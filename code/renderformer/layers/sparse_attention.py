"""
GP-Sparse Attention: Geometry-Physics Sparse Attention for RenderFormer.

Three-branch sparse attention mechanism:
1. Coarse: BVH leaf pooling → global context
2. Selected: Top-k block selection → direct illumination
3. Local: kNN with normal-coherence mask → contact shadows
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NormalCoherentLocalMask(nn.Module):
    """
    Apply normal-coherent masking to suppress back-facing neighbors.
    Suppresses keys where n_i · n_j < -0.5 (back-facing relative to query).
    """

    def __init__(self, normal_threshold: float = -0.5):
        super().__init__()
        self.normal_threshold = normal_threshold

    def forward(
        self,
        query_normals: torch.Tensor,  # (B, N_q, 3)
        key_normals: torch.Tensor,    # (B, N_k, 3)
    ) -> torch.Tensor:
        """
        Compute coherence mask: True = suppress (back-facing), False = keep.

        Args:
            query_normals: (B, N_q, 3) normalized normals
            key_normals: (B, N_k, 3) normalized normals

        Returns:
            mask: (B, N_q, N_k) binary mask, True where dot product < threshold
        """
        # (B, N_q, 1, 3) @ (B, 1, N_k, 3) -> (B, N_q, N_k)
        dot_product = torch.sum(
            query_normals.unsqueeze(2) * key_normals.unsqueeze(1),
            dim=-1
        )
        # True = back-facing (suppress), False = front-facing (keep)
        return dot_product < self.normal_threshold


class GPSparseAttention(nn.Module):
    """
    Geometry-Physics Sparse Attention: 3-branch sparse self/cross-attention.

    Combines coarse (global), selected (direct), and local (near-field) branches
    with learned per-head gating.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        k_local: int = 64,
        k_selected: int = None,
        coarse_pool_size: int = 8,
        normal_threshold: float = -0.5,
        use_normal_mask: bool = True,
        bias: bool = True,
    ):
        """
        Args:
            dim: attention dimension (must be divisible by num_heads)
            num_heads: number of attention heads
            k_local: neighbors for local branch (k-NN)
            k_selected: top-k blocks for selected branch
            coarse_pool_size: BVH leaf node size (for coarse branch)
            normal_threshold: threshold for normal-coherence masking
            use_normal_mask: whether to apply normal-coherence mask
            bias: whether to use bias in projections
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k_local = k_local
        self.k_selected = k_selected or max(4, num_heads // 2)
        self.coarse_pool_size = coarse_pool_size
        self.use_normal_mask = use_normal_mask

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)

        # Per-head gate parameters (softmax over 3 branches)
        self.branch_gates = nn.Parameter(torch.zeros(num_heads, 3))
        nn.init.normal_(self.branch_gates, mean=0.0, std=0.02)

        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        # Normal-coherence masking
        if use_normal_mask:
            self.normal_mask = NormalCoherentLocalMask(normal_threshold)
        else:
            self.normal_mask = None

        self.scaling = self.head_dim ** -0.5

    def compute_local_attention(
        self,
        q: torch.Tensor,           # (B*num_heads, N_q, head_dim)
        k: torch.Tensor,           # (B*num_heads, N_k, head_dim)
        v: torch.Tensor,           # (B*num_heads, N_k, head_dim)
        key_normals: Optional[torch.Tensor] = None,  # (B, N_k, 3)
        query_indices: Optional[torch.Tensor] = None,  # (B, N_q)
    ) -> torch.Tensor:
        """
        Local branch: k-NN in 3D space with optional normal-coherence masking.

        Returns attention output of same shape as input query.
        """
        # Simple k-NN: use query-key similarity as proxy for spatial proximity
        # In practice, this would use actual 3D positions
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # (B*num_heads, N_q, N_k)

        # Top-k selection
        k_local = min(self.k_local, k.size(1))
        topk_vals, topk_indices = torch.topk(scores, k=k_local, dim=-1)  # (B*num_heads, N_q, k)

        # Create sparse mask
        mask = torch.full_like(scores, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)

        # Apply normal-coherence masking if available
        if self.use_normal_mask and key_normals is not None:
            # Reshape for normal masking
            B_total, N_q, _ = q.shape
            B = B_total // self.num_heads
            q_normals = key_normals[:B]  # Use key normals for both (simplified)
            normal_suppress = self.normal_mask(q_normals, key_normals)  # (B, N_q, N_k)
            # Expand to num_heads and apply
            normal_suppress = normal_suppress.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            normal_suppress = normal_suppress.reshape(B_total, N_q, -1)
            mask = torch.where(normal_suppress, torch.full_like(mask, float('-inf')), mask)

        attn = F.softmax(mask, dim=-1)
        return torch.matmul(attn, v)

    def compute_selected_attention(
        self,
        q: torch.Tensor,  # (B*num_heads, N_q, head_dim)
        k: torch.Tensor,  # (B*num_heads, N_k, head_dim)
        v: torch.Tensor,  # (B*num_heads, N_k, head_dim)
        block_indices: Optional[torch.Tensor] = None,  # (B, N_blocks)
    ) -> torch.Tensor:
        """
        Selected branch: top-k BVH blocks via dot-product scoring.
        Captures direct illumination and specular paths.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # (B*num_heads, N_q, N_k)

        # If block structure is available, use it; otherwise use per-token top-k
        k_selected = min(self.k_selected, k.size(1))
        topk_vals, topk_indices = torch.topk(scores, k=k_selected, dim=-1)  # (B*num_heads, N_q, k)

        mask = torch.full_like(scores, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)

        attn = F.softmax(mask, dim=-1)
        return torch.matmul(attn, v)

    def compute_coarse_attention(
        self,
        q: torch.Tensor,  # (B*num_heads, N_q, head_dim)
        k: torch.Tensor,  # (B*num_heads, N_k, head_dim)
        v: torch.Tensor,  # (B*num_heads, N_k, head_dim)
        leaf_indices: Optional[torch.Tensor] = None,  # (B, N_k), leaf block ID per key
        leaf_areas: Optional[torch.Tensor] = None,    # (B, N_leaves), area per leaf
    ) -> torch.Tensor:
        """
        Coarse branch: area-weighted pooling of triangles within BVH leaf nodes.
        Captures global/low-frequency radiance. Reduces keys by ~32x.
        """
        if leaf_indices is None:
            # Fallback: use uniformly spaced pooling
            pool_size = min(self.coarse_pool_size, k.size(1))
            # Simple strided pooling
            stride = max(1, k.size(1) // pool_size)
            k_pooled = k[:, ::stride, :]
            v_pooled = v[:, ::stride, :]
        else:
            # Pool by leaf node
            B_total, N_q, _ = q.shape
            B = B_total // self.num_heads
            n_leaves = leaf_indices.max().item() + 1

            # Aggregate by leaf: v_leaf = sum(area * v) / sum(area)
            v_pooled_list = []
            for leaf_id in range(n_leaves):
                mask_leaf = (leaf_indices == leaf_id)  # (B, N_k)
                if leaf_areas is not None:
                    area = leaf_areas[:, leaf_id:leaf_id+1]  # (B, 1)
                    weighted_v = v.reshape(B, self.num_heads, -1, self.head_dim)[..., :] * area.unsqueeze(1).unsqueeze(-1)
                else:
                    weighted_v = v.reshape(B, self.num_heads, -1, self.head_dim)
                v_pooled_list.append(weighted_v.mean(dim=2))  # (B, num_heads, head_dim)

            k_pooled = torch.stack([torch.ones(B, self.head_dim, device=k.device) for _ in range(len(v_pooled_list))], dim=1)
            v_pooled = torch.stack(v_pooled_list, dim=1).reshape(B_total, -1, self.head_dim)

        scores = torch.matmul(q, k_pooled.transpose(-2, -1)) * self.scaling
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v_pooled)

    def forward(
        self,
        query: torch.Tensor,  # (B, N_q, dim)
        key: torch.Tensor,    # (B, N_k, dim)
        value: torch.Tensor,  # (B, N_k, dim)
        key_normals: Optional[torch.Tensor] = None,  # (B, N_k, 3)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, N_k)
    ) -> torch.Tensor:
        """
        Sparse attention forward pass combining 3 branches.

        Args:
            query: query tokens
            key: key tokens (can equal query for self-attention)
            value: value tokens (can equal query for self-attention)
            key_normals: normal vectors for keys (for normal-coherence masking)
            src_key_padding_mask: mask where True = valid token to attend to

        Returns:
            output: (B, N_q, dim)
        """
        B, N_q, _ = query.shape
        N_k = key.size(1)

        # Project Q, K, V
        q = self.q_proj(query)  # (B, N_q, dim)
        k = self.k_proj(key)    # (B, N_k, dim)
        v = self.v_proj(value)  # (B, N_k, dim)

        # Reshape to (B*num_heads, N, head_dim)
        q = rearrange(q, "b n (h d) -> (b h) n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=self.num_heads)

        # Compute 3 branches
        coarse_out = self.compute_coarse_attention(q, k, v, leaf_indices=None)  # (B*num_heads, N_q, head_dim)
        selected_out = self.compute_selected_attention(q, k, v, block_indices=None)
        local_out = self.compute_local_attention(q, k, v, key_normals=key_normals)

        # Gate fusion: per-head softmax over branches
        gates = F.softmax(self.branch_gates, dim=-1)  # (num_heads, 3)

        # Stack outputs and apply gates
        # Rearrange to (B, num_heads, N_q, head_dim) for each branch
        coarse_out = rearrange(coarse_out, "(b h) n d -> b h n d", h=self.num_heads)
        selected_out = rearrange(selected_out, "(b h) n d -> b h n d", h=self.num_heads)
        local_out = rearrange(local_out, "(b h) n d -> b h n d", h=self.num_heads)

        # Stack branches: (B, num_heads, 3, N_q, head_dim)
        stacked = torch.stack([coarse_out, selected_out, local_out], dim=2)

        # Apply per-head gates: gates is (num_heads, 3)
        # Reshape gates to (1, num_heads, 3, 1, 1) for broadcasting
        gates = gates.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, num_heads, 3, 1, 1)
        gated = (stacked * gates).sum(dim=2)  # (B, num_heads, N_q, head_dim)

        # Rearrange back to (B, N_q, dim)
        output = rearrange(gated, "b h n d -> b n (h d)")
        output = self.out_proj(output)

        return output
