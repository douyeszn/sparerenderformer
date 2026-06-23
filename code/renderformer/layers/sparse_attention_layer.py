"""
Sparse Attention Layer: integrates GP-Sparse into transformer encoder/decoder.
"""

from typing import Literal, Optional
import torch
import torch.nn as nn
from renderformer.layers.sparse_attention import GPSparseAttention
from renderformer.layers.attention import FeedForwardSwiGLU, FeedForwardGeLU, EPS


class SparseAttentionLayer(nn.Module):
    """
    Transformer layer with sparse self-attention and feed-forward.
    Pre-norm architecture (norm → attn → residual → norm → ffn → residual).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        k_local: int = 64,
        k_selected: int = None,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        use_normal_mask: bool = True,
        bias: bool = True,
    ):
        """
        Args:
            dim: hidden dimension
            num_heads: number of attention heads
            ffn_hidden_dim: feed-forward hidden dimension
            k_local: k-NN neighbors for local branch
            k_selected: top-k blocks for selected branch
            dropout: dropout rate
            activation: 'swiglu' or 'gelu'
            norm_type: 'layer_norm' or 'rms_norm'
            use_normal_mask: whether to apply normal-coherence masking
            bias: whether to use bias in projections
        """
        super().__init__()

        # Sparse self-attention
        self.attn = GPSparseAttention(
            dim=dim,
            num_heads=num_heads,
            k_local=k_local,
            k_selected=k_selected,
            use_normal_mask=use_normal_mask,
            bias=bias,
        )
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Normalization
        if norm_type == 'layer_norm':
            norm_module = nn.LayerNorm
        elif norm_type == 'rms_norm':
            norm_module = nn.RMSNorm
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.attn_norm = norm_module(dim, eps=EPS)

        # Feed-forward
        if activation == 'swiglu':
            self.ffn = FeedForwardSwiGLU(dim, ffn_hidden_dim, dropout, bias)
        elif activation == 'gelu':
            self.ffn = FeedForwardGeLU(dim, ffn_hidden_dim, dropout, bias)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.ffn_norm = norm_module(dim, eps=EPS)
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,  # (B, N, dim)
        key_normals: Optional[torch.Tensor] = None,  # (B, N, 3)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, N)
    ) -> torch.Tensor:
        """
        Args:
            x: input tokens
            key_normals: normal vectors for sparse masking
            src_key_padding_mask: padding mask (True = valid)

        Returns:
            output: (B, N, dim)
        """
        # Sparse self-attention with residual
        x_norm = self.attn_norm(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, key_normals=key_normals, src_key_padding_mask=src_key_padding_mask)
        x = x + self.attn_dropout(attn_out)

        # Feed-forward with residual
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))

        return x


class SparseTransformerEncoder(nn.Module):
    """
    Transformer encoder using sparse self-attention layers.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        ffn_hidden_dim: int,
        k_local: int = 64,
        k_selected: int = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        use_normal_mask: bool = True,
        bias: bool = True,
    ):
        """
        Args:
            num_layers: number of transformer layers
            num_heads: number of attention heads
            hidden_dim: hidden dimension
            ffn_hidden_dim: feed-forward hidden dimension
            k_local: k-NN neighbors per layer
            k_selected: top-k blocks per layer
            dropout: dropout rate
            activation: activation function
            norm_type: normalization type
            use_normal_mask: whether to use normal-coherence masking
            bias: whether to use bias
        """
        super().__init__()

        self.layers = nn.ModuleList([
            SparseAttentionLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                k_local=k_local,
                k_selected=k_selected,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type,
                use_normal_mask=use_normal_mask,
                bias=bias,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,  # (B, N, dim)
        key_normals: Optional[torch.Tensor] = None,  # (B, N, 3)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, N)
    ) -> torch.Tensor:
        """
        Args:
            x: input tokens (B, N, dim)
            key_normals: normal vectors (B, N, 3)
            src_key_padding_mask: padding mask (B, N), True = valid token

        Returns:
            output: (B, N, dim)
        """
        for layer in self.layers:
            x = layer(x, key_normals=key_normals, src_key_padding_mask=src_key_padding_mask)
        return x
