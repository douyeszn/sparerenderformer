"""
Tests for GP-Sparse Attention mechanism.
"""

import pytest
import torch
import torch.nn as nn


def test_gp_sparse_attention_basic():
    """Test basic GP-Sparse attention forward pass."""
    from renderformer.layers.sparse_attention import GPSparseAttention

    B, N, dim, num_heads = 2, 32, 768, 6
    attn = GPSparseAttention(dim=dim, num_heads=num_heads, k_local=8, k_selected=4)

    query = torch.randn(B, N, dim)
    key = torch.randn(B, N, dim)
    value = torch.randn(B, N, dim)

    output = attn(query, key, value)

    assert output.shape == (B, N, dim), f"Expected {(B, N, dim)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ Basic forward pass test passed")


def test_gp_sparse_attention_with_normals():
    """Test sparse attention with normal-coherence masking."""
    from renderformer.layers.sparse_attention import GPSparseAttention

    B, N, dim, num_heads = 2, 32, 768, 6
    attn = GPSparseAttention(
        dim=dim, num_heads=num_heads, k_local=8, use_normal_mask=True
    )

    query = torch.randn(B, N, dim)
    key = torch.randn(B, N, dim)
    value = torch.randn(B, N, dim)
    key_normals = torch.randn(B, N, 3)
    key_normals = torch.nn.functional.normalize(key_normals, dim=-1)

    output = attn(query, key, value, key_normals=key_normals)

    assert output.shape == (B, N, dim)
    assert not torch.isnan(output).any()
    print("✓ Forward pass with normals test passed")


def test_normal_coherent_masking():
    """Test normal-coherence mask computation."""
    from renderformer.layers.sparse_attention import NormalCoherentLocalMask

    mask_module = NormalCoherentLocalMask(normal_threshold=-0.5)

    B, N_q, N_k = 2, 16, 32

    # Create aligned normals (should not suppress)
    query_normals = torch.tensor([[0.0, 0.0, 1.0]]).expand(B, N_q, 3).float()
    key_normals = torch.tensor([[0.0, 0.0, 1.0]]).expand(B, N_k, 3).float()

    mask = mask_module(query_normals, key_normals)
    assert not mask.any(), "Aligned normals should not be suppressed"

    # Create opposite normals (should suppress)
    query_normals = torch.tensor([[0.0, 0.0, 1.0]]).expand(B, N_q, 3).float()
    key_normals = torch.tensor([[0.0, 0.0, -1.0]]).expand(B, N_k, 3).float()

    mask = mask_module(query_normals, key_normals)
    assert mask.all(), "Opposite normals should be suppressed"

    print("✓ Normal-coherence masking test passed")


def test_sparse_attention_layer():
    """Test sparse attention layer with residual + FFN."""
    from renderformer.layers.sparse_attention_layer import SparseAttentionLayer

    B, N, dim = 2, 32, 768
    layer = SparseAttentionLayer(
        dim=dim,
        num_heads=6,
        ffn_hidden_dim=dim * 4,
        k_local=8,
        dropout=0.1,
    )

    x = torch.randn(B, N, dim)
    output = layer(x)

    assert output.shape == (B, N, dim)
    assert not torch.isnan(output).any()
    print("✓ Sparse attention layer test passed")


def test_sparse_transformer_encoder():
    """Test sparse transformer encoder."""
    from renderformer.layers.sparse_attention_layer import SparseTransformerEncoder

    B, N, dim = 2, 32, 768
    encoder = SparseTransformerEncoder(
        num_layers=2,
        num_heads=6,
        hidden_dim=dim,
        ffn_hidden_dim=dim * 4,
        k_local=8,
    )

    x = torch.randn(B, N, dim)
    output = encoder(x)

    assert output.shape == (B, N, dim)
    assert not torch.isnan(output).any()
    print("✓ Sparse transformer encoder test passed")


def test_config_sparse_attention():
    """Test RenderFormer config with sparse attention settings."""
    from renderformer.models.config import RenderFormerConfig

    # Default: dense attention
    config = RenderFormerConfig()
    assert not config.use_sparse_attention

    # Sparse attention enabled
    config_sparse = RenderFormerConfig(
        use_sparse_attention=True, sparse_k_local=64, sparse_use_normal_mask=True
    )
    assert config_sparse.use_sparse_attention
    assert config_sparse.sparse_k_local == 64
    assert config_sparse.sparse_use_normal_mask

    print("✓ Config test passed")


def test_gradient_flow():
    """Test that gradients flow through sparse attention."""
    from renderformer.layers.sparse_attention import GPSparseAttention

    B, N, dim = 2, 16, 768
    attn = GPSparseAttention(dim=dim, num_heads=6)

    query = torch.randn(B, N, dim, requires_grad=True)
    key = torch.randn(B, N, dim, requires_grad=True)
    value = torch.randn(B, N, dim, requires_grad=True)

    output = attn(query, key, value)
    loss = output.sum()
    loss.backward()

    assert query.grad is not None, "Query grad is None"
    assert key.grad is not None, "Key grad is None"
    assert value.grad is not None, "Value grad is None"
    assert not torch.isnan(query.grad).any(), "NaN in query grad"

    print("✓ Gradient flow test passed")


def test_branch_gating():
    """Test that branch gating produces valid weights."""
    from renderformer.layers.sparse_attention import GPSparseAttention

    attn = GPSparseAttention(dim=768, num_heads=6)

    # Check that branch gates sum to 1 per head
    gates = torch.nn.functional.softmax(attn.branch_gates, dim=-1)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(6))

    print("✓ Branch gating test passed")


if __name__ == "__main__":
    test_gp_sparse_attention_basic()
    test_gp_sparse_attention_with_normals()
    test_normal_coherent_masking()
    test_sparse_attention_layer()
    test_sparse_transformer_encoder()
    test_config_sparse_attention()
    test_gradient_flow()
    test_branch_gating()
    print("\n✅ All tests passed!")
