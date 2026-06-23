# GP-Sparse Attention API Reference

## Core Module: GPSparseAttention

```python
from renderformer.layers.sparse_attention import GPSparseAttention

attn = GPSparseAttention(
    dim=768,                    # Hidden dimension
    num_heads=6,                # Attention heads
    k_local=64,                 # Local branch neighbors (kNN)
    k_selected=None,            # Selected branch blocks (auto-scales with heads)
    coarse_pool_size=8,         # Coarse branch pooling factor
    normal_threshold=-0.5,      # Normal coherence threshold
    use_normal_mask=True,       # Enable normal-coherence masking
    bias=True                   # Bias in linear layers
)

# Forward pass
output = attn(
    query,                      # (B, N_q, dim)
    key,                        # (B, N_k, dim)
    value,                      # (B, N_k, dim)
    key_normals=None,           # (B, N_k, 3) optional
    src_key_padding_mask=None   # (B, N_k) optional, True=valid
)
# Returns: (B, N_q, dim)
```

### Methods

#### compute_local_attention()
Implements local branch: k-NN with optional normal-coherence masking.

```python
local_out = attn.compute_local_attention(
    q,                      # (B*num_heads, N_q, head_dim)
    k,                      # (B*num_heads, N_k, head_dim)
    v,                      # (B*num_heads, N_k, head_dim)
    key_normals=None,       # (B, N_k, 3)
    query_indices=None      # (B, N_q) unused
)
# Returns: (B*num_heads, N_q, head_dim)
```

**Algorithm:**
1. Compute attention scores: `scores = (q @ k^T) / sqrt(d)`
2. Top-k selection: keep k_local highest scores
3. Create sparse mask (set others to -∞)
4. If normals provided:
   - Compute dot product: `n_q · n_k`
   - Suppress where dot < -0.5 (back-facing)
   - Set suppressed positions to -∞
5. Softmax + weighted sum: `out = softmax(mask) @ v`

#### compute_selected_attention()
Implements selected branch: top-k block selection.

```python
selected_out = attn.compute_selected_attention(
    q,                      # (B*num_heads, N_q, head_dim)
    k,                      # (B*num_heads, N_k, head_dim)
    v,                      # (B*num_heads, N_k, head_dim)
    block_indices=None      # (B, N_blocks) unused
)
# Returns: (B*num_heads, N_q, head_dim)
```

**Algorithm:**
1. Similar to local branch but without normal masking
2. Selects `k_selected` highest-scoring keys per query
3. Captures direct illumination paths (highest affinity)

#### compute_coarse_attention()
Implements coarse branch: BVH leaf pooling.

```python
coarse_out = attn.compute_coarse_attention(
    q,                      # (B*num_heads, N_q, head_dim)
    k,                      # (B*num_heads, N_k, head_dim)
    v,                      # (B*num_heads, N_k, head_dim)
    leaf_indices=None,      # (B, N_k) unused (fallback to strided)
    leaf_areas=None         # (B, N_leaves) unused
)
# Returns: (B*num_heads, N_q, head_dim)
```

**Algorithm (current fallback):**
1. Strided pooling: `stride = max(1, N_k // coarse_pool_size)`
2. Downsample: `k_pool = k[:, ::stride, :]`
3. Standard attention: `out = softmax(q @ k_pool^T) @ v_pool`

**When BVH leaf indices available:**
1. Group keys by leaf_id
2. Area-weight values: `v_leaf = sum(area_i * v_i) / sum(area_i)`
3. Attend to leaf-pooled values

### Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `dim` | - | 256-2048 | Hidden dimension (must divide evenly by `num_heads`) |
| `num_heads` | - | 1-16 | Attention heads per query |
| `k_local` | 64 | 16-512 | Local branch neighbors (memory-critical) |
| `k_selected` | auto | 4-128 | Selected branch blocks |
| `coarse_pool_size` | 8 | 4-32 | Coarse downsampling factor |
| `normal_threshold` | -0.5 | -1 to 0 | Back-facing suppression angle |
| `use_normal_mask` | True | - | Enable/disable coherence masking |
| `bias` | True | - | Bias in Q/K/V projections |

---

## Layer Wrapper: SparseAttentionLayer

```python
from renderformer.layers.sparse_attention_layer import SparseAttentionLayer

layer = SparseAttentionLayer(
    dim=768,
    num_heads=6,
    ffn_hidden_dim=3072,       # FFN hidden (usually 4×dim)
    k_local=64,
    k_selected=None,
    dropout=0.1,
    activation='swiglu',       # or 'gelu'
    norm_type='layer_norm',    # or 'rms_norm'
    use_normal_mask=True,
    bias=False
)

# Forward (pre-norm architecture)
output = layer(
    x,                         # (B, N, dim)
    key_normals=None,          # (B, N, 3)
    src_key_padding_mask=None  # (B, N)
)
# Returns: (B, N, dim)
```

**Flow:**
```
x → LayerNorm → GPSparseAttention → Dropout
  ↓
  + (residual)
  ↓
  → LayerNorm → FFN (SwiGLU/GeLU) → Dropout
  ↓
  + (residual)
  ↓
  output
```

---

## Encoder: SparseTransformerEncoder

```python
from renderformer.layers.sparse_attention_layer import SparseTransformerEncoder

encoder = SparseTransformerEncoder(
    num_layers=12,             # Stacked layers
    num_heads=6,
    hidden_dim=768,
    ffn_hidden_dim=3072,
    k_local=64,
    k_selected=None,
    dropout=0.0,
    activation='swiglu',
    norm_type='rms_norm',
    use_normal_mask=True,
    bias=False
)

# Forward (applies all layers sequentially)
output = encoder(
    x,                         # (B, N, dim)
    key_normals=None,          # (B, N, 3)
    src_key_padding_mask=None  # (B, N)
)
# Returns: (B, N, dim)
```

---

## Integration: RenderFormerConfig

```python
from renderformer.models.config import RenderFormerConfig

# Dense attention (default)
config = RenderFormerConfig(use_sparse_attention=False)

# Sparse attention
config = RenderFormerConfig(
    use_sparse_attention=True,
    sparse_k_local=64,
    sparse_k_selected=None,      # Auto: max(4, num_heads // 2)
    sparse_use_normal_mask=True,
    # ... other config params
)
```

The `RenderFormer` model will automatically route to `SparseTransformerEncoder` when `use_sparse_attention=True`.

---

## Normal-Coherent Masking: NormalCoherentLocalMask

```python
from renderformer.layers.sparse_attention import NormalCoherentLocalMask

mask_module = NormalCoherentLocalMask(normal_threshold=-0.5)

suppression_mask = mask_module(
    query_normals,              # (B, N_q, 3) normalized
    key_normals                 # (B, N_k, 3) normalized
)
# Returns: (B, N_q, N_k) bool, True = suppress (back-facing)
```

**Logic:**
- Computes `n_q · n_k` (dot product per pair)
- Returns `True` (suppress) where dot < threshold
- Typical usage: threshold = -0.5 allows ~120° cone

**Interpretation:**
- Threshold -1.0: suppress nothing (90°+ cone)
- Threshold -0.5: suppress rear hemisphere (~120° cone)
- Threshold 0.0: suppress ≥90° deviations (frontal only)
- Threshold 0.5: suppress ≥60° deviations (narrow)

---

## Example: Full Training Loop

```python
import torch
from renderformer.models.config import RenderFormerConfig
from renderformer.models.renderformer import RenderFormer

# Config
config = RenderFormerConfig(
    use_sparse_attention=True,
    sparse_k_local=64,
    sparse_use_normal_mask=True,
    latent_dim=768,
    num_layers=12,
    num_heads=6,
    # ... other settings
)

# Model
model = RenderFormer(config).cuda()

# Data
batch_size, max_triangles, num_views = 2, 4096, 8
tri_vpos_list = torch.randn(batch_size, max_triangles, 9)       # 9 coords (3 vertices × 3)
texture_patches = torch.randn(batch_size, max_triangles, 13, 32, 32)
valid_mask = torch.ones(batch_size, max_triangles, dtype=torch.bool)
vns = torch.randn(batch_size, max_triangles, 9)                 # 9 coords (3 vertices × 3)
rays_o = torch.randn(batch_size, num_views, 3)
rays_d = torch.randn(batch_size, num_views, 512, 512, 3)
tri_vpos_view = torch.randn(batch_size, num_views, max_triangles, 9)

# Forward
output = model(
    tri_vpos_list,
    texture_patches,
    valid_mask,
    vns,
    rays_o,
    rays_d,
    tri_vpos_view,
    tf32_view_tf=False
)
# output shape: (batch_size, num_views, height, width, channels)

# Loss & backprop (assumed loss function)
# loss = compute_loss(output, target)
# loss.backward()
# optimizer.step()
```

---

## Debugging & Inspection

### Check sparse attention parameters
```python
model = RenderFormer(config)
attn_layer = model.transformer.layers[0].attn

# Inspect branch gates
gates = torch.nn.functional.softmax(attn_layer.branch_gates, dim=-1)
print("Branch gates per head (coarse, selected, local):")
print(gates)  # Shape: (num_heads, 3)

# Typical output: higher coarse for background, higher local for details
```

### Verify normal-coherence effect
```python
# If normal masking is very restrictive, increase threshold
config.sparse_use_normal_mask = True  # Keep enabled
# (normal_threshold=-0.5 is already permissive)

# To disable masking for comparison:
config_dense_local = RenderFormerConfig(
    use_sparse_attention=True,
    sparse_use_normal_mask=False  # Local branch only, no filtering
)
```

### Profile sparse vs dense
```python
import time

# Dense
config_dense = RenderFormerConfig(use_sparse_attention=False)
model_dense = RenderFormer(config_dense).cuda()

# Sparse
config_sparse = RenderFormerConfig(use_sparse_attention=True)
model_sparse = RenderFormer(config_sparse).cuda()

# Time forward pass
with torch.no_grad():
    for _ in range(5):
        t0 = time.time()
        output_dense = model_dense(...)
        torch.cuda.synchronize()
        t_dense = time.time() - t0

        t0 = time.time()
        output_sparse = model_sparse(...)
        torch.cuda.synchronize()
        t_sparse = time.time() - t0

    speedup = t_dense / t_sparse
    print(f"Speedup: {speedup:.2f}×")
```

---

## Common Issues & Solutions

### "Output contains NaN"
**Cause:** Padding mask not correctly applied (all keys masked)
**Fix:** Ensure `src_key_padding_mask[valid_tokens] = True`

### "Gradient explosion"
**Cause:** Sparse attention concentrates gradients on few keys
**Fix:** Reduce `k_local` or enable gradient clipping

### "Local branch gets zero weight"
**Cause:** Selected branch dominates, gates not updated
**Fix:** Check learning rate, may need separate gate learning rate

### "Normal masking too aggressive"
**Cause:** Threshold -0.5 too strict for your scene
**Fix:** Increase threshold (e.g., -0.7) or disable (`use_normal_mask=False`)
