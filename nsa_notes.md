# Native Sparse Attention (NSA)

**Reference:** Yuan et al. (DeepSeek, 2025). *Native Sparse Attention: Hardware-Aligned and
Natively Trainable Sparse Attention*. arXiv:2502.11089.

---

## Problem Statement

Full self-attention scales quadratically with sequence length, making long-context pretraining
computationally prohibitive. Prior sparse attention methods such as BigBird and Longformer
address this through fixed, pre-defined patterns applied after full-attention pretraining. This
post-hoc approach has two consequences. First, training still requires full attention, so
computational savings apply only at inference. Second, because the model is not trained to be
sparse, quality degrades when sparsity is applied. The theoretical speedups in prior methods do
not translate to real hardware gains because the sparse patterns are not aligned with how modern
GPUs load data.

---

## Contribution

NSA introduces natively trainable sparse attention in which sparsity is part of the model
architecture from the start of pretraining. The method is designed around hardware constraints,
specifically the memory-bandwidth bottleneck in long-sequence attention on GPUs. Speedups are
demonstrated across all three phases: decoding, forward propagation, and backward propagation.
The model is pretrained on 270 billion tokens at a context length of 64k tokens.

---

## Architecture

NSA decomposes attention into three parallel branches that are combined via learned scalar gates.

### Compression Branch

The compression branch produces a coarse-grained representation of the full context. Contiguous
token blocks of length $l$ with stride $d$ are each compressed into a single token using a
learnable MLP with intra-block positional encoding. The model attends over these compressed block
tokens, which reduces the number of keys per query by approximately 32$\times$.

### Selection Branch

The selection branch uses the attention scores from the compression branch as importance
estimates. It selects the top-$k$ raw token blocks ranked by importance score and attends to
their full, uncompressed tokens. Selection is dynamic: each query selects different blocks
depending on content, so the model learns which parts of the context are relevant rather than
following a fixed pattern.

### Sliding Window Branch

The sliding window branch attends to a fixed window of the $w$ most recent tokens. It handles
local continuity independently of the other two branches, which prevents gradient interference
between local and long-range learning signals.

### Gating

The three branch outputs are combined as:

$$o_t = \sum_{c \,\in\, \{\text{cmp,\, slc,\, win}\}} g_t^c \cdot \text{Attn}(q_t,\, \tilde{K}_t^c,\, \tilde{V}_t^c)$$

where $g_t^c \in [0,1]$ are gate scores produced by a per-branch MLP with sigmoid activation.
The gates allow the model to weight branches differently depending on the query.

---

## GQA Integration

NSA is designed for Grouped Query Attention (GQA), in which multiple query heads share a single
key-value (KV) head. Block selection in the selection branch is made consistent across all query
heads within a GQA group by aggregating their importance scores before ranking:

$$\tilde{p}'_{\text{slc}} = \sum_{h} \tilde{p}_{\text{slc},(h)}$$

This ensures the same KV blocks are loaded once per group rather than once per head, which
reduces KV-cache bandwidth during decoding. GQA itself reduces the number of KV heads relative
to query heads so that less memory is loaded per decoding step. NSA's group-consistent selection
extends this bandwidth reduction to the selection branch.

---

## Hardware Motivation

Long-sequence attention is memory-bandwidth bound on modern GPUs. The time to load key-value
tensors from high-bandwidth memory (HBM) to on-chip SRAM exceeds the time spent on matrix
multiplication. NSA reduces the volume of KV data loaded per query through three mechanisms.
The compression branch replaces $l$ raw tokens with one compressed token. The selection branch
loads only the KV blocks ranked as relevant by the query. GQA-consistent selection loads each
block once per query group rather than once per head. The paper characterises this design as
"arithmetic intensity-balanced."

---

## Comparison to Prior Methods

| Property | BigBird / Longformer | NSA |
|---|---|---|
| Sparse patterns | Fixed, pre-defined | Learned during pretraining |
| Training mode | Full attention pretraining required | Native sparse pretraining |
| Speedup phases | Inference only | Pretraining + inference |
| Hardware alignment | Not optimised | Explicitly hardware-aligned |

---

## Results

| Benchmark | Full Attention | NSA |
|---|---|---|
| LongBench (64k avg) | baseline | +0.032 |
| Needle-in-Haystack (64k) | — | Perfect retrieval |
| AIME reasoning (16k) | 0.092 | 0.146 |

NSA matches or exceeds full attention on general benchmarks and outperforms it on long-context
retrieval and reasoning tasks.

---

## Relation to SparseRenderFormer

NSA's three-branch decomposition maps closely to the GP-Sparse module in SparseRenderFormer:

| NSA Branch | GP-Sparse Equivalent |
|---|---|
| Compression | Coarse branch (BVH-pooled block tokens) |
| Selection | Selected branch (top-$k$ BVH block selection) |
| Sliding window | Local branch (kNN in 3D with normal-coherent mask) |
| Learned gates | Per-head scalar gates (softmax-normalised $\lambda_1, \lambda_2, \lambda_3$) |

The primary distinction is that GP-Sparse replaces the position-agnostic sliding window with a
geometry-aware local branch that uses 3D kNN and a normal-coherent mask (suppressing neighbours
where $\mathbf{n}_i \cdot \mathbf{n}_j < -0.5$). This substitution is appropriate for
triangle-mesh rendering, where spatial proximity and surface orientation are more informative
locality signals than token recency.
