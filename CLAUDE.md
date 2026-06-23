# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Compiler

Always use `tectonic` to compile LaTeX. Never use `pdflatex` or `xelatex`.

### Folder structure

| Folder | Contents |
|--------|---------|
| `tex/` | All `.tex` source files + style deps (`.cls`, `.sty`, `.bib`, assets) |
| `pdf/` | All compiled PDF outputs |
| `papers/` | Downloaded reference PDFs |
| `overleaf/` | Mirror of `main.tex` and its assets for Overleaf upload |

### Compile commands (run from repo root)

```bash
tectonic -o pdf/ tex/main.tex
tectonic -o pdf/ tex/about_sparserenderformer.tex
tectonic -o pdf/ tex/nsa_notes.tex
tectonic -o pdf/ tex/action_steps.tex
tectonic -o pdf/ tex/transformer_encoder.tex
```

All `.cls` and `.sty` dependencies (`egpubl.cls`, `eg2026.sty`, `dfadobe.sty`, `egweblnk.sty`) are in `tex/`. Tectonic resolves them relative to the source file location automatically.

After editing `tex/main.tex`, keep `overleaf/main.tex` in sync by copying:

```bash
cp tex/main.tex overleaf/main.tex
cp tex/references.bib overleaf/references.bib
```

## Project Overview

This is a research paper repository for **SparseRenderFormer**, a method that extends RenderFormer (Zeng et al., SIGGRAPH 2025) to scale transformer-based neural global illumination from ~4k to 32k+ triangles via hierarchical sparse attention.

The paper targets the Eurographics / EGSR format (`egpubl.cls` + `eg2026.sty`).

### File Map

| File | Purpose |
|------|---------|
| `main.tex` | Primary submission paper (Eurographics format) |
| `about_sparserenderformer.tex` | Internal overview: RF essentials + two contributions |
| `addition1.tex` | Technical note: Form-Factor-Guided Block Selection |
| `prerequisites_addition2.tex` | Background doc for Normal-Coherent Local Masking |
| `references.bib` | All citations (BibTeX) |
| `CHECKLIST.md` | Master task tracker — consult before starting any section |

### Architecture (GP-Sparse)

SparseRenderFormer replaces RenderFormer's O(N²) view-independent self-attention with **GP-Sparse (Geometry-Physics Sparse Attention)**, a three-branch module:

1. **Coarse branch** — area-weighted pooling of triangles within BVH leaf nodes → attend to compressed block tokens. Captures global/low-frequency radiance. Reduces keys per query by 32×.
2. **Selected branch** — top-k BVH block selection per query using dot-product scores. Captures direct illumination and specular paths.
3. **Local branch** — kNN (k=64) in 3D space with a **normal-coherent mask** that suppresses back-facing neighbours (`n_i · n_j < −0.5`). Captures contact shadows and near-field bleeding.

Outputs are combined via per-head learned scalar gates (softmax-normalised λ₁, λ₂, λ₃).

The view-dependent cross-attention stage is also sparsified via frustum culling + kNN (k_r=256).

### RenderFormer Token Format

Each triangle token is 13-dimensional: `[normal(3), diffuse_albedo(3), specular_albedo(3), roughness(1), emission(3)]`. Normal slice is `[:, 0:3]`. Geometric position is encoded via 3D RoPE on attention logits, not in the token.

## Writing Rules

These apply to every edit of `main.tex` (see `CHECKLIST.md §8` for the full list):

- No em dashes (`---`) as prose connectors. Rewrite as a full clause or use parentheses.
- No spaced hyphens (` - `) in prose. Hyphens are compound modifiers only (`well-known`, `back-face`, `fine-tuned`).
- No vague adjectives: "remarkable", "impressive", "strong", "novel", "intricate", "delve", "underscore", "pivotal".
- Unhedged novelty claims must use "To our knowledge,".
- Run dash audit before any commit: `grep -n "\-\-\-" main.tex | grep -v "^%"`

## Citations

Before adding any citation to `main.tex`, add it to the Citation Tracker table in `CHECKLIST.md` with arXiv ID or DOI, verify the link loads, and mark it VERIFIED. All current citations are listed there with status.

## Submission Modes

To switch between submission and camera-ready in `main.tex`, toggle the publication type near the top (only one active at a time):

```latex
% \ConferenceSubmission   % anonymous — comment out author block too
\ConferencePaper          % camera-ready
```

## Open TODOs

All unfilled experimental numbers are marked `\todo{...}` in `main.tex`. Search with:

```bash
grep -n "\\\\todo" main.tex
```
