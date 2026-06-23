# SparseRenderFormer — Team Action Steps
> Team size: 4 | Transformer knowledge: beginner | Estimated timeline: 13 weeks

---

## Phase 0 — Learning Sprint (Weeks 1–2, all 4)

**All team members complete these independently, then discuss as a group:**

- [ ] Watch Andrej Karpathy "Let's build GPT from scratch" (YouTube, ~2 hrs)
- [ ] Read "Attention Is All You Need" — Vaswani et al., NeurIPS 2017
- [ ] Read RenderFormer paper — arXiv:2505.21925 (SIGGRAPH 2025)
- [ ] Read NSA paper — arXiv:2502.11089 (direct inspiration for GP-Sparse)
- [ ] Read `about_sparserenderformer.tex` in this repo
- [ ] Read `CLAUDE.md` to understand the method design
- [ ] Clone RenderFormer: `git clone https://github.com/microsoft/renderformer`
- [ ] Run RenderFormer inference on Cornell box scene

**Exit criterion:** every team member can explain what the coarse, selected, and local branches do and why they reduce memory from O(N²) to O(N).

---

## Phase 1 — Engineering Setup (Week 3)

**P1 — ML infrastructure:**
- [ ] Confirm GPU allocation (A100-80GB recommended; A6000 workable for ablations)
- [ ] Set up Python env: PyTorch 2.x, Triton ≥ 3.x
- [ ] Install Weights & Biases (or MLflow) for experiment tracking
- [ ] Back up RenderFormer pretrained checkpoint to shared storage before any fine-tuning

**P2 — Baseline reproduction:**
- [ ] Download pretrained checkpoint `microsoft/renderformer-v1.1-swin-large` from HuggingFace
- [ ] Run inference on one Objaverse scene (confirm pipeline works end to end)
- [ ] Reproduce one PSNR number from Table 1 of the RenderFormer paper
- [ ] Document the exact command used and the result

**P3 — Rendering stack:**
- [ ] Install Dr.Jit / Mitsuba 3 for reference renders
- [ ] Install Blender headless (bpy) for dataset generation
- [ ] Render Cornell box with Blender Cycles (512 spp, HDR output) as a smoke test

**P4 — Repo and paper setup:**
- [ ] Create public GitHub repo (suggested name: `sparserenderformer`)
- [ ] Add MIT or Apache 2.0 LICENSE file
- [ ] Write README skeleton: installation, quick-start, citation block
- [ ] Register account on target submission system (CMT / OpenReview — confirm venue first)

---

## Phase 2 — Data Pipeline (Weeks 4–5)
> Start this as early as possible. Rendering is the slowest step in the whole project.

**P3 — Dataset construction:**
- [ ] Get access to RenderFormer's Objaverse subset (check repo README for download link)
- [ ] Filter Objaverse by triangle count; write remesh script to target N ∈ [4k, 32k]
- [ ] Set up Blender Cycles render pipeline: 512 spp, HDR output, consistent lighting

**P1 — Dataset generation:**
- [ ] Generate at minimum 1,000 scenes at N = 16k for fine-tuning
- [ ] Generate 50–100 held-out scenes at N = 32k for evaluation
- [ ] Write a data loader that handles variable triangle counts

---

## Phase 3 — Implementation (Weeks 5–8)

**P1 — BVH and coarse branch:**
- [ ] Implement BVH construction (SAH splitting, block size b = 32)
- [ ] Implement coarse branch: area-weighted average pooling within BVH leaf nodes
- [ ] Extend 3D RoPE to coarse branch (use block centroid as position)
- [ ] Unit test: coarse branch output shape and gradient flow

**P2 — Selected and local branches:**
- [ ] Implement selected branch: top-k BVH block selection per query via dot-product scores
- [ ] Implement local branch: 3D kNN (k = 64) per query
- [ ] Add normal-coherent mask to local branch: suppress pairs where `n_i · n_j < −0.5`
- [ ] Unit test: selected branch retrieves expected blocks on a synthetic scene
- [ ] Unit test: local branch kNN distances and mask correctness

**P1 + P2 — Integration:**
- [ ] Implement gated combination (Eq. 5): per-head learned scalars λ₁, λ₂, λ₃ with softmax normalisation
- [ ] Implement sparse view-dependent cross-attention: frustum culling + kNN (k_r = 256)
- [ ] Fork/adapt RenderFormer codebase to slot in GP-Sparse module
- [ ] **Sanity check:** at N = 4,096 with k_sel = 128, GP-Sparse output must match dense attention exactly

**P2 — Optional kernel optimisation (if time allows):**
- [ ] Profile naive PyTorch masking implementation first
- [ ] Port NSA Triton kernel from DeepSeek-V3 repo for memory-efficient sparse attention

---

## Phase 4 — Experiments (Weeks 8–10)

**P4 — Baselines:**
- [ ] Record: RenderFormer pretrained, N = 4,096 — PSNR = `___`
- [ ] Record: RenderFormer fine-tuned, N = 8,192 (dense, max memory) — PSNR = `___`

**P1 + P2 — Main results:**
- [ ] Run ours: N = 8,192 — PSNR = `___`, memory = `___` GB, time = `___` ms
- [ ] Run ours: N = 16,384 — PSNR = `___`, memory = `___` GB, time = `___` ms
- [ ] Run ours: N = 32,768 — PSNR = `___`, memory = `___` GB, time = `___` ms
- [ ] Log-log memory vs N plot (RF dense vs GP-Sparse)
- [ ] Log-log render time vs N plot (RF dense vs GP-Sparse)

**P4 — Ablations:**
- [ ] Remove coarse branch, N = 16k — PSNR = `___`
- [ ] Remove selected branch, N = 16k — PSNR = `___`
- [ ] Remove local branch, N = 16k — PSNR = `___`
- [ ] Block size b ∈ {16, 32, 64} — PSNR at N = 16k for each
- [ ] k_sel ∈ {2, 4, 8} — PSNR at N = 16k for each

**P3 — Visual results:**
- [ ] Render visual gallery: Cornell box, 3 Objaverse scenes, 1 room scene
- [ ] Side-by-side comparison figures: RF dense vs GP-Sparse at N = 32k

---

## Phase 5 — Paper Completion (Weeks 11–12)

All `\todo{...}` items in `overleaf/main.tex` must be filled before submission.

**P4 — Fill experimental numbers:**
- [ ] Abstract: fill PSNR gain (dB), memory reduction (%), render time reduction (%)
- [ ] Table 1: main results table — all PSNR/SSIM/LPIPS numbers
- [ ] Table 2: branch ablation table — all 7 branch subsets at N = 16,384
- [ ] Figure: memory/time scaling log-log plot (generated in Phase 4)
- [ ] §4.3: fill complexity ratio (XX× fewer token pairs)
- [ ] §5.3: fine-tuning epoch count
- [ ] Conclusion: update all numbers

**P4 — Writing pass (follow CHECKLIST §8b rules in order):**
- [ ] Abstract: verify no `\cite{todo}` remain; target 150–220 words
- [ ] Introduction: check no consecutive sentences start with the same word
- [ ] Related Work: same sentence-start check
- [ ] Method (§4): verify all equations are referenced in text
- [ ] Experiments (§5): no vague descriptors; report ± std where applicable
- [ ] Conclusion: no new results introduced; keep to 150 words
- [ ] Dash audit: `grep -n "\-\-\-" overleaf/main.tex | grep -v "^%"` — must return zero hits
- [ ] Hyphen audit: `grep -n " - " overleaf/main.tex | grep -v "^%"` — must return zero hits
- [ ] Banned-word grep: "remarkable", "impressive", "strong", "novel", "intricate", "delve", "underscore", "pivotal"

**All — Remaining sections:**
- [ ] Acknowledgments: funding source, compute credits, dataset access
- [ ] Ethics / Broader Impact statement
- [ ] Supplementary material outline: extended ablations, additional renders, implementation details

---

## Phase 6 — Submission (Week 13)

- [ ] Confirm venue and enter hard deadline: `___`
  - SIGGRAPH Asia 2026: check https://sa2026.siggraph.org (abstract deadline ~late May/June 2026)
  - 3DV 2026: check https://3dvconf.github.io
  - ICCV 2027: paper deadline ~March 2027
- [ ] arXiv preprint submitted at least 1 week before conference deadline
- [ ] Switch to anonymous submission: comment out `\ConferencePaper`, uncomment `\ConferenceSubmission`, remove author block
- [ ] Page limit check: 8 pages + refs; count before submitting
- [ ] Upload supplementary PDF + video; confirm venue size limit
- [ ] Upload fine-tuned checkpoint to HuggingFace Hub
- [ ] Tag `v1.0` release on GitHub at submission time

**After arXiv is live:**
- [ ] Post to r/MachineLearning and r/computergraphics
- [ ] Email RenderFormer authors (Chong Zeng, Microsoft Research Asia) — polite one-paragraph note with preprint link
- [ ] Add to Papers With Code

---

## Compute Credits (apply early if needed)

- AWS Research Credits: https://aws.amazon.com/research-credits
- Google TPU Research Cloud: https://sites.research.google/trc
- Microsoft AFMR: https://www.microsoft.com/en-us/research/collaboration/accelerate-foundation-models-research

---

## Timeline Summary

| Weeks | Phase | Who |
|-------|-------|-----|
| 1–2 | Phase 0: Learning | All 4 |
| 3 | Phase 1: Setup | Split by role |
| 4–5 | Phase 2: Data pipeline | P1 + P3 |
| 5–8 | Phase 3: Implementation | P1 + P2 |
| 8–10 | Phase 4: Experiments | P1 + P2 run; P3 + P4 visualise |
| 11–12 | Phase 5: Paper completion | P4 leads, all review |
| 13 | Phase 6: Submission | All |

> Critical path: start data rendering (Phase 2) in Week 3, not after implementation. Blender renders are slow. If you wait until the code is done you will have nothing to train on.
