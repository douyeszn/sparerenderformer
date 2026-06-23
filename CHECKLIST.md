# SparseRenderFormer — Running Checklist
> Updated: 2026-05-20. Tick boxes as work is completed.
> Writing skill active: apply rules from `~/.claude/skills/writing` to every section before submission.

---

## 0. Citation Tracker
*All citations verified against arxiv / ACM DL on 2026-05-20.*

| Key | Authors | Title | Venue / arXiv | Status |
|-----|---------|-------|---------------|--------|
| zeng2025renderformer | Zeng, Dong, Peers, Wu, Tong | RenderFormer: Transformer-based Neural Rendering of Triangle Meshes with Global Illumination | SIGGRAPH 2025 / arXiv:2505.21925 | ✅ VERIFIED |
| yuan2025nsa | Yuan et al. (DeepSeek) | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | arXiv:2502.11089 | ✅ VERIFIED |
| deepseek2025dsa | DeepSeek-AI | DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention | https://aarnphm.xyz/thoughts/papers/DeepSeek_V3_2.pdf | ⚠️ VERIFY arxiv ID before submission |
| liang2025diffusionrenderer | Liang, Gojcic, Ling et al. (NVIDIA) | DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models | CVPR 2025 Oral / arXiv:2501.18590 | ✅ VERIFIED |
| liao2023equiformerv2 | Liao, Wood, Das, Smidt | EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations | arXiv:2306.12059 | ✅ VERIFIED |
| ye2024differential | Ye, Dong, Xia et al. (Microsoft) | Differential Transformer | ICLR 2025 Oral / arXiv:2410.05258 | ✅ VERIFIED |
| shah2024flashattention3 | Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | arXiv:2407.08608 | ✅ VERIFIED |
| su2021roformer | Su, Lu, Pan, Murtadha, Wen, Liu | RoFormer: Enhanced Transformer with Rotary Position Embedding | arXiv:2104.09864 | ✅ VERIFIED |
| deitke2022objaverse | Deitke, Schwenk, Salvador et al. | Objaverse: A Universe of Annotated 3D Objects | arXiv:2212.08051 | ✅ VERIFIED |
| kerbl2023gaussian | Kerbl, Kopanas, Leimkühler, Drettakis | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | SIGGRAPH 2023 / arXiv:2308.04079 | ✅ VERIFIED |
| jakob2022drjit | Jakob, Speierer, Roussel, Vicini | Dr.Jit: A Just-In-Time Compiler for Differentiable Rendering | SIGGRAPH 2022 / arXiv:2202.01284 | ✅ VERIFIED |
| vaswani2017attention | Vaswani et al. | Attention Is All You Need | NeurIPS 2017 | ✅ WELL-KNOWN — cited in §Related Work (Efficient Transformers) |
| dao2022flashattention | Dao, Fu, Ermon, Rudra, Ré | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | NeurIPS 2022 | ✅ WELL-KNOWN |
| dao2023flashattention2 | Dao | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | ICLR 2024 / arXiv:2307.08691 | ✅ WELL-KNOWN |
| liu2021swin | Liu, Lin, Cao, Hu et al. | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | ICCV 2021 | ✅ WELL-KNOWN |
| mildenhall2020nerf | Mildenhall, Srinivasan, Tancik et al. | NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis | ECCV 2020 | ✅ WELL-KNOWN |
| walter2007microfacet | Walter, Marschner, Li, Torrance | Microfacet Models for Refraction through Rough Surfaces | EGSR 2007 | ✅ WELL-KNOWN |
| kajiya1986rendering | Kajiya | The Rendering Equation | SIGGRAPH 1986 | ✅ WELL-KNOWN |
| zhang2018lpips | Zhang, Isola, Efros, Shechtman, Wang | The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS) | CVPR 2018 / arXiv:1801.03924 | ✅ VERIFIED |
| bengio2013estimating | Bengio, Léonard, Courville | Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation | arXiv:1308.3432 | ✅ VERIFIED |

**Rule:** Before adding any new citation to main.tex, add it here first with arXiv ID or DOI, verify the page loads, then mark VERIFIED.

---

## 1. Understand the Prior Work

- [x] Read RenderFormer paper end to end (architecture, ablations, limitations)
- [ ] Clone RenderFormer repo: `git clone https://github.com/microsoft/renderformer`
- [ ] Download pretrained checkpoint `microsoft/renderformer-v1.1-swin-large` from HuggingFace
- [ ] Run inference on Cornell box scene (should take ~seconds on A100)
- [ ] Run inference on one Objaverse scene (confirm pipeline works)
- [ ] Reproduce one PSNR number from Table 1 in the paper
- [ ] Read all four "future work" bullet points in RenderFormer §6 and annotate here

---

## 2. Scope Your Contribution

- [x] Thesis: Hierarchical Sparse Attention to scale from 4k → 32k+ triangles
- [x] Method: three-branch (coarse / selected / local) BVH-grounded attention
- [ ] Define headline metric: **PSNR at 32,768 triangles vs dense-attention extrapolation**
- [ ] Secondary metrics: peak GPU memory (GB), render time per frame (ms)
- [ ] Monthly arXiv watch: search `renderformer`, `neural global illumination`, `transformer mesh rendering`
  - [ ] 2026-05-20 — first check (no competing preprint found)
  - [ ] 2026-06-20 — second check
  - [ ] 2026-07-20 — third check

---

## 3. Engineering Setup

- [ ] GPU access confirmed: `___` (A100-80GB recommended; A6000 workable for ablations)
- [ ] Python env: PyTorch 2.x, Triton ≥ 3.x (for NSA-style Triton kernels)
- [ ] Install Dr.Jit / Mitsuba 3 for reference renders
- [ ] Install Blender headless (bpy) for dataset generation
- [ ] BVH implementation: decide `trimesh` + custom BVH vs `embree` Python bindings
- [ ] Sparse attention kernel: start with naive PyTorch masking, then profile
  - [ ] Optional: port NSA Triton kernel (https://github.com/deepseek-ai/DeepSeek-V3 — sparse attn code)

---

## 4. Data

- [ ] Get access to RenderFormer's Objaverse subset (check repo README for download link)
- [ ] Alternatively: filter Objaverse by triangle count, remesh to `N ∈ [4k, 32k]`
- [ ] Set up Blender Cycles render pipeline for pseudo-GT (512 spp, HDR output)
- [ ] Generate at minimum 1,000 scenes at `N = 16k` for fine-tuning
- [ ] Generate 50–100 held-out scenes at `N = 32k` for evaluation

---

## 5. Implementation

- [ ] Fork/adapt RenderFormer codebase
- [ ] Implement BVH construction (SAH splitting) with **two levels exposed**: leaf level and parent level
  - [ ] Leaf level (block size b=32): used by the selected branch and local branch
  - [ ] Parent level (groups of ~2 leaves): used by the coarse branch
- [ ] Implement coarse branch: area-weighted pooling at **parent level** + parent-centroid RoPE
- [ ] Implement boundary fringe: at BVH build time, record k_fringe=4 closest triangles from neighbouring leaves for each leaf; include in coarse pooling
- [ ] Implement block scorer: H^I=4 heads, ReLU activation, FP16; takes triangle token → block token dot-products
- [ ] Implement dense warm-up training stage:
  - [ ] Freeze all params except block scorer
  - [ ] Run dense RenderFormer attention and capture per-head attention weights
  - [ ] L1-normalise summed attention weights across heads to get target p_{i,:}
  - [ ] Train scorer with KL divergence loss (Eq. ref{eq:scorer_warm_up}) for ~1000 steps
- [ ] Implement sparse fine-tuning stage (STE end-to-end):
  - [ ] Activate top-k block selection and normal-coherent mask
  - [ ] STE for top-k: forward = hard argmax; backward = straight-through scorer logits
  - [ ] STE for normal mask: forward = hard threshold; backward = sigmoid gradient (s=20)
  - [ ] Combined loss: L = L_render + 0.01 * L^I
- [ ] Implement selected branch: top-k leaf-level block selection using scorer + fine KV gather
- [ ] Implement local branch (3D kNN per query, k=64)
- [ ] Implement gated combination (Eq. 5 in paper draft)
- [ ] Extend 3D RoPE to coarse branch (parent-centroid as position)
- [ ] Implement sparse view-dependent cross-attention (frustum cull + kNN)
- [ ] Sanity check: at N=4096 with k_sel=128, output should match dense attention exactly
- [ ] Ablation: k_fringe ∈ {0, 2, 4, 8} — measure PSNR delta on emitter-heavy scenes
- [ ] Unit tests for each branch independently

---

## 6. Experiments

- [ ] Baseline: RenderFormer pretrained at N=4096 — PSNR = `___` (fill from Table 1 reproduction)
- [ ] Baseline: RenderFormer fine-tuned, N=8192 (dense, max memory) — PSNR = `___`
- [ ] Ours: N=8192 — PSNR = `___`
- [ ] Ours: N=16384 — PSNR = `___`
- [ ] Ours: N=32768 — PSNR = `___`
- [ ] Memory curve: log-log plot memory vs N for RF vs Ours
- [ ] Speed curve: log-log plot time vs N for RF vs Ours
- [ ] Ablation: -coarse branch, N=16k — PSNR = `___`
- [ ] Ablation: -selected branch, N=16k — PSNR = `___`
- [ ] Ablation: -local branch, N=16k — PSNR = `___`
- [ ] Ablation: block size b ∈ {16, 32, 64}
- [ ] Ablation: k_sel ∈ {2, 4, 8}
- [ ] Visual gallery: Cornell box, 3 Objaverse scenes, 1 room scene

---

## 7. Paper TODOs (in main.tex)

Search for `\todo` in main.tex to find all unfilled items:

- [ ] Abstract: fill PSNR gain, memory %, render time %
- [ ] Table 1: main results table (currently empty)
- [ ] Table 2: branch ablation table
- [ ] Figure: memory/time scaling log-log plot
- [ ] §4.3: fill complexity ratio (XX× fewer token pairs)
- [ ] §5.3: fine-tuning epoch count
- [x] LPIPS citation: `zhang2018lpips` added and verified (arXiv:1801.03924)
- [ ] Conclusion: update numbers once experiments done

---

## 8. Writing (apply skill before each submission milestone)

**Skill rules in force:** no em dashes as asides, no comma-appositives, no banned AI words, adjective economy, hedged claims, audience = academic/research reviewers.

**Dash and hyphen rules (standing — apply at every edit):**
- Hyphens are word separators in compound modifiers only: `well-known`, `state-of-the-art`, `back-face`, `fine-tuned`.
- Do NOT use `---` (em dash) or ` - ` (spaced hyphen) as an aside or explanation connector. Rewrite as a full clause or use parentheses.
- Do NOT use `--` as a prose connector. It is for numeric ranges (`30--40\%`) and LaTeX list rules only.
- Before every commit, run: `grep -n "\-\-\-" main.tex | grep -v "^%"` and fix every hit outside comments and header rules.

### 8a. Violations resolved (2026-05-20)

All items below were found and fixed in main.tex. Kept for audit trail.

- [x] Abstract: "remarkable quality" removed; replaced with specific claim structure
- [x] Abstract: em dashes around `\emph{coarse}, \emph{selected}, and \emph{local}` removed; rewritten as full clause
- [x] Introduction: comma-appositive on 3D~RoPE removed; rewritten as full clause
- [x] Related Work: em dashes around NSA branch list removed; rewritten with parentheses
- [x] Related Work: unhedged "first method" claim hedged with "To our knowledge,"
- [x] Related Work novelty sentence: `---the hemisphere constraint...---` rewritten as direct clause
- [x] Conclusion: comma-appositive "presented \ours, a hierarchically..." rewritten as two sentences
- [x] 3DGS defined on first use (line 105); HDR defined on first use (line 125)
- [x] LPIPS `\cite{todo}` replaced with `\cite{zhang2018lpips}` (verified arXiv:1801.03924)
- [x] `vaswani2017attention` cited at first O(N²) mention in §Related Work

### 8b. Section-by-section writing pass (do in order before arXiv)

- [ ] Abstract: fill numbers from experiments; verify no citations remain; target 150–220 words
- [ ] Introduction: re-read for sentence-length variety; check no consecutive sentences start the same word
- [ ] Related Work: check no consecutive sentences start the same word
- [ ] Background (§3): scan for passive voice; justify each design choice with a citation or explicit reasoning
- [ ] Method (§4): verify all equations are referenced in text; check no adjectives without specific meaning
- [ ] Experiments (§5): all numbers in tables; no vague descriptors ("significant", "substantial"); report ± std where applicable
- [ ] Conclusion: do not introduce new results; keep to 150 words
- [ ] **Dash audit:** `grep -n "\-\-\-" main.tex | grep -v "^%"` — must return zero prose hits
- [ ] **Hyphen audit:** `grep -n " - " main.tex | grep -v "^%\|verbatim\|math"` — must return zero prose hits
- [ ] Full pass: grep for "remarkable", "impressive", "strong", "novel", "intricate", "delve", "underscore", "pivotal" and replace each one
- [ ] Full pass: read aloud for sentence length variety (aim: short, medium, longer, short pattern)

### 8c. Sections still to be written

- [ ] Acknowledgments (funding source, compute, dataset access)
- [ ] Ethics / Broader Impact statement (required by NeurIPS; optional but advisable for CVPR/SIGGRAPH)
- [ ] Supplementary material outline: extended ablations, additional scene renderings, implementation details

---

## 9. Submission and Venue

- [ ] Confirm deadlines:
  - SIGGRAPH Asia 2026 — abstract deadline typically late May/early June 2026; check https://sa2026.siggraph.org
  - ICCV 2027 — paper deadline typically March 2027
  - 3DV 2026 — check https://3dvconf.github.io
- [ ] Choose one venue and commit; enter hard deadline here: `___`
- [ ] arXiv preprint submitted at least 1 week before conference deadline
- [ ] Anonymous submission: comment out `\cvprfinalcopy` and remove author block
- [ ] Page limit check: CVPR/SIGGRAPH = 8 pages + refs; count pages before submitting
- [ ] Supplementary uploaded separately (PDF + video); confirm venue's size limit
- [ ] Confirm CMT / OpenReview / Submission system account is registered

---

## 10. Compute and Infrastructure

- [ ] Confirm GPU allocation: `___` (note: 8×A100-80GB for full retrain; single A100 sufficient for fine-tuning)
- [ ] Apply for cloud compute credits if needed:
  - [ ] AWS Research Credits: https://aws.amazon.com/research-credits
  - [ ] Google TPU Research Cloud: https://sites.research.google/trc
  - [ ] Microsoft Accelerate Foundation Models Research (AFMR): https://www.microsoft.com/en-us/research/collaboration/accelerate-foundation-models-research
- [ ] Set up experiment tracking (Weights & Biases or MLflow) before first training run
- [ ] Back up pretrained checkpoint to CMU storage or Google Drive before fine-tuning

---

## 11. Code and Release

- [ ] Create public GitHub repo (name suggestion: `sparserenderformer`)
- [ ] Add MIT or Apache 2.0 LICENSE file
- [ ] Write README: installation, quick-start demo, pretrained checkpoint link, citation block
- [ ] Jupyter notebook demo: load checkpoint → render Cornell box → render one Objaverse scene
- [ ] Upload fine-tuned checkpoint to HuggingFace Hub under your account
- [ ] Add `requirements.txt` pinning PyTorch, Triton, trimesh versions
- [ ] Tag a `v1.0` release at submission time

---

## 12. Promotion (after arXiv)

- [ ] Tweet/X thread linking arXiv: 3–5 key figures, one-sentence plain-language summary
- [ ] Post to r/MachineLearning and r/computergraphics
- [ ] Email RenderFormer authors (Chong Zeng: Microsoft Research Asia) to notify — polite one-paragraph note, link to preprint. They may retweet or provide feedback.
- [ ] Add to Papers With Code once arXiv is live

---

## 13. Upcoming arXiv Checks (to avoid being scooped)

| Date | Search terms | Result |
|------|-------------|--------|
| 2026-05-20 | renderformer, neural global illumination transformer, BVH attention | Nothing found |
| 2026-06-20 | (fill in) | |
| 2026-07-20 | (fill in) | |
| 2026-08-20 | (fill in) | |

---

## Notes

- RenderFormer paper: arXiv:2505.21925 (SIGGRAPH 2025). Authors: Zeng, Dong, Peers, Wu, Tong (Microsoft Research Asia).
- NSA paper: arXiv:2502.11089. The three-branch design (compressed, selected, local) is the direct inspiration.
- Do NOT use Equiformer "V3" as a citation — no confirmed V3 paper as of 2026-05-20; use EquiformerV2 (arXiv:2306.12059).
- DiffusionRenderer is a real CVPR 2025 Oral (arXiv:2501.18590); cite it in related work, not as a direct competitor (different input modality: G-buffers vs triangles).
- LPIPS citation to verify before use: Zhang et al. CVPR 2018, arXiv:1801.03924.
