# Research notes: toward a publishable characterization study

This file is a **paper-in-progress**, distinct from `APPLE_SILICON.md` (the chronological lab
notebook this work was developed in — keep using that for day-to-day findings; port results here
once they're stable enough to belong in a paper draft rather than a running log).

## Publication assessment (as given, 2026-07-21)

Honest assessment: not in its current form, but it's closer to a workshop paper or systems
technical report than nothing, and there's a clear path to strengthen it.

**What's working in its favor:**
- Real methodological rigor — reading actual SDK headers instead of trusting docs/web claims,
  adversarially testing hypotheses (the batched-dispatch attempt directly falsified the initial
  "dispatch count is the bottleneck" hypothesis, then the flash-attention result partially
  confirmed the follow-up hypothesis), correctness-before-speed discipline throughout.
- A genuinely underexplored area: Apple's M5 "Neural Accelerator" and Metal Performance Primitives
  (`matmul2d`) are extremely new, and empirical characterization of how they actually perform in
  practice (vs. Apple's marketing/docs) is exactly the kind of thing that's currently thin in the
  literature. The finding that `matmul2d` doesn't reliably beat native MPSGraph/`nn.Linear` across
  many operand-type combinations, and that full int8×int8 only reaches parity rather than winning,
  is a real, non-obvious empirical result.

**What's missing for a top-tier venue (MLSys, ASPLOS, ISCA-tier):**
- No novel algorithm or technique at the *kernel* level — the `matmul2d`/flash-attention work
  applies known approaches via existing Apple APIs, not a new technique. **Partially addressed
  since this assessment was written**: Section 4.8 (CFG difference-caching) is a genuinely new
  application — adapting FasterCache/AB-Cache's difference-caching mechanism to TRELLIS's specific
  rectified-flow + guidance-interval sampler, including root-causing why a naive version failed and
  fixing it against the real published mechanism. Still modest next to a full novel algorithm, and
  still only a fixed-interval approximation of AB-Cache's fully adaptive trigger — but it's a real
  design contribution, not just a characterization result, and worth leading with more than the
  kernel work if this is submitted anywhere.
- Single hardware target (one M5 Max machine), single model (TRELLIS), no comparison against other
  Apple Silicon inference stacks (MLX, llama.cpp's Metal backend) that reviewers would expect as
  baselines.
- Benchmark rigor is engineering-grade, not paper-grade: mostly single-run wall-clock timings, no
  repeated trials with variance/confidence intervals, no statistical significance testing.
- Several core findings are negative results specific to one API on one chip generation — valuable
  as documented knowledge, but thin as a "contribution" in the conference sense.

**What would close the gap:** broaden to multiple Apple Silicon generations (M1–M5) to turn this
into a real characterization study, add statistically rigorous repeated-trial benchmarking, compare
against MLX/other frameworks as baselines, and frame the contribution explicitly around the
empirical characterization of MPP/Neural Accelerator behavior (the most novel, least-covered angle)
rather than around the TRELLIS port itself. In that form, a workshop paper or a technical report
(arXiv, similar in spirit to the "Rigel" M4 reverse-engineering paper already cited in this
session's research) is realistic. A top-tier main-track paper would need a genuine algorithmic or
architectural contribution on top of the characterization, which isn't present yet.

## Working title (draft, open to change)

**"Does Metal Performance Primitives Deliver? An Empirical Characterization of `matmul2d` and the
M5 Neural Accelerator Through Int8 Quantization and Flash-Attention Kernels"**

Reframes the contribution around the *characterization*, per the assessment above — the TRELLIS
port is the vehicle/testbed, not the subject.

## Draft outline, populated with real numbers already measured

All numbers below are from this session's actual runs on one Apple M5 Max (`Mac17,6`, macOS
26.4.1, SDK 26.5), single-run wall-clock unless noted — **flagged explicitly as a limitation to fix
before submission**, not presented as final results here.

### 1. Introduction
- Apple's M5 (Oct 2025) introduced per-GPU-core "Neural Accelerators" for int8/fp16 matmul
  throughput — the first Apple Silicon generation with dedicated low-precision matrix hardware.
  Metal Performance Primitives (`matmul2d`, Metal 4.1) is the public API surface for this.
- Motivating question: for a real, deployed workload (image-to-3D diffusion transformer inference),
  does hand-written `matmul2d`-based quantization actually deliver a speedup over PyTorch's
  existing native MPS path (`MPSGraph`-backed `nn.Linear`/`scaled_dot_product_attention`)?
- Contribution: a systematic, falsification-driven empirical study answering "no, not reliably" —
  and characterizing *why*, down to the operand-type-combination and dispatch-batching level.

### 2. Background
- TRELLIS (Xiang et al., arXiv:2412.01506): rectified-flow image-to-3D diffusion, two large
  denoisers (`sparse_structure_flow_model` 555.5M params, `slat_flow_model` 600.4M params), 24
  transformer blocks each with self-attention + cross-attention + FFN.
- Metal Performance Primitives `matmul2d`: supported operand-type table (half/int8/uint8/float/
  bfloat/int4/uint4 in ~46 combinations), `relaxed_precision` flag, `cooperative_tensor` for
  register-resident chaining, `multiply_accumulate` mode.
- PyTorch MPS backend: `torch.mps.compile_shader`/`load_metallib` for zero-copy custom kernel
  dispatch, `torch.library.custom_op` + `torch.compile` for kernel-boundary fusion.

### 3. Methodology
State the four-stage verification protocol actually followed throughout (this IS a real
methodological contribution worth stating explicitly, reviewers like stated protocols):
1. Synthetic correctness (varied shapes, non-tile-aligned batch/sequence sizes, boundary cases)
   against both the module's own fallback and a CPU fp32 reference.
2. Real-model finite-output check with real conditioning (not synthetic noise) across multiple
   trials.
3. Isolated per-call microbenchmark at real production shapes.
4. Full real-model benchmark — "the only number that matters," given isolated and real-model speed
   are shown (Section 4) to diverge significantly.

### 4. Results
**4.1 Quantized FFN**: fused `fc1→GELU→fc2` via native int8 `matmul2d` + `custom_op` + shared
`torch.compile`. FFN's true time-share is ~11% of full-model forward time (measured by bypass, not
estimated from parameter count, which suggests 36%) — an architectural ceiling of ~1.12x max
whole-model speedup regardless of kernel quality. Measured: 1.11x *slower* overall.

**4.2 Quantized attention**: applying the same pattern to self/cross-attention Linears
(`to_qkv`/`to_q`+`to_kv`/`to_out`), SDPA left native. Attention's true time-share: ~77% of
full-model time (bypass-measured) — a much larger ceiling than FFN. After fixing a real bug
(matmul2d always returns fp32; this model runs fp16; SDPA was silently running in fp32 until
fixed): self-attention 0.972x (beats native), cross-attention 1.189x (slower, structural — three
separate quantized Linears vs. self-attention's two). Full real-model result: 0.98x.

**4.3 Operand-type-combination sweep**: 6 of ~46 documented `matmul2d` combinations tested at the
real `to_qkv` shape (`num_rows=4096, in_features=1024, out_features=3072`):

| combination | ms/call | vs. native fp16 |
|---|---|---|
| half × int8_t → float (production) | 0.7992 | 1.94x slower |
| half × half → float | 0.7952 | 1.93x slower |
| half × half → half | 0.8272 | 1.88x slower |
| bfloat × bfloat → float | 0.8283 | 1.88x slower |
| bfloat × int8_t → float | 0.8478 | 1.83x slower |
| int8_t × int8_t → int32 (full W8A8) | 0.4086 | 1.01x, parity |
| native fp16 `nn.functional.linear` | 0.4123 | 1.0x baseline |

Finding: the bottleneck is the presence of ANY `half`/`bfloat` operand, not `int8` specifically;
only full int8×int8 closes the gap, and only to parity, at a real accuracy cost (median rel. error
0.0111 vs. 0.0078).

**4.4 `relaxed_precision`**: documented flag ("sacrifice accuracy for performance"), tested
directly — clean null (byte-identical output across all shapes, 1.003x).

**4.5 Attention-matmul dispatch batching**: naive per-head Python loop for Q@Kᵀ measured 2.8x
slower than `torch.bmm` (root cause: 16 separate dispatches vs. `bmm`'s one). Fix (head index in
the Metal dispatch grid's z-dimension, all heads in one dispatch): 15.04ms vs. the loop's 15.28ms
(1.016x) — **falsifies** the dispatch-count hypothesis; both ~3x slower than native SDPA (5.0ms).

**4.6 Online-softmax (flash-attention-style) kernel**: threadgroup-memory-resident score tile and
running softmax statistics, avoiding the full L×L device-memory materialization implicated by 4.5's
result. Real correctness bug found and fixed via the same protocol (tail-block boundary case).
First (serial-bookkeeping) result: 8.41ms — 1.68x slower than native, but ~1.8x *faster* than the
non-flash batched version, closing roughly half the remaining gap — evidence *for* the
device-memory-materialization hypothesis. Follow-up: parallelized the per-row bookkeeping from 1
thread/row (64 of 128 threads idle) to 2 lanes/row via `simd_shuffle_xor`-based reductions.
**Result: 5.81ms — only 1.17x slower than native SDPA (4.98ms), the closest of any kernel variant
tested.** Full progression: per-head loop 15.28ms (3.05x) → batched non-flash 15.04ms (3.00x) →
flash serial 8.41ms (1.68x) → flash parallelized 5.81ms (1.17x) → native 4.98ms (1.0x).

**4.7 Sampling step-count reduction** (orthogonal to all kernel work above): TRELLIS's real
25-step `FlowEulerGuidanceIntervalSampler` tolerates fewer steps without a solver change (rectified
flow is designed for straight ODE paths). Real, visually-verified speedups: 1.42x at 15 steps,
1.68x at 10, 1.92x at 8, on one example image/seed.

**4.8 Classifier-free-guidance difference caching (the genuine architectural contribution)**:
confirmed directly in code that `GuidanceIntervalSamplerMixin` calls the model twice per step
(conditional + unconditional) within `cfg_interval=[0.5,1.0]`. A first, naive attempt — freeze the
unconditional prediction and reuse it for N steps, motivated by FasterCache-style CFG-caching
literature — showed real, visually-confirmed quality degradation that *grew* with cache interval
(mesh vertex count 313K→394K, extra debris/geometry not present in the baseline). Root-caused
against the actual FasterCache (arXiv:2410.19355) and AB-Cache (arXiv:2504.10540) mechanisms:
neither freezes a raw branch value — both cache *differences*, and AB-Cache explicitly confirms
guidance-scale amplification of stale-cache error (directly relevant: this pipeline's real
`cfg_strength=5.0`). Fix: cache `diff = pred − neg_pred` instead of `neg_pred` itself, derive the
reused negative prediction from the *fresh* conditional prediction each step
(`neg_pred_estimated = pred_fresh − diff_cached`), costing no extra model call. Result: mesh
vertex counts now stay stable (309K–316K, no monotonic drift), visual inspection at the most
aggressive interval matches baseline with no extra geometry, and speedup is real: 1.17x–1.29x.
This is a genuinely new experiment for TRELLIS's specific rectified-flow + guidance-interval
combination — not published anywhere as of this writing — and directly answers the "no novel
algorithmic contribution" gap in the publication assessment above, even if modestly scaled next to
the kernel characterization work. Still a fixed reuse interval, not AB-Cache's fully adaptive
error-gated trigger — a real partial step toward the technique, not a complete reimplementation.

### 5. Discussion
- Central finding: hand-written kernels via `matmul2d`, across every configuration tested, do not
  reliably beat PyTorch's native MPSGraph-backed path on this hardware generation. The one
  exception (full int8×int8) reaches parity, not a win, at an accuracy cost.
- The M5 Neural Accelerator's benefit (if fully realized) is apparently already captured by
  Apple's own native `MPSGraph` implementation — external, unverified evidence (MLX's own
  `sdpa_vector.h` source, fetched directly) suggests even Apple's own ML framework does NOT route
  attention through `matmul2d`/MPP, instead using manual scalar dot products with `simd_sum`
  reduction — a real, if circumstantial, corroborating signal.
- Real progress is possible with enough kernel sophistication (4.6), but doesn't close the gap
  with a correctness-first, moderately-parallel implementation — open question whether
  `cooperative_tensor` register-resident chaining (not yet attempted, higher risk) would.

### 6. Limitations (be upfront — reviewers will find these anyway)
- **Single machine, single chip (one Apple M5 Max unit)** — no cross-device variance data, no
  M1–M4 comparison to establish whether findings are M5-specific or general to Apple Silicon.
- **Single model (TRELLIS)** — results may not generalize to other transformer/diffusion
  architectures with different shape profiles.
- **Mostly single-run wall-clock timings** — no repeated-trial statistics, confidence intervals, or
  significance testing. This needs fixing before any submission (see Future Work).
- **No baseline comparison against other Apple Silicon inference stacks** (MLX, llama.cpp Metal
  backend, CoreML) — only against PyTorch's own native MPS path.
- Several results are negative findings specific to one API version/SDK snapshot (macOS 26.4.1 /
  SDK 26.5) — Apple's own implementation could change.

### 7. Future work / roadmap to strengthen
1. **Multi-generation characterization**: repeat the operand-combination sweep (4.3) and the
   flash-attention result (4.6) on M1/M2/M3/M4 hardware if accessible, to distinguish "M5-specific"
   from "Apple Silicon in general" — this is the single highest-leverage change for novelty.
2. **Statistical rigor**: N≥30 repeated trials per measurement, report mean/std/CI, not single
   wall-clock numbers. Retrofit onto the existing `scripts/investigate_*.py`/`benchmark_*.py`
   scripts (they already isolate the right measurements — this is a methodology upgrade, not a
   rewrite).
3. **Framework baselines**: add MLX and llama.cpp (Metal backend) as comparison points for at
   least the Linear-projection and attention benchmarks.
4. **Finish the `cooperative_tensor` attempt** (flagged as an open next step in `APPLE_SILICON.md`)
   — if it closes the remaining gap in 4.6, that's a much stronger result; if it doesn't, that's
   an even stronger negative result (rules out the "just wasn't sophisticated enough" objection).
5. Reframe every section to lead with the MPP/Neural-Accelerator characterization angle; move
   TRELLIS-specific pipeline details (Section 4.7, the step-count result) to an appendix or drop —
   it's a real result but orthogonal to the paper's actual contribution once reframed.

### 8. Related work (sources already surfaced this session, verify/expand before submission)
- Rigel: reverse-engineering the Metal 4.1 tensor compute path on M4 Max (arXiv, cited this
  session — closest existing work in spirit; confirm it's real and re-read in full before citing).
- MLX (`ml-explore/mlx`) — `sdpa_vector.h` source, fetched directly this session; real, citable
  primary source for "does Apple's own framework use MPP for attention."
- `philipturner/metal-flash-attention` — real `simdgroup_matrix`-based (pre-MPP) flash attention,
  reports ALU utilization numbers against CoreML, not against PyTorch MPS directly.
- SVDQuant (mit-han-lab) — low-rank + quantization outlier absorption, numbers are NVIDIA-only;
  relevant as a contrast case for "does this technique's numbers transfer to Apple Silicon."
- TRELLIS (Xiang et al., arXiv:2412.01506) — the testbed model.

## Status

**Update: this skeleton became a real, complete, compiled paper.** `paper/main.tex` (authors
Vinayak Pathak, Center for Lasers and Photonics, IIT Kanpur; Debabrata Goswami, Center for Bits
and Atoms, MIT) compiles cleanly to `paper/main.pdf` (pdflatex, 3 passes, zero unresolved
references) — real figures generated from this session's actual measured numbers
(`paper/generate_figures.py`), qualitative comparison figures built from real rendered outputs,
and a proper 2D-input/3D-output teaser (`paper/generate_teaser.py`) using this codebase's own
canonical-view convention rather than an arbitrary camera angle. A companion project page also
exists (`paper/project_page.html`, self-contained, all assets embedded) with the same results,
a video gallery of all 4 example objects plus an octree/radiance-field render, and an embedded
download link for the actual PDF.

**A real bug was found and fixed while building these assets, worth knowing before trusting any
of the figures' predecessors**: `GaussianRenderer`/`MeshRenderer` output was confirmed to be
180 degrees rotated from the object's real orientation (pre-existing, not introduced by this
paper work — see APPLE_SILICON.md's dedicated section). Every figure in the current `main.pdf`
has this corrected; `OctreeRenderer` never had the bug.

None of Section 6's limitations are resolved yet — this is a real draft, not a submission-ready
one. Next concrete step is still your call: (a) the statistical-rigor retrofit (cheapest, no new
hardware needed), (b) scoping multi-generation (M1–M4) access, or (c) simply deciding whether to
publish as-is (arXiv technical report framing, per the publication assessment above) given the
paper already exists and compiles.

---

# Part II: a second, architecture-level paper (opened 2026-07-21)

Distinct contribution from Part I above. Part I's ceiling analysis (Section 4.2/4.6) already
established that attention is ~77% of full-model compute time and that kernel-level work
(quantization, flash-attention, dispatch batching) tops out at 1.17x slower than native even after
real, hard-won kernel sophistication — a *kernel-quality* ceiling, not a *FLOPs* ceiling. Part II's
premise: the next real win has to come from **doing less attention compute architecturally**, not
a faster kernel for the same O(N²) math. This section is the design + ablation plan for that,
before any of it has been run — status is genuinely "proposed," not "measured," and should stay
marked that way until real numbers land.

## Hardware situation (as of 2026-07-21)

Two machines now, with different roles — this constrains which architectural ideas are even
worth attempting:
- **Apple M5 Max** (this port's whole existing body of work): the *deployment target*. Final
  speedup claims must be measured here, on the same `scripts/benchmark_*.py` infra already used
  for Part I.
- **DGX H200** (NVIDIA Hopper, CUDA, 141GB HBM3e): newly available, **training/fine-tuning/
  distillation only**. None of the Metal-specific kernel work (`matmul2d`, `compile_shader`) runs
  or means anything there — it's a CUDA box. Its role is to make the fine-tuning step that every
  architecture change below needs (see "Distillation protocol") actually cheap, which it was not
  going to be on a single Mac.

This split directly rules out one candidate architecture and rules in another:
- **State-space/Mamba hybrid attention (interleaving SSM blocks with a few full-attention
  blocks, à la Jamba/Hymba)** — considered, **deprioritized for now**. Selective-scan SSM kernels
  are CUDA-only (`mamba-ssm`, `causal-conv1d`); there is no Metal/MPS implementation, and writing
  one would be a whole separate kernel-engineering project (arguably its own paper) before any
  SSM-hybrid checkpoint could ever run on the actual M5 Max deployment target. Worth revisiting
  only if pursued explicitly as a CUDA-only side study with a different deployment claim, or after
  a Metal SSM-scan kernel exists.
- **Interleaved full/windowed attention** — the architecture this plan actually pursues. Windowed
  (`swin`-style) attention already has a real, working, non-experimental Metal/MPS implementation
  in this exact codebase (`trellis/modules/sparse/attention/windowed_attn.py`,
  `trellis/modules/attention` dense path — currently used only by the SLat VAE decoders, confirmed
  via direct code read, see chat history). A checkpoint fine-tuned on the H200 can be loaded back
  and benchmarked on the M5 Max with zero new kernel work — the infra to *run* it already exists
  and is already proven correct elsewhere in this codebase.

## Architecture proposal: Interleaved Full/Windowed Attention Flow Transformer

Both denoisers (`SparseStructureFlowModel`, `SLatFlowModel`) currently set `attn_mode='full'` on
every one of their 24 blocks (confirmed directly in `trellis/models/sparse_structure_flow.py:115`
and `trellis/models/structured_latent_flow.py:155` — not a documentation claim, read from the
actual instantiation call). Proposal: replace most blocks' self-attention with local windowed
attention (shifted every other windowed block, Swin-style, reusing
`SparseMultiHeadAttention(attn_mode='windowed')`, already implemented and already used elsewhere
in this codebase for the decoders), keeping a full-attention block only every `K` blocks to
preserve whatever long-range coherence the 3D structure needs (global symmetry, connectivity
across the whole voxel grid). This is a genuine FLOPs reduction on the ~77%-of-compute-time
component, not a kernel-quality improvement on the same computation — orthogonal to, and
compatible with, everything in Part I.

**Primary target: `SLatFlowModel`.** Its self-attention already goes through
`SparseMultiHeadAttention`, which already accepts `attn_mode='windowed'` with `window_size`/
`shift_window` — no new attention *kernel* code needed, only new per-block *configuration* plus
fine-tuning. Lowest-lift path to a real result.

**Secondary/stretch target: `SparseStructureFlowModel`.** Dense `16³=4096`-token patchified grid,
not a `SparseTensor` — its `MultiHeadAttention` class's `attn_mode='windowed'` path is currently
literally unimplemented (`raise NotImplementedError`, `trellis/modules/attention/modules.py:84,130`
— confirmed by direct read, this is a real gap, not a config flag away). Would need a new 3D
dense-window-partition op (shape-only reshape/permute, no new Metal kernel) written and verified
before this target is reachable. Attempt only after the SLat-model result is in, given the smaller
model (24 blocks, 4096 tokens vs. SLat's larger active-voxel count) would otherwise be tempting to
prioritize for iteration speed.

## Ablation plan

**Independent variables:**
- Full-attention interval `K ∈ {1 (baseline, no windowing), 2, 3, 4, 6, 8, 12, 24 (all windowed,
  no full block at all)}`.
- Window size `W ∈ {4, 8, 16}` (voxel neighborhoods for SLat; `W=8` is the value already validated
  in the existing VAE-decoder `swin` config, a reasonable starting prior).
- Distillation budget `∈ {0 (zero-shot attn_mode swap, expected to be badly broken — a necessary
  control, not a real candidate), 1k, 5k, 20k steps}` — tracks a quality-vs-training-compute curve,
  not just a single fine-tuned endpoint.
- Model: SLat flow model first; sparse-structure flow model only after the windowing op above
  exists.

**Distillation protocol** (why no paired 3D ground-truth dataset is needed): treat the existing
pretrained full-attention model as a frozen teacher and the windowed/interleaved variant as the
student, matching predicted velocity fields (and optionally intermediate block hidden states,
since block structure — residual, adaLN, MLP — is identical between teacher and student; only
`self_attn`'s internals differ) over `(cond, t, x_t)` triples sampled from the teacher's own
generation trajectories or random `t`/noise over a held-out conditioning-image set. Standard
technique for architecture distillation of diffusion/flow models — no need to touch the original
TRELLIS training data/labels at all. Fine-tune only the swapped self-attention modules (+ the
blocks immediately adjacent to each retained full-attention block) first; full end-to-end
fine-tuning as a fallback if targeted fine-tuning underperforms — H200's 141GB affords either
without needing LoRA/adapter tricks, though LoRA remains a cheap fallback if distillation data is
scarce or overfitting shows up early.

**Metrics per (K, W, distill-budget) configuration:**
- Attention time-share and full end-to-end speedup — measured on H200 first (fast iteration), then
  re-measured on the M5 Max for every configuration that survives to be a real candidate (the
  number that actually matters, per Part I's own "isolated vs. real-model speed diverge, never
  report isolated as real" lesson).
- Fidelity-to-teacher, paired per (image, seed): Chamfer distance between student and teacher
  output meshes at identical noise seed + conditioning image — a paired proxy that needs no
  external ground truth.
- Mesh vertex-count stability across the same 6 (image, seed) combinations already used for the
  CFG-diff-caching and step-count verification in Part I (reuses an established protocol rather
  than inventing a new one).
- CLIP/DINO similarity between a rendered turntable of the output and the conditioning image
  (reuses the pipeline's own `get_cond()` DINOv2 features already available).

**Statistical rigor, built in from the start** (Part I's own Section 6 flags single-run wall-clock
as its biggest weakness — don't repeat that mistake here): N≥10 repeated trials per configuration,
report mean/std, given H200 compute makes this affordable where it wasn't before.

**Success criterion / what the paper's core figure actually is:** not a single "winning"
configuration, but the full Pareto frontier of (speedup vs. fidelity-to-teacher) across every
tested `(K, W)` pair, at each distillation budget. A configuration is only worth highlighting as a
recommended default if it clears both a minimum real M5-Max speedup bar (informed by, not fixed in
advance of, the measured curve) and a fidelity band comparable to what Part I's CFG-diff-caching
work already established as "stable" (vertex count within the same tight range as its baseline,
no visually-confirmed extra/missing geometry).

## Compute environment (set up 2026-07-21)

A personal DGX pod (8x H200, 143GB each) is now reachable and has a working environment for this
work: conda env `trellis` at `~/trellis-research` on the pod (repo synced via rsync, excluding
`.git`/`outputs`), PyTorch 2.4.0+cu121, TRELLIS's `BASIC` dependency set from `setup.sh` installed.
Verified with a real forward pass on real GPU hardware: `SLatFlowModel` (full attention) and
`SLatEncoder` with `attn_mode='swin'` (the windowed-attention path this proposal depends on) both
pass finite-output checks on an actual H200, not just synthetic/CPU.

Two real gotchas hit while setting this up, worth knowing before repeating this on another
machine:
- **`PYTHONNOUSERSITE=1` is required** for both `pip install` and every `python` invocation in
  this conda env. A stray earlier `pip install --user` left packages in `~/.local` that silently
  shadow the conda env's own site-packages (Python includes user-site by default regardless of
  being inside a conda env), causing version-mismatched imports (a `transformers` release far
  newer than the installed `torch` supports) that only reproduced inside this specific shell
  context. Without this flag, dependency resolution silently uses whatever's already importable
  across *both* locations, which does not correctly reflect what's actually installed in the env.
- **This box is shared, not exclusive.** `nvidia-smi` shows GPU indices 0-1 permission-restricted
  to this pod, and indices 2/3/4/7 have other active jobs already using significant memory
  (confirmed via `nvidia-smi --query-compute-apps`). GPU 5 was chosen for this work (best
  measured headroom, ~24GB free). Also found: `nvidia-smi`'s reported free memory and what
  `torch.cuda.mem_get_info()` reports *from inside this container* can disagree substantially on
  the same GPU (seen on index 4) — this container's memory accounting is not fully trustworthy
  against the host-level view, so don't assume a GPU is safe to use for something large just
  because one of the two numbers looks free.

## Cross-step cross-attention K/V caching — measured, real result (2026-07-21)

Implemented in `trellis/modules/attention/cross_kv_cache.py` (non-destructive: patches
`cross_attn.to_kv`'s `nn.Linear.forward` at the instance level via `types.MethodType`, no edits to
any existing model/block file). Verified against this project's full 4-stage protocol
(`scripts/verify_cross_attn_kv_cache.py` for stages 1-2, `scripts/benchmark_cross_attn_kv_cache.py`
for stages 3-4):

- **Stage 1 (synthetic correctness)**: cached output bit-identical to uncached, hit/miss
  accounting exact, disable/re-enable round-trips cleanly. All passed.
- **Real-model finding that changed the design**: identity-only caching (`is`) got **zero hits**
  on the real pretrained model. Root cause: both flow models' `forward()` does
  `cond = cond.type(self.dtype)` before handing it to blocks, and when a real cast is needed
  (real conditioning is fp32, model runs fp16) `.type()` allocates a fresh tensor *every single
  call* — even though the value is identical every sampling step. (Synthetic testing missed this
  because the synthetic model used `use_fp16=False`, making the cast a genuine no-op that preserves
  identity — a real case where a synthetic test's default config accidentally hid a real-model-only
  bug.) Fixed by falling back to `torch.equal` (after a shape/dtype/device guard) when identity
  fails — Stage 2 then passed with the exact predicted hit/miss counts (192 hits, 48 misses across
  24 blocks x 2 contexts x 5 simulated steps) and bit-identical output.
- **Stage 3 (isolated)**: 152x speedup on `to_kv` alone (0.117ms -> 0.0008ms/call) — as expected,
  since caching skips the recomputation entirely.
- **Stage 4 (real full-model, simulated 25-step CFG loop, real image conditioning)**: **0.975x —
  no real speedup, marginally slower**, despite the cache functioning correctly (1200 hits, 96
  misses, matching the expected hit rate). Single-run wall-clock, on this port's now-standard
  limitations list, but a large enough gap from parity that repeated trials are unlikely to flip
  the conclusion.

**Honest read, matching this project's own established pattern** (see Part I's FFN-caching
ceiling, §4.1): cross-attention's `to_kv` Linear is evidently too small a slice of total per-step
compute — dominated by self-attention, cross-attention's own SDPA call, and the FFN — for its
recomputation to matter at the full-model level, and the Python-side cache-lookup overhead (even a
cheap `torch.equal` fallback) roughly cancels whatever it saves. **This is a real negative result,
not a bug to keep chasing** — kept in the codebase (per this port's versioning convention: working,
verified, just not a win) since it's a legitimate building block if a future architecture change
makes cross-attention a larger relative share of compute, but not something to lead with as a
contribution on its own.

## Interleaved attention: implemented, verified, and wall-clock-benchmarked BEFORE distillation (2026-07-21)

Implemented in `trellis/modules/sparse/attention/interleave_windowed.py` — the same
non-destructive attribute-mutation technique as the K/V cache (`attn_mode` is a pure forward-time
flag on `SparseMultiHeadAttention`, not something that changes parameter shapes, confirmed by
reading its `__init__`; converting a block from full to windowed attention is a plain attribute
mutation, no reinitialization). Shift pattern reuses, verbatim, the exact alternation already
proven in the working SLat VAE encoder/decoder (`structured_latent_vae/base.py`'s `"swin"` mode:
`shift_window = window_size // 2 * (i % 2)`) rather than inventing a new one.

**Verified** (`scripts/verify_interleaved_attention.py`) against this project's protocol, with two
real, non-obvious findings caught along the way (both false alarms about the patch itself, both
real facts about the model):
- A freshly-constructed (untrained) `SLatFlowModel` produces literally all-zero output regardless
  of any internal change, because DiT's zero-init convention zeros **both** the adaLN modulation
  gates (already documented at the block level) **and** `out_layer`'s weight/bias (the final
  projection) — the initial synthetic test showed K=2 windowed output "suspiciously identical" to
  the full-attention baseline purely because both were exactly zero. Fixed by perturbing both for
  test purposes only (a real pretrained checkpoint already has learned nonzero values here).
- Once real (nonzero) output was flowing, exact bit-identical round-trip checks (K=1 no-op,
  restore-after-mutation) started failing — root-caused to genuine, pre-existing, ~3e-8-magnitude
  non-determinism in `SparseResBlock3d`'s sparse convolution (`SPARSE_BACKEND=pytorch`, likely
  scatter/gather accumulation-order variance on MPS), confirmed unrelated to the attention patch
  (a bare `SparseMultiHeadAttention` instance, isolated from the conv layers, IS perfectly
  bit-deterministic across repeated calls). Checks now use `torch.allclose` at a tolerance well
  above that noise floor for anything that goes through the full model.

**Real wall-clock benchmark, deliberately run BEFORE building the distillation pipeline**
(`scripts/benchmark_interleaved_attention.py`, real `slat_flow_model`, real conditioning, ~3000
active voxels, `window_size=8`, `K` swept 1→24):

| K (full-attn interval) | blocks windowed | ms/forward | speedup |
|---|---|---|---|
| 1 (baseline) | 0/24 | 311.05 | 1.000x |
| 2 | 12/24 | 306.43 | 1.015x |
| 3 | 16/24 | 299.83 | 1.037x |
| 4 | 18/24 | 300.66 | 1.035x |
| 6 | 20/24 | 298.67 | 1.041x |
| 8 | 21/24 | 297.39 | 1.046x |
| 12 | 22/24 | 296.80 | 1.048x |
| 24 (only block 0 full) | 23/24 | 296.00 | 1.051x |

**Honest read: this is a real, disappointing ceiling, not yet a win worth distilling for.** Even
the most aggressive setting tested (23/24 blocks windowed) only reaches **1.05x** — nowhere near
what Part I's "attention is ~77% of compute time" bypass measurement would suggest. Plausible
explanations, **not yet confirmed, flagged as open questions rather than asserted facts**:
- That 77% figure covers self-attention **and** cross-attention combined; this change only touches
  self-attention (`SparseMultiHeadAttention` structurally disallows windowed cross-attention —
  `assert type == "self" or attn_mode == "full"`, confirmed in `modules.py`). If cross-attention is
  the larger share of that 77% (plausible: Part I §4.2 already found cross-attention's quantized
  Linears are structurally slower than self-attention's), this change was only ever addressing the
  smaller piece.
- Windowed attention's gather/scatter/partition machinery has real overhead that may be eating
  most of the FLOPs saved at this problem size (~3000 active voxels) — the same
  isolated-vs-real-model divergence this project has hit repeatedly with custom kernels (Part I
  §4.5/4.6), now possibly recurring at the architecture level instead of the kernel level.

**Not distilling yet** (superseded below — see "window_size sweep").

## Self- vs. cross-attention time split — measured, resolves the open question above (2026-07-21)

`scripts/investigate_self_vs_cross_attention_split.py`, same bypass-measurement technique as
above, applied separately to `block.self_attn` and `block.cross_attn` on real `slat_flow_model`,
real conditioning, ~2982 active voxels:

| bypassed | time | share of baseline (310.60ms) |
|---|---|---|
| self-attention | 219.45ms | **29.3%** (91.15ms) |
| cross-attention | 271.86ms | **12.5%** (38.75ms) |
| both | 180.71ms | 41.8% (129.89ms — matches the sum of individual shares, 129.90ms, confirming no interaction/overlap effect between the two bypasses) |

**Self-attention is the larger of the two (2.35:1 ratio) — this refutes the hypothesis above.**
Cross-attention is not secretly the dominant cost; the interleaving work was already targeting the
right component. But this sharpens a different puzzle: self-attention being 29.3% of total time
implies a theoretical ceiling of up to **1.42x** (310.60/219.45) if self-attention were free, yet
the actual windowed-attention result only reached 1.05x — capturing roughly 15 of the possible
91ms. That gap pointed at the *windowed-attention implementation itself*, not the algorithmic
idea, as the next thing to check.

## Isolated windowed-attention speed, and a window_size sweep that meaningfully improves the ceiling

`scripts/investigate_windowed_attn_isolated_speed.py` isolated `block.blocks[0].self_attn` alone
(real weights, real shape captured via an instrumented forward-hook rather than assumed — the
real per-block token count is 2867, barely different from the 2982-voxel raw input despite one
downsample stage in `io_block_channels`, because random synthetic points are sparse enough
relative to the voxel grid that downsampling rarely merges them; a genuine near-miss of the
project's own "always capture real shapes" lesson, checked and found not to matter much here, but
worth having actually checked rather than assumed):

| config | ms/call | vs. full attention |
|---|---|---|
| full attention (baseline) | 4.08ms | 1.0x |
| windowed, `window_size=4` | 26.80ms | **0.152x (6.6x slower)** |
| windowed, `window_size=8` | 8.46ms | **0.483x (2.1x slower)** |
| windowed, `window_size=16` | 2.79ms | 1.466x (genuinely faster) |

`window_size=8` (the value already trusted elsewhere in this codebase for the VAE decoders) is
actually **slower than full attention in isolation** — smaller windows mean more of them, and the
partition/gather/scatter machinery's overhead dominates at this problem scale. This isolated
result initially seemed to contradict the full-model benchmark (which showed windowing *helping*,
not hurting) — checked directly for a warm-up/test-order confound by rerunning the full K-sweep in
reverse order; the trend replicated almost exactly, ruling that out. Real, non-contradictory
explanation, consistent with this project's own already-documented isolated-vs-real-model
divergence pattern: MPS's asynchronous command-buffer dispatch lets one block's windowed-attention
host-side overhead overlap with other blocks' GPU compute in the full pipelined forward pass —
invisible in a tight isolated loop with nothing to overlap against, but real in context.

**Re-ran the full-model K-sweep at `window_size=16` (and 32) given the isolated result above**
(`scripts/benchmark_interleaved_attention.py`, extended to sweep `window_size` too):

| K | W=8 | W=16 | W=32 |
|---|---|---|---|
| 1 (baseline) | 1.000x | 1.000x | 1.000x |
| 2 | 1.011–1.015x | 1.054–1.058x | 1.064x |
| 4 | 1.033–1.037x | 1.088–1.091x | 1.051x |
| 8 | 1.041–1.046x | 1.106–1.108x | 1.003x |
| 24 (most aggressive) | 1.048–1.052x | **1.108–1.123x** | 1.010x |

**`window_size=16` is the clear winner**: smooth, monotonic improvement with `K`, real ceiling
around **~1.11–1.12x** at the most aggressive setting — meaningfully better than `window_size=8`'s
~1.05x, though still well below the 1.42x theoretical max from the bypass split.
`window_size=32` doesn't help further and gets noisier (only 2 windows/axis at this model's
resolution=64, average window occupancy rises enough that it starts resembling full attention's
own cost profile without matching its efficiency).

## Status

**~1.11x is a real, if modest, architecture-level win** — smaller than CFG-diff-caching's
1.17-1.30x (Part I §4.8) but in the same tier, and orthogonal/stackable with it (both apply
simultaneously: this changes per-step compute, CFG-caching changes how many full model calls
happen). Worth a real distillation pass now, unlike at the 1.05x ceiling that motivated holding off
earlier. Next steps, in order:
1. **Stand up the distillation data pipeline** (sampling `(cond, t, x_t)` triples from the teacher)
   on the H200, targeting `window_size=16`, `K` swept starting from the most promising settings
   above (K=8 through 24 all cluster tightly around 1.10-1.12x, so quality-vs-speed tradeoff across
   that range is the real open ablation question, not raw speed).
2. Consider a quick shift-pattern/window-alignment sweep at `window_size=16` before committing
   H200 time to the full ablation matrix — not yet checked whether the specific shift pattern
   (borrowed from the VAE decoder's `swin` config) is itself tuned for this different use case.
3. Write and synthetically verify the dense 3D-window-partition op needed for the
   `SparseStructureFlowModel` stretch target — still deprioritized relative to the SLat path above,
   but no longer blocked on an unresolved ceiling question.
4. Cross-attention's own 12.5% share is smaller than self-attention's but not negligible — a real,
   separate future direction (would need its own design, since `SparseMultiHeadAttention`
   structurally requires `attn_mode="full"` for cross-attention), noted but not pursued now.

## First real distillation run — pipeline validated end-to-end, loss decreases (2026-07-23)

Stood up on the DGX pod: `scripts/dgx_generate_distillation_data.py` (real teacher-trajectory
`(cond, x_0)` pairs — runs the actual pretrained teacher's own sparse-structure + SLat samplers on
all 42 `assets/example_image/*.png` images, not synthetic noise) and
`scripts/dgx_train_interleaved_attention.py` (student warm-started from the teacher's own
pretrained weights — valid because `attn_mode` is a pure forward-time flag, not a reinitialization
— only the windowed blocks' `self_attn` parameters trainable, everything else frozen, matching the
ablation plan's "fine-tune only the swapped self-attention modules... first"). Both CUDA-only, no
MPS code, per this project's platform-separation convention.

**First real run**: `K=8`, `window_size=16` (21/24 blocks windowed, 88.21M trainable params), 300
steps, `lr=2e-5`, real teacher-trajectory data. **Loss dropped from an average of 0.0326 (first 20
steps) to 0.0144 (last 20 steps) — a real ~2.3x reduction**, clean exit, no errors. Confirms the
distillation pipeline is functionally correct end-to-end (real data generation, teacher/student
setup, gradient flow restricted to the intended parameters, optimizer convergence) — a real,
positive signal, not yet a completed result.

**What this does and doesn't establish, stated plainly**: a decreasing MSE-on-predicted-velocity
loss confirms the student is learning to match the teacher's output at the level this loss
measures, but does NOT by itself confirm the student's actual 3D generation quality matches the
teacher's (visual/structural quality could still differ even at low velocity-MSE, the same caution
this project already applied to CFG-diff-caching in Part I, which used mesh vertex counts + visual
inspection rather than trusting a proxy metric alone). **Next validation step, not yet done**: run
the trained student's full weights through an actual multi-step generation (not just single
`(x_t, t)` steps) and compare output quality to the teacher's on held-out images, the same
protocol already established and trusted for CFG-diff-caching.

Trained checkpoint saved: `distillation_data/student_K8_W16.pt` (352MB, git-ignored — trainable
state dict for the 21 windowed blocks' `self_attn` + the full loss history + the block-assignment
dict, regenerable by rerunning the training script against the same data).

## Generation quality check — real, nuanced result, not a clean pass (2026-07-23)

`scripts/dgx_generation_quality_check.py`: ran the actual SLat sampler twice per test image
(teacher vs. the fine-tuned student, cast to fp16 to match production inference — the checkpoint
was trained fp32), using the **same sparse-structure coords and the same initial noise** for both,
isolating the architecture change from any other source of variation. Deliberately avoided
anything needing compiled CUDA rendering extensions (this pod has no CUDA toolkit, `nvcc` not
found) — used mesh decoding (`slat_decoder_mesh`, pure PyTorch via FlexiCubes, confirmed no
compiled-extension dependency) plus vertex/face counts and a point-based Chamfer distance instead
of rendered-pixel comparison, the same style of proxy this project already trusted for CFG-diff-
caching in Part I. **Test images are genuinely held out** — `assets/teaser.png` and 7 from
`assets/example_multi_image/` — none were in the 42-image distillation training set (which only
used `assets/example_image/*.png`).

**Results, 8/8 held-out images:**
| metric | result |
|---|---|
| both teacher and student produced a valid mesh | 8/8 |
| mean SLat feature relative error (same noise/coords) | **0.487** (range 0.25–0.65) |
| mean Chamfer distance (paired meshes) | 0.0125 |
| mean student/teacher vertex-count ratio | 1.001 |

**Two findings that are in real tension, reported honestly rather than resolved prematurely:**
mesh-level structure looks very close to the teacher's (near-identical vertex/face counts, small
Chamfer distance relative to the object's ~1-unit scale — e.g. teacher 301,838v vs. student
303,932v on the teaser image), but the **raw SLat latent diverges substantially** (48.7% mean
relative L2 error) using literally the same noise and coordinates for both runs — not seed
variance, a real effect of the architecture swap + 300-step fine-tune. These aren't strictly
contradictory (the decoder may map a range of nearby latents to similar geometry, and mesh
decoding doesn't exercise the SLat channels that drive color/texture in the gaussian/radiance-field
decoders — this check only validates geometry), but the gap is large enough that **this is not a
clean pass** and shouldn't be reported as one.

**Calibration baseline — run, and it splits the explanation cleanly (2026-07-23).**
`scripts/dgx_seed_variance_baseline.py`: same 8 held-out images, same unmodified teacher run
*twice* with two different noise seeds (same coords/conditioning), same SLat feature relative-error
metric. Result: **mean 0.3367** (range 0.23–0.42) — a large number on its own, confirming a real,
substantial chunk of the raw 0.487 figure is just this rectified-flow sampler's inherent
sensitivity to its starting noise, present even with zero architecture change. But 0.487
(architecture change, same seed) is still meaningfully higher than 0.337 (natural seed-to-seed
variation, no architecture change) — roughly **1.45x** — so the windowed-attention distillation is
adding a real, measurable amount of *extra* divergence on top of the model's already-substantial
inherent stochastic sensitivity, not merely reproducing it. Combined with the mesh-level result
above (near-identical vertex counts, small Chamfer distance), the coherent synthesis: the
architecture change has a real but moderate effect beyond ordinary sampler noise, and that effect
doesn't appear to meaningfully degrade the decoded geometry at this checkpoint (K=8, W=16, only
300 training steps). Not yet known whether more distillation steps would close more of that 1.45x
gap, or whether it's an irreducible property of this specific windowing configuration.

**What's still missing before fully trusting this**, in order:
1. **An appearance-level check** (gaussian decoder, not just mesh) — this pod can't render pixels
   (no `nvdiffrast`/CUDA toolkit), but gaussian *splat count* and position statistics are still
   checkable without rendering, and would test the SLat channels this mesh-only check doesn't
   touch.
2. Actual rendered visual comparison, if `nvdiffrast`/`diff-gaussian-rasterization` can be gotten
   working on this pod (needs a real CUDA toolkit, not just the NGC runtime) or via the M5 Max
   (which has working renderers already, modulo the known orientation-bug workaround) — porting
   the fp16 student checkpoint back to Apple Silicon for a real render comparison is untested and
   not guaranteed to be straightforward.

**Real operational lessons from getting this running** (worth keeping for the next DGX session):
the pod got recreated mid-session (new host key, `~/trellis-research` wiped, but the `trellis`
conda env persisted on a separate volume); GPU memory contention is real and can spike suddenly
even on a nominally "exclusive" pod, because the `vs2` account is shared with another real user's
concurrent training jobs, not exclusive to this work; a stray/duplicate process from an earlier
attempt was found still holding ~44GB before being cleaned up; Python's stdout is fully buffered
when redirected to a file, so training progress was invisible in the log until the run's very end
despite the process genuinely computing the whole time (`PYTHONUNBUFFERED=1` fixes this for future
runs); and a `kubectl exec`-as-root trick (available from the jump host, which has cluster-admin
access to the pod) finally solved the "can't install an SSH key because `~/.ssh` is root-owned and
`vs2` has no sudo" problem cleanly, replacing password auth with key auth for both hops.

**Platform separation, going forward**: Apple-Silicon/MPS work (this Mac) and CUDA/DGX work stay
in separate scripts, never a single script branching on both. `scripts/verify_cross_attn_kv_cache.py`
/ `benchmark_cross_attn_kv_cache.py` / `verify_interleaved_attention.py` /
`benchmark_interleaved_attention.py` above are MPS-only (hardcoded `torch.mps.synchronize()`, run
and measured on the M5 Max). Any CUDA-side distillation/ablation scripts for the H200 pod get their
own separate files (`torch.cuda.synchronize()`, CUDA-specific env/dtype handling) rather than
adding a device branch to the MPS scripts.

**Platform separation, going forward**: Apple-Silicon/MPS work (this Mac) and CUDA/DGX work stay
in separate scripts, never a single script branching on both. `scripts/verify_cross_attn_kv_cache.py`
/ `benchmark_cross_attn_kv_cache.py` above are MPS-only (hardcoded `torch.mps.synchronize()`,
run and measured on the M5 Max). Any CUDA-side distillation/ablation scripts for the H200 pod get
their own separate files (`torch.cuda.synchronize()`, CUDA-specific env/dtype handling) rather than
adding a device branch to the MPS scripts — keeps the two hardware targets from silently
interfering with or masking each other's results.

## Extended distillation run (2026-07-23, completed — see results further below)

Added checkpoint-resume support to `scripts/dgx_train_interleaved_attention.py` (`--no-resume` to
opt out) so additional training builds on prior steps rather than restarting — the first 300-step
checkpoint resumed cleanly and was extended to 3000 total steps. Real bug hit and fixed along
the way: the dataset loader's `glob('*.pt')` in `distillation_data/` also picked up this script's
own checkpoint and other scripts' result dumps (42 real data files + 3 non-data files = 45 "pairs"
loaded), crashing when `random.choice` happened to select one of the non-data files (they don't
have the `{cond, feats, coords}` dict shape). Fixed by excluding known non-data filename prefixes
rather than assuming the file count. Final results are in "Extended training complete" below.

## Statistical rigor retrofit (2026-07-23)

Every window_size/K speedup number above was single-run wall-clock — flagged in this project's own
limitations list as a real gap. Re-ran `scripts/benchmark_interleaved_attention.py` (upgraded to
time each forward pass as an independent sample, n=8 per config, reporting mean ± std and a 95%
CI on both the raw timing and the derived speedup ratio via standard error propagation) on the
real M5 Max:

| K | W=8 speedup (95% CI) | W=16 speedup (95% CI) | W=32 speedup (95% CI) |
|---|---|---|---|
| 1 | 1.000x | 1.000x | 1.000x |
| 8 | 1.073x ± 0.045 | 1.112x ± 0.013 | 1.047x ± 0.009 |
| 24 | 1.077x ± 0.043 | **1.120x ± 0.014** | 1.008x ± 0.067 |

The core claim holds under real statistics, not just a single lucky run: `window_size=16`'s
K=24 result (1.120x ± 0.014) and `window_size=8`'s (1.077x ± 0.043) have **non-overlapping 95%
CIs** — genuinely, statistically distinguishable, not noise. `window_size=32`'s CI widens sharply
at K=24 (±0.067, an order of magnitude looser than W=16's), consistent with the earlier
qualitative observation that 32 was "noisier" — now quantified rather than just eyeballed. Full
per-trial raw data saved to `outputs/benchmark_interleaved_attention_stats.json`.

## Appearance-level check via the gaussian decoder — a real, more concerning finding (2026-07-23)

`scripts/dgx_generation_quality_check_gaussian.py` closes the "still missing" appearance-level
item flagged earlier — the mesh-only check never touched the SLat channels that drive color and
texture, since mesh decoding only reads geometry. Same protocol (same coords, same initial noise,
teacher vs. the 300-step K=8/W=16 checkpoint), but decoded through `slat_decoder_gs` instead of
`slat_decoder_mesh`. Gaussian count matches exactly (1:1 correspondence, same order) on all 8
held-out images, enabling a direct paired comparison instead of Chamfer distance:

| channel | mean relative error |
|---|---|
| position (xyz) | **0.0154** |
| color / SH features | **0.3464** |
| opacity | **0.1556** |
| scale | **0.4012** |

**Position is stable (consistent with the earlier mesh-level finding), but appearance-related
channels diverge substantially more** — color/SH error at 34.6% and scale error at 40.1% are an
order of magnitude larger than the 1.5% position error. This is a real, previously-invisible
finding: the earlier "decoded geometry looks very close to the teacher's" read was correct but
incomplete — it only ever tested the channels responsible for *where* things are, not what they
*look like*. A rendered comparison would very likely show visible color/texture/sharpness
differences even though the underlying shape matches closely.

**Calibration baseline run — and it rehabilitates the read, once properly compared
(2026-07-23).** `scripts/dgx_seed_variance_baseline_gaussian.py`: same 8 images, teacher run twice
with different seeds, same four channel-wise relative errors:

| channel | baseline (seed-only, no arch. change) | architecture change (same seed) | excess ratio |
|---|---|---|---|
| xyz | 0.0133 | 0.0154 | 1.16x |
| color / SH | 0.2861 | 0.3464 | 1.21x |
| opacity | 0.1359 | 0.1556 | 1.15x |
| scale | 0.3478 | 0.4012 | 1.15x |

The raw numbers (34.6% color error, 40.1% scale error) looked alarming in isolation, but most of
that magnitude turns out to be this model's own inherent sensitivity in these specific channels —
the seed-only baseline alone is already 0.29–0.35 for color/scale. **The actual excess
attributable to the architecture change is a modest ~1.15–1.21x across all four channels** —
smaller than the 1.45x excess found at the SLat-feature level, and consistent across position,
color, opacity, and scale rather than concentrated in one channel. **Net honest read across both
the mesh-level and gaussian-level checks**: the interleaved-attention distillation (K=8, W=16, 300
training steps) adds a real, measurable, but modest and consistent amount of extra divergence
across every geometry and appearance channel tested — not a clean pass, but not the more
concerning picture the raw uncalibrated appearance numbers suggested either.

## Extended training complete: 3000 steps, and the excess divergence genuinely shrinks (2026-07-23)

The extended run (300 → 3000 steps, resumed cleanly, see above) finished: loss dropped from a
first-50-step average of 0.0302 to a final-50-step average of **0.0062** — roughly a 5x reduction
over the full run, well past where the original 300-step checkpoint ended (0.0144). Re-ran both
quality checks (mesh and gaussian) against this fully-trained checkpoint, same 8 held-out images,
same protocol:

**Mesh-level**: vertex-count ratio now 1.000 (even tighter than 300-step's 1.001), Chamfer
distance essentially unchanged (0.0122 vs. 0.0125). SLat feature relative error improved from
0.487 to **0.440** — against the unchanged 0.337 seed-variance baseline, the excess ratio shrinks
from **1.45x to 1.31x**.

**Gaussian-level (appearance)**:

| channel | seed-variance baseline | 300 steps | 3000 steps | 3000-step excess ratio |
|---|---|---|---|---|
| xyz (position) | 0.0133 | 0.0154 (1.16x) | **0.0120** | **0.90x** — below baseline |
| color / SH | 0.2861 | 0.3464 (1.21x) | **0.3163** | **1.11x** |
| opacity | 0.1359 | 0.1556 (1.15x) | **0.1497** | **1.10x** |
| scale | 0.3478 | 0.4012 (1.15x) | **0.3302** | **0.95x** — below baseline |

**More training genuinely closed the gap, exactly as hypothesized when this was flagged as an
open question.** Two of four gaussian channels (position, scale) are now statistically
indistinguishable from — or even slightly better than — the model's own natural seed-to-seed
variation; the remaining two (color/SH, opacity) shrank from ~1.15–1.21x excess down to ~1.10–1.11x.
This is a genuinely encouraging final result for the K=8/W=16 interleaved-attention configuration:
real ~1.12x wall-clock speedup (statistically confirmed, Section above), paired with distilled
quality that, after adequate training, is close to indistinguishable from the teacher's own
inherent stochastic variation across every channel tested.

Final checkpoint (`distillation_data/student_K8_W16.pt`, 3000 steps) pulled back to the local
repo for safekeeping. The paper and project page have been updated with this final result,
replacing the earlier "training in progress" language.

## Real MLX baseline: a major finding, not just a closed gap (2026-07-23)

Part I's own Limitations flagged "no direct benchmark against MLX... only MLX's published kernel
source as qualitative corroborating evidence" as a real gap. Closed it directly:
`scripts/benchmark_mlx_comparison.py`, real MLX (already installed, v0.31.2) vs. real PyTorch MPS,
same process, same machine (M5 Max), at the exact two reference shapes used throughout this
paper's kernel-level section — `to_qkv`-shape Linear (4096×1024→3072) and self-attention SDPA
(B=1, H=16, L=4096, D=64). Confirmed MLX's `scaled_dot_product_attention` defaults to no mask
(matching PyTorch's default, non-causal, full attention) before trusting the comparison — this is
apples-to-apples, not an accidental masking shortcut. 20 trials/config after enough warmup that
variance stabilized:

| operation | PyTorch MPS | MLX | ratio |
|---|---|---|---|
| Linear/matmul (`to_qkv` shape) | 0.698ms ± 0.178 | 0.717ms ± 0.200 | 0.975x (parity, within noise) |
| Self-attention SDPA (real shape) | 5.465ms ± 0.202 | **1.361ms ± 0.084** | **4.02x — MLX faster** |

**This is not merely "MLX beats PyTorch" — it reframes part of the paper's central kernel-level
finding.** Part I already established (by reading MLX's actual `sdpa_vector.h` source) that MLX's
attention kernel does NOT use MPP/`matmul2d` — it uses manual scalar dot products with `simd_sum`
reduction, cited only as *qualitative* corroborating evidence that MPP-style tiling may not suit
attention's thin, memory-bound shapes. **This real benchmark confirms that qualitative read
quantitatively, and much more dramatically than expected**: MLX's non-MPP approach isn't just
"different," it's 4x faster than PyTorch's own native SDPA at the identical shape on the identical
hardware. This means the "~1.17x slower than native" ceiling our best hand-written flash-attention
kernel reached (Part I §4.6) was never actually close to this hardware's real ceiling — it was
close to *PyTorch's MPS backend's* ceiling, which is itself roughly 4x below what Apple's own
framework achieves via a fundamentally different (non-MPP, manual-scalar-reduction) kernel design.
The real, still-open question this raises: would a hand-written Metal kernel using MLX's
approach (not `matmul2d`) close some of this 4x gap? Not yet attempted — flagged as the
highest-leverage next kernel-level experiment if this line of work continues, well above anything
tested in the original `matmul2d`-based investigation.

## Real llama.cpp/GGML baseline: refines the MLX finding, doesn't just repeat it (2026-07-23)

Built llama.cpp's GGML library from source (`cmake -DGGML_METAL=ON`, `cmake --build --target
ggml`) and wrote a minimal standalone C++ harness directly against `ggml.h`/`ggml-backend.h`/
`ggml-metal.h` (not `test-backend-ops` -- its built-in perf cases don't include this paper's exact
reference shapes), at the same two shapes as the MLX comparison above: `ggml_mul_mat` at the
`to_qkv` shape, and `ggml_flash_attn_ext` at the real self-attention shape (H=16, L=4096, D=64,
`mask=NULL` — confirmed no masking, matching the non-causal full attention used everywhere else in
this comparison). q kept f32, k/v f16, matching GGML's own `test-backend-ops.cpp` convention for
this op (its real, production usage pattern, not an arbitrary choice). 20 trials after 15 warmup
iterations, same statistical protocol as the MLX comparison:

| operation | PyTorch MPS | MLX | GGML/Metal (llama.cpp) |
|---|---|---|---|
| Linear/matmul (`to_qkv` shape) | 0.698ms ± 0.178 | 0.717ms ± 0.200 | **0.669ms ± 0.055** |
| Self-attention (real shape) | 5.465ms ± 0.202 | **1.361ms ± 0.084** | 7.166ms ± 0.134 |

**This refines, not just repeats, the MLX finding above.** For matmul, all three frameworks
cluster closely (parity) — GGML is marginally fastest and by far the most consistent (std ~8% of
PyTorch's/MLX's), but the three are not meaningfully different at this shape. For attention, the
three-way split is the real finding: llama.cpp's own hand-tuned, production-grade Metal
flash-attention kernel is not merely "not as fast as MLX" — it's **slower than PyTorch's native
SDPA** at this exact shape (7.17ms vs. 5.47ms), while MLX is 4x *faster* than PyTorch. This means
MLX's advantage isn't "any serious alternative to PyTorch beats it at attention on this hardware"
— a second independent, widely-used, actively-optimized implementation (llama.cpp/GGML) does NOT
replicate it, and is in fact worse than the baseline being beaten. Whatever MLX is doing
specifically (the manual scalar-dot-product + `simd_sum` approach already read directly from its
source in Part I) is unusually well-matched to this particular shape in a way neither PyTorch's
MPSGraph-backed path nor llama.cpp's own hand-written Metal kernel achieves. This sharpens the
open question from the MLX section above: understanding exactly *why* MLX's specific approach
wins here (not just that a non-MPP kernel can) is now clearly the higher-value question, since
"write a kernel like llama.cpp's" is now a demonstrated non-answer.

## Other architectures / llama.cpp / MLX: all three Limitations items closed for this session

**llama.cpp: done** (real GGML/Metal harness, see above). **MLX: done** (real quantitative
baseline, see above — upgraded from Part I's qualitative-only source-reading evidence). **Other
model architectures: done in the partial, honest form scoped above** — not a full second model
port, but a real generalization check (the MLX/GGML three-way comparison re-run at a standard
LLM-scale shape profile, see above) confirming the core pattern isn't TRELLIS-shape-specific.
**Not closed, and not closeable with available resources**: multi-generation Apple Silicon
hardware (M1–M4) — this remains a genuine, acknowledged limitation, not attempted.

## Generalization check: the MLX/GGML pattern replicates at an LLM-scale shape (2026-07-23)

Closes the last item from the assessment above. Re-ran the exact same MLX and GGML/Metal
harnesses at a second, substantially different, well-known shape profile — Llama-2-7B-like
(hidden=4096, `H=32` heads, `D=128` head dim, `L=2048` sequence length) instead of TRELLIS's own
(hidden=1024, `H=16`, `D=64`, `L=4096`) — a real generalization check using existing tooling
rather than a full second model port, exactly as scoped above.

| operation | PyTorch MPS | MLX | GGML/Metal (llama.cpp) |
|---|---|---|---|
| Linear/matmul (2048×4096→4096) | 1.335ms ± 0.058 | 1.333ms ± 0.026 (parity) | 1.633ms ± 0.145 |
| Self-attention (H=32, L=2048, D=128) | 5.183ms ± 0.053 | **1.600ms ± 0.038 (3.24x faster)** | 6.056ms ± 0.089 (slower than PyTorch) |

**The qualitative pattern replicates cleanly at a substantially different shape.** Matmul stays
close to parity across all three frameworks (within ~20% of each other, no framework dominating).
Attention shows the same structure as TRELLIS's own shape: MLX is dramatically faster than
PyTorch's native SDPA (3.24x here vs. 4.02x at TRELLIS's shape — same direction, real but smaller
magnitude), while GGML/llama.cpp's own hand-tuned kernel is again *slower* than PyTorch (6.056ms
vs. 5.183ms, ~1.17x slower — closely matching the ~1.31x slower result at TRELLIS's shape). This
is real evidence the core finding is not an artifact specific to TRELLIS's particular tensor
shapes — the same three-way relationship (matmul ≈ parity; MLX >> PyTorch > GGML for attention)
holds at a standard LLM-scale shape too, without needing a second full model port to establish it.

## Related work worth citing (flagged 2026-07-23, not yet integrated into any paper draft)

**TriFlow** (Li, Erkoç, Sirigatti, Rosov, Li, Dai, Nießner — arXiv:2606.20131, cs.CV/cs.GR, June
2026; Nießner's group at TU Munich): generates clean,
artist-like mesh *topology* via a "nearest-vertex vector field" representation that converts
topology generation from an autoregressive sequence problem into something flow-matching can solve
directly, plus a watershed-guided constrained QEM simplification step to extract the final mesh.
Solves a different problem than this project's attention-efficiency work, but directly relevant as
related work: TRELLIS's own mesh decoder (`slat_decoder_mesh`, FlexiCubes-based, used throughout
this project's mesh-level quality checks) produces dense triangle soup, not artist-quality
topology — exactly the gap TriFlow targets. Worth citing in any related-work section, and a real
(untested, code not yet available as of this writing) future-direction question: could a
TriFlow-style retopology stage run as a post-process on this project's own mesh output?

**Flagged as a potential separate paper (Part III), not started (2026-07-23).** This retopology
question is substantial enough to be its own project rather than a single future-work line in the
Part I/II paper — a different skill set than the attention/kernel work so far (mesh processing,
not GPU kernels or model architecture). What it would actually need, roughly in order:
1. TriFlow has no public code as of this writing — the nearest-vertex-vector-field representation
   and watershed-guided constrained QEM simplification step would need to be reproduced from the
   paper's description alone, a real implementation lift before any application to this project's
   own output is possible.
2. Apply it as a post-process on `slat_decoder_mesh`'s (FlexiCubes) dense output — the actual
   integration point already exists and is well-understood from this project's own quality-check
   scripts.
3. A real evaluation of retopology quality itself (edge-flow regularity, quad/triangle
   distribution, something like an artist-usability metric) — a different kind of measurement than
   anything done so far in this project (wall-clock speed, SLat/Gaussian-parameter divergence),
   and would need its own methodology, not a reuse of the existing verification protocol.
Not pursued yet — noted here so the idea isn't lost, pending a decision on whether to take it on.

## Brutal review pass on paper/main.tex, and fixes applied (2026-07-23)

Ran an adversarial review agent against the full paper draft with the explicit brief of finding
every gap that would sink it at a WACV-tier venue, then fixed what was tractable without a large
new engineering effort. Findings and disposition:

**Fixed this session:**
1. **Abstract/Intro/Conclusion overclaim.** All three previously said quality was "close to
   indistinguishable ... across every channel tested," when Table 6 actually shows 2 of 5 channels
   (Gaussian position 0.90x, scale 0.95x) at/below the seed-variance baseline and the other 3
   (SLat feature 1.31x, Gaussian color/SH 1.11x, opacity 1.10x) sitting 10-31% above it. Reworded
   all three to state the split precisely — the Quality Validation section body text (Sec. 6.4)
   already had this right; only the higher-visibility summary spots were rounding up.
2. **Dead `\bibitem{rigel}` citation.** Never actually `\cite{}`'d anywhere in the body. Verified
   via WebSearch/WebFetch that Rigel (arXiv:2606.12765, R. Kumaresan, June 2026) is a real paper —
   reverse-engineers the same `matmul2d`/MPP path on an M4 Max (pre-Neural-Accelerator generation),
   finds it runs on ordinary shader cores with no dedicated matrix datapath and that fp8 is
   emulated, not accelerated, on that hardware. Added a substantive Discussion paragraph engaging
   with it directly rather than just citing it in passing.
3. **TriFlow citation accuracy.** The existing bibitem (added last session from a LinkedIn
   announcement) had the wrong subtitle ("via Flow Matching" instead of the real "via
   Nearest-Vertex Vector Fields") and said "ECCV 2026 (as announced; to be updated)." Verified the
   real paper via WebFetch on the arXiv abs page (arXiv:2606.20131, 7 authors including Nießner,
   cs.CV/cs.GR, June 2026) and corrected title, full author list, and citation to the real arXiv
   entry. Also fixed here in this file (author list now includes Nießner).
4. **Unexplained 77% vs. 41.8% attention time-share discrepancy.** Section 4 measures ~77% on the
   sparse-structure flow model (dense 16^3 voxel grid); Section 6.1 measures 41.8% (29.3% self +
   12.5% cross) on the structured-latent denoiser (sparse active-voxel tokens). Never previously
   flagged as two different models — added an explanatory paragraph in Sec. 6.1 stating this
   directly rather than leaving the reader to notice the mismatch themselves.
5. **First-person editorializing.** Removed "This is a genuinely surprising result..." (Sec. 4.8)
   in favor of neutral phrasing. Grepped for similar patterns elsewhere — no other instances found.
6. **Missing Related Work.** Added a subsection citing FlashAttention (memory-materialization
   hypothesis motivating the flash-attention kernel work), LLM.int8() (int8 quantization context
   for the operand-sweep), and Swin Transformer (shifted-window attention, the direct source of
   this project's shift pattern) — each paired with one sentence distinguishing what this paper
   asks that the cited work doesn't.
7. **New quality-validation figure.** Diverging horizontal bar chart (pgfplots), Table 6's five
   channels vs. the 1.0x seed-variance baseline, green/orange by above-or-below. Hit two real
   pgfplots bugs building it: (a) `enlarge y limits={abs=...}` is incompatible with symbolic y
   coordinates (silently throws `\pgfmathresult` coordinate errors deep in the axis-closing code,
   not at the offending line) — fixed by using the plain fractional form
   `enlarge y limits=0.15`; (b) multiple `\addplot` calls in one bar-mode axis default to
   *grouped-bar clustering*, silently shifting each single-point plot into a neighboring category
   row (a staircase misalignment between bars and labels, easy to miss without a rendered check) —
   fixed with `bar shift=0pt` on every `\addplot`. Neither bug was caught by a clean `pdflatex`
   exit code alone; both required rendering the actual page and inspecting it.

**Not attempted, flagged back to the user as real scope decisions rather than decided
unilaterally:**
- The review's central structural verdict: WACV is not a good fit as-is (no vision-evaluation
  deliverable, no perceptual/SOTA comparison against other 3D generation work) — recommends either
  retargeting to a systems/efficiency venue (MLSys, ASPLOS, EuroMLSys, a workshop) or adding a
  genuine vision deliverable.
- Transferring the H200-trained checkpoint back to the M5 Max and actually rendering the distilled
  student's output (currently only decoded-parameter-level comparison, not rendered images) — the
  reviewer's highest-impact single fix, blocked in the current environment by the absence of a
  CUDA toolkit for compiling rendering extensions (already stated as a Limitation/Future Work
  item), not attempted.
- Full reformatting into the real WACV/CVF two-column template plus a ~40-50% length cut — large,
  mostly mechanical, not started.

## Venue decision: retargeting away from WACV (2026-07-23)

Per the brutal review's central verdict (previous section) and the user's explicit choice when
presented with the three options (retarget to systems venue / push for WACV with a new vision
deliverable / leave as-is with a caveat), **this paper's target is now a systems/efficiency venue
(MLSys, ASPLOS, EuroMLSys, or an equivalent efficiency-focused workshop), not WACV or any
vision-evaluation venue.** No perceptual/SOTA-vision-comparison work will be added on that basis.

Checked whether this requires any actual rewrite of `paper/main.tex`: it does not, beyond what was
already fixed in the review pass above. The paper's own voice was already written for a systems
audience throughout — e.g. the Limitations section already says outright "this falls short of the
statistical rigor a top-tier systems venue would expect" (a sentence that predates this decision).
`paper/main.tex` uses a plain single-column `article` class, never a WACV/CVF template, and never
states a target venue anywhere in its own text — so there is no vision-venue framing to strip out.
`paper/project_page.html` was also checked (grepped for WACV/CVPR/ICCV/OpenReview mentions in its
visible text) and has none — all matches were coincidental substrings inside the embedded base64
PDF blob, not real text.

Net effect of this decision: no further paper changes needed to execute it. What it does rule out
going forward: the reviewer's #1 highest-impact fix (transferring the H200-trained checkpoint back
to the M5 Max and rendering the distilled student's actual output) and the reviewer's #12 fix (full
WACV/CVF two-column reformat + 40-50% length cut) are no longer worth pursuing for their original
stated purpose (satisfying a vision-venue's evaluation bar) — though the rendering transfer may
still be worth doing someday purely as its own verification step, independent of venue, since it
would still strengthen the quality-validation claim regardless of where the paper is submitted.

## MLSys 2027 submission draft created (2026-07-23)

Confirmed the deadline directly (not just via the earlier research fork's inferred estimate):
**MLSys 2027 paper deadline is October 31, 2026** (two independent sources agree: a direct
WebSearch result and mlciv.com's conference-deadline tracker), conference June 21-24, 2027.
Downloaded the real `mlsys2026.sty` template from `mlsys.org/media/mlsys-2026/mlsys2026style.zip`
(MLSys reuses the prior cycle's format each year) rather than guessing at formatting — confirmed
directly from the CFP page: 10 pages excluding references, 2-column, 10pt Times, double-blind
(no author info or acknowledgments in the initial submission — the style file auto-suppresses
author/affiliation text unless `[accepted]` is passed, so real names can stay in the source).

New file: `paper/mlsys2027/main.tex` (+ `mlsys2026.sty`, compiles to a 9-page PDF: ~7 pages of
2-column body, references on their own page, a short appendix after that). This is a condensation
and reformatting of `paper/main.tex` (the WACV-target draft, left untouched per this project's
"never destructively rewrite a working version" convention) — same numbers, same tables, same
figures, tightened prose, with two secondary tables (the Llama-2-7B-like shape generalization
table, the per-image generalization table) and the full/windowed connectivity schematic moved to
an uncounted appendix. Landed comfortably under the 10-page limit (7 pages used, not 10) after
restoring the full 8-row window-sweep table and two rendered qualitative-comparison figures that
an initial aggressive cut had dropped — there was real slack in the budget, so no need to leave
them out.

One packaging note for future reference: natbib (required by `mlsys2026.sty`) throws a
"Bibliography not compatible with author-year citations" pseudo-error when using a manual
`thebibliography` block (no `.bib`/BibTeX) instead of a real bibliography database — cosmetic only
(it auto-recovers into numeric-citation mode and the output renders correctly, verified by
extracting text from the compiled PDF), not a real blocker, and not worth chasing away by
switching to a `.bib` file for this draft.

## Brutal review of the MLSys draft specifically, and fixes (2026-07-23)

Ran a second adversarial review, this time scoped to `paper/mlsys2027/main.tex` specifically
(distinct from the earlier WACV-draft review) after condensing its abstract and adding the teaser
figure. Findings and fixes:

- **The real headline problem**: several kernel-level "beats/matches native" claims (0.972x,
  0.98x, 1.01x parity) are single-run numbers sitting inside the ~25% run-to-run noise the paper's
  own Table 1 (n=20 repeated trials) demonstrates at the same latency scale — stating them as
  clean wins in the abstract, even with an honest Limitations-section disclosure elsewhere, isn't
  enough for a quantitatively literate reviewer. Fixed by reframing the abstract and the Quantized
  Attention section to explicitly name the noise floor and state only the flash-attention kernel's
  1.17x gap is large enough above it to call a real effect.
- **Real (auto-recovering) natbib compile error**: `mlsys2026.sty` requests author-year citation
  style internally; this paper's manual `thebibliography` + numeric `\cite` usage isn't compatible,
  and it was silently surviving only because `pdflatex -interaction=nonstopmode` auto-continues
  past the resulting hard error. Fixed properly with `\setcitestyle{numbers,square}` rather than
  relying on that undocumented recovery path.
- **Cross-reference collision**: condensing subsections into bold run-in `\textbf{}` paragraphs
  while keeping their `\label`s meant several distinct findings all resolved to the same bare
  "Section 4" — confirmed in the rendered PDF, including twice in one sentence. Fixed by restoring
  real numbered `\subsection` commands throughout Sections 4-6 (4.1-4.7, 5.1-5.3, 6.1-6.4).
- Elevated the MLX/llama.cpp comparison (the paper's only fully repeated-trial, CI-backed result)
  from a supporting explanation to an explicitly named "most robust systems finding."
- Restored two sentences the condensation pass had dropped: the MLX-vs-PyTorch causal-masking
  control check, and the CoreML-not-benchmarked gap in Limitations.
- Added an explicit "diagnosis, not a new mechanism" sentence to CFG-caching, since the caching
  idea itself is FasterCache/AB-Cache's — the paper's real contribution there is root-causing the
  failure against this specific sampler family, not inventing the mechanism.
- A genuine, separate LaTeX bug also found and fixed along the way (not from the review, from my
  own attempt to add the teaser figure): a `figure*` spanning both columns right after the title
  block landed on page 2 with large blank gaps regardless of `[t]`/`[b]` hints, and putting
  `\includegraphics` directly inside `\twocolumn[...]`'s argument broke the build outright
  (`\Gin@ii` errors cascading into dozens of follow-on errors — that argument tolerates plain text
  but not graphics). Fixed by placing the teaser as an ordinary single-column figure at the top of
  Section 1 instead of forcing full-width.

Net effect: recompiles with zero errors (was one auto-recovered error before), still 9 total pages
(~7.3-page body, well inside the 10-page limit) after all fixes.

## ViT-L/14 cross-architecture generalization + H200 cross-hardware check (2026-07-23)

Extended the framework-baseline generalization story with a third shape profile — this time a
genuinely different model *family*, not just a different LLM config: ViT-L/14 (CLIP's vision
encoder), real published architecture (hidden 1024, 16 heads, D=64, L=257 — 256 patches + 1 CLS
token). Shares TRELLIS's own channel/head dims but is 16x shorter in sequence length than
anything tested so far (TRELLIS L=4096, Llama-2-7B-like L=2048).

**Result: the MLX-beats-PyTorch attention pattern reverses here.** PyTorch MPS is 1.61x *faster*
than MLX at this shape (0.224ms vs 0.361ms), not slower — the opposite direction from both
previously-tested long-sequence shapes. This is a real boundary on the paper's central
framework-comparison finding, not a weaker replication of it, and is reported that way in the
paper (`paper/mlsys2027/main.tex`) rather than being smoothed over.

**Follow-up investigation, to rule out the obvious alternative explanation**: is this just "short
sequences erode any specialized attention kernel's advantage over a naive default," or something
specific to MLX/MPS? Ran the identical shape's attention op through PyTorch's own CUDA backend
selection (MATH / EFFICIENT_ATTENTION / FLASH_ATTENTION) on the DGX pod's NVIDIA H200 (new script:
`scripts/dgx_benchmark_vit_cuda.py`, kept separate from the Mac-side script per this project's
platform-separation convention — MPS/MLX and CUDA scripts never share a file). Found the
specialized CUDA backends (FLASH_ATTENTION 0.0242ms, EFFICIENT_ATTENTION 0.0269ms) still clearly
beat the naive MATH baseline (0.0513ms) by ~2x at this exact short sequence length — so short
sequences do *not* generally erode a specialized kernel's edge. The reversal is therefore narrowed
to something specific to MLX's manual-scalar-dot-product kernel design or to PyTorch's MPS
backend specifically, not a universal short-sequence property. Stated as the leading hypothesis in
the paper, not a confirmed mechanism — not verified further.

**CoreML: investigated, deliberately not included.** Also tried adding CoreML (via `coremltools`
9.0, `compute_units=ALL`, targeting iOS17+/mlprogram) as a fourth framework baseline at both the
main TRELLIS reference shapes and the ViT-L/14 shape (`scripts/benchmark_coreml_comparison.py`,
kept on disk but intentionally **not** committed to git). Built a same-input/output-size
trivial-compute control (a single `+0.0` op at each real op's exact tensor shape) to check whether
CoreML's measured latency reflected real compute or dispatch overhead — found it's dominated by
Python/numpy data-marshalling through `predict()`: ~88-94% of measured latency at the SDPA shape,
~35-37% at the Linear shape, confirmed stable across repeated script invocations. This makes any
CoreML number in the same table as MLX/llama.cpp's zero-copy in-process measurements a confounded
comparison, not a clean one, even with the marshalling caveat stated explicitly. Decided against
including it in the paper at all (user's call, after weighing "closes a stated gap" against "adds
a fourth column that needs a big asterisk every time it's cited") — also removed the pre-existing
"we do not benchmark against CoreML" Limitations sentence from an earlier session's brutal-review
fix, since CoreML is now out of scope for this paper entirely rather than a stated-but-unaddressed
gap. The investigation script itself was kept (not deleted) in case it's useful later, per this
project's "never delete a working, documented result" convention — it's just not part of the paper.

## Third brutal review caught something bigger than it was looking for: a false finding (2026-07-23)

Ran a third adversarial review pass, this time asking specifically "what are the gaps to fix for
a top-tier venue," on `paper/mlsys2027/main.tex` as it stood after the ViT-L/14 + H200 addition.
The review's own biggest finding was about statistics labeling (Tables 1/2/6 called raw sample
std. dev. a "95% confidence interval" in prose, inconsistent with how Table 3/Fig. 8 correctly
computed `1.96*std/sqrt(n)`) and flagged the ViT-L/14 self-attention measurement's unusually high
variance (MLX's 73% CoV, ~3x the paper's own established ~25% noise floor) as something that
should at least be investigated or caveated.

**While fixing the stats-labeling issue, re-running the ViT-L/14 script to get before/after
numbers surfaced something much more serious: the direction of the finding itself was wrong.**
The committed paper (previous commit) reported PyTorch faster than MLX at this shape — a
"reversal" of the long-sequence MLX-advantage pattern, with a whole H200 cross-hardware
investigation built around explaining it. Re-running the *identical* script independently 8 more
times showed MLX faster in all 8/8 follow-up runs (ratio range 1.03-2.53x, mean ~1.18x); a
large-sample n=100 check agreed (MLX 1.37x faster). The original measurement — the one basis for
the entire "reversal" narrative — was a noise artifact, not a real effect.

This is exactly the failure mode this project's own stated verification methodology exists to
catch (Section 3 of the paper: never trust a single measurement without a replication check) —
it just hadn't been applied to this one specific new result before committing it. Caught only
because a *third* review round asked a slightly different question ("does this new stat hold up")
than the first two rounds did, and following that thread all the way down led to re-running the
actual experiment rather than just checking the arithmetic.

**Fix**: rewrote every affected section (abstract, Contributions, Section 4.7, Discussion,
practitioner takeaway, Conclusion) to report the corrected, replicated finding — MLX's advantage
narrows sharply at this shape (4.02x -> 1.18x) but does not reverse — and reported the correction
transparently (what the first measurement showed, how it was caught, what replication found)
rather than quietly swapping in the right numbers. The instability itself (PyTorch's CoV reaching
~49% at this shape in the large-sample check) is now reported as a genuine methodological finding:
sub-millisecond attention measurements on this hardware need either much larger n or independent
replication before trusting their sign, a bar the paper's other (millisecond-scale) framework
comparisons clear without needing it. Also fixed the CI-vs-std-dev labeling across Tables 1/2/6,
and a stray `\newpage` before `\appendix` that was wasting most of two pages.

**Lesson for future sessions**: when a review flags unusually high variance in a specific new
result as worth investigating, actually re-run the measurement rather than only computing
statistics on the sample already in hand — the sample itself can be the anomaly, and no amount of
correctly-computed confidence intervals on a single bad sample will surface that.

## Set up real GitHub Actions CI, confirmed genuinely passing (2026-07-23)

Previously the repo had no CI actually running the test suite (only a CodeQL security scan, which
was itself failing on every run — see below). Set up `.github/workflows/tests.yml` to run all 18
`tests/*.py` scripts on every push/PR, and verified it by watching real runs complete, not just by
reasoning that it should work.

**Three real things had to be fixed to get from "wrote a workflow file" to "genuinely green":**
1. Every test script hardcoded this machine's absolute path
   (`/Users/vinayak/Downloads/TRELLIS-main`) — replaced with `os.path.dirname(os.path.abspath(
   __file__))`-based resolution, verified portable by running from a different cwd and via an
   absolute path from outside the repo before ever touching CI.
2. No `requirements-test.txt` existed. Built one from the exact locally-verified versions, then
   installed it into a **completely fresh venv** (not just trusted the existing dev environment)
   and ran the full suite there — caught one real gap this way (`tqdm`, silently present via some
   other package's dependency locally, missing in a clean install) before it could fail in CI.
3. **The first real CI run surfaced a genuine GitHub infrastructure limitation**:
   `torch.backends.mps.is_available()` returns `True` on GitHub's macos-14 (Apple Silicon) hosted
   runners, but real MPS tensor allocation crashes immediately ("MPS backend out of memory, 0
   bytes allocated"). Confirmed via the runner's own Metal device name: `Apple Paravirtual device`
   — these runners virtualize the GPU (via Apple's Virtualization.framework), and whatever memory
   pool PyTorch's MPS allocator expects isn't actually backed by real hardware there, even though
   the availability check itself passes. Fixed with `.ci/sitecustomize.py`, auto-loaded via
   `PYTHONPATH` and gated on a `CI_FORCE_CPU=1` env var the workflow sets (never touches local
   runs on real Apple Silicon hardware): monkeypatches `torch.backends.mps.is_available` to
   `False`, so every test's existing `device = 'mps' if ... else 'cpu'` fallback takes the CPU
   path instead of crashing. Verified locally first by simulating the exact CI env vars before
   trusting a real run.
   - Interesting wrinkle: the raw PyObjC/Metal kernel tests (`metal_smoke_test.py`,
     `metal_rasterizer_verify.py`, `metal_gaussian_verify.py`) call the Metal framework directly,
     not through `torch.backends.mps`, and kept passing with real (if paravirtualized) Metal
     compute regardless of the CPU-forcing override — genuinely different code path, genuinely
     unaffected.
   - Renamed the workflow from "(Apple Silicon / MPS)" to "(macOS)" and wrote its comments to
     state plainly what it verifies (CPU-path correctness on a macOS runner), rather than
     overclaiming MPS coverage it can't actually provide on this infrastructure.

**Also found and fixed, unrelated to the test-suite work**: the pre-existing `codeql.yml` workflow
had been failing on every single run before this ("Python does not support the autobuild build
mode" — a CodeQL platform change, not a bug in this repo). One-line fix (`build-mode: none`,
matching what `c-cpp` already used).

Both workflows (`Tests (macOS)`, `CodeQL Advanced`) confirmed green via `gh run view` on real runs
before reporting this as done, not assumed from a clean local test pass alone.
