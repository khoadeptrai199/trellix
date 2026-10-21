# CLAUDE.md — TRELLIS Apple Silicon port

Instructions for continuing work on this repo in future sessions.

## What this project is

A port of Microsoft's TRELLIS (image-to-3D generation) to run natively on Apple Silicon (MPS),
including hand-written Metal kernels for parts of the pipeline that don't have good MPS coverage.
**`APPLE_SILICON.md` is the primary living document** — it has the full, honest history of what
was tried, what worked, what didn't, and why, in far more detail than this file. Read it first,
especially these sections (most recent/relevant to ongoing work at the top):

- "Renderer-specific 180-degree orientation bug" — recent, real, affects `GaussianRenderer` and
  `MeshRenderer` (not `OctreeRenderer`); if you're about to trust any single rendered image's
  absolute orientation for anything (not just a relative before/after comparison), read this first.
- "Fixed CFG caching via difference-caching" and "Classifier-free-guidance caching" — the
  algorithmic contribution (real, working, generalizes) — read this before touching sampler code.
- "Applying the FFN pattern to attention" and everything under "Beyond Linear quantization: SDPA
  itself, and what actually reduces real wall-clock time" — the attention-kernel work (quantized
  attention, dispatch batching, online-softmax flash-attention, tile tuning, `cooperative_tensor`)
  — read this before doing any more kernel work in this area, several avenues are already
  investigated and closed with real evidence, not guessed away.
- "Fused FFN kernel: matmul2d + native int8 + torch.compile, and the real ceiling on FFN alone" —
  includes the ceiling analysis (see below).
- "Real PyTorch/MPS bugs found along the way"

**A paper and project page exist** (`paper/main.tex` → `paper/main.pdf`, `paper/project_page.html`)
covering the kernel characterization and CFG-caching results — see "Current state" below before
starting new work that would make either stale, and re-run `paper/generate_figures.py` /
`build_project_page.py` / recompile `main.tex` if you change any number they cite.

## Current state (as of this writing)

**Kernel-level (int8 quantization via `matmul2d`) — treated as functionally complete for this
hardware generation.** Summary of the real, measured results; full history in APPLE_SILICON.md.

- **FFN** (`quantized_ffn_matmul2d.py`): fused `fc1->GELU->fc2` through `matmul2d`. FFN is ~36% of
  params but only **~11% of actual compute time** (bypass-measured) — an architectural ceiling of
  ~1.12x max whole-model speedup, regardless of kernel quality. Measured: **1.11x slower** overall.
- **Attention** (`quantized_attention_matmul2d.py`): same pattern applied to
  `to_qkv`/`to_q`+`to_kv`/`to_out`, native SDPA left untouched. Attention is **~77% of compute
  time** (bypass-measured) — a much bigger ceiling than FFN. Self-attention's quantized Linears
  **beat native** (0.972x); cross-attention's three separate quantized Linears remain slower
  (1.189x, structural). Full-model: 0.98x.
- **Operand-type sweep**: 6 of ~46 documented `matmul2d` combinations tested at the real
  `to_qkv` shape. Every combination with a `half`/`bfloat` operand clusters ~1.9x slower than
  native `float16` `nn.Linear`, **regardless of the other operand's type** — ruling out int8
  specifically as the bottleneck. Only full `int8xint8->int32` reaches parity (1.01x), at a real
  accuracy cost (median rel. error 0.0111 vs. 0.0078). `relaxed_precision` flag: clean null
  (byte-identical output, 1.003x).
  scripts: `investigate_matmul2d_dtype_combos.py`/`_v2.py`, `investigate_relaxed_precision.py`.
- **Custom attention kernel progression** (`quantized_linear_matmul2d.metal`'s
  `attention_flash_online_softmax*` kernels): per-head Python loop (3.05x slower than native) →
  batched dispatch (falsified the dispatch-count hypothesis, 3.00x slower) → online-softmax
  flash-attention, serial bookkeeping (1.68x slower, real partial win, confirms the
  device-memory-materialization hypothesis) → **parallelized bookkeeping (1.17x slower — closest
  yet)**. Tile-size tuning: `KVBLOCK=64` infeasible (memory budget), `KVBLOCK=16` regresses
  (7.97ms) — `KVBLOCK=32` stays best. `cooperative_tensor` register-resident chaining: **hard
  architectural blocker, confirmed by reading the actual impl header** — chaining requires
  single-simdgroup scope, incompatible with this kernel's `execution_simdgroups<4>` tiling.
  scripts: `investigate_batched_attention_matmul2d.py`, `investigate_flash_attention_matmul2d.py`,
  `investigate_flash_attention_tile_sizes.py`.
- **Corroborating evidence**: Apple's own MLX framework's `sdpa_vector.h` does NOT use
  `matmul2d`/MPP for attention either (manual scalar dot products + `simd_sum`) — independent
  signal this isn't an implementation-quality problem specific to this port's kernels.

**Algorithmic-level — where the real wins actually came from**, orthogonal to all kernel work
above (both apply together):
- **Sampling step-count reduction**: TRELLIS's real 25-step `FlowEulerGuidanceIntervalSampler`
  tolerates fewer steps (rectified flow → straight ODE path) with no solver change. **1.42x–1.92x**
  at 15/10/8 steps, visually + structurally verified across 6 (image, seed) combinations
  (avg 1.98x). The cleanest win found. `scripts/investigate_step_count.py`,
  `scripts/verify_wins_multi_image.py`.
- **CFG difference-caching** (`trellis/pipelines/samplers/flow_euler_cached_cfg.py`): TRELLIS's
  guidance-interval sampler calls the model twice/step (cond + uncond). Naive caching (freeze
  `neg_pred`) causes real, growing visual drift (root-caused: large `cfg_strength=5.0` amplifies
  staleness, confirmed against FasterCache/AB-Cache's actual published mechanisms). Fix: cache the
  *difference* `pred - neg_pred`, derive the reused value from the always-fresh `pred` instead of
  freezing raw `neg_pred`. Real fix: **1.17x–1.30x**, stable mesh vertex counts, verified across 6
  combinations (avg 1.22x). `scripts/investigate_cfg_caching_diff.py`. (The naive, broken version
  is kept in the same file for comparison, per this port's versioning convention — do not use it.)

**A real, renderer-specific bug, found while preparing paper figures — see APPLE_SILICON.md's
"Renderer-specific 180-degree orientation bug" section before trusting any render's absolute
orientation.** `GaussianRenderer` and `MeshRenderer` output comes out point-reflected 180 degrees
from the object's actual orientation (confirmed pre-existing, not introduced this session —
`outputs/preview_gaussian_color.png` from an earlier session shows the identical issue).
`OctreeRenderer` does NOT have this bug (confirmed the opposite way — applying the same fix broke
an already-correct render). Not fixed in `render_utils.py` itself (unverified whether that would
break other established callers) — worked around locally (rotate the frame 180 degrees) wherever
`paper/*.py` needs a human-viewable render.

## Hard-won technical facts (don't re-derive these from scratch)

- **`torch.mps.compile_shader(source)`** JIT-compiles a Metal kernel and lets you call it directly
  on `torch.Tensor`s already on `mps` — zero-copy, no `.to('cpu')`/`.numpy()` round trip, no manual
  `MTLBuffer` management. `torch.mps.load_metallib(path)` does the same for a precompiled
  `.metallib`. Both return an object whose functions take `threads=` (total grid, NOT threadgroup
  count) and `group_size=` (threads per threadgroup).
- **`threads=`/`group_size=` dispatch is `dispatchThreads`-semantics, not `dispatchThreadgroups`.**
  If `threads` isn't an exact multiple of `group_size`, the last threadgroup on each grid edge is
  **non-uniform** (partial) — confirmed by direct probe: `threads=70, group_size=16` gives a last
  group where `thread_index_in_threadgroup` only reaches 0-5, not 0-15. Any kernel with cooperative
  shared-memory loads (assumes all N threads of a threadgroup participate) will silently corrupt
  data at grid edges unless you round `threads` up to a whole number of threadgroups yourself. Pure
  elementwise kernels (no cross-thread cooperation) don't need this.
- **matmul2d supports native mixed precision, confirmed from the actual Apple SDK header**
  (`/Applications/Xcode.app/.../MetalPerformancePrimitives.framework/.../Headers/
  MPPTensorOpsMatMul2d.h`, read directly — always prefer this over web-search summaries for MPP
  questions). `half x int8_t -> float` is natively supported — pass raw quantized int8 weights
  directly, no dequantize kernel needed. Must declare the buffer as `device int8_t*` in the .metal
  source, NOT `device char*` — they're bit-identical but the header's internal `static_assert`
  type-dispatch fails to match plain `char` on this SDK version.
- **A real, confirmed (not theoretical) bug:** feeding `matmul2d_linear`'s second call in a chain
  a buffer written by a *different custom Metal kernel* (not a native torch op) produced NaN on the
  real pretrained model — passed every synthetic test, only showed up with real weights/real
  conditioning. Not a race condition (reproduced identically with a forced `torch.mps.synchronize()`
  between every single kernel dispatch). Not a range-overflow issue (clamping every fp16 cast
  boundary made no difference). The fix: only feed `matmul2d` operands produced by native PyTorch
  ops, never by another custom kernel's output. This constrains what you can safely fuse — see
  `quantized_ffn_matmul2d.py`'s module docstring for the full reasoning.
- **`torch.compile` cannot trace a raw `torch._C._mps_MetalKernel` call** (from `compile_shader`/
  `load_metallib`) — throws `InternalTorchDynamoError`. Register it as a `torch.library.custom_op`
  (with a `register_fake` for shape inference) first; then Dynamo treats it as an opaque node and
  can fuse the surrounding eager ops. This mattered a lot: eager-mode per-op dispatch overhead
  (clamp, cast, elementwise scale/bias, GELU — each a separate MPS command-buffer submission) was
  large enough to fully erase a kernel that's independently 2.24x faster than `torch.matmul`.
  Compile a **shared function at module scope**, not per-instance — verify via
  `torch._dynamo.utils.counters['stats']['unique_graphs']` that it's actually being reused (== 1),
  don't assume.
- **PyTorch's own `do_metal_mm` fallback kernel** (`aten/src/ATen/native/mps/kernels/
  LinearAlgebra.metal`) is plain `TILE_DIM=16` shared-memory tiling with no register blocking or
  `simdgroup_matrix` — i.e. *less* sophisticated than this port's hand-rolled v3/v5. PyTorch's
  actual speed advantage for the shapes in this model comes from the `MPSGraph`
  `matrixMultiplicationWithPrimaryTensor` path (confirmed by reading PyTorch's own
  `LinearAlgebra.mm`), which very likely connects to the same tuned internals `matmul2d` exposes —
  not a hand-writable trick this port was missing elsewhere.
- **MTLBuffer leak, PyObjC-specific:** repeatedly calling `newBufferWithBytes_length_options_`/
  `newBufferWithLength_options_` in a tight loop leaks deterministically regardless of `del`/
  `gc.collect()`/autorelease pools/`torch.mps.empty_cache()`. Fix: allocate once, overwrite
  contents in place via `buf.contents().as_buffer(...)`. Only relevant to the PyObjC-based modules
  (`quantized_linear_metal.py`, `quantized_linear_matmul2d.py`) — irrelevant to the
  `torch.mps.compile_shader`/`load_metallib` modules, which use PyTorch's own tensor allocator.
- **This machine is an Apple M5 Max**, confirmed via `system_profiler SPHardwareDataType` (also
  visible in the Metal toolchain's own build-tag path). M5 is the first Apple Silicon generation
  with dedicated per-GPU-core int8/fp16 matmul hardware ("Neural Accelerators"), per external
  research — but **`matmul2d` still doesn't reliably beat native `MPSGraph`/`nn.Linear` on this
  hardware anyway** (see "Current state" above) — don't assume newer hardware alone fixes this.
- **`matmul2d`'s `cooperative_tensor` chaining (`get_left_input_cooperative_tensor`) requires
  single-simdgroup execution scope — a hard `static_assert` in the actual impl header**
  (`MPPTensorOpsMatMul2dImpl.h`, not documented anywhere in the public header's examples). Only
  `get_destination_cooperative_tensor` (no chaining) works with multi-simdgroup scopes. Relevant
  to any future attempt at reducing device-memory round trips between chained `matmul2d` calls.
- **Native `scaled_dot_product_attention` on MPS has no alternative backend worth switching to.**
  `torch.nn.attention.sdpa_kernel` with `MATH`/`EFFICIENT_ATTENTION`/`FLASH_ATTENTION`/
  `CUDNN_ATTENTION`/`OVERRIDEABLE` all measured identically (~5.0ms at the real self-attention
  shape) — PyTorch appears to resolve all of them to the same MPS implementation. It's also not a
  naive wrapper: beats a hand-rolled 3-op reimplementation by 1.43x, confirmed by direct comparison.
- **`GaussianRenderer`/`MeshRenderer` (in `trellis/utils/render_utils.py`) render output 180
  degrees rotated from the object's real orientation; `OctreeRenderer` does not.** See "Current
  state" above and APPLE_SILICON.md's dedicated section — confirmed pre-existing, not introduced
  by any change this session, and confirmed to NOT be a blanket bug across every renderer.

## Verification methodology (follow this, don't skip steps)

Every kernel change in this codebase goes through, in order:
1. **Synthetic correctness check** (`scripts/verify_*.py`) — varied shapes, an odd/non-tile-aligned
   batch size, a no-bias case. Compares kernel output against the same module's own PyTorch
   fallback (checks the kernel is a correct implementation) AND against the true fp32 reference
   (checks quantization error is small and bounded). **Passing this is necessary but NOT
   sufficient** — see the NaN bug above, which passed every synthetic check.
2. **Real-model finite-output check**, with *real* conditioning (an actual image through
   `pipeline.get_cond()`), not just `torch.randn` — across multiple trials (different random `x`/
   `t` each time). `torch.randn` conditioning is out-of-distribution enough that it's not a
   reliable proxy for whether the real model stays numerically stable.
3. **Isolated per-call benchmark** — sanity-checks the kernel itself is fast, using realistic
   shapes for this model (`channels=1024`, `hidden=4096`, `num_rows` in the thousands).
4. **Real full-model benchmark** (`scripts/benchmark_*.py`) — the only number that actually
   matters for reporting a speedup. Isolated speed and full-model speed have repeatedly diverged
   in this project (interleaving costs, dispatch overhead, cache effects) — never report an
   isolated number as if it were the real one.

## Working-style notes for this repo (from direct user feedback this session)

- **Never delete or destructively rewrite a working kernel/module version.** Add a new file
  instead (`_zerocopy`, `_matmul2d`, `_ffn_matmul2d` suffixes are the existing convention). If
  something in a shared file (e.g. the `.metal` source) becomes unused, leave it in place with a
  comment noting it's unused and why, rather than deleting it.
- **Commit real checkpoints as you go**, scoped precisely to the files that make up one coherent
  piece of work (`git add <specific files>`, `git commit -- <specific files>` if other unrelated
  work is staged) — don't bundle unrelated staged/uncommitted work into a commit, and don't leave
  large amounts of verified-working, uncommitted work sitting only in the working tree.
- **Document findings in `APPLE_SILICON.md` as you go**, not just at the end — treat it as a lab
  notebook, not a final report. Include real measured numbers, not estimates, and say clearly when
  something is hypothesis vs. confirmed. When a finding turns out to be wrong (e.g. "clamping will
  fix this" turned out not to be the real fix), document the correction rather than silently
  replacing the old text.
- Be honest about negative/null results and hard ceilings (e.g. the ~11%-time-share finding) rather
  than continuing to chase a target that's been shown to be architecturally unreachable via the
  current approach — but do keep looking for what *would* reach it (attention, in this case).
- This project is being treated as a real kernel-engineering case study (possible future
  publication/writeup) — favor rigorous, falsifiable diagnosis (reading actual SDK headers instead
  of guessing, ruling hypotheses out with direct probes, measuring instead of estimating) over fast
  guesses, even when it takes longer.

## Practical environment notes

- `SPARSE_BACKEND` and `ATTN_BACKEND` env vars must be set before importing `trellis` in any
  script — `SPARSE_BACKEND=pytorch` (avoids requiring `spconv`), `ATTN_BACKEND=sdpa` (avoids
  requiring `flash_attn`, which isn't installed on this machine). Every script in `scripts/`
  already does this at the top — copy that pattern for new scripts.
- Model used for all benchmarks: `pipeline.models['sparse_structure_flow_model']`
  (555.5M params, TRELLIS-image-large). Standard benchmark shape: `B=1`, `x=torch.randn(1,8,16,16,16)`,
  `t=500.0`, `cond` either `torch.randn(1,1374,1024)` (synthetic, fine for speed benchmarks) or real
  via `pipeline.get_cond([image])['cond']` (required for correctness/finite-output checks).
- Xcode (full `.app`, not just Command Line Tools) is installed, licensed, with the Metal Toolchain
  component — needed for anything touching `MetalPerformancePrimitives.h`/`matmul2d`. SDK headers
  live under `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/
  MacOSX.sdk/System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers/` —
  read these directly for any MPP question rather than relying on web search/training data, which
  is likely stale or wrong for this very recently introduced API.
- **Save generated results (reports, plots, benchmark artifacts, HTML/visualizations) into this
  repo, not the private scratchpad directory.** `outputs/` is the existing convention (already
  holds benchmark logs, preview images/videos, `.glb`/`.ply` exports) — put new result artifacts
  there, not under `/private/tmp/.../scratchpad`. Scratch/intermediate working files that don't
  need to survive the session are still fine in the scratchpad; anything meant to be a kept result
  (a benchmark write-up, a comparison chart) belongs in `outputs/`.

## Immediate next steps (genuinely open, not yet done)

None of these are guessed-at — pick based on what's actually worth the effort, per the honest
ceilings already established above:

- **`QBLOCK=32` tuning** for the flash-attention kernel (flagged, not attempted — needs reworking
  the row/lane assignment scheme, a bigger lift than the `KVBLOCK` variants already tried).
- **AB-Cache's fully adaptive, error-gated trigger** as a refinement over the current fixed-interval
  CFG difference-caching — should recover more speedup in "easy" regions without paying the quality
  cost seen at the most aggressive fixed interval tested.
- **Root-cause the `GaussianRenderer`/`MeshRenderer` 180-degree orientation bug** in
  `render_utils.py` itself, rather than the current per-script workaround — verify a real fix
  doesn't break other established callers before changing the shared file.
- **Multi-generation (M1–M4) characterization**: repeat the operand-combination sweep and
  flash-attention result on older Apple Silicon, to tell M5-specific findings apart from
  general-to-Apple-Silicon ones — the single highest-leverage change if pursuing the paper further.
- **Statistical rigor**: most benchmark numbers above are single-run wall-clock, not
  repeated-trial with confidence intervals — real gap if targeting a venue with that bar.
- Verify the two real algorithmic wins (step count, CFG-diff-caching) hold beyond the current
  6-combination sample before treating a specific setting (e.g. 15 steps, cache interval 2) as a
  settled default for production use.
