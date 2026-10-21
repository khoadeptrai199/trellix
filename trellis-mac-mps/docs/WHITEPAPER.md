# trellis-mac-mps: Porting a CUDA-Only 3D Generation Pipeline to Apple Silicon

**A technical report on porting Microsoft's TRELLIS image-to-3D pipeline to run natively on Apple Silicon (MPS + Metal), and on finding a real 14–43x speedup in the process.**

*This document is written to double as a project whitepaper, an arXiv-style technical report, and source material for a blog writeup — it favors precise, reproducible claims over marketing language throughout.*

---

## Abstract

[TRELLIS](https://github.com/microsoft/TRELLIS) (Xiang et al., 2024) is a large-scale image/text-to-3D generation model producing meshes, 3D Gaussian splats, and radiance fields from a unified structured-latent representation. The reference implementation depends on CUDA-only components at nearly every stage — sparse 3D convolution, a differentiable rasterizer (nvdiffrast), and a Gaussian splat rasterizer — none of which run on Apple Silicon. We describe a from-scratch port of the full image-to-3D pipeline to PyTorch's MPS backend, replacing each CUDA-only component with a pure-PyTorch equivalent or a hand-written Metal compute kernel. We report two classes of finding: (1) correctness issues that would affect any platform (a pre-existing upstream bug, and a mesh-decimation library quirk we diagnosed and fixed), and (2) performance work specific to the Metal port, including a case where a plausible, verified-correct, microbenchmark-faster optimization produced **zero** measurable improvement on the real workload, and a tile-based rasterizer binning scheme that did — a 14–15x speedup on the pipeline's hole-filling stage and ~43x on mesh preview rendering, confirmed end-to-end, not just in isolation.

## 1. Introduction

TRELLIS's reference implementation targets Linux + NVIDIA CUDA exclusively: `README.md` states the code is "currently tested only on Linux" with "an NVIDIA GPU with at least 16GB of memory." Three components in particular have no CPU or Metal fallback in the original codebase:

- **Sparse 3D convolution** (`spconv`), used throughout the structured-latent encoder/decoder.
- **A differentiable rasterizer** (`nvdiffrast`), used for mesh preview rendering and for the multi-view hole-filling pass in mesh postprocessing.
- **A 3D Gaussian splat rasterizer**, used for Gaussian preview rendering and for multi-view texture baking.

None of these have upstream CUDA-free implementations suitable for Apple Silicon. This report describes replacing each with either a pure-PyTorch implementation (portable, but not always fast enough) or a hand-written Metal compute kernel (compiled at runtime via `MTLDevice.newLibraryWithSource_options_error_`, requiring no Xcode installation), and documents the performance investigation that followed once "it runs" was no longer the bottleneck — "it runs slowly" was.

## 2. Method

### 2.1 Component replacement

| Component | Original | Replacement |
|---|---|---|
| Sparse 3D convolution | `spconv` (CUDA) | Pure-PyTorch backend |
| Attention (sparse, windowed, serialized) | CUDA kernels / `flash-attn` | PyTorch SDPA backend |
| Mesh rasterization | `nvdiffrast` (CUDA/OpenGL) | Pure-PyTorch chunked rasterizer, or a hand-written Metal kernel (`metal_rasterizer_binned.py`, §2.3) |
| Gaussian splat rasterization | Custom CUDA kernel | Pure-PyTorch EWA splatting + alpha compositing, or a hand-written Metal kernel |
| Radiance field (octree) rendering | `diffoctreerast` (CUDA) | Not yet ported — see §5 |

Each pure-PyTorch fallback was cross-checked against the CPU-executed reference math it replaces (e.g. the Metal rasterizer's forward pass matched a CPU reference rasterizer to within float32 precision on both a hand-constructed unit scene and randomized meshes; a manually reimplemented bilinear texture sampler — needed because `F.grid_sample`'s backward pass is unimplemented for MPS in this PyTorch build — matched `F.grid_sample`'s own forward output to within 8e-7 max absolute difference).

### 2.2 A verified-correct optimization that didn't help

The first hole-filling speedup attempt targeted MPS↔CPU synchronization overhead. `metal_rasterizer.py`'s `rasterize()` function separately calls `.to('cpu')` on three tensors (`px`, `py`, `pz`) that share an upstream MPS computation graph; profiling with macOS's `sample` tool during a different, related slowdown (in the Gaussian rasterizer) showed the process parked in `at::mps::MPSStream::synchronize`, not in kernel execution — consistent with each separate `.to('cpu')` call forcing an independent re-execution of the shared upstream graph rather than a cheap readback. Applying the same fix that resolved that case (batch into one `torch.stack` + single `.to('cpu')` call) to the mesh rasterizer, plus caching the (call-invariant) triangle-index buffer across the hundreds of camera views a single hole-filling pass rasterizes, was verified correct: cached-path output was bit-identical to a forced cold call across repeated views, and correctly invalidated on mesh switch.

It produced no measurable speedup on the real workload. `_fill_holes`'s actual call runs at 1024×1024 against a ~35K-face mesh; before/after rasterization throughput was statistically indistinguishable (~3.2–3.3 it/s either way). The reason: `metal_rasterizer.py`'s kernel is a brute-force per-pixel loop over every triangle — O(H×W×F) — and at this resolution/triangle-count that term dominates total cost by a wide enough margin that eliminating a few-hundred-KB buffer re-upload made no observable difference. The fix was reverted rather than kept as unused complexity. We note this not as a failure but as a methodological point: *verified correct* and *faster in a microbenchmark* were both true and both insufficient evidence that the change mattered at real scale — only measuring the actual workload settled the question.

### 2.3 Tile-based rasterizer binning

The real bottleneck was the O(H×W×F) term itself. The standard fix for this — used by nvdiffrast's own CUDA rasterizer — is a two-stage bin-then-rasterize architecture (Laine & Karras, 2011): bin primitives into screen-space tiles by bounding box, then have each pixel only test the (small) set of triangles binned to its tile, rather than every triangle in the mesh.

`metal_rasterizer_binned.py` implements this: triangle screen-space AABBs are computed on the CPU from the already-resident (post-sync) projected vertex positions — no additional MPS↔CPU transfer beyond what the brute-force path already performs — and binned into 32×32-pixel tiles via a vectorized (no explicit Python loop over triangles) construction of a CSR-style `(tile_offsets, bin_triangle_ids)` structure. Triangles whose AABB spans more tiles than a configurable threshold are added to every tile explicitly (an exact, not approximate, fallback) rather than risking a dropped triangle at scale. The Metal kernel is otherwise unchanged: each pixel-thread looks up its tile's candidate range and loops only over that range instead of the full triangle list.

**This is an exact fix, not an approximation.** Every triangle whose AABB overlaps a tile is included in that tile's candidate list; output is bit-identical to the brute-force kernel for the same input, verified against: a hand-constructed overlapping-triangle unit scene, a full-screen single triangle (exercising the oversized-triangle fallback path), and dense random meshes at multiple scales (`scripts/verify_metal_rasterizer_binned.py`).

An initial speed test used a synthetic mesh with `torch.randint`-selected triangle vertex indices (three independently random vertices per face) and appeared to show a *regression* — binned rasterization was 2–5x **slower**. This scene is close to the technique's worst case: independently-random vertices produce mostly huge, screen-spanning triangles, unlike a real mesh's small, locally-coherent faces, which is what binning is designed to exploit. Testing against a real decimated mesh (18,282 vertices / 34,605 faces, decimated from a 692K-face raw decode) reversed the result completely: 19.6x faster at 1024×1024, 7.0x faster at 512×512 in isolation (average 1.37 tiles touched per triangle on that mesh), and — critically — a 14–15x speedup on the actual 1000-view `_fill_holes` loop end to end (§3).

### 2.4 A pre-existing mesh-decimation bug

While benchmarking a second test object (a dragon mesh, 658,060 faces raw-decoded), we noticed the postprocessing pipeline's mesh-simplification step reported "before" and "after" face counts that were identical. Diagnosis (`scripts/diagnose_decimate_dragon.py`) ruled out a broken input mesh — the mesh was 99.99% clean by standard measures (66/658,060, or 0.01%, near-zero-area triangles; zero duplicate vertex positions; 440 boundary edges and 1 non-manifold edge out of ~987K total edges) — and instead isolated the cause to the decimation algorithm itself: PyVista's default `.decimate()` (wrapping VTK's `vtkQuadricDecimation`) reduced the mesh by 0.0% (658,060 → 658,010 faces) at a requested 95% target reduction, while `.decimate_pro()` (`vtkDecimatePro`) hit the target exactly (658,060 → 32,903 faces, 95.0%) on the byte-identical input. This is a pre-existing bug independent of the Apple Silicon port — it would misbehave identically on the original Linux/CUDA codebase — and we fix it with an automatic fallback: attempt quadric decimation first (unchanged behavior for the common case, verified unaffected on a second test mesh that decimates correctly under the default algorithm), and fall back to `decimate_pro()` only when the achieved reduction falls far short of the requested ratio.

## 3. Results

All end-to-end numbers are from real pipeline runs (`scripts/run_real_pipeline.py`) on the same machine, same input image, same seed — not synthetic microbenchmarks — measured before and after the fixes in §2.3–2.4 landed.

### 3.0 Full pipeline, end to end

The headline number: same object (vase, 692,142-face raw decode), same seed, same settings, run twice — once before the tile-binning fix (§2.3) landed, once after — image in to textured `.glb`/`.ply` out.

| | Time |
|---|---:|
| Before | 901.2s (15.0 min) |
| After | 365.4s (6.1 min) |
| **Speedup** | **2.47x** |

The gap between this (2.47x) and the per-stage numbers below (14–43x) is expected and informative: roughly half of total pipeline time (diffusion sampling, Gaussian preview rendering, multi-view rendering for texture baking, gradient-descent texture optimization) doesn't touch the mesh rasterizer at all, so it's untouched by this fix and dominates the "after" total. The stages that *do* touch it improved by an order of magnitude or more.

### 3.1 Rasterizer, isolated (real mesh: 18,282v / 34,605f)

| Resolution | Brute-force kernel | Binned kernel | Speedup |
|---|---:|---:|---:|
| 1024×1024 | 0.245–0.401s | 0.012–0.013s | 19.6–30x |
| 512×512 | 0.065–0.066s | 0.008–0.009s | 7.0–8.0x |

### 3.2 Hole-filling stage, end to end (1000 views, 1024×1024, real mesh)

| | Time | Rate | Output quality |
|---|---:|---:|---|
| Before (brute-force kernel) | ~300–320s | ~3.2–3.3 it/s | 34,428 / 34,603 faces kept |
| After (binned kernel) | 21.0s | 47.1 it/s | 34,428 / 34,606 faces kept |

Reproduced twice: an isolated harness (`scripts/bench_fill_holes_binned.py`, 20.9s / 47.9 it/s) and a full, independent pipeline run (21.0s / 47.14 it/s) agree to within measurement noise. Output quality is unchanged — this is a pure speedup, not a quality/speed tradeoff.

### 3.3 Mesh preview rendering, end to end (30 frames, raw un-decimated mesh)

| Object | Faces | Before | After | Speedup |
|---|---:|---:|---:|---:|
| Vase | 692,142 | 181.5s | 4.2s | 43.2x |
| Dragon | 658,136 | ~175–210s (projected) | 4.2s | ~42–50x |

### 3.4 Resolution/quality tradeoff considered and superseded

Before the binning fix was found, we evaluated the alternative of simply lowering the hole-filling visibility pass's resolution (it only needs a `face_id`/`mask` readback, not a full-resolution image). This is a real, if inferior, alternative worth recording:

| Resolution | Time | Speedup | Faces kept | Quality delta |
|---|---:|---:|---:|---:|
| 1024 (baseline) | 300.2s | 1.0x | 34,428 | — |
| 512 | 89.8s | 3.3x | 33,978 | −1.3% |
| 256 | 30.5s | 9.8x | 31,902 | −7.3% |
| 128 | 17.4s | 17.2x | 30,006 | −12.8% |

The binning fix (§2.3) makes this tradeoff unnecessary: it achieves a *larger* speedup (14–15x vs. 512-res's 3.3x) at *full* 1024 resolution, with no quality cost.

### 3.5 Mesh decimation fix

| | Requested reduction | Achieved reduction | Resulting faces |
|---|---:|---:|---:|
| `decimate()` (quadric, buggy on this mesh) | 95% | 0.0% | 658,010 |
| `decimate_pro()` (fallback) | 95% | 95.0% | 32,903 |

## 4. Discussion and limitations

- **Sparse 3D convolution** has a working pure-PyTorch backend but has not been verified against pretrained checkpoint weights — only against random-weight forward passes matching the CUDA implementation's expected shapes/behavior.
- **`diffoctreerast`** (the radiance-field/octree rasterizer) has not been ported. It is currently skipped in the default pipeline (`formats=['mesh', 'gaussian']`, omitting `'radiance_field'`). Porting it is comparable in scope to the mesh and Gaussian rasterizer ports already completed — a new Metal kernel, not a configuration change — and is left as future work.
- **The reverted sync-batching fix (§2.2)** is documented, not deleted from history, specifically because "verified correct + faster in isolation" is a claim we found to be an unreliable predictor of real-workload impact in this codebase. We'd recommend any similar optimization in this port be benchmarked against the actual call site's real input scale before being trusted.

## 5. Reproducing these results

```sh
# Full pipeline, real timing printed at every stage:
python3 scripts/run_real_pipeline.py assets/example_image/object2_vase.png vase

# Rasterizer correctness + isolated speed:
python3 scripts/verify_metal_rasterizer_binned.py

# Hole-filling end-to-end, binned vs brute-force:
python3 scripts/bench_fill_holes_binned.py
python3 scripts/bench_fill_holes_resolution.py

# Decimation fallback fix:
python3 scripts/verify_decimate_fallback_fix.py
```

## References

- Xiang, J. et al. "Structured 3D Latents for Scalable and Versatile 3D Generation." arXiv:2412.01506, 2024. https://arxiv.org/abs/2412.01506
- Laine, S. and Karras, T. "High-Performance Software Rasterization on GPUs." *High Performance Graphics (HPG)*, 2011. https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-Software-Rasterization/laine2011hpg_paper.pdf
- Full implementation notes, including every bug found and fixed along the way: [APPLE_SILICON.md](../APPLE_SILICON.md)
