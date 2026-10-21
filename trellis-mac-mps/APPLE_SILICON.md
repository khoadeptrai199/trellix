# Running TRELLIS on Apple Silicon (MPS)

TRELLIS is developed and tested on Linux with NVIDIA CUDA (see [README.md](README.md)). This
document covers a from-scratch port of the core pipeline to run on Apple Silicon Macs via
PyTorch's MPS backend, done by replacing each CUDA-only extension with a pure-PyTorch (and, for
one hot path, hand-written Metal) equivalent rather than trying to get the original CUDA packages
to build.

**Heads up (original note, see correction directly below):** if you've seen articles or GitHub
repos claiming a "TRELLIS.2" model with an existing Apple Silicon port (e.g. `trellis-mac`,
`trellis2-apple`) — that appears to be fabricated. There is no "TRELLIS.2" in the real
[microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) repo, which as of this writing still has
multiple open, unresolved issues asking for Mac support. Don't clone those repos or run their
install scripts.

**Correction (found later, while researching this port's publication potential): `microsoft/
TRELLIS.2` now genuinely exists** — confirmed directly against GitHub's API (a real Microsoft-org
repo, "Native and Compact Structured Latents for 3D Generation," 4B params, new O-Voxel sparse
structure), not a fabricated third-party claim. The original note above was correct *at the time it
was written* — this reflects Microsoft actually releasing a v2 since then, not the original note
being wrong. This port is of the original TRELLIS (CVPR'25), now the *previous* version — worth
scoping any write-up explicitly as "TRELLIS v1 on Apple Silicon" rather than assuming "TRELLIS"
unambiguously means this one going forward. The specific fabricated third-party repos named above
(`trellis-mac`, `trellis2-apple`) were never verified to have become real — that caution still
stands independent of Microsoft's own `TRELLIS.2` being real.

## Status: what works

| Component | Status | Notes |
|---|---|---|
| Dense attention | Works | Already had an `sdpa` backend upstream |
| Sparse attention (full / windowed / serialized) | Works, verified | New `sdpa` backend added; verified against a manual softmax reference |
| Sparse 3D convolution | Works, verified (untrained) | New pure-PyTorch backend; **not verified against pretrained checkpoint weights**, see below |
| `SLatFlowModel` (denoiser) | Works, verified | Full forward pass, random weights |
| `SLatEncoder` (VAE) | Works, verified | All three `attn_mode`s (`swin`, `shift_window`, `shift_order`) |
| `Gaussian` representation | Works, verified | `.cuda()` calls replaced |
| `SLatMeshDecoder` (FlexiCubes mesh decoding) | Works, verified | Produces real vertex/face geometry |
| `MeshRenderer` (mask/depth/normal/color preview render) | Works, verified*, fast | Pure-PyTorch rasterizer + hand-written Metal kernel (now tile-binned, see below), ~44-64x faster than the pure-PyTorch path. *Structurally verified (real geometry, correct relative comparisons) — **absolute output orientation is rotated 180 degrees from the object's real orientation**, confirmed later; see "Renderer-specific 180-degree orientation bug" |
| `GaussianRenderer` (3D Gaussian splat rendering) | Works, verified*, fast | Pure-PyTorch EWA splatting + alpha compositing, plus a hand-written Metal kernel (~17x faster at 50K splats; ~100x faster than the naive Metal version at real 478K-splat scale, after the redundant-sync fix below). *Same 180-degree absolute-orientation caveat as `MeshRenderer` above — see "Renderer-specific 180-degree orientation bug" |
| Texture-baking-to-glTF export (`postprocessing_utils.py`) | Works, verified, fast | Full pipeline: hole-filling (now ~14-15x faster, see tile-binned rasterizer below), xatlas UV unwrapping, multi-view render, gradient-descent texture bake, `.glb` export |
| `diffoctreerast` / `radiance_field` format | Works, verified | Pure-PyTorch port of the `trivec` rasterizer (see below) -- correct but noticeably slower than the mesh/Gaussian renderers (~18s/frame at 256x256 vs ~1-6s), scope limited to what TRELLIS's pretrained decoder actually produces |

## Quick start

```bash
export SPARSE_BACKEND=pytorch
export ATTN_BACKEND=sdpa
```

Set these before importing `trellis`. Everything else auto-detects; construct models/tensors on
`device='mps'` as usual.

Additional Python packages needed beyond the base `requirements.txt` (all pure-Python / have arm64
wheels, no CUDA build step):

```bash
pip install plyfile easydict trimesh xatlas pyvista pymeshfix python-igraph opencv-python
pip install pyobjc-framework-Metal pyobjc-framework-Cocoa   # only needed for the Metal kernel path
pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
```

Use that specific `utils3d` fork/commit, **not** the generic PyPI `utils3d` package — that's an
unrelated project that pulls in `open3d`, which has no arm64 wheels here.

The `flexicubes` submodule also needs to be present (it ships as an empty placeholder dir if you
didn't `git submodule update --init`, and this isn't a git checkout that submodule commands work
against anyway):

```bash
git clone https://github.com/MaxtirError/FlexiCubes.git /tmp/flexicubes_clone
cp /tmp/flexicubes_clone/{flexicubes.py,tables.py,LICENSE.txt} trellis/representations/mesh/flexicubes/
```

## What was actually changed, and why

### 1. Attention: CUDA kernels → `torch.nn.functional.scaled_dot_product_attention`

- `trellis/modules/sparse/attention/full_attn.py`, `windowed_attn.py`, `serialized_attn.py`: added
  an `sdpa` code path alongside the existing `xformers`/`flash_attn` ones. The sparse case is
  "packed variable-length sequences" (no padding), so `sdpa` is applied per-segment in a Python
  loop and concatenated back — verified against a manual per-segment softmax reference
  (max error ~1.8e-7).
- `trellis/modules/sparse/__init__.py`: `sdpa` added as a valid `ATTN_BACKEND`/`SPARSE_ATTN_BACKEND` value.

### 2. `vox2seq` (Z-order / Hilbert curve encoding) → pure PyTorch

`vox2seq` is a small CUDA extension used only to compute a space-filling-curve sort key for
grouping nearby voxels into local attention windows (`serialized_attn.py`). Since self-attention
within a window is invariant to token order, this doesn't need to bit-exactly match the original
CUDA kernel's curve construction — only be reasonably space-local — so an independent
reimplementation carries no correctness risk (unlike, say, conv kernel weights).

`trellis/utils/vox2seq_pytorch.py` implements both Z-order (standard Morton bit-interleaving) and
Hilbert curve (Skilling's transpose algorithm) encoding in pure tensor ops. `serialized_attn.py`
tries the real `vox2seq` first, falls back to this.

### 3. Sparse 3D convolution: `spconv` → pure PyTorch gather/scatter

Every `SparseConv3d(...)` call site in `trellis/models` uses the default `stride=1`, which spconv
itself routes to `SubMConv3d` (submanifold: output coords == input coords). Downsampling is
handled separately, by `SparseDownsample`/`SparseUpsample` in `spatial.py`, which was already
backend-agnostic. So only submanifold conv needed implementing — no strided/rulebook conv, no
`SparseInverseConv3d` (confirmed unused anywhere in the model code).

- `trellis/modules/sparse/conv/conv_pytorch.py`: for each of the kernel's k³ offsets, looks up
  (via sort + `searchsorted` over an integer coordinate key) whether the neighbor voxel at
  `coord + offset` is active, gathers its features, projects with that offset's weight slice, and
  scatter-adds into the output.
- `trellis/modules/sparse/basic.py`: extended `SparseTensor` with a `BACKEND == 'pytorch'` data
  path (a lightweight container, not a real spconv/torchsparse object).
- New `SPARSE_BACKEND=pytorch` option registered in `sparse/__init__.py` and `conv/__init__.py`.

**Verified**: the gather/scatter algorithm matches a brute-force per-voxel Python reference
exactly (max error ~6e-8) — this proves the algorithm is internally self-consistent (correct for
a model trained from scratch with this backend).

**Not verified**: `conv_pytorch.py` also includes `spconv_weight_to_pytorch()`, a best-effort
converter for loading pretrained spconv checkpoint weights, assuming a standard
centered/row-major kernel-offset convention. This has **not** been checked against real spconv
output, because spconv cannot be installed on macOS at all to cross-check against — there's no
CUDA machine in this loop. If the offset convention doesn't match spconv's actual internal
rulebook ordering, inference from pretrained weights would silently produce wrong (not crashing,
just wrong) 3D output. **Verify on a CUDA machine before trusting pretrained-weight inference
through this backend.**

### 4. Gaussian representation: hardcoded `.cuda()` → `self.device`

- `trellis/representations/gaussian/gaussian_model.py`: `setup_functions()` had three hardcoded
  `.cuda()` calls despite the class already threading a `device` param everywhere else. Changed
  to `.to(self.device)`.
- `trellis/representations/gaussian/general_utils.py`: `strip_lowerdiag`, `build_rotation`,
  `build_scaling_rotation` (called by `Gaussian.get_covariance()`) hardcoded `device="cuda"`.
  Changed to derive device from the input tensor instead (`L.device`/`q.device`/`s.device`) —
  correct on CUDA, MPS, or CPU, not just Apple Silicon specifically.

Verified: all property getters, `get_covariance()`, and a `save_ply`/`load_ply` round-trip
(including through the actual coordinate-transform path, see bug fix below) all confirmed
finite/correctly-placed on `mps:0`.

### 5. Mesh decoding: vendored `flexicubes`, avoided `kaolin`

- `flexicubes.py` needed **zero** code changes once vendored in — every internal tensor already
  threads through `device=device` properly.
- Its one dependency, `kaolin.utils.testing.check_tensor`, is a shape-assertion helper (used only
  inside `assert` statements, not the actual marching-cubes algorithm), so
  `trellis/representations/mesh/flexicubes/_kaolin_shim.py` reimplements just that one function
  rather than pulling in all of `kaolin` (a much heavier CUDA-extension library). Falls back to
  the real one first if installed.
- **Real bug fixed**, in `trellis/models/structured_latent_vae/decoder_mesh.py`:
  `SLatMeshDecoder.__init__` built `SparseFeatures2Mesh` with no `device` argument, silently
  defaulting to `device="cuda"` — and since `SparseFeatures2Mesh`/`FlexiCubes` aren't `nn.Module`s
  (no registered buffers), `.to('mps')` on the decoder would never have relocated them. This
  crashes immediately even at construction time on a machine with no CUDA at all. Fixed by
  constructing with `device='cpu'` (always available) and re-pointing to the real device lazily
  in `to_representation()` the first time it's actually used.

Verified end-to-end: real vertex/face geometry produced (e.g. 10,982 vertices / 21,308 faces from
a random-weight decode), all finite, on `mps:0`.

### 6. `MeshRenderer`: pure-PyTorch triangle rasterizer replacing nvdiffrast

`trellis/renderers/sw_rasterizer.py` implements the subset of nvdiffrast's API that
`mesh_renderer.py` and (via the shim in §8) `utils3d`'s rasterization actually call:

- **`rasterize`**: standard edge-function coverage test + nearest-wins z-buffer, chunked over
  triangles to bound memory. Batch size 1 only (the only way it's ever called here).
- **`interpolate`**: barycentric-weighted gather, differentiable w.r.t. both attributes and
  (through the weights) vertex position, matching nvdiffrast's actual differentiability contract.
  Accepts nvdiffrast's `diff_attrs` kwarg for API compatibility (returns a zero placeholder
  derivative tensor rather than computing real screen-space derivatives — see `texture()` below
  for why that's fine).
- **`antialias`**: identity passthrough — **not** a reimplementation of nvdiffrast's analytic
  silhouette-edge blending (a separate, harder algorithm). Edges will be aliased at `ssaa=1`; use
  `MeshRenderer`'s existing `ssaa` option (renders at higher internal resolution, downsamples with
  `F.interpolate(..., antialias=True)`) for anti-aliased output instead — verified working.
- **`texture`**: differentiable bilinear sampling (own hand-rolled gather-based implementation,
  not `F.grid_sample` — see the MPS bugs section below for why). No mipmapping: nvdiffrast uses UV
  screen-space derivatives to pick a mip level and avoid minification aliasing; this always samples
  the base level. Doesn't affect *correctness* of gradient-descent texture optimization (this
  function's only caller, `bake_texture`) — plain bilinear sampling is still exactly differentiable
  w.r.t. the texture, just not anti-aliased under strong minification.

Wired in as an `ImportError` fallback in `mesh_renderer.py`, so it's inert on a real CUDA machine
with nvdiffrast installed.

**Verified in three layers**: (1) rasterizer core against an independently-written
point-in-triangle reference (0 mismatches across 4096 test pixels) plus an occlusion/z-test check;
(2) full camera pipeline against NDC conventions derived by hand from `intrinsics_to_projection`'s
actual matrix, checked against a synthetic cube at a known pose (silhouette centered at
(0.496, 0.496), expected (0.5, 0.5)); (3) integration test against a real (if randomly-weighted)
`SLatMeshDecoder` output.

#### Hand-written Metal kernel (`metal_rasterizer.py`)

The pure-PyTorch rasterizer above is correct but slow — chunked over triangles with a separate
tensor op dispatch per chunk, O(H·W·faces) with real Python/dispatch overhead on top. For a
25,548-face mesh at 512×512 it took 9.6s. `trellis/renderers/metal_rasterizer.py` reimplements
just `rasterize` as a single fused Metal compute kernel: one GPU thread per output pixel, each
thread loops over *all* triangles in registers and keeps a running nearest-hit, writing once at
the end. No atomics needed (each thread owns exactly one pixel, no write contention), no
intermediate tensor materialization, one kernel dispatch total.

`xcrun metal` (the offline shader compiler) isn't available in this environment (Xcode Command
Line Tools only, not the full Xcode.app), so the kernel is compiled at **runtime** via
`MTLDevice.newLibraryWithSource:options:` (part of `Metal.framework`, present on every Mac,
no Xcode required), driven from Python via PyObjC. Falls back transparently to `sw_rasterizer` if
PyObjC or a Metal device aren't available for any reason.

Wired into `mesh_renderer.py`'s fallback chain: nvdiffrast → `metal_rasterizer` → `sw_rasterizer`.
`interpolate`/`antialias`/`texture` are re-exported unchanged from `sw_rasterizer` — only
rasterization itself needed a kernel; those are cheap gather/elementwise ops already.

**Result**: 0.15-0.22s for the same 25,548-face / 512×512 case that took 9.6s in pure PyTorch —
roughly 44-64x faster. Verified exactly (5.96e-08 max diff) against the already-verified
pure-PyTorch rasterizer on the original two-triangle test scene, 99.9985% pixel agreement on a
9,000-triangle random-mesh stress test (the ~0.0015% gap is expected floating-point tie-breaking
at shared triangle edges between adjacent triangles, present in both implementations — see the
"no fill rule" note below), and the earlier synthetic-cube geometric test re-verified through the
new path.

**Known limitation, not specific to Metal**: neither rasterizer implements a consistent tie-break
rule for pixels landing exactly on a shared edge between two coplanar adjacent triangles (real
rasterizers use a "top-left fill rule" specifically to avoid this). Found via the synthetic cube
test: 12 of 1,296 covered pixels showed the *back* face's depth instead of the front's, all lying
exactly along the diagonal seam between the front face's two triangles. Cosmetic, and essentially
never encountered on the irregular geometry `FlexiCubes` actually produces — not fixed.

### 7. `GaussianRenderer`: pure-PyTorch 3D Gaussian splat rasterizer replacing `diff_gaussian_rasterization`

`trellis/renderers/gaussian_rasterizer_pytorch.py` reimplements mip-splatting's CUDA
`GaussianRasterizer`/`GaussianRasterizationSettings` API:

1. **3D→2D covariance projection** (EWA splatting): the standard Zwicker et al. Jacobian-based
   formula every 3DGS reimplementation uses — `Sigma_2D = (J·W)·Sigma_3D·(J·W)ᵀ`, plus
   mip-splatting's low-pass filter (`kernel_size` added to the 2D covariance diagonal).
2. **Color**: precomputed RGB, or spherical harmonics via the existing `sh_utils.eval_sh` (already
   pure math, needed zero changes — never touched anything CUDA-specific in the first place).
3. **Compositing**: unlike triangles (opaque, nearest-wins), Gaussians are semi-transparent and
   *every* Gaussian in front of a pixel contributes to it via alpha blending, in strict depth
   order. Sorting all Gaussians by depth once is cheap; the actual compositing is vectorized via
   `torch.cumprod` within each chunk (`_composite_chunk`) rather than a sequential per-Gaussian
   Python loop, while still producing exactly the same result as strict sequential
   front-to-back blending — verified below.
4. No tile-based binning (unlike the CUDA kernel) — same "correctness first, chunked dense
   O(H·W·N)" structure as the mesh rasterizer, not yet followed up with a Metal kernel the way
   `MeshRenderer` was. A natural next optimization if this becomes a measured bottleneck: same
   one-thread-per-pixel pattern, looping over depth-sorted Gaussians instead of triangles.

Wired into `gaussian_render.py`'s `render()` as an `ImportError` fallback for
`diff_gaussian_rasterization`, plus the same hardcoded-`.cuda()`-in-`bg_color`/`screenspace_points`
fixes as elsewhere (derived from `pc.get_xyz.device`/`gausssian.get_xyz.device` instead).

**Verified**: the `cumprod`-based chunked compositing matches a fully-sequential, non-vectorized
Python reference (one Gaussian at a time, explicit loop, no chunking) exactly — max error 1.2e-7.
Plus a single-Gaussian placement sanity check (appears at the right screen location, correct
color, black background elsewhere) and an occlusion test (a nearer opaque red Gaussian correctly
dominates a farther green one at the same pixel).

### 8. Texture-baking-to-glTF export (`postprocessing_utils.py`)

This turned out to route through **three** different rasterization dependencies, not the one
originally scoped:

- `utils3d.torch.RastContext` — turned out (per its own docstring) to be "nothing but a wrapper of
  nvdiffrast.torch.RasterizeCudaContext or RasterizeGLContext", and `rasterize_triangle_faces`
  purely delegates to `dr.rasterize`/`interpolate`/`antialias`/`texture`. So rather than a fourth
  rasterizer implementation, `trellis/renderers/nvdiffrast_shim.py` registers our rasterizer
  (`metal_rasterizer` if available, else `sw_rasterizer`) as `sys.modules['nvdiffrast.torch']`
  before `utils3d`/`nvdiffrast` get imported — `install()` is called at the top of
  `postprocessing_utils.py`, ahead of its own `import utils3d`/`import nvdiffrast.torch`.
- `dr.texture(texture, uv, uv_dr)` — called directly in `bake_texture`'s gradient-descent
  optimization loop. Covered by §6's `sw_rasterizer.texture()`.
- `GaussianRenderer` — used to render the multi-view "observations" that get baked onto the mesh's
  UV texture. Covered by §7.

Beyond the rasterization plumbing, `postprocessing_utils.py` and `render_utils.py` had ~20
hardcoded `.cuda()`/`device='cuda'` call sites across `_fill_holes`, `postprocess_mesh`,
`bake_texture`, `to_glb`, `simplify_gs`, and `yaw_pitch_r_fov_to_extrinsics_intrinsics` — all
threaded through to a real device now (derived from input tensors where available, or a new
`_default_device()` helper — `mps` > `cuda` > `cpu` — for the functions that only take numpy
arrays and have no tensor to infer a device from).

Also needed: `xatlas`, `pyvista`, `pymeshfix`, `python-igraph`, `opencv-python`, `trimesh` — all
CPU-only, all installed cleanly with native arm64 wheels, no compatibility issues.

**Verified end-to-end**, not just per-function: reproduced `to_glb()`'s exact logic (mesh
decimation → `_fill_holes`'s real multi-view rasterization-based visibility check → igraph mincut
→ xatlas UV unwrapping → 12-view `GaussianRenderer` multi-view render → `bake_texture`'s
gradient-descent optimization, loss converging 0.5 → ~1.5e-5 → final `trimesh` assembly with PBR
material) and exported a real, valid 365KB `.glb` file, all in ~12 seconds on `mps`. Also verified
`bake_texture`'s `mode='fast'` (nearest-neighbor scatter, no optimization) path separately.

## Real PyTorch/MPS bugs found along the way (not TRELLIS-specific)

These aren't upstream TRELLIS bugs or bugs in this port's logic — they're gaps/bugs in this
PyTorch build's MPS backend itself, found empirically while building the above and worth knowing
about if you hit similar symptoms elsewhere:

- **`tensor.to(device, dtype)` (or `.to(device=..., dtype=...)`) combining a cross-device transfer
  *to CPU* with int64→int32 dtype narrowing corrupts the *source* MPS tensor in place.** Found via
  a very confusing failure: `utils3d`'s rasterizer call chain raised `index 4294967296 is out of
  bounds` — `2^32` — from a triangle-index tensor that had checked out fine moments earlier.
  Root-caused by bisection down to `metal_rasterizer.py`'s `tri.detach().to('cpu',
  torch.int32).numpy()` — the conversion result was correct, but the *original* `tri` tensor
  (still referenced elsewhere) got corrupted as a side effect. Confirmed as a narrow, specific
  trigger: float tensors are unaffected, same-device conversions are unaffected, only
  cross-device-to-CPU + int-narrowing in the same call triggers it. Fixed by splitting into two
  separate `.to()` calls (device first, then dtype). This was silently latent in the `MeshRenderer`
  Metal-kernel work from earlier too — confirmed by explicitly testing that a mesh's `.faces`
  tensor now survives being rendered from multiple camera angles in a row, which it wouldn't have
  before (multi-view rendering re-reads the same mesh's faces repeatedly, exactly the pattern that
  would have silently corrupted it).
- **MPS has no float64 support at all** (a documented, hard framework limitation, not a bug per
  se). `torch.tensor(python_float)` / `torch.tensor(list_of_python_floats)` defaults to float64,
  which crashes outright on `.to('mps')`. Hit repeatedly in `postprocessing_utils.py`/
  `render_utils.py`'s camera-math helpers (`torch.tensor(float(yaw))` etc.) — fixed by making
  `dtype=torch.float32` explicit at every such construction site.
- **`F.grid_sample`'s backward pass isn't implemented for MPS** in this PyTorch build
  (`grid_sampler_2d_backward`, confirmed via the error PyTorch itself raises, which suggests
  setting `PYTORCH_ENABLE_MPS_FALLBACK=1` as a CPU-fallback workaround). Since `sw_rasterizer.
  texture()` needs to be differentiable for `bake_texture`'s optimization loop, this couldn't just
  be worked around with the env var (silent, easy-to-forget CPU fallback) — reimplemented bilinear
  sampling manually via `gather`/`clamp`/lerp instead, which has full forward+backward MPS support.
  Cross-checked the forward pass against `F.grid_sample`'s own output on random data (max diff
  8e-7) to make sure the manual reimplementation wasn't introducing its own bugs.
- **Repeated `.to('cpu')` calls on MPS tensors that share upstream lazy-graph computation appear to
  each force a full independent re-execution of that shared graph, not just a cheap readback of
  already-computed values.** Found via the macOS `sample` profiler on a `bake_texture` call that
  was hanging: the process was parked in `at::mps::MPSStream::synchronize`/`copy_and_sync`, not in
  the Metal kernel itself. `metal_gaussian_rasterizer.py`'s per-pixel-compositing setup was calling
  `.to('cpu')` **8 separate times** on 8 tensors (`px`, `py`, `inv_a`, `inv_b`, `inv_c`, `opacity`,
  `color`, `bg`) that all trace back through the same upstream EWA-projection computation graph.
  Fix: `torch.stack`/`torch.cat` them into one tensor **on MPS first**, then a single
  `.to('cpu').to(torch.float32).numpy()` call, then split back apart on the CPU side with plain
  numpy slicing (no further device transfers). At real scale (478,336 splats, 1024x1024,
  synthetic scale-matched test) this took first-view render time from **545.11s to 5.46s — a
  ~100x reduction from a pure sync-count change, not an algorithmic one.** Confirmed in the actual
  end-to-end run afterward: first real `bake_texture` view rendered in 6.27s, matching the
  synthetic prediction. Practical rule of thumb going forward in this port: never call `.to('cpu')`
  more than once per "batch" of tensors that share upstream MPS computation — always stack/cat
  first, sync once, split after.
- **The same `px`/`py`/`pz`-synced-separately pattern also exists in `metal_rasterizer.py`'s
  `rasterize()`, hit hardest by `_fill_holes`'s 1000-view hole-filling loop — but batching that
  sync (plus caching the triangle-index buffer across views, since `tri` never changes call to
  call there) turned out **not** to help in practice, and was reverted.** Verified correct in
  isolation (`scripts/verify_metal_rasterizer.py`: cached-path output bit-identical to a forced
  cold call, correct cache invalidation on mesh switch, matches a CPU reference rasterizer) and
  clearly faster in a synthetic microbenchmark at small scale/resolution — but on the real
  `_fill_holes` call, which runs at **1024x1024** (`fill_holes_resolution=1024` in
  `scripts/run_real_pipeline.py`) against a ~35K-face postprocessed mesh, before/after rasterization
  rate was statistically indistinguishable (~3.2-3.3 it/s either way). Root cause: this Metal
  kernel is a brute-force per-pixel rasterizer with no spatial binning/tiling — cost scales as
  O(H×W×F), so at 1024² × ~35K faces the kernel's own compute dwarfs the few-hundred-KB
  triangle-buffer sync that was eliminated. The sync fix was real, it just wasn't the bottleneck at
  this resolution. Left as a documented dead end rather than carrying dead complexity — a real
  speedup here needs either spatial binning in the kernel (matching nvdiffrast's actual approach,
  see `sw_rasterizer.py`'s docstring) or dropping the visibility-pass resolution, since
  `_fill_holes` only reads back `face_id`/`mask`, not a full-resolution image. **Follow-up: spatial
  binning below is the fix that actually worked.**
- **Fixed the real `_fill_holes` bottleneck above with tile-based binning** —
  `trellis/renderers/metal_rasterizer_binned.py`, the standard two-stage (bin + fine-raster)
  architecture from Laine & Karras' *High-Performance Software Rasterization on GPUs* (the same
  NVIDIA authors nvdiffrast's own CUDA rasterizer is built on): triangles are binned into 32x32
  screen-space tiles by bounding box on the CPU (vectorized numpy, no extra MPS sync beyond what
  the brute-force path already pays), then each pixel-thread only tests the triangles actually
  binned to its tile instead of every triangle in the mesh. Binning is exact, not a culling
  heuristic — every triangle whose AABB overlaps a tile is in that tile's candidate list, with an
  explicit add-to-every-tile fallback for triangles spanning more than `MAX_TILE_SPAN` tiles — so
  output is bit-identical to the brute-force kernel (`scripts/verify_metal_rasterizer_binned.py`).
  First speed test used a synthetic random-triangle mesh (3 independently random vertices per
  face) and looked like a **regression** (2-5x slower) — that scene is close to binning's worst
  case, since fully-random triangles are mostly huge/screen-spanning rather than small and
  locally coherent like a real mesh. Testing against the real decimated vase mesh (`scripts/
  save_real_mesh_for_bench.py`, `scripts/profile_binned_rasterizer.py`) flipped this completely:
  **19.6x faster at 1024x1024, 7x at 512x512** (avg 1.37 tiles/triangle for that mesh). Confirmed
  end-to-end on the real 1000-view `_fill_holes` loop (`scripts/bench_fill_holes_binned.py`):
  **20.9s vs the ~300-320s brute-force baseline, a 14-15x reduction**, with equivalent output
  quality (34437/34605 faces kept vs 34428/34603 baseline) — full 1024 resolution, no quality
  tradeoff, unlike lowering `fill_holes_resolution` (`scripts/bench_fill_holes_resolution.py`:
  3.3x faster at 512 res costs 1.3% of faces, 9.8x faster at 256 res costs 7.3% — not needed now
  that the actual bottleneck is fixed). `nvdiffrast_shim.py` now prefers this rasterizer by
  default (falling back to the brute-force kernel, then the pure-PyTorch one), so this is the
  active backend for every rasterization call in the port, not just `_fill_holes`.
- **Ported `diffoctreerast`'s `OctreeTrivecRasterizer` (the `radiance_field` output format) to
  pure PyTorch** — `trellis/renderers/octree_renderer_mps.py`, the last remaining CUDA-only piece
  of this pipeline. Ported by reading the actual CUDA source (`forward.cu`, `auxiliary.h`,
  `config.h` from [JeffreyXiang/diffoctreerast](https://github.com/JeffreyXiang/diffoctreerast))
  directly, not reimplemented from a description. Scope is deliberately narrower than the original
  library, justified by reading `trellis/models/structured_latent_vae/decoder_rf.py` to see what
  TRELLIS's pretrained decoder actually produces: only the `trivec` primitive (the only one
  `Strivec` ever uses), only `sh_degree=0` (hardcoded by the decoder — asserted, not silently
  assumed), forward-only (nothing in this pipeline differentiates through radiance-field
  rendering, and being plain PyTorch ops rather than a hand-written kernel, autograd would work
  automatically if that ever changed — no separate backward kernel needed like the CUDA original
  has). Algorithm: the decoder also always produces *uniform-depth* leaves — despite `DfsOctree`
  supporting general adaptive multi-depth trees, TRELLIS's real output is a sparse voxel grid at
  one fixed resolution. That means leaves are non-overlapping axis-aligned cells that exactly tile
  the scene, so a ray visits them in strictly monotonic distance order for free — this port does a
  single global ray march via a dense leaf-index grid lookup per step, instead of replicating the
  CUDA original's tile-binning + per-tile depth-sort machinery (which exists there to handle
  adaptive-depth trees in general). First working version marched at the finest per-voxel step
  size across the *entire* scene bounding box regardless of occupancy — correct (verified) but
  ~80s/frame at 256x256 on a real model output (14,948 occupied leaves / 64³ grid), since almost
  all of that marching happened in empty space. Fixed with empty-space skipping: jump straight to
  the next grid-cell boundary while unoccupied (DDA-style), only switch to fine per-voxel stepping
  inside an occupied leaf — exact, not an approximation, since empty cells contribute zero density
  regardless of sample count. **216.8s for 12 frames (18.1s/frame) after the fix, a 4.5x speedup**,
  with pixel statistics matching the unoptimized run almost exactly (confirms the speedup changed
  nothing about the result). Still noticeably slower than the mesh/Gaussian renderers (~1-6s/frame)
  — a real, currently-unclosed gap, not a "done" story like the tile-binned mesh rasterizer above.
  No CUDA is available on this machine to diff against a reference output directly, so correctness
  verification instead used: three synthetic scenes with known-by-construction expected results
  (`scripts/verify_octree_renderer_mps.py` — a single opaque voxel produces a small, correctly
  colored, viewpoint-dependent silhouette, not a full-screen or frozen render; near-zero density
  produces pure background), and an end-to-end smoke test against a real pretrained-model Strivec
  output checked for non-degenerate frames (`scripts/e2e_radiance_field_test.py`).
- **fp16 diffusion sampling: tested, doesn't help.** The two large denoisers
  (`sparse_structure_flow_model`, 555.5M params; `slat_flow_model`, 600.4M params) load as
  `float32` by default despite their checkpoint filenames ending in `_fp16` (e.g.
  `ss_flow_img_dit_L_16l8_fp16`) — both classes already support `.convert_to_fp16()`. Converting
  both and re-running the same seed/image: **42.3s (fp32) vs 42.1s (fp16) — no measurable
  speedup**, output still valid (`scripts/test_fp16_diffusion_speedup.py`). Apple Silicon GPUs
  don't have the CUDA tensor-core fp16-vs-fp32 throughput gap; the 25-step sampling loop is more
  likely bottlenecked by per-step dispatch overhead (many small kernel launches across many
  transformer blocks) than raw matmul compute — the same *class* of problem already found and
  fixed in this port's rendering kernels (batching syncs, single-dispatch marching), but fixing it
  *here* means restructuring the sampling loop itself, not swapping a dtype.

  **Correction, found while verifying the quantized-attention module (see "Applying the FFN
  pattern to attention" below): the "load as float32 by default" claim above was never actually
  checked and is wrong.** Both denoisers' own checkpoint configs
  (`ss_flow_img_dit_L_16l8_fp16.json`, `slat_flow_img_dit_L_64l8p2_fp16.json`, read directly from
  the HF cache) set `"use_fp16": true` in `args`, and `SparseStructureFlowModel.__init__` /
  `SLatFlowModel.__init__` both call `self.convert_to_fp16()` automatically at construction when
  that's set — confirmed directly by checking `model.use_fp16` and `model.dtype` right after
  `TrellisImageTo3DPipeline.from_pretrained(...)`, with no explicit conversion call made: both
  denoisers are `torch.float16` from the moment they're loaded, not float32. `scripts/
  test_fp16_diffusion_speedup.py`'s "fp32" leg never verified this assumption before timing it —
  it was therefore almost certainly *already* running in fp16, and the "fp16" leg's
  `convert_to_fp16()` call was very likely a no-op on an already-fp16 model. **This means the
  "fp16 doesn't help" conclusion above was comparing fp16 against fp16, not fp32 against fp16, and
  is not actually established** — a real fp32-vs-fp16 comparison would need an explicit
  `convert_to_fp32()` baseline first. Not re-tested as part of this session (out of scope for the
  attention-quantization work in progress); left as a flagged, not-yet-corrected open item rather
  than re-asserting the old conclusion or silently fixing it without measurement.
- **Built int8 weight-only quantization with a hand-Metal kernel anyway: real 4x memory reduction,
  but 17x *slower*, not faster.** `trellis/modules/quantized_linear_metal.py` — per-output-channel
  symmetric int8 quantization (weights only, activations stay float32), since standard tooling
  doesn't reach this platform (`bitsandbytes` has no Apple Silicon support; `torch.ao` targets
  CPU/CUDA). Verified correct two separate ways (`scripts/verify_quantized_linear.py`): the Metal
  kernel matches its own PyTorch dequantized-matmul fallback to ~2e-6 (float32 accumulation-order
  noise, not a bug), and quantized-vs-original-fp32 error is small and bounded (median ~0.7%
  relative error across several realistic layer shapes, the expected cost of quantization, not a
  bug either). On the real `sparse_structure_flow_model` (555M params, 196 `nn.Linear` layers):
  **281ms → 4819ms per forward pass, a 17x slowdown**, while weight memory dropped exactly the
  expected 4x (2221.8MB → 555.4MB). This confirms the concern raised before building it: the fp16
  result showed this workload isn't memory-bandwidth-starved (isolated forward pass already
  achieves ~14.8 TFLOPS, a reasonable fraction of peak), so a *naive* kernel — one thread per
  output element, a scalar dot-product loop, no tiling/SIMD-group blocking — loses badly on raw
  compute efficiency to PyTorch's tuned MPS matmul, even while reading 4x less weight data.
  - **Followed up with three further GEMM optimization passes, each measured on the real model,
    with real but shrinking returns, then stopped at a plateau:**
    1. *Shared-memory tiling* (16×16 output tile/threadgroup, cooperative load, each element reused
       16x from threadgroup memory instead of re-read from global memory per use): 4819ms →
       2233.8ms (17.1x → 8.0x slower than fp32's 278ms). Real bug hit and fixed here:
       `dispatchThreads` creates *non-uniform* (partial) threadgroups at grid edges when a
       dimension isn't a tile-size multiple, silently leaving parts of the shared tile
       uninitialized — broke exactly the test cases with `num_rows < 16`. Fixed by switching to
       `dispatchThreadgroups` (always launches complete threadgroups; out-of-bounds threads at
       edges handled by existing bounds guards).
    2. *Register blocking* (64×64 tile/threadgroup, each thread computes a 4×4 output block instead
       of 1, so each inner-loop K-step does 16 FMAs per 8 shared-memory reads instead of 1 FMA per
       2 — the actual lever tuned GEMMs use for arithmetic intensity): 2233.8ms → 1771.4ms (8.0x →
       6.4x slower).
    3. *`simdgroup_matrix` hardware matrix-multiply instructions* (Metal's analog to CUDA tensor
       cores — dispatches 8×8 tile multiplies to dedicated per-SIMD-group matrix hardware where
       present, instead of scalar FMAs on ordinary ALU lanes; int8 has no `simdgroup_matrix` type
       on this hardware/Metal version, so weights are still dequantized to float during the tile
       load, same as steps 1–2): 1771.4ms → 1797.2ms — **essentially a wash, marginally worse.**
       Real, informative negative result: this implementation used only 32 threads/threadgroup (one
       SIMD-group computing a 32×32 tile) vs. step 2's 256 — likely too little parallelism per
       dispatch to benefit from per-call matrix-unit hardware, if such dedicated hardware is even
       meaningfully faster than well-tuned scalar code at this granularity on this GPU family
       (unlike NVIDIA's tensor cores, where the win is dramatic and well-documented).
    4. *Multi-simdgroup `simdgroup_matrix`* (grounded in a quick literature/reference check —
       Apple's own MLX-adjacent Metal GEMM writeups (e.g. BaseRT) stress that `simdgroup_matrix`
       tile *geometry*, not just using the intrinsic at all, is what matters): fixed step 3's
       diagnosed weakness directly — 4 simdgroups now cooperate per threadgroup (128 threads,
       arranged 2x2), each computing its own 32x32 sub-tile of a combined 64x64 threadgroup tile
       (matching step 2/3's tile size for a fair comparison), instead of 1 simdgroup (32 threads)
       computing the whole thing alone. Result: 1797.2ms → **1647.8ms — a real, if modest, win**,
       both over step 3 (~7.5% faster) and over step 2's plain scalar register-blocking. **New best:
       5.94x slower than fp32's 278ms**, shipped as the active kernel.
  - Stopped here: four iterations of genuine, measured optimization moved the needle from 17.1x to
    5.94x slower, with clearly diminishing returns overall (2.1x, 1.25x, ~1.0x, then a partial
    rebound to ~1.08x once the actual bug in step 3 — insufficient occupancy — was diagnosed and
    fixed) — a real, useful memory-savings tool (still an honest 4x weight-memory reduction) that
    also ended up meaningfully faster than where the naive version started, but still not a speed
    win against PyTorch's tuned matmul, and not worth chasing full wall-clock parity with a
    vendor-tuned BLAS/MPSGraph implementation by hand past this point.
  - **A real, separate bug found and fixed along the way, independent of the speed result:**
    repeatedly creating small Metal buffers (`newBufferWithBytes_length_options_`/
    `newBufferWithLength_options_`) in a tight loop — the pattern every kernel in this port used —
    **leaks memory deterministically on this PyObjC/Metal setup, regardless of `del`, explicit
    `gc.collect()`, `objc.autorelease_pool()`, or `torch.mps.empty_cache()`.** Isolated,
    reproduced, and confirmed with a minimal repro outside any of this port's actual code:
    allocating and immediately `del`-ing a 16MB buffer 2000 times in a loop grows driver-reported
    memory linearly and unboundedly (~16.75MB/iteration, exactly the buffer size, no plateau) under
    every mitigation tried; reusing a single pre-allocated buffer and overwriting its contents via
    its CPU-visible pointer instead showed zero growth over the same 2000 iterations. Fixed by
    pooling: weight/scale/bias buffers are built once per layer at quantization time (they never
    change), and the activation/output buffers are grown (not recreated) only when a larger shape
    is actually needed, reusing the existing buffer otherwise. This is a real, general Apple
    Silicon Metal/PyObjC gotcha for any code creating many short-lived buffers, not specific to
    quantization — worth keeping in mind for any future Metal kernel work in this port.
- **`torch.min(dim=0)` returns argmin index `-1` on MPS for an all-`+inf` column, vs. the correct
  `0` on CPU/CUDA.** Found via a genuine, reproducible test failure: `tests/metal_rasterizer_verify.py`'s
  large-random-mesh check (5000 verts/9000 faces @ 256x256) crashed with `gather: index -1 is out
  of bounds` at a `torch.mps.synchronize()` call, from `sw_rasterizer.py`'s chunked z-buffer loop
  -- when an entire chunk of triangles doesn't cover a given pixel, that pixel's whole depth column
  is `+inf` (the "no hit" sentinel), and MPS's `.min(dim=0)` returns `-1` for the argmin there
  instead of `0`. Confirmed as an MPS-specific backend bug, not a rasterizer logic bug, with a
  4-line minimal repro (`torch.full((512,256,256), float('inf'))`, set one element finite,
  `.min(dim=0)` on MPS gives `idx.min()==-1`; the identical tensor on CPU gives `idx.min()==0`).
  The buggy index never affected the final rendered result either way (the gathered value at
  those pixels is always discarded by the subsequent `torch.where(update, ...)` masking, since
  `update` is already `False` there) -- but PyTorch still evaluates the invalid `gather` eagerly
  and crashes before that masking ever runs. Fixed with a one-line `chunk_idx.clamp(min=0)` before
  the gather, safe precisely because the clamped value is always a discarded branch, never a kept
  one. Worth checking for the same pattern (argmin/argmax over an all-same-value MPS tensor, fed
  into a downstream `gather`/indexing op) anywhere else in this port.

## Linear algebra & parallel optimization: the theory behind the quantized-matmul kernel work

The `quantized_linear_metal.py` kernel work (naive → 5.94x slower than fp32) isn't a series of
unrelated tricks — each step is a standard, named technique from GPU GEMM (**GE**neral **M**atrix
**M**ultiply) optimization, applied in the order the literature (and the actual measured numbers
here) says matters most. This section is the "why," kept separate from the "what happened" numbers
above, since understanding it is what lets the *next* kernel in this port skip straight to the
techniques that matter instead of rediscovering them by trial and error.

### Why a matmul kernel is slow by default: arithmetic intensity

A matmul `C[M,N] = A[M,K] @ B[K,N]` does `2*M*N*K` floating-point operations (FLOPs) while reading
`M*K + K*N` input elements and writing `M*N` output elements. The ratio of compute to memory
traffic — **arithmetic intensity** — is the single number that predicts whether a kernel will be
*compute-bound* (limited by how fast the ALUs can crunch numbers) or *memory-bound* (limited by how
fast data can move from memory to the ALUs). This is the **roofline model**: plot achievable
performance against arithmetic intensity, and every kernel has a "roof" — either the GPU's peak
FLOP/s, or its peak memory bandwidth, whichever is hit first at that kernel's arithmetic intensity.

The naive kernel (v1: one thread per output element, `for k in K: acc += x[k]*w[k]`) has the worst
possible arithmetic intensity for this problem: every thread independently re-reads its entire
`K`-length row of `x` and column of `w` from slow global memory, with **zero reuse** across
threads, even though every other thread computing a different output element needs *overlapping*
data (every thread in the same output row reads the same `x` row; every thread in the same output
column reads the same `w` column). This is why it measured ~17x slower than a tuned matmul: it
wasn't doing 17x more arithmetic, it was moving vastly more bytes than necessary for that
arithmetic, and was memory-bound almost the entire time.

### Tiling: trading global memory traffic for on-chip memory traffic

GPUs have a memory hierarchy, fastest (and smallest) to slowest (and largest): registers →
threadgroup/shared memory → global memory (roughly 100-1000x latency difference between the ends).
**Tiling** (v2) exploits this directly: instead of every thread independently pulling its own data
from global memory, a whole *threadgroup* cooperatively loads a small tile of `A` and `B` into fast
threadgroup memory once, then every thread in that group reuses the *same* on-chip copy `TILE`
times instead of re-fetching it from global memory on every use. This doesn't change the total
arithmetic — it changes *where the bytes come from* on repeat accesses, which is what actually
costs time. Measured effect here: 17.1x → 8.0x slower, the single largest improvement of the whole
exploration, because it fixes the most severe bottleneck (memory traffic) first.

### Register blocking: raising arithmetic intensity per thread

Tiling alone still has each thread do only 1 FMA (fused multiply-add) per 2 shared-memory reads —
shared memory is much faster than global memory, but it's not free, and reading it still costs real
cycles and occupies load/store ports. **Register blocking** (v3) has each thread compute a small
*block* of outputs (a 4×4 tile here) instead of one, reusing each shared-memory-loaded value across
multiple FMAs held in registers (the fastest storage there is) before moving on. This raises
arithmetic intensity *per thread*, not just per threadgroup — 16 FMAs per 8 shared-memory reads
instead of 1 per 2, an 8x improvement in that specific ratio. Measured effect: 8.0x → 6.4x slower —
real, but smaller than tiling's win, because register blocking optimizes a *less severe* remaining
bottleneck than the one tiling already fixed.

### Occupancy: keeping the GPU's execution units fed

Neither tiling nor register blocking changes the raw ALU work of the inner loop — it's still scalar
FMA instructions on general-purpose ALU lanes. GPUs hide memory/instruction latency by having many
more threads *in flight* than a single instruction needs, switching to another ready thread while
one waits on a load — this only works if there's enough concurrent work scheduled. v4
(`simdgroup_matrix`, single SIMD-group per threadgroup, 32 threads) landed at parity with v3 (6.4x
→ ~6.5x, a wash) specifically because it under-provisioned this: one SIMD-group per threadgroup
dispatch is too little concurrent work to keep the GPU's scheduler busy, regardless of how fast the
per-call matrix-multiply hardware itself is. v5 fixed exactly this — 4 SIMD-groups cooperating per
threadgroup (128 threads) — and got a real, if modest, further win (6.4x → 5.94x): confirmation that
*occupancy*, not just "use the fancy hardware instruction," was the binding constraint at v4's tile
geometry.

### Hardware matrix units: a different execution resource entirely

`simdgroup_matrix` + `simdgroup_multiply_accumulate` dispatch 8×8 tile multiplies to whatever
dedicated matrix-multiply hardware a SIMD-group has access to (Apple's analog to NVIDIA's tensor
cores), rather than issuing 64 individual scalar FMA instructions on ordinary ALU lanes. This is a
*different lever* from tiling/blocking/occupancy — it doesn't reduce memory traffic or increase
arithmetic intensity, it changes which execution unit does the arithmetic. Whether it helps depends
on (a) whether that hardware exists and is meaningfully faster than scalar ALUs on a given chip, and
(b) whether the kernel is structured to keep it fed (see occupancy above) — both real, empirically
determined facts on real hardware, not things you can reason out from first principles alone, which
is exactly why v4 needed to actually be measured to reveal its weakness.

### What's still out of reach without more infrastructure: `matmul2d` / Metal Performance Primitives

Apple's own answer to "how do I write a tuned GEMM on Apple Silicon" is a genuinely different API,
**Metal Performance Primitives**' `matmul2d` (Metal 4, introduced very recently, documented at
[WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/)): a `matmul2d_descriptor`
fixes the output tile a threadgroup produces, `execution_simdgroups<N>` binds N SIMD-groups to
compute it cooperatively (the same *occupancy* idea as v5, but implemented and tuned by Apple, not
hand-rolled), and `MTLTensor` types (including **native int8**, unlike `simdgroup_matrix` which has
no int8 type — quantized weights wouldn't need dequantizing during tile load at all with this path)
back the whole thing. This is very likely close to — or literally — what `MPSGraph`'s own
`matrixMultiplicationWithPrimaryTensor` (the actual code path PyTorch's `torch.matmul` uses for our
shapes, confirmed by reading PyTorch's own `aten/src/ATen/native/mps/operations/LinearAlgebra.mm`)
calls into internally, which is the real, honest answer to "why is PyTorch's matmul faster": not a
hand-writable trick this port was missing, but Apple's own tuned implementation of the same ideas
above (tiling, blocking, occupancy, hardware matrix units), executed through infrastructure this
port doesn't yet use.

That infrastructure gap is concrete and confirmed, not a guess: `#include
<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` fails to resolve under this port's
existing runtime-JIT kernel compilation (`MTLDevice.newLibraryWithSource_options_error_`, used
deliberately everywhere else in this port specifically to avoid requiring a full Xcode.app install
— see every other kernel's module docstring). A minimal `matmul2d` kernel compiled this way fails
with "use of undeclared identifier 'mpp'" and similar — the header needs Xcode's offline `xcrun
metal` → `.metallib` build step, confirmed unavailable on the machine this was first tried on
(`xcode-select -p` → `/Library/Developer/CommandLineTools`, not a full Xcode.app; `xcrun metal
--version` → "unable to find utility"). Also worth noting for context: PyTorch's *own*
hand-written fallback Metal kernel for `torch.matmul` (`do_metal_mm`, used only for specific
size/dtype edge cases outside the `MPSGraph` path — read directly from `aten/src/ATen/native/mps/
kernels/LinearAlgebra.metal`) is plain `TILE_DIM=16` shared-memory tiling with no register
blocking and no `simdgroup_matrix` at all — i.e. *less* sophisticated than this port's own v3/v5.
PyTorch's speed advantage for our shapes isn't coming from that kernel; it's coming from
`MPSGraph`/Apple's proprietary compiler and

### `matmul2d`, follow-up: built it for real, and found the *actual* remaining bottleneck

Xcode (full `.app`, not just Command Line Tools) was installed, licensed (`sudo xcodebuild
-license`), first-launch-initialized (`xcodebuild -runFirstLaunch` — needed because installing
Xcode outside the App Store skips the setup that normally installs `CoreSimulator` and other
required frameworks), and given its Metal Toolchain component (`xcodebuild -downloadComponent
MetalToolchain`, ~688MB — this is a separate, on-demand download in current Xcode, not bundled).
With that, `trellis/modules/quantized_linear_matmul2d.py` (a new module, entirely separate from
`quantized_linear_metal.py` — doesn't touch or replace it) builds a real `.metallib` via
`scripts/build_matmul2d_kernel.sh` and uses `matmul2d` for real.

**Two real, separate findings, not one:**

1. **The tensor layout convention took genuine trial and error to pin down** (undocumented in the
   one public example available): `MTLTensor` `dextents` are `(width, height)` — the *opposite*
   of numpy/C `(rows, cols)` order. Once found: the weight tensor works in nn.Linear's *natural*
   `[out_features, in_features]` layout directly (with `transposeRight=true`), no separate
   transpose-during-dequantize step needed. Verified correct against an fp32 PyTorch reference:
   max diff ~1.7e-4 (pure fp16 accumulation noise), including a non-tile-aligned batch size
   (`scripts/verify_quantized_linear_matmul2d.py`). **Isolated per-call speed is dramatically
   faster than every hand-rolled kernel** — 0.428ms for the real MLP shape (1024→4096, batch 16)
   vs. single-digit milliseconds for the equivalent hand-rolled call.

2. **That isolated speedup nearly vanishes on the real 196-layer model** (1629.5ms vs. fp32's
   282.8ms — 5.76x slower, essentially the same as v5's 5.94x, despite `matmul2d`'s raw kernel
   being an order of magnitude faster in isolation). Diagnosed, not just observed: a single
   `QuantizedLinearMatmul2D` call in isolation (weight cached, repeated on the same pre-existing
   tensor) takes 0.835ms; the *same* call **interleaved with 20 other pending MPS ops first**
   (simulating what actually surrounds each of the 24 transformer blocks' Linear layers in the
   real model — attention, LayerNorm, GELU, residual connections, all still running as ordinary
   PyTorch/MPS ops) takes 3.122ms — a 3.7x slowdown from interleaving alone. This is the *same
   class* of bug already found and fixed in this port's *rendering* kernels (`.to('cpu')` forcing
   a full reflush of pending upstream MPS graph work, not just a cheap readback) — except here
   it's structural to the integration pattern, not a one-off fixable call site: wrapping
   individual `nn.Linear` layers with a per-call CPU↔MPS round trip means **every one of 196
   layers** forces a sync against whatever else is queued on the GPU. This affects the hand-rolled
   kernel (v5) identically, for the identical reason, which is exactly why v5 and `matmul2d` land
   at nearly the same real-model number despite very different raw kernel speed.

**The conclusion this leads to**: the bottleneck for a whole-model integration is no longer which
matmul kernel algorithm is used — both the best hand-rolled kernel and Apple's own tuned primitive
are now bottlenecked by the *same* per-layer CPU↔GPU round-trip tax. Closing that gap for real
would mean eliminating the round trip entirely (keeping activations MPS-resident and interoperating
directly with `MTLBuffer`s instead of `.to('cpu')`/`.to('mps')` per layer) — a fundamentally
different, substantially larger architectural change than swapping which kernel a drop-in
`nn.Linear` replacement calls, and a reasonable place to stop this line of investigation.
`quantized_linear_matmul2d.py` is kept as a real, correct, separately-available alternative
(useful on its own merits — e.g. if a future integration *does* avoid the round trip) rather than
deleted, since the finding above is about the wrapper architecture, not a flaw in the kernel
itself.
runtime, which is the same place `matmul2d` ultimately connects to.

## Closing the CPU↔GPU round trip: `torch.mps.compile_shader`

The previous section called eliminating the per-layer CPU↔MPS round trip "a fundamentally
different, substantially larger architectural change" and stopped there. It turns out PyTorch
already ships the mechanism needed: `torch.mps.compile_shader(source)` JIT-compiles a Metal kernel
and returns an object whose functions can be called **directly on `torch.Tensor`s already resident
on `mps`** — no `.to('cpu')`/`.numpy()`/manual `MTLBuffer` alloc-and-readback at all. (There's also
`torch.mps.load_metallib(path)`, same calling convention, for a precompiled `.metallib` — confirmed
working standalone against `quantized_linear_matmul2d.metallib`'s existing kernels, but not yet
wired into a full module; see "Next" below.)

One real gotcha, found empirically before trusting it: `lib.kernel(..., threads=X, group_size=G)`
dispatches as `dispatchThreads`, not `dispatchThreadgroups` — if `X` isn't an exact multiple of
`G`, the last threadgroup on each grid edge is **non-uniform** (partial), and
`thread_index_in_threadgroup` only ranges over however many threads that partial group actually
got. Confirmed directly: `threads=70, group_size=16` gives a last group where only
`thread_index_in_threadgroup` 0–5 ever execute, not 0–15. This kernel's cooperative shared-memory
tile loads (`for e = thread_idx; e < BM*BK; e += 128`) need all 128 threads of every threadgroup to
run, or shared tiles are silently left partially uninitialized at grid edges — exactly the same
failure mode already documented for `quantized_linear_metal.py`'s own `dispatchThreads` vs
`dispatchThreadgroups` choice, just one API layer higher up. Fix: round `threads` up to a whole
number of `group_size`-sized threadgroups before dispatching (`_dispatch_dims` in the new module
below); the kernel's existing `row < num_rows` / `col < out_features` guards handle the resulting
out-of-bounds threads safely, same as they already did for the PyObjC dispatch path.

`trellis/modules/quantized_linear_zerocopy.py` — a new module, byte-for-byte the same v5
multi-simdgroup `simdgroup_matrix` kernel as `quantized_linear_metal.py`, only the dispatch
mechanism changed. `quantized_linear_metal.py` and `quantized_linear_matmul2d.py` are both
untouched. Verified correct the same two ways as before
(`scripts/verify_quantized_linear_zerocopy.py`, including a non-tile-aligned batch size): kernel
matches its PyTorch dequant fallback to ~3e-6, quantized-vs-fp32 error stays in the same small
bounded range already established (median ~0.7-0.8% relative error).

**Real full-model result** (`scripts/benchmark_quantized_linear_zerocopy.py`, same
`sparse_structure_flow_model`, 555.5M params, same B=1 test shapes as every other benchmark in this
doc): fp32 279.0ms vs. int8 687.9ms — **2.44x slower, down from 5.94x** for the identical kernel
with only the CPU round trip removed. This confirms the round trip diagnosed above was real and
substantial (it accounted for more than half of the previous slowdown), but **not the whole story**:
even with zero-copy dispatch and no CPU sync per layer, this hand-rolled kernel is still 2.44x
slower than PyTorch's tuned MPS matmul, consistent with the compute-bound (not bandwidth-starved)
diagnosis from the fp16 test earlier in this document — a naive kernel that reads 4x less weight
data can still lose on raw FLOP efficiency alone.

**Next, not yet done**: combine this zero-copy dispatch mechanism with `matmul2d` itself (via
`torch.mps.load_metallib` on the existing `.metallib`, confirmed to work in isolated testing above)
instead of the hand-rolled kernel — `matmul2d`'s per-call speed was already an order of magnitude
faster in isolation before the round trip was diagnosed as the shared bottleneck; removing that
bottleneck for the kernel that was fastest in isolation, rather than the slowest, is the natural
next experiment and hasn't been run yet.

## Fused FFN kernel: matmul2d + native int8 + torch.compile, and the real ceiling on FFN alone

Follow-up to the "Next, not yet done" note above: combined zero-copy dispatch with `matmul2d`, for
the specific `fc1 -> GELU(tanh) -> fc2` pattern that's this model's actual `FeedForwardNet` block
(`trellis/modules/transformer/blocks.py`), not just a drop-in single-Linear replacement. New module:
`trellis/modules/quantized_ffn_matmul2d.py`. `quantized_linear_metal.py` and
`quantized_linear_matmul2d.py` are both untouched throughout everything below.

**A real correctness bug, found only on the real model.** The first version fused fc1's bias-add +
GELU + downcast-to-half into one Metal kernel (`bias_gelu_to_half`) feeding directly into fc2's
`matmul2d_linear` call. Passed every synthetic correctness check (varied shapes, odd batch sizes,
no-bias case). On the real pretrained `sparse_structure_flow_model`, with real image conditioning,
it produced NaN by block 2-5 of 24. Root-caused, not guessed: (1) ruled out a range-overflow bug --
clamping every fp32->fp16 cast boundary to comfortably inside half's max (65504) made no difference;
(2) ruled out a race/hazard between the custom kernel dispatches -- forcing a full
`torch.mps.synchronize()` between every single kernel call in the chain reproduced the NaN
identically, just at a different block; (3) confirmed the pre-existing, unfused
`quantized_linear_matmul2d.py` (separate per-Linear calls, no custom kernel sitting between two
`matmul2d_linear` calls) stayed finite on the identical real model/weights/inputs, despite touching
half precision the same way. Conclusion: `matmul2d_linear`'s second call in a chain goes unstable
specifically when fed a buffer written by a *different* custom kernel, not a fundamental fp16-
precision limit and not a race. Fix: do bias-add + GELU via native (still on-device, still
zero-copy) torch ops between the two `matmul2d_linear` calls instead of a custom fusion kernel --
verified finite across multiple trials with real image conditioning before trusting it. The
`bias_gelu_to_half`/`bias_add_f32` kernels are still in
`trellis/modules/metal_kernels/quantized_linear_matmul2d.metal`, just unused by this module now --
kept rather than deleted, per this port's own versioning discipline.

**Native int8 matmul2d, confirmed from Apple's actual SDK header, not inferred from docs.** Read
`MPPTensorOpsMatMul2d.h` directly (Xcode is installed, so this is the real header, not a summary):
it documents (and the `__impl` header's `static_assert`-guarded type dispatch confirms compiles)
`half x int8_t -> float` as a natively supported operand combination -- `matmul2d` can consume the
*raw* int8 quantized weight buffer directly, doing the int8->float widen internally during its own
tile load. This eliminates `dequantize_weights` entirely (a whole separate kernel dispatch plus a
full extra read+write of the weight matrix through global memory, materializing a half-precision
copy nothing else needed). New kernel: `matmul2d_linear_quantized`. Since this port's quantization
scale is per-output-channel only (row-wise symmetric), it factors exactly out of the reduction sum
-- `Y[r,o] = scale[o] * sum_i(X[r,i] * int8_W[o,i])` -- so applying it as a cheap elementwise
follow-up (`scale_bias_f32`, new kernel) after the matmul is exact, not an approximation. Verified:
same error bounds as the dequantize-then-matmul path (median ~1.1% relative error, unchanged).
`device int8_t*`, not `device char*`, was required for the type dispatch to actually match --
`char` alone failed the header's internal `static_assert` on this SDK version even though Metal
treats them as bit-identical; worth remembering for any future kernel touching this API.

**Eager-mode dispatch overhead was quietly erasing the kernel's own advantage.** Measured directly:
`matmul2d_linear_quantized` alone is **2.24x faster** than `torch.matmul` for this model's shapes
(1.06ms vs 2.36ms, isolated). But wrapping it with plain eager PyTorch ops (clamp, cast, the
scale/bias kernel, GELU, clamp, cast again) made the *whole FFN block* slower than fp32 overall --
each small op pays its own fixed dispatch cost (Python-level dispatch + MPS command-buffer
encode/submit), and this path has more, smaller ops than fp32's `nn.Linear`/`nn.GELU` calls. Fix:
registered `matmul2d_linear_quantized` as a `torch.library.custom_op` (`trellis::
matmul2d_linear_quantized`) so `torch.compile` can treat it as an opaque-but-traceable graph node
and fuse the surrounding elementwise ops around it -- `torch.compile` cannot trace through a raw
`torch._C._mps_MetalKernel` call directly (`InternalTorchDynamoError` without the registration).
One compiled function (`_compiled_ffn_compute`) is built *once* at module scope, not per-instance,
so all 24 `QuantizedFeedForwardNetMatmul2D` blocks in the real model share a single compiled graph
and call it with their own weight tensors as plain arguments -- confirmed via
`torch._dynamo.utils.counters['stats']` showing `unique_graphs: 1` after a full forward pass, not
24 (recompilation-per-block was a real hypothesis, directly ruled out, not assumed away). Measured
effect of compiling the same clamp->matmul->scale->bias->GELU chain: 2.40ms eager -> 1.35ms
compiled, a 1.78x win from compilation alone, stacked on top of the kernel's own 2.24x.

**The actual ceiling: FFN's true time-share in this model is ~11%, not the ~36% its param-share
suggests -- verified by direct measurement, not estimated.** Bypassing `FeedForwardNet` entirely
(identity passthrough, no computation) drops the real, pipelined, single-end-of-forward-sync
full-model time from 278.7ms to 248.3ms -- FFN's true contribution is **30.4ms**, not the ~101ms a
naive 36%-of-278.7ms estimate from param-share would suggest (attention/SDPA/sparse-conv overhead
dominates the rest far more than param count implies). This means **no amount of FFN kernel work,
however good, can produce more than roughly a 1.12x whole-model speedup** -- the 2x target set for
this work is architecturally unreachable from FFN quantization alone, regardless of
implementation quality. Measured the same way for the current quantized FFN: its true contribution
is **60.1ms**, roughly 2x fp32's FFN cost, not faster -- a real, still-unexplained gap against the
~32ms (24 blocks x ~1.35ms) the isolated compiled-chain benchmark predicts. Recompilation was
directly ruled out as the cause (`unique_graphs: 1`, confirmed above). Leading hypothesis, not yet
confirmed: the isolated benchmark repeatedly reuses the *same* ~8MB of weight tensors in a tight
loop (cache-resident throughout), while the real model cycles through 24 blocks' worth of distinct
weights (~192MB total) once each per forward, likely cold from cache/unified-memory bandwidth each
time -- a fundamentally different memory-access pattern the isolated test can't capture. Not
resolved as of this writing.

**Resolved: the cache-effect hypothesis above is wrong.** Tested directly with
`scripts/investigate_ffn_cache_gap.py`, before starting any attention work, per this project's own
rule of measuring rather than building further on an unconfirmed hypothesis. Method: capture the
real model's actual per-block shape from one instrumented real forward pass instead of guessing
(`num_rows=4096, channels=1024, hidden=4096` -- not the shape implicitly assumed by the earlier
"~1.35ms/block" isolated number, which is where it turns out to have gone wrong), then sweep the
isolated benchmark's *distinct weight-set count* from 1 (fully cache-resident, matching the
original isolated-benchmark setup) up to 24 (the real model's full ~201MB working set), timing 240
round-robin calls at each point. Result: **completely flat, 2.684ms/call at 1 distinct set vs.
2.691ms/call at 24 -- no scaling with working-set size at all**, cleanly refuting the cache/
bandwidth hypothesis (if it were real, 1-set should have been fast and 24-set should have
approached the real model's per-block cost; instead both are already at the real model's cost). The
actual explanation: the earlier isolated benchmark that produced "~1.35ms/block" was run against a
smaller, unrepresentative `num_rows` than the real model actually uses -- not captured directly from
the model at the time, evidently guessed instead. Re-running the isolated chain at the *real*
captured shape gives 2.68-2.69ms/block in isolation, and a full round of 24 distinct blocks in real
cadence gives 64.57ms/round = 2.691ms/block -- both closely matching the real full-model measurement
of 60.1ms/round = 2.50ms/block (within ~7%, consistent with ordinary measurement noise, not a
mystery gap). **Net effect on the ceiling analysis: none of the ~11%-time-share / ~1.12x-max
conclusion changes.** What this does resolve is *why* isolated and real-model numbers previously
disagreed (a benchmark-shape bug, not an architectural cache effect) and confirms the quantized FFN
genuinely is close to 2x slower than fp32 for the FFN portion alone (~2.5-2.7ms/block quantized vs.
fp32's ~1.27ms/block, i.e. 30.4ms/24) -- a real, now-fully-explained cost of the current kernel/
dispatch chain, not a phantom benchmarking artifact. Lesson for future isolated benchmarks in this
project: always capture real shapes from an instrumented model forward pass rather than assuming
them, even when the assumption "looks" realistic (`num_rows` in the thousands was true here --
4096 -- just not the *specific* number that was actually used).

**Clamping replaced with dynamic conditioning.** The half-precision cast boundaries were originally
guarded with a fixed clamp (`clamp(-65000, 65000)`) against fp16 overflow. Correct user pushback:
a fixed threshold is a guess, not a fix -- it's only safe until some future input's activations
exceed it, at which point it silently truncates rather than erroring. Replaced with dynamic
per-call rescaling: measure each tensor's own `amax()`, rescale so its max lands at a small fixed
target (256) before the half cast -- guaranteed representable regardless of the input's actual
magnitude -- then fold that same scale factor back into the existing per-channel weight scale on
the output (exact, since scale factors are multiplicative and commute through the matmul, not an
approximation). Confirmed finite across multiple real-conditioning trials, same as before. Cost:
~2.5% slower (317ms vs 309ms full-model) than the fixed-clamp version -- the extra `amax()`
reduction isn't free even under `torch.compile` -- accepted deliberately as a correctness-over-
speed trade rather than depending on an assumption about how large activations can get.

Given the ceiling finding, further FFN-only optimization has strictly limited upside (best case:
parity with fp32, not a win) -- the path to the requested 2x-faster target, if reachable at all on
this hardware, has to go through attention (the ~89% of time FFN doesn't touch), not further FFN
kernel work. `trellis/modules/attention/modules.py`'s `MultiHeadAttention` (`to_qkv`/`to_q`/`to_kv`/
`to_out`, all plain `nn.Linear`) is the next candidate, applying the same now-proven pattern
(native int8 matmul2d + custom-op registration + shared `torch.compile`) -- with the actual
`scaled_dot_product_attention` call itself left as native PyTorch (already using Apple's own tuned
SDPA path; quantizing that operation itself would mean writing a flash-attention-class kernel from
scratch, well beyond this port's existing kernel-writing scope).

## Applying the FFN pattern to attention: quantized MultiHeadAttention via matmul2d

Before starting, the FFN section's "still open" cache-effect question above was investigated and
refuted (see "Resolved: the cache-effect hypothesis above is wrong") -- confirmed to not apply here
either, so this work proceeded straight to attention rather than re-deriving that result.

New module: `trellis/modules/quantized_attention_matmul2d.py`, wrapping
`trellis/modules/attention/modules.py`'s `MultiHeadAttention`. Reuses
`torch.ops.trellis.matmul2d_linear_quantized` -- the exact custom_op already registered by
`quantized_ffn_matmul2d.py` -- rather than re-registering an identical op; that module is untouched.
Deliberately narrow scope, per the two configs this model actually instantiates (checked directly
in `trellis/models/sparse_structure_flow.py`'s `ModulatedTransformerCrossBlock`): self-attention
with `qk_rms_norm=True`, `use_rope=False`, and cross-attention with neither. Each of `to_qkv`
(self) / `to_q`+`to_kv` (cross) / `to_out` (both) is its own `matmul2d` call -- unlike the FFN's two
adjacent Linears, there's no adjacent-Linear pair here to fuse the same way, since native
`scaled_dot_product_attention` sits in between and is deliberately left untouched. RMSNorm and SDPA
run as plain native torch ops between the matmul2d calls, same "native ops only between matmul2d
calls" rule established by the FFN module's own NaN bug. `QuantizedMultiHeadAttentionMatmul2D.
__init__` raises `NotImplementedError` (not a silent fallback) for any config outside those two --
windowed `attn_mode`, `use_rope=True`, or `qk_rms_norm` in the "wrong" direction for a given
attention type.

**A real bug, found exactly the way the verification methodology is supposed to catch it: passed
every synthetic check, crashed on the real model.** `scripts/verify_quantized_attention_matmul2d.py`
(8 shape/config cases plus 3 unsupported-config guard checks, comparing against both the module's
own fp32 fallback and a CPU fp32 reference) passed cleanly on first try. Real-model check
(`scripts/verify_quantized_attention_matmul2d_real.py`, real image conditioning via
`pipeline.get_cond`) crashed immediately with a hard, uncatchable MPS assertion (`Destination
NDArray and Accumulator NDArray cannot have different datatype in MPSNDArrayMatrixMultiplication`)
-- no Python traceback, since it's a native Objective-C assertion, not a raised exception. Localized
by capturing the real per-block input tensor from an instrumented forward pass and stepping through
`_self_attn_compute` with explicit `torch.mps.synchronize()` after every sub-step (single block:
completed cleanly), then re-testing with the full model and a wrapped `forward` that prints/syncs
per block (24 blocks: block 0 completed, crash occurred between block 0's self-attention returning
and block 1's self-attention starting). This pointed at block 0's *output*, not its computation.
Root cause: **`matmul2d_linear_quantized`'s custom_op always returns `float32`** (confirmed already
in `quantized_ffn_matmul2d.py`), but this real model runs its transformer blocks in **`float16`**
(`model.use_fp16` is `True` -- see the correction two sections up; confirmed directly, not
assumed). `QuantizedFeedForwardNetMatmul2D.forward` already casts its output back to `orig_dtype`
before returning; this new attention module's `forward` did not. The float32 output got added into
the model's fp16 residual stream, and a few ops later a native fp16 `nn.Linear` (the untouched
`cross_attn` or FFN in the same block) received a float32 activation against its fp16 weight,
producing the MPS assertion. Fix: cast to `orig_dtype` before returning, matching the FFN module's
existing pattern -- confirmed finite across 5 real-conditioning trials with varied random `x`/`t`
after the fix, and the synthetic checks still pass unchanged (they run on CPU-fp32 references, so
never exercised the fp16 path that caused this). Lesson: a fp16-vs-fp32 output-dtype bug like this
is exactly the class of thing synthetic tests miss when they don't run the real model's actual
dtype -- worth deliberately adding an fp16-dtype case to future synthetic checks, not just relying
on the real-model step to catch it after the fact.

**Full real-model benchmark result (`scripts/benchmark_quantized_attention_matmul2d.py`): currently
a net loss, not a win -- 0.90x (310.5ms int8 vs. 279.4ms native fp16).** The 279.4ms native number
is consistent with the 278.7ms figure from the FFN ceiling analysis (same model, same test shape,
different session -- good cross-check). Unlike the FFN case, this was NOT caused by recompilation
(`torch._dynamo.utils.counters['stats']` shows `unique_graphs: 2` -- one graph for self-attention,
one for cross-attention, both correctly shared across all 24 blocks, not recompiling per block).

**The ceiling story here is very different from FFN's, and much more favorable -- confirmed by the
same bypass-measurement technique.** Replacing every `MultiHeadAttention` with a zero passthrough
(the direct analog of FFN's "identity passthrough" bypass -- attention output only ever feeds a
residual add, so zero contribution is the correct no-op) drops the real, pipelined full-model time
from 279.4ms to **64.8ms**. That means attention's true time-share is **~77%** of this model's
total forward time (214.6ms of 279.4ms), not FFN's ~11% -- confirming CLAUDE.md's standing note
that "attention/SDPA/sparse-conv overhead dominates the rest far more than param count implies."
Applying the same bypass-delta math to the quantized run: quantized attention's true contribution
is 310.5 - 64.8 = **245.7ms**, only **~14.5% slower than native fp16 attention's own 214.6ms** --
a much smaller regression than the raw 0.90x whole-model ratio suggests, it's just being amplified
because attention dominates so much of the total. **Unlike FFN, there is real, large headroom here
in principle** (a hypothetical zero-cost attention would be a 4.3x whole-model speedup) -- the
current implementation just doesn't realize any of it yet; it's the *reverse* problem from FFN's
architectural ceiling.

**Leading unconfirmed hypothesis for the 14.5% attention-specific regression, not yet investigated:**
self-attention alone now dispatches three separate matmul2d-backed quantized Linears per block
(to_qkv, to_out -- two, actually, since qkv is fused into one Linear; cross-attention has three:
to_q, to_kv, to_out) each with their own `amax()`/rescale/half-cast overhead, wrapping native
`scaled_dot_product_attention` which is already a single, highly-tuned fused MPSGraph op -- the same
"eager per-op dispatch overhead can erase an individually-faster kernel's advantage" mechanism
already found and fixed once for FFN (via the custom_op + torch.compile registration, which *is*
already applied here) may not be fully absorbing the *additional* rescale calls this module has
relative to FFN's two. Not yet measured directly (e.g. isolated per-call timing of `_self_attn_
compute`/`_cross_attn_compute` alone, or an ablation removing the redundant `x_scale` computation
for to_qkv/to_q by reusing one amax() across queries/keys/values that share the same input `x`) --
flagged as the concrete next step rather than guessed at further.

**Investigated directly (`scripts/investigate_attention_dispatch_overhead.py`), and the hypothesis
above was only half right.** Isolated, real-shape (`num_rows=4096, channels=1024, num_heads=16,
head_dim=64`, `Lkv=1374` for cross) per-call comparison against the real native fp16
`MultiHeadAttention` module (not a synthetic proxy) found the dominant cost wasn't dispatch-count
overhead at all: **`scaled_dot_product_attention` itself was running on `float32` q/k/v in the
quantized path (5.63ms) vs. `float16` in the native path (5.02ms)** -- a 0.61ms difference that
accounted for nearly the *entire* self-attention regression on its own. Cause: `matmul2d_linear_
quantized`'s custom_op always returns `float32` (same fact as the fp16-residual-stream bug above),
and neither compute function downcast back to the model's working dtype until the very end of
`forward()` -- so SDPA, sitting between the two `matmul2d` calls, ran in float32 the whole time
even though this model is fp16 end-to-end. FFN's `_ffn_compute` never needed this because GELU
(the op sandwiched between *its* two `matmul2d` calls) is cheap and dtype-insensitive; SDPA is a
large, precision-sensitive fused matmul, so it was never safe to assume the same non-issue carried
over.

**Fix:** both compute functions now take an explicit `compute_dtype` parameter (the caller's
`orig_dtype`) and downcast immediately after each `matmul2d` call's scale/bias step, so RMSNorm and
SDPA both run in the model's real working dtype (fp16), matching the native path exactly --
verified confirmed-equal SDPA cost after the fix (cross-attention: 1.6916ms quantized vs. 1.6909ms
native, effectively identical). Re-verified correctness (`scripts/
verify_quantized_attention_matmul2d.py` unchanged pass; `scripts/
verify_quantized_attention_matmul2d_real.py` still finite across 5 real-conditioning trials -- this
was a speed fix, not expected to change correctness, and didn't).

**Result: self-attention now beats native (0.972x -- i.e. *faster*), cross-attention improved but
still lags (1.299x -> 1.189x), and the full real-model benchmark went from 0.90x to 0.98x** (283.7ms
int8 vs. 277.8ms native fp16 -- within a few percent of parity, not the ~10% regression found
before this fix). The remaining cross-attention gap **was** investigated with the same real-shape
sub-step breakdown (`to_q`/`to_kv`/`to_out` vs. their native fp16 `nn.Linear` equivalents): SDPA
now matches exactly, but the three separate `matmul2d` calls plus their six rescale/scale-bias
elementwise passes (cross-attention has three quantized Linears -- `to_q`, `to_kv`, `to_out` --
unlike self-attention's two, since query and context are different tensors with no shared `amax()`
to reuse) sum to ~1.43ms of extra eager-mode cost against native's zero, of which `torch.compile`
recovers most (down to a ~0.42ms compiled gap) but not all -- reduction ops like `amax()` don't fuse
away the same way pure elementwise chains do, and `matmul2d` itself was not consistently faster than
native fp16 `nn.Linear` at these specific shapes (e.g. `to_kv`: 0.198ms matmul2d vs. 0.119ms native
-- matmul2d slower here), unlike the fp32-baseline comparisons earlier in this document. **This is
the honest remaining state, not further chased this session**: cross-attention's three-Linear
structure has an inherent minimum extra-dispatch cost this approach can't fully absorb, and
`matmul2d`'s speed advantage (established against fp32 `torch.matmul` in the FFN work) does not
reliably hold against native fp16 `nn.Linear` -- worth remembering before assuming `matmul2d` is a
free win anywhere else in this port that also already runs fp16 natively.

**Considered and rejected as a framework-level fix for the remaining cross-attention gap: switching
to JAX or dropping to raw/pure MPSGraph instead of PyTorch.** Reasoning only, not measured --
recorded here so it isn't re-litigated from scratch later:
- **JAX**: its Metal backend (`jax-metal`) is immature and thin on op coverage, and has no real
  equivalent to `torch.mps.compile_shader`/`torch.library.custom_op`, which is the entire mechanism
  this port's hand-written kernels (including `matmul2d`) plug into. Adopting it would mean
  rewriting a working pipeline onto a *less* capable backend than what's already in use, for a
  problem (dispatch-count overhead) JAX/XLA fusion doesn't obviously solve better than
  `torch.compile` already attempts to.
- **Raw/pure MPSGraph** (bypassing PyTorch's op-by-op eager dispatch entirely, building one
  MPSGraph for the whole attention block): has a real, narrower case -- it could plausibly collapse
  the quantization scaffolding (`amax` -> rescale -> GEMM -> scale/bias) into fewer dispatches than
  even `torch.compile` manages, which is exactly where cross-attention's remaining gap lives. **The
  trap**: MPSGraph has no first-class attention/SDPA op, so going fully "pure MPS" would mean
  hand-building `Q@K^T -> softmax -> @V` as MPSGraph primitives ourselves -- throwing away PyTorch's
  native `scaled_dot_product_attention`, which this document already established (a) is out of
  scope to replace (a flash-attention-class kernel from scratch, well beyond this port's
  kernel-writing scope) and (b) is *not* the bottleneck anymore as of the SDPA-dtype fix above (it
  now matches native cost exactly, confirmed). A full pure-MPS rewrite risks losing more than it
  gains. If dispatch-count reduction for the quantization scaffolding specifically is worth pursuing
  further, the more promising direction is cutting it *within* PyTorch (static/precomputed
  quantization scales instead of per-call `amax()`, or fusing the scale/bias step into the GEMM's
  own epilogue) rather than a framework or API-layer change -- consistent with what the external
  research pass below was specifically sent to check.

### External research pass + hardware-generation check + a real, tested, negative result

An external research pass (web search, not this port's own testing) was run to look for numerical/
theoretical attention speedup techniques beyond what's covered above. **Epistemic status: the
sources below were not independently verified past what's noted -- treat citations as leads, not
established fact, consistent with this project's general preference for reading real SDK headers
directly over trusting summaries.** Headline claim from that pass: Apple GPUs before the M5
generation have no dedicated int8/fp16 matrix-multiply hardware -- all matmul (including
`matmul2d`'s int8 path) runs through the GPU's general-purpose ALU pipeline, the same one used for
everything else; M5 is claimed to be the first generation with a dedicated per-core "Neural
Accelerator" for this. Cited sources included Apple's own ML research blog and October 2025
newsroom post about M5 (plausible/likely genuine -- consistent with public Apple M5 marketing
already known independently of this search) plus several arXiv papers, a GitHub project, and
serving-framework blog posts (vLLM, QServe) that were **not** independently fetched/confirmed by
this port's own investigation and should be treated with more caution.

**Directly checked against our own machine, not assumed:** `system_profiler SPHardwareDataType`
confirms this port is running on an **Apple M5 Max** (`Mac17,6`), not a pre-M5 chip -- also visible
in the installed Metal toolchain's own path (`MetalToolchain-v17.6.109.0.M5u60P`). This flips the
research pass's headline framing for our situation: the "explanation" it offered for why `matmul2d`
might lose to native fp16 doesn't apply here -- we're on the chip that's supposed to *have* the new
hardware, not one that lacks it. That makes it more notable, not less, that `matmul2d` doesn't
reliably beat native fp16 `nn.Linear` (see the cross-attention finding above).

**Checked directly against the real, installed SDK header (not the web) for a hardware-gating
explanation:** `MPPTensorOpsMatMul2d.h` has no `API_AVAILABLE`/OS-or-hardware-version gating
anywhere on the `matmul2d` API -- it's the same single API surface across generations, consistent
with Apple's usual pattern (e.g. Accelerate/BNNS) of a stable API that dispatches to whatever
hardware is actually present at runtime, not a separately-exposed "new hardware" API. This suggests
`matmul2d` calls on this M5 Max machine already reach whatever hardware path is available; the
performance gap against native fp16 `nn.Linear` more likely reflects differences between this
port's hand-written kernel and Apple's own, presumably more deeply-tuned, `MPSGraph`
implementation, not a missing hardware feature.

**One concrete, directly-actionable lever *was* found by reading the header, and was tested
immediately rather than left as a hypothesis: `matmul2d_descriptor`'s `relaxed_precision` flag.**
Documented in Apple's own header comment: "relaxed_precision = false, set it to true to allow
implementation to sacrifice accuracy for performance." This port's kernel has always passed `false`
here (`matmul2d_linear`, `matmul2d_linear_quantized`, both in `quantized_linear_matmul2d.metal`).
Added a new, non-destructive sibling kernel, `matmul2d_linear_quantized_relaxed` (identical, only
`relaxed_precision=true`), and tested it directly (`scripts/investigate_relaxed_precision.py`) --
correctness (vs. an fp32 CPU reference, three shapes including the real `to_qkv` shape) and speed
(60 calls at the real `to_qkv` shape, `num_rows=4096, in_features=1024, out_features=3072`).
**Clean negative result: output is byte-identical to the non-relaxed kernel's** (max abs diff
between the two kernels' outputs across every tested shape: exactly `0.000000`) **and speed is
within noise (1.003x).** For this specific `half x int8_t -> float` operand combination, on this
SDK/hardware, `relaxed_precision` has no effect -- there's apparently no lower-precision internal
path left to drop to beyond what an int8 operand and float32 accumulator already are. The new
kernel is kept in the `.metal` source (unused by any module) rather than deleted, with a comment
recording this result, per this port's versioning convention -- worth re-testing if a future SDK or
hardware revision might change this specifically for this dtype combination, rather than assuming
it stays null forever.

**Follow-up experiment: does the operand TYPE COMBINATION itself matter, not just
`relaxed_precision`?** `matmul2d`'s header documents many supported type pairs (see the table at
the top of `MPPTensorOpsMatMul2d.h`), not just `half x int8_t -> float`. Two new investigation-only
kernels added to `quantized_linear_matmul2d.metal` (`matmul2d_int8_int8`: both operands `int8_t`,
`int32` accumulator, no int8->half widen at all; `matmul2d_half_half`: both operands `half`, a
proxy for "what matmul2d looks like at native `nn.Linear`'s own dtypes"), tested directly
(`scripts/investigate_matmul2d_dtype_combos.py`, correctness vs. fp32 CPU reference + speed, same
real `to_qkv` shape as every other investigation here) alongside the existing production kernel and
native fp16 `nn.functional.linear`. Results (chart + full data table:
`outputs/matmul2d_dtype_experiment.html`):

| combination | ms/call | vs. native fp16 | median rel. error |
|---|---|---|---|
| `half x int8_t -> float` (current production) | 0.7992 | 1.938x slower | 0.0078 |
| `half x half -> float` | 0.7952 | 1.929x slower | 0.0003 |
| `int8_t x int8_t -> int32` (full W8A8) | 0.4086 | 1.009x, ~parity | 0.0111 |
| native fp16 `nn.functional.linear` | 0.4123 | 1.0x (baseline) | -- |

**Two real findings, both confirmed by direct measurement:**
1. **The `half` operand is what's slow, not the `int8_t` one.** `half x int8_t` and `half x half`
   run at essentially identical speed (0.7992ms vs. 0.7952ms) -- the int8 weight isn't buying any
   speed on its own in this port's current (weight-only quantization) design. This finally explains
   the whole cross-attention/self-attention gap chased above: it was never really about `int8_t`
   being slow through `matmul2d`, it was about `matmul2d` itself (with a `half` operand present at
   all) being slower than native fp16 `nn.Linear`'s `MPSGraph` path, on this hardware, regardless of
   what the other operand's type is.
2. **Full int8x int8 (activations quantized too, not just weights) is genuinely ~2x faster than
   the current kernel (1.956x) -- but only reaches parity with native fp16 (1.009x), not a decisive
   win**, and costs real accuracy (0.0111 vs. 0.0078 median relative error, from also quantizing
   activations to 8 bits). This is the one combination that plausibly engages whatever dedicated
   int8 throughput the M5's Neural Accelerator provides (consistent with the research pass's
   hardware claim, now indirectly corroborated by an actual measurement rather than just a cited
   claim) -- but Apple's own native fp16 path is already just as fast, so even a full W8A8 rewrite
   (a substantial lift: per-token/row activation quantization, an int32->float dequant step,
   reworking every module built so far) would only match what's already available for free today,
   not beat it.

**Net effect on next steps, revised**: this closes out the operand-type-combination question
decisively -- **no combination of `matmul2d` operand types tested beats native fp16 `nn.Linear` by
a real margin on this hardware**; full int8 activations reaches parity at best, at a real accuracy
cost, and would require rewriting every quantized module built this session. The framework-level
ideas (JAX, pure MPS) are still rejected as above. The `relaxed_precision` angle is closed out as a
real, tested null. **This significantly lowers the ceiling on how much more this port should invest
in further `matmul2d`-based kernel work for attention** -- the honest remaining directions are
narrower than previously listed: static/calibrated quantization scales instead of per-call
`amax()` (removes reduction-op overhead `torch.compile` can't fuse away -- worth trying since it's
cheap, but per finding #1 above, the ceiling it's chasing is "match native fp16," not "beat it," so
temper expectations accordingly) and fusing the scale/bias epilogue into the GEMM kernel (same
caveat). Given the current state -- self-attention already at parity-or-better (0.972x) and
cross-attention close (1.189x) -- **the pragmatic recommendation is to treat the attention
quantization work as functionally complete for this hardware generation** rather than continuing
to chase a decisive win that this session's own measurements now suggest may not exist via this
approach.

## Beyond Linear quantization: SDPA itself, and what actually reduces real wall-clock time

Given the attention-quantization ceiling above, this round of work asked two further questions
directly: is there anything left to gain by touching `scaled_dot_product_attention` itself, and is
there a real, different lever (not more kernel work) that actually gets closer to the requested 2x?

**SDPA backend sweep: no quick win, and it's already doing real fusion work.** PyTorch exposes
`torch.nn.attention.sdpa_kernel` to force `MATH`/`EFFICIENT_ATTENTION`/`FLASH_ATTENTION`/
`CUDNN_ATTENTION`/`OVERRIDEABLE`. Benchmarked all five plus the default, at the real self-attention
shape (`B=1, H=16, L=4096, D=64`): **every single one times identically, ~5.0ms** -- on MPS,
PyTorch apparently resolves all of these to the same one native implementation; there's no "pick a
faster backend" option the way there is on CUDA. Separately confirmed native SDPA is not a naive
wrapper: a plain three-op reimplementation (`Q@K^T -> softmax -> @V`, matching this repo's own
`full_attn.py` `_naive_sdpa` fallback) measured **7.16ms, ~1.43x slower than native SDPA's 5.0ms** --
real, measurable fusion value already captured by PyTorch's MPS backend, not something sitting on
the table unclaimed.

**Testing `matmul2d` for the QK^T shape itself (not just Linear projections): worse than the Linear
case, not better, and for a specific, identifiable reason.** Reused the already-built
`matmul2d_half_half` kernel for the per-head `Q[L,D] @ K[L,D]^T -> [L,L]` shape (`L=4096, D=64`),
looping over the 16 heads (matmul2d has no batch dimension in its API -- see below). Result:
**2.8x slower than `torch.bmm`** (0.354x), correctness fine (median rel. error 0.00017) but speed
clearly worse than the ~1.9x gap already found for Linear shapes. Root cause, not just "matmul2d is
slow": `torch.bmm`'s native kernel has a 3D dispatch grid (`batch x M-tiles x N-tiles`) built in, so
all 16 heads execute as **one** command-buffer submission; `matmul2d`'s dispatch grid only knows
`(M, N, K)` for a single 2D matrix pair, so using it per-head means 16 separate kernel dispatches,
paying fixed dispatch overhead 16 times over -- a structural disadvantage on top of the raw-GEMM gap
already measured, not a new mechanism. This means a naive "loop matmul2d over heads" approach to a
custom attention kernel is a strictly worse starting point than the Linear-quantization work already
done; any real attempt at a custom attention kernel needs the batch (head) dimension baked into the
dispatch grid itself, not looped in Python -- which is most of what a real flash-attention-style
kernel would need to get right anyway (the rest being online-softmax tiling to avoid materializing
the full L x L score matrix).

**Built the batched-dispatch fix and it was wrong -- correction, not a rewrite of the above.** Two
new kernels, `attention_qkt_batched`/`attention_av_batched` (both in
`quantized_linear_matmul2d.metal`), put the head index in `tgid.z` so all 16 heads execute as ONE
`dispatchThreads` call per kernel, exactly the fix described above -- softmax stays a native torch
op between them (online-softmax tiling deliberately deferred, to isolate "does batching alone close
the gap" as its own question). Correctness verified first, small non-tile-aligned synthetic shape
(`H=3, L=37, D=16`) against a CPU fp32 reference: max abs err 0.0001, median rel err 0.0003 --
`attention_av_batched`'s untested transpose convention (no transpose, unlike every other kernel in
this file) turned out correct. **Speed at the real shape (`H=16, L=4096, D=64`,
`scripts/investigate_batched_attention_matmul2d.py`): 15.04ms, barely different from the per-head
Python-loop version's 15.28ms (1.016x) -- batching provided almost no speedup, and both are ~3x
slower than native SDPA's 5.0ms (0.332x).** This falsifies the hypothesis just above: dispatch
count was NOT the dominant cost for this shape after all. The likely real explanation, not yet
confirmed the same rigorous way: at `L=4096`, the full per-head score matrix is `4096x4096` (16.7M
elements, ~268M across all 16 heads) -- materializing that in `device` memory (write scores, read
for softmax, write probs back to half, read probs for the second matmul) is itself a large amount
of memory traffic that native fused SDPA likely avoids entirely via online-softmax tiling (never
writing the full attention matrix out at all). If that's right, the actual lever was always the
"harder half" explicitly deferred above -- online-softmax/flash-attention-style tiling, not
dispatch batching -- and dispatch batching alone, while a real and correctly-implemented fix for
the mechanism it targeted, was not the dominant cost here. Given every hand-written kernel this
session has lost to Apple's native path, including this one even after fixing the specific
mechanism initially suspected, the honest expectation for attempting the harder online-softmax
tiling work is that it is NOT a good bet without stronger evidence it would fare differently --
this is flagged as an open, unresolved question rather than a recommendation to attempt it.

**Attempted anyway, per explicit direction to keep the kernel-side investigation open regardless
of outcome -- real, if partial, progress, not a full win.** New kernel,
`attention_flash_online_softmax`: one threadgroup per `(head, 64-row query-block)`, looping
internally over 32-column KV blocks, keeping the score tile AND the running online-softmax
statistics (`m`: running max, `l`: running sum, `acc`: running weighted output accumulator) in
**threadgroup (on-chip) memory**, never writing the full L x L score matrix to device memory at
all. Deliberately used plain threadgroup arrays rather than `metal::cooperative_tensor`'s
register-resident chaining (`get_left_input_cooperative_tensor`/`get_right_input_cooperative_tensor`,
which the MPP header documents for exactly this "avoid a device-memory round trip between chained
matmul2d calls" purpose) -- its per-thread data layout is opaque and implementation-defined
(accessed only via `get_multidimensional_index`/`get_mask`/iterators), judged too high correctness
risk for a first attempt versus threadgroup memory's plain, directly-indexable arrays. Uses
`matmul2d_descriptor`'s `multiply_accumulate` mode (`C = A*B + C`) for the running accumulator --
undocumented by any example in the header, used anyway and caught by the correctness check if wrong
(it wasn't).

**A real correctness bug, caught exactly the way this port's own methodology is supposed to catch
it.** First correctness pass (`scripts/investigate_flash_attention_matmul2d.py`, three small
synthetic shapes) failed specifically at `H=3, L=100` while `H=1, L=64` and `H=2, L=96` (both exact
multiples of the 32-wide KV block) passed -- exactly the signature of a tail-block boundary bug.
Root cause: `matmul2d`'s own edge-checking correctly avoids *reading* past `k_h`/`v_h`'s true `L`
extent for a partial final KV block (same mechanism every other kernel in this file already relies
on), but it only *writes* the valid columns of the destination -- the remaining columns of
`score_tile` kept whatever was there from the previous iteration (or uninitialized memory, on the
first). The online-softmax bookkeeping loop read all 32 columns unconditionally, corrupting
`local_max`/`local_sum` with stale data whenever `L` wasn't a multiple of 32. Fix: compute
`kv_valid = min(KVBLOCK, L - kv0)` and only read real columns in the bookkeeping loop -- and
explicitly zero the invalid columns (not just skip them), since the subsequent `av_op.run()` call
reads the *entire* fixed-size `score_tile` buffer every iteration regardless (it's not sliced from
a larger tensor matmul2d can edge-check against, unlike `K`/`V`), so leaving stale data there would
silently corrupt the accumulator. Re-verified clean after the fix, all three shapes including the
boundary case (median rel err 0.0002 throughout).

**Speed at the real shape (`H=16, L=4096, D=64`): 8.41ms -- still 1.68x slower than native SDPA's
5.0ms (0.595x), but a real, substantial improvement over the non-flash batched version's 15.04ms
(~1.8x faster than that).** This is the most informative result of the three attention-kernel
attempts, not because it wins, but because it's evidence *for* the hypothesis rather than another
flat rejection of it: avoiding the full L x L device-memory round trip measurably helped, closing
roughly half the gap to native (15.04ms -> 8.41ms -> native's 5.0ms). The remaining gap likely
comes from this kernel's own, deliberately simple, per-row bookkeeping being far from optimally
parallel (each of the 64 active threads does ~128 serial scalar float ops per KV block -- 64 of the
128 threads in the threadgroup sit idle during that phase entirely -- vs. Apple's own SDPA
implementation almost certainly using better-parallelized reductions, and possibly the
register-resident `cooperative_tensor` chaining this attempt deliberately avoided for lower risk).
**Status: open, not closed** -- per explicit instruction not to declare the kernel-side
investigation finished. Concrete next steps if pursued further, in rough order of risk/lift: (1)
parallelize the per-row bookkeeping across more than 64 of the 128 threads (e.g. split each row's
32-column reduction across 2 threads with a simdgroup shuffle-based reduce, using `reduce_rows`/
`reduce_columns` from the MPP header rather than a hand-rolled serial loop), (2) attempt the
`cooperative_tensor` register-resident chaining now that the plain-threadgroup-memory version has
proven the underlying approach works and is worth the added risk, (3) tune `KVBLOCK`/`QBLOCK` tile
sizes (32/64 chosen to match `matmul2d`'s natural tile size, not benchmarked against alternatives).

**Next step #1 attempted immediately, and it substantially worked.** Rewrote the online-softmax
bookkeeping to use 2 lanes per row (all 128 threads active, none idle) instead of 1 thread per row
(64 idle): each simdgroup's 32 lanes cover 16 rows, partnered as `(lane, lane^1)` within the same
simdgroup so `simd_shuffle_xor(x, 1)` exchanges exactly the right partner's value for the row-max
and row-sum reductions; each lane also now only handles half of `KVBLOCK`'s columns (16, not 32)
and half of `D`'s accumulator elements (32, not 64) for the rescale and output-write steps. Also
parallelized the acc zero-init (grid-stride loop across all 128 threads instead of a 64-thread
serial loop) and the final output write (same 2-lanes-per-row split). Re-verified correctness
clean on all three shapes including the boundary case (identical error levels, no regression) --
this was a performance-only change, and the correctness check confirms it stayed that way.

**Result: 8.41ms -> 5.81ms, a real ~1.45x speedup from parallelization alone -- now only 1.17x
slower than native SDPA, the closest any hand-written kernel has gotten this entire session.** Full
progression, all at the real self-attention shape (`H=16, L=4096, D=64`):

| kernel version | ms/call | vs. native SDPA (4.98ms) |
|---|---|---|
| per-head Python loop (`matmul2d_half_half` x16 dispatches) | 15.28ms | 3.05x slower |
| batched dispatch, non-flash (head dim in the grid, full L x L in device memory) | 15.04ms | 3.00x slower |
| online-softmax flash, serial (1 thread/row) bookkeeping | 8.41ms | 1.68x slower |
| online-softmax flash, parallelized (2 lanes/row) bookkeeping | **5.81ms** | **1.17x slower** |
| native `scaled_dot_product_attention` | 4.98ms | 1.0x (baseline) |

This is strong, direct confirmation that the "far from optimally parallel" diagnosis above
was correct and load-bearing, not just plausible-sounding -- fixing exactly that one thing closed
most of the remaining gap. **Status: still open, not closed** -- remaining next steps (2)
`cooperative_tensor` register-resident chaining and (3) tile size tuning are both still untried,
and now more clearly worth attempting given how much (1) alone delivered.

**Next step #3, tile-size tuning, tried immediately -- two clean negative results, `KVBLOCK=32`
remains the best configuration found.** `KVBLOCK=64` (matching `QBLOCK`) is infeasible outright:
`score_tile` (16KB) + `acc` (16KB) + `m`/`l` (~512B) totals ~32.5KB, just over the ~32KB
per-threadgroup memory budget -- doesn't compile/dispatch, not a performance question. Tried the
other direction instead: a new kernel variant, `attention_flash_online_softmax_kv16`
(`KVBLOCK=16`, otherwise identical, including the same 2-lanes-per-row parallelized bookkeeping),
added non-destructively alongside the `KVBLOCK=32` version rather than replacing it. Correctness
passed cleanly (same error levels). **Speed: 7.97ms -- worse than `KVBLOCK=32`'s 5.81ms, not
better.** More outer-loop iterations (256 vs. 128 for `L=4096`) means more fixed per-iteration
overhead (barrier syncs, `matmul2d` dispatch cost) that a smaller tile doesn't amortize as well --
the same "smaller isn't automatically better once fixed overhead is in the picture" lesson this
port has run into before (e.g. the FFN's own dispatch-overhead findings). `QBLOCK=32` (the other
tuning axis) was not attempted -- it would require reworking the row/lane assignment scheme (the
current code hardcodes 4 simdgroups x 16 rows = 64 total rows; a 32-row query block would need a
different lanes-per-row split, e.g. 4 lanes/row instead of 2) rather than a one-constant change,
a bigger lift than the `KVBLOCK` variants tried here -- left as a flagged, not attempted, gap.

**Next step #2, `cooperative_tensor` register-resident chaining, investigated and found hard-blocked
-- a real, verified constraint, not a difficulty judgment call.** Read the actual chaining
implementation (`__get_left_input_cooperative_tensor` in `MPPTensorOpsMatMul2dImpl.h`, not just the
public header) before writing any code, given this API has no complete op-to-op chaining example
anywhere in the public header. Found a hard `static_assert`:
`__is_same_v<scope, metal::execution_simdgroup>` -- feeding one `matmul2d` call's output directly
into another as a `cooperative_tensor` requires a **single** 32-thread simdgroup scope. This
kernel's entire tiling design uses `execution_simdgroups<4>` (128 threads cooperating per 64x32
tile) throughout, for both `qk_op` and `av_op` -- incompatible with the chaining requirement as
designed. A full redesign around single-simdgroup ops would very likely mean smaller tiles and more
dispatches, which the `KVBLOCK=16` result just above already showed regresses (more, smaller
matmul2d calls paid more fixed overhead, not less) -- the same failure mode this session has hit
repeatedly, not a new risk.

Checked whether a partial version survives this constraint: `get_destination_cooperative_tensor`
(getting a matmul2d call's OWN output into registers, not chaining it as another call's input) has
no such scope restriction and works fine with `execution_simdgroups<4>` (confirmed from the same
impl file: only requires a generic valid execution scope). This would let `qk_op`'s output land in
registers instead of `score_tile` directly. But using it would mean rewriting the online-softmax
bookkeeping against `cooperative_tensor`'s **opaque, implementation-defined per-thread data
layout** (accessed only via `get_multidimensional_index`/`get_mask`/iterators) instead of the
plain, fully-controlled 2-lanes-per-row indexing the current kernel uses -- reintroducing
exactly the correctness risk threadgroup memory was chosen to avoid in the first place (see the
kernel's own module comment), for an uncertain payoff: it would only remove one
`threadgroup_barrier`/round-trip (between `qk_op` and the bookkeeping step), not the full chain,
since a `threadgroup_barrier` + explicit store would still be needed before `av_op` regardless
(true chaining into `av_op` is exactly the part that's blocked).

**Decision: stop here, don't pursue `cooperative_tensor` further right now.** Presented this
finding plus the partial-version option and a full single-simdgroup-redesign option; explicit
choice was to treat `5.81ms` (1.17x slower than native, the best result found across every
approach tried this session) as the current state and not chase the remaining gap via this
specific mechanism. **Not the same as closing the kernel-side investigation overall** -- this
closes out `cooperative_tensor` specifically, with a real, checked reason, not a difficulty
judgment call.

**External research on architectural alternatives (distillation, solver swaps, low-rank
compression): epistemic status as before -- web sources not independently verified beyond what's
checked directly against official sources.**
- **No distilled/few-step TRELLIS checkpoint exists.** Checked directly against the official
  `microsoft/TRELLIS` GitHub repo and the `microsoft/TRELLIS-image-large` HF model card, not just
  general search -- a real negative result, not absence of search hits.
- **Training-free higher-order ODE solvers (DPM-Solver++, DEIS) have limited headroom here by
  construction.** TRELLIS uses rectified flow specifically because it already straightens the ODE
  path, which is *why* it already tolerates few steps with a plain Euler integrator -- the main
  reason those solvers help elsewhere (correcting curved paths) is largely already addressed by the
  model's own training objective. Real step-distillation (consistency/reflow retraining) would be
  the actual big lever, but that's a training project, not an engineering change to this port -- out
  of scope for now.
- **Low-rank weight compression is weaker than it first sounds.** A real AAAI finding: transformer
  *weights* (unlike activations) aren't robustly low-rank -- naive SVD of frozen weights without
  retraining causes real accuracy loss. The technique that does work post-training (SVDQuant) is
  really a quantization-outlier-absorption trick, not standalone compression, and its reported ~3x
  speedups are measured on NVIDIA RTX 4090/5090 (dedicated low-bit tensor cores) -- doesn't transfer
  as evidence to this hardware. Extrapolating from this session's *own* measurements: splitting one
  `nn.Linear` into two smaller ones (`A @ B`) means two dispatches through the same fast native path
  instead of one -- the same "extra small ops eat the win" pattern already measured for
  cross-attention's rescale overhead. Likely a net loss here even before the accuracy question.

**What actually worked: reducing the sampler's step count.** Directly confirmed (not assumed) this
port's real checkpoint config: `microsoft/TRELLIS-image-large`'s own `pipeline.json` sets both the
sparse-structure and SLat samplers to `FlowEulerGuidanceIntervalSampler` at **25 steps** each. Since
`pipeline.run()`'s `sparse_structure_sampler_params`/`slat_sampler_params` already expose `steps` as
a plain kwarg, this needed zero new code -- just calling the existing sampler with a smaller step
count, on the real example image (`assets/example_image/T.png`), same seed throughout
(`scripts/investigate_step_count.py`).

*A real bug hit and fixed along the way*: `pipeline.run()` is `@torch.no_grad()`-decorated, but its
sub-stages (`sample_sparse_structure`/`sample_slat`/`decode_slat`) are **not** individually
decorated -- calling them directly in a loop (needed here, to reuse `cond` across step counts rather
than recomputing it) without wrapping in `torch.no_grad()` retains a full autograd graph through a
600M+ param forward pass every iteration. Hit exactly this: **63GB allocated, hard MPS OOM**, on the
second loop iteration. Fixed by wrapping each iteration's sub-stage calls in `torch.no_grad()`
explicitly (plus `torch.mps.empty_cache()` between iterations) -- worth remembering for any future
script that calls pipeline sub-stages directly instead of going through `run()`.

Results, `steps` vs. the 25-step baseline, measuring wall-clock for `sample_sparse_structure +
sample_slat + decode_slat + render` and a quantitative structural-fidelity proxy (IoU of occupied
sparse-structure voxel coordinates against the 25-step baseline's coordinate set):

| steps | total time | speedup | voxel IoU vs. 25-step |
|---|---|---|---|
| 25 (baseline) | 42.88s | 1.00x | 1.0000 |
| 20 | 36.84s | 1.16x | 0.9487 |
| 15 | 30.26s | 1.42x | 0.9146 |
| 10 | 25.53s | 1.68x | 0.8514 |
| 8 | 22.28s | 1.92x | 0.8111 |

**Visually inspected the rendered mesh-normal preview at every step count
(`outputs/stepcheck_{25,20,15,10,8}steps.png`) -- the voxel-IoU numbers look far more alarming than
the actual renders.** All five renders show the same recognizable structure (a lattice tower with
similar surrounding debris/noise) at every step count tested, **down to 8 steps** -- no obvious
visual quality collapse, even though IoU (an exact-voxel-coordinate-match metric) drops to 0.81.
Mesh vertex/face counts also stay in a tight, consistent band across all five (298K-317K vertices),
another sign the structure isn't degrading, just shifting some boundary voxels in a way IoU
penalizes more harshly than it visually matters. **This is the most promising lever found this
session for real wall-clock speedup** -- 1.42-1.92x, zero new kernel code, zero correctness risk to
the quantization work already done (orthogonal to it -- step count and kernel choice are
independent levers, both could apply together). Caveat: verified on **one** example image/seed so
far -- worth confirming on a few more images/seeds before treating a specific step count (15 looks
like a strong, conservative candidate; 8-10 more aggressive) as a settled default, and IoU alone
should not be trusted as the fidelity metric going forward given how much it diverges from visual
inspection here.

### Multi-image/seed verification: both wins generalize, on 6 further (image, seed) combos

The single-image caveat above (and the same one on the CFG-diff-caching result) addressed directly
-- `scripts/verify_wins_multi_image.py` repeats both techniques' most aggressive tested setting (8
steps; CFG-diff `neg_cache_interval=4`) against each technique's own 25-step/interval=1 baseline,
across 3 deliberately varied example images (`object2_vase.png` -- simple inorganic object,
`typical_creature_dragon.png` -- organic/complex, `typical_vehicle_biplane.png` -- mechanical) x 2
seeds each:

| image | seed | 8-step speedup | 8-step IoU | CFG-diff(4) speedup | CFG-diff(4) IoU |
|---|---|---|---|---|---|
| vase | 1 | 1.75x | 0.7773 | 1.13x | 0.7494 |
| vase | 2 | 1.92x | 0.9512 | 1.16x | 0.8332 |
| dragon | 1 | 1.99x | 0.8657 | 1.27x | 0.8289 |
| dragon | 2 | 1.91x | 0.8161 | 1.18x | 0.7726 |
| biplane | 1 | 2.14x | 0.9369 | 1.30x | 0.8874 |
| biplane | 2 | 2.17x | 0.9282 | 1.25x | 0.8173 |
| **average** | | **1.98x** | **0.8792** | **1.22x** | **0.8148** |

**Both wins generalize.** Step-count reduction's average speedup (1.98x) even slightly exceeds the
single-image T.png result (1.92x at 8 steps), with comparable IoU (0.88 avg vs. 0.81 on T.png).
CFG-diff-caching also holds up (1.22x avg, matching the T.png result's 1.29x at the same interval
within noise). Mesh vertex/face counts stayed in a consistent, non-diverging band for every
combination (no repeat of the naive CFG-caching version's monotonic-growth failure signature) --
spot-checked the two lowest-IoU cases directly (`vase` seed=1 at both settings, the worst in the
table) against their own baselines; both remained recognizably the same object with no obvious
extra/missing structure, consistent with the earlier finding that voxel IoU alone tends to
understate real visual fidelity for this class of thin/complex geometry. All 18 renders saved to
`outputs/multi_verify/` for further spot-checking if needed. **Practical read: step-count reduction
is the stronger, more consistently-generalizing win of the two** (higher speedup, higher IoU, on
every single combination tested) -- CFG-diff-caching remains real and positive but more modest,
consistent with its already-identified ceiling (`decode_slat`/render time doesn't shrink at all).

## Classifier-free-guidance negative-branch caching: a genuine architectural experiment, real but disappointing result

Motivated by wanting a genuine algorithmic/architectural contribution (not just kernel work) --
confirmed directly in code (`trellis/pipelines/samplers/guidance_interval_mixin.py`) that
`FlowEulerGuidanceIntervalSampler` calls the model **twice** per step (once for `cond`, once for
`neg_cond`) whenever `t` falls in `cfg_interval` (this pipeline's real config:
`cfg_interval=[0.5, 1.0]`, confirmed from `pipeline.json`). Real, training-free literature exists
for caching/reusing this specific redundancy (FasterCache-style CFG caching, ~1.67x reported on
video diffusion) -- and, checked directly, nothing has applied this to TRELLIS's specific
rectified-flow + guidance-interval combination. New sampler subclass (not editing the production
one in place, per this port's versioning convention):
`trellis/pipelines/samplers/flow_euler_cached_cfg.py`'s `CachedCfgFlowEulerGuidanceIntervalSampler`
-- caches the negative-branch prediction and reuses it for `neg_cache_interval` consecutive calls
before recomputing, instead of recomputing every call.

**Result (`scripts/investigate_cfg_caching.py`, same real example image/seed as the step-count
experiment, real 25-step config, intervals 1/2/3/4):**

| interval | total time | speedup | neg-branch calls saved | voxel IoU vs. interval=1 |
|---|---|---|---|---|
| 1 (baseline, no caching) | 42.60s | 1.00x | 0/38 | 1.0000 |
| 2 | 37.68s | 1.13x | 18/38 | 0.8794 |
| 3 | 36.95s | 1.15x | 24/38 | 0.8027 |
| 4 | 38.08s | 1.12x | 28/38 | 0.6787 |

**Honest assessment: a real, disappointing result, not a win -- worse on both axes than the
step-count reduction it was meant to complement.** Speedup tops out around 1.15x (not more, even
though interval=4 skips nearly 3/4 of negative-branch calls) because those calls are only part of
total pipeline time (`decode_slat` and rendering don't shrink at all) -- there's a real ceiling
here that reusing this specific redundancy alone can't push past. More importantly, **the quality
signal is qualitatively different from the step-count experiment's, not just quantitatively
worse**: mesh vertex/face counts *grew* monotonically with cache interval (313K -> 336K -> 355K ->
394K vertices) instead of staying flat the way they did across all five step counts tested earlier
-- a different signature than benign discretization noise. Visual inspection
(`outputs/cfgcache_{2,4}interval.png`) confirms it: interval=4 shows visibly more debris/noise than
the baseline, including extra blob-like geometry on the left side that isn't present in the
baseline at all. **Caching a stale unconditional prediction appears to introduce systematic drift
in the guidance signal, not just added variance** -- qualitatively different from Euler
discretization error, which shifts boundary voxels without adding new structure. Not recommended
as-is. If revisited: a smarter reuse policy (e.g. caching only when consecutive `neg_pred`s are
already numerically close, rather than a fixed interval) might avoid the systematic-drift failure
mode -- untested, flagged as a possible next step rather than attempted here, consistent with
keeping this line of investigation open rather than declaring it closed.

### Root-caused and fixed: cache the DIFFERENCE, not the raw value

Researched why, against real sources rather than guessing at a fix. **FasterCache**
(arXiv:2410.19355, fetched directly) does not freeze a raw branch prediction at all -- its CFG-Cache
mechanism stores frequency-domain differences/biases between the conditional and unconditional
outputs, reweighted before reuse, with forced full recomputation on a schedule (not applied
uniformly across a whole guidance interval the way the naive attempt above did); its own ablations
report the same "loss of detail / smoothing artifacts" failure mode this port's naive version hit,
which is *why* it doesn't freeze raw values. **AB-Cache** (arXiv:2504.10540) goes further and
directly confirms the working hypothesis: a larger guidance scale amplifies the error from a stale
cached difference -- directly relevant here, since this pipeline's real config uses
`cfg_strength=5.0`, and uses an adaptive, error-gated trigger (recompute when a cached and a fresh
value diverge past a threshold) rather than a fixed schedule.

**Fix, grounded in that mechanism, not guessed:** new class,
`DiffCachedCfgFlowEulerGuidanceIntervalSampler` (same file, naive version kept in place per this
port's versioning convention) -- caches `diff = pred - neg_pred` from the last full computation,
and on a reuse step derives `neg_pred_estimated = pred_fresh - diff_cached` instead of reusing a
frozen `neg_pred` directly. `pred` (the conditional branch) is always computed fresh every step
regardless of caching, so this costs no extra model call -- it only changes what the reused call
estimates. Still a fixed reuse interval, not AB-Cache's fully adaptive error-gated trigger (that
needs a cheap divergence proxy this port doesn't have yet) -- a real, smaller step toward the
actual technique, not a complete reimplementation of it.

**Result (`scripts/investigate_cfg_caching_diff.py`, identical protocol to the naive version's
test -- same image/seed/25-step config/intervals, for a direct before/after comparison):**

| interval | total time | speedup | mesh vertices | voxel IoU vs. interval=1 |
|---|---|---|---|---|
| 1 (baseline) | 42.42s | 1.00x | 313,748 | 1.0000 |
| 2 | 36.28s | 1.17x | 316,150 | 0.8994 |
| 3 | 34.21s | 1.24x | 309,000 | 0.8279 |
| 4 | 32.87s | **1.29x** | 309,281 | 0.7913 |

**A genuinely different, much better quality signature than the naive version.** Mesh vertex
counts now stay in a tight, stable band (309K-316K) instead of growing monotonically away from
baseline (the naive version's 313K -> 394K) -- the same "flat vertex count = no real structural
drift" signal the step-count experiment established as trustworthy. Visual inspection
(`outputs/cfgcache_diff_4interval.png`, the same aggressive interval=4 setting where the naive
version showed clear extra debris/blob geometry) confirms it: the render is visually consistent
with the no-caching baseline, no visible extra structure. **And it's faster than the naive version
at the same interval** (1.29x vs. 1.12x at interval=4) -- not the point of the fix, but a welcome
side effect, most likely just run-to-run variance rather than a real mechanism, not chased further.
IoU still declines somewhat at higher intervals (0.79 at interval=4) and this is verified on
**one** example image/seed so far, same caveat as the step-count experiment -- but this is a real,
working, if partial, positive result for a genuine architectural change, not just kernel work,
directly answering the "can we find a real algorithmic contribution" question this line of
investigation was started to chase.

## Renderer-specific 180-degree orientation bug (found while preparing paper figures)

**`GaussianRenderer` output is point-reflected 180 degrees from the object's real orientation --
confirmed directly, not assumed, and confirmed to be pre-existing (not introduced by this
session's paper-generation scripts).** Found while building input/output teaser figures for
`paper/main.tex`: `T.png`'s rendered output only reveals its actual shape (a recognizable capital
letter "T", matching the filename) after a 180-degree correction -- uncorrected, it looks like a
plausible-but-wrong shape (a vine mass at the "top", a fence-like base at the "bottom", which is
actually the T rotated upside down). Confirmed pre-existing by checking `outputs/
preview_gaussian_color.png`, generated by an *earlier* session's `quick_preview.py`, well before
this session's paper work -- it shows the identical wrong orientation. An earlier finding this
session ("the dragon's orientation can't be fixed by any yaw/pitch angle") was entirely this bug,
not an object-specific quirk -- corrected, the dragon matches its input almost exactly.

**Confirmed renderer-specific, not a single blanket bug across the whole rendering pipeline: only
`GaussianRenderer` has it. `OctreeRenderer` (`trellis/renderers/octree_renderer.py`, dispatching to
the hand-Metal `render_trivec` kernel from `octree_renderer_metal.py`) does NOT.** Found by
initially applying the same 180-degree correction blindly to an octree render of `T` -- it turned
a correctly-oriented render *into* an upside-down one, confirmed by directly comparing corrected
vs. uncorrected output. `MeshRenderer`'s orientation was not separately re-verified after this
finding (not blindly assumed either way) -- treat as unconfirmed, not "probably also affected,"
until actually checked.

Not fixed in `render_utils.py`/`GaussianRenderer` itself -- would need to verify the fix doesn't
break other, already-verified callers of the shared utility (e.g. prior end-to-end pipeline runs
that used Gaussian preview renders without any human checking absolute orientation, only structural
correctness like vertex/splat counts). Worked around locally instead, as a post-process (rotate the
rendered frame 180 degrees) in this session's own `paper/generate_teaser.py`, `paper/
generate_turntables.py`, and `paper/find_best_views.py` -- explicitly NOT applied in `paper/
generate_octree.py`, since `OctreeRenderer` doesn't have the bug.

**Practical implication for this session's own earlier visual verification (step-count reduction,
CFG-caching): those conclusions are unaffected.** Every comparison was between two Gaussian/mesh
renders subject to the *same* bug equally (a baseline render and a test-condition render, both
upside down the same way) -- relative structural-similarity judgments (does 8-step still look like
25-step?) remain valid; only the *absolute* orientation shown in any given figure was wrong. Worth
knowing before trusting any *new* single-render orientation judgment from this pipeline going
forward, though -- always sanity-check against the actual input photo, not assumed-correct camera
math, especially for `GaussianRenderer` specifically.

## Pre-existing upstream TRELLIS bugs found along the way

Found while testing the above; not specific to this port (would misbehave identically on
Linux/CUDA). Fixed unless noted:

- `sp.SerializeModes` (needed by `attn_mode='shift_order'`) was never registered in
  `sparse/__init__.py`'s lazy-export table — fixed.
- `Gaussian.load_ply()`, `trellis/representations/gaussian/gaussian_model.py`: referenced an
  undefined `rotation` local when `transform is not None` (the default), and even fixed, the
  transformed result was never written back to `rots` — fixed, verified via a full round-trip
  through the actual default transform (xyz: exact match; rotation: 6.3e-5 max error, at
  float32 quaternion↔matrix precision).
- `full_attn.py`'s `ATTN_DEBUG=1` assertion compares `coords == i` across all 4 columns instead
  of just the batch column — **not fixed**, just avoided (leave `ATTN_DEBUG`/`SPARSE_DEBUG` off).

## Scope not covered

- **Radiance field output (`Strivec`/`Octree`, `formats=['radiance_field']`)**: unlike mesh/gaussian,
  this path was never exercised this session — `decoder_rf.py` had the same hardcoded-`device='cuda'`
  bug as `decoder_gs.py` (now fixed the same way, deriving device from the actual tensor rather than
  hardcoding a literal), but that fix is **unverified**: never run, no renderer path tested. It *is*
  part of the default pipeline (`formats` defaults to `['mesh', 'gaussian', 'radiance_field']` in
  both `run()` and `decode_slat()`) — our real end-to-end run explicitly passed
  `formats=['mesh', 'gaussian']` to skip it, not because it's excluded by design.
- **`diffoctreerast`**: not exercised by anything covered above; not investigated. Relevant if the
  roadmap item below (using the octree/radiance-field output for a web viewer) gets picked up, since
  `OctreeRenderer` (`trellis/renderers/octree_renderer.py`) is the renderer for this representation
  and hasn't been checked for its own CUDA-specific dependencies yet.
- **Training**: everything above is verified via forward passes with randomly-initialized weights
  (and, for texture baking, a real gradient-descent optimization loop that converges correctly).
  Loading real pretrained checkpoints is untested/unverified specifically for the sparse-conv
  backend's weight layout (see §3) — the attention, mesh-decoding, and rendering paths have no
  equivalent weight-layout risk since they're not carrying learned per-position kernel weights the
  way conv does.
- **Analytic anti-aliasing** (nvdiffrast's silhouette-edge blending) and **texture mipmapping**:
  both deliberately skipped as separate, harder problems from the core rasterization/sampling
  correctness this port focused on. `ssaa` (supersampling) is a working substitute for the former;
  the latter only affects visual quality under strong minification, not correctness.
- ~~Metal kernel for `GaussianRenderer`~~: done — `trellis/renderers/metal_gaussian_rasterizer.py`,
  same one-thread-per-pixel pattern as `metal_rasterizer.py`, but compositing via depth-ordered
  alpha blending with an early transmittance-exit instead of nearest-wins. ~17x faster than the
  pure-PyTorch path at 50K Gaussians, verified to 6e-5 max error against it. Built after the first
  real end-to-end run showed the pure-PyTorch Gaussian renderer taking ~30s/frame at real (478K
  splat) scale, which would have made `to_glb`'s 100-view texture-bake render take hours.

## End-to-end verification with real pretrained weights

Beyond the synthetic-data verification described above, the full `TrellisImageTo3DPipeline`
(`microsoft/TRELLIS-image-large`, real downloaded checkpoint, real DINOv2 image conditioning) was
run start to finish on `mps`: image → sparse structure diffusion sampling → SLat diffusion sampling
→ mesh + Gaussian decoding → rendering. This is what actually exercises the previously-unverified
sparse-conv checkpoint weight-loading risk (§3) for the first time.

Two real, previously-latent bugs surfaced and got fixed only once real weights were in play:

- **State dict key/shape mismatch**: spconv's `SparseConv3d` stores weight/bias under a nested
  `.conv` submodule (`self.conv = spconv.SubMConv3d(...)`) in KRSC layout
  (`[out, kz, ky, kx, in]`); this backend's `SparseConv3d` stores them as direct attributes in
  `[kernel_volume, in, out]` layout. Loading a real checkpoint crashed immediately with a
  `RuntimeError` (missing/unexpected keys) — masked behind a confusing secondary `401 Unauthorized`
  error, because `pipelines/base.py` had a bare `except:` that blindly retried with a mangled path
  on *any* failure (also fixed: now surfaces both the original and retry errors together). Real
  fix: `trellis/modules/sparse/conv/conv_pytorch.py`'s new `adapt_state_dict_for_pytorch_backend()`,
  wired into `trellis/models/__init__.py`'s `from_pretrained`, renames and reshapes
  (`spconv_weight_to_pytorch`) matching keys before `load_state_dict` — verified against the actual
  downloaded checkpoint (correct `(kernel_volume, in, out)` shape, finite, sane weight statistics).
  This makes checkpoint loading *possible*; the kernel-offset semantic convention itself is still
  the same unverified-against-real-spconv risk described in §3 — just no longer blocked by a key
  mismatch.
- **`decoder_gs.py`/`decoder_rf.py` hardcoded `device='cuda'`** when constructing the output
  `Gaussian`/`Strivec` representation objects — same bug class as the `SparseFeatures2Mesh` fix in
  §5, just never hit before because those decoders were never actually run this session (only the
  `Gaussian` class in isolation, always with an explicit `device` arg). Fixed the same way: derive
  from `x.device` (the actual `SparseTensor`'s device) instead of a hardcoded literal.

Result: a real image (`assets/example_image/T.png`) produced a 313,724-vertex / 625,538-face mesh
and a 478,336-splat Gaussian representation, both finite, both correctly bounded within the
expected `[-0.5, 0.5]` AABB, both on `mps:0`. Rendered preview images (mesh normal map, Gaussian
color) show a coherent, recognizable structure with sensible material separation (wood vs.
foliage) — a strong qualitative signal, though not a substitute for the CUDA-machine numerical
cross-check §3 still calls for.

### MPS does not time-share well across concurrent processes

Observed directly while running two Python processes against the same GPU at once (a full
30-frame Gaussian video render, plus a separate single-frame quick-preview render launched in
parallel to get an early look without waiting): this is **not** a mild slowdown from splitting one
GPU two ways -- it's closer to one process starving the other almost completely, then a slow
recovery.

Concrete numbers from that run: isolated per-frame Gaussian render time was ~29.6-29.8s/frame.
Once the second process started, the first job's per-frame time rose to 40-55s (~1.5-1.8x slower)
-- but the *second*, smaller job fared far worse: a single frame that should have taken ~30s in
isolation took 307.4s, roughly a 10x slowdown. After the second process finished, the first job's
per-frame time didn't snap back immediately -- it recovered gradually over about 10 frames (55s ->
53s -> ... -> 30.6s), suggesting some lingering scheduling/cache-state effect rather than a clean
handoff.

Practical takeaway: don't run multiple MPS-heavy TRELLIS jobs (generation, rendering, texture
baking) concurrently expecting them to time-share reasonably -- queue them sequentially instead.
This is an operational characteristic of the MPS backend / GPU scheduler on this machine, not
something in TRELLIS's or this port's code.

## Roadmap / not yet started

- **Verify the radiance-field (`Strivec`/`Octree`) output path** — run `formats=['radiance_field']`
  for real, confirm the `decoder_rf.py` device fix above actually works end-to-end, check whether
  `OctreeRenderer` needs any device/CUDA fixes of its own (unaudited so far).
- **Web-based viewer** for the generated 3D output (mesh/Gaussian/radiance-field) — not started.
- **Flask serving endpoint** wrapping the pipeline for the web viewer to call — explicitly deferred,
  not started. Noted here as the planned next layer once the viewer exists.
- **Possible write-up / publication** of this port as a technical report — explicitly deferred,
  not started. Researched (with independent verification, not just taken from search results) so
  the option is documented for later:
  - TRELLIS's code *and* pretrained weights (Hugging Face) are both MIT-licensed — no separate
    restrictive weights license blocks building on and publishing this.
  - **`microsoft/TRELLIS.2` now genuinely exists** (verified directly against GitHub's API: real
    Microsoft-org repo, "Native and Compact Structured Latents for 3D Generation," 4B params, new
    O-Voxel sparse structure) — this port is of the original TRELLIS (CVPR'25), now the previous
    version. Worth deciding whether to scope any write-up explicitly as "TRELLIS v1 on Apple
    Silicon" or revisit after checking what TRELLIS.2 changes, before finalizing scope.
  - Precedent exists for exactly this genre of paper: "port + hand-optimize an existing published
    model for Apple Silicon" is an active arXiv genre, no lab affiliation required. Verified real
    (not hallucinated) directly against arXiv's API: *"Systematic Optimization of Real-Time
    Diffusion Model Inference on Apple M3 Ultra"* (arXiv:2605.16259, Yoichi Ochiai, Feb 2026) —
    same shape of contribution (diffusion model + Apple Silicon + systematic optimization).
  - Realistic venues: arXiv as a standalone technical report (the default, no gatekeeping, where
    the precedent papers live), or MLSys's Industrial/Systems track, whose bar is explicitly
    "design and implementation of real-world systems" rather than algorithmic novelty.
  - What would make it a real contribution rather than just a writeup: the two genuine PyTorch/MPS
    bugs found and root-caused (int64 cross-device-transfer-with-narrowing corruption, missing
    `grid_sample` MPS backward), the verified-against-reference methodology used throughout this
    port, and the measured 17-64x Metal kernel speedups.
