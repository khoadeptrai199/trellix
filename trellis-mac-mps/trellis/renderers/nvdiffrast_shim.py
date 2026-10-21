"""
Registers our pure-PyTorch rasterizer (metal_rasterizer_binned if available, else metal_rasterizer,
else sw_rasterizer) as `nvdiffrast`/`nvdiffrast.torch` in sys.modules, if the real nvdiffrast isn't
installed.

Why: `utils3d.torch.RastContext` is -- by its own docstring -- "nothing but a wrapper of
nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext", and
`rasterize_triangle_faces` purely delegates to `dr.rasterize`/`interpolate`/`antialias`/`texture`.
So rather than reimplementing a second/third rasterizer for utils3d specifically, installing this
shim before `import utils3d` (or before `import nvdiffrast.torch` anywhere else, e.g. in
postprocessing_utils.py itself) makes utils3d's rasterization transparently use ours instead.

metal_rasterizer_binned bins triangles into screen-space tiles before rasterizing (see its module
docstring), giving bit-identical output to metal_rasterizer at real workload scale (postprocessing_
utils._fill_holes: ~35K faces, 1024x1024) but ~14-15x faster end to end (verified via
scripts/verify_metal_rasterizer_binned.py and scripts/bench_fill_holes_binned.py) -- the O(H*W*F)
brute-force kernel's cost dwarfed everything else at that resolution/triangle count (see
APPLE_SILICON.md), and binning is the standard fix for exactly that bottleneck.

Call `install()` explicitly, early, in any module that imports `utils3d` (for its torch
rasterization functions) or `nvdiffrast` directly -- importing this shim module has no side
effects by itself; only `install()` mutates sys.modules, and only if the real nvdiffrast can't be
imported.
"""
import sys
import types

_installed = False


def install():
    global _installed
    if _installed:
        return
    _installed = True

    try:
        import nvdiffrast.torch  # noqa: F401
        return  # real nvdiffrast is available -- nothing to do, don't shadow it
    except ImportError:
        pass

    try:
        from . import metal_rasterizer_binned as impl
    except ImportError:
        try:
            from . import metal_rasterizer as impl
        except ImportError:
            from . import sw_rasterizer as impl

    fake_nvdiffrast = types.ModuleType('nvdiffrast')
    fake_nvdiffrast.torch = impl
    sys.modules.setdefault('nvdiffrast', fake_nvdiffrast)
    sys.modules.setdefault('nvdiffrast.torch', impl)
