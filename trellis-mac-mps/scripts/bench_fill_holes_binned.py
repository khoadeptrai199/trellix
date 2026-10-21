"""
Runs the REAL postprocessing_utils._fill_holes loop (1000 views, real vase mesh, resolution=1024
-- exactly what run_real_pipeline.py's bake_texture stage does) with metal_rasterizer_binned
swapped in for utils3d's rasterizer backend, to get a genuine views/sec number comparable to the
documented brute-force baseline (~3.2 it/s, see APPLE_SILICON.md).
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import numpy as np
import torch
from trellis.renderers.nvdiffrast_shim import install as _install_nvdiffrast_shim
_install_nvdiffrast_shim()
import utils3d.torch.rasterization as u3d_rast  # noqa: E402  (explicit import -- utils3d.torch's
    # own __getattr__ is a lazy-export shim that doesn't expose submodules as plain attributes)
from trellis.renderers import metal_rasterizer_binned as binned  # noqa: E402
from trellis.utils import postprocessing_utils as pp  # noqa: E402

# Swap utils3d's rasterizer backend (bound at its own import time) from metal_rasterizer to the
# binned variant, for this process only.
u3d_rast.dr = binned

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
data = np.load("outputs/vase_decimated_mesh.npz")
verts_t = torch.tensor(data['vertices'], dtype=torch.float32).to(device)
faces_t = torch.tensor(data['faces'], dtype=torch.int32).to(device)
print(f"real mesh: {verts_t.shape[0]}v / {faces_t.shape[0]}f, device={device}")
print(f"rasterizer backend: {u3d_rast.dr.__name__}\n")

t0 = time.time()
out_v, out_f = pp._fill_holes(
    verts_t.clone(), faces_t.clone(),
    max_hole_size=0.04,
    max_hole_nbe=int(250 * np.sqrt(1 - 0.95)),
    resolution=1024,
    num_views=1000,
    verbose=True,
)
elapsed = time.time() - t0
print(f"\n_fill_holes (binned rasterizer, 1000 views @ 1024x1024): {elapsed:.1f}s "
      f"({1000/elapsed:.2f} it/s)")
print(f"faces kept: {out_f.shape[0]} / {faces_t.shape[0]}")
print("\n(brute-force baseline from the real pipeline run: ~300-320s for 1000 views @ ~3.2 it/s, "
      "34428/34603 faces kept)")
