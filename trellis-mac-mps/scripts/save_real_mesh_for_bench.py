"""
One-off: decode + decimate the real vase mesh (same seed/settings as run_real_pipeline.py /
bench_fill_holes_resolution.py) and save vertices/faces to outputs/vase_decimated_mesh.npz, so
later benchmark scripts (e.g. profiling the binned rasterizer) don't have to re-pay pipeline
load + diffusion sampling every time -- they can just load this file.
"""
import os, sys
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import numpy as np
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
import pyvista as pv

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

image = Image.open("assets/example_image/object2_vase.png")
outputs = pipeline.run(image, seed=1, formats=['mesh'])
mesh = outputs['mesh'][0]

vertices_np = mesh.vertices.cpu().numpy()
faces_np = mesh.faces.cpu().numpy()
pvmesh = pv.PolyData(vertices_np, np.concatenate([np.full((faces_np.shape[0], 1), 3), faces_np], axis=1))
pvmesh = pvmesh.decimate(0.95)
vertices_np, faces_np = pvmesh.points, pvmesh.faces.reshape(-1, 4)[:, 1:]

np.savez("outputs/vase_decimated_mesh.npz", vertices=vertices_np.astype(np.float32), faces=faces_np.astype(np.int32))
print(f"saved: {vertices_np.shape[0]}v / {faces_np.shape[0]}f -> outputs/vase_decimated_mesh.npz")
