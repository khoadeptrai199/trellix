import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
import numpy as np
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}", flush=True)

out_dir = "/Users/vinayak/Downloads/TRELLIS-main/outputs"
os.makedirs(out_dir, exist_ok=True)

t0 = time.time()
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
print(f"pipeline loaded (cached) in {time.time()-t0:.1f}s", flush=True)

image = Image.open("assets/example_image/T.png")

t1 = time.time()
outputs = pipeline.run(image, seed=1, formats=['mesh', 'gaussian'])
print(f"generation done in {time.time()-t1:.1f}s", flush=True)

mesh = outputs['mesh'][0]
gaussian = outputs['gaussian'][0]
print(f"mesh: {mesh.vertices.shape[0]}v/{mesh.faces.shape[0]}f, gaussian: {gaussian.get_xyz.shape[0]} splats", flush=True)

# quick single-view mesh normal preview (fast: uses the Metal-kernel MeshRenderer)
t2 = time.time()
frame = render_utils.render_frames(
    mesh,
    *render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.8], [0.4], 2, 40, device=mesh.vertices.device),
    options={'resolution': 512},
)
normal_img = (frame['normal'][0]).astype(np.uint8)
Image.fromarray(normal_img).save(f"{out_dir}/preview_mesh_normal.png")
print(f"mesh normal preview saved in {time.time()-t2:.1f}s -> {out_dir}/preview_mesh_normal.png", flush=True)

# quick single-view gaussian color preview (slow-ish per frame, but just ONE frame)
t3 = time.time()
frame2 = render_utils.render_frames(
    gaussian,
    *render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.8], [0.4], 2, 40, device=gaussian.get_xyz.device),
    options={'resolution': 512},
)
color_img = (frame2['color'][0]).astype(np.uint8)
Image.fromarray(color_img).save(f"{out_dir}/preview_gaussian_color.png")
print(f"gaussian color preview saved in {time.time()-t3:.1f}s -> {out_dir}/preview_gaussian_color.png", flush=True)

print(f"TOTAL: {time.time()-t0:.1f}s", flush=True)
print("DONE", flush=True)
