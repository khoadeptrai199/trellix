"""
Generates an octree/radiance-field turntable render for the project page (a representation
format this session hadn't produced yet, alongside mesh and Gaussian). Uses OctreeRenderer via
render_utils' existing Octree-representation dispatch (trellis/renderers/octree_renderer_mps.py).
Deliberately slow (~18s/frame, documented in APPLE_SILICON.md) -- kept to a modest frame count and
one object (T, this project's clearest orientation test case) rather than all 4, given the cost.

IMPORTANT, confirmed directly (not assumed): the 180-degree render-orientation bug documented in
find_best_views.py's module docstring is specific to GaussianRenderer -- OctreeRenderer does NOT
have it. Applying the same rotation fix here was tried first and produced an upside-down T where
the uncorrected render was already correct -- confirmed by comparing both directly. So: no
orientation correction applied in this script, unlike its Gaussian-renderer siblings.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
out_dir = "/Users/vinayak/Downloads/TRELLIS-main/paper/videos"
os.makedirs(out_dir, exist_ok=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

image = Image.open("assets/example_image/T.png")
N_FRAMES = 12
RESOLUTION = 384

print("sampling + decoding radiance_field ...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = pipeline.run(image, seed=1, formats=['radiance_field'])
rf = out['radiance_field'][0]
print(f"  done in {time.time()-t0:.1f}s -- {rf.position.shape[0]} occupied leaves / {rf.resolution}^3 grid", flush=True)

print(f"rendering {N_FRAMES}-frame octree turntable (~18s/frame expected) ...", flush=True)
t1 = time.time()
with torch.no_grad():
    video = render_utils.render_video(rf, resolution=RESOLUTION, num_frames=N_FRAMES)['color']
# No orientation correction here -- confirmed directly that OctreeRenderer does not have the
# 180-degree bug GaussianRenderer has (see module docstring above).
imageio.mimsave(f"{out_dir}/T_octree.mp4", video, fps=6)
imageio.mimsave(f"{out_dir}/T_octree.gif", video, fps=6)
print(f"  rendered in {time.time()-t1:.1f}s -> {out_dir}/T_octree.mp4 (+.gif)", flush=True)

# Also a single still frame for the paper/page (front view, matching T's picked yaw=0).
yaw = np.deg2rad(0)
pitch = np.deg2rad(15)
with torch.no_grad():
    frame = render_utils.render_frames(
        rf,
        *render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], 2, 40, device=rf.position.device),
        options={'resolution': 512},
    )
still = frame['color'][0].astype(np.uint8)
Image.fromarray(still).save(f"{out_dir}/../figures/fig7_octree_still.png")
print("wrote fig7_octree_still.png")
print("done")
