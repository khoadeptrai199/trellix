"""
Generates turntable .mp4/.gif preview videos (Gaussian-splat color render) for the project page,
for the same 4 example objects used in the paper's teaser figure. Lighter-weight than
scripts/run_real_pipeline.py: skips mesh texture-baking/.glb export (not needed for a video
preview), reusing render_utils.render_video directly.
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

OBJECTS = {
    'T': 'assets/example_image/T.png',
    'vase': 'assets/example_image/object2_vase.png',
    'dragon': 'assets/example_image/typical_creature_dragon.png',
    'biplane': 'assets/example_image/typical_vehicle_biplane.png',
}

RESOLUTION = 512
N_FRAMES = 60

for name, path in OBJECTS.items():
    print(f"generating {name} ...", flush=True)
    image = Image.open(path)
    t0 = time.time()
    with torch.no_grad():
        out = pipeline.run(image, seed=1, formats=['gaussian'])
    gaussian = out['gaussian'][0]
    print(f"  sampled in {time.time()-t0:.1f}s, {gaussian.get_xyz.shape[0]} splats", flush=True)

    t1 = time.time()
    video = render_utils.render_video(gaussian, resolution=RESOLUTION, num_frames=N_FRAMES)['color']
    # Confirmed 180-degree render-orientation bug (see find_best_views.py) -- corrected per frame.
    video = [np.rot90(f, 2).copy() for f in video]
    imageio.mimsave(f"{out_dir}/{name}.mp4", video, fps=20)
    imageio.mimsave(f"{out_dir}/{name}.gif", video, fps=20)
    print(f"  turntable rendered in {time.time()-t1:.1f}s -> {out_dir}/{name}.mp4 (+.gif)", flush=True)
    torch.mps.empty_cache()

print("done")
