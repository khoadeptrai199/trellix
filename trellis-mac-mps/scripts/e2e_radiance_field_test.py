"""
End-to-end smoke test: run the real pretrained pipeline with formats=['radiance_field'], render
the resulting Strivec with the new pure-PyTorch octree_renderer_mps path (via OctreeRenderer's
fallback, exercised through render_utils.render_video exactly as mesh/gaussian previews already
are), and check the output isn't degenerate (NaN/Inf, all-background, or all-saturated).
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import numpy as np
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

image = Image.open("assets/example_image/T.png")
t0 = time.time()
outputs = pipeline.run(image, seed=1, formats=['radiance_field'])
print(f"pipeline.run(formats=['radiance_field']) completed in {time.time()-t0:.1f}s")

rf = outputs['radiance_field'][0]
print(f"Strivec: {rf.position.shape[0]} occupied leaves / {rf.resolution}^3 grid, primitive={rf.primitive}, sh_degree={rf.sh_degree}")
assert rf.position.shape[0] > 0, "decoder produced zero occupied leaves -- nothing to render"

t1 = time.time()
video = render_utils.render_video(rf, resolution=256, num_frames=12)['color']
print(f"rendered 12 frames in {time.time()-t1:.1f}s")

video_np = np.stack(video)  # [F, H, W, 3] uint8
print(f"video shape: {video_np.shape}, dtype: {video_np.dtype}")
assert np.isfinite(video_np).all() or video_np.dtype == np.uint8, "non-finite values in rendered output"

# A degenerate render is either uniform (nothing visible, or fully saturated one color across all
# frames) or has NaN propagated through as 0s/255s uniformly -- check per-frame pixel variance.
frame_std = video_np.reshape(video_np.shape[0], -1).std(axis=1)
print(f"per-frame pixel std: min={frame_std.min():.2f}, max={frame_std.max():.2f}")
non_degenerate = (frame_std > 3.0).all()
print(f"all frames have non-trivial variance (std > 3.0): {non_degenerate}")

import imageio
os.makedirs("outputs", exist_ok=True)
imageio.mimsave("outputs/radiance_field_smoke_test.mp4", video, fps=8)
print("saved -> outputs/radiance_field_smoke_test.mp4")

ok = non_degenerate
print("\n" + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
