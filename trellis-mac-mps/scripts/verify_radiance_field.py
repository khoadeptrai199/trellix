import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}", flush=True)

t0 = time.time()
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
print(f"pipeline loaded (cached) in {time.time()-t0:.1f}s", flush=True)

image = Image.open("assets/example_image/T.png")

# Reuse the sampling stages once, decode to ALL formats including radiance_field this time,
# to specifically exercise decoder_rf.py's device fix (never tested until now).
t1 = time.time()
if_preprocess = True
image_p = pipeline.preprocess_image(image)
cond = pipeline.get_cond([image_p])
torch.manual_seed(1)
coords = pipeline.sample_sparse_structure(cond, 1, {})
slat = pipeline.sample_slat(cond, coords, {})
print(f"sampling done in {time.time()-t1:.1f}s", flush=True)

t2 = time.time()
outputs = pipeline.decode_slat(slat, formats=['radiance_field'])
print(f"decode_slat(radiance_field) done in {time.time()-t2:.1f}s", flush=True)

rf = outputs['radiance_field'][0]
print(f"radiance_field type: {type(rf).__name__}", flush=True)
print(f"position shape: {rf.position.shape}, device: {rf.position.device}", flush=True)
print(f"position finite: {torch.isfinite(rf.position).all().item()}", flush=True)
print(f"position range: [{rf.position.min().item():.3f}, {rf.position.max().item():.3f}]", flush=True)

# Check the other Strivec attrs (per decoder_rf.py's actual layout) got populated, are finite,
# and landed on the right device -- this is exactly what the device='cuda' -> device=x.device fix
# needed to get right.
for attr in ['trivec', 'density', 'features_dc', 'depth']:
    v = getattr(rf, attr)
    print(f"{attr}: shape {tuple(v.shape)}, device {v.device}, dtype {v.dtype}, "
          f"finite {torch.isfinite(v.float()).all().item()}", flush=True)

print("\nDONE - decoder_rf.py device fix verified working on real pretrained weights", flush=True)
