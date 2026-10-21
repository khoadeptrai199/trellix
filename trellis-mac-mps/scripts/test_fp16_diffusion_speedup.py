"""
Tests whether converting the two large diffusion denoisers (sparse_structure_flow_model, 555.5M
params; slat_flow_model, 600.4M params -- both loaded as float32 by default despite their
checkpoint filenames ending in "_fp16", e.g. ss_flow_img_dit_L_16l8_fp16) to float16 via their
existing convert_to_fp16() method actually speeds up sampling on MPS.

Result (see APPLE_SILICON.md): fp16 runs correctly (no crash, valid output) but is NOT faster --
42.3s (fp32) vs 42.1s (fp16), within noise. Apple Silicon GPUs don't have the same fp16-vs-fp32
throughput gap CUDA GPUs get from dedicated tensor cores; the 25-step sampling loop is more likely
bottlenecked by per-step dispatch overhead (many small kernel launches) than raw matmul compute --
the same class of problem already found and fixed in this port's *rendering* kernels, but fixing it
in the sampling loop would mean restructuring that loop itself, a different and larger undertaking.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

image = Image.open("assets/example_image/T.png")

t0 = time.time()
out_fp32 = pipeline.run(image, seed=1, formats=['mesh'])
t_fp32 = time.time() - t0
print(f"fp32 sampling: {t_fp32:.1f}s")

pipeline.models['sparse_structure_flow_model'].convert_to_fp16()
pipeline.models['slat_flow_model'].convert_to_fp16()
print("convert_to_fp16() succeeded on both flow models")

t0 = time.time()
out_fp16 = pipeline.run(image, seed=1, formats=['mesh'])
t_fp16 = time.time() - t0
print(f"fp16 sampling: {t_fp16:.1f}s")
print(f"fp16 mesh success: {out_fp16['mesh'][0].success}, verts: {out_fp16['mesh'][0].vertices.shape[0]}")

speedup = t_fp32 / t_fp16
print(f"\nspeedup: {speedup:.2f}x")
print("CONCLUSION: fp16 is not meaningfully faster than fp32 for diffusion sampling on this MPS "
      "setup" if speedup < 1.15 else "CONCLUSION: fp16 gave a real speedup, worth pursuing further")
