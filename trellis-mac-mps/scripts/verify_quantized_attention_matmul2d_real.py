"""
Real-model finite-output check for quantized_attention_matmul2d.py, step 2 of this project's
verification methodology (APPLE_SILICON.md): synthetic checks are necessary but not sufficient (a
past FFN kernel bug passed every synthetic test and only produced NaN on the real pretrained model
with real conditioning). Loads the real sparse_structure_flow_model, quantizes every
MultiHeadAttention, and runs multiple forward passes with real image conditioning (not
torch.randn -- out-of-distribution conditioning is not a reliable stability proxy) and varied
random x/t each trial.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules.quantized_attention_matmul2d import quantize_model_attention_matmul2d, available

print(f"matmul2d available: {available()}\n")

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
model = pipeline.models['sparse_structure_flow_model']
model.eval()

image = Image.open("assets/example_image/T.png")
image_p = pipeline.preprocess_image(image)
cond = pipeline.get_cond([image_p])['cond']
print(f"real cond shape: {cond.shape}\n")

n_quantized = quantize_model_attention_matmul2d(model)
print(f"quantized {n_quantized} MultiHeadAttention modules\n")

B = 1
all_finite = True
with torch.no_grad():
    for trial in range(5):
        torch.manual_seed(trial)
        x = torch.randn(B, 8, 16, 16, 16, device=device)
        t = torch.full((B,), float(torch.randint(1, 1000, (1,)).item()), device=device)
        out = model(x, t, cond)
        finite = torch.isfinite(out).all().item()
        all_finite = all_finite and finite
        print(f"trial {trial}: t={t.item():.1f}, output shape={tuple(out.shape)}, "
              f"finite={finite}, max abs={out.abs().max().item():.2f}")

print()
print("=== " + ("ALL TRIALS FINITE" if all_finite else "NON-FINITE OUTPUT DETECTED") + " ===")
sys.exit(0 if all_finite else 1)
