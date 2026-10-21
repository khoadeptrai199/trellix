"""
Exports downsampled (6000-point) colored point clouds from each example object's Gaussian
representation, base64-encoded, for paper/viewer/index.html's interactive view-angle finder --
lets a human manually orbit a lightweight proxy of the real output and read off the exact
(yaw, pitch) that matches an input photo's framing, instead of guessing from a discrete 8-way
turntable scan (paper/find_best_views.py's approach). Color uses the Gaussian's 0th-order
spherical-harmonics coefficient (the standard 3DGS DC-term-to-RGB conversion), a reasonable flat-
lighting proxy -- not the full view-dependent color model, which isn't needed just to recognize
silhouette/color layout for angle-matching.

Run from the repo root: python paper/viewer/export_pointclouds.py
Writes /tmp/pointclouds.json; paste into paper/viewer/index.html's POINTCLOUDS constant by hand
(kept as a separate step rather than writing the HTML directly, so re-running this never
clobbers hand-edits to the viewer's UI code).
"""
import os, sys, json, base64, struct
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
import numpy as np
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

OBJECTS = {
    'T': 'assets/example_image/T.png',
    'vase': 'assets/example_image/object2_vase.png',
    'dragon': 'assets/example_image/typical_creature_dragon.png',
    'biplane': 'assets/example_image/typical_vehicle_biplane.png',
}

N_POINTS = 6000
out = {}
for name, path in OBJECTS.items():
    print(f"processing {name}...", flush=True)
    image = Image.open(path)
    with torch.no_grad():
        result = pipeline.run(image, seed=1, formats=['gaussian'])
    g = result['gaussian'][0]
    xyz = g.get_xyz.detach().cpu().numpy()
    if hasattr(g, 'get_features'):
        feat = g.get_features.detach().cpu().numpy()
        C0 = 0.28209479177387814  # standard 3DGS SH-DC-term-to-RGB constant
        rgb = np.clip(0.5 + C0 * feat[:, 0, :3], 0, 1)
    else:
        rgb = np.ones((xyz.shape[0], 3)) * 0.6

    n = xyz.shape[0]
    if n > N_POINTS:
        idx = np.random.RandomState(0).choice(n, N_POINTS, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    # Center + unit-normalize -- matches the world-space scale the yaw/pitch camera convention
    # (r=2-2.4 in the viewer) assumes.
    center = xyz.mean(axis=0)
    xyz = xyz - center
    scale = np.abs(xyz).max()
    xyz = xyz / scale

    buf = bytearray()
    for i in range(xyz.shape[0]):
        buf += struct.pack('<fff', float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
        buf += struct.pack('<BBB', int(rgb[i, 0] * 255), int(rgb[i, 1] * 255), int(rgb[i, 2] * 255))
    b64 = base64.b64encode(bytes(buf)).decode('ascii')
    out[name] = {'n': xyz.shape[0], 'data': b64}
    print(f"  {xyz.shape[0]} points, {len(b64)} base64 chars", flush=True)
    torch.mps.empty_cache()

with open('/tmp/pointclouds.json', 'w') as f:
    json.dump(out, f)
print("wrote /tmp/pointclouds.json")
