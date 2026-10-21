"""
Generates a proper input-2D-image -> rendered-3D-output teaser figure for the paper, using actual
COLOR renders (gaussian splat color channel, not the normal-map previews used elsewhere in this
port for fast structural checks) -- a reader needs to see what the model actually produces, not
just a normal-encoded structural diagnostic.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
out_dir = "/Users/vinayak/Downloads/TRELLIS-main/paper/figures"
os.makedirs(out_dir, exist_ok=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

OBJECTS = {
    'T': ('assets/example_image/T.png', 0),
    'vase': ('assets/example_image/object2_vase.png', 180),
    'dragon': ('assets/example_image/typical_creature_dragon.png', 180),
    'biplane': ('assets/example_image/typical_vehicle_biplane.png', 225),
}
# Per-object yaw chosen by visually inspecting an 8-angle turntable scan against each input photo
# (paper/find_best_views.py) -- TRELLIS's generated assets don't share one consistent canonical
# orientation across objects, so no single fixed camera angle matches every input's framing;
# confirmed empirically per-object rather than assumed.
PITCH_DEG = 15

tiles = []
for name, (path, yaw_deg) in OBJECTS.items():
    print(f"generating {name} (yaw={yaw_deg}) ...", flush=True)
    image = Image.open(path)
    t0 = time.time()
    with torch.no_grad():
        out = pipeline.run(image, seed=1, formats=['gaussian'])
    gaussian = out['gaussian'][0]
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(PITCH_DEG)
    frame = render_utils.render_frames(
        gaussian,
        *render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], 2, 40, device=gaussian.get_xyz.device),
        options={'resolution': 512},
    )
    # Confirmed bug (see find_best_views.py's module docstring for the full story): this render
    # pipeline's output comes out rotated 180 degrees from the object's real orientation, for
    # every object -- not fixed at the source (render_utils.py), corrected here as a local
    # post-process instead.
    corrected = np.rot90(frame['color'][0].astype(np.uint8), 2).copy()
    color_img = Image.fromarray(corrected)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    input_img = image.convert('RGB').resize((512, 512))
    tiles.append((name, input_img, color_img))
    torch.mps.empty_cache()

# Build a 2-row x N-col grid: row 1 = input image, row 2 = rendered 3D output, labeled per column.
tile_w, tile_h = 512, 512
label_h = 30
n = len(tiles)
grid = Image.new('RGB', (n * tile_w, 2 * tile_h + 2 * label_h), '#ffffff')
draw = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
except Exception:
    font = ImageFont.load_default()

for i, (name, input_img, color_img) in enumerate(tiles):
    x = i * tile_w
    draw.text((x + 10, 5), f"input: {name}", fill='#111111', font=font)
    grid.paste(input_img, (x, label_h))
    draw.text((x + 10, label_h + tile_h + 5), "TRELLIS output (rendered)", fill='#111111', font=font)
    grid.paste(color_img, (x, 2 * label_h + tile_h))

grid.save(os.path.join(out_dir, 'fig0_teaser_input_output.png'))
print(f"wrote fig0_teaser_input_output.png")
