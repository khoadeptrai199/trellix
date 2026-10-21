"""
Builds paper/project_page.html: a self-contained project page (all assets embedded as data URIs,
per the artifact sandbox's CSP -- no external requests) for the TRELLIS-on-Apple-Silicon kernel
characterization + CFG-caching work. Run after generate_teaser.py / generate_turntables.py /
generate_octree.py / generate_figures.py / main.tex have all been (re)built, so it picks up
current assets rather than stale ones.
"""
import base64
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def b64(path, mime):
    with open(os.path.join(ROOT, path), 'rb') as f:
        data = base64.b64encode(f.read()).decode('ascii')
    return f"data:{mime};base64,{data}"


ASSETS = {
    'teaser': b64('figures/fig0_teaser_input_output.png', 'image/png'),
    'chart_progression': b64('figures/fig2_kernel_progression.png', 'image/png'),
    'chart_combos': b64('figures/fig1_matmul2d_combos.png', 'image/png'),
    'chart_steps': b64('figures/fig3_step_count_tradeoff.png', 'image/png'),
    'chart_cfg': b64('figures/fig4_cfg_caching_drift.png', 'image/png'),
    'octree_still': b64('figures/fig7_octree_still.png', 'image/png'),
    'video_T': b64('videos/T.mp4', 'video/mp4'),
    'video_vase': b64('videos/vase.mp4', 'video/mp4'),
    'video_dragon': b64('videos/dragon.mp4', 'video/mp4'),
    'video_biplane': b64('videos/biplane.mp4', 'video/mp4'),
    'video_octree': b64('videos/T_octree.mp4', 'video/mp4'),
    'paper_pdf': b64('main.pdf', 'application/pdf'),
}

print("asset sizes (base64 chars):")
for k, v in ASSETS.items():
    print(f"  {k}: {len(v)//1024}KB")

TEMPLATE_PATH = os.path.join(ROOT, 'project_page_template.html')
with open(TEMPLATE_PATH, 'r') as f:
    html = f.read()

for key, data_uri in ASSETS.items():
    html = html.replace(f"{{{{{key}}}}}", data_uri)

OUT_PATH = os.path.join(ROOT, 'project_page.html')
with open(OUT_PATH, 'w') as f:
    f.write(html)

print(f"\nwrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
