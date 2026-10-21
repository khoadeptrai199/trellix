"""
Generates all figures for paper/main.tex from this session's real, measured data (see
APPLE_SILICON.md for the full record each number is drawn from). Run from the repo root:
    python paper/generate_figures.py
Outputs PDF (vector) figures to paper/figures/, matplotlib only (no extra deps).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#111111',
    'text.color': '#111111',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.grid': True,
    'grid.color': '#dddddd',
    'grid.linewidth': 0.6,
    'axes.axisbelow': True,
})

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

ACCENT = '#2a6fbb'
ACCENT2 = '#c0392b'
GOOD = '#1a8a4a'
GRAY = '#888888'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {name}")


# ---------- Figure 1: matmul2d operand-combination sweep ----------
combos = [
    'half×int8_t→float\n(production)',
    'half×half→float',
    'half×half→half',
    'bfloat×bfloat→float',
    'bfloat×int8_t→float',
    'int8_t×int8_t→int32\n(full W8A8)',
]
ms = [0.7992, 0.7952, 0.8272, 0.8283, 0.8478, 0.4086]
native_ms = 0.4123

fig, ax = plt.subplots(figsize=(6.4, 3.4))
colors = [GRAY] * len(combos)
colors[-1] = GOOD
y_pos = range(len(combos))
ax.barh(y_pos, ms, color=colors, height=0.6)
ax.axvline(native_ms, color=ACCENT2, linestyle='--', linewidth=1.3)
ax.text(native_ms + 0.01, len(combos) - 0.3, f'native fp16 Linear\n({native_ms:.3f} ms)',
        color=ACCENT2, fontsize=8, va='top')
ax.set_yticks(list(y_pos))
ax.set_yticklabels(combos, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('ms / call (real to\\_qkv shape: 4096×1024×3072)')
ax.set_title('matmul2d operand-type combinations vs. native fp16', fontsize=10)
for i, v in enumerate(ms):
    ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=8)
save(fig, 'fig1_matmul2d_combos.pdf')

# ---------- Figure 2: attention kernel progression ----------
labels = ['per-head\nPython loop', 'batched\ndispatch', 'flash\n(serial)', 'flash\n(parallel)', 'native\nSDPA']
vals = [15.28, 15.04, 8.41, 5.81, 4.98]
colors2 = [GRAY, GRAY, GRAY, ACCENT, ACCENT2]

fig, ax = plt.subplots(figsize=(6.0, 3.2))
bars = ax.bar(labels, vals, color=colors2, width=0.55)
ax.set_ylabel('ms / call (H=16, L=4096, D=64)')
ax.set_title('Attention kernel progression toward native', fontsize=10)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f'{v:.2f}', ha='center', fontsize=8)
save(fig, 'fig2_kernel_progression.pdf')

# ---------- Figure 3: step-count speedup vs. quality tradeoff ----------
steps = [25, 20, 15, 10, 8]
speedup = [1.00, 1.16, 1.42, 1.68, 1.92]
iou = [1.00, 0.9487, 0.9146, 0.8514, 0.8111]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax1.plot(steps, speedup, marker='o', color=ACCENT, linewidth=1.8)
ax1.set_xlabel('sampling steps')
ax1.set_ylabel('speedup vs. 25-step baseline')
ax1.set_title('(a) Speedup', fontsize=10)
ax1.invert_xaxis()

ax2.plot(steps, iou, marker='o', color=GOOD, linewidth=1.8)
ax2.set_xlabel('sampling steps')
ax2.set_ylabel('voxel IoU vs. 25-step baseline')
ax2.set_title('(b) Structural fidelity (IoU)', fontsize=10)
ax2.invert_xaxis()
ax2.set_ylim(0.75, 1.02)
save(fig, 'fig3_step_count_tradeoff.pdf')

# ---------- Figure 4: CFG-caching naive vs. diff-based (vertex-count drift) ----------
intervals = [1, 2, 3, 4]
naive_verts = [313772, 336719, 354917, 394129]
diff_verts = [313748, 316150, 309000, 309281]

fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.plot(intervals, naive_verts, marker='o', color=ACCENT2, linewidth=1.8, label='naive (freeze raw value)')
ax.plot(intervals, diff_verts, marker='s', color=GOOD, linewidth=1.8, label='fixed (cache difference)')
ax.axhline(naive_verts[0], color=GRAY, linestyle=':', linewidth=1.0, label='baseline (no caching)')
ax.set_xlabel('negative-branch cache interval')
ax.set_ylabel('mesh vertex count')
ax.set_title('CFG-caching: raw-value freezing drifts, difference-caching does not', fontsize=10)
ax.set_xticks(intervals)
ax.legend(fontsize=8, frameon=False)
save(fig, 'fig4_cfg_caching_drift.pdf')

# ---------- Figure 5: qualitative results grid, actual rendered outputs ----------
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(REPO, 'outputs')


def load_labeled(path, label):
    img = Image.open(path).convert('RGB')
    # Confirmed bug (see APPLE_SILICON.md's "Renderer-specific 180-degree orientation bug"):
    # MeshRenderer's normal-map previews (what stepcheck_*.png/cfgcache_*.png/multi_verify/*.png
    # all are) come out point-reflected 180 degrees from the object's real orientation, same as
    # GaussianRenderer -- confirmed directly by rotating T's render and getting a clean "T" back.
    # Corrected here, since these are pre-existing renders from earlier investigation scripts that
    # predate the fix and aren't worth regenerating from scratch just for this.
    img = img.rotate(180)
    W, H = img.size
    strip_h = 34
    canvas = Image.new('RGB', (W, H + strip_h), '#ffffff')
    canvas.paste(img, (0, strip_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) / 2, 6), label, fill='#111111', font=font)
    return canvas


def make_grid(rows, out_name, row_labels=None):
    """rows: list of lists of (path, label). Builds a grid, one row per sublist."""
    tiles = [[load_labeled(p, lbl) for p, lbl in row] for row in rows]
    tile_w, tile_h = tiles[0][0].size
    n_cols = max(len(r) for r in tiles)
    row_label_w = 90 if row_labels else 0
    grid = Image.new('RGB', (row_label_w + n_cols * tile_w, len(tiles) * tile_h), '#ffffff')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except Exception:
        font = ImageFont.load_default()
    for r, row in enumerate(tiles):
        if row_labels:
            draw.text((8, r * tile_h + tile_h // 2 - 10), row_labels[r], fill='#111111', font=font)
        for c, tile in enumerate(row):
            grid.paste(tile, (row_label_w + c * tile_w, r * tile_h))
    grid.save(os.path.join(OUT, out_name))
    print(f"wrote {out_name}")


# Row 1: step-count qualitative comparison (T.png object), Row 2: CFG-caching (naive vs. fixed)
step_row = [
    (os.path.join(OUTPUTS, 'stepcheck_25steps.png'), '25 steps (baseline)'),
    (os.path.join(OUTPUTS, 'stepcheck_15steps.png'), '15 steps (1.42x)'),
    (os.path.join(OUTPUTS, 'stepcheck_10steps.png'), '10 steps (1.68x)'),
    (os.path.join(OUTPUTS, 'stepcheck_8steps.png'), '8 steps (1.92x)'),
]
cfg_row = [
    (os.path.join(OUTPUTS, 'cfgcache_1interval.png'), 'baseline (no caching)'),
    (os.path.join(OUTPUTS, 'cfgcache_4interval.png'), 'naive freeze, interval=4'),
    (os.path.join(OUTPUTS, 'cfgcache_diff_4interval.png'), 'diff-cache (ours), interval=4'),
]
if all(os.path.exists(p) for p, _ in step_row) and all(os.path.exists(p) for p, _ in cfg_row):
    make_grid([step_row], 'fig5a_step_count_qualitative.png')
    make_grid([cfg_row], 'fig5b_cfg_caching_qualitative.png')
else:
    print("WARNING: some source renders missing, skipping qualitative figure")

# Multi-image generalization grid: baseline vs. 8-step vs. CFG-diff(4), 3 objects
MV = os.path.join(OUTPUTS, 'multi_verify')
gen_rows, gen_labels = [], []
for obj in ['vase', 'dragon', 'biplane']:
    row = [
        (os.path.join(MV, f'{obj}_seed1_baseline25.png'), 'baseline'),
        (os.path.join(MV, f'{obj}_seed1_steps8.png'), '8 steps'),
        (os.path.join(MV, f'{obj}_seed1_cfgdiff4.png'), 'CFG-diff(4)'),
    ]
    if all(os.path.exists(p) for p, _ in row):
        gen_rows.append(row)
        gen_labels.append(obj)
if gen_rows:
    make_grid(gen_rows, 'fig6_generalization_qualitative.png', row_labels=gen_labels)
else:
    print("WARNING: multi_verify renders missing, skipping generalization figure")

print("done")
