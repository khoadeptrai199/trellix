"""
CUDA-only (DGX pod). Distills the interleaved full/windowed-attention student from the frozen,
pretrained full-attention teacher, per research.md Part II's ablation plan. Requires
scripts/dgx_generate_distillation_data.py to have already produced `distillation_data/*.pt`.

Distillation protocol (matches the plan, not a from-scratch reinvention): student starts from the
teacher's own pretrained weights (valid because `attn_mode` is a pure forward-time flag -- see
trellis/modules/sparse/attention/interleave_windowed.py's docstring -- so this is a warm-started
fine-tune adapting existing weights to a new computation pattern, not training a differently-shaped
model from scratch). Only the windowed blocks' self_attn parameters are trainable; everything else
(cross_attn, mlp, full-attention blocks, input/output layers) stays frozen, matching the plan's
"fine-tune only the swapped self-attention modules... first." Loss matches the student's predicted
velocity to the TEACHER's prediction (not the true flow-matching target) over `(cond, t, x_t)`
triples -- `x_t` derived from the real teacher-generated `x_0` latents via the same
`diffuse()`/timestep convention as trellis/trainers/flow_matching/flow_matching.py (logit-normal
`t`, `sigma_min=1e-5`), not a reinvented noise schedule.

Per this project's platform-separation convention: CUDA-only, no MPS code, never merged into the
MPS-side verify/benchmark scripts.
"""
import os, sys, glob, copy, time, argparse
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

# See scripts/dgx_generate_distillation_data.py's identical comment: trellis/pipelines/__init__.py
# unconditionally imports TrellisTextTo3DPipeline, which needs open3d only for type annotations on
# methods this script never calls. Stubbed rather than chasing this pod's missing libGL.so.1 for
# a pipeline path that's genuinely irrelevant here.
class _DummyO3D:
    def __getattr__(self, name):
        return _DummyO3D()
    def __call__(self, *a, **kw):
        return _DummyO3D()
if 'open3d' not in sys.modules:
    sys.modules['open3d'] = _DummyO3D()

import torch
import torch.nn.functional as F
from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.modules.sparse as sp
from trellis.modules.sparse.attention.interleave_windowed import configure_interleaved_attention, attention_mode_summary

parser = argparse.ArgumentParser()
parser.add_argument("--full-attention-interval", type=int, default=8, help="K -- see research.md Part II sweep")
parser.add_argument("--window-size", type=int, default=16, help="window_size -- 16 beat 8 in the real benchmark")
parser.add_argument("--num-steps", type=int, default=300, help="ADDITIONAL steps to run this invocation, not a total")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--log-every", type=int, default=20)
parser.add_argument("--save-path", type=str, default=None)
parser.add_argument("--no-resume", action="store_true",
                     help="start fresh from the teacher's own weights instead of continuing from an "
                          "existing checkpoint at --save-path (auto-detected by default if present)")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})\n")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'distillation_data')
SIGMA_MIN = 1e-5


def diffuse(x_0, t, noise):
    t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
    return (1 - t) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise


def sample_t(batch_size, device, mean=0.0, std=1.0):
    return torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)


# --- load teacher, build student ---
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
teacher = pipeline.models['slat_flow_model']
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

student = copy.deepcopy(teacher)
student = student.float()  # fp32 for stable training; teacher stays native fp16 for inference
student.dtype = torch.float32  # forward() casts cond/t_emb via self.dtype -- must match the fp32 weights now
student.train()

n_blocks = len(student.blocks)
assignment = configure_interleaved_attention(student, full_attention_interval=args.full_attention_interval,
                                              window_size=args.window_size)
print(f"Student attn_mode assignment (K={args.full_attention_interval}, window_size={args.window_size}): "
      f"{attention_mode_summary(student)}\n")

save_path = args.save_path or os.path.join(
    REPO_ROOT, 'distillation_data', f'student_K{args.full_attention_interval}_W{args.window_size}.pt')

prior_losses = []
if not args.no_resume and os.path.exists(save_path):
    prior_ckpt = torch.load(save_path, map_location=device, weights_only=False)
    assert prior_ckpt['assignment'] == assignment, "existing checkpoint's block assignment doesn't match this run's K/window_size"
    student_sd = student.state_dict()
    for key, val in prior_ckpt['trainable_state_dict'].items():
        student_sd[key] = val.to(device).to(student_sd[key].dtype)
    student.load_state_dict(student_sd)
    prior_losses = prior_ckpt['losses']
    print(f"Resumed from {save_path}: {len(prior_losses)} prior steps, "
          f"last loss={prior_losses[-1]:.6f} -- training {args.num_steps} more.\n")
else:
    print(f"Starting fresh from the teacher's own pretrained weights (no checkpoint found at "
          f"{save_path}, or --no-resume set).\n")

for p in student.parameters():
    p.requires_grad_(False)
trainable_params = []
for i, block in enumerate(student.blocks):
    if assignment[i] == "windowed":
        for p in block.self_attn.parameters():
            p.requires_grad_(True)
            trainable_params.append(p)
n_trainable = sum(p.numel() for p in trainable_params)
print(f"Trainable: {n_trainable/1e6:.2f}M params across {sum(1 for v in assignment.values() if v=='windowed')} "
      f"windowed blocks' self_attn (out of {n_blocks} total blocks)\n")

optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

# --- load cached (cond, x_0) pairs ---
# DATA_DIR also holds this script's own checkpoints (student_K*_W*.pt) and other scripts'
# result dumps (quality_check_results.pt, seed_variance_baseline.pt) -- a bare '*.pt' glob picks
# those up too (a real bug hit once: 42 real data files + 3 non-data files = 45 "pairs" loaded,
# crashing when random.choice happened to pick one, since they don't have the {cond, feats,
# coords} dict shape). Excluding known non-data prefixes rather than assuming the count.
_NON_DATA_PREFIXES = ('student_', 'quality_check_results', 'seed_variance_baseline')
data_files = sorted(
    f for f in glob.glob(os.path.join(DATA_DIR, '*.pt'))
    if not os.path.basename(f).startswith(_NON_DATA_PREFIXES)
)
assert len(data_files) > 0, f"no distillation data found in {DATA_DIR} -- run scripts/dgx_generate_distillation_data.py first"
print(f"Loaded {len(data_files)} (cond, x_0) pairs from {DATA_DIR}\n")
dataset = [torch.load(f, map_location=device) for f in data_files]
for f, d in zip(data_files, dataset):
    assert isinstance(d, dict) and 'cond' in d, f"{f} doesn't look like training data (got {type(d)}) -- check _NON_DATA_PREFIXES"

import random
losses = []
t0 = time.time()
for step in range(args.num_steps):
    sample = random.choice(dataset)
    cond = sample['cond'].to(device)
    x_0_feats = sample['feats'].to(device)
    coords = sample['coords'].to(device)

    t = sample_t(1, device)
    noise = torch.randn_like(x_0_feats)
    x_t_feats = diffuse(x_0_feats, t, noise)
    t_scaled = t * 1000

    # Both models' input_layer/t_embedder/pos_embedder stay fp32 even when the model's "torso"
    # (self.blocks) runs fp16 -- convert_to_fp16 only touches input_blocks/blocks/out_blocks, not
    # these (confirmed by reading SLatFlowModel.convert_to_fp16). forward() does its own internal
    # `.type(self.dtype)` casts for cond/t_emb/h at the right points; passing pre-cast fp16 inputs
    # here (an earlier version of this script did) double-casts and crashes on the fp32-weighted
    # input_layer. Pass fp32 straight through for both models and let forward() handle it.
    with torch.no_grad():
        x_t_teacher = sp.SparseTensor(feats=x_t_feats, coords=coords)
        teacher_pred = teacher(x_t_teacher, t_scaled, cond)

    x_t_student = sp.SparseTensor(feats=x_t_feats, coords=coords)
    student_pred = student(x_t_student, t_scaled, cond)

    loss = F.mse_loss(student_pred.feats.float(), teacher_pred.feats.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (step + 1) % args.log_every == 0:
        recent = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
        print(f"step {step+1}/{args.num_steps}: loss={loss.item():.6f}, "
              f"avg(last {args.log_every})={recent:.6f}, {time.time()-t0:.1f}s elapsed")

print(f"\nDone (this run). Final loss: {losses[-1]:.6f}, "
      f"first-{args.log_every} avg: {sum(losses[:args.log_every])/args.log_every:.6f}, "
      f"last-{args.log_every} avg: {sum(losses[-args.log_every:])/args.log_every:.6f}")

all_losses = prior_losses + losses
if prior_losses:
    print(f"Total steps across all resumed runs: {len(all_losses)}. "
          f"Very first {args.log_every} avg: {sum(all_losses[:args.log_every])/args.log_every:.6f} -> "
          f"latest {args.log_every} avg: {sum(all_losses[-args.log_every:])/args.log_every:.6f}")

torch.save({
    'assignment': assignment,
    'full_attention_interval': args.full_attention_interval,
    'window_size': args.window_size,
    'trainable_state_dict': {
        f'blocks.{i}.self_attn.{k}': v
        for i, block in enumerate(student.blocks) if assignment[i] == "windowed"
        for k, v in block.self_attn.state_dict().items()
    },
    'losses': all_losses,
}, save_path)
print(f"Saved trainable state dict + loss history ({len(all_losses)} total steps) -> {save_path}")
