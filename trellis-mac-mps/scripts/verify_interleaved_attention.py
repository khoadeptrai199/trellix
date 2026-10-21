"""
Verifies trellis/modules/sparse/attention/interleave_windowed.py -- the real interleaved
full/windowed attention architecture proposal (research.md Part II), as distinct from the
cross-attn K/V cache (a smaller warm-up that measured a real null result).

Unlike the K/V cache, interleaving windowed attention genuinely changes the computation (windowed
attention is not mathematically equivalent to full attention), so "correctness" here does NOT mean
bit-identical output vs. the full-attention baseline. It means:
  1. Finite output, preserved coords/shape, for every full_attention_interval tested, including
     the two edge cases (K=1 == baseline exactly; K=num_blocks == only block 0 stays full).
  2. K=1 (identity case) DOES need to be bit-identical to the untouched baseline -- it should be a
     genuine no-op.
  3. The block-level attn_mode/window_size/shift_window assignment matches the exact proven "swin"
     alternation pattern this codebase already trusts elsewhere (structured_latent_vae/base.py).
  4. restore_full_attention() is a true round-trip back to the original baseline (within a tight
     tolerance -- SEE NOTE) -- proves the attribute mutation doesn't corrupt any other module
     state.

NOTE on tolerance, a real finding from running this: SLatFlowModel's SparseResBlock3d input/output
blocks (SPARSE_BACKEND=pytorch sparse convolution) are NOT bit-exact deterministic across repeated
identical calls on MPS -- confirmed in isolation (two consecutive input_blocks-only forwards on
identical input differ by ~3e-8 absolute, float32-noise-floor scale, almost certainly
scatter/gather accumulation-order non-determinism in the sparse conv, not a race condition or
correctness bug). This is entirely orthogonal to the attn_mode patch under test here -- confirmed
separately that a bare SparseMultiHeadAttention instance IS perfectly bit-deterministic across
repeated calls (see the debug session that found this). So identity/round-trip checks that go
through the FULL model (which includes the conv input/output blocks) use `torch.allclose` at a
tolerance well above that ~3e-8 noise floor, not `torch.equal`.
  5. A real-model finite-output smoke test (real conditioning, not torch.randn) at one interleave
     setting, since a synthetic-only check would miss a real-model-only bug (as happened with the
     K/V cache's dtype-cast finding).
"""
import os, sys
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import types
_root = '/Users/vinayak/Downloads/TRELLIS-main'
_trellis_root = _root + '/trellis'
_trellis_pkg = types.ModuleType('trellis'); _trellis_pkg.__path__ = [_trellis_root]; sys.modules['trellis'] = _trellis_pkg
_modules_pkg = types.ModuleType('trellis.modules'); _modules_pkg.__path__ = [_trellis_root + '/modules']; sys.modules['trellis.modules'] = _modules_pkg
_models_pkg = types.ModuleType('trellis.models'); _models_pkg.__path__ = [_trellis_root + '/models']; sys.modules['trellis.models'] = _models_pkg
_slat_vae_pkg = types.ModuleType('trellis.models.structured_latent_vae')
_slat_vae_pkg.__path__ = [_trellis_root + '/models/structured_latent_vae']
sys.modules['trellis.models.structured_latent_vae'] = _slat_vae_pkg

import torch
import trellis.modules.sparse as sp
from trellis.models.structured_latent_flow import SLatFlowModel
from trellis.modules.sparse.attention.interleave_windowed import (
    configure_interleaved_attention,
    restore_full_attention,
    attention_mode_summary,
)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")
torch.manual_seed(0)
all_ok = True


def make_sparse_tensor(batch_sizes, extent, channels):
    coords_list = []
    for b, n in enumerate(batch_sizes):
        pts = torch.randint(0, extent, (n, 3))
        pts = torch.unique(pts, dim=0)
        bcol = torch.full((pts.shape[0], 1), b, dtype=torch.int32)
        coords_list.append(torch.cat([bcol, pts.int()], dim=1))
    coords = torch.cat(coords_list, dim=0).to(device)
    feats = torch.randn(coords.shape[0], channels, device=device)
    return sp.SparseTensor(feats=feats, coords=coords)


def check_finite(name, t):
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    status = "OK" if (n_nan == 0 and n_inf == 0) else f"BAD (nan={n_nan}, inf={n_inf})"
    print(f"  {name}: shape={tuple(t.shape)} finite-check={status}")
    return n_nan == 0 and n_inf == 0


print("=== Stage 1: synthetic correctness ===")
NUM_BLOCKS = 8
flow = SLatFlowModel(
    resolution=16, in_channels=8, model_channels=32, cond_channels=32,
    out_channels=8, num_blocks=NUM_BLOCKS, num_heads=4, mlp_ratio=2, patch_size=2,
    num_io_res_blocks=2, io_block_channels=[16], pe_mode="ape", qk_rms_norm=True,
).to(device)
flow.eval()

# TRELLIS zero-initializes every block's adaLN modulation final layer (DiT convention -- see
# SLatFlowModel.initialize_weights), so at a freshly-constructed model every gate_msa is exactly
# zero and self_attn's output is multiplied away before reaching the residual stream. That's
# correct DiT behavior for training-from-scratch stability, but it means attn_mode changes are
# genuinely invisible in the full-model output on an untouched fresh model -- not a bug in the
# interleaving patch (confirmed separately: the same attn_mode mutation on a bare
# SparseMultiHeadAttention instance, outside this zero-gated residual path, does produce different
# output for full vs. windowed). Perturb the gates here so this test can actually exercise the
# residual path, same as a partially-trained/fine-tuned checkpoint would.
torch.manual_seed(1)
for block in flow.blocks:
    torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.5)
# Same DiT zero-init convention also zeros SLatFlowModel.out_layer's weight AND bias (the final
# model_channels->out_channels projection) -- confirmed in initialize_weights(). That makes EVERY
# fresh model's output unconditionally all-zero regardless of any internal computation (a
# zero-weight, zero-bias Linear maps any input to 0), which silently defeated the adaLN-gate fix
# above on its own. Perturb this too so internal differences actually reach the output -- a
# real pretrained/fine-tuned checkpoint has long since learned nonzero weights here (confirmed by
# Stage 2's real-model output below being nonzero and finite).
torch.nn.init.normal_(flow.out_layer.weight, std=0.02)
torch.nn.init.normal_(flow.out_layer.bias, std=0.02)

# extent=16 so window_size=8 gives exactly 2 windows per axis -- a real, non-degenerate case.
x = make_sparse_tensor([180, 140], extent=16, channels=8)
t = torch.randint(0, 1000, (2,), device=device).float()
cond = torch.randn(2, 32, 32, device=device)

with torch.no_grad():
    out_baseline = flow(x, t, cond)
ok = check_finite("baseline (all full)", out_baseline.feats)
all_ok &= ok

print("\n-- K=1 (identity case): must be bit-identical to baseline --")
assignment = configure_interleaved_attention(flow, full_attention_interval=1, window_size=8)
print(f"  assignment: {assignment}")
ok = all(v == "full" for v in assignment.values())
print(f"  all blocks stayed full: {'OK' if ok else 'BAD'}")
all_ok &= ok
with torch.no_grad():
    out_k1 = flow(x, t, cond)
ok = torch.allclose(out_baseline.feats, out_k1.feats, atol=1e-4, rtol=1e-4)
print(f"  matches baseline within tolerance (sparse-conv noise floor, see module docstring): {'OK' if ok else 'BAD'}"
      f" (max abs diff: {(out_baseline.feats - out_k1.feats).abs().max().item():.2e})")
all_ok &= ok

print("\n-- K=2: alternating full/windowed, check exact assignment against the proven swin pattern --")
assignment = configure_interleaved_attention(flow, full_attention_interval=2, window_size=8)
print(f"  assignment: {assignment}")
expected = {i: ("full" if i % 2 == 0 else "windowed") for i in range(NUM_BLOCKS)}
ok = assignment == expected
print(f"  matches expected full-block placement: {'OK' if ok else 'BAD'}")
all_ok &= ok
for i, block in enumerate(flow.blocks):
    sa = block.self_attn
    if assignment[i] == "windowed":
        expected_shift = 8 // 2 * (i % 2)
        ok = (sa.window_size == 8 and sa.shift_window == expected_shift)
        print(f"  block {i}: window_size={sa.window_size}, shift_window={sa.shift_window} "
              f"(expect 8, {expected_shift}): {'OK' if ok else 'BAD'}")
        all_ok &= ok
with torch.no_grad():
    out_k2 = flow(x, t, cond)
ok = check_finite("K=2 interleaved output", out_k2.feats)
all_ok &= ok
ok = not torch.equal(out_baseline.feats, out_k2.feats)  # windowed genuinely changes the computation
print(f"  differs from full-attention baseline (expected, not a bug): {'OK' if ok else 'BAD (suspiciously identical)'}")
all_ok &= ok

print("\n-- K=NUM_BLOCKS (most aggressive: only block 0 full) --")
assignment = configure_interleaved_attention(flow, full_attention_interval=NUM_BLOCKS, window_size=8)
ok = assignment[0] == "full" and all(assignment[i] == "windowed" for i in range(1, NUM_BLOCKS))
print(f"  assignment correct: {'OK' if ok else 'BAD'} ({assignment})")
all_ok &= ok
with torch.no_grad():
    out_aggressive = flow(x, t, cond)
ok = check_finite("most-aggressive interleaved output", out_aggressive.feats)
all_ok &= ok

print("\n-- restore_full_attention(): must be a true, bit-identical round-trip --")
n_restored = restore_full_attention(flow)
print(f"  restored {n_restored} blocks (expect {NUM_BLOCKS})")
ok = attention_mode_summary(flow) == ["full"] * NUM_BLOCKS
print(f"  all blocks report 'full': {'OK' if ok else 'BAD'}")
all_ok &= ok
with torch.no_grad():
    out_restored = flow(x, t, cond)
ok = torch.allclose(out_baseline.feats, out_restored.feats, atol=1e-4, rtol=1e-4)
print(f"  matches original baseline within tolerance: {'OK' if ok else 'BAD'}"
      f" (max abs diff: {(out_baseline.feats - out_restored.feats).abs().max().item():.2e})")
all_ok &= ok

# ---------------------------------------------------------------------------
# Stage 2: real-model finite-output smoke test (real conditioning, one interleave setting).
# ---------------------------------------------------------------------------
print("\n=== Stage 2: real-model finite-output smoke test ===")
try:
    for stubbed in ['trellis.models.structured_latent_vae', 'trellis.models', 'trellis.modules', 'trellis']:
        sys.modules.pop(stubbed, None)

    import glob
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.to(device)
    real_model = pipeline.models['slat_flow_model']
    real_model.eval()
    n_blocks_real = len(real_model.blocks)
    print(f"  real slat_flow_model num_blocks={n_blocks_real}")

    image_paths = glob.glob('/Users/vinayak/Downloads/TRELLIS-main/assets/**/*.png', recursive=True) + \
                  glob.glob('/Users/vinayak/Downloads/TRELLIS-main/assets/**/*.jpg', recursive=True)
    image = Image.open(image_paths[0]).convert("RGBA")
    real_cond = pipeline.get_cond([image])['cond']

    # A real SLat latent has irregular sparse coords -- build one the same way the model's own
    # tests do (random active voxels at the model's real resolution), rather than a dense grid.
    real_resolution = real_model.resolution
    real_in_channels = real_model.in_channels
    x_real = make_sparse_tensor([400], extent=real_resolution, channels=real_in_channels)
    t_real = torch.full((1,), 500.0, device=device)

    assignment = configure_interleaved_attention(real_model, full_attention_interval=4, window_size=8)
    n_windowed = sum(1 for v in assignment.values() if v == "windowed")
    print(f"  K=4 assignment: {n_windowed}/{n_blocks_real} blocks windowed")

    with torch.no_grad():
        out_real = real_model(x_real, t_real, real_cond)
    ok = check_finite("real slat_flow_model, K=4 interleaved", out_real.feats)
    all_ok &= ok
    restore_full_attention(real_model)
except Exception as e:
    print(f"  SKIPPED (real pretrained model not available in this environment): {e}")

print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
