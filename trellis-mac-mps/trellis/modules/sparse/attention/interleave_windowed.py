"""
Interleaved full/windowed self-attention for TRELLIS's flow (denoiser) transformers -- the actual
architecture proposal in research.md Part II, not the smaller cross-attn K/V-cache warm-up (which
measured a real null result, see research.md).

Key fact this whole module leans on: `attn_mode` is a pure forward-time behavior flag on
SparseMultiHeadAttention, not something that changes parameter shapes. `to_qkv`/`to_out`/
`q_rms_norm`/`k_rms_norm` are constructed identically regardless of attn_mode (confirmed by
reading trellis/modules/sparse/attention/modules.py's __init__ -- nothing there branches on
attn_mode). So converting an already-constructed (and possibly already pretrained-loaded) block's
self-attention from full to windowed is a pure attribute mutation -- no reinitialization, no shape
change, no need to rebuild the block or touch trellis/models/structured_latent_flow.py at all.
Matches this project's non-destructive-patch convention (same technique as
trellis/modules/attention/cross_kv_cache.py): a new file, zero edits to any existing model/block
file.

Shift pattern matches, exactly, the one already proven in the working SLat VAE encoder/decoder
classes (attn_mode="swin"): trellis/models/structured_latent_vae/base.py's `block_attn_config`
generator alternates `shift_window = window_size // 2 * (i % 2)` -- even blocks unshifted, odd
blocks shifted by half the window. Reused verbatim here rather than invented, since it's already
trusted elsewhere in this codebase on real sparse voxel data.

Cross-attention is never touched -- SparseMultiHeadAttention's own constructor asserts
`type == "self" or attn_mode == "full"` (cross-attention only supports full attention), and this
module only ever mutates `block.self_attn`, never `block.cross_attn`.
"""
from typing import Dict, List
import torch.nn as nn


def configure_interleaved_attention(model: nn.Module, full_attention_interval: int, window_size: int = 8) -> Dict[int, str]:
    """
    Converts most of `model`'s self-attention blocks to windowed attention, keeping a full block
    every `full_attention_interval` blocks (block 0 is always full; blocks 0, K, 2K, ... stay
    full). `full_attention_interval=1` is the identity operation (every block stays full, exactly
    the pretrained baseline). Windowed blocks alternate shift_window 0/(window_size//2) by parity,
    matching the proven "swin" pattern in structured_latent_vae/base.py.

    Returns {block_index: "full"|"windowed"} recording the assignment actually made, so a caller
    can log/assert what a given (K, W) configuration produced.
    """
    assert full_attention_interval >= 1, "full_attention_interval must be >= 1"
    assignment = {}
    for i, block in enumerate(model.blocks):
        self_attn = block.self_attn
        if i % full_attention_interval == 0:
            self_attn.attn_mode = "full"
            assignment[i] = "full"
        else:
            self_attn.attn_mode = "windowed"
            self_attn.window_size = window_size
            self_attn.shift_window = window_size // 2 * (i % 2)
            assignment[i] = "windowed"
    return assignment


def restore_full_attention(model: nn.Module) -> int:
    """Reverts every block's self-attention back to attn_mode='full' (the pretrained default)."""
    restored = 0
    for block in model.blocks:
        block.self_attn.attn_mode = "full"
        restored += 1
    return restored


def attention_mode_summary(model: nn.Module) -> List[str]:
    """Returns the current attn_mode of every block's self-attention, in order."""
    return [block.self_attn.attn_mode for block in model.blocks]
