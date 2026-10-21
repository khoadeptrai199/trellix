"""
Pure-PyTorch replacement for the `vox2seq` CUDA extension.

`vox2seq` is only used (trellis/modules/sparse/attention/serialized_attn.py) to compute a
space-filling-curve index for 3D voxel coordinates, which is then used purely as a *sort key*
to group spatially-nearby points into local attention windows. Self-attention within a window
is invariant to the order of its tokens (no positional encoding is derived from this ordering
here), so this implementation does not need to be bit-identical to the original CUDA kernel's
curve construction -- it only needs to produce a reasonably space-local ordering. That means a
standard, independently-implemented Morton/Hilbert encoder is safe to substitute, unlike e.g.
sparse-conv kernel weights, which must match a specific layout exactly to give correct results
for a pretrained checkpoint.

Coordinates are assumed to be non-negative integers (TRELLIS keeps them in [0, 1023], i.e. 10 bits).
"""
from typing import List
import torch

__all__ = ['encode']

_NBITS = 16  # supports coords in [0, 65535); TRELLIS coords live in [0, 1023]


def _morton_encode(coords: torch.Tensor) -> torch.Tensor:
    """Z-order (Morton) encode. coords: [N, 3] int64 -> [N] int64."""
    def part1by2(x: torch.Tensor) -> torch.Tensor:
        x = x & 0x1fffff
        x = (x | (x << 32)) & 0x1f00000000ffff
        x = (x | (x << 16)) & 0x1f0000ff0000ff
        x = (x | (x << 8)) & 0x100f00f00f00f00f
        x = (x | (x << 4)) & 0x10c30c30c30c30c3
        x = (x | (x << 2)) & 0x1249249249249249
        return x

    x, y, z = coords.unbind(-1)
    return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)


def _axes_to_transpose(coords: torch.Tensor, nbits: int) -> torch.Tensor:
    """Skilling's AxesToTranspose: Cartesian coords -> Hilbert 'transpose' representation."""
    X = coords.clone().to(torch.int64)
    D = X.shape[1]
    M = 1 << (nbits - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(D):
            mask = (X[:, i] & Q) != 0
            X[:, 0] = torch.where(mask, X[:, 0] ^ P, X[:, 0])
            t = (X[:, 0] ^ X[:, i]) & P
            X[:, 0] = torch.where(~mask, X[:, 0] ^ t, X[:, 0])
            X[:, i] = torch.where(~mask, X[:, i] ^ t, X[:, i])
        Q >>= 1
    for i in range(1, D):
        X[:, i] = X[:, i] ^ X[:, i - 1]
    t2 = torch.zeros(X.shape[0], dtype=torch.int64, device=coords.device)
    Q = M
    while Q > 1:
        mask = (X[:, D - 1] & Q) != 0
        t2 = torch.where(mask, t2 ^ (Q - 1), t2)
        Q >>= 1
    for i in range(D):
        X[:, i] = X[:, i] ^ t2
    return X


def _transpose_to_index(X: torch.Tensor, nbits: int) -> torch.Tensor:
    """Interleave the per-axis bits of the Hilbert transpose (MSB-first) into a linear index."""
    H = torch.zeros(X.shape[0], dtype=torch.int64, device=X.device)
    for b in range(nbits - 1, -1, -1):
        for d in range(X.shape[1]):
            bit = (X[:, d] >> b) & 1
            H = (H << 1) | bit
    return H


def _hilbert_encode(coords: torch.Tensor, nbits: int = _NBITS) -> torch.Tensor:
    return _transpose_to_index(_axes_to_transpose(coords, nbits), nbits)


def encode(coords: torch.Tensor, mode: str = 'z_order', permute: List[int] = [0, 1, 2]) -> torch.Tensor:
    """
    Args:
        coords (torch.Tensor): [N, 3] non-negative integer voxel coordinates.
        mode (str): 'z_order' or 'hilbert'.
        permute (List[int]): axis order to apply before encoding (matches vox2seq's `permute` arg).

    Returns:
        torch.Tensor: [N] int64 curve index, monotonic in space-filling-curve order.
    """
    assert coords.dim() == 2 and coords.shape[1] == 3, f"Expected [N, 3] coords, got {coords.shape}"
    coords = coords[:, permute].to(torch.int64)
    if mode == 'z_order':
        return _morton_encode(coords)
    elif mode == 'hilbert':
        return _hilbert_encode(coords)
    else:
        raise ValueError(f"Unknown mode: {mode}")
