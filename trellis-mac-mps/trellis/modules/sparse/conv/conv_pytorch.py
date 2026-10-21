"""
Pure-PyTorch submanifold sparse 3D convolution backend (no CUDA extension required).

Every `SparseConv3d(...)` call site in `trellis/models` uses the default `stride=1`, which the
spconv backend itself routes to `SubMConv3d` (submanifold conv: output coordinates equal input
coordinates, downsampling is instead handled separately by `SparseDownsample`/`SparseUpsample`
in `trellis/modules/sparse/spatial.py`). So this module only implements the submanifold case --
strided/rulebook sparse conv and `SparseInverseConv3d` are not needed by the current model code
and are not implemented here.

Algorithm: for each of the kernel's k^3 offsets, look up -- for every active voxel -- whether the
voxel at (coord + offset) is also active (via sort + searchsorted over an integer coordinate key),
gather that neighbor's features, project with that offset's weight slice, and scatter-add into the
output. This is O(k^3) vectorized gather/matmul/scatter passes instead of one fused CUDA kernel, so
it is meaningfully slower than spconv's `flex_gemm`, but every op it uses (index_select, matmul,
searchsorted, index_add_) has a real Metal/MPS implementation.

IMPORTANT -- weight layout is NOT verified against spconv: the algorithm here is self-consistent
(a network trained from scratch with this module will work correctly), but if you load pretrained
spconv checkpoint weights via `spconv_weight_to_pytorch`, the mapping from kernel offset to weight
slice is this module's own best-effort convention (standard centered/row-major, matching e.g.
`torch.nn.Conv3d`), not something verified bit-for-bit against spconv's internal (CUDA-only)
rulebook ordering -- spconv cannot be installed on macOS to cross-check against directly. Verify
numerically against a real spconv forward pass on a CUDA machine before trusting inference output
from pretrained weights.
"""
from typing import *
import itertools
import torch
import torch.nn as nn
from .. import SparseTensor
from .. import DEBUG

__all__ = [
    'PytorchSparseTensorData',
    'SparseConv3d',
    'SparseInverseConv3d',
]


class PytorchSparseTensorData:
    """Minimal spconv-SparseConvTensor-like container backing SparseTensor when BACKEND == 'pytorch'."""

    def __init__(self, features: torch.Tensor, indices: torch.Tensor, spatial_shape=None, batch_size=None):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def replace_feature(self, features: torch.Tensor) -> 'PytorchSparseTensorData':
        return PytorchSparseTensorData(features, self.indices, self.spatial_shape, self.batch_size)

    def dense(self) -> torch.Tensor:
        """Returns a [batch_size, C, *spatial_shape] dense tensor (matches spconv's .dense() layout)."""
        C = self.features.shape[1]
        out = torch.zeros(self.batch_size, C, *self.spatial_shape, dtype=self.features.dtype, device=self.features.device)
        b, x, y, z = self.indices.unbind(-1)
        out[b.long(), :, x.long(), y.long(), z.long()] = self.features
        return out


def _encode_coords(coords: torch.Tensor, spatial_size: int = 1 << 20) -> torch.Tensor:
    """Pack [N, 4] (batch, x, y, z) non-negative int coords into a single sortable int64 key."""
    b, x, y, z = coords.long().unbind(-1)
    return ((b * spatial_size + x) * spatial_size + y) * spatial_size + z


def _kernel_offsets(kernel_size: Tuple[int, int, int], dilation: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    ranges = [range(-(k // 2), k - k // 2) for k in kernel_size]
    return [tuple(o * d for o, d in zip(offset, dilation)) for offset in itertools.product(*ranges)]


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super().__init__()
        stride_t = stride if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        assert all(s == 1 for s in stride_t) and padding is None, (
            "The pure-PyTorch sparse conv backend only implements submanifold (stride=1, padding=None) "
            "convolution -- every SparseConv3d(...) call in trellis/models already uses stride=1; spatial "
            "downsampling/upsampling goes through SparseDownsample/SparseUpsample instead."
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size, kernel_size)
        self.dilation = dilation if isinstance(dilation, (list, tuple)) else (dilation, dilation, dilation)
        self.offsets = _kernel_offsets(self.kernel_size, self.dilation)
        self.kernel_volume = len(self.offsets)
        self._center_idx = self.offsets.index((0, 0, 0)) if (0, 0, 0) in self.offsets else None

        self.weight = nn.Parameter(torch.empty(self.kernel_volume, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_channels * self.kernel_volume
        bound = fan_in ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: SparseTensor) -> SparseTensor:
        coords = x.coords          # [N, 4] (batch, x, y, z)
        feats = x.feats            # [N, Cin]
        N = coords.shape[0]
        device = feats.device

        key = _encode_coords(coords)
        sorted_key, sort_idx = torch.sort(key)

        out = torch.zeros(N, self.out_channels, device=device, dtype=feats.dtype)

        if self._center_idx is not None:
            out = out + feats @ self.weight[self._center_idx]

        for i, offset in enumerate(self.offsets):
            if offset == (0, 0, 0):
                continue
            neighbor_coords = coords.clone()
            neighbor_coords[:, 1:] += torch.tensor(offset, device=device, dtype=coords.dtype)
            nkey = _encode_coords(neighbor_coords)

            pos = torch.searchsorted(sorted_key, nkey)
            pos_clamped = pos.clamp(max=sorted_key.shape[0] - 1)
            found = (pos < sorted_key.shape[0]) & (sorted_key[pos_clamped] == nkey)
            if not torch.any(found):
                continue

            query_idx = found.nonzero(as_tuple=True)[0]
            src_idx = sort_idx[pos_clamped[query_idx]]
            contrib = feats[src_idx] @ self.weight[i]
            out.index_add_(0, query_idx, contrib)

        if self.bias is not None:
            out = out + self.bias

        if DEBUG:
            assert out.shape == (N, self.out_channels)

        return x.replace(out)


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super().__init__()
        raise NotImplementedError(
            "SparseInverseConv3d is not used by any model in trellis/models (all SparseConv3d calls are "
            "stride=1 submanifold; spatial changes go through SparseDownsample/SparseUpsample), so the "
            "pure-PyTorch backend does not implement it. If new code needs it, it requires the general "
            "strided rulebook conv, not just the submanifold case this backend covers."
        )


def spconv_weight_to_pytorch(spconv_weight: torch.Tensor, kernel_size: Tuple[int, int, int], dilation: Tuple[int, int, int] = (1, 1, 1)) -> torch.Tensor:
    """
    Best-effort conversion of a pretrained spconv SubMConv3d weight (KRSC layout:
    [out_channels, kz, ky, kx, in_channels]) into this module's [kernel_volume, in, out] layout,
    assuming the standard centered/row-major kernel-index convention.

    NOT verified against spconv on real hardware (spconv cannot be installed on this machine to
    cross-check) -- treat output correctness as unverified until compared against a real spconv
    forward pass on a CUDA machine.
    """
    out_channels, kz, ky, kx, in_channels = spconv_weight.shape
    assert (kz, ky, kx) == tuple(kernel_size), f"kernel_size mismatch: weight has {(kz, ky, kx)}, expected {tuple(kernel_size)}"
    offsets = _kernel_offsets(tuple(kernel_size), tuple(dilation))
    out = torch.empty(len(offsets), in_channels, out_channels, dtype=spconv_weight.dtype)
    c = [k // 2 for k in kernel_size]
    for idx, offset in enumerate(offsets):
        oz, oy, ox = (o // d for o, d in zip(offset, dilation))
        iz, iy, ix = oz + c[0], oy + c[1], ox + c[2]
        out[idx] = spconv_weight[:, iz, iy, ix, :].transpose(0, 1)
    return out


def adapt_state_dict_for_pytorch_backend(model: nn.Module, state_dict: dict) -> dict:
    """
    Rewrite a spconv-trained checkpoint's state dict to load into this pure-PyTorch backend.

    spconv's SparseConv3d/SubMConv3d module stores its weight/bias under a nested `.conv`
    submodule (`self.conv = spconv.SubMConv3d(...)`), in KRSC layout
    (`[out_channels, kz, ky, kx, in_channels]`). This module's SparseConv3d stores weight/bias as
    direct attributes, in `[kernel_volume, in_channels, out_channels]` layout -- so loading a real
    checkpoint needs both a key rename (`X.conv.weight` -> `X.weight`) and a layout conversion
    (`spconv_weight_to_pytorch`), not just a rename.

    Only touches keys belonging to actual `SparseConv3d` modules in `model` (found via
    `model.named_modules()`), so this is a safe no-op when the active sparse backend isn't
    'pytorch' -- there won't be any modules of this class to match against.

    NOTE: this makes real pretrained-checkpoint loading *possible* (correct key names and shapes),
    but the underlying kernel-offset convention used by `spconv_weight_to_pytorch` is still
    unverified against real spconv output -- see that function's docstring.
    """
    state_dict = dict(state_dict)
    for name, module in model.named_modules():
        if not isinstance(module, SparseConv3d):
            continue
        w_key, new_w_key = f"{name}.conv.weight", f"{name}.weight"
        b_key, new_b_key = f"{name}.conv.bias", f"{name}.bias"
        if w_key in state_dict:
            raw_weight = state_dict.pop(w_key)
            state_dict[new_w_key] = spconv_weight_to_pytorch(raw_weight, module.kernel_size, module.dilation)
        if b_key in state_dict:
            state_dict[new_b_key] = state_dict.pop(b_key)
    return state_dict
