"""
Minimal stand-in for `kaolin.utils.testing.check_tensor`, used only so that `flexicubes.py`
doesn't need the full `kaolin` package (a large, mostly CUDA-extension-based library) installed
just for a shape-assertion helper. `flexicubes.py` calls this exclusively inside `assert`
statements to validate tensor shapes (with `None` as a wildcard dimension) -- it is not part of
the actual marching-cubes algorithm, so an independent reimplementation carries no numerical risk.
"""
from typing import Optional, Sequence
import torch

__all__ = ['check_tensor']


def check_tensor(tensor: torch.Tensor, shape: Sequence[Optional[int]], throw: bool = True) -> bool:
    ok = torch.is_tensor(tensor) and tensor.dim() == len(shape) and all(
        expected is None or actual == expected
        for actual, expected in zip(tensor.shape, shape)
    )
    if not ok and throw:
        raise ValueError(f"Expected tensor of shape {shape}, got {tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor)}")
    return ok
