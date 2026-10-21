#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that chunked SDPA returns the same result as the unpatched one.

`compat_patches._patch_sdpa_chunking` slices attention along the query axis to
bound peak memory. Softmax is applied per query row, so this must be exactly
equivalent — this test is what makes that claim checkable rather than asserted.

Run directly:

    python tests/test_sdpa_chunking.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend as backend_mod  # noqa: E402  (before torch — sets the MPS fallback)
import torch  # noqa: E402

import compat_patches  # noqa: E402


def main() -> int:
    import torch.nn.functional as F

    reference = F.scaled_dot_product_attention  # capture before patching

    # A tiny budget forces chunking on small tensors, so the test is fast and
    # still exercises many chunks.
    assert compat_patches._patch_sdpa_chunking(budget=8 * 1024)
    chunked = F.scaled_dot_product_attention
    assert chunked is not reference, "patch did not take effect"

    device = torch.device(backend_mod.detect().device)
    print(f"device: {device}")

    cases = [
        ("basic 4d", (2, 4, 64, 16), None, torch.float32),
        ("many heads", (1, 12, 128, 32), None, torch.float32),
        ("fp16", (2, 8, 96, 32), None, torch.float16),
        ("3d (no head axis)", (6, 80, 24), None, torch.float32),
        ("additive mask", (2, 4, 48, 16), "additive", torch.float32),
        ("bool mask", (2, 4, 48, 16), "bool", torch.float32),
        ("broadcast mask", (2, 4, 48, 16), "broadcast", torch.float32),
        ("uneven split", (1, 3, 37, 8), None, torch.float32),
    ]

    ok = True
    for name, shape, mask_kind, dtype in cases:
        torch.manual_seed(0)
        len_k = shape[-2] + 7  # keys longer than queries, as in cross-attention
        q = torch.randn(*shape, dtype=dtype, device=device)
        k = torch.randn(*shape[:-2], len_k, shape[-1], dtype=dtype, device=device)
        v = torch.randn(*shape[:-2], len_k, shape[-1], dtype=dtype, device=device)

        if mask_kind == "additive":
            mask = torch.zeros(shape[-2], len_k, dtype=dtype, device=device)
            mask[:, len_k // 2:] = float("-inf")
        elif mask_kind == "bool":
            mask = torch.rand(shape[-2], len_k, device=device) > 0.3
            mask[:, 0] = True  # never mask a whole row: that is a NaN, not a test
        elif mask_kind == "broadcast":
            mask = torch.zeros(1, len_k, dtype=dtype, device=device)
            mask[:, -1] = float("-inf")
        else:
            mask = None

        expected = reference(q, k, v, mask)
        got = chunked(q, k, v, mask)

        same_shape = expected.shape == got.shape
        max_diff = float((expected.float() - got.float()).abs().max()) if same_shape else float("nan")
        tol = 2e-3 if dtype == torch.float16 else 1e-5
        passed = same_shape and max_diff <= tol
        ok &= passed
        print(
            f"  [{'PASS' if passed else 'FAIL'}] {name:20s} shape={tuple(shape)} "
            f"mask={mask_kind or 'none':10s} max_diff={max_diff:.2e} (tol {tol:.0e})"
        )

    # The wrapper must stay out of the way when the matrix is small.
    q = torch.randn(1, 2, 8, 8, device=device)
    small_same = torch.equal(reference(q, q, q), chunked(q, q, q))
    print(f"  [{'PASS' if small_same else 'FAIL'}] below-budget calls are bit-identical")
    ok &= small_same

    print()
    print("ALL SDPA CHUNKING TESTS PASSED" if ok else "SOME SDPA CHUNKING TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
