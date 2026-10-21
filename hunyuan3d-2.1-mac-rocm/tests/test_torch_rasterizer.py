#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check `torch_rasterizer` against a literal transcription of upstream's C++.

The reference below is a line-by-line port of `rasterize_image_cpu` /
`rasterizeTriangleCPU` / `barycentricFromImgcoordCPU` from
`custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp`, using Python
ints and floats so the integer truncation and the packed z-buffer token behave
the same way the compiled kernel does. It is deliberately slow and dumb — that
is what makes it a usable oracle.

Run directly (no pytest needed):

    python tests/test_torch_rasterizer.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# `backend` before torch: it is what sets PYTORCH_ENABLE_MPS_FALLBACK, and torch
# only reads that during initialisation. Without it an MPS run of this test can
# fail on an op that the app itself would have run happily.
import backend as backend_mod  # noqa: E402
import torch  # noqa: E402
import torch_rasterizer as tr  # noqa: E402

MAXINT = 2147483647


# ---------------------------------------------------------------------------
# Reference implementation (transcribed from rasterizer.cpp)
# ---------------------------------------------------------------------------
def _signed_area2(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1])


def _bary(a, b, c, p):
    beta_tri = _signed_area2(a, (p[0], p[1]), c)
    gamma_tri = _signed_area2(a, b, (p[0], p[1]))
    area = _signed_area2(a, b, c)
    if area == 0:
        return [-1.0, -1.0, -1.0]
    tri_inv = 1.0 / area
    beta = beta_tri * tri_inv
    gamma = gamma_tri * tri_inv
    alpha = 1.0 - beta - gamma
    return [alpha, beta, gamma]


def _in_bounds(b):
    return all(0.0 <= v <= 1.0 for v in b)


def reference_rasterize(V, F, width, height, D=None, use_depth_prior=0):
    """V: (N,4) float32, F: (T,3) int32. Returns (findices, barycentric)."""
    occlusion_truncation = 1e-6
    n_pix = width * height
    maxint = MAXINT * MAXINT + (MAXINT - 1)
    zbuf = [maxint] * n_pix

    def screen(vi):
        vp = V[vi]
        w = float(vp[3])
        return (
            (float(vp[0]) / w * 0.5 + 0.5) * (width - 1) + 0.5,
            (0.5 + 0.5 * float(vp[1]) / w) * (height - 1) + 0.5,
            float(vp[2]) / w * 0.49999 + 0.5,
        )

    for f_idx in range(F.shape[0]):
        vt0 = screen(F[f_idx][0])
        vt1 = screen(F[f_idx][1])
        vt2 = screen(F[f_idx][2])

        x_min = min(vt0[0], min(vt1[0], vt2[0]))
        x_max = max(vt0[0], max(vt1[0], vt2[0]))
        y_min = min(vt0[1], min(vt1[1], vt2[1]))
        y_max = max(vt0[1], max(vt1[1], vt2[1]))

        # `int px = x_min` in C truncates toward zero.
        px = int(x_min)
        while px < x_max + 1:
            if 0 <= px < width:
                py = int(y_min)
                while py < y_max + 1:
                    if 0 <= py < height:
                        vt = (px + 0.5, py + 0.5)
                        b = _bary(vt0, vt1, vt2, vt)
                        if _in_bounds(b):
                            pixel = py * width + px
                            depth = b[0] * vt0[2] + b[1] * vt1[2] + b[2] * vt2[2]
                            depth_thres = 0.0
                            if D is not None and use_depth_prior:
                                depth_thres = (
                                    float(D[pixel]) * 0.49999 + 0.5 + occlusion_truncation
                                )
                            z_quantize = int(depth * (2 << 17))  # truncating
                            token = z_quantize * MAXINT + (f_idx + 1)
                            if depth >= depth_thres:
                                zbuf[pixel] = min(zbuf[pixel], token)
                    py += 1
            px += 1

    def c_mod(t: int, d: int) -> int:
        """C's `%`: quotient truncates toward zero.

        Must stay in exact integer arithmetic. Going through float (`int(t / d)`)
        loses precision at the z-buffer's initial value (~4.6e18), which silently
        turns the empty marker into -1.
        """
        q = t // d if t >= 0 else -((-t) // d)
        return t - q * d

    findices = [0] * n_pix
    barycentric = [[0.0, 0.0, 0.0] for _ in range(n_pix)]
    for pix in range(n_pix):
        t = zbuf[pix]
        f = c_mod(t, MAXINT)
        if f == MAXINT - 1:
            continue
        findices[pix] = f
        f -= 1
        if f < 0:
            continue
        vt = (pix % width + 0.5, pix // width + 0.5)
        i0, i1, i2 = F[f][0], F[f][1], F[f][2]
        s0, s1, s2 = screen(i0), screen(i1), screen(i2)
        b = _bary(s0, s1, s2, vt)
        w0, w1, w2 = float(V[i0][3]), float(V[i1][3]), float(V[i2][3])
        b = [b[0] / w0, b[1] / w1, b[2] / w2]
        s = b[0] + b[1] + b[2]
        if s == 0:
            continue
        barycentric[pix] = [v / s for v in b]

    return (
        np.array(findices, dtype=np.int64).reshape(height, width),
        np.array(barycentric, dtype=np.float64).reshape(height, width, 3),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def _make_scene(seed: int, n_tri: int, spread: float = 1.0):
    """Random triangles in front of the camera, with varied depths and sizes."""
    rng = np.random.default_rng(seed)
    n_vert = n_tri * 3
    xy = rng.uniform(-spread, spread, size=(n_vert, 2)).astype(np.float32)
    z = rng.uniform(-0.6, 0.6, size=(n_vert, 1)).astype(np.float32)
    w = np.ones((n_vert, 1), dtype=np.float32)
    V = np.concatenate([xy, z, w], axis=1).astype(np.float32)
    F = np.arange(n_vert, dtype=np.int32).reshape(n_tri, 3)
    return V, F


def _compare(name, V, F, width, height, device, D=None, use_depth_prior=0):
    ref_f, ref_b = reference_rasterize(V, F, width, height, D=D, use_depth_prior=use_depth_prior)

    pos = torch.from_numpy(V).unsqueeze(0).to(device)
    tri = torch.from_numpy(F).to(device)
    dprior = None if D is None else torch.from_numpy(D).to(device)
    got_f, got_b = tr.rasterize(
        pos, tri, (height, width), clamp_depth=dprior, use_depth_prior=use_depth_prior
    )

    got_f_np = got_f.cpu().numpy().astype(np.int64)
    got_b_np = got_b.cpu().numpy().astype(np.float64)

    same_f = np.array_equal(got_f_np, ref_f)
    covered = ref_f > 0
    if covered.any():
        max_db = float(np.abs(got_b_np[covered] - ref_b[covered]).max())
    else:
        max_db = 0.0
    n_cov = int(covered.sum())

    ok = same_f and max_db < 2e-4
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name:38s} device={str(device):5s} "
        f"covered={n_cov:6d}/{width*height:6d} findices_equal={same_f} "
        f"max_bary_diff={max_db:.2e}"
    )
    if not same_f:
        diff = np.argwhere(got_f_np != ref_f)
        print(f"        first {min(5, len(diff))} mismatching pixels:")
        for y, x in diff[:5]:
            print(f"          (y={y}, x={x}) ref={ref_f[y, x]} got={got_f_np[y, x]}")
    return ok


def main() -> int:
    # CPU always, plus whatever accelerator this machine actually has. The CPU
    # run is the one compared against the C++ oracle; the accelerator run is what
    # catches device-specific arithmetic — the packed z-buffer token needs real
    # int64, which is exactly the sort of thing a GPU backend gets wrong.
    devices = [torch.device("cpu")]
    accel = torch.device(backend_mod.detect().device)
    if accel.type != "cpu":
        devices.append(accel)

    all_ok = True
    for device in devices:
        print(f"device: {device}")

        # Small overlapping triangles: exercises the z-buffer tie-breaking.
        V, F = _make_scene(seed=0, n_tri=8, spread=0.9)
        all_ok &= _compare("8 overlapping triangles 64x64", V, F, 64, 64, device)

        # Mixed footprints: forces several power-of-two buckets in one call.
        V, F = _make_scene(seed=1, n_tri=40, spread=1.4)
        all_ok &= _compare("40 triangles, clipped, 96x64", V, F, 96, 64, device)

        # Non-square resolution, to catch width/height transposition.
        V, F = _make_scene(seed=2, n_tri=25, spread=0.7)
        all_ok &= _compare("25 triangles 128x48 (non-square)", V, F, 128, 48, device)

        # Degenerate triangle (all three vertices collinear).
        V, F = _make_scene(seed=3, n_tri=4, spread=0.8)
        V[3] = V[4] = V[5] = V[3]
        all_ok &= _compare("with a degenerate triangle", V, F, 64, 64, device)

        # Fully off-screen geometry: every triangle must be dropped.
        V, F = _make_scene(seed=4, n_tri=6, spread=0.4)
        V[:, 0] += 8.0
        all_ok &= _compare("all off-screen", V, F, 48, 48, device)

        # Depth prior path.
        V, F = _make_scene(seed=5, n_tri=12, spread=0.9)
        D = np.full((64 * 64,), -0.4, dtype=np.float32)
        all_ok &= _compare("depth prior active", V, F, 64, 64, device, D=D, use_depth_prior=1)

        # Perspective divide: non-unit w must change the barycentrics.
        V, F = _make_scene(seed=6, n_tri=10, spread=0.6)
        V[:, 3] = np.linspace(1.0, 2.5, V.shape[0]).astype(np.float32)
        V[:, :3] *= V[:, 3:4]
        all_ok &= _compare("perspective (non-unit w)", V, F, 64, 64, device)

    print()
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
