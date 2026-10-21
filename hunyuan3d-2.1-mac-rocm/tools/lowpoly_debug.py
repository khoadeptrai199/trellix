#!/usr/bin/env python3
"""Debug CLI for low-poly optimization.

Run every mode/level combination on an input GLB and produce a side-by-side
comparison table (vertices, faces, file size, deviation from original,
elapsed time). Simplified meshes are written to an output directory so they
can be inspected in a 3D viewer.

Usage:
    python tools/lowpoly_debug.py INPUT.glb
    python tools/lowpoly_debug.py INPUT.glb --modes quadric,planar,hybrid --levels medium
    python tools/lowpoly_debug.py INPUT.glb --out outputs/lowpoly_debug
    python tools/lowpoly_debug.py INPUT.glb --samples 40000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

# Allow running as `python tools/lowpoly_debug.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lowpoly  # noqa: E402


ALL_MODES = ("quadric", "planar", "hybrid", "iso", "auto")
ALL_LEVELS = ("weak", "medium", "strong")


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n} B"


def fmt_stats(m: trimesh.Trimesh) -> str:
    return f"V={len(m.vertices):>6}  F={len(m.faces):>6}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input .glb/.obj/.ply/... mesh file")
    ap.add_argument("--modes", default=",".join(ALL_MODES),
                    help=f"Comma-separated list from {ALL_MODES}")
    ap.add_argument("--levels", default=",".join(ALL_LEVELS),
                    help=f"Comma-separated list from {ALL_LEVELS}")
    ap.add_argument("--out", default="outputs/lowpoly_debug",
                    help="Output directory for simplified meshes")
    ap.add_argument("--samples", type=int, default=20000,
                    help="Sample count for Hausdorff-style deviation (default 20000). "
                         "0 disables measurement.")
    ap.add_argument("--skip-write", action="store_true",
                    help="Don't save simplified GLBs, only measure")
    ap.add_argument("--format", default="glb", choices=("glb", "obj", "ply"),
                    help="Output format for simplified meshes")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"error: input not found: {in_path}", file=sys.stderr)
        return 2

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    for m in modes:
        if m not in ALL_MODES:
            print(f"error: unknown mode {m!r}", file=sys.stderr); return 2
    for l in levels:
        if l not in ALL_LEVELS:
            print(f"error: unknown level {l!r}", file=sys.stderr); return 2

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {in_path} ...", flush=True)
    t0 = time.time()
    original = trimesh.load(str(in_path), force="mesh")
    load_elapsed = time.time() - t0
    orig_size = in_path.stat().st_size

    if not isinstance(original, trimesh.Trimesh) or len(original.faces) == 0:
        print("error: input has no triangles", file=sys.stderr)
        return 2

    print(f"  loaded in {load_elapsed:.2f}s — {fmt_stats(original)}  "
          f"area={original.area:.4f}  bbox_diag={original.scale:.4f}  "
          f"size={human_bytes(orig_size)}", flush=True)

    # Mesh-type hint
    try:
        kind, ratio = lowpoly.classify_mesh(original)
        print(f"  classifier: {kind}  (large-planar area ratio = {ratio*100:.1f}%)",
              flush=True)
    except Exception as e:
        print(f"  classifier failed: {e}", flush=True)

    # Table header
    header = (
        "Mode       Level     V         F        Δ  bytes      "
        "time    dev_max    dev_p95    dev_mean   notes"
    )
    rows: list[str] = []
    print("\n" + header)
    print("-" * len(header))

    # Also report original as a baseline row
    rows.append(
        f"original   —         {len(original.vertices):<9} {len(original.faces):<8} "
        f"—          {human_bytes(orig_size):<10} —       —          —          —          "
        f"input"
    )
    print(rows[-1], flush=True)

    stem = in_path.stem
    for mode in modes:
        for level in levels:
            t1 = time.time()
            try:
                res = lowpoly.optimize(original, mode=mode, level=level)
            except Exception as e:
                elapsed = time.time() - t1
                err_str = str(e)[:80]
                row = f"{mode:<10} {level:<9} ERROR: {err_str}  ({elapsed:.1f}s)"
                print(row, flush=True)
                rows.append(row)
                continue
            total_elapsed = time.time() - t1
            mesh = res.mesh

            # Optionally save
            saved_size = -1
            rel_path = ""
            if not args.skip_write:
                name = f"{stem}__{mode}_{level}"
                if mode == "auto":
                    name += f"_as_{res.mode_used}"
                out_file = out_dir / f"{name}.{args.format}"
                try:
                    mesh.export(str(out_file))
                    saved_size = out_file.stat().st_size
                    rel_path = str(out_file.relative_to(Path.cwd())) \
                        if out_file.is_relative_to(Path.cwd()) else str(out_file)
                except Exception as e:
                    rel_path = f"save_failed:{str(e)[:40]}"

            # Metrics
            dev = {}
            if args.samples > 0 and len(mesh.faces) > 0:
                try:
                    dev = lowpoly.measure_deviation(original, mesh, samples=args.samples)
                except Exception as e:
                    dev = {"error": str(e)[:40]}

            v_before, f_before = len(original.vertices), len(original.faces)
            v_after, f_after = len(mesh.vertices), len(mesh.faces)
            d_ratio = (1 - f_after / max(f_before, 1)) * 100

            dev_str = ""
            if "error" in dev:
                dev_str = f"dev_err:{dev['error']}"
            elif dev:
                dev_str = (
                    f"{dev['hausdorff_rel']*100:>7.2f}%  "
                    f"{dev['p95_rel']*100:>7.2f}%  "
                    f"{dev['mean_rel']*100:>7.2f}%"
                )
            else:
                dev_str = "—          —          —         "

            notes = ""
            if res.classifier_ratio is not None:
                notes += f"auto→{res.mode_used} (ratio={res.classifier_ratio*100:.0f}%)  "
            if res.planar_stats:
                s = res.planar_stats
                notes += (f"planar:{s.n_regions_retriangulated}/{s.n_regions_kept}/"
                          f"{s.n_regions_total} regions"
                          f" ({s.area_retriangulated/max(s.area_total,1e-9)*100:.0f}% area)")
                if s.fallback_regions:
                    notes += f" fallback={s.fallback_regions}"
            if rel_path:
                notes += f"  → {rel_path}"

            row = (
                f"{mode:<10} {level:<9} "
                f"{v_after:<9} {f_after:<8} "
                f"-{d_ratio:>5.1f}%   "
                f"{human_bytes(saved_size) if saved_size >= 0 else '—':<10} "
                f"{total_elapsed:>5.1f}s  "
                f"{dev_str}  {notes}"
            )
            print(row, flush=True)
            rows.append(row)

    print("-" * len(header))
    print(f"\nOutput dir: {out_dir}")
    print("Columns: dev_* = sample-based distance to original, "
          "normalized by bbox diagonal (% of bbox_diag).")
    print("")
    print("Guide:")
    print("  quadric  — universal, uniform reduction; best dev_p95/F ratio")
    print("             but scatters many small triangles across flat faces.")
    print("  planar   — consolidates large coplanar regions into few big")
    print("             triangles (best for CAD / architectural). Keeps")
    print("             curved regions original so face count stays higher.")
    print("  hybrid   — planar + quadric on top. Experimental: quadric tends")
    print("             to damage the clean planar work at region boundaries.")
    print("  iso      — uniform-edge remeshing with feature-edge preservation.")
    print("             CAD-friendly look, well-shaped triangles everywhere.")
    print("  auto     — dispatches to planar (hard-surface) or quadric (organic).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
