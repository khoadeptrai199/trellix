"""Convert a GLB triangle mesh into an STL with an explicit height in mm.

Preserves scene transforms, converts standard glTF Y-up to Z-up, scales, then
places the bottom at Z=0. This does not repair holes or guarantee printability.
Dependencies: numpy, trimesh >= 4.4.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height-mm", type=float, default=80.0)
    parser.add_argument("--source-up", choices=("y", "z"), default="y")
    args = parser.parse_args()
    import numpy as np
    import trimesh

    if args.input.suffix.lower() != ".glb" or not args.input.is_file():
        parser.error("Input must be an existing .glb file")
    if args.output.suffix.lower() != ".stl":
        parser.error("Output must end in .stl")
    if args.output.exists():
        parser.error("Output already exists; choose another filename")
    if not np.isfinite(args.height_mm) or args.height_mm <= 0:
        parser.error("--height-mm must be a finite positive number")
    scene = trimesh.load_scene(args.input)
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        parser.error("GLB does not contain a triangle mesh")
    if not np.isfinite(mesh.vertices).all():
        parser.error("Mesh has non-finite coordinates")
    if args.source_up == "y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    height = float(mesh.extents[2])
    if height <= 0:
        parser.error("Mesh height is zero")
    mesh.apply_scale(args.height_mm / height)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh.apply_translation([-center[0], -center[1], -mesh.bounds[0, 2]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    print(f"Saved: {args.output.resolve()}")
    print(f"Dimensions X/Y/Z (mm): {mesh.extents.round(3)}")
    print(f"Watertight: {mesh.is_watertight}; triangles: {len(mesh.faces)}")
    print("Check orientation, thickness, connected parts and sliced layers in Bambu Studio.")


if __name__ == "__main__":
    main()
