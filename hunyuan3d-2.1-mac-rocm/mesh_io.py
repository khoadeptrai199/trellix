#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blender-free mesh I/O for the Hunyuan3D paint pipeline.

`hy3dpaint/DifferentiableRenderer/mesh_utils.py` does `import bpy` at module
scope, and `MeshRender` imports `load_mesh` and `save_mesh` from it inside a
bare `try/except` that only prints "Bpy IO CAN NOT BE Imported!!!". Without
Blender those two names are simply undefined, so the paint pipeline dies with a
NameError at its *first* step — `self.render.load_mesh(mesh=mesh)` — not at save
time. That is why the stub below is required, not merely convenient.

What actually needs Blender is narrower than it looks: `save_obj_mesh` (and thus
`save_mesh`) and `load_mesh` are pure numpy + cv2. Only `convert_obj_to_glb`
drives Blender. So we stub `bpy`, keep upstream's OBJ/MTL/texture writer
verbatim, and supply our own GLB exporter.

The GLB path is ours rather than upstream's `convert_utils.create_glb_with_pbr_materials`
for two concrete reasons: that function writes `temp.glb` and `mr_combined.png`
into the *current working directory*, and it labels every embedded image as
`image/png` even though the textures upstream just wrote are JPEG. Building the
GLB through trimesh's PBRMaterial avoids both and needs no temp files.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np

_BPY_STUBBED = False


def install_bpy_stub() -> bool:
    """Make `import bpy` succeed so `mesh_utils` is importable.

    Returns True if a stub was installed, False if a real Blender module is
    already available (in which case we leave it alone — upstream's GLB path
    would then work too, though we still prefer our own).

    Must be called before anything imports
    `hy3dpaint.DifferentiableRenderer.mesh_utils`.
    """
    global _BPY_STUBBED
    if "bpy" in sys.modules:
        return _BPY_STUBBED
    try:
        import bpy  # noqa: F401

        return False
    except ImportError:
        pass

    class _Stub(types.ModuleType):
        """Returns further stubs for any attribute access.

        mesh_utils only touches `bpy` inside the GLB conversion helpers, which we
        never call, so nothing has to behave correctly — it only has to import.
        Anything that *does* reach through here raises, loudly and specifically,
        instead of failing somewhere confusing later.
        """

        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            raise RuntimeError(
                f"bpy.{name} was accessed, but Blender is not installed. "
                f"mesh_io provides a Blender-free path for everything this port "
                f"uses; something called into upstream's bpy-based code."
            )

    sys.modules["bpy"] = _Stub("bpy")
    _BPY_STUBBED = True
    print("[mesh_io] bpy is absent — installed an import stub (Blender is not needed)", flush=True)
    return True


def _to_uint8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Upstream hands textures over as float arrays in [0, 1]."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def pack_metallic_roughness(
    metallic: Optional[np.ndarray], roughness: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Pack into the glTF metallicRoughness layout: R=AO, G=roughness, B=metallic.

    `MeshRender.get_texture_mr` returns both channels already broadcast to 3
    channels, so only one channel of each is meaningful.
    """
    m = _to_uint8(metallic)
    r = _to_uint8(roughness)
    if m is None and r is None:
        return None
    ref = m if m is not None else r
    h, w = ref.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[..., 0] = 255  # ambient occlusion: unused, fully lit
    if r is not None:
        out[..., 1] = r[..., 0] if r.ndim == 3 else r
    if m is not None:
        out[..., 2] = m[..., 0] if m.ndim == 3 else m
    return out


def export_glb_pbr(
    vtx_pos: np.ndarray,
    pos_idx: np.ndarray,
    vtx_uv: np.ndarray,
    albedo: np.ndarray,
    out_path: str,
    metallic: Optional[np.ndarray] = None,
    roughness: Optional[np.ndarray] = None,
    normal: Optional[np.ndarray] = None,
) -> str:
    """Write a GLB with a full PBR material.

    UVs are per-vertex here: the paint pipeline runs `mesh_uv_wrap` (xatlas),
    which remaps vertices so that `uv_idx == pos_idx`, and upstream's
    `load_mesh` relies on exactly that. So one index buffer is correct and no
    vertex splitting is needed.

    `vtx_uv` must already be in glTF convention (V measured from the top). Note
    that `MeshRender.get_mesh` does NOT return that — it flips V for OBJ — so see
    `save_textured_mesh`, which converts before calling this.
    """
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vtx_pos, dtype=np.float64),
        faces=np.asarray(pos_idx, dtype=np.int64),
        process=False,
        validate=False,
    )

    kwargs = {}
    base = _to_uint8(albedo)
    if base is not None:
        kwargs["baseColorTexture"] = Image.fromarray(base[..., :3])
    mr = pack_metallic_roughness(metallic, roughness)
    if mr is not None:
        kwargs["metallicRoughnessTexture"] = Image.fromarray(mr)
        # With a metallicRoughness texture present the factors must be 1.0 or
        # they scale the sampled values down.
        kwargs["metallicFactor"] = 1.0
        kwargs["roughnessFactor"] = 1.0
    nrm = _to_uint8(normal)
    if nrm is not None:
        kwargs["normalTexture"] = Image.fromarray(nrm[..., :3])

    material = PBRMaterial(name="PBR_Material", **kwargs)
    mesh.visual = TextureVisuals(uv=np.asarray(vtx_uv, dtype=np.float64), material=material)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    return out_path


def save_textured_mesh(render, obj_path: str, glb_path: Optional[str] = None,
                       downsample: bool = True) -> dict:
    """Save a textured MeshRender result as OBJ (+MTL +maps) and GLB.

    Mirrors `MeshRender.save_mesh` — including its texture downsampling — but
    routes the geometry and maps through our exporters instead of Blender.

    Returns {"obj": path, "glb": path or None, "maps": {...}}.
    """
    import cv2

    vtx_pos, pos_idx, vtx_uv, uv_idx = render.get_mesh(normalize=False)
    albedo = render.get_texture()
    metallic, roughness = render.get_texture_mr()
    normal = render.get_texture_normal()

    if downsample:
        def half(img):
            if img is None:
                return None
            return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

        albedo = half(albedo)
        metallic = half(metallic)
        roughness = half(roughness)
        normal = half(normal)

    # OBJ + MTL + texture files: upstream's writer, which is bpy-free.
    from DifferentiableRenderer.mesh_utils import save_obj_mesh

    Path(obj_path).parent.mkdir(parents=True, exist_ok=True)
    save_obj_mesh(
        obj_path, vtx_pos, pos_idx, vtx_uv, uv_idx, albedo,
        metallic=metallic, roughness=roughness, normal=normal,
    )

    base = os.path.splitext(obj_path)[0]
    maps = {"albedo": f"{base}.jpg"}
    if metallic is not None:
        maps["metallic"] = f"{base}_metallic.jpg"
    if roughness is not None:
        maps["roughness"] = f"{base}_roughness.jpg"
    if normal is not None:
        maps["normal"] = f"{base}_normal.jpg"
    maps = {k: v for k, v in maps.items() if os.path.exists(v)}

    result = {"obj": obj_path, "glb": None, "maps": maps}

    if glb_path:
        # `MeshRender.get_mesh` returns V flipped for the OBJ convention
        # (`vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]`, MeshRender.py:841), where V is
        # measured from the bottom. glTF measures it from the top, so it has to be
        # flipped back — otherwise every UV island samples a different island of
        # the atlas. With xatlas's chaotic packing that does not look like a
        # mirrored texture, it looks like colour noise smeared over the model,
        # which is easy to misread as a corrupted bake.
        uv_gltf = np.asarray(vtx_uv, dtype=np.float64).copy()
        uv_gltf[:, 1] = 1.0 - uv_gltf[:, 1]
        result["glb"] = export_glb_pbr(
            vtx_pos, pos_idx, uv_gltf, albedo, glb_path,
            metallic=metallic, roughness=roughness, normal=normal,
        )
    return result
