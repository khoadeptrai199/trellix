"""
Verifies the decimate_pro() fallback fix (trellis/utils/postprocessing_utils.py's postprocess_mesh)
against the real dragon mesh that exposed the bug (outputs/dragon_raw_mesh.npz -- vtkQuadricDecimation
silently reduced it 658060 -> 658010 faces, 0.0%, see scripts/diagnose_decimate_dragon.py).
CPU-only, no pipeline/GPU needed.
"""
import sys
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import numpy as np
from trellis.utils.postprocessing_utils import postprocess_mesh

data = np.load("/Users/vinayak/Downloads/TRELLIS-main/outputs/dragon_raw_mesh.npz")
vertices, faces = data['vertices'], data['faces']
print(f"input: {vertices.shape[0]}v / {faces.shape[0]}f")

v_out, f_out = postprocess_mesh(
    vertices, faces,
    simplify=True, simplify_ratio=0.95,
    fill_holes=False,  # isolate the decimate step only -- fast, no rasterization needed
    verbose=True,
    device='cpu',
)
print(f"\noutput: {v_out.shape[0]}v / {f_out.shape[0]}f "
      f"({100*(1 - f_out.shape[0]/faces.shape[0]):.1f}% reduced)")
ok = f_out.shape[0] < faces.shape[0] * 0.5
print("PASS" if ok else "FAIL -- still not meaningfully reduced")
sys.exit(0 if ok else 1)
