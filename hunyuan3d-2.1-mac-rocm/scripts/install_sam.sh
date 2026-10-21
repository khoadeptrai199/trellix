#!/usr/bin/env bash
# Install the optional 2D SAM proposals model into this project's venv.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/venv/bin/python"
CHECKPOINT="${PROJECT_DIR}/weights/sam_vit_b_01ec64.pth"
EXPECTED_SHA="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Hunyuan venv: run install.sh first." >&2
    exit 1
fi

"$PYTHON" -m pip install \
    'git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf'

if [[ ! -f "$CHECKPOINT" ]]; then
    mkdir -p "$(dirname "$CHECKPOINT")"
    curl --location --fail --retry 3 \
        --output "${CHECKPOINT}.download" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    mv "${CHECKPOINT}.download" "$CHECKPOINT"
fi

"$PYTHON" - "$CHECKPOINT" "$EXPECTED_SHA" <<'PY'
import hashlib
import sys

path, expected = sys.argv[1:]
digest = hashlib.sha256()
with open(path, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected:
    raise SystemExit("SAM checkpoint checksum mismatch; remove it and rerun this script.")
print("SAM ViT-B ready:", path)
PY
