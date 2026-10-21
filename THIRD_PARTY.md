# Upstream sources

This repository vendors source snapshots with local modifications. Original license and notice files remain inside each project directory.

| Directory | Upstream | Snapshot commit |
|---|---|---|
| `hunyuan3d-2.1-mac-rocm/` | https://github.com/VladimirTalyzin/hunyuan3d-2.1-mac-rocm | `6f4b63bd6419406f40291bca23a7c849bcaae2a3` |
| `trellis-mac-mps/` | https://github.com/vinayapathak/trellis-mac-mps | `f7c175a8620c2c23923d5b6fca3bf7f53604335f` |

Hunyuan3D's Tencent source and pretrained weights are downloaded during setup and are not bundled here. Optional SAM source and weights are also installed separately. Refer to their upstream licenses and model terms.

Run Git operations from the `trellix` root. The two directories above are regular tracked source directories, not Git submodules. Existing nested Git metadata was backed up outside this working tree during import; it is not distributed with the source.
