# Hướng dẫn Windows: từ clone code đến Split ảnh và tạo 3D

Cập nhật: **21/10/2026**. Dành cho **Windows 11 x64**, PowerShell, Python **3.12 x64**. Chạy trực tiếp trên Windows, không cần WSL.

**Mức kiểm chứng:** đã đối chiếu đường dẫn, tham số và code của checkout này; Split/SAM và shape đã chạy trên máy Mac của dự án. Quy trình bên dưới **chưa được chạy end-to-end trên Windows**. Các kết quả Windows được nhắc trong README gốc là báo cáo của tác giả upstream. Mở được UI chưa có nghĩa là model đã chạy thành công: phải hoàn thành bước tạo và mở GLB ở mục 8.

## 1. Chọn đúng bản code và cấu hình

### Bản code cần clone

Clone **[khoadeptrai199/trellix](https://github.com/khoadeptrai199/trellix)**, branch `main`. Repo hiện là private: tài khoản GitHub của người nhận cần được cấp quyền truy cập trước khi clone.

Source Hunyuan có tính năng Split/SAM nằm trong thư mục **`hunyuan3d-2.1-mac-rocm/`**. Tất cả lệnh cài và chạy bên dưới thực hiện từ thư mục con này, sau bước clone.

```text
trellix/
  hunyuan3d-2.1-mac-rocm/
    gradio_app.py
    image_parts.py
    sam_parts.py
    launch.ps1
    requirements_rocm.txt
    scripts/install_sam.ps1
    docs/HUONG_DAN_WINDOWS.md
  trellis-mac-mps/
  mac_setup/
```

Hai dự án được lưu dưới dạng source trong repo cha, không cần `git submodule update`. Không dùng URL `Tencent-Hunyuan/Hunyuan3D-2.1` làm repo chính: đó là repo model sẽ được tải vào thư mục con ở mục 4.

### Phần cứng

| Máy Windows | Nhánh cài | Phạm vi |
|---|---|---|
| NVIDIA có CUDA, nên có từ 12–16 GB VRAM | 3A | Lựa chọn ưu tiên cho shape; texture cần thêm bộ nhớ |
| AMD nằm trong danh sách hỗ trợ Windows của AMD | 3B | Cần đúng GPU, driver, ROCm và Python; không áp dụng cho mọi Radeon |
| Không có GPU được hỗ trợ / Intel GPU | 3C | CPU: dùng thử UI/Split, tạo 3D rất chậm |
| Windows ARM / Snapdragon | Ngoài phạm vi guide | Các wheel x64 bên dưới không phải bản native ARM |

Nên có **32 GB RAM**, SSD còn **40 GB trống cho shape + SAM**, **80 GB nếu thử texture**, mạng ổn định khi cài lần đầu. Đây là mức dự trù thực tế, không phải mức bảo đảm mọi preset chạy được. Tencent công bố khoảng 10 GB VRAM cho shape, 21 GB cho texture; lượng dùng của port phụ thuộc preset và backend. Xem [yêu cầu của Tencent](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1#-models-zoo).

### Phần mềm cần có

1. [Git for Windows](https://git-scm.com/downloads/win).
2. [Python 3.12 x64](https://www.python.org/downloads/windows/), cài Python Launcher (`py`). Nếu chọn bản tải thủ công, dùng bản có Windows installer.
3. Driver GPU tương ứng: NVIDIA cho 3A; đúng driver mà AMD chỉ định cho 3B.
4. PowerShell 5.1 có sẵn trên Windows hoặc PowerShell 7. Mở terminal thường; chỉ installer hệ thống có thể cần quyền admin.

Kiểm tra sau khi cài và mở lại terminal:

```powershell
git --version
py -3.12 --version
py -3.12 -c "import struct; print(struct.calcsize('P') * 8)"
```

Kết quả: Git có version, Python `3.12.x`, dòng cuối `64`. Các lệnh trong guide dùng **PowerShell**, không dán nguyên vào CMD/Git Bash. Chạy từng khối; nếu có lỗi thì sửa trước khi đi tiếp.

## 2. Clone và tạo môi trường riêng

Dùng tài khoản có quyền đọc repo private. Nếu Git yêu cầu đăng nhập, hoàn thành đăng nhập GitHub qua credential manager; không dùng mật khẩu tài khoản làm token.

```powershell
$RepoUrl = 'https://github.com/khoadeptrai199/trellix.git'
$RepoBranch = 'main'
$RepoRoot = Join-Path $env:USERPROFILE 'AI\trellix'
$Project = Join-Path $RepoRoot 'hunyuan3d-2.1-mac-rocm'
New-Item -ItemType Directory -Force (Split-Path -Parent $RepoRoot) | Out-Null
git clone --branch $RepoBranch $RepoUrl $RepoRoot
if ($LASTEXITCODE -ne 0) { throw 'Clone failed: check repo URL and branch.' }
Set-Location $Project

$Required = @('gradio_app.py', 'image_parts.py', 'sam_parts.py', 'launch.ps1', 'scripts/install_sam.ps1')
foreach ($File in $Required) {
    if (-not (Test-Path $File)) { throw "Wrong checkout: missing $File" }
}
git rev-parse HEAD

py -3.12 -m venv venv
if ($LASTEXITCODE -ne 0) { throw 'Cannot create venv.' }
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'Cannot install pip tooling.' }
```

Các lệnh sau giả định bạn đang ở thư mục này. Guide gọi thẳng `venv\Scripts\python.exe`, không cần `Activate.ps1` và không cài package vào Python hệ thống. Đừng copy `venv` từ Mac sang Windows. Chọn thư mục local ngắn; tránh OneDrive để các file model không bị đồng bộ ngoài ý muốn.

## 3. Cài PyTorch — chỉ chọn MỘT nhánh

### 3A. NVIDIA / CUDA

Kiểm tra driver:

```powershell
nvidia-smi
```

Phải thấy tên NVIDIA GPU và driver. Sau đó cài một bộ version cố định có wheel Windows chính thức:

```powershell
.\venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
.\venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert not torch.version.hip; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.ones(32,32,device='cuda'); assert (x@x).mean().item()==32; print('GPU arithmetic OK')"
if ($LASTEXITCODE -ne 0) { throw 'Fix CUDA/driver before continuing.' }
```

Kết quả cần có: `2.8.0+cu128`, tên GPU và `GPU arithmetic OK`. Bộ version này có trong [danh sách wheel chính thức của PyTorch](https://pytorch.org/get-started/previous-versions/); đây chưa phải lockfile Windows đã được dự án kiểm thử. Nếu GPU/driver không hỗ trợ bộ này, chọn cặp torch/torchvision khác theo tài liệu PyTorch và chạy lại phép thử. Cài CUDA Toolkit riêng chỉ cần khi build CUDA extension; quy trình shape ban đầu dùng wheel có sẵn.

### 3B. AMD / ROCm trên Windows

Đọc đúng [matrix Radeon Windows](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html) hoặc [matrix Ryzen Windows](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html). Cùng kiến trúc `gfx` không đủ để kết luận một model GPU được hỗ trợ.

Ví dụ cố định dưới đây dành cho **Radeon thuộc matrix ROCm 7.2.1**, **Python 3.12 x64**, **driver 26.2.2** theo [hướng dẫn cài AMD 7.2.1](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html). Nếu tài liệu AMD đã chuyển version, dùng trọn bộ tương ứng cùng một release. Với Ryzen APU, theo trang cài Ryzen được liên kết từ matrix, không mặc định dùng bộ Radeon này.

```powershell
$AmdRelease = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1'
.\venv\Scripts\python.exe -m pip install --no-cache-dir "$AmdRelease/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" "$AmdRelease/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" "$AmdRelease/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" "$AmdRelease/rocm-7.2.1.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'ROCm runtime installation failed.' }
.\venv\Scripts\python.exe -m pip install --no-cache-dir "$AmdRelease/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$AmdRelease/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$AmdRelease/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
if ($LASTEXITCODE -ne 0) { throw 'ROCm PyTorch installation failed.' }
.\venv\Scripts\python.exe -c "import torch; assert torch.version.hip, 'Not a ROCm build'; assert torch.cuda.is_available(), 'AMD GPU unavailable'; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0)); x=torch.ones(32,32,device='cuda'); assert (x@x).mean().item()==32; print('GPU arithmetic OK')"
if ($LASTEXITCODE -ne 0) { throw 'Fix ROCm/driver before continuing.' }
```

ROCm cũng dùng API `torch.cuda` và device `cuda`; đây là bình thường. Đừng cài wheel ROCm dành cho Linux trên Windows. Script cũ `install_rocm_windows.ps1` có gợi ý nightly dành riêng cho một số GPU; guide này dùng các bước tường minh để tránh chọn nhầm wheel hoặc ghi đè môi trường NVIDIA. GPU không có trong matrix cần quy trình thử nghiệm riêng hoặc nhánh CPU.

### 3C. CPU

```powershell
.\venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__); x=torch.ones(32,32); assert (x@x).mean().item()==32; print('CPU arithmetic OK')"
```

CPU chạy UI, connected regions và SAM được về mặt code, nhưng thời gian inference phụ thuộc máy. Không dùng thời gian đo trên Mac/GPU làm dự báo cho CPU.

## 4. Cài dependencies và source model

Áp dụng cho cả ba nhánh. Ghi lại version torch đang dùng thành constraints để pip báo xung đột thay vì tự thay runtime GPU:

```powershell
New-Item -ItemType Directory -Force .cache | Out-Null
.\venv\Scripts\python.exe -c "from importlib.metadata import version; from pathlib import Path; Path('.cache/torch-constraints.txt').write_text('\n'.join(n+'=='+version(n) for n in ('torch','torchvision','torchaudio'))+'\n', encoding='utf-8')"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch is incomplete.' }
.\venv\Scripts\python.exe -m pip install -c .cache\torch-constraints.txt -r requirements_rocm.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Do not start the app yet.' }
.\venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Resolve dependency conflicts first.' }
```

Tên `requirements_rocm.txt` là tên hiện có trong port; file này chứa dependencies chung và **không cài torch**. Không thay bằng toàn bộ `requirements.txt` của Tencent vì các package/build CUDA bổ sung không cần cho luồng shape của port.

Clone source Tencent vào đúng thư mục con. Checkout commit đang dùng trong lần chạy shape local đã thành công để hạn chế thay đổi upstream:

```powershell
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git Hunyuan3D-2.1
if ($LASTEXITCODE -ne 0) { throw 'Upstream clone failed.' }
git -C Hunyuan3D-2.1 checkout 82920d643c0dc2f7bfd7255f45f62d386edfe60c
if ($LASTEXITCODE -ne 0) { throw 'Cannot select upstream revision.' }
```

`detached HEAD` ở thư mục con là bình thường khi pin commit. Nếu thư mục đã tồn tại, kiểm tra `git -C Hunyuan3D-2.1 status` và revision; bỏ qua clone, không xóa công việc đang có trong đó.

## 5. Tải model shape

```powershell
.\venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2.1', local_dir='weights/Hunyuan3D-2.1', allow_patterns=['hunyuan3d-dit-v2-1/*','hunyuan3d-vae-v2-1/*','README*'])"
if ($LASTEXITCODE -ne 0) { throw 'Shape download failed; rerun after fixing the connection/authentication.' }
```

Download nhiều GB, có thể tiếp tục bằng cách chạy lại lệnh khi mạng gián đoạn. Model dùng `.ckpt`, không tự đổi sang `.safetensors`. Cấu trúc cần có:

```text
weights/Hunyuan3D-2.1/
  hunyuan3d-dit-v2-1/config.yaml
  hunyuan3d-dit-v2-1/model.fp16.ckpt
  hunyuan3d-vae-v2-1/...
Hunyuan3D-2.1/hy3dshape/hy3dshape/pipelines.py
```

Nếu gặp `401/403`, mở trang model trên Hugging Face để kiểm tra quyền truy cập/điều khoản, rồi đăng nhập từ thư mục dự án:

```powershell
.\venv\Scripts\hf.exe auth login
```

Sau đó chạy lại download. Không cần đưa token vào code. Lần đầu xóa nền còn có thể tải model `rembg` vào cache người dùng; giữ mạng trong lần thử đầu.

## 6. Cài SAM để Split vật thể liền nhau

Bỏ qua mục này nếu chỉ cần tạo 3D hoặc tách các vùng rời nhau.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_sam.ps1
if ($LASTEXITCODE -ne 0) { throw 'SAM setup failed.' }
```

Script cài source [Meta Segment Anything](https://github.com/facebookresearch/segment-anything) ở commit cố định, tải ViT-B khoảng 375 MB và xác minh SHA256. Kết quả cần thấy `SAM import OK` và `SAM ready; checkpoint SHA256 verified`. File đích: `weights/sam_vit_b_01ec64.pth`. Script dùng venv của dự án và không tự thay torch. `-ExecutionPolicy Bypass` chỉ áp dụng cho process PowerShell này; không cần đổi policy toàn máy.

## 7. Kiểm tra và mở UI

```powershell
.\venv\Scripts\python.exe scripts\doctor.py
```

Cần kiểm tra:

- `compute backend`: `cuda` cho NVIDIA, `rocm` cho AMD, `cpu` cho nhánh CPU.
- `import hy3dshape.pipelines`, source và weights shape: `OK`.
- `gpu arithmetic`: `OK` nếu dùng GPU.
- `shape generation`: `OK` với GPU; CPU có thể là `WARN` do tốc độ.
- `texturing (PBR): FAIL` và thiếu `mesh_inpaint_processor` là dự kiến khi chỉ cài shape. Không cần sửa phần texture để dùng Split hoặc Generate shape.

Doctor chủ yếu kiểm tra imports, file và phép tính; kết quả `OK` không thay thế chạy inference thật.

**NVIDIA:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1 -Backend cuda -Raster torch -Port 7860
```

**AMD:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1 -Backend rocm -Raster torch -Port 7860
```

**CPU:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1 -Backend cpu -Raster torch -Port 7860
```

Mở **http://127.0.0.1:7860** khi terminal báo Gradio đang chạy. Giữ terminal mở. Model shape được load khi bấm Generate lần đầu nên lần đầu có thể chậm hơn. `-Raster torch` dùng renderer có sẵn, giúp quy trình ban đầu không phụ thuộc build CUDA extension.

Mở PowerShell thứ hai để kiểm tra HTTP:

```powershell
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7860).StatusCode
```

Kết quả mong đợi `200`. App mặc định chỉ nghe `127.0.0.1`, truy cập từ chính máy đang chạy.

## 8. Thử tính năng và xác định “chạy thành công”

Chọn English trong UI để tên nút khớp guide.

### Split ảnh 2D

1. Mở **Split 2D image into parts**.
2. Upload ảnh gốc vào **Image to split (upload here)**.
3. Chọn **Connected regions (fast)** cho các vật thể/vùng rời nhau; chọn **SAM AI candidates (GPU/CPU)** cho các bộ phận chạm nhau.
4. Giữ `Minimum area per part` ở `100`, bấm **Split image**.
5. Kiểm tra ảnh trong **2D parts**, tải ZIP, mở PNG kiểm tra vùng trong suốt. Output được lưu ở `outputs/image_parts_*/`.
6. Chọn một part, bấm **Use selected part as input image**, kiểm tra ảnh input phía dưới trước khi Generate.

**Kết quả thực tế cần hiểu:** connected regions không tự hiểu tóc/đầu/tay/chân. SAM đề xuất vùng thị giác, có thể chồng lên nhau, thiếu bộ phận hoặc gộp nhiều bộ phận. Không có cam kết tự tạo đủ các nhãn ngữ nghĩa đó.

Nếu cần chọn chính xác vùng: mở **Advanced: color mask for touching parts**, upload PNG mask **cùng kích thước ảnh gốc**; nền đen, mỗi part một màu RGB đặc khác nhau. Không dùng JPEG, gradient hay chống răng cưa cho mask. Không upload ảnh gốc vào ô mask. Bỏ mask cũ khi đổi ảnh hoặc muốn thử SAM tự động.

### Tạo model 3D nguyên khối

1. Upload ảnh đầy đủ vào **Input image (object on a plain background)**. Nếu vừa chọn một crop, thay lại bằng ảnh đầy đủ.
2. Bật xóa nền nếu ảnh còn nền. Chọn ảnh rõ, không cắt mất chân/tóc; lần đầu có thể dùng `assets/examples/robot.jpg`.
3. Mở tham số: **10 steps**, **octree 128**, **guidance 5**, **seed 42** để thử pipeline.
4. Bấm Generate, chờ log hoàn thành, xoay model ở preview và tải GLB.
5. Sau khi chạy thành công, thử **30 steps / octree 192**; tăng lên **50 / 256** nếu đủ bộ nhớ. Số steps chủ yếu ảnh hưởng thời gian; giảm octree giúp giảm tải bước trích mesh.

Shape tạo hình học, không tự có màu/texture. AI suy đoán mặt sau từ một ảnh nên không thể bảo đảm đúng từng chi tiết mẫu. Generate mỗi crop riêng cũng không bảo đảm các mesh cùng tỷ lệ, khớp nhau hay có mộng nối.

**Điều kiện hoàn thành:** log không có traceback, có GLB tải được, mở được trong trình xem 3D/Blender và thấy hình học hợp lý. Chất lượng hình học và cài đặt thành công là hai tiêu chí riêng.

### Chạy kiểm tra bằng CLI (tùy chọn)

Dừng UI trước để tránh load hai model cùng lúc. Từ thư mục dự án:

```powershell
.\venv\Scripts\python.exe tests\e2e_visual.py --image assets\examples\robot.jpg --skip-texture --steps 10 --octree 128 --views 4 --resolution 256 --out outputs\windows_smoke
if ($LASTEXITCODE -ne 0) { throw 'Inference or rendering failed; inspect the report/log.' }
Start-Process .\outputs\windows_smoke\report.html
```

CLI tự phát hiện backend. Đọc trạng thái từng stage trong report và mở GLB; nếu preview bị thu nhỏ do mảnh mesh xa vật thể, kiểm tra model trong Blender trước khi kết luận mất hình học. Lệnh này có chạy model thật, không chỉ kiểm tra cài đặt.

## 9. Xuất GLB/STL và đưa vào Bambu Studio

1. Trong **Download mesh**, chọn `glb`, bấm **Save and download** để giữ bản gốc.
2. Có thể dùng post-processing của UI để giảm polygon; so sánh với bản gốc vì có thể mất chi tiết.
3. Chọn `stl`, bấm **Save and download** rồi tải file. STL không giữ màu/texture.
4. Import vào Bambu Studio, đặt kích thước theo mm, ví dụ cao 80 mm. Mesh AI không có kích thước vật lý đáng tin mặc định.
5. Kiểm tra các mảnh rời, mặt hở, thành mỏng, overhang; repair nếu cần, chọn hướng in và support.
6. Slice và xem từng lớp trước khi in. `watertight=True` riêng lẻ chưa bảo đảm in được hoặc đủ độ bền.

Split PNG chỉ là crop 2D. Muốn mesh 3D có các phần lắp ráp cần bước phân đoạn/chỉnh sửa 3D khác; repo này chưa tích hợp pipeline Hunyuan3D-Part hoàn chỉnh.

## 10. Texture PBR (tùy chọn, làm sau khi shape chạy được)

Texture cần thêm model, native extension và bộ nhớ. Dừng UI. Cài [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022), chọn workload **Desktop development with C++**, MSVC x64 và Windows SDK. Mở lại PowerShell tại thư mục dự án.

```powershell
.\venv\Scripts\python.exe scripts\build_extensions.py --raster torch
if ($LASTEXITCODE -ne 0) { throw 'Extension build failed; read the compiler output.' }
.\venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2.1', local_dir='weights/Hunyuan3D-2.1', allow_patterns=['hunyuan3d-paintpbr-v2-1/*'])"
if ($LASTEXITCODE -ne 0) { throw 'Paint model download failed.' }
New-Item -ItemType Directory -Force .\Hunyuan3D-2.1\hy3dpaint\ckpt | Out-Null
Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth' -OutFile .\Hunyuan3D-2.1\hy3dpaint\ckpt\RealESRGAN_x4plus.pth
.\venv\Scripts\python.exe scripts\doctor.py
```

`--raster torch` bỏ build rasterizer native, nhưng vẫn build `mesh_inpaint_processor`. Nếu thiếu compiler thì mục này chưa hoàn tất. Paint còn có thể tải DINOv2 và dependency model vào Hugging Face cache khi chạy lần đầu; cần mạng và dung lượng dự phòng.

Chỉ thử khi doctor báo texture khả dụng. Start lại UI, Generate shape, rồi chạy texture với **Safe**, 6 views / 256 px. Không tăng preset chỉ vì model load được. Nếu preflight từ chối bộ nhớ, giữ nguyên giới hạn và hạ cấu hình. Texture bổ sung màu/vật liệu, không sửa đảm bảo hình học sai so với ảnh gốc.

## 11. Dùng hằng ngày: start, stop, đổi port

Không cần cài lại hay tải model mỗi lần. Ví dụ NVIDIA:

```powershell
Set-Location (Join-Path $env:USERPROFILE 'AI\trellix\hunyuan3d-2.1-mac-rocm')
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1 -Backend cuda -Raster torch -Port 7860
```

AMD đổi `cuda` thành `rocm`; CPU đổi thành `cpu`. Các cờ backend của launcher điều khiển pipeline chính; SAM hiện tự chọn GPU khả dụng trước CPU.

**Dừng bình thường:** quay lại terminal chạy app, nhấn **Ctrl+C**, chờ terminal trả prompt và port được giải phóng. Đóng tab trình duyệt không dừng Python. Output đã lưu trên đĩa vẫn còn; trạng thái chọn mesh của phiên UI có thể mất sau reload/restart.

**Port bận:** dùng `-Port 7861`, mở `http://127.0.0.1:7861`. Hoặc kiểm tra tiến trình đang nghe:

```powershell
$Listeners = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue
foreach ($Listener in $Listeners) {
    Get-CimInstance Win32_Process -Filter "ProcessId = $($Listener.OwningProcess)" |
        Select-Object ProcessId, ExecutablePath, CommandLine
}
```

Nếu mất terminal, chỉ sau khi xác nhận `ExecutablePath` là `venv\Scripts\python.exe` của đúng dự án và command có `gradio_app.py`, mới dừng PID đó:

```powershell
$AppProcessId = 12345  # Thay bang PID vua xac minh
Stop-Process -Id $AppProcessId
```

Đây là dừng cưỡng bức, tác vụ đang chạy sẽ mất. Không dùng `taskkill /IM python.exe /F` vì có thể tắt Python của dự án khác. Sau đó kiểm tra lại port rồi start một instance.

## 12. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `py` / Git không tìm thấy | Cài đúng phần mềm, mở terminal mới; dùng `py -3.12`, không dựa vào alias Python của Microsoft Store |
| Không có Split/SAM trong UI | Kiểm tra repo, branch, commit và các file ở mục 1; thay đổi local của người khác không có trong clone gốc |
| `running scripts is disabled` | Dùng nguyên lệnh `powershell -NoProfile -ExecutionPolicy Bypass -File ...`; policy do tổ chức quản lý có thể cần quản trị viên xử lý |
| `Torch not compiled with CUDA`, `cuda.is_available() == False` | Kiểm tra đúng venv, đúng nhánh torch, driver và GPU; chạy lại phép thử mục 3 |
| `DLL load failed` / `WinError 126` | Kiểm tra Python x64, driver; cài [Microsoft Visual C++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist); không tải DLL rời từ website lạ |
| pip báo cần Microsoft C++ Build Tools | Cài workload C++ như mục 10, mở lại terminal, chạy lại đúng lệnh pip bị lỗi |
| pip đổi torch hoặc `operator torchvision::nms does not exist` | Torch/torchvision sai cặp; cài lại trọn bộ của mục 3, tạo lại constraints rồi `pip check` |
| `Cannot import hy3dshape` | Kiểm tra source ở `Hunyuan3D-2.1\hy3dshape\hy3dshape`; đừng clone thành hai lớp thư mục trùng tên |
| Thiếu model / file `.ckpt` lỗi | Chạy lại download mục 5; chỉ có thư mục weights chưa chứng minh download đã đủ |
| `SAM weights are missing` / `SAM is not installed` | Chạy `scripts\install_sam.ps1`, kiểm tra checkpoint và đúng venv, restart UI |
| Split chỉ trả một part | Vật thể liền nhau: chuyển sang SAM; connected regions không tách ngữ nghĩa |
| SAM thiếu tay/chân hoặc có mask chồng | Giới hạn thuật toán; kiểm tra từng crop hoặc cung cấp color mask thủ công |
| CUDA/ROCm out of memory | Dừng job, restart app, đóng ứng dụng chiếm GPU, giảm octree và texture preset; chạy từng tác vụ |
| Trình duyệt báo refused / không mở được | Xem traceback terminal, kiểm tra port và HTTP; phải giữ tiến trình launcher chạy |
| `launch() got an unexpected keyword argument 'theme'` | Dùng checkout mới có theme trong `gr.Blocks`; giữ Gradio 5 theo requirements, không nâng Gradio 6 tùy ý |
| Màn hình chớp đen / driver reset | Dừng app, khởi động lại Windows, chạy doctor để kiểm tra phép tính GPU; giảm tải và kiểm tra driver trước khi chạy lại |
| Không có màu trong GLB/STL | Shape chỉ có hình học; cần PBR cho màu trong GLB, STL không lưu texture |

Không coi sửa registry TDR là bước cài mặc định. README upstream có gợi ý nâng timeout; [Microsoft dành các khóa TDR cho kiểm thử driver và không khuyến nghị người dùng cuối tự chỉnh](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys). Khi gặp reset, ưu tiên quy trình kiểm tra và giảm tải ở trên.

Khi báo lỗi, gửi command đã chạy, traceback, GPU/driver và các file chẩn đoán:

```powershell
New-Item -ItemType Directory -Force outputs\diagnostics | Out-Null
.\venv\Scripts\python.exe -m torch.utils.collect_env > outputs\diagnostics\environment.txt
.\venv\Scripts\python.exe scripts\doctor.py > outputs\diagnostics\doctor.txt
.\venv\Scripts\python.exe -m pip freeze > outputs\diagnostics\packages.txt
git rev-parse HEAD > outputs\diagnostics\app_commit.txt
git -C Hunyuan3D-2.1 rev-parse HEAD > outputs\diagnostics\upstream_commit.txt
```

## 13. Dành cho người bàn giao và cập nhật code

Trước khi gửi link clone:

- Commit/push các thay đổi từ repo cha `trellix`. Không khởi tạo thêm `.git` trong hai thư mục source. Kiểm tra `git diff` để giữ các thay đổi UI/Gradio cần thiết.
- Gửi URL, branch và commit cụ thể; cấp quyền đọc repo private cho người nhận trước khi họ clone.
- Không commit `venv/`, `weights/`, `outputs/`, cache, token hay ảnh riêng. Các file trong `outputs/character_test_20261020/` của máy phát triển không đi theo clone; guide không phụ thuộc chúng.
- Thử trên một máy Windows sạch: clone mới → mục 2–7 → Split connected → Split SAM → Generate → tải/mở GLB → xuất STL. Ghi GPU, driver, version packages và thời gian.
- Nếu muốn cam kết “clone và chạy được”, cần hoàn thành phép thử Windows này, sửa các lỗi thực tế và công bố bộ dependencies đã kiểm chứng. Các khoảng version trong requirements hiện tại chưa phải lockfile tái lập tuyệt đối.

Người nhận clone repo ở mục 2 rồi làm theo các mục 3–9; mục 10 là tùy chọn. Khi cập nhật, dừng app, kiểm tra `git status` từ repo cha rồi dùng `git pull --ff-only`; không ghi đè các thay đổi local chưa lưu.
