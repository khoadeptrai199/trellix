# Cài Hunyuan3D 2.1 và TRELLIS Mac trên M4 Pro

Ngày đối chiếu: 18/10/2026. Máy đã kiểm tra: M4 Pro 24 GB, macOS 15.1 ARM64,
Command Line Tools có sẵn. Homebrew ở `/opt/homebrew/bin/brew`, chưa nằm trong
PATH của phiên kiểm tra; Python hệ thống 3.9.6. Đã cài Python 3.12.15, Git LFS
và libusb qua Homebrew. Hai repo Mac đã clone trong workspace.

Đây là hướng dẫn cài và kiểm tra theo đúng máy này. Trạng thái xác minh thực tế ở
cuối tài liệu; chỉ coi một model chạy hoàn chỉnh khi lệnh tạo mẫu cho ra mesh.
Nếu cài lại trên máy khác, chạy từng khối trong Terminal và dừng khi có lỗi.
Hai dự án là bản port cộng đồng; phiên bản macOS dùng trong benchmark của tác giả
khác máy hiện tại.

## Kết quả kiểm tra trên máy này

| Repo | Đã chạy thật | Giới hạn hiện tại |
|---|---|---|
| Hunyuan Mac (`6f4b63b`) | `doctor.py` báo shape OK trên MPS; `tests/e2e_visual.py --skip-texture --steps 20 --octree 192` tạo mesh 88.980 đỉnh/177.960 mặt, render được; STL 80 mm đọc lại được, tọa độ hợp lệ và `watertight=True`; web UI trả HTTP 200 ở localhost:7860 | Chỉ cài shape weights; PBR chưa cài/chưa thử |
| TRELLIS Mac (`f7c175a`) | MPS matmul, kernel Metal và smoke test sparse đều qua; ảnh mẫu tạo GLB 648.667 mặt và STL 80 mm đọc lại được; web UI trả HTTP 200 ở localhost:5050 | GLB/STL mesh-only là `watertight=False`; chế độ GLB có texture mặc định hết MPS memory trên máy 24 GB |

Ngày 19/10/2026 đã kiểm tra lại Hunyuan: `doctor.py` báo shape OK trên MPS;
test ảnh mẫu 10 steps/octree 128 qua cả tạo hình và render (40.530 đỉnh,
81.064 mặt). GLB mới có tọa độ hữu hạn và `watertight=True`. Web UI khởi động
trên localhost:7860, trả HTTP 200 và có endpoint `/generate_3d`.
Một lượt tạo từ ảnh mèo với 50 steps/octree 256 đã lưu
`outputs/hunyuan3d_20261019_212625_raw.glb` (204.690 đỉnh, 411.600 mặt,
`watertight=True`). STL chuyển từ mesh này đọc lại được, nhưng vẫn cần xem
trong Bambu Studio trước khi in.

Hai môi trường đều qua `python -m pip check`. File
đầu ra thực tế nằm ở `hunyuan3d-2.1-mac-rocm/outputs/hunyuan_demo_80mm.stl` và
`trellis-mac-mps/outputs/dragon_80mm.stl`. Mẫu Hunyuan đã thử là STL kín; mỗi
ảnh mới vẫn cần kiểm tra. Với mẫu TRELLIS này, hãy sửa lỗ trong slicer hoặc phần
mềm sửa mesh trước khi in.

## 1. Chuẩn bị chung

Mở Terminal. Dùng thư mục gốc workspace hiện tại, không chạy trong repo TRELLIS cũ.

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install python@3.12 git-lfs libusb
/opt/homebrew/bin/python3.12 --version
uname -m
xcode-select -p
cd /Users/kmacbookpro/Desktop/trellix
```

Kết quả mong đợi: Python 3.12.x; `arm64`; đường dẫn CommandLineTools.
Nếu Homebrew yêu cầu cập nhật Command Line Tools, xử lý thông báo đó trước.
Dùng `venv` riêng cho từng repo; không cần cài Conda. Chuẩn bị thực tế khoảng
60–80 GB trống cho hai môi trường, weights, cache và output; đây là mức dự phòng,
không phải dung lượng tải chính xác. Máy hiện tại có đủ khoảng trống đó.

Cấu trúc sau khi làm xong:

```text
trellix/
  trellix/TRELLIS/                  repo CUDA đã có
  hunyuan3d-2.1-mac-rocm/           bản Mac đã clone, môi trường venv/
  trellis-mac-mps/                  bản Mac đã clone, môi trường .venv/
  mac_setup/                       hướng dẫn và script hỗ trợ
```

## 2. Hunyuan3D: cài phần tạo hình cho STL

### Tách một ảnh 2D thành nhiều ảnh bộ phận

Web UI Hunyuan3D local có mục **Split 2D image into parts** ở đầu trang. Chạy:

```bash
cd /Users/kmacbookpro/Desktop/trellix/hunyuan3d-2.1-mac-rocm
bash launch.sh --backend mps --port 7861
```

Mở `http://127.0.0.1:7861/`, mở **Split 2D image into parts**, tải ảnh gốc vào **Image to split (upload here)** rồi bấm **Split image**. App cũng chấp nhận ảnh trong ô **Input image** của luồng tạo 3D nếu ô trong khung tách ảnh đang trống. Chọn **Connected regions (fast)** để tách các vùng alpha rời nhau. Nếu ảnh có nền kín, app dùng `rembg` để lấy vật thể trước; các vùng dính nhau vẫn được xem là một phần. Các pixel alpha gần 0 do nhiễu xóa nền được bỏ qua.

Với ảnh một vật thể liền khối như áo giáp, chọn **SAM AI candidates (MPS/CPU)** trong **Split method**, rồi bấm **Split image**. SAM ViT-B tạo các vùng gợi ý từ ảnh; từng vùng có thể chồng lấn hoặc chưa đúng ý nghĩa bộ phận. Xem gallery, chọn vùng phù hợp trước khi tạo 3D. SAM đã được cài và chạy thử trên MPS của máy này. Nếu cần cài lại trong một venv mới, chạy:

```bash
cd /Users/kmacbookpro/Desktop/trellix/hunyuan3d-2.1-mac-rocm
bash scripts/install_sam.sh
```

Script tải checkpoint ViT-B khoảng 369 MB vào `weights/` và kiểm tra SHA-256. Lần đầu chạy SAM có thể chậm hơn chế độ vùng rời. Chỉ nên chạy một tác vụ nặng (SAM hoặc Hunyuan tạo hình) tại một thời điểm trên máy 24 GB.

Để chủ động tách các phần chạm nhau (ví dụ mái và thân nhà), vẽ thêm **color label mask** dạng PNG có **cùng kích thước pixel** với ảnh gốc: nền đen, mỗi bộ phận một màu RGB đặc khác nhau. Mở **Advanced: color mask for touching parts**, tải mask vào ô **Color label mask (not the original photo)**, rồi bấm **Split image**. Không tải ảnh gốc vào ô mask. Có thể tăng **Minimum area per part** để bỏ nhiễu. App tạo PNG trong suốt cho từng phần và file ZIP dưới `outputs/image_parts_*/`.

Chọn một mục trong **Part to use for 3D generation**, bấm **Use selected part as input image**, sau đó dùng **Generate 3D object** và xuất GLB/STL như quy trình bên dưới. Lặp lại cho từng phần. Đây là phân vùng **ảnh 2D**; các mesh 3D sinh riêng sẽ không tự căn đúng vị trí lắp ráp, tạo mộng nối hay kiểm tra khả năng in. Kiểm tra tỷ lệ, độ kín, độ dày và slice preview trong Bambu Studio.

**Kiểm thử ảnh nhân vật ngày 20/10/2026:** ảnh `Downloads/images.jpeg` chạy thành công qua callback thật trên MPS. SAM trả 11 gợi ý trong khoảng 6 giây, nhưng chưa tách đủ đầu, hai tay và hai chân riêng; có mask chồng lấn. Hunyuan tạo GLB trong khoảng 9 phút 44 giây ở 50 bước/octree 256, có 75.972 mặt và `watertight=True`, nhưng gồm một thành phần chính cùng hai mảnh nhỏ. Hình học chỉ gần giống mẫu, không giữ chính xác mặt/tóc/chi tiết trang phục và chưa có texture. Xem [báo cáo cùng ảnh đối chiếu](../hunyuan3d-2.1-mac-rocm/outputs/character_test_20261020/assessment.md). Chạy thành công không đồng nghĩa với đạt yêu cầu phân vùng theo tên bộ phận hoặc sao chép chính xác ảnh.

Repo [Hunyuan3D-Part](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part) xử lý **mesh 3D**, không nhận ảnh 2D trực tiếp. Bản P3-SAM upstream kiểm thử trên CUDA 12.1; README X-Part còn đánh dấu weights là chưa phát hành. Vì vậy mục tách ảnh ở đây là chức năng local riêng, không phải P3-SAM/X-Part.

```bash
cd /Users/kmacbookpro/Desktop/trellix
git clone https://github.com/VladimirTalyzin/hunyuan3d-2.1-mac-rocm.git
cd hunyuan3d-2.1-mac-rocm
git checkout 6f4b63b
bash install.sh --shape-only
python /Users/kmacbookpro/Desktop/trellix/mac_setup/prepare_hunyuan_gradio.py \
  /Users/kmacbookpro/Desktop/trellix/hunyuan3d-2.1-mac-rocm
```

Repo đã có trong workspace hiện tại, nên nếu tiếp tục từ lần cài này hãy bỏ
`git clone` và chỉ `cd` vào thư mục trên. Không chạy lại installer khi đã cài xong.

`install.sh` tự tạo `venv/`, tải source Tencent, cài PyTorch và dependency, rồi tải
weights. `--shape-only` bỏ tải weights texture/PBR; ở phiên bản đã đọc, installer
vẫn cài danh sách thư viện chung và thử build extension. Dung lượng shape weights
được tác giả ước tính khoảng 15 GB; lần cài này hai weight file chính chiếm khoảng
7,5 GiB. Nhiều thông báo installer hiện viết tiếng Nga. Script `prepare_hunyuan_gradio.py`
sửa vị trí tham số `theme` cho Gradio 5; thiếu sửa này, `launch.sh` báo
`Blocks.launch() got an unexpected keyword argument 'theme'`.
Quá trình cài kết thúc không tự chứng minh inference đã hoạt động.

Kiểm tra đúng môi trường và thử phép tính trên GPU Apple:

```bash
source venv/bin/activate
python - <<'PY'
import platform
import torch
print('Architecture:', platform.machine())
print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
assert torch.backends.mps.is_available(), 'MPS is unavailable'
x = torch.ones((32, 32), device='mps')
y = x @ x
torch.mps.synchronize()
assert y[0, 0].item() == 32
print('GPU arithmetic: OK')
PY
python scripts/doctor.py
python -m pip check
```

`doctor.py` có thể báo paint/texture chưa sẵn sàng vì bạn chọn shape-only; kiểm tra
riêng shape, backend, weights và phép tính GPU. Nếu các mục này lỗi thì chưa chạy
Generate. Lần chạy đầu còn có thể tải thêm model phụ.

Khởi động:

```bash
bash launch.sh --backend mps
```

Mở URL được in trong Terminal, mặc định `http://127.0.0.1:7860`.
Giữ Terminal mở trong lúc dùng app.

### Test một ảnh và xuất STL

Bài thử tự động đã chạy thành công trên máy này:

```bash
venv/bin/python tests/e2e_visual.py --skip-texture --steps 20 --octree 192 \
  --views 1 --resolution 256 --out outputs/e2e_shape
```

Nó tạo báo cáo `outputs/e2e_shape/report.html` và một GLB raw trong `outputs/`.
GLB thử lần này là `outputs/hunyuan3d_20261018_174935_raw.glb`. Để lấy STL cao
80 mm và đặt đáy ở Z=0:

```bash
venv/bin/python /Users/kmacbookpro/Desktop/trellix/mac_setup/glb_to_stl.py \
  outputs/hunyuan3d_20261018_174935_raw.glb outputs/hunyuan_demo_80mm.stl \
  --height-mm 80
```

Lần thử đã cho STL kích thước 50,136 × 45,672 × 80 mm, kín và 177.960 mặt.
Nếu chạy lại, tên GLB có timestamp mới; thay đúng tên file trong lệnh chuyển đổi.

1. Chọn ảnh mẫu hoặc upload PNG/JPG có một vật thể rõ ràng.
2. Bật xóa nền nếu ảnh chưa có nền trong suốt.
3. Lần thử đầu dùng 20–30 steps; octree 192 nếu giao diện cho chọn, hoặc mức gần
   đó trong phạm vi được giao diện hỗ trợ. Chưa tăng lên độ phân giải cao.
4. Bấm Generate và kiểm tra hình dáng ở cửa sổ preview.
5. Trong phần download/export chọn `stl`, tải file về máy. Nếu cần chính xác cao
   80 mm và đặt đáy ở Z=0, chuyển từ GLB bằng script ở trên.
6. Nhập STL vào Bambu Studio; đặt kích thước mm, hướng đặt và support rồi slice.

Đối với STL không cần chạy texture/PBR. Chỉ chạy một app tạo 3D tại một thời điểm
để hai model không tranh RAM 24 GB. Nhấn Ctrl+C ở Terminal để dừng server.

### Mở lại và dừng Hunyuan vào lần sau

Mở **một Terminal mới** và chạy đúng hai lệnh này. Không cần kích hoạt `venv`
trước: `launch.sh` tự dùng môi trường `venv/` trong repo.

```bash
cd /Users/kmacbookpro/Desktop/trellix/hunyuan3d-2.1-mac-rocm
bash launch.sh --backend mps --port 7860
```

Đợi Terminal in URL và dòng Gradio đang phục vụ; mở
<http://127.0.0.1:7860>. Terminal này phải tiếp tục mở khi dùng. Có thể kiểm
tra từ **Terminal thứ hai**:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/
```

Kết quả `200` nghĩa là web UI đã phản hồi. Để dừng, quay lại Terminal đang chạy
`bash launch.sh` và bấm **Ctrl+C** một lần; đợi prompt trở lại. Nếu đang Generate,
hãy chờ xuất file xong trước khi dừng. Đóng tab trình
duyệt không dừng server. Sau khi dừng, lệnh `curl` ở trên phải báo không kết nối
được. Nếu Terminal chạy server đã mất, tìm đúng PID đang nghe cổng:

```bash
lsof -nP -iTCP:7860 -sTCP:LISTEN
ps -p <PID> -o pid=,command=
lsof -a -p <PID> -d cwd
kill -INT <PID>
```

Thay `<PID>` bằng số PID do `lsof` in ra. Chỉ gửi `kill -INT` nếu dòng `ps`
cho thấy `python gradio_app.py` và dòng `cwd` trỏ vào repo Hunyuan này.
Đừng dùng `pkill python`, vì có thể
dừng các ứng dụng Python khác. Nếu cổng 7860 đã thuộc ứng dụng khác, chạy
`bash launch.sh --backend mps --port 7861` rồi mở
<http://127.0.0.1:7861>; dùng số cổng đó trong các lệnh kiểm tra/dừng.

Không cài lại và không clone lại khi mở app thường ngày. Nếu kiểm tra cài đặt
trước khi chạy, dùng `venv/bin/python scripts/doctor.py`; mục shape phải OK.
Trên máy 24 GB, lần thử đầu nên dùng 10–20 steps, octree 128 hoặc 192, và bỏ
chọn **Measure GPU vs CPU op share** nếu không cần chẩn đoán. 50 steps/octree
256 có thể mất nhiều phút và dùng gần hết bộ nhớ; chỉ bấm Generate một lần rồi
đợi file mới xuất hiện trong `outputs/`. Nếu giao diện cứ báo `processing`, xem
tiến độ `Diffusion Sampling` ở Terminal; file GLB chỉ được lưu sau khi dựng mesh.

Nếu sau này cần PBR, `bash install.sh` bổ sung weights texture; nó cũng có thể cập
nhật source/dependency nên không cần chạy lại cho việc mở app thường ngày.

## 3. TRELLIS Mac MPS: cài riêng

Dừng Hunyuan bằng Ctrl+C. Mở một tab Terminal mới để tránh dùng nhầm môi trường.

```bash
cd /Users/kmacbookpro/Desktop/trellix
git clone --recurse-submodules https://github.com/vinayapathak/trellis-mac-mps.git
cd trellis-mac-mps
git checkout f7c175a
git apply --check ../mac_setup/trellis_mac_24gb.patch
git apply ../mac_setup/trellis_mac_24gb.patch
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Repo và `.venv` đã có trong workspace hiện tại. Khi mở lại chỉ `cd` và
`source .venv/bin/activate`; không clone, áp patch hay tạo venv lần nữa. File
`mac_setup/trellis_mac_24gb.patch` chứa sửa đường dẫn tác giả, chế độ `--mesh-only`
và lựa chọn mặc định tương ứng trên web UI. Lệnh `git checkout` cố định đúng
phiên bản đã kiểm tra để patch áp dụng được; nếu muốn cập nhật upstream, phải
kiểm tra lại patch và chạy thử lại.

Danh sách `requirements-test.txt` của repo khóa phiên bản dùng trên máy tác giả.
Nó chưa bao gồm toàn bộ dependency cho tải model và chạy web UI. Cài chúng trong
cùng một lần để pip phát hiện xung đột thay vì âm thầm đổi phiên bản đã khóa:

```bash
python -m pip install -r requirements-test.txt \
  torchvision 'transformers<5' 'huggingface_hub<1' safetensors \
  'rembg[cpu]' scipy imageio imageio-ffmpeg flask 'open3d>=0.19'
python -m pip check
```

Đây là phần bổ sung mình suy ra từ import trong source; cần kiểm tra runtime sau
khi cài. Nếu gặp `No matching distribution` hoặc `ResolutionImpossible`, giữ
nguyên thông báo và dừng tại đây. Không xóa hàng loạt version pin hoặc dùng
`--no-deps` để ép cài. Các gói được khóa trên macOS 26.4.1/M5 Max, nên việc pip
cài được trên macOS 15.1 chưa đảm bảo mọi kernel đều chạy đúng.

Kiểm tra FlexiCubes (trong checkout đã kiểm tra, file này có sẵn):

```bash
git submodule update --init --recursive
test -f trellis/representations/mesh/flexicubes/flexicubes.py && echo 'FlexiCubes: OK'
```

Chỉ khi file vẫn thiếu, dùng fallback được tài liệu repo đề cập:

```bash
mkdir -p vendor
git clone https://github.com/MaxtirError/FlexiCubes.git vendor/FlexiCubes
mkdir -p trellis/representations/mesh/flexicubes
cp vendor/FlexiCubes/flexicubes.py vendor/FlexiCubes/tables.py \
  vendor/FlexiCubes/LICENSE.txt trellis/representations/mesh/flexicubes/
```

### Đường dẫn cố định của tác giả

Trong source đã đọc, `scripts/run_real_pipeline.py` chdir tới
`/Users/vinayak/Downloads/TRELLIS-main`. Đường dẫn đó không tồn tại trên máy bạn.
Patch trong bước clone đã sửa entry point chạy model để tính đường dẫn từ vị trí
file. Checkout hiện tại cũng có bản sao `scripts/run_real_pipeline.py.before-mac-setup`
để đối chiếu source gốc. Kiểm tra thay đổi:

```bash
git diff -- scripts/run_real_pipeline.py
```

`prepare_trellis_paths.py` vẫn có trong `mac_setup/` nếu chỉ cần sửa đường dẫn
trên một checkout không dùng patch đầy đủ. Không chạy nó sau `git apply`, vì
phần sửa đường dẫn đã có trong patch.

### Kiểm tra trước khi tải model

```bash
export SPARSE_BACKEND=pytorch
export ATTN_BACKEND=sdpa
export PYTORCH_ENABLE_MPS_FALLBACK=1
python - <<'PY'
import torch
assert torch.backends.mps.is_available(), 'MPS is unavailable'
x = torch.ones((32, 32), device='mps')
assert (x @ x)[0, 0].item() == 32
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from trellis.renderers import metal_rasterizer_binned
assert metal_rasterizer_binned.available(), 'Metal rasterizer could not initialize'
print('MPS arithmetic + inference imports + Metal rasterizer: OK')
PY
```

Nếu báo `ModuleNotFoundError`, đọc tên module và traceback. Nếu có `spconv`,
`flash_attn`, `nvdiffrast` thì kiểm tra đang ở đúng repo Mac, đúng `.venv` và đã
export hai biến backend trước khi import. Không cài các extension CUDA vào Mac.
Nếu kernel Metal báo không tương thích, cần xử lý phiên bản macOS/toolchain theo
traceback; chưa coi máy macOS 15.1 là đã được kiểm chứng chỉ dựa vào README.

### Test ảnh mẫu bằng CLI (đường xuất mesh phù hợp 24 GB)

```bash
python cli.py generate assets/example_image/typical_creature_dragon.png dragon_mesh \
  --device mps --mesh-only
```

Lần đầu tải weights Hugging Face, DINOv2 và có thể model xóa nền. Output mong đợi:
`outputs/dragon_mesh.glb` (mesh không texture). Chế độ này bỏ giải mã Gaussian,
vá lỗ và texture baking để giảm bộ nhớ. Khi giải mã có vài mặt chứa tọa độ không
hữu hạn, script loại chúng nếu dưới 5% tổng số mặt và báo số lượng cụ thể; nếu
quá ngưỡng, nó dừng thay vì xuất file hỏng. Kiểm tra tính kín của mesh trong
Bambu Studio vì bỏ vá lỗ. Chỉ khi lệnh hoàn tất và GLB đọc lại được mới coi test
end-to-end thành công.

Đường xuất GLB có texture của port gốc đã hết bộ nhớ trên máy này ở bước vá lỗ
(MPS allocated 12.11 GiB, tổng các cấp phát khác 15.08 GiB). Vì vậy, với 24 GB
hãy dùng `--mesh-only` để lấy hình học cho STL.

### Mở giao diện web

```bash
python cli.py serve
```

Mở `http://127.0.0.1:5050`. Chọn ảnh mẫu hoặc upload ảnh, đặt tên output, Generate,
xem tiến độ rồi tải GLB. Trong Advanced settings, `Mesh only` được bật sẵn trên
checkout đã chuẩn bị cho máy 24 GB này. Giao diện này phục vụ image-to-3D. Không dùng lệnh
`python app.py`/`example_text.py` của upstream làm điểm khởi động cho Mac.

### Chuyển GLB sang STL cao 80 mm

Sau khi tạo được `outputs/dragon_mesh.glb`, chạy trong môi trường `.venv` của TRELLIS:

```bash
python /Users/kmacbookpro/Desktop/trellix/mac_setup/glb_to_stl.py \
  outputs/dragon_mesh.glb outputs/dragon_80mm.stl --height-mm 80
```

Script áp dụng transform trong GLB, đổi trục Y-up của glTF thành Z-up, đặt chiều
cao 80 mm và đáy ở Z=0. Nó giữ hình học, bỏ texture theo định dạng STL, không tự
sửa lỗ. Nếu model nguồn thực tế đã Z-up, thêm `--source-up z`. Script báo kích
thước và watertight để bạn kiểm tra; watertight không bảo đảm độ dày hay độ bền.

### Mở lại TRELLIS vào lần sau

```bash
cd /Users/kmacbookpro/Desktop/trellix/trellis-mac-mps
source .venv/bin/activate
export SPARSE_BACKEND=pytorch
export ATTN_BACKEND=sdpa
export PYTORCH_ENABLE_MPS_FALLBACK=1
python cli.py serve
```

Ctrl+C để dừng. `deactivate` để rời môi trường Python.

## 4. Kiểm tra trong Bambu Studio

Import STL, chọn máy in/nozzle/filament, xác nhận kích thước mm và hướng model.
Kiểm tra mảnh rời, lỗ, thành mỏng; chọn support rồi Slice Plate. Xem Preview từng
lớp trước khi in. File tạo bởi AI có thể cần sửa mesh hoặc làm dày chi tiết.

## 5. Khi gặp lỗi

| Hiện tượng | Việc cần kiểm tra |
|---|---|
| `brew: command not found` | Chạy lại `eval "$(/opt/homebrew/bin/brew shellenv)"` trong tab hiện tại |
| `python: command not found` | Activate đúng `venv/` hoặc `.venv/` |
| `MPS available: False` | Python phải ARM64; kiểm tra phiên bản torch và macOS; không tiếp tục như thể đang dùng GPU |
| TRELLIS truy cập `/Users/vinayak/...` | Chạy helper sửa đường dẫn trong bước 3 |
| Thiếu `flexicubes` | Kiểm tra submodule và fallback ở bước 3 |
| Cài gói xung đột | Gửi nguyên lỗi pip, `python --version`, `python -m pip check`; không ép bỏ dependency |
| `MPS out of memory` | Dừng app còn lại, giảm độ phân giải hình học, chạy một ảnh; không tăng giới hạn RAM tùy tiện |
| Model download bị ngắt | Chạy lại bước tải/chạy model để tận dụng cache; đọc lỗi 401/403 nếu có |
| Không vào được localhost | Terminal phải còn chạy server và đã in URL; dùng đúng port 7860 hoặc 5050 |

Nếu cần gửi lỗi: lấy câu lệnh đã chạy và khoảng 30–50 dòng cuối traceback. Không
cần gửi token Hugging Face. Lưu môi trường sau khi đã tạo model thành công:

```bash
python -m pip freeze > installed-working.txt
git rev-parse HEAD > source-working-commit.txt
```

## Nguồn đã đối chiếu

- [Hunyuan Mac installer](https://github.com/VladimirTalyzin/hunyuan3d-2.1-mac-rocm/blob/main/install.sh): tạo môi trường, shape-only, tải weights và doctor.
- [Hunyuan launcher](https://github.com/VladimirTalyzin/hunyuan3d-2.1-mac-rocm/blob/main/launch.sh): backend MPS, môi trường và port.
- [Hunyuan UI](https://github.com/VladimirTalyzin/hunyuan3d-2.1-mac-rocm/blob/main/gradio_app.py): tạo hình và xuất STL.
- [TRELLIS Mac](https://github.com/vinayapathak/trellis-mac-mps): CLI và web UI.
- [TRELLIS dependency pins](https://github.com/vinayapathak/trellis-mac-mps/blob/main/requirements-test.txt): môi trường máy tác giả.
- [TRELLIS runner](https://github.com/vinayapathak/trellis-mac-mps/blob/main/scripts/run_real_pipeline.py): đường dẫn cố định và output GLB.
- [TRELLIS image pipeline](https://github.com/vinayapathak/trellis-mac-mps/blob/main/trellis/pipelines/trellis_image_to_3d.py): torchvision, rembg và DINOv2.
