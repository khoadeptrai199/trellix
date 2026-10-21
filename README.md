# trellix

Workspace cho quy trình ảnh 2D → model 3D → GLB/STL → Bambu Studio.

## Bắt đầu

- **Windows / Hunyuan3D:** [Hướng dẫn cài và chạy đầy đủ](hunyuan3d-2.1-mac-rocm/docs/HUONG_DAN_WINDOWS.md).
- **macOS / Apple Silicon:** [Hướng dẫn Mac](mac_setup/HUONG_DAN_MAC.md). Các đường dẫn `/Users/kmacbookpro/Desktop/trellix` trong tài liệu cũ là ví dụ; thay bằng vị trí clone hiện tại.
- **TRELLIS Mac:** [README của port MPS](trellis-mac-mps/README.md).

```bash
git clone https://github.com/khoadeptrai199/trellix.git
cd trellix/hunyuan3d-2.1-mac-rocm
```

Repo private yêu cầu tài khoản được cấp quyền đọc. Sau khi clone, làm theo guide của hệ điều hành để tạo môi trường Python và tải weights. Clone source chưa tự cài model.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `hunyuan3d-2.1-mac-rocm/` | Hunyuan3D UI, Split 2D, SAM, export mesh và hướng dẫn Windows |
| `trellis-mac-mps/` | Source TRELLIS port cho Apple Silicon, có các chỉnh sửa local |
| `mac_setup/` | Hướng dẫn và công cụ hỗ trợ Mac/STL |

Các thư mục source được quản lý trực tiếp bởi repo này. `venv/`, `.venv/`, weights, cache và outputs không được đưa vào Git. Giữ các file LICENSE/NOTICE của dự án gốc; xem [nguồn upstream](THIRD_PARTY.md).

## Giới hạn đã xác định

- Split connected regions tách vùng rời nhau; SAM đề xuất các mask có thể chồng lấp, không bảo đảm đủ tóc/đầu/tay/chân.
- Tạo 3D từ một ảnh cần suy đoán hình học khuất và không bảo đảm giống mẫu từng chi tiết.
- Shape/SAM đã chạy local trên Mac. Quy trình Windows chưa được kiểm thử end-to-end trên Windows; guide có các bước xác minh.
