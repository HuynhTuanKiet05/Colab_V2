# Colab_V2 - Drug Disease AI Predictor

Ứng dụng web PHP + MySQL tích hợp Python API để dự đoán liên kết giữa **thuốc** và **bệnh lý** bằng mô hình HGT cải tiến.

## Tổng quan
Dự án gồm 3 phần chính:

- **Web PHP**: giao diện người dùng, trang admin, lịch sử tra cứu
- **MySQL**: lưu tài khoản, dữ liệu thực thể, liên kết, lịch sử dự đoán
- **Python API**: phục vụ suy luận AI và trả kết quả cho web qua HTTP

## Tính năng chính

- Đăng nhập / đăng xuất
- Tra cứu **Thuốc -> Bệnh**
- Tra cứu **Bệnh -> Thuốc**
- Chọn dataset và top-k
- Hiển thị đồ thị 3D tương tác
- Lưu lịch sử dự đoán
- Trang admin quản lý:
  - thuốc
  - bệnh lý
  - liên kết sinh học
- Giao diện hiện đại, tối ưu cho màn hình desktop

## Cấu trúc thư mục

```text
CaiTien_HGT_repo/
├─ app/
├─ public/
│  ├─ assets/
│  ├─ index.php
│  ├─ login.php
│  ├─ history.php
│  ├─ admin.php
│  ├─ admin_drugs.php
│  ├─ admin_diseases.php
│  └─ admin_links.php
├─ python_api/
├─ model/
├─ data/
└─ README.md
```

## Yêu cầu môi trường

### Web
- Windows 10/11
- XAMPP
- PHP 8.x
- MySQL / MariaDB

### Python
- Python 3.10+ khuyến nghị
- `pip`
- Các thư viện theo `requirements.txt` của phần Python API / training

## Hướng dẫn cài đặt

### 1) Clone hoặc copy source vào XAMPP
Đặt project vào thư mục:

```bash
C:\xampp\htdocs\DoANBase_Final\CaiTien_HGT_repo
```

### 2) Tạo database
- Mở **phpMyAdmin**
- Tạo database theo cấu hình trong `app/config.php`
- Import file SQL nếu dự án có sẵn schema/dump dữ liệu

> Nếu bạn chưa có file SQL, hãy kiểm tra trong repo xem có file như `database_schema.sql`, `schema.sql` hoặc file dump tương tự.

### 3) Cấu hình kết nối CSDL
Mở file:

```bash
app/config.php
```

Kiểm tra các thông số:
- host
- database name
- username
- password

Ví dụ thường dùng trên XAMPP:
- host: `127.0.0.1`
- user: `root`
- password: rỗng

### 4) Cài Python dependencies
Mở terminal tại thư mục dự án hoặc thư mục `python_api`, sau đó cài thư viện:

```bash
python -m pip install -r requirements.txt
```

Nếu `requirements.txt` nằm trong `python_api/` thì chạy:

```bash
cd python_api
python -m pip install -r requirements.txt
```

### 5) Chạy Python API
Khởi động API tại cổng `8000`:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Kiểm tra health:

```bash
http://127.0.0.1:8000/health
```

### 6) Chạy web PHP
- Mở XAMPP
- Start **Apache** và **MySQL**
- Truy cập web:

```bash
http://localhost/DoANBase_Final/CaiTien_HGT_repo/public/login.php
```

## Tài khoản mặc định

Nếu hệ thống đã có seed dữ liệu tài khoản, thông tin mặc định thường là:

- `admin / password`
- `user1 / password`

> Nếu mật khẩu khác, kiểm tra dữ liệu seed hoặc bảng `users` trong database.

## Cách sử dụng

### Trang người dùng
1. Đăng nhập
2. Vào trang Dashboard
3. Nhập tên thuốc / bệnh / ID
4. Chọn dataset
5. Chọn top-k
6. Nhấn **Dự đoán**
7. Xem:
   - bảng kết quả
   - đồ thị 3D
   - lịch sử tra cứu

### Trang quản trị
1. Đăng nhập bằng tài khoản admin
2. Vào `Admin`
3. Quản lý:
   - thuốc
   - bệnh
   - liên kết
4. Theo dõi thống kê hệ thống

## Ghi chú về AI / mô hình
- Web PHP gọi Python API qua HTTP.
- Python API là lớp trung gian để phục vụ dự đoán.
- Nếu muốn đổi sang model train thật, cần đảm bảo API Python load được checkpoint / pipeline suy luận tương ứng.

## Lỗi thường gặp

### 1) Web báo API offline
- Kiểm tra Python API đã chạy chưa
- Kiểm tra cổng `8000`
- Kiểm tra file cấu hình endpoint trong `PredictionService.php`

### 2) Không đăng nhập được
- Kiểm tra bảng `users`
- Kiểm tra seed data
- Kiểm tra kết nối MySQL

### 3) Không load được dữ liệu hoặc đồ thị
- Kiểm tra database đã import đủ chưa
- Kiểm tra dữ liệu đầu vào của dataset
- Kiểm tra API trả JSON đúng định dạng

## Phát triển thêm
Bạn có thể tiếp tục mở rộng dự án theo các hướng:

- tích hợp model inference thật cho Python API
- thêm validation split / model selection
- thêm contrastive learning / multi-view fusion
- thêm trang báo cáo kết quả train

## License
Chưa xác định.

## Tác giả
Dự án phát triển bởi nhóm/ cá nhân thực hiện luận văn / đồ án.
