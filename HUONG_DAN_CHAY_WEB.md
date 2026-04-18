# 🚀 Hướng Dẫn Khởi Chạy Hệ Thống HGT AI Dashboard

Tài liệu này hướng dẫn bạn cách khởi chạy toàn bộ hệ thống dự đoán liên kết Thuốc - Bệnh sử dụng mô hình Giao thoa Đồ thị (HGT).

---

## 📋 Yêu cầu hệ thống
*   **PHP**: 7.4 trở lên.
*   **Python**: 3.9 (Khuyến nghị dùng Conda/Miniconda).
*   **Cơ sở dữ liệu**: MySQL.
*   **Trình duyệt**: Chrome, Edge hoặc Brave (để hiển thị tốt nhất đồ thị 3D).

---

## 🛠️ Bước 1: Khởi chạy Backend AI (Python API)

Backend AI chịu trách nhiệm tính toán và chạy mô hình học sâu.

1.  **Mở Terminal** và di chuyển đến thư mục dự án.
2.  **Kích hoạt môi trường Conda**:
    ```powershell
    conda activate amdgt_env
    ```
3.  **Di chuyển vào thư mục API**:
    ```powershell
    cd python_api
    ```
4.  **Chạy Server Uvicorn**:
    ```powershell
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    ```
    > [!IMPORTANT]
    > Giữ cửa sổ Terminal này luôn mở. Server phải chạy ở cổng **8000** để Frontend có thể kết nối.

---

## 🌐 Bước 2: Khởi chạy Frontend (PHP Web)

Frontend cung cấp giao diện người dùng và quản lý dữ liệu lịch sử.

1.  **Mở một Terminal mới** (không tắt Terminal Python).
2.  **Khởi chạy PHP Server**:
    ```powershell
    php -S localhost:8080 -t public
    ```
3.  **Truy cập hệ thống**:
    Mở trình duyệt và nhập địa chỉ: `http://localhost:8080`

---

## 🗄️ Bước 3: Cấu hình Cơ sở dữ liệu (Nếu cần)

Nếu bạn thay đổi mật khẩu hoặc tên DB, hãy cập nhật tại:
*   File: `app/config.php`

---

## 🔑 Thông tin quản trị viên
*   **Trang quản trị**: `http://localhost:8080/admin.php`
*   **Tài khoản demo**: `admin@hgt.com` / `password`

---

## 💡 Lưu ý quan trọng
*   **Thứ tự**: Bạn nên chạy Backend Python trước, sau đó mới chạy Frontend PHP.
*   **Đồ thị 3D**: Nếu đồ thị không hiện, hãy kiểm tra kết nối Internet (hệ thống sử dụng thư viện Three.js từ CDN).
*   **Cảnh báo Đỏ**: Nếu Dashboard hiện thông báo "Python HGT-API ngắt kết nối", hãy kiểm tra lại Terminal ở Bước 1.

---
*Chúc bạn có những trải nghiệm tuyệt vời với HGT AI Dashboard!*
