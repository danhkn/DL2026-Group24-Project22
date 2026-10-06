# DATA.md - Dataset & Artifact Documentation

Tài liệu này cung cấp chi tiết nguồn gốc dữ liệu, phiên bản, phân chia tập dữ liệu (data split), quy trình tiền xử lý (preprocessing) và hướng dẫn tái lập thực nghiệm theo yêu cầu dự án.

---

## 1. Danh sách Datasets sử dụng

| Dataset | Vai trò | Phiên bản / Nguồn | URL chính thức | Data Split sử dụng | Kích thước & Kênh màu |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **CIFAR-10** | **In-Distribution (ID)** | Python version (Alex Krizhevsky) | https://www.cs.toronto.edu/~kriz/cifar.html | **Train:** 50,000 ảnh (Feature Bank)<br>**Test:** 10,000 ảnh (ID Eval) | 32×32 (RGB, 3 channels) |
| **CIFAR-100** | **Near-OOD** | Python version (Alex Krizhevsky) | https://www.cs.toronto.edu/~kriz/cifar.html | **Test:** 10,000 ảnh (OOD Eval) | 32×32 (RGB, 3 channels) |
| **MNIST** | **Far-OOD** | Yann LeCun et al. | https://yann.lecun.com/exdb/mnist/ | **Test:** 10,000 ảnh (OOD Eval) | 28×28 (Grayscale) -> Resize 32×32 (RGB) |

---

## 2. Quy trình tiền xử lý dữ liệu (Preprocessing Procedure)

Tất cả các tập dữ liệu đều được chuẩn hóa và đưa về cùng kích thước không gian để truyền qua mạng trích xuất đặc trưng `WideResNet-40-2` (`wrn-40-2/cifar10/crossentropy`):

### A. CIFAR-10 & CIFAR-100 (ID & Near-OOD)
* **Kích thước đầu vào:** Giữ nguyên 32×32 pixel.
* **Chuẩn hóa giá trị pixel:** Đưa về khoảng [0, 1] qua `transforms.ToTensor()`.
* **Standardization:** Trừ giá trị trung bình (mean) và chia độ lệch chuẩn (std) của tập huấn luyện CIFAR-10:
  * Mean = [0.4914, 0.4822, 0.4465]
  * Std = [0.2470, 0.2435, 0.2616]

### B. MNIST (Far-OOD)
* **Chuyển đổi kích thước không gian:** Dùng phép biến đổi nội suy `transforms.Resize((32, 32))` để đưa ảnh chữ số từ 28×28 lên 32×32.
* **Mở rộng kênh màu:** Nhân bản kênh đơn (grayscale, 1 channel) thành 3 kênh màu (RGB) bằng `transforms.Grayscale(num_output_channels=3)`.
* **Standardization:** Áp dụng cùng thông số Mean và Std của CIFAR-10 như trên để đảm bảo tính đồng nhất phân phối đầu vào của backbone.

---

## 3. Tạo và Tải Preprocessed Artifacts (`ood_artifacts.npz`)

Để người đánh giá có thể chạy demo tức thì trong ~10 giây mà không cần tải lại toàn bộ 335 MB dataset gốc, nhóm đã tạo file nén cache:

* **File:** `data/ood_artifacts.npz` (Dung lượng: ~25.2 MB)
* **Vị trí trong repo:** Được lưu trực tiếp trong thư mục `data/` của GitHub repository: https://github.com/danhkn/DL2026-Group24-Project22/tree/main/data
* **Nội dung bên trong file:**
  1. `feature_bank`: Mảng ma trận (50000, 128) kiểu `float32` chứa 50.000 vector đặc trưng latent từ CIFAR-10 train phục vụ thuật toán KNN (K=50).
  2. `label_bank`: Mảng nhãn tương ứng (50000,) của CIFAR-10 train.
  3. `vim_principal_subspace`: Ma trận không gian con chính (128, 64) trích xuất qua SVD/PCA trên feature covariance phục vụ thuật toán ViM (d=64).
  4. `vim_alpha`: Hệ số scale năng lượng logit của ViM.
  5. `vim_u`: Vector trọng số bias/offset tính từ mô hình WideResNet.
  6. `pool_cifar10`, `pool_cifar100`, `pool_mnist`: 200 ảnh mẫu của mỗi tập kèm nhãn phục vụ visual trực tiếp trên `OOD_Demo.ipynb`.
  7. `feature_sha256`: Mã hash SHA-256 xác thực tính toàn vẹn của dữ liệu feature bank.

---

## 4. Kịch bản tái lập dữ liệu (Scripts to Reproduce)

### Bước 1: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt