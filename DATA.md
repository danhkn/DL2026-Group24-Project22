# Tài liệu Dữ liệu và Artifacts (DATA.md)

Tài liệu này mô tả các tập dữ liệu, quy trình tiền xử lý và cấu trúc file artifact trong dự án.

### 1. Các tập dữ liệu (Datasets)

| Dataset | Số lượng mẫu (Test) | Kích thước gốc | Vai trò | Ghi chú tiền xử lý |
| :--- | :---: | :---: | :---: | :--- |
| **CIFAR-10** | 10,000 | 32×32 (RGB) | **In-Distribution (ID)** | Chuẩn hóa theo chuẩn CIFAR-10: Mean `[0.4914, 0.4822, 0.4465]`, Std `[0.2470, 0.2435, 0.2616]`. |
| **CIFAR-100** | 10,000 | 32×32 (RGB) | **Near-OOD** | Dùng chung transform với CIFAR-10. Không có lớp nào trùng với CIFAR-10 nhưng tương đồng về ngữ nghĩa. |
| **MNIST** | 10,000 | 28×28 (Grayscale) | **Far-OOD** | 1. Resize lên 32×32.<br>2. Chuyển kênh đơn sang 3 kênh (Grayscale to RGB).<br>3. Chuẩn hóa theo Mean và Std của CIFAR-10. |

> *Ghi chú:* Tập huấn luyện **CIFAR-10 train (50,000 ảnh)** được sử dụng làm dữ liệu tham chiếu (Reference data) để trích xuất Feature Bank cho KNN và khớp tham số (Principal Subspace, Alpha) cho ViM. Mô hình không được train lại.

---

### 2. Artifact trích xuất (`ood_artifacts.npz`)

Để phục vụ demo nhanh mà không cần tải lại 335 MB dữ liệu và quét qua 50,000 ảnh huấn luyện, notebook `OOD_MSP_KNN_ViM_demo.ipynb` xuất ra file nén `ood_artifacts.npz` (dung lượng ~25 MB):

#### Các trường dữ liệu trong archive:
1. **KNN Data:**
   - `feature_bank`: Mảng numpy kích thước `(50000, 128)` kiểu `float32` chứa đặc trưng latent của CIFAR-10 train.
   - `label_bank`: Mảng numpy nhãn tương ứng `(50000,)`.
2. **ViM Parameters:**
   - `vim_principal_subspace`: Ma trận không gian con chính `(128, 64)`.
   - `vim_alpha`: Hệ số tỉ lệ năng lượng (scalar `float32`).
   - `vim_u`: Vector trọng số bias/offset tính từ lớp fully-connected.
3. **Demo Image Pool:**
   - 200 ảnh mẫu cho mỗi tập (`pool_cifar10`, `pool_cifar100`, `pool_mnist`) kèm nhãn và chỉ số index gốc để vẽ ảnh minh họa.
4. **Provenance & Metadata:**
   - `feature_sha256`: Mã băm SHA-256 của feature bank để kiểm tra tính toàn vẹn khi load vào `OOD_Demo.ipynb`.
   - `model_id`: Checkpoint định danh (`wrn-40-2/cifar10/crossentropy`).
   - `knn_k`: Số lân cận $K=50$.
   - `vim_d`: Số chiều subspace $d=64$.

---

### 3. File kết quả đánh giá (Evaluation Outputs)

Quá trình phân tích trong `ood_comparison.ipynb` sinh ra và đối chiếu các file:
- `msp_results.csv`, `msp_scores.npz`: Điểm số và metric của phương pháp MSP.
- `KNN_results.csv`, `KNN_scores.npz`: Điểm số và metric của phương pháp KNN.
- `ViM_results.csv`, `ViM_scores.npz`: Điểm số và metric của phương pháp ViM.
