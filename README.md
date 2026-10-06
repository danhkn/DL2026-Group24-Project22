# DL2026 - Group 24 - Project 22
## Out-of-Distribution (OOD) Detection: MSP vs KNN vs ViM

### 1. Giới thiệu dự án
Dự án nghiên cứu, thực nghiệm và so sánh 3 phương pháp phát hiện mẫu ngoài phân phối (**OOD Detection**) trên cùng một mô hình:
- **Model:** WideResNet-40-2 (`wrn-40-2/cifar10/crossentropy`) huấn luyện trên CIFAR-10 đạt độ chính xác **94.84%**.
- **In-Distribution (ID):** CIFAR-10 test (10,000 ảnh).
- **Near-OOD:** CIFAR-100 test (10,000 ảnh) — ảnh tự nhiên 32×32, nhiều lớp tương đồng ngữ nghĩa.
- **Far-OOD:** MNIST test (10,000 ảnh) — chữ số viết tay đơn sắc, được resize sang 32×32×3.

### 2. Các phương pháp thực nghiệm
Cả 3 phương pháp dùng chung biểu diễn đặc trưng 128 chiều (*128-D latent feature*) để đảm bảo tính công bằng:
1. **MSP (Maximum Softmax Probability):** Đánh giá dựa trên xác suất lớn nhất từ đầu ra Softmax. Điểm cao = giống ID.
2. **KNN (K-Nearest Neighbors):** Tính khoảng cách Euclidean tới lân cận thứ $K=50$ trong ngân hàng 50,000 đặc trưng CIFAR-10 train. Điểm cao = giống OOD.
3. **ViM (Virtual-logit Matching):** Kết hợp khoảng cách residual ngoài không gian con chính (principal subspace, $d=64$) và năng lượng logit. Điểm cao = giống OOD.

### 3. Kết quả đánh giá Benchmark

| Phương pháp | OOD Dataset | Phân loại OOD | AUROC (%) ↑ | FPR95 (%) ↓ | AUPR-IN (%) ↑ | AUPR-OUT (%) ↑ | AUTC (%) ↓ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MSP** | CIFAR-100 | Near-OOD | 87.83 | 43.08 | 88.42 | 85.20 | 40.69 |
| **MSP** | MNIST | Far-OOD | 92.66 | 22.47 | 94.33 | 90.29 | 37.24 |
| **KNN** ($K=50$) | CIFAR-100 | Near-OOD | **88.95** | **41.73** | 89.49 | 87.45 | **38.89** |
| **KNN** ($K=50$) | MNIST | Far-OOD | **93.73** | 22.46 | 94.93 | **91.30** | **32.28** |
| **ViM** ($d=64$) | CIFAR-100 | Near-OOD | 82.40 | 61.67 | 82.38 | 80.75 | 44.55 |
| **ViM** ($d=64$) | MNIST | Far-OOD | 93.30 | **19.41** | **95.09** | 88.81 | 38.93 |

### 4. Danh sách Notebooks
1. **`notebooks/OOD_MSP_KNN_ViM_demo.ipynb`**: Chạy toàn bộ luồng thực nghiệm, trích xuất đặc trưng chung 128 chiều, fit ViM và tạo feature bank cho KNN.
2. **`notebooks/ood_comparison.ipynb`**: Phân tích chi tiết số liệu, so sánh tương quan score, vẽ biểu đồ phân phối và đường ROC của 3 phương pháp.
3. **`notebooks/OOD_Demo.ipynb`**: Demo chấm điểm nhanh trên 3 ảnh bất kỳ (chạy ~10s không cần tải lại toàn bộ dataset).

### 5. Hướng dẫn cài đặt & Thực thi
```bash
# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# Khởi chạy demo nhanh (~10s)
jupyter notebook notebooks/OOD_Demo.ipynb
```
### 6. Cấu trúc thư mục
```text
DL2026-Group24-Project22/
├── notebooks/
│   ├── OOD_MSP_KNN_ViM_demo.ipynb
│   ├── ood_comparison.ipynb
│   └── OOD_Demo.ipynb
├── data/
├── requirements.txt
├── .gitignore
├── DATA.md
└── README.md
