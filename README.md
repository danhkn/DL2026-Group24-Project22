# DL2026 - Group 24 - Project 22
## Out-of-Distribution (OOD) Detection on CIFAR-10

### 1. Giới thiệu bài toán
Dự án nghiên cứu và so sánh khả năng phát hiện mẫu ngoại phân phối (Out-of-Distribution - OOD) trên mô hình phân loại hình ảnh **WideResNet-40-2** huấn luyện trên **CIFAR-10** (In-Distribution, Test Accuracy: 94.84%).

Các phương pháp được khảo sát trên hai cấp độ:
- **Near-OOD (CIFAR-100)**: Miền ảnh tự nhiên tương tự, cùng độ phân giải 32x32 và có nhiều lớp ngữ nghĩa gần với CIFAR-10.
- **Far-OOD (MNIST / MNIST-C)**: Chữ số viết tay (MNIST và bộ 16 corruption MNIST-C), được tiền xử lý resize sang 32x32x3.

### 2. Các phương pháp thực nghiệm
1. **MSP (Maximum Softmax Probability)**: Đánh giá độ tự tin đầu ra Softmax.
2. **KNN (K-Nearest Neighbors)**: Tính khoảng cách Euclidean tới $K=50$ hàng xóm gần nhất trong không gian embedding 128 chiều.
3. **ViM (Virtual-logit Matching)**: Kết hợp không gian đặc trưng chính (principal subspace, $d=64$) và logit của mô hình.

### 3. Bảng kết quả tổng hợp

| Phương pháp | OOD Dataset | Phân loại OOD | AUROC (%) ↑ | FPR95 (%) ↓ | AUPR-IN (%) ↑ | AUPR-OUT (%) ↑ | AUTC (%) ↓ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MSP** | CIFAR-100 | Near-OOD | **87.83** | **43.08** | 88.42 | 85.20 | 40.69 |
| **MSP** | MNIST-C (16 corruptions) | Far-OOD | **94.07** | **18.51** | 79.67 | 99.46 | 35.86 |
| **KNN** ($K=50$) | CIFAR-100 | Near-OOD | **88.95** | **41.73** | 89.49 | 87.45 | 38.89 |
| **KNN** ($K=50$) | MNIST | Far-OOD | **93.73** | **22.46** | 94.93 | 91.30 | 32.28 |
| **ViM** ($d=64$) | CIFAR-100 | Near-OOD | **82.40** | **61.70** | 82.38 | 80.75 | 44.55 |
| **ViM** ($d=64$) | MNIST | Far-OOD | **93.29** | **19.41** | 95.09 | 88.79 | 38.94 |

### 4. Cấu trúc thư mục
