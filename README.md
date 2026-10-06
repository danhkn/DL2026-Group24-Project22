# DL2026 - Group 24 - Project 22

## Out-of-Distribution (OOD) Detection: MSP vs KNN vs ViM

## 1. Project Overview
This project benchmarks and compares three prominent Out-of-Distribution (OOD) detection methods on a unified model backbone:
- **Model:** WideResNet-40-2 (`wrn-40-2/cifar10/crossentropy`) trained on CIFAR-10 with an accuracy of **94.84%**.
- **In-Distribution (ID):** CIFAR-10 test set (10,000 images).
- **Near-OOD:** CIFAR-100 test set (10,000 images) — natural images (32×32) sharing semantic domain similarities.
- **Far-OOD:** MNIST test set (10,000 images) — grayscale handwritten digits, resized to 32×32×3.

## 2. Methodology
All three methods leverage a shared **128-D latent feature space** to ensure fair benchmarking:
1. **MSP (Maximum Softmax Probability):** Scores samples based on the maximum predicted softmax probability. Higher score indicates higher likelihood of being ID.
2. **KNN (K-Nearest Neighbors):** Computes the Euclidean distance to the $K = 50$-th nearest neighbor across a 50,000-sample CIFAR-10 training feature bank. Higher distance indicates higher likelihood of being OOD.
3. **ViM (Virtual-logit Matching):** Couples residual distance orthogonal to the principal feature subspace ($d = 64$) with logit energy. Higher score indicates higher likelihood of being OOD.

## 3. Benchmark Results
| Method | OOD Dataset | OOD Type | AUROC (%) ↑ | FPR95 (%) ↓ | AUPR-IN (%) ↑ | AUPR-OUT (%) ↑ | AUTC (%) ↓ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MSP** | CIFAR-100 | Near-OOD | 87.83 | 43.08 | 88.42 | 85.20 | 40.69 |
| **MSP** | MNIST | Far-OOD | 92.66 | 22.47 | 94.33 | 90.29 | 37.24 |
| **KNN** ($K = 50$) | CIFAR-100 | Near-OOD | **88.95** | **41.73** | 89.49 | 87.45 | **38.89** |
| **KNN** ($K = 50$) | MNIST | Far-OOD | **93.73** | 22.46 | 94.93 | **91.30** | **32.28** |
| **ViM** ($d = 64$) | CIFAR-100 | Near-OOD | 82.40 | 61.67 | 82.38 | 80.75 | 44.55 |
| **ViM** ($d = 64$) | MNIST | Far-OOD | 93.30 | **19.41** | **95.09** | 88.81 | 38.93 |

## 4. Notebooks
1. `notebooks/OOD_Detection_MSP_KNN_ViM.ipynb`: End-to-end experiment pipeline, feature extraction (128-D), ViM subspace fitting, and KNN feature bank construction.
2. `notebooks/OOD_Comparison_Analysis.ipynb`: Detailed quantitative evaluation, score correlation analysis, distribution plots, and ROC curve comparisons across all three methods.
3. `notebooks/OOD_Demo.ipynb`: Rapid inference demo on 3 arbitrary samples (~10s execution, no full dataset download required).

## 5. Quickstart & Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run the lightweight demo (~10s)
jupyter notebook notebooks/OOD_Demo.ipynb
```

## 6. Directory Structure
```text
DL2026-Group24-Project22/
├── notebooks/
│   ├── OOD_Detection_MSP_KNN_ViM.ipynb
│   ├── OOD_Comparison_Analysis.ipynb
│   └── OOD_Demo.ipynb
├── data/
├── requirements.txt
├── .gitignore
├── DATA.md
└── README.md
```
