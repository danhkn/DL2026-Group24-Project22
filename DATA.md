# DATA.md - Dataset & Artifact Documentation

This document provides detailed information regarding data sources, versions, dataset splits, preprocessing pipelines, and experimental reproducibility instructions.

---

## 1. Datasets Used

| Dataset | Role | Version / Source | Official URL | Split Used | Dimensions & Channels |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CIFAR-10** | In-Distribution (ID) | Python version (Alex Krizhevsky) | [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) | **Train:** 50,000 images (Feature Bank)<br>**Test:** 10,000 images (ID Eval) | 32×32 (RGB, 3 channels) |
| **CIFAR-100** | Near-OOD | Python version (Alex Krizhevsky) | [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) | **Test:** 10,000 images (OOD Eval) | 32×32 (RGB, 3 channels) |
| **MNIST** | Far-OOD | Yann LeCun et al. | [MNIST Dataset](https://yann.lecun.com/exdb/mnist/) | **Test:** 10,000 images (OOD Eval) | 28×28 (Grayscale) → Resized to 32×32 (RGB) |

---

## 2. Preprocessing Procedure

All datasets are standardized and mapped to identical spatial dimensions before being fed into the WideResNet-40-2 backbone (`wrn-40-2/cifar10/crossentropy`):

### A. CIFAR-10 & CIFAR-100 (ID & Near-OOD)
- **Input Resolution:** Kept at native 32×32 pixels.
- **Pixel Scaling:** Scaled to $[0, 1]$ via `transforms.ToTensor()`.
- **Standardization:** Normalized using CIFAR-10 training set mean and standard deviation:
  - `Mean = [0.4914, 0.4822, 0.4465]`
  - `Std = [0.2470, 0.2435, 0.2616]`

### B. MNIST (Far-OOD)
- **Spatial Resizing:** Interpolated from 28×28 to 32×32 using `transforms.Resize((32, 32))`.
- **Channel Expansion:** Replicated single-channel grayscale into 3-
