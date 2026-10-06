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
- **Channel Expansion:** Replicated single-channel grayscale into 3-channel RGB via `transforms.Grayscale(num_output_channels=3)`.
- **Standardization:** Applied the identical CIFAR-10 mean and std values above to preserve input distribution consistency for the backbone.

---

## 3. Preprocessed Artifacts (`ood_artifacts.npz`)

To enable instant demo execution (~10 seconds) without downloading the full 335 MB raw datasets, preprocessed artifacts are cached:
- **File:** `data/ood_artifacts.npz` (Size: ~25.2 MB)
- **Location:** Tracked and stored in the `data/` directory of the GitHub repository.
- **Contents:**
  1. `feature_bank`: `(50000, 128)` float32 array containing latent feature vectors from CIFAR-10 train set for KNN ($K=50$).
  2. `label_bank`: Corresponding labels `(50000,)` of CIFAR-10 train set.
  3. `vim_principal_subspace`: Principal subspace projection matrix `(128, 64)` fitted via SVD/PCA for ViM ($d=64$).
  4. `vim_alpha`: ViM scaling parameter balancing residual distance and logit energy.
  5. `vim_u`: Bias/offset vector computed from the WideResNet classifier.
  6. `pool_cifar10`, `pool_cifar100`, `pool_mnist`: 200 sample images per dataset with labels for visualization in `OOD_Demo.ipynb`.
  7. `feature_sha256`: SHA-256 hash verifying feature bank integrity.

---

## 4. Reproducibility & Execution Scripts

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
