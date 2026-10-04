# Dataset Documentation

## 1. Official Dataset URLs & Versions
- **In-Distribution (ID): CIFAR-10** (`torchvision.datasets.CIFAR10`)
  - Official URL: https://www.cs.toronto.edu/~kriz/cifar.html
  - Split: 50,000 training images (used for feature bank in KNN and principal subspace fitting in ViM), 10,000 test images (used for ID evaluation).
- **Out-of-Distribution (OOD): CIFAR-100** (`torchvision.datasets.CIFAR100`)
  - Official URL: https://www.cs.toronto.edu/~kriz/cifar.html
  - Split: 10,000 test images (evaluated strictly as OOD, class labels discarded and mapped to -1).

## 2. Preprocessing Procedure
- Convert to Tensor: `transforms.ToTensor()`
- Normalization (standard CIFAR-10 statistics):
  - Mean: `(0.4914, 0.4822, 0.4465)`
  - Std: `(0.2470, 0.2435, 0.2616)`
- No data augmentation applied during evaluation or feature extraction.

## 3. Data Reproduction
Both datasets are automatically downloaded and cached into `./data/` directly via torchvision scripts provided in the notebooks.
