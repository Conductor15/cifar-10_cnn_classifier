# CIFAR-10 Image Classifier (PyTorch)

A convolutional neural network (CNN) trained on CIFAR-10 with PyTorch.

## Setup

```bash
git clone https://github.com/your-username/cifar10-classifier.git
cd cifar10-classifier
pip install -r requirements.txt
```
### Architecture Overview

Input: 3x32x32 image (CIFAR-10)

Block 1:
- Conv2d(3 → 32), SiLU, BatchNorm
- Conv2d(32 → 32), SiLU, BatchNorm
- MaxPool, Dropout

Block 2:
- Conv2d(32 → 64), SiLU, BatchNorm
- Conv2d(64 → 64), SiLU, BatchNorm
- MaxPool, Dropout

Block 3:
- Conv2d(64 → 128), SiLU, BatchNorm
- Conv2d(128 → 128), SiLU, BatchNorm
- Conv2d(128 → 128), SiLU, BatchNorm
- MaxPool, Dropout

Block 4:
- Conv2d(128 → 256), SiLU, BatchNorm
- Conv2d(256 → 256), SiLU, BatchNorm
- Conv2d(256 → 256), SiLU, BatchNorm
- MaxPool, Dropout

Classifier:
- Flatten
- Linear(1024 → 512), SiLU
- Linear(512 → 10)


Accuracy in test set ~90% after 100 epochs