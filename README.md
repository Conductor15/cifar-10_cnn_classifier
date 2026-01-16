# CIFAR-10 Image Classifier (PyTorch)

A convolutional neural network (CNN) trained on CIFAR-10 with PyTorch.

## Information about dataset

Object Recognition in Images - Identify the subject of 60,000 labeled images
[See more](https://www.cs.toronto.edu/~kriz/cifar.html)


## Architecture Overview

**Input:** `3×32×32` image (CIFAR-10)  
**Kernel size:** All `Conv2d` layers use `3×3` kernels

---

### Block 1
- `Conv2d(3 → 32, 3×3)` + **SiLU** + *BatchNorm*
- `Conv2d(32 → 32, 3×3)` + **SiLU** + *BatchNorm*
- **MaxPooling** + *Dropout*

---

### Block 2
- `Conv2d(32 → 64, 3×3)` + **SiLU** + *BatchNorm*
- `Conv2d(64 → 64, 3×3)` + **SiLU** + *BatchNorm*
- **MaxPooling** + *Dropout*

---

### Block 3
- `Conv2d(64 → 128, 3×3)` + **SiLU** + *BatchNorm*
- `Conv2d(128 → 128, 3×3)` + **SiLU** + *BatchNorm*
- `Conv2d(128 → 128, 3×3)` + **SiLU** + *BatchNorm*
- **MaxPooling** + *Dropout*

---

### Block 4
- `Conv2d(128 → 256, 3×3)` + **SiLU** + *BatchNorm*
- `Conv2d(256 → 256, 3×3)` + **SiLU** + *BatchNorm*
- `Conv2d(256 → 256, 3×3)` + **SiLU** + *BatchNorm*
- **MaxPooling** + *Dropout*

---

### Classifier
- `Flatten`
- `Linear(1024 → 512)` + **SiLU**
- `Linear(512 → 10)`

---

**Test Accuracy:** ~**90%** after **100 epochs**



## Run code

```bash
pip install -r requirements.txt
```

### Customize your params in `config.py`

```
BATCH_SIZE_TRAIN = 
BATCH_SIZE_TEST = 
LEARNING_RATE =
NUM_EPOCHS =
LR_GAMMA = 
MODEL_PATH = "your_output_file"
```
### Training the Model

Once dependencies are installed, you can train the model using:

```
python train.py
```

### Evaluating the Model
After training, you can evaluate the model on the CIFAR-10 test set or your own model:

```
python eval.py
```
