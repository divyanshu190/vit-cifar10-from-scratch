# Vision Transformer (ViT-Small) on CIFAR-10 — Training from Scratch

This repository demonstrates how to train a Vision Transformer (ViT-Small) from scratch on the CIFAR-10 dataset using PyTorch and timm, without any ImageNet pretraining.

The project shows that with a strong training strategy—specifically heavy data augmentation, Mixup/CutMix, Exponential Moving Average (EMA), and cosine learning rate scheduling—a transformer-based model can achieve competitive performance (~90% accuracy) on a small dataset like CIFAR-10.

---

## Key Features

- Vision Transformer trained entirely from scratch (no pretrained weights)
- ViT-Small architecture adapted for small-resolution images
- Patch size reduced to 4x4 for effective tokenization of 32x32 images
- Strong data augmentation using RandomCrop, HorizontalFlip, and RandAugment
- Mixup and CutMix for improved generalization with soft-label supervision
- Exponential Moving Average (EMA) for stable evaluation
- AdamW optimizer with cosine annealing learning rate schedule

---

## Model Architecture

- Model: Vision Transformer (ViT-Small)
- Image Size: 32 x 32
- Patch Size: 4 x 4
- Number of Patches: 8 x 8 = 64
- Embedding Dimension: 384
- Attention Heads: 12
- Transformer Blocks: ViT-Small configuration
- Classifier: CLS token with linear head
- Number of Classes: 10 (CIFAR-10)

Reducing the patch size is critical for CIFAR-10. Using the default 16x16 patch size would result in too few tokens and limit the effectiveness of self-attention.

---

## Dataset

CIFAR-10:
- 60,000 RGB images
- 10 classes
- Image resolution: 32 x 32

The dataset is automatically downloaded using torchvision.

---

## Training Strategy

### Data Augmentation
- Random Crop with padding
- Random Horizontal Flip
- RandAugment

Strong augmentation is essential for Vision Transformers due to the lack of convolutional inductive bias.

---

### Mixup and CutMix
- Enabled during training
- Produces soft labels
- Improves robustness and generalization

Loss Function:
- SoftTargetCrossEntropy

---

### Exponential Moving Average (EMA)
- Maintains a moving average of model parameters
- EMA weights are used for evaluation
- Improves training stability and final accuracy

---

### Optimization

- Optimizer: AdamW
- Learning Rate: 3e-4
- Weight Decay: 0.05
- Scheduler: Cosine Annealing Learning Rate
- Epochs: 200

This configuration follows best practices for training transformer-based vision models.

---

## Project Structure

```

.
├── train.py          # Training loop and evaluation
├── requirements.txt  # Python dependencies
├── data/             # CIFAR-10 dataset (auto-downloaded)
└── README.md         # Project documentation

```

---

## Installation

Install dependencies:

```

pip install torch torchvision timm

```

---

## Usage

To train the model from scratch:

```

python train.py

```

The script will:
- Download the CIFAR-10 dataset automatically
- Train ViT-Small for 200 epochs
- Evaluate performance using EMA weights after each epoch

---

## Results

- Dataset: CIFAR-10
- Model: ViT-Small (timm)
- Training: From scratch
- Epochs: 200
- Test Accuracy: Approximately 88–90%

Exact results may vary depending on hardware and random seed.

---

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  https://arxiv.org/abs/2010.11929

- timm: PyTorch Image Models
  https://github.com/huggingface/pytorch-image-models

- CIFAR-10 Dataset

---
