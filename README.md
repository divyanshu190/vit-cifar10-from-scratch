# Vision Transformer (ViT-Small) from Scratch on CIFAR-10

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

A Vision Transformer (ViT-Small) model trained from random initialization on the CIFAR-10 dataset using PyTorch, without using any ImageNet pre-trained weights.

This project demonstrates that with a proper training strategy—specifically strong data augmentation and cosine learning rate scheduling—a transformer-based model can achieve competitive performance (~90% accuracy) on small datasets like CIFAR-10, challenging the inductive bias advantage typically held by CNNs.

---

## Key Features

* **Training from Scratch:** Model trained without any pre-trained weights.
* **ViT-Small Architecture:** Adapted for CIFAR-10 with smaller patch size.
* **Optimized Training:** Utilizes cosine learning rate scheduling and strong data augmentation.
* **High Performance:** Achieves **~90% Top-1 Accuracy** on the CIFAR-10 test set after 200 epochs.

---

## Project Structure

```plaintext
├── train.py           # Training loop, data loading, and evaluation
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## Requirements

* Python **3.8 or higher**
* PyTorch **2.0 or higher**
* torchvision
* timm

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/divyanshu190/vit-cifar10-from-scratch
   cd vit-cifar10-from-scratch
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To start training the model from scratch, run the training script. This will download the CIFAR-10 dataset automatically if it is not present.

```bash
python train.py
```

*Training logs and evaluation results are printed after each epoch.*

---

## Hyperparameters

The model was trained with the following configuration:

| Parameter         | Value       | Description             |
| :---------------- | :---------- | :---------------------- |
| Epochs            | 200         | Total training passes   |
| Batch Size        | 128         |                         |
| Optimizer         | AdamW       | Improved regularization |
| Learning Rate     | 3e-4        | Cosine annealing        |
| Weight Decay      | 0.05        | Regularization          |
| Data Augmentation | RandAugment | Strong augmentation     |
| Patch Size        | 4 × 4       | CIFAR-10 adaptation     |
| Embed Dim         | 384         | Hidden size             |
| Heads             | 12          | Attention heads         |
| MLP Dim           | 1536        | 4× expansion            |

---

## Results

The model was trained for 200 epochs using AdamW and cosine annealing learning rate scheduling.

| Metric        | Value     |
| :------------ | :-------- |
| Dataset       | CIFAR-10  |
| Model         | ViT-Small |
| Epochs        | 200       |
| Test Accuracy | ~88–90%   |

---

## References

* Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*
  [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

* timm: PyTorch Image Models
  [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

---

## License

This project is licensed under the MIT License.

---
