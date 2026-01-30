# Vision Transformer (ViT-Small) from Scratch on CIFAR-10

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

A pure PyTorch implementation of the Vision Transformer (ViT-Small) architecture, trained from scratch on the CIFAR-10 dataset without using any pre-trained weights.

This project demonstrates that with a proper training strategy—specifically strong data augmentation and cosine learning rate scheduling—a transformer-based model can achieve competitive performance (~90% accuracy) on small datasets like CIFAR-10, challenging the inductive bias advantage typically held by CNNs.

## 🚀 Key Features

* **Clean Implementation:** Built entirely from scratch in PyTorch (no external model libraries).
* **Full Architecture:** Includes custom implementations of Patch Embedding, Multi-Head Self-Attention (MSA), MLP blocks, and the CLS token mechanism.
* **Optimized Training:** Utilizes Cosine Learning Rate Scheduling and heavy data augmentation (RandomCrop, HorizontalFlip, etc.) to prevent overfitting.
* **High Performance:** Achieves **~90% Top-1 Accuracy** on the CIFAR-10 test set after 200 epochs.

## 📂 Project Structure

```plaintext
├── vit.py             # Model architecture definition (ViT-Small)
├── train.py           # Training loop, data loading, and evaluation
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/vit-cifar10.git](https://github.com/yourusername/vit-cifar10.git)
   cd vit-cifar10
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

To start training the model from scratch, run the training script. This will download the CIFAR-10 dataset automatically if it is not present.

```bash
python train.py
```

*Note: Training logs and checkpoints will be saved to the current directory unless configured otherwise.*

## 📊 Results

The model was trained for 200 epochs using the Adam optimizer and a cosine annealing scheduler.

| Metric | Value |
| :--- | :--- |
| **Dataset** | CIFAR-10 |
| **Model** | ViT-Small (Custom) |
| **Epochs** | 200 |
| **Test Accuracy** | **~90.0%** |

## 📜 References

This implementation is based on the concepts introduced in the original paper:
* *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))

## 📄 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
