# Vision Transformer (ViT-Small) from Scratch on CIFAR-10

Pure PyTorch implementation of a Vision Transformer (ViT-Small) trained on CIFAR-10 from scratch (no pretrained weights). Achieves ~90% test accuracy with proper training strategy, cosine LR scheduling, and strong data augmentation.

## Features

- Implemented completely from scratch in PyTorch
- Patch Embedding, Multi-Head Self-Attention, MLP blocks, CLS token
- Cosine learning rate scheduler and data augmentation
- ~90% CIFAR-10 test accuracy

## Project Structure

- vit.py — model definition
- train.py — training loop
- requirements.txt — dependencies

## Installation

```bash
pip install -r requirements.txt
    
## Usage/Examples

```javascript
python train.py
```


## Results
~90% test accuracy after 200 epochs on CIFAR-10
## License

[MIT](https://choosealicense.com/licenses/mit/)
