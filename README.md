# Vision Transformer (ViT) from Scratch on CIFAR-10

This project implements a Vision Transformer (ViT-Small) completely from scratch in PyTorch and trains it on the CIFAR-10 dataset to achieve ~90% test accuracy.

## Features
- Pure PyTorch implementation (no pretrained models)
- Patch embedding, MHSA, MLP blocks, CLS token
- Cosine LR scheduler and data augmentation
- ~90% CIFAR-10 accuracy

## Project Structure
vit.py — model  
train.py — training loop  
requirements.txt — dependencies  

## Setup
pip install -r requirements.txt

## Train
python train.py

## Result
~90% Test Accuracy after 200 epochs

## License
MIT