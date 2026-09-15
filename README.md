# Phobos-GPT-v1

GPT From-scratch implementation of GPT-2 (124M) language model in PyTorch, covering the transformers architectire and training.

---

## Overview

This was an project I made in a week to learn more about LLM's, the components are:

- Byte Pair Encoding (BPE) tokenization via 'tiktoken' (Note: This was not written from scratch and was imported from the tiktoken library)
- Token and positional embeddings
- Multi-Head casual attention with scaled dot-product scores
- Transformers vlocks with pre-norm residual connections
- Next-token prediction training loop with loss tracking

The implementation is presented in python files as it is easier for me to work with files instead of notebooks in my IDE.

---

## Prerequisities

- Python 3.14
- CUDA-capable GPU for training (Google Colab is fine)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Ansh-droid-glitch/phobos-gpt-v1.git
cd phobos-gpt-v1

# Create and activate a virtual python enviornment
python -m venv
source venv/bin/activate # macOS / Linux
# venv\bin\activate # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Make sure you are in your virtual python enviornment
python train.py
```

---

## Todo

- [ ] Add ROC-m suppot
- [ ] Make and train Byte-Pair Encoding (BPE) from scratch