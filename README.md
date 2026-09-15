# Phobos-GPT-v1

GPT From-scratch implementation of GPT-2 (124M) language model in PyTorch, covering the transformers architectire and training.

---

## Overview

This was an project i made in a week to learn more about LLM's, the components are:

- Byte Pair Encoding (BPE) tokenization via 'tiktoken' (Note: This as not written from scratch and was imported from the tiktoken library)
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
```

---

## Todo

- [ ] Add ROC-m suppot
- [ ] Make and train Byte-Pair Encoding (BPE) from scratch