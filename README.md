# WaveletTransforms
Applying Discrete Wavelet Transform (DWT) for neural network model compression.

# Wavelet Transforms for Model Compression

## Project Overview

This project aims to leverage Discrete Wavelet Transform (DWT) to compress neural network models. By applying DWT on the weight matrices of neural networks, we threshold small values to zero, increase sparsity, and employ KMeans clustering on the coefficients to reduce the model's size while striving to maintain performance.

## Getting Started

To get started with this project, clone this repository and install the required dependencies listed in `requirements.txt`.

```bash
git clone https://github.com/<your-username>/wavelet-transforms.git
cd wavelet-transforms
pip install -r requirements.txt
```
