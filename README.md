# WaveletTransforms

Applying Discrete Wavelet Transform (DWT) for neural network model compression.

## Wavelet Transforms for Model Compression

Apply, sciencefy and evolve.

## Project Overview

This project aims to leverage Discrete Wavelet Transform (DWT) to compress neural network models. By applying DWT on the weight matrices of neural networks, we threshold small values to zero, increase sparsity, and employ KMeans clustering on the coefficients to reduce the model's size while striving to maintain performance.

## Getting Started

To get started with this project, clone this repository and install the required dependencies listed in `requirements.txt`.

```bash
git clone git@github.com:iAmGiG/WaveletTransforms.git
cd wavelet-transforms
pip install -r requirements.txt
```

### Conda Environment Setup

1. If you haven't already, install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your environments.

2. Create a new Conda environment named `wavelet_env` with Python 3.8.18 for compatibility with PyWavelets. Open your terminal or command prompt and run the following command:

```bash
conda create --name wavelet_env python=3.8.18
```

### Activate the newly created environment

```bash
conda activate wavelet_env
```

### Install the required dependencies. Ensure you are in the project directory where the requirements.txt file is located, then run

```bash
pip install -r requirements.txt
```

This will install the necessary libraries, such as numpy, pywt, scikit-learn, and others, within your wavelet_env environment.

Now, you're ready to start working on the project within this environment. Remember to activate the wavelet_env environment whenever you are working on this project.
