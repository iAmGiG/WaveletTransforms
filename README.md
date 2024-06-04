# WaveletTransforms

Applying Discrete Wavelet Transform (DWT) for neural network model compression.

## Wavelet Transforms for Model Compression

Apply, sciencefy, and evolve.

## Project Overview

This project aims to leverage Discrete Wavelet Transform (DWT) to compress neural network models. By applying DWT on the weight matrices of neural networks, we threshold small values to zero, increase sparsity, and employ KMeans clustering on the coefficients to reduce the model's size while striving to maintain performance.

## Tools and Methods

### Neural Network Models
- **Initial Phase**: Multi-Layer Perceptrons (MLP) trained on the MNIST dataset.
- **Current Focus**: ResNet-18 models trained on the ImageNet-1k dataset using PyTorch.

### Wavelet Transform
- **Wavelet Types**: Various types of wavelets, with a focus on the Haar wavelet.
- **Decomposition Levels**: Multiple levels of decomposition (0 to 3) to analyze and prune different layers of the model.

### Pruning Methodology
- **Discrete Wavelet Transform (DWT)**: Applied to the weight matrices of neural networks.
- **Thresholding**: Threshold small values to zero to increase sparsity.
- **KMeans Clustering**: Employed on the coefficients to further reduce model size.

### Evaluation Metrics
- **Performance Metrics**: Classification accuracy, loss, F1 score, recall.
- **Model Metrics**: Sparsity, model size, precision, average inference time.
- **Visualization Tools**: Matplotlib, Seaborn for visualizing results and comparisons.

## Getting Started

To get started with this project, clone this repository and install the required dependencies listed in `requirements.txt`.

```bash
git clone git@github.com:iAmGiG/WaveletTransforms.git
cd WaveletTransforms
pip install -r requirements.txt
```

### Conda Environment Setup

1. If you haven't already, install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your environments.

2. Create a new Conda environment named wavelet_env with Python 3.9.15 for compatibility with the project's dependencies. Open your terminal or command prompt and run the following command:
```bash
conda create --name wavelet_env python=3.9.15
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

### Project Structure
- **models/:** Contains the neural network models used in the project.
- **data/:** Directory for storing datasets.
- **notebooks/:** Jupyter notebooks for experimentation and analysis.
- **scripts/:** Python scripts for various tasks such as training, evaluation, and pruning.
- **results/:** Directory for storing results, including model weights and performance metrics.

### Recent Developments
- **Wavelet Pruning:** Continued development of a detailed methodology for applying wavelet-based pruning using PyWavelet.
- **Model Testing:** Transitioned to testing with CIFAR-10 dataset for effective evaluation results but this might need to be changed.
- **Expanded Metrics:** Expanded the evaluation metrics to include precision, average inference time, and size comparisons, despite some challenges in obtaining significant insights from inference time measurements.
