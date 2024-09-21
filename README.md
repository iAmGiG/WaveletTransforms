# WaveletTransforms

Applying Discrete Wavelet Transform (DWT) and Minimum Weight Pruning for neural network model compression.

## Project Overview

This project explores advanced model compression techniques for neural networks, focusing on applying the Discrete Wavelet Transform (DWT) and Minimum Weight Pruning methods. By leveraging these techniques, we aim to reduce model sizes significantly while maintaining or minimally impacting performance. The ultimate goal is to enhance the efficiency of neural networks, making them more suitable for deployment in resource-constrained environments.

## Table of Contents

- [WaveletTransforms](#wavelettransforms)
  - [Project Overview](#project-overview)
  - [Key Techniques](#key-techniques)
    - [Discrete Wavelet Transform (DWT) Pruning](#discrete-wavelet-transform-dwt-pruning)
    - [Minimum Weight Pruning](#minimum-weight-pruning)
    - [Random Pruning](#random-pruning)
  - [Theoretical Background](#theoretical-background)
    - [Why Wavelets?](#why-wavelets)
    - [Comparison of Pruning Methods](#comparison-of-pruning-methods)
  - [Tools and Methods](#tools-and-methods)
    - [Neural Network Models](#neural-network-models)
    - [Wavelet Transform Details](#wavelet-transform-details)
    - [Pruning Methodology](#pruning-methodology)
    - [Evaluation Metrics](#evaluation-metrics)
  - [Getting Started](#getting-started)
    - [Conda Environment Setup](#conda-environment-setup)
    - [Project Structure](#project-structure)
  - [Recent Developments](#recent-developments)

## Key Techniques

### Discrete Wavelet Transform (DWT) Pruning

DWT pruning leverages the mathematical properties of wavelets to analyze and compress the weights of neural network models:

- **Multi-Resolution Analysis:** Decomposes weight tensors into various frequency components, capturing information at different scales.
- **Thresholding:** Applies percentile-based thresholding to wavelet coefficients, setting insignificant coefficients to zero, thus increasing sparsity.
- **Reconstruction:** Pruned coefficients are reconstructed back into the spatial domain, resulting in sparser weight tensors.

**Advantages:**

- Preserves important features across multiple scales.
- Introduces structured sparsity, which can be more hardware-friendly.
- Potentially maintains model performance better due to multi-scale analysis.

### Minimum Weight Pruning

Minimum weight pruning is a straightforward method where weights with the smallest absolute values are set to zero:

- **Selection Criterion:** Prunes a percentage of weights with the smallest magnitudes within each layer.
- **Application:** Applied specifically to convolutional layers to ensure consistency and fair comparison with DWT pruning.

**Advantages:**

- Simple and easy to implement.
- Computationally efficient with minimal overhead.
- Effective in reducing model size.

### Random Pruning

As a baseline comparison, random pruning removes weights at random:

- **Selection Criterion:** Randomly selects a percentage of weights to prune.
- **Purpose:** Provides a benchmark to assess the effectiveness of structured pruning methods like DWT and minimum weight pruning.

## Theoretical Background

### Why Wavelets?

Wavelets provide a powerful tool for signal processing and analysis due to their ability to represent data at multiple resolutions:

- **Localized Analysis:** Wavelets can capture both frequency and location information, making them ideal for analyzing weight tensors in neural networks.
- **Compression Capability:** By focusing on significant wavelet coefficients, less important details can be discarded, leading to model compression.
- **Structured Sparsity:** Wavelet-based pruning can create sparsity patterns that are more structured compared to unstructured methods, potentially benefiting hardware acceleration.

### Comparison of Pruning Methods

- **DWT Pruning vs. Minimum Weight Pruning:**
  - **DWT Pruning** considers the spatial and frequency information of weights, potentially preserving important features better.
  - **Minimum Weight Pruning** is simpler but may prune weights that are small in magnitude yet important for certain features.

- **Theoretical Performance:**
  - DWT pruning is expected to perform better in theory due to its multi-resolution analysis and structured approach.
  - Empirical results are necessary to validate theoretical expectations.

## Tools and Methods

### Neural Network Models

- **Current Focus:** ResNet-18 models trained on the ImageNet-1k dataset using PyTorch.
- **Previous Work:** Initial experiments were conducted on Multi-Layer Perceptrons (MLP) trained on the MNIST dataset.

### Wavelet Transform Details

- **Wavelet Types:** Utilizing various discrete wavelets from PyWavelets, with a focus on biorthogonal wavelets like `bior` and `rbio` for their symmetry properties.
- **Decomposition Levels:** Multiple levels (up to 5) to capture features at different scales.

### Pruning Methodology

- **DWT Pruning:**
  - **Process:**
    - Apply DWT to convolutional layer weights.
    - Perform percentile-based thresholding on wavelet coefficients.
    - Reconstruct pruned weights using inverse DWT.
  - **Implementation Notes:**
    - Only convolutional layers (`nn.Conv2d`) are pruned for consistency.
    - Thresholds are adjusted based on percentiles (e.g., 10%, 50%).

- **Minimum Weight Pruning:**
  - **Process:**
    - Calculate the overall pruning percentage from DWT pruning logs.
    - Prune the same percentage of weights with the smallest absolute values in convolutional layers.
  - **Implementation Notes:**
    - Ensures a fair comparison by pruning the same set of layers as DWT pruning.

- **Random Pruning:**
  - **Process:**
    - Randomly prune weights in convolutional layers based on the pruning percentage from DWT pruning.

### Evaluation Metrics

- **Performance Metrics:**
  - Classification accuracy
  - Loss
  - F1 score
  - Recall
- **Model Metrics:**
  - Sparsity levels
  - Model size reduction
  - Average inference time
- **Visualization Tools:**
  - Matplotlib and Seaborn for plotting results and comparisons.

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
- **utils/:** Utility functions for loading models, logging, and other common tasks.

### Recent Developments

**Refactored Pruning Logic:**
- Ensured consistency between DWT and Minimum Weight Pruning by pruning only convolutional layers.
- Resolved discrepancies in layer selection and pruning statistics.
  
**Expanded Theoretical Framework:**
- Added detailed comparisons between DWT pruning and minimum weight pruning.
- Included discussions on the advantages, disadvantages, and theoretical expectations of each method.
  
**Wavelet Selection:**
- Experimented with different wavelet types, focusing on biorthogonal wavelets for their symmetry and effectiveness in preserving convolutional filter structures.
  
**Improved Logging and Reporting:**
- Enhanced logging to include detailed per-layer pruning statistics.
- Standardized log formats for easier comparison and analysis.
  
**Evaluation Enhancements:**
- Expanded evaluation metrics to provide a comprehensive assessment of model performance post-pruning.
- Addressed challenges in measuring inference time and its significance.
