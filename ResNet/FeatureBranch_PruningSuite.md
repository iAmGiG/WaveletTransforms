## DWT Pruning Update

### Overview

This update introduces an improved method for Discrete Wavelet Transform (DWT) pruning in Conv2D layers of ResNet models.

### Changes

- **Efficient Multi-Resolution Analysis**: Utilizing `pywt.dwt2` and `pywt.idwt2` for direct multi-dimensional operations, avoiding the inefficiency of flattening and reshaping tensors.
- **Layer Handling**: Explicitly targeting Conv2D layers, with the potential to extend to other layer types.
- **Error Handling**: Added robust error handling for wavelet transformation and reconstruction.
- **Validation**: Ensuring reconstructed weights match original shapes.

### Benefits

- **Controlled Pruning**: More accurate thresholding reduces over-pruning.
- **Accurate Counts**: Improved error handling and validation lead to better parameter counting.

### Upcoming Experiment

We will compare the DWT pruning with a new minimum impact pruning method based on the percentage prune from the DWT solution.

### Future Work

- **Implement Minimum Impact Pruning**: Identify and prune weights with the least impact.
- **Run Comparative Experiments**: Utilize HPC and Docker containers to conduct and manage these experiments.

### Docker Integration

We are also setting up Docker containers to facilitate GPU-based experiments. This will enable more efficient and scalable testing environments.

### Branch Information

All updates are being implemented in the `feature/dwt-pruning-update` branch.
