# Wavelet-Based Pruning of ResNet Models

This project aims to prune pre-trained ResNet models using wavelet-based techniques and random pruning. The pruning process involves removing less significant weights from the model to reduce its size and computational complexity while maintaining acceptable performance levels. The project is organized into modular components for better maintainability and reusability.

## Suggested Architecture

1. **Main Pruning Script (`main_pruning.py`)**
2. **Wavelet-Based Pruning Module (`dwt_pruning.py`)**
3. **Random Pruning Script (`random_pruning.py`)**
4. **Utility Functions (`utils.py`)**

## Detailed Breakdown

### 1. Main Pruning Script (`main_pruning.py`)

Handles the overall flow and coordinates the pruning process.

### 2. Wavelet-Based Pruning Module (`dwt_pruning.py`)

Contains the logic for wavelet-based pruning.

### 3. Random Pruning Script (`random_pruning.py`)

Executes random pruning based on the prune count obtained from the wavelet-based pruning.

### 4. Utility Functions (`utils.py`)

Contains common functions for loading models, saving models, logging, and setting up the CSV writer.

## Workflow

### Main Pruning Script

- Loads the model.
- Performs wavelet-based pruning using `dwt_pruning.py`.
- Saves the pruned model and logs the details.
- Prepares parameters for the random pruning step.

### Wavelet-Based Pruning Module

- Handles wavelet-based pruning and logging of details.

### Random Pruning Script

- Handles random pruning based on prune count and logging of details.

### Utility Functions

- Common functions for model loading, saving, logging, and setting up the CSV writer.
