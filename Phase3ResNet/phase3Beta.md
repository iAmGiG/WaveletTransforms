# Deep Learning Model Pruning

This repository contains the code for pruning deep learning models using both wavelet-based and random pruning methods. The architecture is modular, making it easy to modify or extend the pruning techniques.

## Suggested Architecture

1. **Main Pruning Script (`main_pruning.py`)**: Orchestrates the entire pruning process.
2. **Wavelet-Based Pruning Module (`dwt_pruning.py`)**: Implements the wavelet-based pruning logic.
3. **Random Pruning Script (`random_pruning.py`)**: Performs random pruning based on the output from the wavelet-based pruning.
4. **Utility Functions (`utils.py`)**: Provides common functions used across the project.

## Workflow

### Main Pruning Script (`main_pruning.py`)

- Loads the model from a specified path.
- Performs wavelet-based pruning using the `dwt_pruning.py` module.
- Saves the pruned model and logs the details to a CSV file.
- Prepares parameters (such as the total prune count and model path) for the random pruning step.

### Wavelet-Based Pruning Module (`dwt_pruning.py`)

- Handles the wavelet-based pruning of the model.
- Logs detailed information about the pruning process, including parameters and outcomes for each layer.

### Random Pruning Script (`random_pruning.py`)

- Executes random pruning based on the prune count obtained from the wavelet-based pruning.
- Saves the randomly pruned model and logs the details.

### Utility Functions (`utils.py`)

- Contains functions for loading models, saving models, logging pruning details, and setting up CSV writers.

## Usage

To use this repository, follow these steps:

1. Configure the paths and parameters in the flags section of `main_pruning.py`.
2. Run `main_pruning.py` to perform wavelet-based pruning and prepare for random pruning.
3. Execute `random_pruning.py` with the appropriate parameters to perform random pruning.
