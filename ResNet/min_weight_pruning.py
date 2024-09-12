"""
Minimum Weight Pruning Module

This module implements a minimum weight pruning technique for neural network models.
It prunes a specified percentage of the smallest weights in each layer of the model,
based on their absolute values.

Key Components:
1. percentage_min_pruning: Function to prune a percentage of the smallest weights in a tensor.
2. min_weight_pruning: Main function to apply minimum weight pruning across the model.

The pruning process uses a log file from a previous selective (DWT) pruning step to
identify layers and their original parameter counts. It then applies percentage-based
pruning to each layer, removing the specified proportion of the smallest weights.

Usage:
This module is typically called from the main pruning script (main_pruning.py) and
works in conjunction with other pruning techniques for comparison purposes.

The threshold parameter (0.0 to 1.0) determines the percentage of weights to prune
in each layer. For example, a threshold of 0.1 will prune the smallest 10% of weights
in each layer.

Dependencies:
- torch: For tensor operations
- transformers: For working with pre-trained models
- utils: Custom utility functions for logging and model operations

Note:
This pruning technique aims to provide a consistent pruning approach across all layers,
making it more comparable to other pruning methods like DWT-based pruning.
"""
import os
import csv
from typing import Optional
from queue import Queue
import torch
from transformers import PreTrainedModel
from utils import setup_csv_writer, log_pruning_details, check_and_set_pruned_instance_path, get_layer, save_model, append_to_experiment_log


def calculate_dwt_pruning_percentage(selective_log_path: str) -> float:
    """
    Calculate the overall pruning percentage from the DWT-based pruning log.

    Args:
        selective_log_path (str): Path to the selective pruning log file.

    Returns:
        float: The overall pruning percentage.
    """
    total_params = 0
    total_pruned = 0

    with open(selective_log_path, 'r') as log_file:
        log_reader = csv.DictReader(log_file)
        for row in log_reader:
            total_params += int(row['Original Parameter Count'])
            total_pruned += int(row['Pruned Parameter Count'])

    return total_pruned / total_params if total_params > 0 else 0.0


def percentage_min_pruning(weights: torch.Tensor, prune_percentage: float) -> torch.Tensor:
    """
    Prune a percentage of weights with the smallest absolute values.

    Args:
        weights (torch.Tensor): Weight tensor from a model layer.
        prune_percentage (float): Percentage of weights to prune (0.0 to 1.0).

    Returns:
        torch.Tensor: The pruned weight tensor.
    """
    flatten_weights = weights.view(-1)
    total_weights = flatten_weights.numel()
    prune_count = int(total_weights * prune_percentage)

    abs_weights = torch.abs(flatten_weights)
    _, indices_to_prune = torch.topk(abs_weights, prune_count, largest=False)
    flatten_weights[indices_to_prune] = 0
    return flatten_weights.view_as(weights)


def min_weight_pruning(selective_log_path: str, model: PreTrainedModel, guid: str, wavelet: str,
                       level: int, threshold: float, csv_path: str, log_queue: Optional[Queue] = None) -> None:
    """
    Apply minimum weight pruning based on the overall pruning percentage from DWT-based pruning.

    Args:
        selective_log_path (str): Path to the selective pruning log file.
        model (PreTrainedModel): The pre-trained Hugging Face transformer model to be pruned.
        guid (str): Unique identifier for the pruning session.
        wavelet (str): Wavelet type used in the previous pruning step.
        level (int): Level of wavelet decomposition used in the previous pruning step.
        threshold (float): Threshold value used in the previous pruning step (not used for percentage-based pruning).
        csv_path (str): Path to the CSV file for logging overall experiment results.
        log_queue (Optional[Queue]): Queue for thread-safe logging (if used in threading context).

    Returns:
        None
    """
    min_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/min_pruned")
    min_log_path = os.path.join(min_pruned_dir, 'log.csv')
    min_csv_writer, min_log_file = setup_csv_writer(min_log_path, mode='w')

    # Calculate the overall pruning percentage from DWT-based pruning
    overall_prune_percentage = calculate_dwt_pruning_percentage(
        selective_log_path)
    print(
        f"Overall pruning percentage from DWT: {overall_prune_percentage:.2%}")

    total_params = 0
    total_pruned = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            original_param_count = module.weight.numel()
            pruned_weights = percentage_min_pruning(
                module.weight.data, overall_prune_percentage)

            with torch.no_grad():
                module.weight.copy_(pruned_weights)

            non_zero_params_after_pruning = torch.count_nonzero(
                pruned_weights).item()
            actual_pruned_count = original_param_count - non_zero_params_after_pruning

            log_pruning_details(min_csv_writer, guid, wavelet, level, threshold, 'min',
                                original_param_count, non_zero_params_after_pruning, actual_pruned_count, name)

            total_params += original_param_count
            total_pruned += actual_pruned_count

    save_model(model, min_pruned_dir)
    if log_queue is not None:
        log_queue.put((guid, wavelet, level, threshold, 'min',
                      total_pruned, total_params - total_pruned, min_pruned_dir))
    else:
        append_to_experiment_log(csv_path, guid, wavelet, level, threshold,
                                 'min', total_pruned, total_params - total_pruned, min_pruned_dir)

    min_log_file.close()
    print(
        f"Minimum weight pruning completed. Pruned {total_pruned} out of {total_params} parameters.")
    print(f"Actual pruning percentage: {total_pruned/total_params:.2%}")
