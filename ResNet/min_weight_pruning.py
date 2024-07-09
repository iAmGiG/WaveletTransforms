import os
import csv
from typing import List, Optional
from queue import Queue
import torch
from transformers import PreTrainedModel
from utils import setup_csv_writer, log_pruning_details, check_and_set_pruned_instance_path, get_layer, save_model

def absolute_min_pruning(weights: torch.Tensor, prune_count: int) -> torch.Tensor:
    """
    Prune weights with the smallest absolute values.

    Args:
        weights (torch.Tensor): Weight tensor from a model layer.
        prune_count (int): Number of weights to prune to zero.

    Returns:
        torch.Tensor: The pruned weight tensor.
    """
    flatten_weights = weights.view(-1)
    abs_weights = torch.abs(flatten_weights)
    _, indices_to_prune = torch.topk(abs_weights, prune_count, largest=False)
    flatten_weights[indices_to_prune] = 0
    return flatten_weights.view_as(weights)

def min_weight_pruning(selective_log_path: str, model: PreTrainedModel, guid: str, wavelet: str, 
                       level: int, threshold: float, csv_path: str, log_queue: Optional[Queue] = None) -> None:
    """
    Apply minimum weight pruning based on the selective pruning log.

    Perform minimum weight pruning on a model post-wavelet-based pruning. This function targets 
    the smallest weights left after initial pruning to refine the model's efficiency. It uses a 
    log from the prior pruning phase to identify layers and their state post-pruning, applying 
    additional pruning to further reduce weight counts by targeting the least impactful weights.

    Args:
        selective_log_path (str): Path to the selective pruning log file.
        model (PreTrainedModel): The pre-trained Hugging Face transformer model to be pruned.
        guid (str): Unique identifier for the pruning session.
        wavelet (str): Wavelet type used in the previous pruning step.
        level (int): Level of wavelet decomposition used in the previous pruning step.
        threshold (float): Threshold value used in the previous pruning step.
        csv_path (str): Path to the CSV file for logging overall experiment results.
        log_queue (Optional[Queue]): Queue for thread-safe logging (if used in threading context).

    Returns:
        None
    """
    min_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/min_pruned")
    min_log_path = os.path.join(min_pruned_dir, 'log.csv')
    min_csv_writer, min_log_file = setup_csv_writer(min_log_path, mode='w')

    with open(selective_log_path, 'r') as log_file:
        log_reader = csv.DictReader(log_file)
        for row in log_reader:
            layer_name = row['Layer Name']
            original_param_count = int(row['Original Parameter Count'])
            already_pruned_count = int(row['Total Pruned Count'])
            layer = get_layer(model, layer_name)
            if layer is not None and hasattr(layer, 'weight'):
                prune_count = max(0, original_param_count - already_pruned_count) // 10  # Further prune 10% of the remaining weights
                pruned_weights = absolute_min_pruning(layer.weight.data, prune_count)
                with torch.no_grad():
                    layer.weight.copy_(pruned_weights)

                non_zero_params_after_pruning = torch.count_nonzero(pruned_weights).item()
                actual_pruned_count = original_param_count - non_zero_params_after_pruning
                log_pruning_details(min_csv_writer, guid, wavelet, level, threshold, 'min',
                                    original_param_count, non_zero_params_after_pruning, actual_pruned_count, layer_name)
            else:
                print(f"Layer not found or does not have weights: {layer_name}")

    save_model(model, min_pruned_dir)
    if log_queue is not None:
        log_queue.put((guid, wavelet, level, threshold, 'min', actual_pruned_count, non_zero_params_after_pruning, min_pruned_dir))
    else:
        append_to_experiment_log(csv_path, guid, wavelet, level, threshold, 'min', actual_pruned_count, non_zero_params_after_pruning, min_pruned_dir)

    min_log_file.close()
    print("Minimum weight pruning completed and model saved.")

