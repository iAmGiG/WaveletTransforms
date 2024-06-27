import os
import csv
from typing import List
import torch
from tqdm import tqdm
from transformers import PreTrainedModel
from main_pruning import log_queue
from utils import setup_csv_writer, log_pruning_details, check_and_set_pruned_instance_path, get_layer, save_model


def absolute_min_pruning(weights: List[torch.Tensor], prune_count: int) -> List[torch.Tensor]:
    """
    Prune weights with the smallest absolute values until the count matches the given prune count.

    This function implements the minimum weight pruning algorithm. It flattens each weight tensor,
    selects the smallest absolute values, and sets them to zero.

    Args:
        weights (List[torch.Tensor]): List of weight tensors to be pruned.
        prune_count (int): The number of weights to prune.

    Returns:
        List[torch.Tensor]: List of pruned weight tensors.
    """
    pruned_weights = []
    for weight in weights:
        flatten_weights = weight.view(-1)
        _, indices_to_prune = torch.topk(
            torch.abs(flatten_weights), prune_count, largest=False)
        flatten_weights[indices_to_prune] = 0
        pruned_weights.append(flatten_weights.view_as(weight))
    return pruned_weights


def min_weight_pruning(selective_log_path: str, model: PreTrainedModel, guid: str, wavelet: str,
                       level: int, threshold: float, csv_path: str) -> None:
    """
    Apply minimum weight pruning to the model based on the selective pruning log.

    This function prunes the model's smallest weights based on the selective pruning log,
    aiming to further reduce the model size while maintaining performance. It processes
    each layer mentioned in the selective pruning log, applies minimum weight pruning,
    and logs the results.

    Args:
        selective_log_path (str): Path to the selective pruning log file.
        model (PreTrainedModel): The pre-trained Hugging Face transformer model to be pruned.
        guid (str): Unique identifier for the pruning session.
        wavelet (str): Wavelet type used in the previous pruning step.
        level (int): Level of wavelet decomposition used in the previous pruning step.
        threshold (float): Threshold value used in the previous pruning step.
        csv_path (str): Path to the CSV file for logging overall experiment results.

    Returns:
        None
    """
    total_pruned_count = 0
    total_non_zero_params = 0
    min_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_min_pruning_threshold-{threshold}_level-{level}_guid-{guid[:4]}/min_pruned")
    min_log_path = os.path.join(min_pruned_dir, 'log.csv')
    min_csv_writer, min_log_file = setup_csv_writer(
        os.path.normpath(min_log_path), mode='w')

    with open(selective_log_path, 'r') as log_file:
        log_reader = csv.DictReader(log_file)
        total_layers = sum(1 for _ in log_reader)  # Count total layers
        log_file.seek(0)  # Reset file pointer
        next(log_reader)  # Skip header

        for row in tqdm(log_reader, total=total_layers, desc="Pruning Progress"):
            layer_name = row['Layer Name']
            original_param_count = int(row['Original Parameter Count'])
            prune_count = int(row['Total Pruned Count'])
            print(
                f"Processing layer: {layer_name} with prune count: {prune_count}")

            layer = get_layer(model, layer_name)
            if layer is not None and hasattr(layer, 'weight'):
                weights = [layer.weight]
                pruned_weights = absolute_min_pruning(weights, prune_count)

                with torch.no_grad():
                    layer.weight.copy_(pruned_weights[0])

                non_zero_params_after_pruning = torch.count_nonzero(
                    pruned_weights[0]).item()

                # Calculate the actual pruned count
                actual_pruned_count = original_param_count - non_zero_params_after_pruning

                log_pruning_details(min_csv_writer, guid, wavelet, level, threshold, 'min',
                                    original_param_count, non_zero_params_after_pruning, actual_pruned_count, layer_name)
                total_pruned_count += actual_pruned_count
                total_non_zero_params += non_zero_params_after_pruning
            else:
                print(
                    f"Layer not found or does not have weights: {layer_name}")

    # Save the minimum weight pruned model
    save_model(model, min_pruned_dir)

    # Use the log queue instead of directly calling append_to_experiment_log
    log_queue.put((guid, wavelet, level, threshold, 'min',
                  total_pruned_count, total_non_zero_params, min_pruned_dir))

    min_log_file.close()
    print("Minimum weight pruning completed.")
