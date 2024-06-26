import os
import csv
import torch
import torch.nn as nn
from utils import setup_csv_writer, log_pruning_details, append_to_experiment_log, check_and_set_pruned_instance_path, get_layer, save_model


def absolute_min_pruning(weights, prune_count):
    """
    Prune weights with the smallest absolute values until the count matches the given prune count.

    Args:
        weights (list of torch.Tensor): List of weight tensors to be pruned.
        prune_count (int): The number of weights to prune.

    Returns:
        pruned_weights (list of torch.Tensor): List of pruned weight tensors.
    """
    pruned_weights = []
    for weight in weights:
        flatten_weights = weight.view(-1)
        _, indices_to_prune = torch.topk(
            torch.abs(flatten_weights), prune_count, largest=False)
        flatten_weights[indices_to_prune] = 0
        pruned_weights.append(flatten_weights.view_as(weight))
    return pruned_weights


def min_weight_pruning(selective_log_path, model, guid, wavelet, level, threshold, csv_path):
    """
    Apply minimum weight pruning to the model based on the selective pruning log.

    This function prunes the model's smallest weights based on the selective pruning log,
    aiming to further reduce the model size while maintaining performance.

    Args:
        selective_log_path (str): Path to the selective pruning log file.
        model (torch.nn.Module): The model to be pruned.
        guid (str): Unique identifier for the pruning session.
        wavelet (str): Wavelet type to be used.
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for pruning.
        csv_path (str): Path to the CSV file for logging.

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

    # Read the selective pruning log file
    with open(selective_log_path, 'r') as log_file:
        log_reader = csv.DictReader(log_file)
        for row in log_reader:
            layer_name = row['Layer Name']
            original_param_count = int(row['Original Parameter Count'])
            prune_count = int(row['Total Pruned Count'])
            print(
                f"Processing layer: {layer_name} with prune count: {prune_count}")

            layer = get_layer(model, layer_name)
            if layer and isinstance(layer, nn.Conv2d):
                weights = [param for param in layer.parameters()]
                pruned_weights = absolute_min_pruning(weights, prune_count)

                with torch.no_grad():
                    for param, pruned_weight in zip(layer.parameters(), pruned_weights):
                        param.copy_(pruned_weight)

                non_zero_params_after_pruning = sum(torch.count_nonzero(
                    weight).item() for weight in pruned_weights)

                # Calculate the actual pruned count
                actual_pruned_count = original_param_count - non_zero_params_after_pruning

                log_pruning_details(min_csv_writer, guid, wavelet, level, threshold, 'min',
                                    original_param_count, non_zero_params_after_pruning, actual_pruned_count, layer_name)
                total_pruned_count += actual_pruned_count
                total_non_zero_params += non_zero_params_after_pruning
            else:
                print(f"Layer not found or not a Conv2D layer: {layer_name}")

    # Save the minimum weight pruned model
    save_model(model, min_pruned_dir)
    # Append to the combined experiment log
    append_to_experiment_log(os.path.normpath(csv_path), guid, wavelet, level, threshold, 'min',
                             total_pruned_count, total_non_zero_params, min_pruned_dir)
    min_log_file.close()
    print("Minimum weight pruning completed.")
