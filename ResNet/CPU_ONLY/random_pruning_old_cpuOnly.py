import os
import csv
import numpy as np
import torch
import torch.nn as nn
from utils import setup_csv_writer, log_pruning_details, append_to_experiment_log, check_and_set_pruned_instance_path, get_layer

def random_pruning(selective_pruning_log, model, guid, wavelet, level, threshold):
    """
    Apply random pruning to the model based on the selective pruning log.
    
    This function randomly prunes the model's weights based on a selective pruning
    log, aiming to further reduce the model size while maintaining performance.

    Args:
        selective_pruning_log (str): Path to the selective pruning log file.
        model (torch.nn.Module): The model to be pruned.
        guid (str): Unique identifier for the pruning session.
        wavelet (str): Wavelet type to be used.
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for pruning.

    Returns:
        None
    """
    total_pruned_count = 0
    total_non_zero_params = 0
    random_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/random_pruned")
    random_log_path = os.path.join(random_pruned_dir, 'log.csv')
    random_csv_writer, random_log_file = setup_csv_writer(
        os.path.normpath(random_log_path), mode='w')

    # Read the selective pruning log file
    with open(selective_pruning_log, 'r') as log_file:
        log_reader = csv.DictReader(log_file)
        for row in log_reader:
            layer_name = row['Layer Name']
            original_param_count = int(row['Original Parameter Count'])
            prune_count = int(row['Total Pruned Count'])
            print(
                f"Processing layer: {layer_name} with prune count: {prune_count}")

            layer = get_layer(model, layer_name)
            if layer and isinstance(layer, nn.Conv2d):
                weights = layer.weight.data.cpu().numpy()
                flatten_weights = weights.flatten()
                indices_to_prune = np.random.choice(
                    len(flatten_weights), prune_count, replace=False)
                flatten_weights[indices_to_prune] = 0
                weights = flatten_weights.reshape(weights.shape)
                layer.weight.data = torch.from_numpy(
                    weights).to(layer.weight.device)
                non_zero_params_after_pruning = np.count_nonzero(weights)

                # Calculate the actual pruned count
                actual_pruned_count = original_param_count - non_zero_params_after_pruning

                log_pruning_details(random_csv_writer, guid, wavelet, level, threshold, 'random',
                                    original_param_count, non_zero_params_after_pruning, actual_pruned_count, layer_name)
                total_pruned_count += actual_pruned_count
                total_non_zero_params += non_zero_params_after_pruning
            else:
                print(f"Layer not found or not a Conv2D layer: {layer_name}")

    # Save the randomly pruned model
    save_model(model, random_pruned_dir)
    # Append to the combined experiment log
    append_to_experiment_log(os.path.normpath(FLAGS.csv_path), guid, wavelet, level, threshold, 'random',
                             total_pruned_count, total_non_zero_params, random_pruned_dir)
    random_log_file.close()
    print("Random pruning completed.")
