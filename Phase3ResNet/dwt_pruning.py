import os
import pywt
import numpy as np
import torch
import torch.nn as nn
from utils import log_pruning_details, setup_csv_writer, check_and_set_pruned_instance_path, save_model, append_to_experiment_log


def multi_resolution_analysis(weights, wavelet, level, threshold):
    """
    Perform multi-resolution analysis and apply wavelet threshold pruning.

    Args:
        weights (list of numpy.ndarray): List of weight arrays to be pruned.
        wavelet (str): Wavelet type to be used.
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for pruning.

    Returns:
        tuple: A tuple containing pruned weights and total pruned count.
    """
    pruned_weights = []
    total_pruned_count = 0
    for weight in weights:
        original_shape = weight.shape
        flattened_weight = np.ravel(weight)
        coeffs = pywt.wavedec(flattened_weight, wavelet,
                              level=level, mode='periodization')
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

        # Apply threshold
        coeff_arr[np.abs(coeff_arr) < threshold] = 0

        # Reconstruct the weight
        pruned_coeffs = pywt.array_to_coeffs(
            coeff_arr, coeff_slices, output_format='wavedec')
        pruned_weight = pywt.waverec(
            pruned_coeffs, wavelet, mode='periodization')
        pruned_weight = pruned_weight[:np.prod(original_shape)].reshape(
            original_shape)  # Corrected reshaping
        pruned_weights.append(pruned_weight)

        # Calculate the number of pruned weights
        original_non_zero_params = np.count_nonzero(flattened_weight)
        pruned_non_zero_params = np.count_nonzero(pruned_weight)
        total_pruned_count += original_non_zero_params - pruned_non_zero_params

    return pruned_weights, total_pruned_count


def prune_layer_weights(layer, wavelet, level, threshold):
    """
    Prune the weights of a given layer using wavelet-based pruning.

    Args:
        layer (torch.nn.Module): The layer to be pruned.
        wavelet (str): Wavelet type to be used.
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for pruning.

    Returns:
        tuple: A tuple containing the original parameter count, non-zero parameter count after pruning, and total pruned count.
    """
    weights = [param.detach().cpu().numpy() for param in layer.parameters()]
    original_param_count = sum(weight.size for weight in weights)
    non_zero_params = sum(np.count_nonzero(weight) for weight in weights)

    pruned_weights, layer_pruned_count = multi_resolution_analysis(
        weights, wavelet, level, threshold)

    # Update layer weights
    for param, pruned_weight in zip(layer.parameters(), pruned_weights):
        param.data = torch.from_numpy(pruned_weight).to(param.device)

    return original_param_count, non_zero_params, layer_pruned_count


def wavelet_pruning(model, wavelet, level, threshold, csv_path, guid):
    """
    Apply wavelet-based pruning to the model, log the pruning details, and save the pruned model.

    Args:
        model (torch.nn.Module): The model to be pruned.
        wavelet (str): Wavelet type to be used.
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for pruning.
        guid (str): Unique identifier for the pruning session.

    Returns:
        str: Path to the log file containing the pruning details.
    """

    total_pruned_count = 0
    selective_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/selective_pruned")
    selective_log_path = os.path.join(selective_pruned_dir, 'log.csv')
    selective_csv_writer, selective_log_file = setup_csv_writer(
        os.path.normpath(selective_log_path), mode='w')

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d)):
            original_param_count, non_zero_params, layer_pruned_count = prune_layer_weights(
                module, wavelet, level, threshold)
            total_pruned_count += layer_pruned_count
            log_pruning_details(selective_csv_writer, guid, wavelet, level, threshold,
                                'selective', original_param_count, non_zero_params, layer_pruned_count, name)
            print(
                f"Layer: {name}, Original Params: {original_param_count}, Non-zero Params: {non_zero_params}, Pruned Count: {layer_pruned_count}")

    selective_log_file.close()

    # Save the selectively pruned model
    save_model(model, selective_pruned_dir)
    append_to_experiment_log(os.path.normpath(csv_path), guid, wavelet, level,
                             threshold, 'selective', total_pruned_count, non_zero_params, selective_pruned_dir)

    print(f"Selectively pruned model saved at {selective_pruned_dir}")

    return selective_log_path
