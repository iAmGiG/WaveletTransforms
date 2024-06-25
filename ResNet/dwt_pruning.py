import os
import pywt
import numpy as np
import torch
import torch.nn as nn
from utils import log_pruning_details, setup_csv_writer, check_and_set_pruned_instance_path, save_model, append_to_experiment_log


def multi_resolution_analysis(weights, wavelet, level, threshold, mode='periodization'):
    """
    Perform multi-resolution analysis and apply wavelet threshold pruning.

    Args:
        weights (list of torch.Tensor): List of weight tensors to be pruned.
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
        device = weight.device

        # Convert tensor to numpy array for wavelet processing
        weight_np = weight.detach().cpu().numpy()

        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(weight_np, wavelet, level=level, mode=mode, axes=(-2, -1))
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs, axes=(-2, -1))

        # Apply threshold
        coeff_arr[np.abs(coeff_arr) < threshold] = 0

        # Reconstruct the weight
        pruned_coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
        pruned_weight_np = pywt.waverec2(pruned_coeffs, wavelet, mode=mode, axes=(-2, -1))


        # Ensure the pruned weight has the same shape as the original
        pruned_weight_np = pruned_weight_np[:original_shape[0], :original_shape[1], :original_shape[2], :original_shape[3]]

        # Convert pruned weight back to tensor
        pruned_weight = torch.tensor(pruned_weight_np, dtype=weight.dtype, device=device)
        pruned_weights.append(pruned_weight)

        # Calculate the number of pruned weights
        original_non_zero_params = torch.count_nonzero(weight).item()
        pruned_non_zero_params = torch.count_nonzero(pruned_weight).item()
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
    weights = [param for param in layer.parameters()]
    original_param_count = sum(weight.numel() for weight in weights)
    non_zero_params = sum(torch.count_nonzero(weight).item() for weight in weights)

    pruned_weights, layer_pruned_count = multi_resolution_analysis(weights, wavelet, level, threshold)

    # Update layer weights
    with torch.no_grad():
        for param, pruned_weight in zip(layer.parameters(), pruned_weights):
            param.copy_(pruned_weight)

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
    selective_pruned_dir = check_and_set_pruned_instance_path(f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/selective_pruned")
    selective_log_path = os.path.join(selective_pruned_dir, 'log.csv')
    selective_csv_writer, selective_log_file = setup_csv_writer(os.path.normpath(selective_log_path), mode='w')

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d)):
            original_param_count, non_zero_params, layer_pruned_count = prune_layer_weights(module, wavelet, level, threshold)
            total_pruned_count += layer_pruned_count
            log_pruning_details(selective_csv_writer, guid, wavelet, level, threshold, 'selective', original_param_count, non_zero_params, layer_pruned_count, name)
            print(f"Layer: {name}, Original Params: {original_param_count}, Non-zero Params: {non_zero_params}, Pruned Count: {layer_pruned_count}")

    selective_log_file.close()

    # Save the selectively pruned model
    save_model(model, selective_pruned_dir)
    append_to_experiment_log(os.path.normpath(csv_path), guid, wavelet, level, threshold, 'selective', total_pruned_count, non_zero_params, selective_pruned_dir)

    print(f"Selectively pruned model saved at {selective_pruned_dir}")

    return selective_log_path
