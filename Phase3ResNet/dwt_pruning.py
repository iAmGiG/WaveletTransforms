import os
import pywt
import numpy as np
import torch
import torch.nn as nn
from utils import log_pruning_details, setup_csv_writer, check_and_set_pruned_instance_path, save_model


def multi_resolution_analysis(weights, wavelet, level, threshold):
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
    weights = [param.detach().cpu().numpy() for param in layer.parameters()]
    original_param_count = sum(weight.size for weight in weights)
    non_zero_params = sum(np.count_nonzero(weight) for weight in weights)

    pruned_weights, layer_pruned_count = multi_resolution_analysis(
        weights, wavelet, level, threshold)

    # Update layer weights
    for param, pruned_weight in zip(layer.parameters(), pruned_weights):
        param.data = torch.from_numpy(pruned_weight).to(param.device)

    return original_param_count, non_zero_params, layer_pruned_count


def wavelet_pruning(model, wavelet, level, threshold, guid):
    total_pruned_count = 0
    selective_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/selective_pruned")
    selective_log_path = os.path.join(selective_pruned_dir, 'log.csv')
    selective_csv_writer, selective_log_file = setup_csv_writer(
        os.path.normpath(selective_log_path), mode='w')

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
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
    print(f"Selectively pruned model saved at {selective_pruned_dir}")

    return selective_log_path
