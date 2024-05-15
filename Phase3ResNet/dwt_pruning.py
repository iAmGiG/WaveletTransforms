import pywt
import numpy as np
from utils import log_pruning_details


def prune_layer_weights(layer, wavelet, level, threshold):
    weights = layer.get_weights()
    if not weights:
        return weights, 0

    pruned_weights = []
    total_pruned_count = 0
    for weight in weights:
        original_shape = weight.shape
        flattened_weight = np.ravel(weight)

        coeffs = pywt.wavedec(flattened_weight, wavelet,
                              level=level, mode='periodization')
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        pruned_count = np.sum(np.abs(coeff_arr) < threshold)
        coeff_arr[np.abs(coeff_arr) < threshold] = 0
        pruned_coeffs = pywt.array_to_coeffs(
            coeff_arr, coeff_slices, output_format='wavedec')
        pruned_weight = pywt.waverec(
            pruned_coeffs, wavelet, mode='periodization')

        if pruned_weight.size > flattened_weight.size:
            pruned_weight = pruned_weight[:flattened_weight.size]
        elif pruned_weight.size < flattened_weight.size:
            pruned_weight = np.pad(
                pruned_weight, (0, flattened_weight.size - pruned_weight.size), 'constant')

        pruned_weight = np.reshape(pruned_weight, original_shape)
        pruned_weights.append(pruned_weight)
        total_pruned_count += pruned_count

    original_param_count = sum(weight.size for weight in weights)
    non_zero_params = original_param_count - total_pruned_count

    return pruned_weights, total_pruned_count, original_param_count, non_zero_params


def wavelet_pruning(model, wavelet, level, threshold, csv_writer, guid):
    total_prune_count = 0
    for layer in model.layers:
        if layer.trainable and layer.get_weights():
            try:
                pruned_weights, layer_pruned_count, original_param_count, non_zero_params = prune_layer_weights(
                    layer, wavelet, level, threshold
                )
                layer.set_weights(pruned_weights)
                total_prune_count += layer_pruned_count
                log_pruning_details(csv_writer, guid, wavelet, level, threshold, 'selective',
                                    original_param_count, non_zero_params, layer_pruned_count, layer.name)
            except Exception as e:
                print(f"Error pruning layer {layer.name}: {e}")
    return model, total_prune_count
