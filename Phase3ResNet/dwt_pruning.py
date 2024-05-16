import pywt
import numpy as np
import tensorflow as tf
from transformers import TFResNetForImageClassification

# Example inspection result adjustment:
# Let's assume the main layer is found to be 'tf_resnet_model' instead


def multi_resolution_analysis(weights, wavelet, level, threshold):
    pruned_weights = []
    total_pruned_count = 0
    for weight in weights:
        original_shape = weight.shape
        flattened_weight = np.ravel(weight)
        coeffs = pywt.wavedec(flattened_weight, wavelet,
                              level=level, mode='periodization')
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        for i in range(len(coeff_arr)):
            if np.abs(coeff_arr[i]).mean() < threshold:
                pruned_count = np.sum(coeff_arr[i] != 0)
                total_pruned_count += pruned_count
                coeff_arr[i] = 0
        pruned_coeffs = pywt.array_to_coeffs(
            coeff_arr, coeff_slices, output_format='wavedec')
        pruned_weight = pywt.waverec(
            pruned_coeffs, wavelet, mode='periodization')
        if pruned_weight.size > flattened_weight.size:
            pruned_weight = pruned_weight[:flattened_weight.size]
        elif pruned_weight.size < flattened_weight.size:
            pruned_weight = np.pad(
                pruned_weight, (0, flattened_weight.size - pruned_weight.size), 'constant')
        pruned_weights.append(pruned_weight.reshape(original_shape))
    return pruned_weights, total_pruned_count


def prune_layer_weights(layer, wavelet, level, threshold, csv_writer, guid, layer_name):
    if hasattr(layer, 'kernel'):
        weights = layer.kernel.numpy()
        pruned_weights, total_pruned_count = multi_resolution_analysis(
            weights, wavelet, level, threshold)
        layer.kernel.assign(pruned_weights)
        return pruned_weights, total_pruned_count
    elif hasattr(layer, 'weights'):
        weights = layer.get_weights()
        pruned_weights, total_pruned_count = multi_resolution_analysis(
            weights, wavelet, level, threshold)
        layer.set_weights(pruned_weights)
        return pruned_weights, total_pruned_count
    else:
        print(f"Layer {layer_name} is not a supported layer type. Skipping...")
        return layer.get_weights(), 0


def recursive_prune(layer, wavelet, level, threshold, csv_writer, guid, layer_name_prefix=""):
    total_prune_count = 0
    pruned_layers_count = 0

    def inner_recursive_prune(current_layer, layer_name_prefix):
        nonlocal total_prune_count, pruned_layers_count
        full_layer_name = f"{layer_name_prefix}/{current_layer.name}"
        # Adjust based on inspection
        if isinstance(current_layer, tf.keras.Model) and current_layer.name == 'tf_resnet_model':
            for sub_layer in current_layer.layers:
                inner_recursive_prune(
                    sub_layer, layer_name_prefix=full_layer_name)
        elif isinstance(current_layer, tf.keras.layers.Layer):
            try:
                pruned_weights, layer_pruned_count = prune_layer_weights(
                    current_layer, wavelet, level, threshold, csv_writer, guid, full_layer_name
                )
                total_prune_count += layer_pruned_count
                pruned_layers_count += 1
                print(
                    f"Layer {full_layer_name} pruned. Total pruned count: {layer_pruned_count}")
            except Exception as e:
                print(f"Error pruning layer {full_layer_name}: {e}")

    inner_recursive_prune(layer, layer_name_prefix)
    print(f"Completed DWT pruning on {pruned_layers_count} layers.")
    return total_prune_count


def wavelet_pruning(model, wavelet, level, threshold, csv_writer, guid):
    total_prune_count = recursive_prune(
        model, wavelet, level, threshold, csv_writer, guid)
    return model, total_prune_count
