import pywt
import numpy as np
from utils import log_pruning_details


def multi_resolution_analysis(weights, wavelet, level, threshold):
    """
    Perform multi-resolution analysis and pruning on the weights.

    Args:
        weights (list of np.ndarray): List of weight arrays to be pruned.
        wavelet (str): Type of wavelet to use for decomposition.
        level (int): Level of decomposition for the wavelet transform.
        threshold (float): Threshold value for pruning wavelet coefficients.

    Returns:
        tuple: Pruned weights and total pruned count.
    """
    pruned_weights = []
    total_pruned_count = 0
    for weight in weights:
        original_shape = weight.shape
        flattened_weight = np.ravel(weight)

        # Perform multi-resolution wavelet decomposition
        coeffs = pywt.wavedec(flattened_weight, wavelet,
                              level=level, mode='periodization')
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

        # Prune low-importance scales
        for i in range(len(coeff_arr)):
            if np.abs(coeff_arr[i]).mean() < threshold:
                total_pruned_count += np.sum(coeff_arr[i] != 0)
                coeff_arr[i] = 0

        # Reconstruct the weights
        pruned_coeffs = pywt.array_to_coeffs(
            coeff_arr, coeff_slices, output_format='wavedec')
        pruned_weight = pywt.waverec(
            pruned_coeffs, wavelet, mode='periodization')

        # Ensure the pruned weight has the same shape
        if pruned_weight.size > flattened_weight.size:
            pruned_weight = pruned_weight[:flattened_weight.size]
        elif pruned_weight.size < flattened_weight.size:
            pruned_weight = np.pad(
                pruned_weight, (0, flattened_weight.size - pruned_weight.size), 'constant')

        pruned_weight = np.reshape(pruned_weight, original_shape)
        pruned_weights.append(pruned_weight)

    return pruned_weights, total_pruned_count


def prune_layer_weights(layer, wavelet, level, threshold):
    """
    Apply wavelet-based pruning to the weights of a given layer.

    Args:
        layer (tf.keras.layers.Layer): Layer to prune.
        wavelet (str): Type of wavelet to use for decomposition.
        level (int): Level of decomposition for the wavelet transform.
        threshold (float): Threshold value for pruning wavelet coefficients.

    Returns:
        tuple: Pruned weights, total prune count, original parameter count, non-zero parameters count.
    """
    if not isinstance(layer, tf.keras.layers.Conv2D):
        return layer.get_weights(), 0

    weights = layer.get_weights()
    if not weights:
        return weights, 0

    pruned_weights, total_pruned_count = multi_resolution_analysis(
        weights, wavelet, level, threshold)

    original_param_count = sum(weight.size for weight in weights)
    non_zero_params = original_param_count - total_pruned_count

    return pruned_weights, total_pruned_count, original_param_count, non_zero_params


def wavelet_pruning(model, wavelet, level, threshold, csv_writer, guid):
    """
    Apply wavelet-based pruning to the entire model.

    Args:
        model (tf.keras.Model): Model to prune.
        wavelet (str): Type of wavelet to use for decomposition.
        level (int): Level of decomposition for the wavelet transform.
        threshold (float): Threshold value for pruning wavelet coefficients.
        csv_writer (csv.DictWriter): CSV writer object for logging.
        guid (str): Unique identifier for the pruning session.

    Returns:
        tf.keras.Model: Pruned model.
    """
    total_prune_count = 0
    for layer in model.layers:
        if layer.trainable and isinstance(layer, tf.keras.layers.Conv2D):
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
