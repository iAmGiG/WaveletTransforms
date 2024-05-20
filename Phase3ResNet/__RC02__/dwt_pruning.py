import numpy as np
import tensorflow as tf
from utils import log_pruning_details


def dwt_pruning(model, wavelet, level, threshold, csv_writer, guid):
    # Placeholder for actual DWT pruning logic
    layer_prune_counts = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            # Placeholder for actual DWT-based pruning logic
            # Replace with actual prune count logic
            prune_count = np.random.randint(100, 1000)
            layer_prune_counts[layer.name] = prune_count

            # Log the pruning details for each layer
            original_param_count = np.prod(layer.get_weights()[0].shape)
            non_zero_params = original_param_count - prune_count
            log_pruning_details(csv_writer, guid, wavelet, level, threshold, 'DWT',
                                original_param_count, non_zero_params, prune_count, layer.name)

            # Simulate the pruning (actual pruning logic will differ)
            weights = layer.get_weights()
            flat_weights = weights[0].flatten()
            prune_indices = np.random.choice(
                flat_weights.size, prune_count, replace=False)
            flat_weights[prune_indices] = 0
            weights[0] = flat_weights.reshape(weights[0].shape)
            layer.set_weights(weights)

    return layer_prune_counts
