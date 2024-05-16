import numpy as np
import tensorflow as tf
from utils import log_pruning_details


def random_pruning(model, prune_count, guid, csv_writer):
    """
    Apply random pruning to the model.

    Args:
        model (tf.keras.Model): The model to prune.
        prune_count (int): The number of weights to prune.
        guid (str): Unique identifier for the pruning session.
        csv_writer (csv.DictWriter): CSV writer object for logging.

    Returns:
        tf.keras.Model: Randomly pruned model.
    """
    total_pruned = 0
    for layer in model.layers:
        if layer.trainable and isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()
            if not weights:
                continue
            pruned_weights = []
            for weight in weights:
                flat_weights = weight.flatten()
                # Select random indices to prune
                prune_indices = np.random.choice(
                    flat_weights.size, prune_count, replace=False)
                flat_weights[prune_indices] = 0
                pruned_weights.append(flat_weights.reshape(weight.shape))
                total_pruned += prune_count

            layer.set_weights(pruned_weights)

            # Log pruning details for the layer
            original_param_count = sum(weight.size for weight in weights)
            non_zero_params = original_param_count - total_pruned
            log_pruning_details(csv_writer, guid, 'N/A', 'N/A', 'N/A',
                                'random', original_param_count, non_zero_params, total_pruned, layer.name)

    return model, total_pruned
