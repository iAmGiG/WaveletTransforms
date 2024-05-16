import csv
import numpy as np
import tensorflow as tf
from utils import log_pruning_details


def random_pruning(model, layer_prune_counts, guid, csv_writer):
    """
    Apply random pruning to the model, ensuring the same number of weights are pruned per layer as in DWT pruning.

    Args:
        model (tf.keras.Model): The model to prune.
        layer_prune_counts (dict): Dictionary with layer names as keys and number of weights to prune as values.
        guid (str): Unique identifier for the pruning session.
        csv_writer (csv.DictWriter): CSV writer object for logging.

    Returns:
        tf.keras.Model: Randomly pruned model.
    """
    for layer in model.layers:
        if layer.trainable and isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()
            if not weights:
                continue
            pruned_weights = []
            for weight in weights:
                flat_weights = weight.flatten()
                prune_count = layer_prune_counts.get(layer.name, 0)
                prune_indices = np.random.choice(
                    flat_weights.size, prune_count, replace=False)
                flat_weights[prune_indices] = 0
                pruned_weights.append(flat_weights.reshape(weight.shape))

            layer.set_weights(pruned_weights)

            # Log pruning details for the layer
            original_param_count = sum(weight.size for weight in weights)
            non_zero_params = original_param_count - prune_count
            log_pruning_details(csv_writer, guid, 'N/A', 'N/A', 'N/A',
                                'random', original_param_count, non_zero_params, prune_count, layer.name)

    return model


def collect_prune_counts(log_file_path):
    """
    Collect the prune counts for each layer from the DWT pruning log.

    Args:
        log_file_path (str): Path to the DWT pruning log file.

    Returns:
        dict: Dictionary with layer names as keys and number of weights pruned as values.
    """
    prune_counts = {}
    with open(log_file_path, 'r') as log_file:
        reader = csv.DictReader(log_file)
        for row in reader:
            if row['DWT Phase'] == 'selective':
                layer_name = row['Layer Name']
                prune_count = int(row['Total Pruned Count'])
                prune_counts[layer_name] = prune_count
    return prune_counts
