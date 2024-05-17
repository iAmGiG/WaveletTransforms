import csv
import numpy as np
import tensorflow as tf
from transformers import TFResNetForImageClassification
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
    def prune_layer(layer, layer_name):
        nonlocal total_pruned

        if hasattr(layer, 'kernel'):
            weights = [layer.kernel.numpy()]
        elif hasattr(layer, 'weights'):
            weights = layer.get_weights()
        else:
            print(f"Layer {layer_name} has no weights. Skipping...")
            return

        if not weights:
            print(f"Layer {layer_name} has no weights. Skipping...")
            return

        pruned_weights = []
        for weight in weights:
            flat_weights = weight.flatten()
            prune_count = layer_prune_counts.get(layer_name, 0)

            if prune_count == 0:
                print(f"No pruning needed for layer {layer_name}")
                continue

            # Debug: Print prune count for the layer
            print(f"Prune count for layer {layer_name}: {prune_count}")

            prune_indices = np.random.choice(
                flat_weights.size, prune_count, replace=False)
            flat_weights[prune_indices] = 0
            pruned_weights.append(flat_weights.reshape(weight.shape))

            # Debug: Print summary of pruned weights for the layer
            non_zero_count = np.count_nonzero(flat_weights)
            zero_count = flat_weights.size - non_zero_count
            print(
                f"Layer {layer_name} pruned: {zero_count} weights set to zero, {non_zero_count} weights remaining.")

        if pruned_weights:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(pruned_weights[0])
                print(
                    f"Assigned pruned weights to kernel of layer {layer_name}")
            else:
                layer.set_weights(pruned_weights)
                print(f"Assigned pruned weights to layer {layer_name}")

            original_param_count = sum(weight.size for weight in weights)
            non_zero_params = original_param_count - prune_count
            log_pruning_details(csv_writer, guid, 'N/A', 'N/A', 'N/A', 'random',
                                original_param_count, non_zero_params, prune_count, layer_name)

            total_pruned += prune_count
        else:
            print(
                f"No pruned weights for layer {layer_name}, skipping assignment.")

    def recursive_prune(current_layer, layer_name_prefix=""):
        layer_name = f"{layer_name_prefix}/{current_layer.name}"
        if isinstance(current_layer, tf.keras.Model):
            for sub_layer in current_layer.layers:
                recursive_prune(sub_layer, layer_name_prefix=layer_name)
        elif isinstance(current_layer, tf.keras.layers.Layer):
            prune_layer(current_layer, layer_name)

    total_pruned = 0
    recursive_prune(model)
    print(f"Total weights pruned: {total_pruned}")
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
