import numpy as np
import tensorflow as tf
from utils import log_pruning_details


def random_pruning(model, layer_prune_counts, guid, csv_writer):
    total_pruned = 0

    def prune_layer(layer, prune_count):
        nonlocal total_pruned

        if hasattr(layer, 'kernel'):
            weights = [layer.kernel.numpy()]
        elif hasattr(layer, 'weights'):
            weights = layer.get_weights()
        else:
            print(f"Layer {layer.name} has no weights. Skipping...")
            return

        if not weights:
            print(f"Layer {layer.name} has no weights. Skipping...")
            return

        pruned_weights = []
        for weight in weights:
            flat_weights = weight.flatten()

            if prune_count == 0:
                print(f"No pruning needed for layer {layer.name}")
                continue

            # Randomly select indices to prune
            prune_indices = np.random.choice(
                flat_weights.size, prune_count, replace=False)
            flat_weights[prune_indices] = 0
            pruned_weights.append(flat_weights.reshape(weight.shape))

            # Log summary of pruned weights for the layer
            print(f"Pruned {prune_count} weights for layer {layer.name}")

        if pruned_weights:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(pruned_weights[0])
                print(
                    f"Assigned pruned weights to kernel of layer {layer.name}")
            else:
                layer.set_weights(pruned_weights)
                print(f"Assigned pruned weights to layer {layer.name}")

            original_param_count = sum(weight.size for weight in weights)
            non_zero_params = original_param_count - prune_count
            log_pruning_details(csv_writer, guid, 'N/A', 'N/A', 'N/A', 'random',
                                original_param_count, non_zero_params, prune_count, layer.name)

            total_pruned += prune_count
            print(f"Logged pruning details for layer {layer.name}")
        else:
            print(
                f"No pruned weights for layer {layer.name}, skipping assignment.")

    for layer in model.layers:
        if layer.name in layer_prune_counts:
            prune_layer(layer, layer_prune_counts[layer.name])

    print(f"Total weights pruned: {total_pruned}")
    return model, total_pruned
