import numpy as np
import tensorflow as tf
from transformers import TFResNetForImageClassification

# Function to load the pre-saved model


def load_saved_model(model_path, config_path):
    model = TFResNetForImageClassification.from_pretrained(
        model_path, config=config_path)
    return model

# Dummy random pruning function for testing


def dummy_random_pruning(model, layer_prune_counts):
    total_pruned = 0

    def prune_layer(layer, layer_path):
        nonlocal total_pruned

        if hasattr(layer, 'kernel'):
            weights = [layer.kernel.numpy()]
        elif hasattr(layer, 'weights'):
            weights = layer.get_weights()
        else:
            print(f"Layer {layer_path} has no weights. Skipping...")
            return

        if not weights:
            print(f"Layer {layer_path} has no weights. Skipping...")
            return

        pruned_weights = []
        for weight in weights:
            flat_weights = weight.flatten()
            prune_count = layer_prune_counts.get(layer_path, 0)

            if prune_count == 0:
                print(f"No pruning needed for layer {layer_path}")
                continue

            # Adjust prune_count if it exceeds the number of weights in the layer
            prune_count = min(prune_count, flat_weights.size)

            # Debug: Print prune count for the layer
            print(f"Prune count for layer {layer_path}: {prune_count}")

            # Randomly select indices to prune
            prune_indices = np.random.choice(
                flat_weights.size, prune_count, replace=False)
            flat_weights[prune_indices] = 0
            pruned_weights.append(flat_weights.reshape(weight.shape))

            # Log summary of pruned weights for the layer
            print(f"Pruned {prune_count} weights for layer {layer_path}")

        if pruned_weights:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(pruned_weights[0])
                print(
                    f"Assigned pruned weights to kernel of layer {layer_path}")
            else:
                layer.set_weights(pruned_weights)
                print(f"Assigned pruned weights to layer {layer_path}")

            total_pruned += prune_count
            print(f"Logged pruning details for layer {layer_path}")
        else:
            print(
                f"No pruned weights for layer {layer_path}, skipping assignment.")

    def recursive_prune(current_layer, layer_path=""):
        layer_name = f"{layer_path}/{current_layer.name}" if layer_path else current_layer.name
        # Debug: Print current layer path
        print(f"Visiting layer: {layer_name}")
        if isinstance(current_layer, tf.keras.Model):
            for sub_layer in current_layer.layers:
                recursive_prune(sub_layer, layer_path=layer_name)
        elif isinstance(current_layer, tf.keras.layers.Layer):
            prune_layer(current_layer, layer_name)

    total_pruned = 0
    recursive_prune(model)
    print(f"Total weights pruned: {total_pruned}")
    return model, total_pruned


# Example prune counts for testing
dummy_prune_counts = {
    "tf_res_net_for_image_classification/resnet": 100,
    "tf_res_net_for_image_classification/classifier.1": 10
}
