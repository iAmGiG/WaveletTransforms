import numpy as np
import tensorflow as tf


def randomly_prune_model(model, num_prune, target_layers=None):
    """
    Randomly prunes a specified number of weights from targeted layers of the model.

    Args:
        model (tf.keras.Model): The pre-trained model to prune.
        num_prune (int): The number of weights to randomly set to zero across all targeted layers.
        target_layers (list): Optional. List of layer indices to prune. If None, all layers except the first
                              and last are considered.

    Returns:
        tf.keras.Model: The pruned model.
    """
    pruned_model = tf.keras.models.clone_model(model)
    original_weights = model.get_weights()
    pruned_weights = []

    # If no specific layers are targeted, prune all except the first and last layers
    if target_layers is None:
        target_layers = [i for i, layer in enumerate(model.layers) if layer.weights and not isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.MaxPooling2D))]

    total_weights_in_target_layers = sum(
        original_weights[i].size for i in target_layers)
    # Ensure we don't exceed available weights
    num_prune = min(num_prune, total_weights_in_target_layers)

    for i, w in enumerate(original_weights):
        if i in target_layers:
            # This layer is targeted for pruning
            w_pruned = np.copy(w).flatten()  # Flatten for pruning
            # Calculate proportional pruning for this layer
            proportion = w.size / total_weights_in_target_layers
            num_prune_this_layer = int(num_prune * proportion)

            # Randomly select indices to prune
            indices_to_prune = np.random.choice(
                range(w.size), size=num_prune_this_layer, replace=False)
            w_pruned[indices_to_prune] = 0  # Prune
            # Reshape back to original shape
            w_pruned = w_pruned.reshape(w.shape)
        else:
            # This layer is not targeted for pruning
            w_pruned = np.copy(w)

        pruned_weights.append(w_pruned)

    pruned_model.set_weights(pruned_weights)

    # Compile the pruned model to make it ready for evaluation
    pruned_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    pruned_model.trainable = False
    pruned_model.summary()
    return pruned_model
