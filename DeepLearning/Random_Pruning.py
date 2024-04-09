import numpy as np
import tensorflow as tf


def randomly_prune_model(model, num_prune):
    """
    take the model, number of weights to prune
    """
    pruned_model = tf.keras.models.clone_model(model)
    pruned_model.set_weights(model.get_weights())  # Clone weights

    weights = pruned_model.get_weights()
    flattened_weights = np.concatenate([w.flatten() for w in weights])
    total_weights = flattened_weights.shape[0]

    # Ensure we don't attempt to prune more weights than exist
    num_prune = min(num_prune, total_weights)

    # Select indices to prune
    indices_to_prune = np.random.choice(
        range(total_weights), size=num_prune, replace=False)

    # Prune selected weights
    flattened_weights[indices_to_prune] = 0

    # Reassign pruned weights back to their original shapes
    start = 0
    for i, w in enumerate(weights):
        end = start + w.size
        weights[i] = flattened_weights[start:end].reshape(w.shape)
        start = end

    pruned_model.set_weights(weights)
    pruned_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    print("Random Model generated")
    return pruned_model
