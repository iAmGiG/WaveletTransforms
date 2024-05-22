import os
import numpy as np
from utils import save_model, log_pruning_details, append_to_experiment_log


def random_pruning(model, layer_prune_counts, csv_writer, guid, wavelet, level, threshold, output_dir):
    """
    Performs random pruning on the model based on the DWT pruning results.

    Args:
        model (tf.keras.Model): The TensorFlow model to be pruned.
        layer_prune_counts (dict): A dictionary containing the prune counts for each layer.
        csv_writer (csv.DictWriter): A CSV writer object for logging pruning details.
        guid (str): A unique identifier for the pruning session.
        wavelet (str): The wavelet used for DWT pruning.
        level (int): The level of wavelet decomposition used for DWT pruning.
        threshold (float): The threshold value used for DWT pruning.
        output_dir (str): The directory path to save the randomly pruned model.

    Returns:
        tf.keras.Model: The randomly pruned model.
    """
    total_pruned_count = 0
    random_pruned_model = model.clone_model()

    for layer in random_pruned_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer_name = layer.name
            prune_count = layer_prune_counts.get(layer_name, 0)
            if prune_count > 0:
                weights = layer.get_weights()
                non_zero_params = np.count_nonzero(weights[0])
                original_param_count = np.prod(weights[0].shape)

                # Determine the number of filters or weight vectors to prune
                num_filters_or_vectors = weights[0].shape[-1]
                num_to_prune = int(prune_count / num_filters_or_vectors)

                # Randomly select and zero out filters or weight vectors
                prune_indices = np.random.choice(
                    num_filters_or_vectors, num_to_prune, replace=False)
                pruned_weights = weights[0].copy()
                pruned_weights[:, :, prune_indices] = 0

                weights[0] = pruned_weights
                layer.set_weights(weights)

                # Log pruning details
                log_pruning_details(csv_writer, guid, wavelet, level, threshold, 'Random',
                                    original_param_count, non_zero_params, prune_count, layer_name)

                total_pruned_count += prune_count

    # Save the randomly pruned model
    output_path = os.path.join(output_dir, f"random_pruned_model_{guid}.h5")
    save_model(random_pruned_model, output_path)

    # Append to the experiment log
    append_to_experiment_log(os.path.join(output_dir, "experiment_log.csv"),
                             guid, wavelet, level, threshold, 'Random', total_pruned_count)

    return random_pruned_model
