import os
from absl import app, flags
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForImageClassification, AutoConfig
from utils import setup_csv_writer, load_model, save_model, log_pruning_details, append_to_experiment_log, check_and_set_pruned_instance_path
from dwt_pruning import wavelet_pruning

FLAGS = flags.FLAGS

# Command line argument setup
flags.DEFINE_string('model_path', '__OGModel__/tf_model.h5',
                    'Path to the pre-trained ResNet model (.h5 file)')
flags.DEFINE_string('config_path', '__OGModel__/config.json',
                    'Path to the model configuration file (.json)')
flags.DEFINE_string('csv_path', 'experiment_log.csv',
                    'Path to the CSV log file')
flags.DEFINE_enum('wavelet', 'haar', ['haar', 'db1', 'db2', 'coif1', 'bior1.3',
                  'rbio1.3', 'sym2', 'mexh', 'morl'], 'Type of wavelet to use for DWT.')
flags.DEFINE_integer(
    'level', 0, 'Level of decomposition for the wavelet transform')
flags.DEFINE_float(
    'threshold', 0.1, 'Threshold value for pruning wavelet coefficients')
flags.DEFINE_string('output_dir', 'SavedModels',
                    'Directory to save the pruned models')


def random_pruning(layer_prune_counts, guid, wavelet, level, threshold):
    total_pruned_count = 0

    # Reload the original model
    random_pruned_model = load_model(FLAGS.model_path, FLAGS.config_path)

    # Set up logging for the random pruned model
    random_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/random_pruned")
    random_log_path = os.path.join(random_pruned_dir, 'log.csv')
    random_csv_writer, random_log_file = setup_csv_writer(
        os.path.normpath(random_log_path), mode='w')

    for layer_name, prune_count in layer_prune_counts.items():
        if prune_count > 0:
            try:
                layer = random_pruned_model.get_layer(layer_name)
            except ValueError:
                # If the layer is not found, try removing the prefix
                layer_name_without_prefix = layer_name.split('/')[-1]
                layer = random_pruned_model.get_layer(
                    layer_name_without_prefix)

            weights = layer.get_weights()
            original_param_count = np.prod(weights[0].shape)
            num_weights = weights[0].size

            # Ensure that the number of weights to prune doesn't exceed the total
            num_to_prune = min(prune_count, num_weights)

            # Randomly prune individual weights
            pruned_weights = weights[0].flatten()
            prune_indices = np.random.choice(
                num_weights, num_to_prune, replace=False)
            pruned_weights[prune_indices] = 0
            pruned_weights = pruned_weights.reshape(weights[0].shape)

            weights[0] = pruned_weights
            layer.set_weights(weights)

            # Calculate the number of non-zero parameters after pruning
            non_zero_params = np.count_nonzero(pruned_weights)

            # Log pruning details
            log_pruning_details(random_csv_writer, guid, wavelet, level, threshold, 'random',
                                original_param_count, non_zero_params, num_to_prune, layer_name)

            total_pruned_count += num_to_prune
            print(
                f"Layer '{layer_name}' pruned with {num_to_prune} parameters in random pruning.")

    # Save the randomly pruned model
    save_model(random_pruned_model, random_pruned_dir)

    # Append to the experiment log
    experiment_log_path = os.path.normpath(FLAGS.csv_path)
    append_to_experiment_log(
        experiment_log_path, guid, wavelet, level, threshold, 'random', total_pruned_count, random_pruned_dir)

    random_log_file.close()

    print("Random pruning completed.")


def selective_pruning(original_model, wavelet, level, threshold, csv_writer, guid):
    selective_pruned_model, layer_prune_counts = wavelet_pruning(
        original_model, wavelet, level, threshold, csv_writer, guid)
    selective_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/selective_pruned")

    # Save the selective pruned model
    save_model(selective_pruned_model, selective_pruned_dir)

    # Append to the experiment log
    append_to_experiment_log(os.path.normpath(FLAGS.csv_path), guid, wavelet, level,
                             threshold, 'selective', sum(layer_prune_counts.values()), selective_pruned_dir)

    # Set up logging for the selective pruned model
    selective_log_path = os.path.join(selective_pruned_dir, 'log.csv')
    selective_csv_writer, selective_log_file = setup_csv_writer(
        os.path.normpath(selective_log_path), mode='w')

    # Log pruning details
    for layer_name, prune_count in layer_prune_counts.items():
        # Remove any potential prefix from the layer names
        normalized_layer_name = layer_name.split('/')[-1]
        try:
            layer = selective_pruned_model.get_layer(normalized_layer_name)
            original_param_count = layer.count_params()
            non_zero_params = np.sum([np.count_nonzero(weight)
                                     for weight in layer.get_weights()])
            log_pruning_details(selective_csv_writer, guid, wavelet, level, threshold, 'selective',
                                original_param_count, non_zero_params, prune_count, normalized_layer_name)
        except ValueError as e:
            print(f"Error retrieving layer '{normalized_layer_name}': {e}")

    selective_log_file.close()

    print(
        f"Selective pruning completed and model saved to {selective_pruned_dir}")
    return layer_prune_counts


def main(argv):
    model = load_model(FLAGS.model_path, FLAGS.config_path)
    print("Pre-trained model loaded successfully.")

    # Append mode for the running experiment log
    csv_writer, running_log_file = setup_csv_writer(FLAGS.csv_path, mode='a')

    guid = os.urandom(4).hex()

    # Perform DWT (Selective) pruning
    layer_prune_counts = selective_pruning(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, csv_writer, guid)

    # Perform Random pruning using the layer_prune_counts obtained from DWT pruning
    random_pruning(layer_prune_counts, guid, FLAGS.wavelet,
                   FLAGS.level, FLAGS.threshold)

    running_log_file.close()


if __name__ == '__main__':
    app.run(main)
