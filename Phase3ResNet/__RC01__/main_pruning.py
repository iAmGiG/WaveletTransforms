import os
import uuid
import numpy as np
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from dwt_pruning import wavelet_pruning
from random_pruning import random_pruning
from utils import load_model, save_model, setup_csv_writer, append_to_experiment_log

# Command line argument setup
flags.DEFINE_string('model_path', '__OGModel__/tf_model.h5',
                    'Path to the pre-trained ResNet model (.h5 file)')
flags.DEFINE_string('config_path', '__OGModel__/config.json',
                    'Path to the model configuration file (.json)')
flags.DEFINE_string('csv_path', 'experiment_log.csv',
                    'Path to the CSV log file')
flags.DEFINE_enum('wavelet', 'haar', ['haar', 'db1', 'db2', 'coif1', 'bior1.3', 'rbio1.3', 'sym2', 'mexh', 'morl'],
                  'Type of wavelet to use for DWT.')
flags.DEFINE_integer(
    'level', 0, 'Level of decomposition for the wavelet transform')
flags.DEFINE_float(
    'threshold', 0.1, 'Threshold value for pruning wavelet coefficients')


def check_and_set_pruned_instance_path(pruned_instance):
    """
    Check if the script is run from the 'Phase3ResNet' directory and set the pruned_instance_path accordingly.

    Args:
        pruned_instance (str): The name of the pruned instance.

    Returns:
        str: The full path to the pruned instance directory.
    """
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'Phase3ResNet':
        pruned_instance_path = os.path.join('SavedModels', pruned_instance)
    else:
        pruned_instance_path = os.path.join(
            current_dir, 'Phase3ResNet', 'SavedModels', pruned_instance)
    os.makedirs(pruned_instance_path, exist_ok=True)
    return pruned_instance_path


def validate_prune_counts(layer_prune_counts):
    """
    Validate the prune counts to ensure they are non-negative and match the expected layers.
    """
    for layer_name, prune_count in layer_prune_counts.items():
        if prune_count < 0:
            raise ValueError(
                f"Prune count for layer {layer_name} is negative: {prune_count}")
        print(f"Layer {layer_name} has a valid prune count: {prune_count}")


def main(_argv):
    """
    Main function to perform wavelet-based and random pruning.
    """
    # Load the pre-trained model
    model = load_model(FLAGS.model_path, FLAGS.config_path)
    print("Loaded model summary:")
    model.summary()

    # Create a unique identifier and setup folder structure
    guid = str(uuid.uuid4())
    pruned_instance = f"wavelet-{FLAGS.wavelet}_threshold-{FLAGS.threshold}_level-{FLAGS.level}_guid-{guid[-4:]}"
    pruned_instance_path = check_and_set_pruned_instance_path(pruned_instance)
    selective_pruned_path = os.path.join(
        pruned_instance_path, 'selective_pruned')
    random_pruned_path = os.path.join(pruned_instance_path, 'random_pruned')

    os.makedirs(selective_pruned_path, exist_ok=True)
    os.makedirs(random_pruned_path, exist_ok=True)

    # Perform wavelet-based pruning and store prune counts in memory
    print("Running Selective DWT pruning")
    selective_log_path = os.path.join(selective_pruned_path, 'log.csv')
    csv_writer, csv_file = setup_csv_writer(
        selective_log_path, mode='a')  # Open in append mode

    pruned_model, layer_prune_counts = wavelet_pruning(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, csv_writer, guid)
    save_model(pruned_model, os.path.join(selective_pruned_path, 'model'))
    csv_file.close()

    # Validate prune counts
    validate_prune_counts(layer_prune_counts)

    print(
        f"Total pruned count after wavelet-based pruning: {sum(layer_prune_counts.values())}")
    print(f"Layer prune counts: {layer_prune_counts}")

    # Update layer_prune_counts with full layer paths
    def update_prune_counts(model, layer_prune_counts):
        updated_prune_counts = {}

        def recursive_update(layer, path=""):
            layer_name = f"{path}/{layer.name}" if path else layer.name
            if layer_name in layer_prune_counts:
                updated_prune_counts[layer_name] = layer_prune_counts[layer_name]
            if isinstance(layer, tf.keras.Model):
                for sub_layer in layer.layers:
                    recursive_update(sub_layer, path=layer_name)

        recursive_update(model)
        return updated_prune_counts

    layer_prune_counts = update_prune_counts(model, layer_prune_counts)
    print("Updated Layer prune counts:", layer_prune_counts)

    # Log experiment details for selective pruning
    experiment_log_path = 'experiment_log.csv'
    append_to_experiment_log(experiment_log_path, guid, FLAGS.wavelet, FLAGS.level,
                             FLAGS.threshold, 'selective', sum(layer_prune_counts.values()))

    print("Starting random pruning")

    # Perform random pruning using in-memory layer prune counts
    random_log_path = os.path.join(random_pruned_path, 'log.csv')
    csv_writer, csv_file = setup_csv_writer(
        random_log_path, mode='a')  # Open in append mode

    random_model = load_model(FLAGS.model_path, FLAGS.config_path)
    print("Layer prune counts before random pruning: ", layer_prune_counts)
    random_pruned_model, total_pruned_random = random_pruning(
        random_model, layer_prune_counts, guid, csv_writer)
    save_model(random_pruned_model, os.path.join(random_pruned_path, 'model'))
    csv_file.close()

    # Log experiment details for random pruning if any pruning was done
    if total_pruned_random > 0:
        append_to_experiment_log(
            experiment_log_path, guid, 'N/A', 'N/A', 'N/A', 'random', total_pruned_random)
    else:
        print("No weights pruned during random pruning. Skipping log update.")
    print("Completed random pruning")


if __name__ == '__main__':
    app.run(main)
