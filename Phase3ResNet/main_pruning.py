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


def random_pruning(original_model_path, config_path, layer_prune_counts, csv_writer, guid, wavelet, level, threshold, output_dir):
    total_pruned_count = 0

    # Load the original model
    random_pruned_model = load_model(original_model_path, config_path)

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
                num_to_prune = int(prune_count)

                # Randomly select and zero out filters or weight vectors
                prune_indices = np.random.choice(
                    num_filters_or_vectors, num_to_prune, replace=False)
                pruned_weights = weights[0].copy()
                pruned_weights[..., prune_indices] = 0

                weights[0] = pruned_weights
                layer.set_weights(weights)

                # Log pruning details
                log_pruning_details(csv_writer, guid, wavelet, level, threshold, 'random',
                                    original_param_count, non_zero_params, prune_count, layer_name)

                total_pruned_count += prune_count

                print(
                    f"Layer '{layer_name}' pruned with {prune_count} parameters in random pruning.")

    # Save the randomly pruned model
    output_path = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/random_pruned")
    save_model(random_pruned_model, output_path)

    # Append to the experiment log
    append_to_experiment_log(
        FLAGS.csv_path, guid, wavelet, level, threshold, 'random', total_pruned_count, output_path)

    print("Random pruning completed.")


def main(argv):
    # Load the pre-trained model
    model = load_model(FLAGS.model_path, FLAGS.config_path)
    print("Pre-trained model loaded successfully.")

    # Set up the CSV writer
    csv_writer, log_file = setup_csv_writer(FLAGS.csv_path)

    # Perform DWT pruning
    guid = os.urandom(4).hex()
    pruned_model, layer_prune_counts = wavelet_pruning(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, csv_writer, guid)

    # Save the DWT pruned model
    pruned_instance = f"{FLAGS.wavelet}_threshold-{FLAGS.threshold}_level-{FLAGS.level}_guid-{guid[:4]}"
    dwt_pruned_dir = check_and_set_pruned_instance_path(pruned_instance)
    print(f"Saving DWT pruned model to: {dwt_pruned_dir}")
    output_path = os.path.join(dwt_pruned_dir, "selective_pruned")
    print(f"Output path: {output_path}")

    try:
        save_model(pruned_model, output_path)
    except Exception as e:
        print(f"Error saving DWT pruned model: {e}")

    # Perform random pruning
    print(
        f"starting random pruning on: {model.summary()}\nImpacting the layers {layer_prune_counts}")
    random_pruning(model, layer_prune_counts, csv_writer, guid,
                   FLAGS.wavelet, FLAGS.level, FLAGS.threshold, dwt_pruned_dir)

    # Close the log file
    log_file.close()


if __name__ == "__main__":
    app.run(main)
