import tensorflow as tf
import pywt
import numpy as np
import os
import csv
import uuid
from absl import app, flags
from absl.flags import FLAGS
from transformers import TFAutoModelForImageClassification, AutoConfig

# Command line argument setup
flags.DEFINE_string('model_path', '__OGModel__/tf_model.h5',
                    'Path to the pre-trained ResNet model (.h5 file)')
flags.DEFINE_string('config_path', '__OGModel__/config.json',
                    'Path to the model configuration file (.json)')
flags.DEFINE_string('output_path', None, 'Path to save the pruned model')
flags.DEFINE_string('csv_path', 'experiment_log.csv',
                    'Path to the CSV log file')
flags.DEFINE_enum('wavelet', 'haar', ['haar', 'db1', 'db2', 'coif1', 'bior1.3', 'rbio1.3', 'sym2', 'mexh', 'morl'],
                  'Type of wavelet to use for DWT.')
flags.DEFINE_integer(
    'level', 1, 'Level of decomposition for the wavelet transform')
flags.DEFINE_float(
    'threshold', 0.5, 'Threshold value for pruning wavelet coefficients')


def load_model(model_path, config_path):
    """
    Load the pre-trained ResNet model from a local .h5 file and configuration file.

    Args:
        model_path (str): Path to the .h5 file of the pre-trained model.
        config_path (str): Path to the model configuration file (.json).

    Returns:
        tf.keras.Model: Loaded ResNet model.
    """
    try:
        # Load configuration
        config = AutoConfig.from_pretrained(config_path)

        # Load model
        model = TFAutoModelForImageClassification.from_pretrained(
            model_path, config=config)

        print(f'Model loaded successfully from {model_path}')
        return model
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        raise


def prune_layer_weights(layer, wavelet, level, threshold):
    """
    Apply wavelet-based pruning to the weights of a given layer.

    Args:
        layer (tf.keras.layers.Layer): Layer to prune.
        wavelet (str): Type of wavelet to use for decomposition.
        level (int): Level of decomposition for the wavelet transform.
        threshold (float): Threshold value for pruning wavelet coefficients.

    Returns:
        tuple: Pruned weights, original parameter count, non-zero parameters count, total pruned count.
    """
    weights = layer.get_weights()
    if not weights:
        return weights, 0, 0, 0

    pruned_weights = []
    total_pruned_count = 0
    for weight in weights:
        original_shape = weight.shape
        flattened_weight = np.ravel(weight)

        coeffs = pywt.wavedec(flattened_weight, wavelet,
                              level=level, mode='periodization')
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        pruned_count = np.sum(np.abs(coeff_arr) < threshold)
        coeff_arr[np.abs(coeff_arr) < threshold] = 0
        pruned_coeffs = pywt.array_to_coeffs(
            coeff_arr, coeff_slices, output_format='wavedec')
        pruned_weight = pywt.waverec(
            pruned_coeffs, wavelet, mode='periodization')

        # Check and adjust the length of pruned_weight
        if pruned_weight.size > flattened_weight.size:
            pruned_weight = pruned_weight[:flattened_weight.size]
        elif pruned_weight.size < flattened_weight.size:
            pruned_weight = np.pad(
                pruned_weight, (0, flattened_weight.size - pruned_weight.size), 'constant')

        pruned_weight = np.reshape(pruned_weight, original_shape)
        pruned_weights.append(pruned_weight)
        total_pruned_count += pruned_count

    original_param_count = sum([weight.size for weight in weights])
    non_zero_params = original_param_count - total_pruned_count

    return pruned_weights, original_param_count, non_zero_params, total_pruned_count


def prune_model(model, wavelet, level, threshold, csv_writer, guid, dwt_phase):
    """
    Apply wavelet-based pruning to the entire model.

    Args:
        model (tf.keras.Model): Model to prune.
        wavelet (str): Type of wavelet to use for decomposition.
        level (int): Level of decomposition for the wavelet transform.
        threshold (float): Threshold value for pruning wavelet coefficients.
        csv_writer (csv.DictWriter): CSV writer object for logging.
        guid (str): Unique identifier for the pruning session.
        dwt_phase (str): DWT Phase, either selective or random pruning.

    Returns:
        tf.keras.Model: Pruned model.
    """
    for layer in model.layers:
        if layer.trainable and layer.get_weights():
            try:
                pruned_weights, original_param_count, non_zero_params, total_pruned_count = prune_layer_weights(
                    layer, wavelet, level, threshold)
                layer.set_weights(pruned_weights)
                csv_writer.writerow({
                    'GUID': guid,
                    'Wavelet': wavelet,
                    'Level': level,
                    'Threshold': threshold,
                    'DWT Phase': dwt_phase,
                    'Original Parameter Count': original_param_count,
                    'Non-zero Params': non_zero_params,
                    'Total Pruned Count': total_pruned_count
                })
                print(f'Layer {layer.name} pruned successfully.')
            except Exception as e:
                print(f"Error pruning layer {layer.name}: {e}")
    return model


def save_model(model):
    """
    Save the pruned model in TensorFlow SavedModel format.

    Args:
        model (tf.keras.Model): The TensorFlow model to save.
    """
    try:
        output_path = os.path.join(os.getcwd(), 'pruned_model')
        model.save(output_path)
        print(f"Model saved successfully at {output_path}")
    except Exception as e:
        print(f"Failed to save the model: {e}")
        raise


def setup_csv_writer(csv_path):
    """
    Set up the CSV writer.

    Args:
        csv_path (str): Path to the CSV log file.

    Returns:
        tuple: CSV writer object and file object.
    """
    file_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, mode='a', newline='')
    fieldnames = ['GUID', 'Wavelet', 'Level', 'Threshold', 'DWT Phase',
                  'Original Parameter Count', 'Non-zero Params', 'Total Pruned Count']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    return writer, csv_file


def validate_flags():
    """
    Validate the command-line arguments to ensure they are within acceptable ranges.
    """
    if not 0 <= FLAGS.threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')


def main(_argv):
    validate_flags()
    print(f'Loading model from {FLAGS.model_path}')
    model = load_model(FLAGS.model_path, FLAGS.config_path)

    print(f'Setting up CSV writer at {FLAGS.csv_path}')
    csv_writer, csv_file = setup_csv_writer(FLAGS.csv_path)

    # Generate a unique identifier for this pruning session
    guid = str(uuid.uuid4())

    print(
        f'Pruning model with wavelet {FLAGS.wavelet}, level {FLAGS.level}, and threshold {FLAGS.threshold}')
    pruned_model = prune_model(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, csv_writer, guid)

    save_model(pruned_model)

    csv_file.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        raise
