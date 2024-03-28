import datetime
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import numpy as np
from absl import app, flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', None,
                    'Full path to the original DWT TensorFlow model.')
flags.DEFINE_string('quantized_model_dir', './SavedTFliteModels',
                    'Directory where the quantized TFLite models are saved')
flags.DEFINE_string("quantization_type", 'DEFAULT',
                    'Quantization strategy to use.')
flags.DEFINE_string('quant_level', 'binary',
                    'Level of quantization (binary, ternary, etc.)')
flags.DEFINE_float('quant_percentage', 50, 'Percentage of weights to quantize')
flags.DEFINE_boolean('random_quantize', False,
                     'Enable random weight quantization')
flags.mark_flag_as_required('model_path')


def parse_model_details_from_filename(model_filename):
    """
    Parses the model filename to extract details such as wavelet type, level, and threshold.

    Args:
    - model_filename: str. Filename of the original TensorFlow model.

    Returns:
    - A dictionary containing the parsed details if a match is found.
    - Raises a ValueError if no pattern matches.
    """
    pattern = r"mnist_model_dwt_(?P<wavelet>\w+)_(?P<level>\d+)_(?P<threshold>\d+_\d+)_(?P<date>\d{2}-\d{2}).h5$"
    match = re.match(pattern, model_filename)
    if match:
        return {
            'wavelet': match.group('wavelet'),
            'level': match.group('level'),
            'threshold': match.group('threshold').replace('_', '.'),
            'date': match.group('date')
        }
    else:
        raise ValueError(
            "The model filename does not match any expected pattern.")


def generate_model_filename(details):
    """
    Generates a descriptive file name for the quantized model.

    Args:
    - wavelet: str. Wavelet type used in the model.
    - level: int. Level of wavelet transformation used in the model.
    - threshold: float. Threshold value used in the model.
    - date: str. Date of model quantization.

    Returns:
    - str. A string representing the file name for the quantized model.
    """
    # date = datetime.datetime.now().strftime('%Y-%m-%d')
    # return f"mnist_model_dwt_{details['wavelet']}_lvl{details['level']}_thresh{threshold_str}_quantized_{date}.tflite"
    date_now = datetime.datetime.now().strftime('%Y-%m-%d')
    threshold_str = str(details['threshold']).replace('.', '_')
    return f"mnist_model_dwt_{details['wavelet']}_lvl{details['level']}_thresh{threshold_str}_quantized_{date_now}.tflite"


def convert_and_quantize_model(model_path, method='default', **kwargs):
    """
    Converts and quantizes a model based on the specified method.

    Args:
    - model_path: Path to the model to convert and quantize.
    - method: The quantization method ('default' or 'random').
    - kwargs: Additional arguments specific to the quantization method.
    """
    if method == 'random':
        # Example: 'save_path' should be provided in kwargs for the random quantization method
        random_quantize_model(model_path, kwargs.get(
            'save_path', './quantized_model.h5'))
    else:
        # Call existing function for the default quantization method
        pass  # Use the existing convert_model_to_tflite here


def random_quantize_weights(weights, quant_level='binary', percentage=50):
    """
    Randomly quantizes a percentage of weights to a specified level.

    Args:
    - weights: Numpy array of weights.
    - quant_level: The level of quantization ('binary' or 'ternary' in this example).
    - percentage: The percentage of weights to quantize.
    Returns:
    - Quantized weights as a numpy array.
    """
    mask = np.random.rand(*weights.shape) < (percentage / 100.0)
    quantized_weights = np.copy(weights)
    if quant_level == 'binary':
        quantized_weights[mask] = np.sign(quantized_weights[mask])
    elif quant_level == 'ternary':
        quantized_weights[mask] = np.sign(
            quantized_weights[mask]) * np.ceil(np.abs(quantized_weights[mask]) / 2)
    return quantized_weights


def ensure_directory_exists(directory):
    """
    Ensures that the specified directory exists. If the directory does not exist, it is created.

    Args:
    - directory_path (str): The path to the directory to ensure exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_save_dir():
    """
    Constructs and returns a directory path for saving models. The directory path reflects the model's configuration,
    incorporating the wavelet type, batch size, threshold value, epochs, and decomposition level into the directory structure.
    This organized approach facilitates easy navigation and identification of models based on their configuration.

    To ensure compatibility with filesystem conventions, especially on Windows, this function replaces '.' in the threshold
    value with '_', as filenames and directories on Windows cannot contain certain characters like the dot character in 
    contexts other than separating the filename from the extension.

    The constructed directory path follows the format:
    `save_dir/wavelet/batch_size/threshold_value/epochs/level`
    where:
    - `save_dir` is the base directory for saving models, specified by the `--save_dir` flag.
    - `wavelet` specifies the wavelet type used in the DWT process.
    - `batch_size` reflects the number of samples processed before the model is updated.
    - `threshold_value` is the threshold applied in the DWT process, with dots replaced by underscores for compatibility.
    - `epochs` represents the number of complete passes through the training dataset.
    - `level` indicates the decomposition level used in the DWT process.

    If the constructed directory does not exist, it is created with `os.makedirs(save_dir, exist_ok=True)`, ensuring
    that the model can be saved without manual directory creation.

    Returns:
        str: The constructed directory path where the model should be saved.

    Example:
        If the flags are set as follows:
        --wavelet 'haar', --batch_size 32, --threshold 0.1, --epochs 10, --level 1
        The returned save directory will be something like:
        'models/haar/32/0_1/10/1'

    Note:
        The use of `os.makedirs(..., exist_ok=True)` ensures that attempting to create an already existing directory
        won't raise an error, facilitating reusability of the function across different runs with the same or different
        configurations.
    """
    # Convert threshold to string and replace dots for filesystem compatibility
    threshold_str = str(FLAGS.threshold).replace('.', '_')
    # Construct the directory path based on flags
    save_dir = os.path.join(FLAGS.save_dir, FLAGS.wavelet,
                            str(FLAGS.batch_size), threshold_str,
                            str(FLAGS.epochs), str(FLAGS.level))
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def convert_model_to_tflite(model_path, output_file, quantization_type='default', representative_dataset_func=None):
    """
    Converts a TensorFlow model to a TensorFlow Lite model with specified post-training quantization.

    Args:
    - model_path (str): Path to the TensorFlow .h5 model file to convert.
    - output_file (str): Path where the TFLite model will be saved.
    - quantization_type (str): Type of quantization to apply. Options are 'float16', 'none', or 'default'.
      'default' will apply no quantization and is equivalent to 'none'.

    Returns:
    - None: The function saves the TFLite model to the specified path.
    """
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization_type == 'float16':
        converter.target_spec.supported_types = [tf.float16]
    elif quantization_type == 'default' or 'DEFAULT':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization_type == 'none':
        pass
    else:
        raise ValueError(
            "Unsupported quantization type. Choose 'float16', or 'none'.")

    tflite_quant_model = converter.convert()
    output_directory = os.path.join(get_save_dir(), output_file)
    # ensure_directory_exists(output_directory)
    with open(output_directory, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"Model saved to: {output_directory}")


def apply_random_quantization(model, quant_level='binary', percentage=50):
    for layer in model.layers:
        if hasattr(layer, 'weights') and layer.trainable:
            weights = layer.get_weights()
            new_weights = [random_quantize_weights(
                w, quant_level, percentage) for w in weights]
            layer.set_weights(new_weights)
    return model


def get_model_size(file_path):
    """
    Returns the size of the model at the given file path in kilobytes (KB).

    Args:
        file_path (str): The path to the model file.

    Returns:
        float: The size of the model in kilobytes (KB).
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / 1024  # Convert bytes to kilobytes


def log_details(directory, filename, details):
    """
    Logs the specified details into a text file within the given directory.

    Args:
        directory (str): The directory where the log file will be saved.
        filename (str): The name of the log file.
        details (str): The details to log.
    """
    # Ensure the directory exists
    ensure_directory_exists(directory)

    log_filepath = os.path.join(directory, filename)
    with open(log_filepath, "a") as log_file:  # Open in append mode
        # Add a newline at the end of the details
        log_file.write(details + "\n")


def get_quantized_model_save_dir(original_model_path):
    """
    Generates the save directory for the quantized model based on the original model's path.
    """
    details = parse_model_details_from_filename(original_model_path)
    threshold_str = details['threshold'].replace('_', '.')
    save_dir = os.path.join(FLAGS.quantized_model_dir,
                            details['wavelet'], details['level'], threshold_str, details['date'], "quantized")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def main():
    model_path = FLAGS.model_path

    if FLAGS.random_quantize:
        # Load the model for random quantization
        model = load_model(model_path)

        # Apply random quantization
        model = apply_random_quantization(
            model, FLAGS.quant_level, FLAGS.quant_percentage)

        # Save the quantized model
        model.save(FLAGS.quantized_model_path)
        print(
            f"Randomly quantized model saved to: {FLAGS.quantized_model_path}")

        # Measure quantized model size
        quantized_model_size_kb = get_model_size(FLAGS.quantized_model_path)
        print(f"Quantized model size: {quantized_model_size_kb:.2f} KB")

        # Optionally, compare to the original model size and display size reduction
        original_model_size_kb = get_model_size(model_path)
        size_reduction_kb = original_model_size_kb - quantized_model_size_kb
        size_reduction_percent = (
            size_reduction_kb / original_model_size_kb) * 100
        print(
            f"Size reduction: {size_reduction_kb:.2f} KB ({size_reduction_percent:.2f}%)")
    else:
        quantization_type = FLAGS.quantization_type.lower()
        print(f'Model path: {model_path}')
        model_filename = os.path.basename(model_path)
        details = parse_model_details_from_filename(model_filename)

        quantized_model_filename = generate_model_filename(details)
        quantized_model_path = os.path.join(
            FLAGS.quantized_model_dir, quantized_model_filename)

        # Perform the original quantization process (e.g., TFLite conversion)
        convert_model_to_tflite(
            model_path, quantized_model_path, quantization_type)

        # Measure original model size
        original_model_size_kb = get_model_size(model_path)
        print(f"Original model size: {original_model_size_kb:.2f} KB")

        # Measure quantized model size
        quantized_model_size_kb = get_model_size(quantized_model_path)
        print(f"Quantized model size: {quantized_model_size_kb:.2f} KB")

        # Display size reduction
        size_reduction_kb = original_model_size_kb - quantized_model_size_kb
        size_reduction_percent = (
            size_reduction_kb / original_model_size_kb) * 100
        log_details = f"Size reduction: {size_reduction_kb:.2f} KB ({size_reduction_percent:.2f}%)"
        print(log_details)
        print(f"Quantized model saved as {quantized_model_path}")


if __name__ == '__main__':
    app.run(main)

# python TensorFlowLite.py --model_dir='./DeepLearning/SavedStandardModels' --model_filename='mnist_model_02-20_17-53.h5' --output_model_filename='mnist_model_quant.tflite' --optimization='DEFAULT'
