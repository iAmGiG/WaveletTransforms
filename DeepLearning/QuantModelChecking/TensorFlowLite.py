import datetime
import os
import tensorflow as tf
import re
from absl import app, flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'use_gpu', False, 'Enable GPU support for TensorFlow operations')
flags.DEFINE_string('model_path', None,
                    'Full path to the original DWT TensorFlow model.')
flags.DEFINE_string('quantized_model_dir', './SavedTFliteModels',
                    'Directory where the quantized TFLite models are saved')
flags.DEFINE_string("quantization_type", 'DEFAULT',
                    'Quantization strategy to use.')
flags.mark_flag_as_required('model_path')


def setup_gpu_configuration():
    """
    Configures TensorFlow to use GPU if available and enabled through flags.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    if gpus and FLAGS.use_gpu:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU for TensorFlow operations")
        except RuntimeError as e:
            print(e)
    else:
        print("GPU support not enabled or GPUs not available, using CPU instead")


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


def representative_dataset_generator():
    # Placeholder: Replace this loop with actual data loading and preprocessing suitable for your model.
    for _ in range(100):
        # Assuming the model expects input shape of [1, 224, 224, 3]. Adjust accordingly.
        yield [tf.random.uniform([1, 28, 28, 3], dtype=tf.float32)]


def ensure_directory_exists(directory):
    """
    Ensures that the specified directory exists. If the directory does not exist, it is created.

    Args:
    - directory_path (str): The path to the directory to ensure exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_model_to_tflite(model_path, output_file, quantization_type='default', representative_dataset_func=None):
    """
    Converts a TensorFlow model to a TensorFlow Lite model with specified post-training quantization.

    Args:
    - model_path (str): Path to the TensorFlow .h5 model file to convert.
    - output_file (str): Path where the TFLite model will be saved.
    - quantization_type (str): Type of quantization to apply. Options are 'float16', 'int8', 'none', or 'default'.
      'default' will apply no quantization and is equivalent to 'none'.
    - representative_dataset_func (function, optional): A function that generates representative dataset samples
      for 'int8' quantization. Required if quantization_type is 'int8'.

    Returns:
    - None: The function saves the TFLite model to the specified path.
    """
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization_type == ['float16', 'int8']:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization_type == 'float16':
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_type == 'int8':
            if representative_dataset_func is None:
                raise ValueError(
                    "representative_dataset_func must be provided for 'int8' quantization.")
            converter.representative_dataset = representative_dataset_generator
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    elif quantization_type == 'default' or 'DEFAULT':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization_type == 'none':
        pass
    else:
        raise ValueError(
            "Unsupported quantization type. Choose 'float16', 'int8', or 'none'.")

    tflite_quant_model = converter.convert()
    output_directory = os.path.dirname(output_file)
    ensure_directory_exists(output_directory)
    with open(output_file, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"Model saved to: {output_file}")


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
        log_file.write(details + "\n")  # Add a newline at the end of the details

def get_quantized_model_save_dir(original_model_path):
    """
    Generates the save directory for the quantized model based on the original model's path.
    """
    details = parse_model_details_from_filename(original_model_path)
    threshold_str = details['threshold'].replace('_', '.')
    save_dir = os.path.join(FLAGS.quantized_model_dir, details['wavelet'], details['level'], threshold_str, details['date'], "quantized")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def main(argv):
    # setup_gpu_configuration() gpus not supported for the TF lite conversion

    # Full path for the original model and the quantized model
    model_path = FLAGS.model_path
    quantization_type = FLAGS.quantization_type.lower()
    print(f'Model path: {model_path}')
    model_filename = os.path.basename(model_path)
    details = parse_model_details_from_filename(model_filename)

    quantized_model_filename = generate_model_filename(details)
    quantized_model_path = os.path.join(
        FLAGS.quantized_model_dir, quantized_model_filename)

    # convert_model_to_tflite(model_path, quantized_model_path,
    #                         FLAGS.quantization_type, representative_dataset_func=representative_dataset_generator)
    
    # Measure original model size
    original_model_size_kb = get_model_size(model_path)
    print(f"Original model size: {original_model_size_kb:.2f} KB")
    
    convert_model_to_tflite(model_path,
                            quantized_model_path, quantization_type)
    
    # Measure quantized model size
    quantized_model_size_kb = get_model_size(quantized_model_path)
    print(f"Quantized model size: {quantized_model_size_kb:.2f} KB")

    # Display size reduction
    size_reduction_kb = original_model_size_kb - quantized_model_size_kb
    size_reduction_percent = (size_reduction_kb / original_model_size_kb) * 100
    print(f"Size reduction: {size_reduction_kb:.2f} KB ({size_reduction_percent:.2f}%)")

    print(f"Quantized model saved as {quantized_model_path}")


if __name__ == '__main__':
    app.run(main)

# python TensorFlowLite.py --model_dir='./DeepLearning/SavedStandardModels' --model_filename='mnist_model_02-20_17-53.h5' --output_model_filename='mnist_model_quant.tflite' --optimization='DEFAULT'
