import datetime
import os
import tensorflow as tf
import re
from absl import app
from absl import flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'use_gpu', True, 'Enable GPU support for TensorFlow operations')
flags.DEFINE_string('model_dir', None,
                    'Full path to the original DWT TensorFlow model.')
flags.DEFINE_string('quantized_model_dir', './DeepLearning/SavedTFliteModels',
                    'Directory where the quantized TFLite models are saved')
flags.DEFINE_string('version', 'v1', 'Version number of the model')
flags.DEFINE_string("quantization_type", 'DEFAULT', 'Quantization strategy to use.')
flags.mark_flag_as_required('model_path')

def setup_gpu_configuration():
    """
    Configures TensorFlow to use GPU if available and enabled through flags.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
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
    - A dictionary containing the parsed details.
    """
    pattern = r"mnist_model_dwt_(?P<wavelet>\w+)_(?P<level>\d+)_(?P<threshold>\d+)_(?P<date>\d{2}-\d{2})"
    match = re.search(pattern, model_filename)
    if match:
        return match.groupdict()
    else:
        raise ValueError(
            "The model filename does not match the expected pattern.")


def generate_model_filename(details, version):
    """
    Generates a descriptive file name for the quantized model.

    Args:
    - wavelet: str. Wavelet type used in the model.
    - level: int. Level of wavelet transformation used in the model.
    - threshold: float. Threshold value used in the model.
    - version: str. Version number of the model.
    - date: str. Date of model quantization.

    Returns:
    - str. A string representing the file name for the quantized model.
    """
    date = datetime.now().strftime('%Y-%m-%d')
    threshold_str = str(details['threshold']).replace('.', '')
    return f"mnist_model_dwt_{details['wavelet']}_lvl{details['level']}_thresh{threshold_str}_quantized_{version}_{date}.tflite"


def convert_model_to_tflite(model_path, output_file, quantization_type):
    """
    Converts a TensorFlow model to a TensorFlow Lite model with post-training quantization.

    Args:
    - model_path: str. Path to the TensorFlow .h5 model file to convert.
    - output_file: str. Path where the TFLite model will be saved.
    - optimizations: list. List of optimizations to apply during conversion.

    Returns:
    - None. The function saves the TFLite model to the specified path.
    """
    if quantization_type == 'float16':
        converter.target_spec.supported_types = [tf.float16]
    elif quantization_type == 'int8':
    
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    with open(output_file, 'wb') as f:
        f.write(tflite_quant_model)


def main(argv):

    setup_gpu_configuration()
    # Full path for the original model and the quantized model
    model_filename = os.path.basename(FLAGS.original_model_path)
    details = parse_model_details_from_filename(model_filename)

    quantized_model_filename = generate_model_filename(details, FLAGS.version)
    quantized_model_path = os.path.join(
        FLAGS.quantized_model_dir, quantized_model_filename)

    convert_model_to_tflite(FLAGS.original_model_path, quantized_model_path)

    print(f"Quantized model saved as {quantized_model_path}")


if __name__ == '__main__':
    app.run(main)

# python TensorFlowLite.py --model_dir='./DeepLearning/SavedStandardModels' --model_filename='mnist_model_02-20_17-53.h5' --output_model_filename='mnist_model_quant.tflite' --optimization='DEFAULT'
