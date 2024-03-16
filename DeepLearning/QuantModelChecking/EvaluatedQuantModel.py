from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
from absl import app
from absl import flags
import os
import re
from datetime import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', None, 'Full path to the TFLite model file.')
flags.DEFINE_boolean(
    'use_gpu', True, 'Enable GPU support for TensorFlow operations')


def setup_gpu_configuration():
    """
    Configures TensorFlow to use GPU if available and enabled through flags.
    """
    if FLAGS.use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Using GPU for TensorFlow operations.")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
        else:
            print("No GPUs found. Using CPU instead.")
    else:
        print("GPU support is disabled. Using CPU instead.")
        tf.config.set_visible_devices([], 'GPU')


def find_most_recent_model(directory):
    """
    Finds the most recent model in the directory, optionally filtering by compression.
    """
    pattern = re.compile(
        r'mnist_model_dwt_(?P<wavelet>\w+)_(?P<level>\d+)_thresh(?P<threshold>\d+)_.*\.tflite$')
    models = [f for f in os.listdir(directory) if pattern.match(f)]
    if not models:
        return None, None
    latest_model = max(models, key=lambda x: datetime.strptime(
        pattern.match(x).group('date'), '%Y-%m-%d'))
    details = pattern.match(latest_model).groupdict()
    return os.path.join(directory, latest_model), details


def load_test_dataset():
    """
    Loads and preprocesses the MNIST test dataset.
    """
    (testX, testY), (_, _) = mnist.load_data()
    testX = testX.astype('float32') / 255.0
    testX = np.expand_dims(testX, -1)  # Add channel dimension
    testY = tf.keras.utils.to_categorical(testY)
    return testX, testY


def make_interpreter(model_path):
    """
    Loads the TFLite model and allocates tensors.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def evaluate_model(interpreter, testX, testY):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    print(f"Input shape expected by the model: {input_shape}")

    prediction_digits = []
    for i, test_image in enumerate(testX):
        # Reshape test_image to match the input shape expected by the model
        test_image_reshaped = np.reshape(test_image, input_shape[1:])

        # Ensure test_image is float32
        test_image_reshaped = test_image_reshaped.astype(np.float32)

        # Add batch dimension
        test_image_reshaped = np.expand_dims(test_image_reshaped, axis=0)

        interpreter.set_tensor(input_index, test_image_reshaped)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_index)
        digit = np.argmax(output_data[0])
        prediction_digits.append(digit)

    accurate_count = sum(1 for i in range(len(prediction_digits))
                         if prediction_digits[i] == np.argmax(testY[i]))
    accuracy = accurate_count / len(prediction_digits)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main(argv):
    setup_gpu_configuration()
    print(f"TensorFlow is running on: {tf.config.list_logical_devices()}")
    if FLAGS.model_path is None:
        raise ValueError(
            "Please provide a model path using the --model_path flag.")
    testX, testY = load_test_dataset()
    interpreter = make_interpreter(FLAGS.model_path)
    accuracy = evaluate_model(interpreter, testX, testY)
    print(f'Quantized model accuracy: {accuracy*100:.2f}%')


if __name__ == '__main__':
    app.run(main)
