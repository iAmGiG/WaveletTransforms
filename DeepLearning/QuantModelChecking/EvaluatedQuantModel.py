from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
from absl import app
from absl import flags
import os
import re
from datetime import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '../SavedTFliteModels',
                    'Directory where TFLite models are saved.')


def find_most_recent_model(directory):
    """
    I want this method to also record the compression from the pattern as well. 
    I might just do the sigular date, the current patter includes time.
    """
    # Regex to extract datetime and possibly threshold from filenames
    pattern = re.compile(r'.*_(\d{8}T\d{6})_.*\.tflite$')
    models = [f for f in os.listdir(directory) if pattern.match(f)]
    if not models:
        return None
    latest_model = max(models, key=lambda x: datetime.strptime(
        pattern.match(x).group(1), '%Y%m%dT%H%M%S'))
    return os.path.join(directory, latest_model)


def load_test_dataset():
    # Load MNIST test dataset
    (testX, testY), (_, _) = mnist.load_data()
    testX = testX.astype('float32') / 255.0
    # Make sure to match the model's input shape
    testX = np.expand_dims(testX, -1)
    testY = tf.keras.utils.to_categorical(testY)
    return testX, testY


def make_interpreter():
    # Load the TFLite model and allocate tensors
    model_path = find_most_recent_model(FLAGS.model_dir)
    if model_path is None:
        raise FileNotFoundError(
            "No TFLite model found in the specified directory.")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_i_o_details(interpreter):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


def evaluate_model(interpreter, testX, testY):
    # Define a function to evaluate accuracy
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    prediction_digits = []
    for test_image in testX:
        # Ensure test_image is float32
        test_image = test_image.astype(np.float32)

        # Make sure test_image is of shape [28, 28] (or [28, 28, 1] and then squeeze if needed)
        if test_image.ndim == 4:  # Assuming the shape is [1, 28, 28, 1]
            test_image = np.squeeze(test_image, axis=0)  # Now [28, 28, 1]
        if test_image.ndim == 3 and test_image.shape[-1] == 1:
            # Squeeze channel dimension if it's 1
            test_image = np.squeeze(test_image, axis=-1)

        # Add batch dimension
        test_image = np.expand_dims(test_image, axis=0)

        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_index)
        digit = np.argmax(output_data[0])
        prediction_digits.append(digit)

    # Compute accuracy
    accurate_count = sum(1 for i in range(len(prediction_digits))
                         if prediction_digits[i] == np.argmax(testY[i]))
    accuracy = accurate_count / len(prediction_digits)
    return accuracy


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # Evaluate the model
    testX, testY = load_test_dataset()
    interpreter = make_interpreter()
    get_i_o_details(interpreter=interpreter)
    accuracy = evaluate_model(interpreter, testX, testY)
    print(f'Quantized model accuracy: {accuracy*100:.2f}%')


if __name__ == '__main__':
    app.run(main)
