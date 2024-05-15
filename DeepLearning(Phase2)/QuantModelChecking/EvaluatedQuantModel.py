from tensorflow.keras.datasets import mnist
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags
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

    # Assuming prediction_digits and true_labels are lists containing model predictions and true labels respectively
    prediction_digits = []
    true_labels = testY.argmax(axis=1)
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

    precision = precision_score(
        true_labels, prediction_digits, average='macro')
    recall = recall_score(true_labels, prediction_digits, average='macro')
    f1 = f1_score(true_labels, prediction_digits, average='macro')
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, prediction_digits)

    accurate_count = sum(1 for i in range(len(prediction_digits))
                         if prediction_digits[i] == np.argmax(testY[i]))
    accuracy = accurate_count / len(prediction_digits)

    # print(f"Model accuracy: {accuracy * 100:.2f}%")
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

    return metrics


def save_evaluation_plot(metrics, plot_filepath):
    """
    Generates and saves a plot of the evaluation metrics.

    Args:
        metrics (dict): A dictionary containing the evaluation metrics.
        plot_filepath (str): The file path where the plot will be saved.
    """
    scalar_metrics = {k: v for k,
                      v in metrics.items() if not isinstance(v, np.ndarray)}
    confusion_matrix = metrics.get('confusion_matrix', None)

    num_plots = len(scalar_metrics) + \
        (1 if confusion_matrix is not None else 0)
    num_rows = int(np.ceil(num_plots / 2))

    fig, axs = plt.subplots(num_rows, 2, figsize=(
        15, num_rows * 5), constrained_layout=True)

    # Make axs a 2D array for easy indexing
    if num_rows == 1:
        axs = np.array([axs])

    # Plot scalar metrics
    for i, (metric_name, metric_value) in enumerate(scalar_metrics.items()):
        row, col = divmod(i, 2)
        axs[row, col].bar([metric_name], [metric_value], color='blue')
        axs[row, col].set_title(metric_name)

    # Plot confusion matrix if it exists
    if confusion_matrix is not None:
        # Place it at the end
        ax_cm = axs[-1, -1] if num_plots % 2 == 0 else axs[-1, 1]
        sns.heatmap(confusion_matrix, annot=True,
                    fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title('Confusion Matrix')
        ax_cm.set_xlabel('Predicted Labels')
        ax_cm.set_ylabel('True Labels')

    # Hide any unused axes
    for i in range(num_plots, num_rows * 2):
        row, col = divmod(i, 2)
        axs[row, col].axis('off')

    plt.suptitle('Model Evaluation Metrics')
    plt.savefig(plot_filepath)
    plt.close()


def main(argv):
    setup_gpu_configuration()
    # print(f"TensorFlow is running on: {tf.config.list_logical_devices()}")
    # if FLAGS.model_path is None:
    #     raise ValueError(
    #         "Please provide a model path using the --model_path flag.")
    testX, testY = load_test_dataset()
    interpreter = make_interpreter(FLAGS.model_path)
    metrics = evaluate_model(interpreter, testX, testY)
    # Generate the file path for the plot
    model_name = os.path.basename(FLAGS.model_path).replace('.tflite', '')
    plot_filename = f"evaluation_{model_name}.pdf"
    plot_filepath = os.path.join(
        os.path.dirname(FLAGS.model_path), plot_filename)

    # Save the evaluation plot
    save_evaluation_plot(metrics, plot_filepath)
    print(metrics)


if __name__ == '__main__':
    app.run(main)
