import time
import os
import numpy as np
import datetime
import re
import tensorflow as tf
from absl import app
from absl import flags
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', None,
                    'Directory where models are saved')
flags.DEFINE_boolean('use_gpu', True, 'Whether to use GPU or not')

# Mandatory flag definitions
flags.mark_flag_as_required('model_dir')


def parse_model_filename(filename):
    # Example pattern: haar_32_0.1_10_notquantized.h5
    patterns = [
        r"mnist_model_dwt_(?P<wavelet>\w+)_(?P<level>\d+)_(?P<threshold>[^_]+)_(?P<date>\d{2}-\d{2})\.h5$",
        r"mnist_model_dwt_(?P<wavelet>\w+)_(?P<level>\d+)(?:_(?P<threshold>[^_]+))?_(?P<date>\d{2}-\d{2})\.h5$"
    ]

    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return {
                'wavelet': match.group('wavelet'),
                'level': int(match.group('level')),
                'threshold': match.group('threshold'),
                'date': match.group('date'),
                'quantized': 'notquantized'
            }

    raise ValueError(f"Filename {filename} does not match expected patterns.")


def load_and_evaluate_model(model_dir):
    """
    Loads the MNIST dataset, prepares the data, and evaluates a pre-trained model's
    performance on this dataset. It calculates and returns the model's accuracy,
    size, inference time, and sparsity.

    This function assumes the existence of global FLAGS defined via the absl library,
    containing the necessary parameters to locate and load the model file.

    Returns:
    - A tuple containing:
        - A dictionary with keys 'accuracy', 'model_size', 'inference_time', and 'sparsity',
          each mapped to their respective evaluation metrics.
        - The path to the loaded model file.
    """
    # Load or prepare your test dataset
    (testX, testY), (_, _) = mnist.load_data()
    testX = testX / 255.0  # Normalize
    testY = to_categorical(testY)
    print(f"Loaded test dataset with {len(testX)} samples.")

    # Model file name pattern based on flags
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    if not model_files:
        print(f"No .h5 model files found in directory: {model_dir}")
        return None, None

    model_name = model_files[0]
    model_path = os.path.join(model_dir, model_name)
    model_details = parse_model_filename(model_name)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to true to avoid consuming all memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Define the distributed strategy
            strategy = tf.distribute.MirroredStrategy()
            print(f"Number of devices: {strategy.num_replicas_in_sync}")

            with strategy.scope():
                # Load your model within the distributed strategy scope
                model = load_model(model_path)

        except RuntimeError as e:
            print(e)
    else:
        # Load your model normally if GPUs are not available
        model = load_model(model_path)

    print(f"Model loaded from {model_path}. Evaluating...")

    # Evaluate accuracy
    loss, accuracy = model.evaluate(testX, testY)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Evaluate model size
    model_size = os.path.getsize(model_path)
    print(f"Model Size: {model_size / 1024:.2f} KB")

    # Evaluate inference time
    start_time = time.time()
    predictions = model.predict(testX)
    end_time = time.time()
    print(f"Inference Time: {end_time - start_time:.4f} seconds")

    # Evaluate sparsity
    weights = model.get_weights()
    zero_weights = np.sum([np.sum(w == 0) for w in weights])
    total_weights = np.sum([w.size for w in weights])
    sparsity = zero_weights / total_weights
    print(f"Sparsity: {sparsity*100:.2f}%")

    # Dictionary to hold metrics
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'model_size': model_size / 1024,  # Convert to KB for consistency
        'inference_time': end_time - start_time,
        'predictions': predictions,
        'sparsity': sparsity * 100,  # Convert to percentage
    }

    return metrics, model_path, model_details


def plot_metrics(metrics, model_path, model_details):
    """
    Generates and saves a plot of the model evaluation metrics as a PDF file.

    This function creates a 2x2 subplot, with each quadrant representing one of
    the following metrics: accuracy, model size, inference time, and sparsity. The plot
    is then saved in the same directory as the model file.

    Parameters:
    - metrics (dict): A dictionary containing the evaluation metrics of the model.
                      The expected keys are 'accuracy', 'model_size', 'inference_time',
                      and 'sparsity', each associated with their respective values.
    - model_path (str): The file path of the model, used to determine the directory
                        where the plot will be saved.

    The plot is saved as 'model_evaluation_metrics.pdf' in the directory of the model file.
    """
    # Generate confusion matrix from predictions
    # Assuming testY is available in metrics
    y_true = np.argmax(metrics['testY'], axis=1)
    y_pred = np.argmax(metrics['predictions'], axis=1)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Adjust the figsize to fit your screen better
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle(
        f"Model Evaluation Metrics - {model_details['wavelet']}, Level: {model_details['level']}, model_details: {model_details['threshold']}"
    )

    # Accuracy
    axs[0, 0].bar(['Accuracy'], [metrics['accuracy']])
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_ylim(0, 1)

    # Model Size (convert to KB)
    axs[0, 1].bar(['Model Size'], [metrics['model_size']])
    axs[0, 1].set_ylabel('Size (KB)')

    # Inference Time
    axs[1, 0].bar(['Inference Time'], [metrics['inference_time']])
    axs[1, 0].set_ylabel('Time (seconds)')

    # Sparsity
    axs[1, 1].bar(['Sparsity'], [metrics['sparsity']])
    axs[1, 1].set_ylabel('Sparsity (%)')
    axs[1, 1].set_ylim(0, 100)

    # Loss
    axs[2, 0].bar(['Loss'], [metrics['loss']])
    axs[2, 0].set_ylabel('Loss')

    # You can leave the last subplot empty or use it for another metric if you have one
    # This will turn off the 6th subplot (no plotting will happen here)
    axs[2, 1].axis('off')

    # Plot the confusion matrix
    axs[2, 1].set_title('Confusion Matrix')
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axs[2, 1], cmap='Blues')
    axs[2, 1].set_xlabel('Predicted Labels')
    axs[2, 1].set_ylabel('True Labels')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot as a PDF in the same directory as the model
    plot_filename = f"evaluation_{model_details['wavelet']}_{model_details['quantized']}.pdf"
    plot_filepath = os.path.join(os.path.dirname(model_path), plot_filename)
    plt.savefig(plot_filepath)
    print(f"Saved plot to {plot_filepath}")


def plot_confusion_matrix(y_true, y_pred, model_details, model_path):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_details['wavelet']}")
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')

    # Adjust this filename as necessary to include all relevant model details
    plot_filename = f"confusion_matrix_{model_details['wavelet']}_{model_details['date']}.pdf"
    plot_filepath = os.path.join(os.path.dirname(model_path), plot_filename)
    plt.savefig(plot_filepath)
    print(f"Saved plot to {plot_filepath}")


def plot_accuracy_over_samples(testY, predictions, model_details, model_path):
    test_labels = np.argmax(testY, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    correct_predictions = test_labels == predicted_labels

    plt.figure(figsize=(15, 5))
    plt.plot(correct_predictions.nonzero()[0], np.ones_like(
        correct_predictions.nonzero()[0]), 'go', label='Correct', markersize=1)
    plt.plot((~correct_predictions).nonzero()[0], np.zeros_like(
        (~correct_predictions).nonzero()[0]), 'ro', label='Incorrect', markersize=1)
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.xlabel('Sample')
    plt.title(
        f"Prediction Accuracy Over Samples - {model_details['wavelet']} - {model_details['quantized']}")
    plt.legend()

    plot_filename = f"accuracy_over_samples_{model_details['wavelet']}_{model_details['quantized']}.pdf"
    plot_filepath = os.path.join(os.path.dirname(model_path), plot_filename)
    plt.savefig(plot_filepath)
    print(f"Saved plot to {plot_filepath}")


def find_model_file(directory, pattern):
    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"The specified directory does not exist: {directory}")

    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            return os.path.join(directory, filename)
    return None


def main(argv):
    # Assuming load_and_evaluate_model returns a dictionary of metrics
    if FLAGS.use_gpu:
        # or another device index if you have multiple GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        # forces TensorFlow to use CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Ensure this function also returns the model_path
    metrics, model_path, model_details = load_and_evaluate_model(
        FLAGS.model_dir)
    if metrics:
        plot_metrics(metrics, model_path, model_details)


if __name__ == '__main__':
    app.run(main)
