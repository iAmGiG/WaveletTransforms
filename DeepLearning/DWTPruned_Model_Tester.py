import uuid
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from datetime import datetime
from absl import app, flags
import time
import os
import numpy as np
import re
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', None,
                    'Full path to the model file to be evaluated')
flags.DEFINE_boolean('use_gpu', False, 'Whether to use GPU or not')
# TODO: Remove when done with this round of testing, and replace with regex.
flags.DEFINE_boolean('wasRandModel', False, "use if model was random pruned")
# Mandatory flag definitions
flags.mark_flag_as_required('model_path')

def parse_model_directory(path):
    """
    Parse the directory path or filename to extract the UUID and the last few digits of the UUID.
    If a UUID is not found, returns the current month and day.

    Parameters:
    - path (str): The directory path or filename to parse.

    Returns:
    - dict: A dictionary containing the parsed elements 'uuid' and 'uuid_digits'.
            If a UUID is not found, 'uuid' will be None, and 'uuid_digits' will be the current month and day.
    """
    # Extract the base name from the path
    base_name = os.path.basename(path)

    # Regular expression pattern to match a UUID
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'

    # Find the UUID in the base name
    match = re.search(uuid_pattern, base_name, re.IGNORECASE)

    if match:
        uuid_str = match.group()
        uuid_obj = uuid.UUID(uuid_str)

        # Get the full UUID
        full_uuid = str(uuid_obj)

        # Get the last few digits of the UUID (adjust the number as needed)
        uuid_digits = full_uuid[-4:]

        return {'uuid': full_uuid, 'uuid_digits': uuid_digits}
    else:
        # If a UUID is not found, use the current month and day as a fallback
        current_date = datetime.now().strftime('%m%d')
        return {'uuid': None, 'uuid_digits': current_date}
    
def load_and_evaluate_model(model_file_path):
    """
    Load the MNIST dataset and a pre-trained model from a specified file path, evaluate the model's performance,
    and calculate metrics such as accuracy, model size, inference time, and sparsity.

    Parameters:
    - model_file_path (str): The full path to the .h5 model file to be evaluated.

    Returns:
    - tuple: Containing:
        - metrics (dict): A dictionary with evaluation metrics of the model.
        - model_file_path (str): The full path to the evaluated model file.
        - model_details (dict): Parsed details from the model filename.
        - testY (np.array): The true labels from the MNIST test dataset.

    If the model file cannot be found, returns None for all elements in the tuple.
    """
    # Load or prepare your test dataset
    (testX, testY), (_, _) = mnist.load_data()
    testX = testX / 255.0  # Normalize
    # Test zone Begin
    print(f"Shape of testY: {testY.shape}\nShape of testX: {testX.shape}")
    # testX = testX.reshape((-1, 28, 28, 1))  # For MNIST dataset
    # testX = testX.reshape((-1, 28, 28))
    # testX = np.expand_dims(testX, axis=-1)
    # testY_categorical = tf.one_hot(testY, depth=10)
    # Test zene ends, , num_classes=10
    testY_categorical = tf.keras.utils.to_categorical(testY)
    print(f"Loaded test dataset with {len(testX)} samples.")

    # Verify the model file exists directly
    if not os.path.isfile(model_file_path):
        print(f"Model file not found: {model_file_path}")
        return None, None, None, None

    model_path, model_filename = os.path.split(model_file_path)
    model_details = parse_model_directory(model_filename)


    # Load the model directly from the provided path
    print(f"Loading model from {model_path}...")
    model = load_model(model_file_path)
    # TODO Look like we need this if we random compile, replace with dynamic elements, so true for rand models.
    if FLAGS.wasRandModel:
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Evaluate accuracy, batch_size=32 - use as needed., verbose=1
    loss, accuracy = model.evaluate(x=testX, y=testY_categorical)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Evaluate model size
    model_size = os.path.getsize(model_file_path)
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
    return metrics, model_file_path, model_details, testY


def plot_metrics(metrics, model_path, model_details, testY):
    """
    Generates a set of plots for various model evaluation metrics and saves them as a PDF file.

    Parameters:
    - metrics (dict): A dictionary containing the evaluation metrics.
    - model_path (str): The file path of the model, to determine save location of the plot.
    - model_details (dict): Details of the model for use in the plot title.
    - testY (np.array): True labels for the test dataset, used for the confusion matrix.

    The function will generate a subplot of 2x3, plotting accuracy, model size, inference time, sparsity, loss,
    and a confusion matrix, and save it as 'model_evaluation_metrics.pdf' in the directory of the model file.
    """
    # Generate confusion matrix from predictions
    # Assuming testY is available in metrics
    y_true = testY  # Directly use testY if it's already the true labels
    # Convert predictions from one-hot encoding to labels
    y_pred = np.argmax(metrics['predictions'], axis=1)

    # Adjust the figsize to fit your screen better
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    # TODO
    # fig.suptitle(
    #     f"Model Evaluation Metrics - {model_details['wavelet']}, Level: {model_details['level']}, model_details: {model_details['threshold']}"
    # )
    fig.suptitle(f"Model Evaluation Metrics {datetime.now().strftime('%m%d')}")

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

    # Adjust this part to plot the confusion matrix within the subplot framework
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axs[2, 1], cmap='Blues')
    axs[2, 1].set_title('Confusion Matrix')
    axs[2, 1].set_xlabel('Predicted Labels')
    axs[2, 1].set_ylabel('True Labels')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot as a PDF in the same directory as the model TODO
    # plot_filename = f"evaluation_{model_details['wavelet']}_{model_details['quantized']}.pdf"
    plot_filename = f"evaluation_{datetime.now().strftime('%m%d')}_wasRand-{FLAGS.wasRandModel}.pdf"
    plot_filepath = os.path.join(os.path.dirname(model_path), plot_filename)
    plt.savefig(plot_filepath)
    print(f"Saved plot to {plot_filepath}")


def plot_confusion_matrix(y_true, y_pred, model_details, model_path):
    """
    Generates and saves a confusion matrix plot for the given predictions and true labels.

    Parameters:
    - y_true (np.array): The true labels.
    - y_pred (np.array): The predicted labels by the model.
    - model_details (dict): Details of the model for use in the plot title.
    - model_path (str): The path to the model file, used to determine save location for the plot.

    Saves the confusion matrix plot as a PDF file named 'confusion_matrix_<wavelet>_<date>.pdf'
    in the same directory as the model file.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    # plt.title(f"Confusion Matrix - {model_details['wavelet']}") TODO
    plt.title(f"Confusion Matrix {datetime.now().strftime('%m%d')}")
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')

    # Adjust this filename as necessary to include all relevant model details TODO
    #  = f"confusion_matrix_{model_details['wavelet']}_{model_details['date']}.pdf"
    plot_filename = f"confusion_matrix_{datetime.now().strftime('%m%d')}"
    plot_filepath = os.path.join(os.path.dirname(model_path), plot_filename)
    plt.savefig(plot_filepath)
    print(f"Saved plot to {plot_filepath}")


def plot_accuracy_over_samples(testY, predictions, model_details, model_path):
    """
    Generates and saves a scatter plot showing prediction accuracy over samples.

    Parameters:
    - testY (np.array): The true labels for the test dataset.
    - predictions (np.array): The predicted probabilities by the model for each class.
    - model_details (dict): Details of the model for use in the plot title.
    - model_path (str): The path to the model file, used to determine save location for the plot.

    Saves the plot as a PDF file named 'accuracy_over_samples_<wavelet>_<quantized>.pdf'
    in the same directory as the model file.
    """
    test_labels = testY
    predicted_labels = np.argmax(predictions, axis=1)
    correct_predictions = test_labels == predicted_labels

    plt.figure(figsize=(15, 5))
    plt.plot(correct_predictions.nonzero()[0], np.ones_like(
        correct_predictions.nonzero()[0]), 'go', label='Correct', markersize=1)
    plt.plot((~correct_predictions).nonzero()[0], np.zeros_like(
        (~correct_predictions).nonzero()[0]), 'ro', label='Incorrect', markersize=1)
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.xlabel('Sample')
    # plt.title( TODO
    #     f"Prediction Accuracy Over Samples - {model_details['wavelet']} - {model_details['quantized']}")
    plt.title(f"Prediction Accuracy Over Samples {datetime.now().strftime('%m%d')}")
    plt.legend()
    

    # f"accuracy_over_samples_{model_details['wavelet']}_{model_details['quantized']}.pdf"
    plot_filename = f"accuracy_over_samples_{datetime.now().strftime('%m%d')}_wasRand-{FLAGS.wasRandModel}.pdf"
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
    """
    Main function that configures GPU settings, loads and evaluates a model,
    and generates evaluation plots.

    Uses command-line flags to control behavior such as the path to the model file
    and whether to use a GPU for evaluation.

    Exits with a status code indicating success or failure of the operations.
    """
    # Assuming load_and_evaluate_model returns a dictionary of metrics
    if FLAGS.use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus}")
            except RuntimeError as e:
                print(f"Error configuring GPU: {e}")
    else:
        print("Using CPU for evaluation.")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Ensure this function also returns the model_path
    # Assuming this now directly contains the model file path
    metrics, model_path, model_details, testY = load_and_evaluate_model(
        FLAGS.model_path)
    if metrics:
        plot_metrics(metrics, model_path, model_details, testY)
        plot_accuracy_over_samples(
            testY, metrics['predictions'], model_details, model_path)


if __name__ == '__main__':
    app.run(main)
