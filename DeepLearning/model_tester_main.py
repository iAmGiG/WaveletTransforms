import time
import os
import numpy as np
import datetime
import re
from absl import app
from absl import flags
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('wavelet', None, 'Wavelet used in the model')
flags.DEFINE_integer('level', None, 'Level of the wavelet transformation')
flags.DEFINE_string('model_dir', './DeepLearning/SavedStandardModels',
                    'Directory where models are saved')
flags.DEFINE_string(
    'threshold', None, 'Threshold used in the model, as a string to match filenames')
flags.DEFINE_boolean('use_gpu', True, 'Whether to use GPU or not')

# Mandatory flag definitions
flags.mark_flag_as_required('use_gpu')
flags.mark_flag_as_required('wavelet')
flags.mark_flag_as_required('level')


def load_and_evaluate_model():
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

    # Model file name pattern based on flags
    today_date = datetime.datetime.now().strftime('%m-%d')
    model_filename_pattern = f"mnist_model_dwt_{FLAGS.wavelet}_{FLAGS.level}_{FLAGS.threshold}_*.h5"
    model_path = find_model_file(FLAGS.model_dir, model_filename_pattern)

    if not model_path:
        print("Model file not found")
        return

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

    # Evaluate accuracy
    loss, accuracy = model.evaluate(testX, testY)
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
        'accuracy': accuracy,
        'model_size': model_size / 1024,  # Convert to KB for consistency
        'inference_time': end_time - start_time,
        'sparsity': sparsity * 100,  # Convert to percentage
    }

    return metrics, model_path

def plot_metrics(metrics, model_path):
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
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Model Evaluation Metrics')
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Model Evaluation Metrics')

    # Accuracy
    axs[0, 0].bar(['Accuracy'], [metrics['accuracy']])
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_ylim(0, 1)

    # Model Size
    axs[0, 1].bar(['Model Size'], [metrics['model_size'] / 1024])  # Convert to KB
    axs[0, 1].set_ylabel('Size (KB)')

    # Inference Time
    axs[1, 0].bar(['Inference Time'], [metrics['inference_time']])
    axs[1, 0].set_ylabel('Time (seconds)')

    # Sparsity
    axs[1, 1].bar(['Sparsity'], [metrics['sparsity']])
    axs[1, 1].set_ylabel('Sparsity (%)')
    axs[1, 1].set_ylim(0, 100)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot as a PDF in the same directory as the model
    model_directory = os.path.dirname(model_path)
    plot_filename = os.path.join(model_directory, 'model_evaluation_metrics.pdf')
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")


def find_model_file(directory, pattern):
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            return os.path.join(directory, filename)
    return None


def main(argv):
    # Assuming load_and_evaluate_model returns a dictionary of metrics
    if FLAGS.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or another device index if you have multiple GPUs
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # forces TensorFlow to use CPU

    metrics, model_path = load_and_evaluate_model()  # Ensure this function also returns the model_path
    plot_metrics(metrics, model_path)


if __name__ == '__main__':
    app.run(main)
