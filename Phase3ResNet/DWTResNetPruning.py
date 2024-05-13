import os
import json
import tensorflow as tf
from transformers import TFResNetModel
import pywt
import numpy as np
from absl import app, flags, logging

FLAGS = flags.FLAGS

# Define command-line flags
flags.DEFINE_string('model_path', None,
                    'Path to the pre-trained model to be pruned.')
flags.DEFINE_float('threshold', None,
                   'Threshold value for identifying insignificant weights.')
flags.DEFINE_string('wavelet_type', None,
                    'Type of wavelet function used in the decomposition.')
flags.DEFINE_integer('decomp_level', None,
                     'Decomposition level for wavelet transform.')
flags.DEFINE_string(
    'save_path', None, 'Directory path where the pruned model and logs are saved.')


def setup_tensorflow_gpu():
    """ Set TensorFlow to use any available GPU. """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set TensorFlow to use only the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print("Error setting up GPU:", e)


def save_and_log_results(model, pruned_weights_info, save_path, model_type):
    """
    Saves the pruned model and logs detailed information about the pruning process to both a text file and a structured JSON file for easier analysis.

    Args:
        model (tf.keras.Model): The pruned model to be saved.
        pruned_weights_info (dict): Detailed information about the weights pruned, including count and percentage of total.
        save_path (str): The directory path where the pruned model and logs are saved.
        model_type (str): Identifier for the type of pruning ('DWT' or 'Random') to distinguish between save files.

    Returns:
        str: The path to the saved model file.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the pruned model
    model_save_path = os.path.join(save_path, f"pruned_model_{model_type}.h5")
    model.save(model_save_path)

    # Log the pruned weights information to a text file
    log_file_path = os.path.join(save_path, f"pruning_log_{model_type}.txt")
    with open(log_file_path, "w") as log_file:
        for layer_name, layer_info in pruned_weights_info.items():
            log_file.write(f"Layer: {layer_name}\n")
            log_file.write(
                f"  Pruned weights count: {layer_info['pruned_weights_count']}\n")
            log_file.write(
                f"  Pruned weights percentage: {layer_info['pruned_weights_percentage']:.2f}%\n")
            log_file.write("\n")

    # Log the pruned weights information to a JSON file
    json_file_path = os.path.join(save_path, f"pruning_info_{model_type}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(pruned_weights_info, json_file, indent=4)

    return model_save_path


def prune_layer_tf(layer, threshold, wavelet_type, decomp_level):
    """
    Prunes the weights of a given TensorFlow layer using a wavelet-based approach.

    Args:
        layer (tf.keras.layers.Layer): TensorFlow layer to be pruned.
        threshold (float): Pruning threshold to determine the insignificance of weights.
        wavelet_type (str): Wavelet type used for the DWT.
        decomp_level (int): Decomposition level for the DWT.

    Returns:
        tf.keras.layers.Layer: Layer with pruned weights.
    """
    # Check if the layer has a kernel attribute
    if hasattr(layer, 'kernel'):
        # Get the current weights
        weights = layer.kernel

        # Apply wavelet decomposition and pruning (TensorFlow-specific implementation)
        pruned_weights = apply_wavelet_decomposition_tf(
            weights, wavelet_type, decomp_level, threshold)

        # Update the layer's weights with the pruned weights
        layer.kernel = pruned_weights

    return layer


def apply_wavelet_decomposition_tf(weights, wavelet_type, decomp_level, threshold):
    """
    TensorFlow-specific implementation of wavelet decomposition and pruning.

    Args:
        weights (Tensor): Weights tensor from a TensorFlow layer.
        wavelet_type (str): Type of wavelet used.
        decomp_level (int): Decomposition level for the DWT.
        threshold (float): Pruning threshold.

    Returns:
        Tensor: Pruned weights tensor.
    """
    # Placeholder for TensorFlow-compatible DWT operations
    pruned_weights = tf.identity(weights)
    # Reshape weights to 1D array
    weights_1d = weights.ravel()

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(weights_1d, wavelet_type,
                          mode='periodic', level=decomp_level)

    # Apply thresholding to prune insignificant coefficients
    pruned_coeffs = []
    for coeff in coeffs:
        pruned_coeff = pywt.threshold(coeff, threshold, mode='soft')
        pruned_coeffs.append(pruned_coeff)

    # Reconstruct the weights using the pruned coefficients
    pruned_weights_1d = pywt.waverec(
        pruned_coeffs, wavelet_type, mode='periodic')

    # Reshape the reconstructed weights back to the original shape
    pruned_weights = pruned_weights_1d.reshape(weights.shape)

    return pruned_weights


def execute_pruning_workflow(model, threshold, wavelet_type, decomp_level, save_path):
    """
    
    """
    pruned_weights_info = {}

    # Iterate over all layers in the model
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            # Prune the layer using wavelet-based pruning (TensorFlow-specific implementation)
            layer = prune_layer_tf(
                layer, threshold, wavelet_type, decomp_level)

            # Update the pruned weights information
            kernel_pruned_weights_info = get_pruned_weights_info(layer.kernel)
            pruned_weights_info[layer.name] = kernel_pruned_weights_info

    # Save the pruned model and log the pruning results
    model_save_path = save_and_log_results(
        model, pruned_weights_info, save_path, 'DWT')
    logging.info(f"Pruned model saved to: {model_save_path}")


def main(argv):
    """
    loads model, then has the execute pruning workflow run, later the random pruning.
    """
    # Load the pre-trained ResNet model
    setup_tensorflow_gpu()
    model = TFResNetModel.from_pretrained(FLAGS.model_path)
    # Load the configuration if needed
    # config = ResNetConfig.from_pretrained('microsoft/resnet-18')
    # # Load the pre-trained model
    # model = TFResNetForImageClassification.from_pretrained(
    #     'microsoft/resnet-18', config=config)

    # Execute the pruning workflow
    execute_pruning_workflow(
        model, FLAGS.threshold, FLAGS.wavelet_type, FLAGS.decomp_level, FLAGS.save_path)


if __name__ == '__main__':
    app.run(main)
