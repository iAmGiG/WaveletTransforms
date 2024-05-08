from absl import app, flags
import tensorflow as tf
import numpy as np
import os
import pywt
import csv
import ResNetRandomPruning as DWTR
from transformers import TFResNetForImageClassification, ResNetConfig

"""
TODO threshold values:
0, 0.236, 0.382, 0.5, 0.618, 0.786, 1
"""
# Define Flags.
FLAGS = flags.FLAGS
'''
- Wavelet: Type of wavelet to use for the Discrete Wavelet Transform (DWT).
    The choice of wavelet affects the base wavelet function used in the DWT,
    which can impact the model's ability to learn from data 
    that has been decomposed in different ways.
    Different wavelets capture different characteristics of the input data,
    potentially influencing the features learned by the model.
- Levels: Decomposition level for the DWT.
    A higher level results in a deeper decomposition of the input data,
        affecting the granularity of the wavelet transform applied to the input data.
    Higher levels may capture more abstract features 
        but also increase computational complexity and the risk of overfitting.
- Threshold: Threshold for thresholding the wavelet coefficients obtained from the DWT.
    A value of 0 means no thresholding is applied, 
        while the maximum value corresponds to the absolute value 
        of the largest wavelet coefficient. 
    Thresholding controls the sparsity of the wavelet coefficients,
    potentially impacting the interpretability of the learned features 
        and the model's ability to generalize.
'''
flags.DEFINE_string('model_path', None,
                    'Full path to the original DWT TensorFlow model.')
flags.DEFINE_enum('wavelet', 'haar', ['haar', 'db1', 'db2', 'coif1', 'bior1.3', 'rbio1.3', 'sym2', 'mexh', 'morl'],
                  'Type of wavelet to use for DWT.')
flags.DEFINE_integer('level', '1', 'Decomposition level for the DWT.')
flags.DEFINE_float('threshold', '1.0',
                   'Threshold for thresholding the wavelet coefficients obtained from the DWT.')
flags.DEFINE_string('csv_path', 'experiment_log.csv',
                    'Path to the CSV log file.')
flags.DEFINE_string('guid', None, 'GUID for the current pruning operation.')
# flags.mark_flag_as_required('model_path')
flags.mark_flag_as_required('guid')


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

# Load model


def get_model(model_path):
    """
    gets the model from the defined location.
    """
    # Load the pre-trained model
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print("Failed to load model:", e)
        return None

# Save model


def save_model(model, original_model_path, guid, isRandomPruned=False):
    """
    Save the model in TensorFlow SavedModel format.

    Args:
        model: The TensorFlow model to save.
        original_model_path: The path to the original model, used to determine save directory.
        guid: Unique identifier for the save directory.
        isRandomPruned: Flag to indicate if the model was pruned randomly.
    """
    # Determine the directory of the original model
    directory = os.path.dirname(
        original_model_path) if original_model_path is not None else os.getcwd()

    # Create a new directory name with the GUID
    new_directory_name = f"{directory}/pruned_{guid}"
    os.makedirs(new_directory_name, exist_ok=True)

    # Determine the save path based on pruning type
    model_type = "RP" if isRandomPruned else "DWT"
    model_save_path = f"{new_directory_name}/model_{model_type}_threshold{FLAGS.threshold}"

    try:
        # Save the model using PretrainedSavedModel
        model.save_pretrained(model_save_path)
        print(f"Model saved successfully at {model_save_path}")
    except Exception as e:
        print(f"Failed to save the model: {e}")
        raise


def log_to_csv(guid, details):
    """
    Log experiment details to a CSV file in a structured manner.

    Args:
        guid (str): Unique identifier for the experiment.
        details (dict): Dictionary containing all details to be logged.
    """
    # Define the CSV path from flags or directly
    csv_path = flags.FLAGS.csv_path

    # Check if the file exists to write headers; otherwise, append data
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as file:
        fieldnames = ['GUID', 'Wavelet', 'Level', 'Threshold', 'DWT Phase',
                      'Original Parameter Count', 'Non-zero Params', 'Total Pruned Count']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write the header only once

        # Construct the row data as a dictionary
        row_data = {'GUID': guid}
        row_data.update(details)
        writer.writerow(row_data)


# DWT process

# DWT all weights


def optimize_model(threshold, wavelet, level, guid, model=None):
    """
    Optimizes the weights of a pre-trained model by applying wavelet-based pruning.

    Args:
        threshold (float): The threshold value used for pruning wavelet coefficients.
        wavelet (str): The type of wavelet to be used for the wavelet transform.
        level (int): The decomposition level for the wavelet transform.

    Returns:
        None
    """
    total_pruned_count = 0
    original_param_count = model.count_params()
    skipped_layers = 0

    # Collect all Conv2D layers
    conv2d_layers = [layer for layer in model.layers if isinstance(
        layer, tf.keras.layers.Conv2D)]

    for layer in conv2d_layers:
        print(f"Processing layer: {layer.name}")
        weights = layer.get_weights()[0]
        print("Getting Coeffecients and Original shape")
        coeffs, original_shape = apply_dwt(
            weights=weights, wavelet=wavelet, level=level)
        print("Beginning the Prune process")
        pruned_coeffs, pruned_count_temp = prune_coeffs(
            coeffs=coeffs, threshold=threshold)
        total_pruned_count += pruned_count_temp
        print("Making the new weights")
        new_weights = reconstruct_weights(
            pruned_coeffs=pruned_coeffs, wavelet=wavelet, original_shape=original_shape)
        print("Setting the new weights")
        layer.set_weights([new_weights] + layer.get_weights()[1:])

    # Handle layers without weights
    non_conv2d_layers = [layer for layer in model.layers if not isinstance(
        layer, tf.keras.layers.Conv2D)]
    for layer in non_conv2d_layers:
        print(f"Layer {layer.name} has no weights and is skipped.")
        skipped_layers += 1

    print(f"Total parameters pruned: {total_pruned_count}")
    print(f"Total layers skipped: {skipped_layers}")

    # Save the optimized model
    save_model(model, original_model_path=FLAGS.model_path, guid=guid)

    # Log model changes
    model_changes = {
        "Wavelet": wavelet,
        "Level": level,
        "Threshold": threshold,
        "Layers skipped": skipped_layers,
        "DWT Phase": 'Yes',
        "Original Parameter count": original_param_count,
        "Total pruned count": total_pruned_count,
    }
    log_to_csv(guid, model_changes)

    return total_pruned_count


def apply_dwt(weights, wavelet='haar', level=1):
    """
    Applies the 2D Discrete Wavelet Transform (DWT) to the input weights matrix.

    Args:
        weights (np.ndarray): The weights matrix to be transformed.
        wavelet (str): The type of wavelet to be used for the transform. Default is 'haar'.
        level (int): The decomposition level for the wavelet transform. Default is 1.

    Returns:
        tuple: A tuple containing the wavelet coefficients and the original shape of the weights matrix.
    """
    # Handle weights with more than 2 dimensions by reshaping
    original_shape = weights.shape
    if len(original_shape) > 2:
        weights = weights.reshape((-1, original_shape[-1]))

    coeffs = pywt.wavedec2(weights, wavelet, level=level)

    # Coefficients need to be handled depending on further processing
    return coeffs, original_shape


def prune_coeffs(coeffs, threshold=0.85):
    """
    Prunes the wavelet coefficients based on the given threshold and counts the number of coefficients pruned.

    Args:
        coeffs (list): A list of wavelet coefficients, where each element can be an ndarray or a tuple of ndarrays.
        threshold (float): The threshold value for pruning coefficients.

    Returns:
        tuple: A tuple containing:
            - list: A list of pruned wavelet coefficients.
            - int: The total number of coefficients pruned.
    """
    pruned_coeffs = []
    total_pruned = 0  # Initialize the count of pruned coefficients

    for coeff in coeffs:
        if isinstance(coeff, tuple):
            # Initialize a list to hold pruned coefficients for this level
            pruned_level = []
            for c in coeff:
                # Count before pruning
                pre_prune_count = np.count_nonzero(c)

                # Prune coefficients
                pruned_c = np.where(np.abs(c) < threshold, 0, c)
                pruned_level.append(pruned_c)

                # Count after pruning and update total_pruned
                post_prune_count = np.count_nonzero(pruned_c)
                total_pruned += (pre_prune_count - post_prune_count)

            pruned_coeffs.append(tuple(pruned_level))
        else:
            # Handle the approximation coefficients similarly
            pre_prune_count = np.count_nonzero(coeff)
            pruned_c = np.where(np.abs(coeff) < threshold, 0, coeff)
            pruned_coeffs.append(pruned_c)
            post_prune_count = np.count_nonzero(pruned_c)
            total_pruned += (pre_prune_count - post_prune_count)

    return pruned_coeffs, total_pruned

# DWT random Weights


def random_pruning(prune_count, guid, model):
    """
    prunes randomly.
    """
    random_pruned_modle = DWTR.randomly_prune_model(
        model=model, num_prune=prune_count)
    # model=get_model(FLAGS.model_path), num_prune=prune_count)
    save_model(random_pruned_modle, original_model_path=FLAGS.model_path,
               guid=guid, isRandomPruned=True)

    # Log model changes
    model_changes = {
        "Random prune completed": True,
    }
    log_to_csv(guid, model_changes)


# IDWT process


def reconstruct_weights(pruned_coeffs, wavelet, original_shape):
    """
    Reconstructs the weight matrix from the pruned wavelet coefficients.

    Args:
        pruned_coeffs (list): A list of pruned wavelet coefficients.
        wavelet (str): The type of wavelet used for the wavelet transform.
        original_shape (tuple): The original shape of the weight matrix.

    Returns:
        np.ndarray: The reconstructed weight matrix.
    """
    # Reconstruct the weights from the pruned coefficients
    reconstructed = pywt.waverec2(pruned_coeffs, wavelet)

    # If original weights were reshaped, reshape back to original
    if len(original_shape) > 2:
        reconstructed = reconstructed.reshape(original_shape)

    return reconstructed

# Main


def main(argv):
    """
    runs the show
    """
    setup_tensorflow_gpu()
    # Parameters
    wavelet_type = FLAGS.wavelet  # Can be 'db1', 'sym2', etc.
    decomposition_level = FLAGS.level  # Level of wavelet decomposition
    threshold = FLAGS.threshold  # Pruning threshold
    guid = FLAGS.guid
    # if FLAGS.model_path is None:

    # Load the configuration if needed
    config = ResNetConfig.from_pretrained('microsoft/resnet-18')
    # Load the pre-trained model
    model = TFResNetForImageClassification.from_pretrained(
        'microsoft/resnet-18', config=config)
    prune_count = optimize_model(threshold=threshold, wavelet=wavelet_type,
                                 level=decomposition_level, guid=guid, model=model)
    random_pruning(prune_count, guid, model=model)

    # else:
    #     print(FLAGS.model_path)
    #     # Percentage of weights to prune selectively
    #     prune_count = optimize_model(threshold=threshold, wavelet=wavelet_type,
    #                                  level=decomposition_level, guid=guid)
    #     random_pruning(prune_count, guid)


if __name__ == '__main__':
    app.run(main)
