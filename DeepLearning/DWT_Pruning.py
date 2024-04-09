from absl import app, flags
import tensorflow as tf
import numpy as np
import os
import datetime
import pywt
import uuid
import json
import Random_Pruning as DWTR

"""
TODO: show the linear relationship of pruning and threshold dropping.
so when we prune how much, 
when we modify the percent

we want to see about pruning a selection by percentage that isn't random.
at what point do we see a large accuracy drop?
how much can we go by this method.

need to understand HOW this pruning is working, 
are the weights begin adjusted, 
the weights are staying?
TODO: HOW ARE THE VALUES BEIGN PRUNNED
"""

# Define Flags.
FLAGS = flags.FLAGS
'''
- Wavelet: Type of wavelet to use for the Discrete Wavelet Transform (DWT).
    The choice of wavelet affects the base wavelet function used in the DWT,
    which can impact the model's ability to learn from data that has been decomposed in different ways.
    Different wavelets capture different characteristics of the input data,
    potentially influencing the features learned by the model.
- Save Directory: Directory to save the trained models.
    If you want to save models to the default location (./SavedDWTModels),
    you need to be in the DeepLearning folder when running the script.
    Otherwise, it will create a new folder and subdirectories in the current working directory.
    This flag determines where the trained models are stored for future use or analysis.
- Batch size: Batch size for training.
    This determines the number of samples that will be propagated through 
        the neural network at once during the training process.
    A larger batch size requires more memory 
        but can lead to faster training and smoother updates to the model's weights.
    However, very large batch sizes can also degrade generalization performance.
- Epochs: Number of epochs or iterations over the entire training dataset.
    This specifies how many times the learning algorithm will 
        work through the entire training dataset.
    More epochs can lead to better convergence and potentially higher accuracy,
    but also increase the risk of overfitting if not appropriately regularized.
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
flags.DEFINE_float('threshold', '0.236',
                   'Threshold for thresholding the wavelet coefficients obtained from the DWT.')
flags.DEFINE_string('quant_level', 'binary',
                    'Level of quantization (binary, ternary, etc.)')
# flags.DEFINE_float('prun_percent', 50, 'Percentage of weights to quantize')
flags.DEFINE_boolean('random_quantize', False,
                     'Enable random(ly selected) weight quantization')
flags.mark_flag_as_required('model_path')


def generate_guid():
    """
    make a new GUID
    """
    return uuid.uuid4().hex

# Load model


def get_model(model_path):
    """
    gets the model from the defined location.

    """
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    return model

# Save model


def save_model(model, original_model_path, guid, isRandomPruned=False):
    """
    save the model
    """
    # Determine the directory of the original model
    directory = os.path.dirname(original_model_path)
    # Create a new directory name with the GUID
    new_directory_name = f"{directory}/optimized_{guid}"
    os.makedirs(new_directory_name, exist_ok=True)
    # Save the model in the new directory
    if not isRandomPruned:
        model_save_path = f"{new_directory_name}/model.h5"
    else:
        model_save_path = f"{new_directory_name}/model_RandPruned.h5"
    model.save(model_save_path)
    # return model_save_path


def log_model_changes(log_path, guid, model_changes):
    """
    helps me keep track of changes
    """
    # Assume model_changes is a dictionary containing change details
    if not os.path.exists(log_path):
        with open(log_path, 'w') as file:
            json.dump([], file)

    with open(log_path, 'r+') as file:
        logs = json.load(file)
        logs.append({"guid": guid, "changes": model_changes})
        file.seek(0)
        json.dump(logs, file, indent=4)

# DWT process

# DWT all weights


def optimize_model(threshold, wavelet, level):
    """
    Optimizes the weights of a pre-trained model by applying wavelet-based pruning.

    Args:
        threshold (float): The threshold value used for pruning wavelet coefficients.
        wavelet (str): The type of wavelet to be used for the wavelet transform.
        level (int): The decomposition level for the wavelet transform.

    Returns:
        None
    """
    model = get_model(FLAGS.model_path)
    guid = generate_guid()
    total_pruned_count = 0

    for layer in model.layers:
        if layer.weights:
            print("Getting Weights")
            weights = layer.get_weights()[0]
            print("Getting Coeffecients and Original shape")
            coeffs, original_shape = apply_dwt(
                weights=weights, wavelet=wavelet, level=level)
            print("Begining the Prune process")
            pruned_coeffs, pruned_count = prune_coeffs(
                coeffs=coeffs, threshold=threshold)
            total_pruned_count += pruned_count
            print("Makinng the new weights")
            new_weights = reconstruct_weights(
                pruned_coeffs=pruned_coeffs, wavelet=wavelet, original_shape=original_shape)
            print("Setting the new weights")
            layer.set_weights([new_weights] + layer.get_weights()[1:])

    # At this point, `total_pruned_count` holds the total number of parameters pruned across the model
    print(f"Total parameters pruned: {total_pruned_count}")
    # Save the optimized model
    save_model(model, original_model_path=FLAGS.model_path, guid=guid, )

    # Log model changes
    model_changes = {
        "wavelet": wavelet,
        "level": level,
        "threshold": threshold,
        "total pruned count": total_pruned_count,
        # Include other relevant details such as pruning percentage, quantization level, etc.
    }
    log_model_changes("model_changes_log.json", guid, model_changes)

    return total_pruned_count, guid


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


def random_pruning(prune_count, guid):
    """
    prunes randomly.
    """
    random_pruned_modle = DWTR.randomly_prune_model(
        model=get_model(FLAGS.model_path), num_prune=prune_count)
    save_model(random_pruned_modle, original_model_path=FLAGS.model_path,
               guid=guid, isRandomPruned=True)

    # Log model changes
    model_changes = {
        "Random prune completed": True,
    }
    log_model_changes("model_changes_log.json", guid, model_changes)


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
    print(FLAGS.model_path)

    # Parameters
    wavelet_type = FLAGS.wavelet  # Can be 'db1', 'sym2', etc.
    decomposition_level = FLAGS.level  # Level of wavelet decomposition
    threshold = FLAGS.threshold  # Pruning threshold

    # Percentage of weights to prune selectively
    """ 
    NOTE: 
    this code was original set here because the initial idea was to test based on a few given criteria,
    that random testing was independent of the selected testing.
    ---The current testing is taking a pruned count from the selected optimization, then perfroming a pruning again, on the model based on the pruned count.
    so we will test if random selection with the same threshold/level. thus prun percent isn't needed at the moment.
        prune_percentage = FLAGS.prun_percent
        if not FLAGS.random_quantize:
            print("Full optimization start")
            optimize_model(threshold=threshold, wavelet=wavelet_type,
                        level=decomposition_level)
            print("Optimization complete")
        else:
            # new_model = random_optimizer(threshold=threshold, wavelet=wavelet_type, level=decomposition_level, prun_percentage)
            print(f"testing for percent of {prune_percentage}")
    """

    prune_count, guid = optimize_model(threshold=threshold, wavelet=wavelet_type,
                                       level=decomposition_level)
    random_pruning(prune_count, guid)


if __name__ == '__main__':
    app.run(main)
