from absl import app, flags
import tensorflow as tf
import numpy as np
import os
import datetime
import pywt

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
flags.DEFINE_float('threshold', '0.85',
                   'Threshold for thresholding the wavelet coefficients obtained from the DWT.')

flags.DEFINE_string('quant_level', 'binary',
                    'Level of quantization (binary, ternary, etc.)')
flags.DEFINE_float('prun_percent', 50, 'Percentage of weights to quantize')
flags.DEFINE_boolean('random_quantize', False,
                     'Enable random weight quantization')
flags.mark_flag_as_required('model_path')


# Load model
def get_model(model_path):
    """
    gets the model from the defined location.

    """
    # Load the pre-trained model
    model = tf.keras.models.load_model('model_path')
    return model

# Save model


def save_model(model, model_path):
    """    
    we'll have to adjust for the unique pathing name, 
    and subdirectory storage.
    these models will be strored within a sub domain 
    along side the original model 
    and we need to store the details 
    of the dwt into the file name
    """

    model.save()
    return str(save_dir)

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
    model = get_model(FLAGS.model_dir)
    for layer in model.layers:
        if layer.weights:
            weights = layer.get_weights()[0]
            coeffs, original_shape = apply_dwt(
                weights=weights, wavelet=wavelet, level=level)
            pruned_coeffs = prune_coeffs(coeffs=coeffs, threshold=threshold)
            new_weights = reconstruct_weights(
                pruned_coeffs=pruned_coeffs, wavelet=wavelet, original_shape=original_shape)
            layer.set_weights([new_weights] + layer.get_weights()[1:])


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
    Prunes the wavelet coefficients based on the given threshold.

    Args:
        coeffs (list): A list of wavelet coefficients.
        threshold (float): The threshold value for pruning coefficients. Default is 0.85.

    Returns:
        list: A list of pruned wavelet coefficients.
    """
    pruned_coeffs = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            # For each level of decomposition, apply thresholding
            pruned_coeffs.append(
                tuple(np.where(np.abs(c) < threshold, 0, c) for c in coeff))
        else:
            # For the approximation coefficients
            pruned_coeffs.append(np.where(np.abs(coeff) < threshold, 0, coeff))
    return pruned_coeffs

# DWT random Weights

def optimize_model(threshold, wavelet, level):
    return "TESTING ONLY"

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
    print(FLAGS.model_dir)

    # Parameters
    wavelet_type = FLAGS.wavelet  # Can be 'db1', 'sym2', etc.
    decomposition_level = FLAGS.level  # Level of wavelet decomposition
    threshold = FLAGS.threshold  # Pruning threshold
    prun_percentage = FLAGS.prun_percentage
    # Percentage of weights to prune selectively
    prune_percentage = FLAGS.prun_percent
    if not FLAGS.random_quantize:
        new_model = optimize_model(threshold=threshold, wavelet=wavelet_type, level=decomposition_level)
    else:
        new_model = random_optimizer(threshold=threshold, wavelet=wavelet_type, level=decomposition_level, prun_percentage)


if __name__ == '__main__':
    app.run(main)
