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
def decompose_and_pruen(weights_fromModel, threshold,
                                    wavelet, level):
    """
    For each layer's weight matrix, 
    perform DWT with the specified wavelet type and decomposition level. 
    Apply threshold-based pruning, possibly with an added step to prune a 
    specific percentage of weights based on their magnitude or significance.
    """
    

# DWT random Weights

# IDWT process
def reconstrucut_weights(approx, coeffs, wavelet):


# Main


def main(argv):
    """
    runs the show
    """
    print(FLAGS.model_dir)
    current_model = get_model(FLAGS.model_dir)
    #
    # extract the weights from the model
    #
    # Parameters
    wavelet_type = FLAGS.wavelet  # Can be 'db1', 'sym2', etc.
    decomposition_level = FLAGS.level  # Level of wavelet decomposition
    threshold = FLAGS.threshold  # Pruning threshold
    prune_percentage = FLAGS.prun_percent  # Percentage of weights to prune selectively
    if not FLAGS.random_quantize:
        new_model = decompose_and_pruen(weights_fromModel, DWT_threshold,
                                    DWT_wavelet, DWT_level)
    else:
        new_model = DWT_random_weights(weights_fromModel, dwt_threshold, dwt_wavelet,
                                       DWT_level, prun_percentage)
    save_model = reconstrucut_weights(new_model)

if __name__ == '__main__':
    app.run(main)
