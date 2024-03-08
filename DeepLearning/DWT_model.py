import numpy as np
import os
import datetime
import pywt
import tensorflow as tf
from absl import app
from absl import flags
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

FLAGS = flags.FLAGS
flags.DEFINE_enum('wavelet', 'haar', ['haar', 'db1', 'db2', 'coif1', 'bior1.3', 'rbio1.3', 'sym2', 'mexh', 'morl'],
                  'Type of wavelet to use for DWT.')
flags.DEFINE_string('save_dir', './SavedDWTModels',
                    'Dir to save trained models.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('epochs', '10', 'Number of episodes/epochs')
flags.DEFINE_integer('level', '1', 'Deeper decompreosition use a ')
flags.DEFINE_float('threshold', '0.1',
                   'Threshold of the appllied dwt weight. 0 lower bounds, max "absolute avlue of coefficients"')


def get_save_dir():
    """
    Constructs and returns a directory path for saving models, 
    incorporating wavelet type, batch size, threshold value, epochs, and decomposition level into the directory structure. 
    It also replaces '.' in the threshold value with '_' to avoid issues on Windows.
    """
    # current_time = datetime.datetime.now()
    # Construct directory path based on flags
    threshold_str = str(FLAGS.threshold).replace('.', '_')
    save_dir = os.path.join(FLAGS.save_dir, FLAGS.wavelet,
                            str(FLAGS.batch_size), threshold_str,
                            str(FLAGS.epochs), str(FLAGS.level))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def load_dataset():
    """
    Loads and preprocesses the MNIST dataset. 
    This includes normalizing the image data 
    and converting the labels to one-hot encoded vectors.
    """
    # Load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # Normalize images
    trainX = trainX / 255.0
    testX = testX / 255.0
    # One-hot encode targets
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def define_model():
    """
    Defines and compiles a simple Sequential neural network model suitable for MNIST digit classification. 
    The model consists of a flatten layer, a dense layer with ReLU activation, 
    and an output dense layer with softmax activation.
    """
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def apply_dwt_and_reconstruct(weights, wavelet, level, threshold_val):
    """
    Apply discrete wavelet transform (DWT) and reconstruct weights with thresholding.

    Args:
        weights (numpy.ndarray): A 2D array representing the weights to be transformed.
        wavelet (pywt.Wavelet): The wavelet to be used for the DWT.
        level (int): The level of decomposition for the DWT.
        threshold_val (float): The threshold value for thresholding the wavelet coefficients.

    Returns:
        numpy.ndarray: The reconstructed weights after applying DWT and thresholding.

    This function performs the following steps:
    1. Compute the wavelet decomposition of the input weights using `pywt.wavedecn`.
    2. Threshold the resulting wavelet coefficients using `pywt.threshold`.
       - For detail coefficients (represented as dictionaries), threshold each value in the dictionary.
       - For approximation coefficients (represented as arrays), threshold the array directly.
    3. Reconstruct the weights from the thresholded wavelet coefficients using `pywt.waverecn`.
    4. Return the reconstructed weights.

    The thresholding operation helps in denoising and sparsifying the weights, potentially improving
    the performance of the deep learning model.
    """
    # Assuming weights are a 2D matrix
    coeffs = pywt.wavedecn(weights, wavelet, level)
    coeffs_thresh = []
    for coeff in coeffs:
        if isinstance(coeff, dict):
            # Handle detail coefficients (dictionary)
            thresholded_coeff = {key: pywt.threshold(coeff[key], threshold_val) for key in coeff.keys()}
            coeffs_thresh.append(thresholded_coeff)
        else:
            # Handle approximation coefficients (single array)
            coeffs_thresh.append(pywt.threshold(coeff, threshold_val))
    weights_reconstructed = pywt.waverecn(coeffs_thresh, wavelet)
    return weights_reconstructed

def train_model_with_dwt(trainX, trainY, testX, testY, wavelet, level, threshold_val):
    """
    Trains the defined model on the MNIST dataset with weights modified by DWT. 
    It applies the apply_dwt_and_reconstruct function to the model's dense layer weights before training.
    """
    model = define_model()

    # Apply DWT to model weights
    for i, layer in enumerate(model.layers):
        if 'dense' in layer.name:
            weights, biases = layer.get_weights()
            transformed_weights = apply_dwt_and_reconstruct(
                weights, wavelet, level, threshold_val)
            model.layers[i].set_weights([transformed_weights, biases])

    model.fit(trainX, trainY, epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size,
              validation_data=(testX, testY))
    return model


def main(argv):
    """
    The main entry point for the script, 
    handling the workflow of loading data, applying DWT to model weights, training the model, 
    and saving the trained model with a filename that reflects the model configuration and training parameters.
    """
    # if len(argv) > 1:
    #     raise app.UsageError('Expected no command-line arguments, '
    #                         'got: {}'.format(argv))
    app.parse_flags_with_usage(argv)

    wavelet = FLAGS.wavelet
    level = FLAGS.level
    threshold_val = FLAGS.threshold

    trainX, trainY, testX, testY = load_dataset()
    # model = define_model()
    # Apply DWT and train
    # Pass datasets as arguments if needed
    model = train_model_with_dwt(
        trainX, trainY, testX, testY, wavelet=wavelet, level=level, threshold_val=threshold_val)

    # Save model with flags in the name for tracking
    threshold_str = str(FLAGS.threshold).replace('.', '_')
    model_filename = f"mnist_model_dwt_{FLAGS.wavelet}_{FLAGS.level}_{threshold_str}_{datetime.datetime.now().strftime('%m-%d')}.h5"
    full_path = os.path.join(get_save_dir(), model_filename)
    model.save(full_path)
    print(f"Model saved to: {full_path}")
    model.summary()


if __name__ == '__main__':
    app.run(main)
