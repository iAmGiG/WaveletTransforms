import numpy as np
import os
import datetime
import pywt
import tensorflow as tf
from absl import app, flags
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard


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
- Use gpu: Whether to use a GPU or not for TensorFlow operations. 
    When enabled and a compatible GPU is available, 
    training can be significantly faster compared to using just the CPU. 
    However, GPU support may not always be present or compatible with the TensorFlow version being used. 
    At the time of this project, 
    the TensorFlow GPU version (2.6.0) was behind the base CPU version (2.10.0) used in this code.
'''
flags.DEFINE_enum('wavelet', 'haar', ['haar', 'db1', 'db2', 'coif1', 'bior1.3', 'rbio1.3', 'sym2', 'mexh', 'morl'],
                  'Type of wavelet to use for DWT.')
flags.DEFINE_string('save_dir', './SavedDWTModels',
                    'Dir location to save trained models.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer(
    'epochs', '10', 'Number of epochs or iterations over the entire training dataset.')
flags.DEFINE_integer('level', '1', 'Decomposition level for the DWT.')
flags.DEFINE_float('threshold', '0.85',
                   'Threshold for thresholding the wavelet coefficients obtained from the DWT.')
flags.DEFINE_boolean(
    'use_gpu', True, 'Whether to use GPU or not for TensorFlow operations.')


def setup_gpu_configuration():
    """
    Configures TensorFlow to use GPU if available and enabled through flags.

    This function checks for the availability of GPU devices and, if found, 
    attempts to configure TensorFlow to use the GPU, specifically by setting
    memory growth to prevent TensorFlow from allocating the GPU's entire memory
    upfront. This is crucial for running multiple TensorFlow processes on the
    same GPU.

    Key considerations:
    - TensorFlow GPU support depends on the compatibility with specific CUDA and
      cuDNN versions. Ensure your TensorFlow version is compatible with the installed
      CUDA and cuDNN versions.
    - The memory growth setting allows gradual memory allocation, which is useful
      to avoid TensorFlow consuming all available GPU memory in environments shared
      with other applications.
    - In some cases, despite having compatible hardware and software, TensorFlow
      might not recognize the GPU due to issues with the installation or environment
      setup. Troubleshooting may involve verifying the CUDA and cuDNN installation,
      ensuring correct environment variables are set, and checking for conflicts
      with other installed Python packages.
    - Version Compatibility: As of the current version of this script, it is developed
      and tested with TensorFlow 2.x. It's crucial to match the TensorFlow version
      with compatible versions of h5py and numpy to avoid binary incompatibility
      errors, which can arise from mismatches in the expected sizes of data structures
      between compiled C extensions and Python objects.
    - Users should weigh the benefits of GPU acceleration against the potential
      need to adjust their environment. Changing CUDA/cuDNN versions or TensorFlow
      itself to gain GPU support might not be desirable if it risks disrupting
      other dependencies or project requirements.

    If GPU support is not enabled or no GPUs are found, TensorFlow operations
    will default to the CPU. Users can explicitly disable GPU support to force
    TensorFlow to use the CPU, which might be useful for debugging or when
    running on systems without a suitable GPU.
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


def get_save_dir():
    """
    Constructs and returns a directory path for saving models. The directory path reflects the model's configuration,
    incorporating the wavelet type, batch size, threshold value, epochs, and decomposition level into the directory structure.
    This organized approach facilitates easy navigation and identification of models based on their configuration.

    To ensure compatibility with filesystem conventions, especially on Windows, this function replaces '.' in the threshold
    value with '_', as filenames and directories on Windows cannot contain certain characters like the dot character in 
    contexts other than separating the filename from the extension.

    The constructed directory path follows the format:
    `save_dir/wavelet/batch_size/threshold_value/epochs/level`
    where:
    - `save_dir` is the base directory for saving models, specified by the `--save_dir` flag.
    - `wavelet` specifies the wavelet type used in the DWT process.
    - `batch_size` reflects the number of samples processed before the model is updated.
    - `threshold_value` is the threshold applied in the DWT process, with dots replaced by underscores for compatibility.
    - `epochs` represents the number of complete passes through the training dataset.
    - `level` indicates the decomposition level used in the DWT process.

    If the constructed directory does not exist, it is created with `os.makedirs(save_dir, exist_ok=True)`, ensuring
    that the model can be saved without manual directory creation.

    Returns:
        str: The constructed directory path where the model should be saved.

    Example:
        If the flags are set as follows:
        --wavelet 'haar', --batch_size 32, --threshold 0.1, --epochs 10, --level 1
        The returned save directory will be something like:
        'models/haar/32/0_1/10/1'

    Note:
        The use of `os.makedirs(..., exist_ok=True)` ensures that attempting to create an already existing directory
        won't raise an error, facilitating reusability of the function across different runs with the same or different
        configurations.
    """
    # Convert threshold to string and replace dots for filesystem compatibility
    threshold_str = str(FLAGS.threshold).replace('.', '_')
    # Construct the directory path based on flags
    save_dir = os.path.join(FLAGS.save_dir, FLAGS.wavelet,
                            str(FLAGS.batch_size), threshold_str,
                            str(FLAGS.epochs), str(FLAGS.level))
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def load_dataset():
    """
    Loads and preprocesses the MNIST dataset for use in training and evaluating deep learning models. The MNIST dataset
    comprises 28x28 grayscale images of handwritten digits (0 through 9) and is commonly used for benchmarking classification
    algorithms.

    - The conversion to `float32` is performed to ensure numerical consistency across different computing platforms and 
      optimize the training performance, especially on GPUs.
    - Normalization helps with model convergence by ensuring input features are on a similar scale.
    - One-hot encoding the labels is necessary for classification tasks to match the expected output structure of neural network models.

    The preprocessing steps applied are as follows:
    1. Normalization: The pixel values of the images, originally ranging from 0 to 255, are normalized to float values 
       between 0 and 1. This normalization is crucial for models to converge faster during training, as it ensures that 
       the input feature values are on a similar scale.
    2. One-hot encoding: The labels, which are integer values representing the digits, are converted into one-hot encoded 
       vectors. A one-hot vector is a vector where all bits are 0 except for one bit, which is set to 1 to indicate the 
       digit class. This conversion is necessary for classification tasks where the output is expected to be a probability 
       distribution across different classes.

    Returns:
        tuple: A tuple containing four elements:
               - trainX: numpy.ndarray of shape (60000, 28, 28), representing the training images.
               - trainY: numpy.ndarray of shape (60000, 10), representing the one-hot encoded labels for the training images.
               - testX: numpy.ndarray of shape (10000, 28, 28), representing the testing images.
               - testY: numpy.ndarray of shape (10000, 10), representing the one-hot encoded labels for the testing images.

    Example usage:
        trainX, trainY, testX, testY = load_dataset()
        # Now, trainX and testX contain normalized images, and trainY and testY contain one-hot encoded labels.

    Note:
        This preprocessing step is crucial for the effective training of deep learning models on the MNIST dataset. It aligns the data 
        format with common practices in machine learning and ensures compatibility with TensorFlow's data processing requirements. The 
        explicit conversion to `float32` and subsequent normalization does not make older models obsolete but ensures consistency and 
        performance for new training sessions.
        Normalizing the dataset and converting the labels into one-hot encoded vectors are common preprocessing steps in 
        machine learning and deep learning. These steps help improve model accuracy, learning efficiency, and convergence speed.
    """
    # Load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # Normalize images to have values between 0 and 1
    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0
    # Convert labels to one-hot encoded vectors
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def define_model():
    """
    Defines and compiles a simple Sequential neural network model for image classification tasks like MNIST digit recognition.

    The model architecture consists of the following layers:
    1. Flatten layer: Converts the input tensor (typically an image) into a 1D array, allowing the subsequent dense layers to operate on flattened feature vectors.
    2. Dense layer with ReLU activation: A fully connected layer with 128 units and ReLU (Rectified Linear Unit) activation, which introduces non-linearity to the model.
    3. Output Dense layer with softmax activation: A fully connected output layer with 10 units (one for each digit class in MNIST) and softmax activation, which produces a probability distribution over the classes.

    The model is compiled with the following settings:
    - Optimizer: Adam optimizer, which is a popular choice for its adaptive learning rate and momentum behavior.
    - Loss function: Categorical cross-entropy loss, suitable for multi-class classification problems.
    - Metrics: Accuracy, which measures the fraction of correctly classified samples.

    Args:
        input_shape (tuple): The expected shape of the input data. For MNIST, it should be (28, 28, 1) for grayscale images or (28, 28, 3) for RGB images.

    Returns:
        model (tensorflow.keras.models.Sequential): The compiled Sequential model.
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
    Apply the Discrete Wavelet Transform (DWT) to the input weights and reconstruct them after thresholding the wavelet coefficients.

    The DWT is a signal processing technique that decomposes a signal (in this case, the weight matrix) into a set of wavelet coefficients, representing different frequency components of the original signal. Thresholding these coefficients can help in denoising and sparsifying the weights, potentially improving the performance of the deep learning model.

    Args:
        weights (numpy.ndarray): A 2D array representing the weights of a neural network layer. These weights will be transformed using the DWT.
        wavelet (pywt.Wavelet): A PyWavelets wavelet object specifying the wavelet function to be used for the DWT. Different wavelet functions capture different characteristics of the input signal.
        level (int): The level of decomposition for the DWT. Higher levels result in a deeper decomposition, capturing more granular details but also increasing computational complexity.
        threshold_val (float): The threshold value for thresholding the wavelet coefficients. Coefficients with absolute values below this threshold will be set to zero. Thresholding helps in denoising and sparsifying the weights.

    Returns:
        numpy.ndarray: The reconstructed weights after applying the DWT, thresholding the wavelet coefficients, and reconstructing the weights using the thresholded coefficients.

    The function follows these steps:
    1. Compute the wavelet decomposition of the input weights using `pywt.wavedecn`. This yields a set of wavelet coefficients, including detail coefficients (represented as dictionaries) and approximation coefficients (represented as arrays).
    2. Threshold the resulting wavelet coefficients using `pywt.threshold`.
        - For detail coefficients (dictionaries), threshold each value in the dictionary.
        - For approximation coefficients (arrays), threshold the array directly.
    3. Reconstruct the weights from the thresholded wavelet coefficients using `pywt.waverecn`.
    4. Return the reconstructed weights.

    The thresholding operation helps in denoising and sparsifying the weights, potentially improving the performance of the deep learning model by removing insignificant or redundant components from the weights.
    """
    # Assuming weights are a 2D matrix
    coeffs = pywt.wavedecn(weights, wavelet, level)
    coeffs_thresh = []
    for coeff in coeffs:
        if isinstance(coeff, dict):
            # Handle detail coefficients (dictionary)
            thresholded_coeff = {key: pywt.threshold(
                coeff[key], threshold_val) for key in coeff.keys()}
            coeffs_thresh.append(thresholded_coeff)
        else:
            # Handle approximation coefficients (single array)
            coeffs_thresh.append(pywt.threshold(coeff, threshold_val))
    weights_reconstructed = pywt.waverecn(coeffs_thresh, wavelet)
    return weights_reconstructed


def train_model_with_dwt(trainX, trainY, testX, testY, wavelet, level, threshold_val):
    """
    Trains a neural network model on the MNIST dataset with weights modified by the Discrete Wavelet Transform (DWT).

    This function applies the `apply_dwt_and_reconstruct` function to the weights of the dense layers in the model
    before training. This step transforms the weights using the specified wavelet, decomposition level, and
    threshold value, potentially improving the model's performance by denoising and sparsifying the weights.

    Args:
        trainX (numpy.ndarray): Training input data.
        trainY (numpy.ndarray): Training labels or target data.
        testX (numpy.ndarray): Testing input data.
        testY (numpy.ndarray): Testing labels or target data.
        wavelet (pywt.Wavelet): A PyWavelets wavelet object specifying the wavelet function to be used for the DWT.
        level (int): The level of decomposition for the DWT.
        threshold_val (float): The threshold value for thresholding the wavelet coefficients.

    Returns:
        model (tensorflow.keras.models.Sequential): The trained Sequential model.

    The function follows these steps:
    1. Define the neural network model using the `define_model` function.
    2. For each dense layer in the model:
        a. Get the weights and biases of the layer.
        b. Apply the `apply_dwt_and_reconstruct` function to the weights, using the specified wavelet, level, and threshold_val.
        c. Set the modified weights and original biases back to the layer.
    3. Train the model on the provided training data (trainX, trainY) using the fit method.
        - The number of epochs and batch size are specified by the FLAGS.epochs and FLAGS.batch_size values, respectively.
        - The testing data (testX, testY) is used for validation during training.
    4. Return the trained model.

    Note: The commented lines for setting up TensorBoard logging and callbacks are provided for reference,
    but they are not included in the current implementation.
    """
    model = define_model()

    # Apply DWT to model weights
    for i, layer in enumerate(model.layers):
        if 'dense' in layer.name:
            weights, biases = layer.get_weights()
            transformed_weights = apply_dwt_and_reconstruct(
                weights, wavelet, level, threshold_val)
            model.layers[i].set_weights([transformed_weights, biases])

    model.fit(trainX, trainY,
              epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size,
              validation_data=(testX, testY)
              )
    # Optional: Uncomment to enable TensorBoard callbacks for visualization
    # callbacks=[tensorboard_callback]
    return model


def main(argv):
    """
    Executes the main workflow of the DWT-based MNIST digit classification model training. 
    This includes setting up GPU configuration (if enabled), loading the MNIST dataset, 
    training a neural network model with Discrete Wavelet Transform (DWT) applied to its weights, 
    and saving the trained model to disk with a filename reflecting its configuration and training parameters.

    This function handles the following tasks:
    1. Parse command-line arguments using the `absl` library.
    2. Set up GPU configuration for TensorFlow operations. [Optional]
    3. Load the MNIST dataset using the `load_dataset` function.
    4. Train a neural network model on the MNIST dataset with weights modified by the Discrete Wavelet Transform (DWT)
       using the `train_model_with_dwt` function. The wavelet type, decomposition level, and threshold value are
       specified by the command-line flags.
    5. Save the trained model to a file with a filename that reflects the model configuration and training parameters.
    6. Print the model summary to the console.

    The script expects the following command-line flags:
    - wavelet: The type of wavelet to use for the DWT.
    - level: The level of decomposition for the DWT.
    - threshold: The threshold value for thresholding the wavelet coefficients.
    - save_dir: The directory to save the trained model.
    - batch_size: The batch size for training.
    - epochs: The number of epochs or iterations over the entire training dataset.
    - use_gpu: Whether to use GPU or not for TensorFlow operations.

    Args:
        argv: List of command-line arguments passed to the script. 
              This function expects no command-line arguments besides the flags defined globally.

    This function showcases how to incorporate DWT into a deep learning model's training process, potentially improving 
    model performance by leveraging the sparsity and denoising capabilities of wavelet transforms.

    These flags can be specified when running the script, e.g.:
    python script.py --wavelet=haar --level=1 --threshold=0.05 --save_dir=./models --batch_size=32 --epochs=10 --use_gpu=True

    Note: The commented lines related to handling command-line arguments are provided for reference,
    but they are not included in the current implementation.
    """
    # Parse command-line flags
    app.parse_flags_with_usage(argv)

    # Configure TensorFlow to use GPU or CPU based on the provided flags
    setup_gpu_configuration()

    # Retrieve training parameters from flags
    wavelet = FLAGS.wavelet
    level = FLAGS.level
    threshold_val = FLAGS.threshold

    # Load and preprocess the MNIST dataset
    trainX, trainY, testX, testY = load_dataset()

    # Train the model with DWT applied to weights
    model = train_model_with_dwt(
        trainX, trainY, testX, testY, wavelet=wavelet, level=level, threshold_val=threshold_val)

    # Construct a filename for the model incorporating training parameters
    threshold_str = str(FLAGS.threshold).replace('.', '_')
    model_filename = f"mnist_model_dwt_{FLAGS.wavelet}_{FLAGS.level}_{threshold_str}_{datetime.datetime.now().strftime('%m-%d')}.h5"

    # Save the trained model
    full_path = os.path.join(get_save_dir(), model_filename)
    model.save(full_path)
    print(f"Model saved to: {full_path}")

    # Optionally, display the model's summary
    model.summary()


if __name__ == '__main__':
    app.run(main)
