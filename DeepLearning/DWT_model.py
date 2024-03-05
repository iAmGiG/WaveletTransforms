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
flags.DEFINE_string('wavelet', 'haar', 'Wavelety type used for DWT')
flags.DEFINE_string('save_dir', './SavedDWTModels',
                    'Dir to save trained models.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('epochs', '10', 'Number of episodes/epochs')


def get_save_dir():
    current_time = datetime.datetime.now()
    # Construct directory path based on flags
    save_dir = os.path.join(FLAGS.save_dir, current_time.strftime(
        '%Y-%m-%d'), FLAGS.wavelet, str(FLAGS.batch_size), str(FLAGS.epochs))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def load_dataset():
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
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def apply_dwt_and_reconstruct(weights, wavelet='haar'):
    # Assuming weights are a 2D matrix
    coeffs = pywt.dwt2(weights, wavelet)
    cA, (cH, cV, cD) = coeffs
    # Use only approximation coefficients for reconstruction
    reconstructed = pywt.idwt2((cA, (None, None, None)), wavelet)
    # Ensure the reconstructed shape matches the original shape
    reconstructed = np.resize(reconstructed, weights.shape)
    return reconstructed


def train_model_with_dwt(trainX, trainY, testX, testY):
    model = define_model()

    # Apply DWT to model weights
    for i, layer in enumerate(model.layers):
        if 'dense' in layer.name:
            weights, biases = layer.get_weights()
            transformed_weights = apply_dwt_and_reconstruct(weights)
            model.layers[i].set_weights([transformed_weights, biases])

    model.fit(trainX, trainY, epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size,
              validation_data=(testX, testY))
    return model


def main(argv):
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()

    # Apply DWT and train
    # Pass datasets as arguments if needed
    model = train_model_with_dwt(trainX, trainY, testX, testY)

    # Save model with flags in the name for tracking
    model_filename = f"mnist_model_dwt_{FLAGS.wavelet}_{datetime.datetime.now().strftime('%m-%d')}.h5"
    full_path = os.path.join(get_save_dir(), model_filename)
    model.save(full_path)
    print(f"Model saved to: {full_path}")
    model.summary()


if __name__ == '__main__':
    app.run(main)
