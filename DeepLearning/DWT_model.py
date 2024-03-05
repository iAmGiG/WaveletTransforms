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
flags.DEFINE_string('save_dir', './DeepLearning/SavedDWTModels',
                    'Dir to save trained models.')
'''
where you call the model.fit() method under the train with dwt model.
Adjusting the batch size affects the number of samples that are processed before the model's internal parameters are updated.
Raising Batch Size: Increasing the batch size will generally 
    lead to faster epoch times since there are fewer updates to be made.
Lowering Batch Size: Decreasing the batch size often 
    leads to longer training times since the model's parameters are updated more frequently.
'''
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('epochs', '10', 'Number of episodes/epochs')
# need flag for the adjustment of the model threshold.


def get_save_dir():
    # I want the save dir to use the flag for threshold change on the weights, for the record.
    current_time = datetime.datetime.now()
    # date_time = current_time.strftime("%m-%d_%H-%M")
    # wavelet_type = FLAGS.wavelet
    save_dir = os.path.join(FLAGS.save_dir, f"mnist_model_dwt_{current_time.strftime('%m-%d')}_{FLAGS.wavelet}_{FLAGS.batch_size}_{FLAGS.epochs}.h5")
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


def train_model_with_dwt():
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()

    # Apply DWT to model weights
    for i, layer in enumerate(model.layers):
        if 'dense' in layer.name:
            weights, biases = layer.get_weights()
            transformed_weights = apply_dwt_and_reconstruct(weights)
            model.layers[i].set_weights([transformed_weights, biases])

    model.fit(trainX, trainY, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
              validation_data=(testX, testY))
    # model_filename = f"mnist_model_dwt_{FLAGS.wavelet}_{date_time}.h5"
    # full_path = os.path.join(get_save_dir(), model_filename)
    # model.save(full_path)
    return model


def main(argv):
    if len(argv) > 4:
        raise app.UsageError('Too many command-line arguments.')
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()
    
    # Apply DWT and train
    model = train_model_with_dwt(trainX, trainY, testX, testY)  # Pass datasets as arguments if needed
    
    # Save model with flags in the name for tracking
    date_time = datetime.datetime.now().strftime("%m-%d")
    model_filename = f"mnist_model_dwt_{FLAGS.wavelet}_{date_time}.h5"
    full_path = os.path.join(FLAGS.save_dir, model_filename)
    model.save(full_path)
    model.summary()


if __name__ == '__main__':
    app.run(main)
