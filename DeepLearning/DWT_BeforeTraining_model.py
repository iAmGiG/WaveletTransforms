import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

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
    
    model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY))
    model.save(f'./DeepLearning/SavedStandardModels/mnist_model_dwt.h5')
    return model

if __name__ == "__main__":
    model = train_model_with_dwt()
    model.summary()
