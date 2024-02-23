import os
import datetime as datetime
import pywt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

save_dir = './DeepLearning/SavedStandardModels'
os.makedirs(save_dir, exist_ok=True)
now = datetime.datetime.now()
date_time = now.strftime("%m-%d_%H-%M")


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


def train_model():
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()
    model.fit(trainX, trainY, epochs=10, batch_size=32,
              validation_data=(testX, testY))
    model.save(model_filename)
    return model

def apply_dwt(weights, wavelet='haar'):
    # Example function to apply DWT to weights
    coeffs = pywt.dwt2(weights, wavelet)
    # Flatten or process coeffs to use in the model
    # This is simplified; actual implementation may vary
    return coeffs

def train_model_with_dwt():
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()
    
    # Apply DWT to model weights
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            weights, biases = layer.get_weights()
            transformed_weights = apply_dwt(weights)  # Simplified
            layer.set_weights([transformed_weights, biases])
    
    # Continue with training as usual
    model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY))
    model.save(os.path.join(save_dir, f"DWT_{model_filename}"))  # Save with DWT prefix for clarity
    return model


if __name__ == "__main__":
    model_filename = f"mnist_DWT_{date_time}.h5"
    full_path = os.path.join(save_dir, model_filename)
    model = train_model()
    model.summary()
    print(f"Model saved as {model_filename}")
