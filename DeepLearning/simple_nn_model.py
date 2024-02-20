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


def train_model():
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()
    model.fit(trainX, trainY, epochs=10, batch_size=32,
              validation_data=(testX, testY))
    return model


if __name__ == "__main__":
    model = train_model()
    model.summary()
