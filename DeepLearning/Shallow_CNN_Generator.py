from absl import app, flags
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import os
import datetime as datetime
import tensorflow as tf
import uuid
import json

#TODO log
"""
TODO: get model changes log.json incoporated into shallow cnn gen
TODO: implement absl flags ~~~See Train Model~~~
TODO: 
"""

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer(
    'epochs', '10', 'Number of epochs or iterations over the entire training dataset.')

save_dir = './DeepLearning/SavedStandardModels'
os.makedirs(save_dir, exist_ok=True)
now = datetime.datetime.now()
date_time = now.strftime("%m-%d_%H-%M")


def generate_guid():
    """
    make a new GUID
    """
    return uuid.uuid4().hex


def save_model(model, original_model_path, guid):
    """
    save the model
    """
    # Determine the directory of the original model
    directory = os.path.dirname(original_model_path)
    # Create a new directory name with the GUID
    new_directory_name = f"{directory}/shallow_cnn_{guid}"
    os.makedirs(new_directory_name, exist_ok=True)
    # Save the model in the new directory
    model_save_path = f"{new_directory_name}/model.h5"
    model.save(model_save_path)
    # return model_save_path


def load_dataset():
    """
    Loads the mnist dataset
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
    a shallow cnn model
    TODO: fix "if needed" the model scope issue.
    """
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model():
    """
    TODO: might rename things?
    """
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()
    model.fit(trainX, trainY, epochs=10, batch_size=32,
              validation_data=(testX, testY))
    model.save(full_path)
    return model


def main(argv):
    """
    does a thing and runs the show
    """
    # model_filename = f"mnist_model_{date_time}.h5"
    # full_path = os.path.join(save_dir, model_filename)
    # model = train_model()
    # model.summary()
    # print(f"Model saved as {model_filename}")
    # Load and preprocess the MNIST dataset
    batch_size = FLAGS.batch_size
    
    trainX, trainY, testX, testY = load_dataset()


if __name__ == '__main__':
    app.run(main)