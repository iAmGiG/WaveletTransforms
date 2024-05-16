import os
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Path to the saved model
model_path = './DeepLearning/SavedStandardModels/mnist_model_02-20_17-44.h5'

if __name__ == "__main__":
    # Load the model
    model = load_model(model_path)

    # Load the dataset
    (_, _), (testX, testY) = mnist.load_data()

    # Normalize images
    testX = testX / 255.0

    # One-hot encode targets
    testY = to_categorical(testY)

    # Evaluate the model
    loss, accuracy = model.evaluate(testX, testY, verbose=0)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Predict the values from the test dataset
    Y_pred = model.predict(testX)
    # Convert predictions classes to one hot vectors 
    Y_pred_classes = np.argmax(Y_pred, axis = 1) 
    # Convert test observations to one hot vectors
    Y_true = np.argmax(testY, axis = 1) 

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    print("Confusion Matrix:")
    print(confusion_mtx)

    # Show classification report
    print("Classification Report:")
    print(classification_report(Y_true, Y_pred_classes))
