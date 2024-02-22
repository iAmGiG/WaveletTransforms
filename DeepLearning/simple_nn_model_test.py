import os
from tensorflow.keras.models import load_model

# Path to the saved model
model_path = './DeepLearning/SavedStandardModels/mnist_model_02-20_17-44.h5'

# Load the model
model = load_model(model_path)
