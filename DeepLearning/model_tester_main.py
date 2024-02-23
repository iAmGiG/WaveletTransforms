import time
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load or prepare your test dataset
(testX, testY), (_, _) = mnist.load_data()
testX = testX / 255.0  # Normalize
testY = to_categorical(testY)

# make this line abstract based on flags.
model_path = './DeepLearning/SavedStandardModels/mnist_model_DWT.h5'

# Load your model
model = load_model(model_path)


# Evaluate accuracy
loss, accuracy = model.evaluate(testX, testY)
print(f"Accuracy: {accuracy*100:.2f}%")

# Evaluate model size
model_size = os.path.getsize(model_path)
print(f"Model Size: {model_size / 1024:.2f} KB")

# Evaluate inference time
start_time = time.time()
predictions = model.predict(testX)
end_time = time.time()
print(f"Inference Time: {end_time - start_time:.4f} seconds")

# Evaluate sparsity
weights = model.get_weights()
zero_weights = np.sum([np.sum(w == 0) for w in weights])
total_weights = np.sum([w.size for w in weights])
sparsity = zero_weights / total_weights
print(f"Sparsity: {sparsity*100:.2f}%")
