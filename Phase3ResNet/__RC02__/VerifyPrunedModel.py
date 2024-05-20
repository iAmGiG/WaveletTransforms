import tensorflow as tf
import numpy as np

# Load the pruned model
pruned_model = tf.keras.models.load_model('pruned_model')

# Print model summary
pruned_model.summary()

# Inspect weights of a specific layer (e.g., the first convolutional layer)
layer = pruned_model.layers[0]
weights = layer.get_weights()
print(f"Weights of the first layer after pruning: {weights}")
