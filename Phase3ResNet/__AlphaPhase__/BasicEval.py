import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from transformers import AutoFeatureExtractor

# Set the environment variable to allow duplicate OpenMP runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the pruned model
pruned_model = tf.keras.models.load_model('pruned_model')

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.0  # Normalize the test images
y_test = tf.keras.utils.to_categorical(y_test, 10)  # One-hot encode the labels

# Initialize the feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")

# Preprocess the dataset


def preprocess(image):
    inputs = feature_extractor(images=image, return_tensors="tf")
    return inputs['pixel_values'][0]


# Preprocess and batch the test dataset
x_test_processed = tf.data.Dataset.from_tensor_slices((x_test, y_test))
x_test_processed = x_test_processed.map(lambda x, y: (preprocess(x), y))
test_data = x_test_processed.batch(32).prefetch(tf.data.AUTOTUNE)

# Evaluate the pruned model
loss, accuracy = pruned_model.evaluate(test_data)
print(f"Pruned model accuracy: {accuracy * 100:.2f}%")
