import os
import tensorflow as tf
from transformers import AutoFeatureExtractor

# Set the environment variable to allow duplicate OpenMP runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the pruned model
pruned_model = tf.keras.models.load_model('pruned_model')

# Attempt to load the ImageNet-1k dataset from Hugging Face
try:
    from datasets import load_dataset
    dataset = load_dataset("imagenet-1k", split="test")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Load an alternative dataset or handle the error appropriately
    dataset = None  # Replace this with alternative dataset loading code

# Initialize the feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")

# Define the preprocess function


def preprocess(example):
    image = example["image"]
    inputs = feature_extractor(images=image, return_tensors="tf")
    return inputs['pixel_values'][0], example["label"]


if dataset:
    # Preprocess and batch the test dataset
    test_data = dataset.map(preprocess)
    test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

    # Evaluate the pruned model
    loss, accuracy = pruned_model.evaluate(test_data)
    print(f"Pruned model accuracy: {accuracy * 100:.2f}%")
else:
    print("Dataset could not be loaded. Evaluation aborted.")
