from datasets import load_dataset
import random

# Load the ImageNet-1k dataset
dataset = load_dataset('imagenet-1k')

# Create a small subset of the test set
small_test_set = dataset['test'].shuffle(seed=42).select(
    range(100))  # Select only 100 samples

# Save the small test set to disk
small_test_set.save_to_disk('imagenet-1k-small-test')
