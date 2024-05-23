from datasets import load_dataset

# Load the ImageNet-1k dataset
dataset = load_dataset('imagenet-1k')
dataset.save_to_disk('imagenet-1k-dataset')
