from datasets import load_from_disk
import os


def inspect_dataset():
    dataset_dir = os.path.abspath('imagenet-1k-dataset')
    dataset = load_from_disk(dataset_dir)
    # Print the first entry to inspect its structure
    print(dataset['validation'][0])
    # Print the second entry to inspect its structure
    print(dataset['validation'][1])
    # Print the third entry to inspect its structure
    print(dataset['validation'][2])


inspect_dataset()
