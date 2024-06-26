# download_data.py
import os
from datasets import load_dataset

def download_and_cache_imagenet(cache_dir='./cache'):
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset('imagenet-1k', split='validation', cache_dir=cache_dir)
    return dataset

if __name__ == '__main__':
    download_and_cache_imagenet()
