import tarfile
import os

def extract_tar_file(file_path, extract_path):
    """
    Extracts a tar.gz file to the specified directory.

    Args:
        file_path (str): The path to the tar.gz file.
        extract_path (str): The directory to extract the contents to.
    """
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

# Extract test_images.tar.gz
extract_tar_file('./imagenet-1k/data/test_images.tar.gz', './imagenet-1k/data/test_images')
