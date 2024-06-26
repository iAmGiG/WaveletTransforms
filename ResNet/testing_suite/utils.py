# utils.py
import os
import json
from safetensors import safe_open


def get_model_folders(base_path):
    """
    Recursively retrieves all subdirectory paths within the given base path.

    Args:
        base_path (str): The root directory to search for model folders.

    Returns:
        list: A list of paths to subdirectories containing model folders.
    """
    model_folders = []
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            model_folders.append(os.path.join(root, dir))
    return model_folders


def load_config(folder_path):
    """
    Loads the configuration file (config.json) from the specified folder.

    Args:
        folder_path (str): The path to the folder containing the configuration file.

    Returns:
        dict: The configuration parameters loaded from the JSON file.
    """
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def load_model(folder_path):
    """
    Loads the model from the safetensors file in the specified folder.

    Args:
        folder_path (str): The path to the folder containing the model.safetensors file.

    Returns:
        Any: The model loaded from the safetensors file.
    """
    model_path = os.path.join(folder_path, 'model.safetensors')
    model = safe_open(model_path)
    return model
