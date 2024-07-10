# utils.py
import os
import json
import logging
from transformers import AutoModelForImageClassification, AutoConfig

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

def load_model(model_path):
    """
    Load a pre-trained model from the given path.

    Args:
        model_path (str): Path to the model directory.

    Returns:
        model (torch.nn.Module): Loaded pre-trained model.
    """
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, 'config.json')
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForImageClassification.from_pretrained(model_path, config=config)
    else:
        raise ValueError(f"Provided model path {model_path} is not a valid directory.")
    
    print("Pre-trained model loaded successfully.")
    return model

def setup_logging(log_dir):
    """
    Sets up logging for the application.

    Args:
        log_dir (str): The directory where log files will be stored.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )