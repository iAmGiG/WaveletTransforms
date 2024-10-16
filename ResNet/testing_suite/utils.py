# utils.py
import os
import json
import logging
import torch
import traceback
from safetensors.torch import load_file as safe_load_file
from transformers import ResNetForImageClassification, ResNetConfig


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
        model_path (str): Path to the directory containing model.safetensor and config.json.

    Returns:
        model (torch.nn.Module): Loaded pre-trained model.
    """
    try:
        if os.path.isdir(model_path):
            # Print the contents of the directory
            print(f"Contents of the directory {model_path}:")
            for item in os.listdir(model_path):
                print(f" - {item}")

            model_file = os.path.join(model_path, 'model.safetensors')
            config_file = os.path.join(model_path, 'config.json')

            if not os.path.exists(model_file):
                logging.warning(
                    f"Model file not found in directory: {model_path}. Checking subdirectories.")

                # Check subdirectories
                for sub_dir in os.listdir(model_path):
                    sub_dir_path = os.path.join(model_path, sub_dir)
                    if os.path.isdir(sub_dir_path):
                        model_file = os.path.join(
                            sub_dir_path, 'model.safetensors')
                        config_file = os.path.join(sub_dir_path, 'config.json')
                        if os.path.exists(model_file) and os.path.exists(config_file):
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            model_config = ResNetConfig.from_dict(config)
                            model = ResNetForImageClassification(model_config)
                            state_dict = safe_load_file(model_file)
                            model.load_state_dict(state_dict, strict=False)
                            logging.info(
                                f"Model loaded successfully from {sub_dir_path}")
                            return model, config

                logging.error(
                    f"Model file not found in directory or its subdirectories: {model_path}")
                return None, None

            with open(config_file, 'r') as f:
                config = json.load(f)
            model_config = ResNetConfig.from_dict(config)
            model = ResNetForImageClassification(model_config)
            state_dict = safe_load_file(model_file)
            model.load_state_dict(state_dict, strict=False)

        else:
            # Assume model is a single file
            state_dict = safe_load_file(model_path)
            model_config = ResNetConfig()
            model = ResNetForImageClassification(model_config)
            model.load_state_dict(state_dict, strict=False)
            config = None

        logging.info(f"Model loaded successfully from {model_path}")
        return model, config

    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None


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


def load_preprocessed_batches(data_path):
    batches = []
    labels = []
    files = sorted(os.listdir(data_path))
    # Print first 10 files for brevity
    print(f"Files in data path: {files[:10]}... (total {len(files)} files)")

    for file in files:
        file_path = os.path.join(data_path, file)
        data = torch.load(file_path)

        if file.startswith("batch_") and file.endswith(".pt"):
            print(f"Loading batch file: {file_path}")
            batches.append(data)
        elif file.startswith("labels_") and file.endswith(".pt"):
            print(f"Loading label file: {file_path}")
            labels.append(data)
        else:
            print(f"Skipping file: {file}")

    print(f"Loaded {len(batches)} batches and {len(labels)} label sets")

    if len(batches) == 0:
        raise ValueError(
            "No batches loaded. Check the data path and file format.")

    if len(labels) == 0:
        print("Warning: No separate label files found. Assuming labels are included in the batches.")

    return batches, labels
