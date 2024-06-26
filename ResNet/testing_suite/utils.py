# utils.py
import os
import json
from safetensors import safe_open

def get_model_folders(base_path):
    model_folders = []
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            model_folders.append(os.path.join(root, dir))
    return model_folders

def load_config(folder_path):
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def load_model(folder_path):
    model_path = os.path.join(folder_path, 'model.safetensors')
    model = safe_open(model_path)
    return model
