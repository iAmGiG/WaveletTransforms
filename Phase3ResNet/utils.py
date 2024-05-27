import os
import csv
from transformers import AutoModelForImageClassification, AutoConfig
import torch


def load_model(model_path, config_path):
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForImageClassification.from_pretrained(
        model_path, config=config)
    print("Pre-trained model loaded successfully.")
    return model


def save_model(model, output_path):
    """
    Saves the model
    """
    output_path = os.path.normpath(os.path.join(os.getcwd(), output_path))
    model.save_pretrained(output_path)
    print(f"Model saved successfully at {output_path}")


def setup_csv_writer(file_path, mode='w'):
    """
    Set up a CSV writer.

    Args:
        file_path (str): Path to the CSV file.
        mode (str): File mode, 'w' for write (default) or 'a' for append.

    Returns:
        tuple: CSV writer and file object.
    """
    try:
        file_exists = os.path.isfile(file_path)
        file = open(file_path, mode=mode, newline='')
        fieldnames = [
            'GUID', 'Wavelet', 'Level', 'Threshold', 'DWT Phase',
            'Original Parameter Count', 'Non-zero Params', 'Total Pruned Count', 'Layer Name'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if mode == 'w' or (mode == 'a' and not file_exists):
            writer.writeheader()
        return writer, file
    except Exception as e:
        print(f"Failed to set up CSV writer: {e}")
        raise


def log_pruning_details(csv_writer, guid, wavelet, level, threshold, phase, original_param_count, non_zero_params, total_pruned_count, layer_name):
    """
    Log details of the pruning process to a CSV file.

    Args:
        csv_writer (csv.DictWriter): CSV writer object for logging.
        guid (str): Unique identifier for the pruning session.
        wavelet (str): Wavelet type used.
        level (int): Level of decomposition.
        threshold (float): Threshold value for pruning.
        phase (str): Pruning phase ('selective' or 'random').
        original_param_count (int): Original number of parameters in the layer.
        non_zero_params (int): Number of non-zero parameters after pruning.
        total_pruned_count (int): Total number of pruned weights.
        layer_name (str): Name of the layer being pruned.
    """
    if not csv_writer or not layer_name:
        return

    try:
        row = {
            'GUID': guid,
            'Wavelet': wavelet,
            'Level': level,
            'Threshold': threshold,
            'DWT Phase': phase,
            'Original Parameter Count': original_param_count,
            'Non-zero Params': non_zero_params,
            'Total Pruned Count': total_pruned_count,
            'Layer Name': layer_name
        }
        csv_writer.writerow(row)
    except Exception as e:
        print(f"Failed to log pruning details for layer {layer_name}: {e}")


def append_to_experiment_log(file_path, guid, wavelet, level, threshold, phase, total_pruned_count, total_non_zero_params, model_path):
    """
    Append details of the pruning experiment to the experiment log CSV file.

    Args:
        file_path (str): Path to the experiment log CSV file.
        guid (str): Unique identifier for the pruning session.
        wavelet (str): Wavelet type used.
        level (int): Level of decomposition.
        threshold (float): Threshold value for pruning.
        phase (str): Pruning phase ('selective' or 'random').
        total_pruned_count (int): Total number of pruned weights.
        total_non_zero_params (int): Total number of non-zero parameters after pruning.
        model_path (str): Path to the saved pruned model.
    """
    try:
        file_path = os.path.normpath(file_path)
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            fieldnames = ['GUID', 'Wavelet', 'Level', 'Threshold', 'Phase',
                          'Total Pruned Count', 'Total Non-Zero Params', 'Model Path']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'GUID': guid,
                'Wavelet': wavelet,
                'Level': level,
                'Threshold': threshold,
                'Phase': phase,
                'Total Pruned Count': total_pruned_count,
                'Total Non-Zero Params': total_non_zero_params,
                'Model Path': model_path
            })
    except Exception as e:
        print(f"Failed to append to experiment log: {e}")


def check_and_set_pruned_instance_path(pruned_instance):
    """
    Check if the script is run from the 'Phase3ResNet' directory and set the pruned_instance_path accordingly.

    Args:
        pruned_instance (str): The name of the pruned instance.

    Returns:
        str: The full path to the pruned instance directory.
    """
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'Phase3ResNet':
        pruned_instance_path = os.path.join('SavedModels', pruned_instance)
    else:
        pruned_instance_path = os.path.join(
            current_dir, 'Phase3ResNet', 'SavedModels', pruned_instance)
    os.makedirs(pruned_instance_path, exist_ok=True)
    return pruned_instance_path


def print_model_summary(model):
    total_params = 0
    print("Model Summary:")
    print("Layer Name" + "\t" * 7 + "Output Shape" + "\t" * 5 + "Param #")
    print("="*100)
    for name, module in model.named_children():
        if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
            layer_info = f"{name}\t" + \
                str(module.weight.size()) + "\t" + f"{module.weight.numel()}"
            total_params += module.weight.numel()
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                total_params += module.bias.numel()
            print(layer_info)
    print("="*100)
    print(f"Total Params: {total_params}")


def print_model_structure(model, depth=0):
    indent = " " * (depth * 2)
    for name, module in model.named_children():
        print(f"{indent}{name} - {module.__class__.__name__}")
        if list(module.children()):
            print_model_structure(module, depth + 1)
