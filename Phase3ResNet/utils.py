import os
import csv
from transformers import TFAutoModelForImageClassification, AutoConfig


def load_model(model_path, config_path):
    """
    Loads the model
    """
    config = AutoConfig.from_pretrained(config_path)
    model = TFAutoModelForImageClassification.from_pretrained(
        model_path, config=config)
    return model


def save_model(model, output_path):
    """
    saves the model
    """
    output_path = os.path.join(os.getcwd(), output_path)
    model.save(output_path)
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


def append_to_experiment_log(file_path, guid, wavelet, level, threshold, phase, total_pruned_count):
    """
    Append details of the pruning experiment to the experiment log.

    Args:
        file_path (str): Path to the experiment log CSV file.
        guid (str): Unique identifier for the pruning session.
        wavelet (str): Wavelet type used.
        level (int): Level of decomposition.
        threshold (float): Threshold value for pruning.
        phase (str): Pruning phase ('selective' or 'random').
        total_pruned_count (int): Total number of pruned weights.
    """
    try:
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            fieldnames = ['GUID', 'Wavelet', 'Level',
                          'Threshold', 'Phase', 'Total Pruned Count']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'GUID': guid,
                'Wavelet': wavelet,
                'Level': level,
                'Threshold': threshold,
                'Phase': phase,
                'Total Pruned Count': total_pruned_count
            })
    except Exception as e:
        print(f"Failed to append to experiment log: {e}")
