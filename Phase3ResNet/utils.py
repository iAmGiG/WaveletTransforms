import tensorflow as tf
import os
import csv


def load_model(model_path, config_path):
    from transformers import TFAutoModelForImageClassification, AutoConfig
    config = AutoConfig.from_pretrained(config_path)
    model = TFAutoModelForImageClassification.from_pretrained(
        model_path, config=config)
    return model


def save_model(model, output_path):
    output_path = os.path.join(os.getcwd(), output_path)
    model.save(output_path)
    print(f"Model saved successfully at {output_path}")


def setup_csv_writer(csv_path):
    file_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, mode='a', newline='')
    fieldnames = ['GUID', 'Wavelet', 'Level', 'Threshold', 'DWT Phase',
                  'Original Parameter Count', 'Non-zero Params', 'Total Pruned Count', 'Layer Name']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    return writer, csv_file


def log_pruning_details(csv_writer, guid, wavelet, level, threshold, dwt_phase, original_param_count, non_zero_params, total_pruned_count, layer_name):
    csv_writer.writerow({
        'GUID': guid,
        'Wavelet': wavelet,
        'Level': level,
        'Threshold': threshold,
        'DWT Phase': dwt_phase,
        'Original Parameter Count': original_param_count,
        'Non-zero Params': non_zero_params,
        'Total Pruned Count': total_pruned_count,
        'Layer Name': layer_name
    })
