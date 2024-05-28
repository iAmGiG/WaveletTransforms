import os
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
from absl import app, flags
# from torchsummary import summary
# from transformers import AutoModelForImageClassification, AutoConfig
from utils import setup_csv_writer, load_model, save_model, log_pruning_details, append_to_experiment_log, check_and_set_pruned_instance_path
from dwt_pruning import wavelet_pruning

FLAGS = flags.FLAGS

"""
TODO threshold values: - might want to do sub 0.0->0.1 domain
0, 0.236, 0.382, 0.5, 0.618, 0.786, 1
"""
# Command line argument setup
flags.DEFINE_string('model_path', '__OGPyTorchModel__/pytorch_model.bin',
                    'Path to the pre-trained ResNet model (bin file)')
flags.DEFINE_string('config_path', '__OGPyTorchModel__/config.json',
                    'Path to the model configuration file (.json)')
flags.DEFINE_string('csv_path', 'experiment_log.csv',
                    'Path to the CSV log file')
flags.DEFINE_enum('wavelet', 'haar', ['haar', 'db1', 'db2', 'coif1', 'bior1.3',
                  'rbio1.3', 'sym2', 'mexh', 'morl'], 'Type of wavelet to use for DWT.')
flags.DEFINE_integer(
    'level', 1, 'Level of decomposition for the wavelet transform')
flags.DEFINE_float(
    'threshold', 0.0236, 'Threshold value for pruning wavelet coefficients')
flags.DEFINE_string('output_dir', 'SavedModels',
                    'Directory to save the pruned models')


def get_layer(model, layer_name):
    # Handle the case where the layer name is prefixed with the model's class name
    prefix = 'ResNetForImageClassification.'
    if layer_name.startswith(prefix):
        layer_name = layer_name[len(prefix):]  # Remove the prefix

    name_parts = layer_name.split('.')
    current_model = model
    for idx, part in enumerate(name_parts):
        if part:  # Only attempt to get attribute if 'part' is not empty
            print(f"Checking for part '{part}' at level {idx}")
            if hasattr(current_model, part):
                current_model = getattr(current_model, part)
                print(
                    f"Found part '{part}', current model: {type(current_model)}")
            else:
                print(
                    f"Layer part '{part}' not found in the model at level {idx}")
                return None
    return current_model


def random_pruning(selective_pruning_log, model, guid, wavelet, level, threshold):
    total_pruned_count = 0
    total_non_zero_params = 0
    random_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/random_pruned")
    random_log_path = os.path.join(random_pruned_dir, 'log.csv')
    random_csv_writer, random_log_file = setup_csv_writer(
        os.path.normpath(random_log_path), mode='w')

    # Read the selective pruning log file
    with open(selective_pruning_log, 'r') as log_file:
        log_reader = csv.DictReader(log_file)
        for row in log_reader:
            layer_name = row['Layer Name']
            original_param_count = int(row['Original Parameter Count'])
            # non_zero_params = int(row['Non-zero Params'])
            prune_count = int(row['Total Pruned Count'])
            print(
                f"Processing layer: {layer_name} with prune count: {prune_count}")

            layer = get_layer(model, layer_name)
            if layer and isinstance(layer, nn.Conv2d):
                weights = layer.weight.data.cpu().numpy()
                flatten_weights = weights.flatten()
                indices_to_prune = np.random.choice(
                    len(flatten_weights), prune_count, replace=False)
                flatten_weights[indices_to_prune] = 0
                weights = flatten_weights.reshape(weights.shape)
                layer.weight.data = torch.from_numpy(
                    weights).to(layer.weight.device)
                non_zero_params_after_pruning = np.count_nonzero(weights)

                # Calculate the actual pruned count
                actual_pruned_count = original_param_count - non_zero_params_after_pruning

                log_pruning_details(random_csv_writer, guid, wavelet, level, threshold, 'random',
                                    original_param_count, non_zero_params_after_pruning, actual_pruned_count, layer_name)
                total_pruned_count += actual_pruned_count
                total_non_zero_params += non_zero_params_after_pruning
            else:
                print(f"Layer not found or not a Conv2D layer: {layer_name}")

    # Save the randomly pruned model
    save_model(model, random_pruned_dir)
    # Append to the combined experiment log
    append_to_experiment_log(os.path.normpath(FLAGS.csv_path), guid, wavelet, level, threshold, 'random',
                             total_pruned_count, total_non_zero_params, random_pruned_dir)
    random_log_file.close()
    print("Random pruning completed.")


def selective_pruning(original_model, wavelet, level, threshold, guid):
    selective_pruned_model = copy.deepcopy(original_model)
    selective_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/selective_pruned")
    selective_log_path = os.path.join(selective_pruned_dir, 'log.csv')
    selective_csv_writer, selective_log_file = setup_csv_writer(
        os.path.normpath(selective_log_path), mode='w')

    # Perform the wavelet pruning
    selective_pruned_model, layer_prune_counts = wavelet_pruning(
        selective_pruned_model, wavelet, level, threshold, selective_csv_writer, guid)

    total_non_zero_params = 0  # Initialize total non-zero parameters
    for layer_name, prune_count in layer_prune_counts.items():
        layer = get_layer(selective_pruned_model, layer_name)
        if layer:
            original_param_count = sum(param.numel()
                                       for param in layer.parameters())
            non_zero_params = sum(np.count_nonzero(
                param.data.cpu().numpy()) for param in layer.parameters())
            total_non_zero_params += non_zero_params  # Accumulate non-zero parameters
            log_pruning_details(selective_csv_writer, guid, wavelet, level, threshold, 'selective',
                                original_param_count, non_zero_params, prune_count, layer_name)

    total_pruned_count = sum(layer_prune_counts.values())

    save_model(selective_pruned_model, selective_pruned_dir)
    append_to_experiment_log(os.path.normpath(FLAGS.csv_path), guid, wavelet, level,
                             threshold, 'selective', total_pruned_count, total_non_zero_params, selective_pruned_dir)
    selective_log_file.close()
    print(
        f"Selective pruning completed and model saved to {selective_pruned_dir}")
    selective_pruning_log_path = os.path.join(selective_pruned_dir, 'log.csv')
    return layer_prune_counts, selective_pruning_log_path

    # Print the model structure for diagnostics - USE THE utils.py import for the print structure.
    # print("Model Structure:")
    # print_model_structure(selective_pruned_model)


def main(argv):
    model = load_model(FLAGS.model_path, FLAGS.config_path)

    # Append mode for the running experiment log
    # csv_writer, running_log_file = setup_csv_writer(FLAGS.csv_path, mode='a')

    guid = os.urandom(4).hex()

    # Create a new instance of the model for random pruning
    random_pruning_model = copy.deepcopy(model)

    # layer_prune_counts, selective_pruning_log_path = selective_pruning(
    #     model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, guid)

    # # Perform Random pruning using the layer_prune_counts obtained from DWT pruning
    # random_pruning(selective_pruning_log_path, random_pruning_model, guid, FLAGS.wavelet,
    #                FLAGS.level, FLAGS.threshold)

    # running_log_file.close()

    # Selective Pruning Phase
    selective_log_path = wavelet_pruning(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, guid)
    print(f"Selective pruning completed. Log saved at {selective_log_path}")

    # Random Pruning Phase
    random_pruning(selective_log_path, random_pruning_model, guid,
                   FLAGS.wavelet, FLAGS.level, FLAGS.threshold)


if __name__ == '__main__':
    app.run(main)
