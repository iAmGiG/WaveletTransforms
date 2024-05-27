import os
import copy
import numpy as np
import torch
import torch.nn as nn
from absl import app, flags
from torchsummary import summary
from transformers import AutoModelForImageClassification, AutoConfig
from utils import setup_csv_writer, load_model, save_model, log_pruning_details, append_to_experiment_log, check_and_set_pruned_instance_path
from dwt_pruning import wavelet_pruning

FLAGS = flags.FLAGS

"""
TODO threshold values:
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
    'level', 0, 'Level of decomposition for the wavelet transform')
flags.DEFINE_float(
    'threshold', 0.0236, 'Threshold value for pruning wavelet coefficients')
flags.DEFINE_string('output_dir', 'SavedModels',
                    'Directory to save the pruned models')


def get_layer(model, layer_name):
    # Remove leading/trailing slashes and handle empty parts that might result from split
    name_parts = layer_name.strip('/').split('/')
    current_model = model
    for part in name_parts:
        if part:  # Only attempt to get attribute if 'part' is not empty
            if hasattr(current_model, part):
                current_model = getattr(current_model, part)
            else:
                print(f"Layer part '{part}' not found in the model.")
                return None
    return current_model


def random_pruning(layer_prune_counts, guid, wavelet, level, threshold, random_pruning_model):
    total_pruned_count = 0
    total_non_zero_params = 0  # Initialize total non-zero parameters
    random_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/random_pruned")
    random_log_path = os.path.join(random_pruned_dir, 'log.csv')
    random_csv_writer, random_log_file = setup_csv_writer(
        os.path.normpath(random_log_path), mode='w')

    # Debug print
    print(f"Random pruning started: Logging to {random_log_path}")
    for layer_name, prune_count in layer_prune_counts.items():
        print(f"Processing layer: {layer_name}")  # Debug print
        layer = get_layer(random_pruning_model, layer_name)
        if layer and isinstance(layer, nn.Conv2d):
            # Debug print
            print(f"Pruning {prune_count} weights from layer: {layer_name}")
            weights = [param.data.cpu().numpy()
                       for param in layer.parameters()]
            pruned_weights = []
            for weight in weights:
                flattened_weight = weight.flatten()
                original_non_zeros = np.count_nonzero(flattened_weight)
                prune_indices = np.random.choice(flattened_weight.size, min(
                    prune_count, flattened_weight.size), replace=False)
                flattened_weight[prune_indices] = 0
                new_non_zeros = np.count_nonzero(flattened_weight)
                pruned_weight = flattened_weight.reshape(weight.shape)
                pruned_weights.append(pruned_weight)
                # Debug print
                print(
                    f"Layer {layer_name}: {original_non_zeros} -> {new_non_zeros} non-zero parameters")

            non_zero_params = sum(np.count_nonzero(weight)
                                  for weight in pruned_weights)
            total_non_zero_params += non_zero_params  # Accumulate non-zero parameters

            log_pruning_details(random_csv_writer, guid, wavelet, level, threshold, 'random',
                                sum(weight.size for weight in weights), non_zero_params, prune_count, layer_name)
            print(f"Details logged for layer {layer_name}")  # Debug print

            # Update the layer weights with the pruned weights
            for param, pruned_weight in zip(layer.parameters(), pruned_weights):
                param.data = torch.from_numpy(pruned_weight)

            total_pruned_count += prune_count
            print(f"Updated layer {layer_name} weights")  # Debug print

    save_model(random_pruning_model, random_pruned_dir)
    print(f"Model saved at {random_pruned_dir}")  # Debug print
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
    return layer_prune_counts

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

    layer_prune_counts = selective_pruning(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, guid)

    # Perform Random pruning using the layer_prune_counts obtained from DWT pruning
    random_pruning(layer_prune_counts, guid, FLAGS.wavelet,
                   FLAGS.level, FLAGS.threshold, random_pruning_model)

    # running_log_file.close()


if __name__ == '__main__':
    app.run(main)
