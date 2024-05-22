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
    'threshold', 0.1, 'Threshold value for pruning wavelet coefficients')
flags.DEFINE_string('output_dir', 'SavedModels',
                    'Directory to save the pruned models')


def get_layer_by_name(model, layer_name):
    parts = layer_name.split('/')
    current_model = model
    for part in parts[:-1]:
        current_model = getattr(current_model, part)
    return getattr(current_model, parts[-1])


def random_pruning(layer_prune_counts, guid, wavelet, level, threshold, random_pruning_model):
    total_pruned_count = 0

    # Reload the original model
    random_pruned_model = random_pruning_model

    # Set up logging for the random pruned model
    random_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/random_pruned")
    random_log_path = os.path.join(random_pruned_dir, 'log.csv')
    random_csv_writer, random_log_file = setup_csv_writer(
        os.path.normpath(random_log_path), mode='w')

    for layer_name, prune_count in layer_prune_counts.items():
        print(f"layer: {layer_name}\nPrune count: {prune_count}")
        if prune_count > 0:
            layer = get_layer_by_name(random_pruned_model, layer_name)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights = [layer.weight.data.cpu().numpy()]
                if layer.bias is not None:
                    weights.append(layer.bias.data.cpu().numpy())
                # For Hugging Face Conv layers
            elif hasattr(layer, 'conv'):
                weights = [layer.conv.weight.data.cpu().numpy()]
                if layer.conv.bias is not None:
                    weights.append(layer.conv.bias.data.cpu().numpy())
            else:
                print(
                    f"Layer {layer_name} is not a supported layer type. Skipping...")
                continue

            # original_param_count = np.prod(weights[0].shape)
            original_param_count = sum(weight.size for weight in weights)
            num_weights = sum(weight.size for weight in weights)

            # Debugging statements
            print(
                f"Layer '{layer_name}' has {num_weights} weights. Trying to prune {prune_count} weights.")

            # Ensure that the number of weights to prune doesn't exceed the total
            num_to_prune = min(prune_count, num_weights)
            print(f"Num to prune: {num_to_prune}")

            # Randomly prune individual weights
            pruned_weights = []
            for weight in weights:
                flattened_weight = weight.flatten()
                prune_indices = np.random.choice(
                    flattened_weight.size, num_to_prune, replace=False)
                flattened_weight[prune_indices] = 0
                pruned_weight = flattened_weight.reshape(weight.shape)
                pruned_weights.append(pruned_weight)

            # Debugging statements
            print(
                f"Pruned weights lengths: {[weight.size for weight in pruned_weights]}")
            print(
                f"Original weights lengths: {[weight.size for weight in weights]}")

            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = torch.from_numpy(pruned_weights[0])
                if layer.bias is not None:
                    layer.bias.data = torch.from_numpy(pruned_weights[1])
            elif hasattr(layer, 'conv'):
                layer.conv.weight.data = torch.from_numpy(pruned_weights[0])
                if layer.conv.bias is not None:
                    layer.conv.bias.data = torch.from_numpy(pruned_weights[1])

            # Calculate the number of non-zero parameters after pruning
            non_zero_params = sum(np.count_nonzero(weight)
                                  for weight in pruned_weights)

            # Log pruning details
            log_pruning_details(random_csv_writer, guid, wavelet, level, threshold, 'random',
                                original_param_count, non_zero_params, num_to_prune, layer_name)

            total_pruned_count += num_to_prune
            print(
                f"Layer '{layer_name}' pruned with {num_to_prune} parameters in random pruning.")

    # Save the randomly pruned model
    save_model(random_pruned_model, random_pruned_dir)

    # Append to the experiment log
    experiment_log_path = os.path.normpath(FLAGS.csv_path)
    append_to_experiment_log(
        experiment_log_path, guid, wavelet, level, threshold, 'random', total_pruned_count, random_pruned_dir)

    random_log_file.close()

    print("Random pruning completed.")


def selective_pruning(original_model, wavelet, level, threshold, csv_writer, guid):
    selective_pruned_model = copy.deepcopy(original_model)
    layer_prune_counts = {}

    def recursive_get_layer(model, layer_name):
        name_parts = layer_name.split('.')
        current_model = model
        for part in name_parts:
            if isinstance(current_model, nn.Module) and hasattr(current_model, part):
                current_model = getattr(current_model, part)
            else:
                return None
        return current_model

    layer_prune_counts = recursive_prune(
        selective_pruned_model, wavelet, level, threshold, csv_writer, guid)

    selective_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/selective_pruned")

    # Save the selective pruned model
    save_model(selective_pruned_model, selective_pruned_dir)

    # Append to the experiment log
    append_to_experiment_log(os.path.normpath(FLAGS.csv_path), guid, wavelet, level,
                             threshold, 'selective', sum(layer_prune_counts.values()), selective_pruned_dir)

    # Set up logging for the selective pruned model
    selective_log_path = os.path.join(selective_pruned_dir, 'log.csv')
    selective_csv_writer, selective_log_file = setup_csv_writer(
        os.path.normpath(selective_log_path), mode='w')

    # Log pruning details
    for layer_name, prune_count in layer_prune_counts.items():
        layer = recursive_get_layer(selective_pruned_model, layer_name)
        if layer is not None:
            try:
                original_param_count = sum(param.numel()
                                           for param in layer.parameters())
                non_zero_params = sum(param.data.cpu().numpy().flatten().nonzero()[
                                      0].size for param in layer.parameters())
                log_pruning_details(selective_csv_writer, guid, wavelet, level, threshold, 'selective',
                                    original_param_count, non_zero_params, prune_count, layer_name)
            except Exception as e:
                print(f"Error logging details for layer '{layer_name}': {e}")
        else:
            print(f"Could not find layer '{layer_name}' in the model.")

    selective_log_file.close()

    print(
        f"Selective pruning completed and model saved to {selective_pruned_dir}")
    return layer_prune_counts


def main(argv):
    model = load_model(FLAGS.model_path, FLAGS.config_path)

    # Recursively print the model structure
    # print_model_structure(model)

    # Append mode for the running experiment log
    csv_writer, running_log_file = setup_csv_writer(FLAGS.csv_path, mode='a')

    guid = os.urandom(4).hex()

    # Create a new instance of the model for random pruning
    random_pruning_model = copy.deepcopy(model)

    # Perform DWT (Selective) pruning
    layer_prune_counts = selective_pruning(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, csv_writer, guid)

    # Perform Random pruning using the layer_prune_counts obtained from DWT pruning
    random_pruning(layer_prune_counts, guid, FLAGS.wavelet,
                   FLAGS.level, FLAGS.threshold, random_pruning_model)

    running_log_file.close()


if __name__ == '__main__':
    app.run(main)
