import pywt
import numpy as np
import torch
import torch.nn as nn
from utils import log_pruning_details


def multi_resolution_analysis(weights, wavelet, level, threshold):
    pruned_weights = []
    total_pruned_count = 0
    for weight in weights:
        original_shape = weight.shape
        flattened_weight = np.ravel(weight)
        coeffs = pywt.wavedec(flattened_weight, wavelet,
                              level=level, mode='periodization')
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        for i in range(len(coeff_arr)):
            if np.abs(coeff_arr[i]).mean() < threshold:
                pruned_count = np.sum(coeff_arr[i] != 0)
                total_pruned_count += pruned_count
                coeff_arr[i] = 0
        pruned_coeffs = pywt.array_to_coeffs(
            coeff_arr, coeff_slices, output_format='wavedec')
        pruned_weight = pywt.waverec(
            pruned_coeffs, wavelet, mode='periodization')
        if pruned_weight.size > flattened_weight.size:
            pruned_weight = pruned_weight[:flattened_weight.size]
        elif pruned_weight.size < flattened_weight.size:
            pruned_weight = np.pad(
                pruned_weight, (0, flattened_weight.size - pruned_weight.size), 'constant')
        pruned_weights.append(pruned_weight.reshape(original_shape))
    return pruned_weights, total_pruned_count


def prune_layer_weights(layer, wavelet, level, threshold, csv_writer, guid, layer_name):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        weights = [layer.weight.data.cpu().numpy()]
        if layer.bias is not None:
            weights.append(layer.bias.data.cpu().numpy())
    elif hasattr(layer, 'conv'):  # For Hugging Face Conv layers
        weights = [layer.conv.weight.data.cpu().numpy()]
        if layer.conv.bias is not None:
            weights.append(layer.conv.bias.data.cpu().numpy())
    elif hasattr(layer, 'fc'):  # For Hugging Face Classifier layer
        weights = [layer.fc.weight.data.cpu().numpy()]
        if layer.fc.bias is not None:
            weights.append(layer.fc.bias.data.cpu().numpy())
    elif isinstance(layer, nn.Module):  # For the root module (ResNetForImageClassification)
        weights = []
        for name, param in layer.named_parameters():
            if 'weight' in name:
                weights.append(param.data.cpu().numpy())
    else:
        print(f"Layer {layer_name} is not a supported layer type. Skipping...")
        return 0

    pruned_weights, total_pruned_count = multi_resolution_analysis(
        weights, wavelet, level, threshold)

    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        layer.weight.data = torch.from_numpy(pruned_weights[0])
        if layer.bias is not None:
            layer.bias.data = torch.from_numpy(pruned_weights[1])
    elif hasattr(layer, 'conv'):  # For Hugging Face Conv layers
        layer.conv.weight.data = torch.from_numpy(pruned_weights[0])
        if layer.conv.bias is not None:
            layer.conv.bias.data = torch.from_numpy(pruned_weights[1])
    elif hasattr(layer, 'fc'):  # For Hugging Face Classifier layer
        layer.fc.weight.data = torch.from_numpy(pruned_weights[0])
        if layer.fc.bias is not None:
            layer.fc.bias.data = torch.from_numpy(pruned_weights[1])
    elif isinstance(layer, nn.Module):  # For the root module (ResNetForImageClassification)
        idx = 0
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.from_numpy(pruned_weights[idx])
                idx += 1

    print(f"Assigned pruned weights to layer {layer_name}")

    original_param_count = sum(weight.size for weight in weights)
    non_zero_params = original_param_count - total_pruned_count
    log_pruning_details(csv_writer, guid, wavelet, level, threshold, 'selective',
                        original_param_count, non_zero_params, total_pruned_count, layer_name)
    return total_pruned_count


def recursive_prune(model, wavelet, level, threshold, csv_writer, guid, layer_name_prefix=""):
    layer_prune_counts = {}
    total_prune_count = 0

    def inner_recursive_prune(module, layer_name_prefix):
        nonlocal total_prune_count
        layer_name = f"{layer_name_prefix}/{module.__class__.__name__}"

        if isinstance(module, nn.Sequential):
            for sub_module in module:
                inner_recursive_prune(sub_module, layer_name)
        elif isinstance(module, nn.ModuleList):
            for idx, sub_module in enumerate(module):
                inner_recursive_prune(sub_module, f"{layer_name}.{idx}")
        else:
            try:
                layer_pruned_count = prune_layer_weights(
                    module, wavelet, level, threshold, csv_writer, guid, layer_name)
                if layer_pruned_count > 0:
                    layer_prune_counts[layer_name] = layer_pruned_count
                total_prune_count += layer_pruned_count
                print(
                    f"Layer {layer_name} pruned. Total pruned count: {layer_pruned_count}")
            except Exception as e:
                print(f"Error pruning layer {layer_name}: {e}")

    inner_recursive_prune(model, layer_name_prefix)
    print(f"Completed DWT pruning on {len(layer_prune_counts)} layers.")
    return layer_prune_counts


def wavelet_pruning(model, wavelet, level, threshold, csv_writer, guid):
    layer_prune_counts = recursive_prune(
        model, wavelet, level, threshold, csv_writer, guid)
    return model, layer_prune_counts
