import os
import pywt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel
from typing import List, Tuple
from utils import log_pruning_details, setup_csv_writer, check_and_set_pruned_instance_path, save_model, append_to_experiment_log, get_layer


def calculate_max_level(shape, wavelet):
    return pywt.dwt_max_level(min(shape[-2:]), pywt.Wavelet(wavelet).dec_len)


def calculate_entropy(coeff_arr):
    """Calculate entropy of the wavelet coefficients array."""
    # Normalize the coefficients to get a probability distribution
    coeff_flat = coeff_arr.flatten()
    coeff_prob = np.abs(coeff_flat) / np.sum(np.abs(coeff_flat))
    # Adding a small constant to avoid log(0)
    entropy = -np.sum(coeff_prob * np.log(coeff_prob + 1e-12))
    return entropy


def entropy_based_thresholding(coeff_arr, entropy, threshold=1.0):
    """Apply entropy-based thresholding to the coefficients array."""
    threshold = threshold * entropy
    # Debugging output
    print(
        f"Entropy: {entropy}, Threshold: {threshold}, Max Coeff: {np.max(np.abs(coeff_arr))}")
    pruned_coeff_arr = np.where(np.abs(coeff_arr) < threshold, 0, coeff_arr)
    return pruned_coeff_arr, threshold


def percentile_based_thresholding(coeff_arr, percentile=90):
    """Apply percentile-based thresholding to the coefficients array."""
    threshold = np.percentile(np.abs(coeff_arr), percentile)
    # Debugging output
    print(
        f"Percentile: {percentile}, Threshold: {threshold}, Max Coeff: {np.max(np.abs(coeff_arr))}")
    pruned_coeff_arr = np.where(np.abs(coeff_arr) < threshold, 0, coeff_arr)
    return pruned_coeff_arr, threshold


def multi_resolution_analysis(weights: List[torch.Tensor], wavelet: str, level: int, percentile: float, mode: str = 'periodization') -> Tuple[List[torch.Tensor], int]:
    """
    Perform multi-resolution analysis and apply percentile-based wavelet threshold pruning.

    Args:
        weights (List[torch.Tensor]): List of weight tensors to be pruned.
        wavelet (str): Wavelet type to be used.
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for pruning.
        mode (str): Mode for wavelet transform (default: 'periodization').

    Returns:
        Tuple[List[torch.Tensor], int]: A tuple containing pruned weights and total pruned count.
    """
    pruned_weights = []
    total_pruned_count = 0

    for weight in weights:
        original_shape = weight.shape
        device = weight.device
        weight_np = weight.detach().cpu().numpy()

        print(f"Original Shape: {original_shape}, Weight Size: {weight_np.size}")

        if len(weight_np.shape) < 2:
            pruned_weight_np, threshold = percentile_based_thresholding(weight_np, percentile)
            pruned_weight = torch.tensor(pruned_weight_np, dtype=weight.dtype, device=device).view(original_shape)
        else:
            max_level = pywt.dwt_max_level(min(weight_np.shape[-2:]), wavelet)
            level = min(level, max_level)

            coeffs = pywt.wavedec2(weight_np, wavelet, level=level, mode=mode, axes=(-2, -1))
            coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs, axes=(-2, -1))
            
            pruned_coeff_arr, threshold = percentile_based_thresholding(coeff_arr, percentile)
            
            pruned_coeffs = pywt.array_to_coeffs(pruned_coeff_arr, coeff_slices, output_format='wavedec2')
            pruned_weight_np = pywt.waverec2(pruned_coeffs, wavelet, mode=mode)

            if pruned_weight_np.shape != original_shape:
                pruned_weight_np = pruned_weight_np[:original_shape[0], :original_shape[1], :original_shape[2], :original_shape[3]]

            pruned_weight = torch.tensor(pruned_weight_np, dtype=weight.dtype, device=device).view(original_shape)

        pruned_weights.append(pruned_weight.to(device))
        pruned_count = np.sum(pruned_weight.cpu().numpy() == 0)
        print(f"Pruned Count for weight: {pruned_count}")
        total_pruned_count += pruned_count

    return pruned_weights, total_pruned_count


def prune_layer_weights(layer: nn.Module, wavelet: str, level: int, percentile: float) -> Tuple[int, int, int]:
    """
    Prune the weights of a given layer using wavelet-based pruning.

    Args:
        layer (nn.Module): The layer to be pruned.
        wavelet (str): Wavelet type to be used.
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for pruning.

    Returns:
        Tuple[int, int, int]: A tuple containing the original parameter count, non-zero parameter count after pruning, and total pruned count.
    """
    # Original snippet (as shown in the image)
    with torch.no_grad():
        weights = [layer.weight]  # Only consider the weight, not the bias
        original_param_count = sum(param.numel() for param in weights)
        
        pruned_weights, total_pruned_count = multi_resolution_analysis(weights, wavelet, level, percentile)
        
        non_zero_params = sum(param.nonzero().size(0) for param in pruned_weights)
        print(f"Original Param Count: {original_param_count}, Non-zero Params: {non_zero_params}, Total Pruned Count: {total_pruned_count}")
        
        # Assign pruned weights back to the layer
        layer.weight.data = pruned_weights[0]
        
        return original_param_count, non_zero_params, total_pruned_count


def wavelet_pruning(model: PreTrainedModel, wavelet: str, level: int, percentile: float, csv_path: str, guid: str) -> str:
    """
    Apply wavelet-based pruning to the model, log the pruning details, and save the pruned model.

    Args:
        model (PreTrainedModel): The pre-trained Hugging Face transformer model to be pruned.
        wavelet (str): Wavelet type to be used.
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for pruning.
        csv_path (str): Path to the CSV file for logging overall experiment results.
        guid (str): Unique identifier for the pruning session.

    Returns:
        str: Path to the log file containing the pruning details.
    """
    total_pruned_count = 0
    total_non_zero_params = 0
    selective_pruned_dir = check_and_set_pruned_instance_path(
        f"{wavelet}_threshold-{percentile}_level-{level}_guid-{guid[:4]}/selective_pruned")
    selective_log_path = os.path.join(selective_pruned_dir, 'log.csv')
    selective_csv_writer, selective_log_file = setup_csv_writer(
        os.path.normpath(selective_log_path), mode='w')

    layers_to_prune = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]

    for name, module in tqdm(layers_to_prune, desc="Pruning Layers"):
        original_param_count, non_zero_params, layer_pruned_count = prune_layer_weights(
            module, wavelet, level, percentile)
        total_pruned_count += layer_pruned_count
        total_non_zero_params += non_zero_params
        log_pruning_details(selective_csv_writer, guid, wavelet, level, percentile,
                            'selective', original_param_count, non_zero_params, layer_pruned_count, name)

    selective_log_file.close()

    save_model(model, selective_pruned_dir)
    append_to_experiment_log(os.path.normpath(csv_path), guid, wavelet, level, percentile,
                             'selective', total_pruned_count, total_non_zero_params, selective_pruned_dir)

    print(f"Selectively pruned model saved at {selective_pruned_dir}")

    return selective_log_path