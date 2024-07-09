import os
import pywt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel
from typing import List, Tuple
from utils import log_pruning_details, setup_csv_writer, check_and_set_pruned_instance_path, save_model, append_to_experiment_log, get_layer


def multi_resolution_analysis(weights: List[torch.Tensor], wavelet: str, level: int, threshold: float, mode: str = 'periodization') -> Tuple[List[torch.Tensor], int]:
    """
    Perform multi-resolution analysis and apply wavelet threshold pruning.

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

        # Check the dimensionality of the weight tensor
        if len(weight_np.shape) < 2:
            # For 1D tensors, apply simple threshold pruning
            pruned_weight_np = np.where(np.abs(weight_np) < threshold, 0, weight_np)
            pruned_weight = torch.tensor(pruned_weight_np, dtype=weight.dtype, device=device)
        else:
            # Perform wavelet decomposition for 2D+ tensors
            coeffs = pywt.wavedec2(weight_np, wavelet, level=level, mode=mode, axes=(-2, -1))
            coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs, axes=(-2, -1))

            # Apply threshold
            coeff_arr[np.abs(coeff_arr) < threshold] = 0

            # Reconstruct the weight
            pruned_coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
            pruned_weight_np = pywt.waverec2(pruned_coeffs, wavelet, mode=mode, axes=(-2, -1))

            # Ensure the pruned weight has the same shape as the original
            pruned_weight_np = pruned_weight_np[tuple(slice(0, dim) for dim in original_shape)]
            pruned_weight = torch.tensor(pruned_weight_np, dtype=weight.dtype, device=device)

        pruned_weights.append(pruned_weight)

        # Calculate the number of pruned weights
        original_non_zero_params = torch.count_nonzero(weight).item()
        pruned_non_zero_params = torch.count_nonzero(pruned_weight).item()
        total_pruned_count += original_non_zero_params - pruned_non_zero_params
    
    return pruned_weights, total_pruned_count


def prune_layer_weights(layer: nn.Module, wavelet: str, level: int, threshold: float) -> Tuple[int, int, int]:
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
    weights = [param for param in layer.parameters()]
    original_param_count = sum(weight.numel() for weight in weights)
    non_zero_params = sum(torch.count_nonzero(weight).item() for weight in weights)

    pruned_weights, layer_pruned_count = multi_resolution_analysis(weights, wavelet, level, threshold)

    # Update layer weights
    with torch.no_grad():
        for param, pruned_weight in zip(layer.parameters(), pruned_weights):
            if param.requires_grad:
                param.copy_(pruned_weight)

    return original_param_count, non_zero_params, layer_pruned_count


def wavelet_pruning(model: PreTrainedModel, wavelet: str, level: int, threshold: float, csv_path: str, guid: str) -> str:
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
        f"{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}/selective_pruned")
    selective_log_path = os.path.join(selective_pruned_dir, 'log.csv')
    selective_csv_writer, selective_log_file = setup_csv_writer(
        os.path.normpath(selective_log_path), mode='w')

    # Get all layers that have weights
    layers_to_prune = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]

    # Use tqdm for progress tracking
    for name, module in tqdm(layers_to_prune, desc="Pruning Layers"):
        original_param_count, non_zero_params, layer_pruned_count = prune_layer_weights(
            module, wavelet, level, threshold)
        total_pruned_count += layer_pruned_count
        total_non_zero_params += non_zero_params
        log_pruning_details(selective_csv_writer, guid, wavelet, level, threshold,
                            'selective', original_param_count, non_zero_params, layer_pruned_count, name)

    selective_log_file.close()

    # Save the selectively pruned model
    save_model(model, selective_pruned_dir)
    append_to_experiment_log(os.path.normpath(csv_path), guid, wavelet, level, threshold,
                             'selective', total_pruned_count, total_non_zero_params, selective_pruned_dir)

    print(f"Selectively pruned model saved at {selective_pruned_dir}")

    return selective_log_path
