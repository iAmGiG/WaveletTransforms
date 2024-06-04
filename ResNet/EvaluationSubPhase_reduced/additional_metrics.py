import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import precision_score, roc_auc_score
import time
import os
from absl import app, flags
import numpy as np
import logging
from torch.nn.functional import softmax
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None,
                    'Directory where pruned models are stored')
flags.DEFINE_string('test_data_dir', None,
                    'Directory where test data is stored')
flags.DEFINE_string('dataset', 'CIFAR10',
                    'Dataset to use for evaluation (e.g., CIFAR10)')
flags.DEFINE_integer('batch_size', 32, 'Batch size for data loader')

# Set up logging


def setup_logging(output_log):
    logging.basicConfig(filename=output_log,
                        level=logging.INFO, format='%(message)s')

# Load the pruned models


def load_model(model_dir):
    config_path = os.path.join(model_dir, 'config.json')
    config = AutoConfig.from_pretrained(config_path, local_files_only=True)
    model = AutoModelForImageClassification.from_config(config)
    model_path = os.path.join(model_dir, 'model.safetensors')

    # Load safetensors
    model_weights = load_safetensors(model_path)
    model.load_state_dict(model_weights)

    return model

# Calculate additional metrics


def calculate_additional_metrics(model, test_loader, n_classes):
    precision = 0
    all_labels = []
    all_preds = []
    all_scores = []
    inference_times = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            start_time = time.time()
            outputs = model(inputs).logits
            end_time = time.time()

            _, preds = torch.max(outputs, 1)
            # Apply softmax to get probabilities
            probabilities = softmax(outputs, dim=1)

            precision += precision_score(labels.cpu(), preds.cpu(),
                                         average='weighted', zero_division=0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())

            inference_time = end_time - start_time
            inference_times.append(inference_time)

    precision /= len(test_loader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)

    # Ensure all_labels are binarized to match the number of classes
    all_labels_binarized = label_binarize(
        all_labels, classes=np.arange(n_classes))

    # Check if the dimensions match
    if all_labels_binarized.shape[1] != all_scores.shape[1]:
        # Align the number of columns in the binarized labels with the predicted scores
        if all_labels_binarized.shape[1] > all_scores.shape[1]:
            all_labels_binarized = all_labels_binarized[:,
                                                        :all_scores.shape[1]]
        else:
            padding = all_scores.shape[1] - all_labels_binarized.shape[1]
            all_labels_binarized = np.pad(
                all_labels_binarized, ((0, 0), (0, padding)), mode='constant')

    # Check for at least two unique classes
    unique_classes = np.unique(all_labels)
    if len(unique_classes) < 2:
        auroc = float('nan')
    else:
        try:
            auroc = roc_auc_score(
                all_labels_binarized, all_scores, multi_class='ovo', average='weighted')
        except ValueError as e:
            auroc = float('nan')
            logging.warning(f"AUROC calculation failed: {e}")

    avg_inference_time = sum(inference_times) / len(inference_times)

    return precision, auroc, avg_inference_time

# Layer-wise information


def layerwise_information(model):
    layer_info = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            num_params = param.numel()
            num_zeros = torch.sum(param == 0).item()
            sparsity = num_zeros / num_params
            layer_info.append(
                (name, num_params, num_zeros, sparsity, param.shape))
    return layer_info

# Model size


def get_model_size(model_path):
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)  # size in MB
    else:
        return None

# Save results to log


def save_results_to_log(model_name, precision, auroc, avg_inference_time, layer_info, original_size, pruned_size, compression_ratio):
    logging.info(f"Model: {model_name}")
    logging.info(f"Precision: {precision:.6f}")
    logging.info(f"AUROC: {auroc:.6f}")
    logging.info(f"Average Inference Time: {avg_inference_time:.6f} seconds")
    logging.info(f"Original Model Size: {original_size:.2f} MB")
    logging.info(f"Pruned Model Size: {pruned_size:.2f} MB")
    logging.info(f"Compression Ratio: {compression_ratio:.2f}")

    for layer in layer_info:
        logging.info(
            f"Layer: {layer[0]}, Params: {layer[1]}, Zeros: {layer[2]}, Sparsity: {layer[3]:.6f}, Shape: {layer[4]}")

# Save results to PDF


def save_results_to_pdf(results, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for model_name, metrics in results.items():
            precision, auroc, avg_inference_time, layer_info, original_size, pruned_size, compression_ratio = metrics

            # Create a figure for the metrics
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{model_name} Evaluation')

            # Plot precision
            axs[0, 0].bar('Precision', precision)
            axs[0, 0].set_ylim(0, 1)
            axs[0, 0].set_title(f'Precision: {precision:.6f}')

            # Plot AUROC
            axs[0, 1].bar('AUROC', auroc)
            axs[0, 1].set_ylim(0, 1)
            axs[0, 1].set_title(f'AUROC: {auroc:.6f}')

            # Plot Average Inference Time
            axs[0, 2].bar('Avg Inference Time', avg_inference_time)
            axs[0, 2].set_title(
                f'Avg Inference Time: {avg_inference_time:.6f} seconds')

            # Plot Original Model Size
            axs[1, 0].bar('Original Size', original_size)
            axs[1, 0].set_title(f'Original Size: {original_size:.2f} MB')

            # Plot Pruned Model Size
            axs[1, 1].bar('Pruned Size', pruned_size)
            axs[1, 1].set_title(f'Pruned Size: {pruned_size:.2f} MB')

            # Plot Compression Ratio
            axs[1, 2].bar('Compression Ratio', compression_ratio)
            axs[1, 2].set_title(f'Compression Ratio: {compression_ratio:.2f}')

            pdf.savefig(fig)
            plt.close(fig)

            # Plot layer-wise sparsity
            fig, ax = plt.subplots(figsize=(15, 10))
            layers = [layer[0] for layer in layer_info]
            sparsities = [layer[3] for layer in layer_info]
            ax.barh(layers, sparsities)
            ax.set_xlabel('Sparsity')
            ax.set_title(f'Layer-wise Sparsity for {model_name}')
            pdf.savefig(fig)
            plt.close(fig)


def main(argv):
    del argv  # Unused

    model_dir = FLAGS.model_dir
    test_data_dir = FLAGS.test_data_dir
    dataset = FLAGS.dataset
    batch_size = FLAGS.batch_size

    output_log = os.path.join(model_dir, 'additional_metrics_log.txt')
    output_pdf = os.path.join(model_dir, 'additional_metrics_results.pdf')
    setup_logging(output_log)

    selective_pruned_model_dir = os.path.join(model_dir, 'selective_pruned')
    random_pruned_model_dir = os.path.join(model_dir, 'random_pruned')

    selective_pruned_model = load_model(selective_pruned_model_dir)
    random_pruned_model = load_model(random_pruned_model_dir)

    selective_pruned_model.eval()
    random_pruned_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    if dataset == 'CIFAR10':
        test_dataset = datasets.CIFAR10(
            root=test_data_dir, train=False, download=True, transform=transform)
        n_classes = 10  # CIFAR-10 has 10 classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    models_to_evaluate = {
        'Selective Pruned Model': selective_pruned_model,
        'Random Pruned Model': random_pruned_model
    }

    original_model_size = get_model_size(
        os.path.join(model_dir, 'model.safetensors'))
    if original_model_size is None:
        original_model_size = 0  # Handle case when original model size is not found

    results = {}

    for model_name, model in models_to_evaluate.items():
        precision, auroc, avg_inference_time = calculate_additional_metrics(
            model, test_loader, n_classes)
        layer_info = layerwise_information(model)
        pruned_model_size = get_model_size(os.path.join(
            model_dir, f'{model_name.lower().replace(" ", "_")}', 'model.safetensors'))
        if pruned_model_size is None:
            pruned_model_size = 0  # Handle case when pruned model size is not found
        compression_ratio = original_model_size / \
            pruned_model_size if pruned_model_size > 0 else 0
        save_results_to_log(model_name, precision, auroc, avg_inference_time,
                            layer_info, original_model_size, pruned_model_size, compression_ratio)
        results[model_name] = (precision, auroc, avg_inference_time, layer_info,
                               original_model_size, pruned_model_size, compression_ratio)

    save_results_to_pdf(results, output_pdf)


if __name__ == '__main__':
    app.run(main)
