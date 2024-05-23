import subprocess
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score
import numpy as np
import os
from absl import app, flags
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', 'C:/Users/gigac/Documents/Projects/WaveletTransforms/Phase3ResNet/SavedModels/haar_threshold-0.786_level-1_guid-6bf8',
                    'Directory where pruned models are stored')
flags.DEFINE_string('eval_dir', 'C:/Users/gigac/Documents/Projects/WaveletTransforms/Phase3ResNet/EvaluationSubPhase_reduced',
                    'Directory where evaluation scripts are stored')
flags.DEFINE_string('test_data_dir', './data',
                    'Directory where test data is stored')
flags.DEFINE_string('dataset', 'CIFAR10',
                    'Dataset to use for evaluation (e.g., CIFAR10)')
flags.DEFINE_integer('batch_size', 32, 'Batch size for data loader')

# Define output paths
flags.DEFINE_string('output_pdf', None,
                    'Output PDF file for charts and results')
flags.DEFINE_string('output_log', None, 'Output log file for text results')

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

# Define metrics


def compute_accuracy(output, target):
    _, preds = torch.max(output, 1)
    return accuracy_score(target.cpu(), preds.cpu())


def compute_loss(output, target):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    return loss.item()


def compute_f1(output, target):
    _, preds = torch.max(output, 1)
    return f1_score(target.cpu(), preds.cpu(), average='weighted')


def compute_recall(output, target):
    _, preds = torch.max(output, 1)
    return recall_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)


def compute_confusion_matrix(output, target):
    _, preds = torch.max(output, 1)
    return confusion_matrix(target.cpu(), preds.cpu())


def compute_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()
    return zero_params / total_params

# Evaluate models on the test set


def evaluate_model(model, test_loader):
    accuracy = 0
    loss = 0
    f1 = 0
    recall = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Adjust this line as per your model's output
            outputs = model(inputs).logits
            accuracy += compute_accuracy(outputs, labels)
            loss += compute_loss(outputs, labels)
            f1 += compute_f1(outputs, labels)
            recall += compute_recall(outputs, labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    accuracy /= len(test_loader)
    loss /= len(test_loader)
    f1 /= len(test_loader)
    recall /= len(test_loader)
    sparsity = compute_sparsity(model)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return accuracy, loss, f1, recall, sparsity, conf_matrix


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))

    # Reorder the confusion matrix based on the class names
    class_indices = list(range(len(class_names)))
    cm = cm[np.argsort(class_indices), :][:, np.argsort(class_indices)]

    # Plot the reordered confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion Matrix', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()


def save_results_to_pdf(results, pdf_path, class_names, dataset, batch_size):
    with PdfPages(pdf_path) as pdf:
        for model_name, metrics in results.items():
            accuracy, loss, f1, recall, sparsity, conf_matrix = metrics

            # Create a figure for the metrics
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(
                f'{model_name} Evaluation\nDataset: {dataset}, Batch Size: {batch_size}')

            # Plot accuracy
            axs[0, 0].bar('Accuracy', accuracy)
            axs[0, 0].set_ylim(0, 1)
            axs[0, 0].set_title(f'Accuracy: {accuracy:.6f}')

            # Plot loss
            axs[0, 1].bar('Loss', loss)
            axs[0, 1].set_title(f'Loss: {loss:.6f}')

            # Plot F1-score
            axs[0, 2].bar('F1-score', f1)
            axs[0, 2].set_ylim(0, 1)
            axs[0, 2].set_title(f'F1-score: {f1:.6f}')

            # Plot recall
            axs[1, 0].bar('Recall', recall)
            axs[1, 0].set_ylim(0, 1)
            axs[1, 0].set_title(f'Recall: {recall:.6f}')

            # Plot sparsity
            axs[1, 1].bar('Sparsity', sparsity)
            axs[1, 1].set_title(f'Sparsity: {sparsity:.6f}')

            # Remove empty subplot
            fig.delaxes(axs[1, 2])

            pdf.savefig(fig)
            plt.close(fig)

            # Print and plot confusion matrix
            print(f"Confusion Matrix for {model_name}:\n{conf_matrix}")
            logging.info(f"Confusion Matrix for {model_name}:\n{conf_matrix}")
            plot_confusion_matrix(conf_matrix, class_names)
            pdf.savefig()
            plt.close()

            # Log the raw values
            logging.info(f"Model: {model_name}")
            logging.info(f"Accuracy: {accuracy:.6f}")
            logging.info(f"Loss: {loss:.6f}")
            logging.info(f"F1-score: {f1:.6f}")
            logging.info(f"Recall: {recall:.6f}")
            logging.info(f"Sparsity: {sparsity:.6f}")


def main(argv):
    del argv  # Unused

    model_dir = FLAGS.model_dir
    output_pdf = os.path.join(model_dir, 'evaluation_results.pdf')
    output_log = os.path.join(model_dir, 'evaluation_log.txt')

    setup_logging(output_log)

    selective_pruned_model = load_model(os.path.normpath(
        os.path.join(model_dir, 'selective_pruned')))
    random_pruned_model = load_model(os.path.normpath(
        os.path.join(model_dir, 'random_pruned')))

    selective_pruned_model.eval()
    random_pruned_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    if FLAGS.dataset == 'CIFAR10':
        test_dataset = datasets.CIFAR10(
            root=FLAGS.test_data_dir, train=False, download=True, transform=transform)
        class_names = test_dataset.classes
    else:
        raise ValueError(f"Unsupported dataset: {FLAGS.dataset}")

    test_loader = DataLoader(
        test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    models_to_evaluate = {
        'Selective Pruned Model': selective_pruned_model,
        'Random Pruned Model': random_pruned_model
    }

    results = {}

    for model_name, model in models_to_evaluate.items():
        accuracy, loss, f1, recall, sparsity, conf_matrix = evaluate_model(
            model, test_loader)
        results[model_name] = (
            accuracy, loss, f1, recall, sparsity, conf_matrix)
        logging.info(f"Model: {model_name}")
        logging.info(f"Accuracy: {accuracy:.6f}")
        logging.info(f"Loss: {loss:.6f}")
        logging.info(f"F1-score: {f1:.6f}")
        logging.info(f"Recall: {recall:.6f}")
        logging.info(f"Sparsity: {sparsity:.6f}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")

    save_results_to_pdf(results, output_pdf, class_names,
                        FLAGS.dataset, FLAGS.batch_size)

    # Run additional metrics script
    subprocess.run(["python", "C:/Users/gigac/Documents/Projects/WaveletTransforms/Phase3ResNet/EvaluationSubPhase_reduced/additional_metrics.py",
                   f"--model_dir={model_dir}", f"--test_data_dir={FLAGS.test_data_dir}", f"--dataset={FLAGS.dataset}", f"--batch_size={FLAGS.batch_size}"])


if __name__ == '__main__':
    app.run(main)
