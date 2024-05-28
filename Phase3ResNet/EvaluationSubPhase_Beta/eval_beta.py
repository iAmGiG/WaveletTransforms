import os
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file as load_safetensors
from datasets import load_from_disk
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score


def load_model(model_dir, model_type='bin'):
    config_path = os.path.join(model_dir, 'config.json')
    config = AutoConfig.from_pretrained(config_path, local_files_only=True)
    model = AutoModelForImageClassification.from_config(config)
    model_path = os.path.join(model_dir, f'pytorch_model.{model_type}')

    if model_type == 'safetensors':
        model_weights = load_safetensors(model_path)
    elif model_type == 'bin':
        model_weights = torch.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_type}")

    model.load_state_dict(model_weights)
    return model


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values']
            labels = batch['labels']
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    cm = confusion_matrix(all_labels, all_predictions)
    return accuracy, f1, recall, cm


def save_metrics_to_pdf(accuracy, f1, recall, cm):
    fig, ax = plt.subplots(2, 1, figsize=(10, 15))

    # Plot confusion matrix
    cax = ax[0].matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax[0].set_title("Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    # Plot metrics
    metrics = {'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall}
    ax[1].bar(metrics.keys(), metrics.values())
    ax[1].set_ylim([0, 1])
    ax[1].set_title("Evaluation Metrics")

    plt.tight_layout()
    plt.savefig("evaluation_metrics.pdf")


def save_metrics_to_txt(accuracy, f1, recall):
    with open("evaluation_metrics.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")


def main():
    original_model_dir = os.path.abspath(
        os.path.join('..', '__OGPyTorchModel__'))
    dataset_dir = os.path.abspath(os.path.join('imagenet-1k-dataset'))

    print(f"Original model directory: {original_model_dir}")
    print(f"Dataset directory: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory {dataset_dir} does not exist.")

    original_model = load_model(original_model_dir, 'bin')
    dataset = load_from_disk(dataset_dir)

    print(dataset)  # Check the structure to find the correct field name

    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

    def preprocess(batch):
        images = [image_processor(image, return_tensors='pt')[
            'pixel_values'] for image in batch['image']]
        return {'pixel_values': torch.stack(images).squeeze(1), 'labels': batch['label']}

    dataset = dataset.map(preprocess, batched=True)
    dataloader = DataLoader(
        dataset['validation'], batch_size=32, shuffle=False)

    accuracy, f1, recall, cm = evaluate_model(original_model, dataloader)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")

    save_metrics_to_pdf(accuracy, f1, recall, cm)
    save_metrics_to_txt(accuracy, f1, recall)


if __name__ == "__main__":
    main()
