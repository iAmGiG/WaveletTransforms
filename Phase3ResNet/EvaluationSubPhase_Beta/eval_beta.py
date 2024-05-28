import os
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file as load_safetensors
from datasets import load_from_disk, DatasetDict, Dataset
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


def preprocess(batch, image_processor):
    images = [image_processor(image.convert("RGB"), return_tensors='pt')[
        'pixel_values'] for image in batch['image']]
    pixel_values = torch.cat(images, dim=0)
    return {'pixel_values': pixel_values, 'labels': batch['label']}


def main():
    # Paths to directories
    original_model_dir = "C:\\Users\\gigac\\Documents\\Projects\\WaveletTransforms\\Phase3ResNet\\__OGPyTorchModel__"
    dataset_dir = "C:\\Users\\gigac\\Documents\\Projects\\WaveletTransforms\\Phase3ResNet\\EvaluationSubPhase_Beta\\imagenet-1k-dataset"
    cache_dir = "C:\\Users\\gigac\\Documents\\Projects\\WaveletTransforms\\Phase3ResNet\\EvaluationSubPhase_Beta\\cache"

    # Load the dataset
    print("Loading dataset from disk...")
    dataset = load_from_disk(dataset_dir)
    print("Dataset loaded successfully.")

    # Ensure the dataset is in the correct format
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(validation_dataset))
    print("Test dataset size:", len(test_dataset))

    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

    # Apply preprocessing and cache
    def preprocess_function(batch):
        return preprocess(batch, image_processor)

    print("Applying preprocessing to the dataset...")
    cache_file = os.path.join(cache_dir, "validation_dataset.arrow")

    if os.path.exists(cache_file):
        print("Loading cached dataset...")
        validation_dataset = Dataset.load_from_disk(cache_file)
    else:
        print("Processing and caching the dataset...")
        validation_dataset = validation_dataset.map(
            preprocess_function, batched=True, remove_columns=["image"])
        validation_dataset.save_to_disk(cache_file)

    original_model = load_model(original_model_dir, 'bin')

    # Use a very small subset of the validation dataset for quicker evaluation
    subset_indices = np.random.choice(
        len(validation_dataset), size=200, replace=False)
    subset_dataset = validation_dataset.select(subset_indices)

    dataloader = DataLoader(subset_dataset, batch_size=32, shuffle=False)

    accuracy, f1, recall, cm = evaluate_model(original_model, dataloader)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")

    save_metrics_to_pdf(accuracy, f1, recall, cm)
    save_metrics_to_txt(accuracy, f1, recall)


if __name__ == "__main__":
    main()
