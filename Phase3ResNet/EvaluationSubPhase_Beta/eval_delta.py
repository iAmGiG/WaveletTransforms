import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoConfig
import os

# Define the data transformation including resizing and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load FashionMNIST dataset
validation_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)

# Use a smaller subset for evaluation
subset_size = 1000
subset_indices = np.random.choice(
    len(validation_dataset), subset_size, replace=False)
validation_subset = Subset(validation_dataset, subset_indices)
validation_loader = DataLoader(validation_subset, batch_size=16, shuffle=False)


def load_model(model_dir):
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(
        model_dir, config=config)
    return model


def evaluate_model(model, dataloader):
    model.eval()
    device = torch.device("cpu")  # Use CPU
    model.to(device)

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Debug: print inputs and outputs
            print(f"Inputs: {inputs.shape}")
            print(f"Labels: {labels}")
            print(f"Outputs: {outputs}")
            print(f"Predicted: {predicted}")

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions,
                  average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions,
                          average='weighted', zero_division=1)
    cm = confusion_matrix(all_labels, all_predictions)

    return accuracy, f1, recall, cm


def save_metrics(model_dir, accuracy, f1, recall, cm):
    metrics_path = os.path.join(model_dir, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # Save the confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))


def main():
    pruned_models = [
        r"C:\Users\gigac\Documents\Projects\WaveletTransforms\Phase3ResNet\__OGPyTorchModel__",
        "C:/Users/gigac/Documents/Projects/WaveletTransforms/Phase3ResNet/SavedModels/db1_threshold-0.001_level-0_guid-8b15/random_pruned",
        "C:/Users/gigac/Documents/Projects/WaveletTransforms/Phase3ResNet/SavedModels/db1_threshold-0.001_level-0_guid-8b15/selective_pruned"
    ]

    for model_dir in pruned_models:
        model = load_model(model_dir)
        print(model)  # Print model summary

        accuracy, f1, recall, cm = evaluate_model(model, validation_loader)
        save_metrics(model_dir, accuracy, f1, recall, cm)

        print(f"Model: {model_dir}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
