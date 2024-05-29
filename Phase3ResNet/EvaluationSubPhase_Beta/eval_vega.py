import safetensors.torch
from transformers import AutoConfig, AutoModelForImageClassification
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the data transformation including resizing and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Convert grayscale to 3 channels
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5 and std=0.5
])

# Load FashionMNIST dataset
validation_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)

# Use a smaller subset for evaluation
subset_size = 1000
subset_indices = np.random.choice(
    len(validation_dataset), subset_size, replace=False)
validation_subset = Subset(validation_dataset, subset_indices)
validation_loader = DataLoader(validation_subset, batch_size=64, shuffle=False)


def load_model(model_dir):
    if model_dir.endswith('.bin'):
        # Load the original PyTorch model
        config = AutoConfig.from_pretrained(model_dir)
        model = AutoModelForImageClassification.from_pretrained(
            model_dir, config=config)
        num_ftrs = model.classifier[-1].in_features
        # Adjust for FashionMNIST's 10 classes
        model.classifier[-1] = torch.nn.Linear(num_ftrs, 10)
    else:
        # Load the pruned model (format not specified)
        config = AutoConfig.from_pretrained(model_dir)
        model = AutoModelForImageClassification.from_pretrained(
            model_dir, config=config)
        num_ftrs = model.classifier[-1].in_features
        # Adjust for FashionMNIST's 10 classes
        model.classifier[-1] = torch.nn.Linear(num_ftrs, 10)
    return model


def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_predictions,
                  average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions,
                          average='weighted', zero_division=1)
    cm = confusion_matrix(all_labels, all_predictions)

    return accuracy, f1, recall, cm


def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.pdf')
    plt.close()


def save_metrics_to_file(metrics, file_path):
    with open(file_path, 'w') as file:
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(f"F1 Score: {metrics['f1']:.4f}\n")
        file.write(f"Recall: {metrics['recall']:.4f}\n")
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(metrics['confusion_matrix']))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [
        r"C:\Users\gigac\Documents\Projects\WaveletTransforms\Phase3ResNet\__OGPyTorchModel__",
        r"C:\Users\gigac\Documents\Projects\WaveletTransforms\Phase3ResNet\SavedModels\haar_threshold-0.0236_level-1_guid-7cfc\random_pruned",
        r"C:\Users\gigac\Documents\Projects\WaveletTransforms\Phase3ResNet\SavedModels\haar_threshold-0.0236_level-1_guid-7cfc\selective_pruned"
    ]

    for model_dir in model_paths:
        print(f"Evaluating model: {model_dir}")
        model = load_model(model_dir)
        accuracy, f1, recall, cm = evaluate_model(
            model, validation_loader, device)

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall,
            "confusion_matrix": cm
        }

        metrics_file_path = model_dir + "/evaluation_metrics.txt"
        save_metrics_to_file(metrics, metrics_file_path)

        plot_confusion_matrix(cm, list(
            validation_dataset.classes), title=f"Confusion Matrix for {model_dir}")

        print(f"Model: {model_dir}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
