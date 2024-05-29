import os
import torch
import torchvision
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score

# Load CIFAR-10 dataset with proper transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),
    # ImageNet normalization
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

validation_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
validation_loader = DataLoader(
    validation_dataset, batch_size=32, shuffle=False)

# Function to load the model


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

# Function to evaluate the model


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            print(
                f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions,
                  average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions,
                          average='weighted', zero_division=1)
    cm = confusion_matrix(all_labels, all_predictions)
    return accuracy, f1, recall, cm

# Function to save metrics to PDF


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

# Function to save metrics to TXT


def save_metrics_to_txt(accuracy, f1, recall):
    with open("evaluation_metrics.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")


def main():
    original_model_dir = "C:\\Users\\gigac\\Documents\\Projects\\WaveletTransforms\\Phase3ResNet\\__OGPyTorchModel__"

    # Load the model
    original_model = load_model(original_model_dir, 'bin')
    original_model.to('cpu')  # Assuming you're running on CPU

    # Verify model structure
    print(original_model)

    # Evaluate the model on the validation dataset
    accuracy, f1, recall, cm = evaluate_model(
        original_model, validation_loader)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")

    save_metrics_to_pdf(accuracy, f1, recall, cm)
    save_metrics_to_txt(accuracy, f1, recall)


if __name__ == "__main__":
    main()
