import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import f1_score, recall_score
import numpy as np
import os
from absl import app, flags

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

# Step 3: Load the pruned models


def load_model(model_dir):
    config_path = os.path.join(model_dir, 'config.json')
    config = AutoConfig.from_pretrained(config_path, local_files_only=True)
    model = AutoModelForImageClassification.from_config(config)
    model_path = os.path.join(model_dir, 'model.safetensors')

    # Load safetensors
    model_weights = load_safetensors(model_path)
    model.load_state_dict(model_weights)

    return model

# Step 5: Define metrics


def compute_accuracy(output, target):
    _, preds = torch.max(output, 1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)


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


def compute_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()
    return zero_params / total_params

# Step 7: Evaluate models on the test set


def evaluate_model(model, test_loader):
    accuracy = 0
    loss = 0
    f1 = 0
    recall = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Adjust this line as per your model's output
            outputs = model(inputs).logits
            accuracy += compute_accuracy(outputs, labels)
            loss += compute_loss(outputs, labels)
            f1 += compute_f1(outputs, labels)
            recall += compute_recall(outputs, labels)
    accuracy /= len(test_loader)
    loss /= len(test_loader)
    f1 /= len(test_loader)
    recall /= len(test_loader)
    sparsity = compute_sparsity(model)
    return accuracy, loss, f1, recall, sparsity


def main(argv):
    del argv  # Unused

    selective_pruned_model = load_model(os.path.normpath(
        os.path.join(FLAGS.model_dir, 'selective_pruned')))
    random_pruned_model = load_model(os.path.normpath(
        os.path.join(FLAGS.model_dir, 'random_pruned')))

    selective_pruned_model.eval()
    random_pruned_model.eval()

    # Step 5: Define transform for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Step 1: Load a smaller dataset
    if FLAGS.dataset == 'CIFAR10':
        test_dataset = datasets.CIFAR10(
            root=FLAGS.test_data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {FLAGS.dataset}")

    test_loader = DataLoader(
        test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    models_to_evaluate = {
        'Selective Pruned Model': selective_pruned_model,
        'Random Pruned Model': random_pruned_model
    }

    for model_name, model in models_to_evaluate.items():
        accuracy, loss, f1, recall, sparsity = evaluate_model(
            model, test_loader)
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {loss:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Sparsity: {sparsity:.4f}")


if __name__ == '__main__':
    app.run(main)
