import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file as load_safetensors
from datasets import load_dataset
from torch.utils.data import DataLoader


def load_model(model_dir, model_type='safetensors'):
    """
    Load a model from the specified directory containing config and safetensors files.

    Args:
        model_dir (str): Path to the directory containing the model files.
        model_type (str): Type of model file ('safetensors' or 'bin').

    Returns:
        torch.nn.Module: Loaded model.
    """
    config_path = os.path.join(model_dir, 'config.json')
    config = AutoConfig.from_pretrained(config_path, local_files_only=True)
    model = AutoModelForImageClassification.from_config(config)
    model_path = os.path.join(model_dir, f'model.{model_type}')

    if model_type == 'safetensors':
        model_weights = load_safetensors(model_path)
    elif model_type == 'bin':
        model_weights = torch.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_type}")

    model.load_state_dict(model_weights)
    return model


def evaluate_model(model, dataloader):
    """
    Evaluate the model on the given dataloader.

    Args:
        model (torch.nn.Module): Model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.

    Returns:
        float: Evaluation metric (e.g., accuracy).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values']
            labels = batch['labels']
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main():
    # Paths to model directories
    original_model_dir = r'Phase3ResNet\__OGPyTorchModel__'  # Adjust path as needed
    # Example path for pruned model
    pruned_model_dir = r'Phase3ResNet\SavedModels\{wavelet}_threshold-{threshold}_level-{level}_guid-{guid[:4]}\selective_pruned'
    # Adjust path as needed
    dataset_dir = r'Phase3ResNet\EvaluationSubPhase_Beta\imagenet-1k-dataset'

    # Load models
    original_model = load_model(original_model_dir, 'bin')
    pruned_model = load_model(pruned_model_dir, 'safetensors')

    # Prepare the dataset
    dataset = load_dataset(
        "imagefolder", data_dir=dataset_dir, split="validation")
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

    def preprocess(batch):
        return image_processor(batch['image'], return_tensors='pt')

    dataset = dataset.map(preprocess, batched=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Evaluate models
    original_accuracy = evaluate_model(original_model, dataloader)
    pruned_accuracy = evaluate_model(pruned_model, dataloader)

    print(f"Original Model Accuracy: {original_accuracy}")
    print(f"Pruned Model Accuracy: {pruned_accuracy}")


if __name__ == "__main__":
    main()
