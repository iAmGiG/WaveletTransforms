# main.py
import os
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from datasets import load_dataset
from absl import app, flags
from utils import get_model_folders, load_config, load_model
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', './ResNet/SavedModels',
                    'Path to the folder containing pruned models')
flags.DEFINE_string('cache_dir', './ResNet/testing_suite/cache',
                    'Path to the cache directory for datasets')


def load_imagenet(cache_dir):
    """
    Loads the ImageNet validation dataset from the specified cache directory.

    Args:
        cache_dir (str): The path to the cache directory where the dataset is stored.

    Returns:
        DataLoader: A DataLoader object for the ImageNet validation dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(os.path.join(
        cache_dir, 'imagenet-1k/validation'), transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4)
    return dataloader


def evaluate_model(model, dataloader):
    """
    Evaluates the given model using the specified DataLoader and computes various metrics.

    Args:
        model (AutoModelForImageClassification): The model to evaluate.
        dataloader (DataLoader): The DataLoader containing the evaluation dataset.
        feature_extractor (AutoFeatureExtractor): The feature extractor for preprocessing images.

    Returns:
        tuple: A tuple containing accuracy, F1 score, recall, and confusion matrix.
    """
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, recall, cm


def evaluate_pruned_models(models, configs, testloader):
    """
    Evaluates multiple pruned models and their configurations using the specified DataLoader.

    Args:
        models (list): A list of models to evaluate.
        configs (list): A list of configurations corresponding to the models.
        dataloader (DataLoader): The DataLoader containing the evaluation dataset.
        feature_extractor (AutoFeatureExtractor): The feature extractor for preprocessing images.

    Returns:
        list: A list of dictionaries containing the evaluation results for each model.
    """
    results = []
    for model, config in zip(models, configs):
        model.eval()
        accuracy, f1, recall, cm = evaluate_model(model, testloader)
        results.append({
            'config': config,
            'accuracy': accuracy,
            'f1_score': f1,
            'recall': recall,
            'confusion_matrix': cm
        })
    return results


def main(argv):
    """
    Main function to load pruned models, evaluate them, and print the results.

    Args:
        argv (list): Command-line arguments passed to the script.
    """
    base_path = FLAGS.model_path
    cache_dir = FLAGS.cache_dir

    # Ensure dataset is cached
    if not os.path.exists(os.path.join(cache_dir, 'imagenet-1k/validation')):
        print("Dataset not found in cache. Downloading...")
        from download_data import download_and_cache_imagenet
        download_and_cache_imagenet(cache_dir)

    testloader = load_imagenet(cache_dir)

    model_folders = get_model_folders(base_path)
    models = []
    configs = []

    for folder in model_folders:
        try:
            config = load_config(folder)
            model = load_model(folder)
            models.append(model)
            configs.append(config)
        except Exception as e:
            print(f"Failed to load model or config from {folder}: {e}")

    results = evaluate_pruned_models(models, configs, testloader)
    for result in results:
        print(result)


if __name__ == '__main__':
    app.run(main)
