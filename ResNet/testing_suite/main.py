# main.py
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from absl import app, flags
from utils import get_model_folders, load_config, load_model

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', './SavedModels', 'Path to the folder containing pruned models')
flags.DEFINE_string('cache_dir', './cache', 'Path to the cache directory for datasets')

def load_imagenet(cache_dir):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(os.path.join(cache_dir, 'imagenet-1k/validation'), transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    return dataloader

def evaluate_model(model, dataloader):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

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
