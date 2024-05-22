import torch
import tensorflow as tf
from transformers import AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from absl import app, flags
from utility import ImageNetDataset

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', '', 'Path to the model')


def create_model_from_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = tf.keras.Sequential()
    for layer_config in config['layers']:
        layer_type = getattr(tf.keras.layers, layer_config['class_name'])
        layer = layer_type(**layer_config['config'])
        model.add(layer)

    return model


def evaluate_model_tf(model, data_loader):
    y_true = []
    y_pred = []

    for images, labels in data_loader:
        outputs = model.predict(images)
        preds = tf.argmax(outputs, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, f1, recall, conf_matrix


def evaluate_model_torch(model, data_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, f1, recall, conf_matrix


def save_confusion_matrix(conf_matrix, model_name):
    os.makedirs('results/confusion_matrices', exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(
        f'results/confusion_matrices/confusion_matrix_{model_name}.png')


def evaluate_and_save_tf(model, model_name, data_loader, results_df):
    accuracy, f1, recall, conf_matrix = evaluate_model_tf(model, data_loader)

    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Recall': recall,
    }

    results_df = results_df.append(results, ignore_index=True)
    results_df.to_csv('results/results_log.csv', index=False)

    save_confusion_matrix(conf_matrix, model_name)
    return results_df


def evaluate_and_save_torch(model, model_name, data_loader, results_df):
    accuracy, f1, recall, conf_matrix = evaluate_model_torch(
        model, data_loader)

    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum(p.nonzero().size(0) for p in model.parameters())
    sparsity = 1 - nonzero_params / total_params

    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Recall': recall,
        'Sparsity': sparsity
    }

    results_df = results_df.append(results, ignore_index=True)
    results_df.to_csv('results/results_log.csv', index=False)

    save_confusion_matrix(conf_matrix, model_name)
    return results_df


def main(argv):
    del argv  # Unused

    if not os.path.exists('test_loader.pth'):
        raise FileNotFoundError(
            "The file 'test_loader.pth' was not found. Please ensure that 'prepare_data.py' has been run successfully.")

    test_loader = torch.load('test_loader.pth')
    results_df = pd.read_csv('results/results_log.csv') if os.path.exists(
        'results/results_log.csv') else pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score', 'Recall', 'Sparsity'])

    model_path = FLAGS.model_path
    model_name = os.path.basename(model_path)

    if model_path.endswith('.h5'):
        try:
            model = tf.keras.models.load_model(model_path)
        except ValueError:
            config_path = os.path.join(
                os.path.dirname(model_path), 'config.json')
            if os.path.exists(config_path):
                model = create_model_from_config(config_path)
                model.load_weights(model_path)
            else:
                raise ValueError(
                    "Model config not found in the file and no config.json present.")
        results_df = evaluate_and_save_tf(
            model, model_name, test_loader, results_df)
    else:
        model = torch.load(model_path)
        results_df = evaluate_and_save_torch(
            model, model_name, test_loader, results_df)


if __name__ == '__main__':
    app.run(main)
