import torch
import tensorflow as tf
from transformers import AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', '', 'Path to the model')


def evaluate_model(model, data_loader):
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
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
        f'results/confusion_matrices/confusion_matrix_{model_name}.pdf')


def evaluate_and_save(model, model_name, data_loader, results_df):
    accuracy, f1, recall, conf_matrix = evaluate_model(model, data_loader)

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

    test_loader = torch.load('test_loader.pth')
    results_df = pd.read_csv('results/results_log.csv') if os.path.exists(
        'results/results_log.csv') else pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score', 'Recall', 'Sparsity'])

    model_path = FLAGS.model_path
    if model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
        model_name = os.path.basename(model_path)
        accuracy, f1, recall, conf_matrix = evaluate_model(model, test_loader)

        # Save results
        results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Recall': recall
        }

        results_df = results_df.append(results, ignore_index=True)
        results_df.to_csv('results/results_log.csv', index=False)

        # Save confusion matrix
        save_confusion_matrix(conf_matrix, model_name)
    else:
        model = torch.load(model_path)
        model_name = os.path.basename(model_path)
        results_df = evaluate_and_save(
            model, model_name, test_loader, results_df)


if __name__ == '__main__':
    app.run(main)
