import torch
from torch.utils.data import DataLoader
from utils import load_model, setup_logging, load_preprocessed_batches
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import numpy as np
from absl import app, flags
import os
from eval_model import evaluate_model

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "../__OGPyTorchModel__",
                    "Path to the model directory")
flags.DEFINE_string("data_path", "../preprocessed_test_data",
                    "Path to the preprocessed test data")
flags.DEFINE_integer("batch_size", 32, "Batch size for evaluation")


def main(argv):
    # Load model
    model = load_model(FLAGS.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load preprocessed test data
    preprocessed_batches, preprocessed_labels = load_preprocessed_batches(
        FLAGS.data_path)

    # Check if data is loaded correctly
    if not preprocessed_batches:
        raise ValueError(
            "No preprocessed batches loaded. Check the data path and files.")
    print(
        f"Model expected input shape: {model.config.num_channels}, {model.config.image_size}, {model.config.image_size}")
    # Evaluate model
    if not preprocessed_labels:
        # If no separate labels were found, assume they're included in the batches
        accuracy, f1, recall, cm = evaluate_model(
            model, preprocessed_batches, device)
    else:
        # If separate labels were found, pass both batches and labels
        accuracy, f1, recall, cm = evaluate_model(model, zip(
            preprocessed_batches, preprocessed_labels), device)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(
        {'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall]})
    metrics_df.to_csv(os.path.join(FLAGS.model_path,
                      'test_metrics.csv'), index=False)

    # Check and log confusion matrix
    print(f"Confusion Matrix:\n{cm}")
    if cm.size == 0:
        raise ValueError(
            "Confusion matrix is empty. Check the model predictions and labels.")

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    # Set annot=False for ImageNet due to large number of classes
    sns.heatmap(cm, annot=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(FLAGS.model_path, 'confusion_matrix.pdf'))


if __name__ == "__main__":
    app.run(main)
