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
from setup_test_dataloader import prepare_test_dataloader

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "../__OGPyTorchModel__",
                    "Path to the model directory")
flags.DEFINE_string("data_path", "imagenet1k/data/test_images",
                    "Path to the preprocessed test data")
flags.DEFINE_integer("batch_size", 32, "Batch size for evaluation")


def main(argv):
    # Load model
    model = load_model(FLAGS.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Model configuration attributes:")
    for attr in dir(model.config):
        if not attr.startswith("_"):
            print(f"{attr}: {getattr(model.config, attr)}")

    # Load preprocessed test data using DataLoader
    test_loader = prepare_test_dataloader(FLAGS.data_path, FLAGS.batch_size)
    
    # Check if DataLoader is correctly prepared
    if not test_loader:
        raise ValueError("Test DataLoader is not correctly prepared. Check the data path and files.")
    
    # Evaluate model
    try:
        accuracy, f1, recall, cm = evaluate_model(model, test_loader, device)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall]})
        metrics_df.to_csv(os.path.join(FLAGS.model_path, 'test_metrics.csv'), index=False)
        
        # Check and log confusion matrix
        print(f"Confusion Matrix:\n{cm}")
        if cm.size == 0:
            raise ValueError("Confusion matrix is empty. Check the model predictions and labels.")
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False)  # Set annot=False for ImageNet due to large number of classes
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(FLAGS.model_path, 'confusion_matrix.pdf'))
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    app.run(main)
