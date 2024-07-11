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
import torchvision.models as models
from imagenet1k.classes import IMAGENET2012_CLASSES

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "../__OGPyTorchModel__", "Path to the model directory")
flags.DEFINE_string("data_path", "imagenet1k/data/test_images", "Path to the ImageNet test data")
flags.DEFINE_integer("batch_size", 32, "Batch size for evaluation")
flags.DEFINE_integer("subset_size", 10, "Subset size for debugging")

def debug_dataset(data_loader, num_samples=5):
    for i, (inputs, labels) in enumerate(data_loader):
        if i >= num_samples:
            break
        print(f"Sample {i}:")
        print(f"Input shape: {inputs.shape}")
        print(f"Label: {labels[0].item()}")
        print(f"Class name: {list(IMAGENET2012_CLASSES.values())[labels[0].item()]}")
        print("\n")

def main(argv):
    # Load pre-trained model from torchvision for comparison
    model = models.resnet18(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Model configuration attributes:")
    for attr in dir(model):
        if not attr.startswith("_"):
            print(f"{attr}: {getattr(model, attr)}")

    # Print data path for verification
    print(f"Data path: {FLAGS.data_path}")

    # Load preprocessed test data using DataLoader with subset for debugging
    try:
        test_loader = prepare_test_dataloader(FLAGS.data_path, FLAGS.batch_size, subset_size=FLAGS.subset_size)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Debug the dataset
    debug_dataset(test_loader)
    
    # Evaluate model on the subset
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
