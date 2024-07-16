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
from setup_test_dataloader import prepare_validation_dataloader
import torchvision.models as models
from imagenet1k.classes import IMAGENET2012_CLASSES

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "../__OGPyTorchModel__",
                    "Path to the model directory")
flags.DEFINE_string("data_path", "imagenet1k/data/val_images",
                    "Path to the ImageNet test data")
flags.DEFINE_integer("batch_size", 32, "Batch size for evaluation")
flags.DEFINE_integer("subset_size", 10, "Subset size for debugging")


def debug_dataset(data_loader, dataset, num_samples=5):
    for i, (inputs, labels) in enumerate(data_loader):
        if i >= num_samples:
            break
        print(f"Sample {i}:")
        print(f"Input shape: {inputs.shape}")
        print(f"Label index: {labels[0].item()}")
        print(f"Class name: {dataset.get_class_name(labels[0].item())}")
        print("\n")


def main(argv):
    print("Starting main function")

    # Load pre-trained model
    print("Loading pre-trained ResNet18 model")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Model loaded and moved to device: {device}")

    # Load validation data using DataLoader
    print(f"Preparing validation data loader from path: {FLAGS.data_path}")
    val_loader, val_dataset = prepare_validation_dataloader(
        FLAGS.data_path, FLAGS.batch_size, subset_size=FLAGS.subset_size)

    print(f"Validation dataset size: {len(val_dataset)}")

    # Debug the dataset
    print("Debugging dataset:")
    debug_dataset(val_loader, val_dataset)

    # Evaluate model on the validation set
    try:
        print("Starting model evaluation")
        accuracy, f1, recall, cm = evaluate_model(model, val_loader, device)

        print(f"Evaluation complete. Metrics:")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"Recall: {recall}")

        # Save metrics to CSV
        metrics_df = pd.DataFrame(
            {'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall]})
        metrics_path = os.path.join(FLAGS.model_path, 'validation_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")

        # Check and log confusion matrix
        print(f"Confusion Matrix shape: {cm.shape}")

        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        cm_path = os.path.join(FLAGS.model_path, 'confusion_matrix.pdf')
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()  # This will print the full stack trace

    print("Main function completed successfully")


if __name__ == "__main__":
    app.run(main)
