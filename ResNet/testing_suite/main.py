import torch
from torch.utils.data import DataLoader
from utils import load_model, setup_logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import numpy as np
from absl import app, flags
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "../__OGPyTorchModel__", "Path to the model directory")
flags.DEFINE_string("data_path", "../preprocessed_test_data", "Path to the preprocessed test data")
flags.DEFINE_integer("batch_size", 32, "Batch size for evaluation")

def load_preprocessed_batches(data_path):
    batches = []
    for file in sorted(os.listdir(data_path)):
        if file.startswith("batch_") and file.endswith(".pt"):
            batch_path = os.path.join(data_path, file)
            batch = torch.load(batch_path)
            batches.append(batch)
    return batches

def evaluate_model(model, preprocessed_batches, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in preprocessed_batches:
            # Check if the batch is a dictionary
            if isinstance(batch, dict):
                inputs = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
            # If it's a tuple or list, assume the first two elements are inputs and labels
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            outputs = model(inputs)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, f1, recall, cm

def main(argv):
    # Setup logging
    setup_logging(FLAGS.model_path)
    
    # Load model
    model = load_model(FLAGS.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load preprocessed test data
    preprocessed_batches = load_preprocessed_batches(FLAGS.data_path)
    
    # Evaluate model
    accuracy, f1, recall, cm = evaluate_model(model, preprocessed_batches, device)
    
    # Log results
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall]})
    metrics_df.to_csv(os.path.join(FLAGS.model_path, 'test_metrics.csv'), index=False)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False)  # Set annot=False for ImageNet due to large number of classes
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(FLAGS.model_path, 'confusion_matrix.pdf'))

if __name__ == "__main__":
    app.run(main)