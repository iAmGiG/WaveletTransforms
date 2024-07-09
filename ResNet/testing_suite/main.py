import torch
from torch.utils.data import DataLoader
from transformers import ResNetImageProcessor
from datasets import load_dataset
from utils import load_model, setup_logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", None, "Path to the model directory")
flags.DEFINE_string("data_path", None, "Path to the ImageNet-1k dataset")
flags.DEFINE_integer("batch_size", 32, "Batch size for evaluation")

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
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
    
    # Prepare dataset
    dataset = load_dataset("imagenet-1k", split="validation", cache_dir=FLAGS.data_path)
    processor = ResNetImageProcessor.from_pretrained("microsoft/resnet-18")
    
    def preprocess_function(examples):
        return processor(examples["image"], return_tensors="pt")
    
    processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    dataloader = DataLoader(processed_dataset, batch_size=FLAGS.batch_size, shuffle=False)
    
    # Evaluate model
    accuracy, f1, recall, cm = evaluate_model(model, dataloader, device)
    
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