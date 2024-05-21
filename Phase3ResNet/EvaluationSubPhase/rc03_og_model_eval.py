from transformers import AutoModelForImageClassification, AutoImageProcessor, DefaultDataCollator
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the original model
model_name = 'microsoft/resnet-18'
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Prepare test data
dataset = load_dataset('imagenet-1k', split='test')


def transform(example):
    example['pixel_values'] = processor(
        images=example['image'], return_tensors='pt')['pixel_values']
    example['label'] = example['label']
    return example


test_dataset = dataset.map(transform, batched=True)
test_loader = DataLoader(test_dataset, batch_size=32,
                         collate_fn=DefaultDataCollator(return_tensors='pt'))

# Evaluate the model


def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['pixel_values'].squeeze().to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = batch['label'].to(torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, f1, recall, conf_matrix


accuracy, f1, recall, conf_matrix = evaluate_model(model, test_loader)

# Save results
results = {
    'Model': 'Original',
    'Accuracy': accuracy,
    'F1 Score': f1,
    'Recall': recall
}

results_df = pd.DataFrame([results])
results_df.to_csv('results_log.csv', index=False)

# Save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix - Original Model')
plt.savefig('confusion_matrix_original.pdf')
