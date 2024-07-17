import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            logging.info(f"Processing batch {i}")

            inputs = inputs.to(device)
            labels = labels.to(device)
            if i == 0:  # Print for the first batch
                print(f"Raw model output shape: {outputs.shape}")
                print(f"Raw model output (first 5 samples, first 10 classes):\n{outputs[:5, :10]}")
            
            try:
                outputs = model(inputs)
                logging.debug(f"Model output type: {type(outputs)}")
                logging.debug(f"Raw outputs: {outputs[:3]}")

                if hasattr(outputs, 'logits'):
                    preds = outputs.logits.argmax(dim=-1)
                else:
                    preds = outputs.argmax(dim=-1)

                logging.debug(f"Predictions: {preds[:10]}")
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                logging.info(f"Processed batch {i}, total predictions: {len(all_preds)}")

            except Exception as e:
                logging.error(f"Error processing batch {i}: {str(e)}")
                continue
            
            if i % 10 == 0:
                print(f"Processed {i+1} batches")

    if not all_preds or not all_labels:
        logging.error("No predictions or labels were generated.")
        raise ValueError("Check the model and input data compatibility.")

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, recall, cm
