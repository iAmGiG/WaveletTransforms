import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score


def evaluate_model(model, data, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data):
            print(f"Processing batch {i}")

            inputs = inputs.to(device)
            labels = labels.to(device)

            try:
                outputs = model(inputs)
                print(f"Model output type: {type(outputs)}")
                # Print raw outputs for first 3 examples
                print(f"Raw outputs: {outputs[:3]}")

                if hasattr(outputs, 'logits'):
                    preds = outputs.logits.argmax(dim=-1)
                else:
                    preds = outputs.argmax(dim=-1)

                # Print first 10 predictions
                print(f"Predictions: {preds[:10]}")

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                print(
                    f"Processed batch {i}, total predictions: {len(all_preds)}")

            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue

    if not all_preds or not all_labels:
        raise ValueError(
            "No predictions or labels were generated. Check the model and input data compatibility.")

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, recall, cm
