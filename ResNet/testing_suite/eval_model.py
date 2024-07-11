import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score

def evaluate_model(model, data, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data):
            if isinstance(inputs, torch.Tensor):
                # Check and adjust the channel dimension if necessary
                if inputs.size(1) != 3:  # Assuming the channel dimension is the second one
                    print(f"Batch {i} has incorrect channel dimension: {inputs.size(1)}, expected 3")
                    continue  # Skip this batch or you could implement a method to correct it

                inputs = inputs.to(device)
                if labels is not None and isinstance(labels, torch.Tensor):
                    labels = labels.to(device).long()
                else:
                    labels = None  # Adjust according to how labels are provided

            else:
                raise ValueError(f"Unexpected batch format at index {i}: {type(inputs)}")

            try:
                outputs = model(inputs)
                preds = outputs.logits.argmax(dim=-1) if hasattr(outputs, 'logits') else outputs.argmax(dim=-1)
                if labels is not None and labels.min() >= 0:  # Only consider non-dummy labels
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue

    if not all_preds or not all_labels:
        raise ValueError("No predictions or labels were generated. Check the model and input data compatibility.")

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, recall, cm
