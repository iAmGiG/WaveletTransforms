import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score


def evaluate_model(model, data, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(data):
            if isinstance(batch, torch.Tensor):
                # Check and adjust the channel dimension if necessary
                if batch.size(1) != 3:  # Assuming the channel dimension is the second one
                    print(f"Batch {i} has incorrect channel dimension: {batch.size(1)}, expected 3")
                    continue  # Skip this batch or you could implement a method to correct it
                
                inputs = batch.to(device)
                # Assuming labels are at the last index if batch includes labels
                if batch.size(-1) == 2:  # You might need to adjust this based on your label setup
                    labels = batch[:, -1].long()
                    inputs = batch[:, :-1]
                else:
                    labels = None  # Adjust according to how labels are provided

            else:
                raise ValueError(f"Unexpected batch format at index {i}: {type(batch)}")

            try:
                outputs = model(inputs)
                preds = outputs.logits.argmax(dim=-1) if hasattr(outputs, 'logits') else outputs.argmax(dim=-1)
                if labels is not None:
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
