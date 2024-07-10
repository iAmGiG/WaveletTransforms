import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score


def evaluate_model(model, data, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(data):
            if isinstance(batch, torch.Tensor):
                # Print shape of the first few batches
                if i < 5:
                    print(f"Batch {i} shape: {batch.shape}")

                # Assuming labels are included in the last column
                inputs = batch[:, :-1].to(device)
                labels = batch[:, -1].to(device).long()
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            # Print shape of inputs
            if i < 5:
                print(f"Input shape: {inputs.shape}")

            outputs = model(inputs)
            preds = outputs.logits.argmax(
                dim=-1) if hasattr(outputs, 'logits') else outputs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, recall, cm
