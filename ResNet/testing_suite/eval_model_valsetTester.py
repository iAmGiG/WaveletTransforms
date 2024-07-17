import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from imagenet1k.classes import IMAGENET2012_CLASSES
from setup_test_dataloader import prepare_validation_dataloader
import torchvision.models as models

def evaluate_model(model, data_loader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            print(f"Processing batch {i}")

            inputs = inputs.to(device)
            labels = labels.to(device)

            try:
                outputs = model(inputs)
                print(f"Model output type: {type(outputs)}")
                
                # Print raw outputs for first 3 examples
                if i == 0:
                    print(f"Raw outputs (first 3 examples): {outputs[:3]}")

                if hasattr(outputs, 'logits'):
                    preds = outputs.logits.argmax(dim=-1)
                else:
                    preds = outputs.argmax(dim=-1)

                # Print first 10 predictions and actual labels
                if i == 0:
                    print(f"Predictions (first 10): {preds[:10]}")
                    print(f"Actual labels (first 10): {labels[:10]}")
                    if class_names:
                        pred_classes = [class_names[p] for p in preds[:10].cpu().numpy()]
                        actual_classes = [class_names[l] for l in labels[:10].cpu().numpy()]
                        print(f"Predicted class names (first 10): {pred_classes}")
                        print(f"Actual class names (first 10): {actual_classes}")

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                print(f"Processed batch {i}, total predictions: {len(all_preds)}")

            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue

    if not all_preds or not all_labels:
        raise ValueError("No predictions or labels were generated. Check the model and input data compatibility.")

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix: \n{cm}")

    return accuracy, f1, recall, cm

# Example usage
if __name__ == "__main__":
    # Load your model, data_loader, and class_names here
    model = models.resnet18(pretrained=True)  # Ensure correct loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Assuming class_names is a list of class names in order of indices
    class_names = list(IMAGENET2012_CLASSES.values())

    # Your data_loader should be prepared with the validation dataset
    val_loader = prepare_validation_dataloader(val_dir='imagenet1k/data/val_images')

    accuracy, f1, recall, cm = evaluate_model(model, val_loader, device, class_names=class_names)
