import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
import logging
from torch.nn import CrossEntropyLoss


def calculate_sparsity(model, threshold=1e-6):
    """Calculate the sparsity of the model."""
    total_params = 0
    near_zero_params = 0
    for param in model.parameters():
        if param.dim() > 1:  # Only consider weights, not biases
            total_params += param.numel()
            near_zero_params += torch.sum(torch.abs(param) < threshold).item()
    
    if total_params == 0:
        return 0.0  # Avoid division by zero
    
    sparsity = near_zero_params / total_params
    return sparsity


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    criterion = CrossEntropyLoss()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            logging.info(f"Processing batch {i}")
            inputs = inputs.to(device)
            labels = labels.to(device)

            try:
                outputs = model(inputs)
                logging.debug(f"Model output type: {type(outputs)}")

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                logging.debug(f"Model logits shape: {logits.shape}")
                preds = logits.argmax(dim=-1)

                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1

                logging.debug(f"Predictions shape: {preds.shape}")
                logging.debug(f"Labels shape: {labels.shape}")

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                logging.info(
                    f"Processed batch {i}, total predictions: {len(all_preds)}")

                if i % 10 == 0:
                    logging.debug(f"Sample predictions: {preds[:10]}")
                    logging.debug(f"Sample true labels: {labels[:10]}")

            except Exception as e:
                logging.error(f"Error processing batch {i}: {str(e)}")
                raise  # Re-raise the exception to see the full traceback

    if not all_preds or not all_labels:
        logging.error("No predictions or labels were generated.")
        raise ValueError("Check the model and input data compatibility.")

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds,
                          average='weighted', zero_division=1)

    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    # Calculate sparsity
    sparsity = calculate_sparsity(model)
    
    logging.info(f"Model sparsity: {sparsity:.6f}")
    
    # Log more details about the model structure
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")
    
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Only log details for weight matrices
            zeros = torch.sum(torch.abs(param) < 1e-6).item()
            total = param.numel()
            logging.info(f"Layer {name}: {zeros}/{total} near-zero parameters")

    return accuracy, f1, recall, avg_loss, sparsity 