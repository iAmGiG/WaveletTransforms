import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForImageClassification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score

# Define the data transformation including resizing, normalization, and augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
validation_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(
    validation_dataset, batch_size=32, shuffle=False)

# Function to load and modify the model for CIFAR-10


def load_model_for_cifar10():
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-18")
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(512, 10)  # CIFAR-10 has 10 classes
    )
    return model

# Function to train the model


def train_model(model, train_loader, validation_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        evaluate_model(model, validation_loader)

# Function to evaluate the model


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions,
                  average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions,
                          average='weighted', zero_division=1)
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")

    return accuracy, f1, recall, cm

# Function to save metrics to PDF


def save_metrics_to_pdf(accuracy, f1, recall, cm):
    fig, ax = plt.subplots(2, 1, figsize=(10, 15))

    cax = ax[0].matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax[0].set_title("Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    metrics = {'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall}
    ax[1].bar(metrics.keys(), metrics.values())
    ax[1].set_ylim([0, 1])
    ax[1].set_title("Evaluation Metrics")

    plt.tight_layout()
    plt.savefig("evaluation_metrics.pdf")

# Function to save metrics to TXT


def save_metrics_to_txt(accuracy, f1, recall):
    with open("evaluation_metrics.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")


def main():
    model = load_model_for_cifar10()

    # Train the model on the CIFAR-10 training dataset
    train_model(model, train_loader, validation_loader)

    # Evaluate the model on the CIFAR-10 validation dataset
    accuracy, f1, recall, cm = evaluate_model(model, validation_loader)

    save_metrics_to_pdf(accuracy, f1, recall, cm)
    save_metrics_to_txt(accuracy, f1, recall)


if __name__ == "__main__":
    main()
