import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch import optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from transformers import AutoModelForImageClassification
import random

# Define the data transformation including resizing and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
validation_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

# Use a smaller subset for evaluation
subset_size = 1000
subset_indices = random.sample(range(len(validation_dataset)), subset_size)
validation_subset = Subset(validation_dataset, subset_indices)
validation_loader = DataLoader(validation_subset, batch_size=32, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


def load_model_for_cifar10():
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-18")
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(512, 10)  # CIFAR-10 has 10 classes
    )
    return model


def fine_tune_model(model, train_loader, epochs=3):
    device = torch.device("cpu")  # Use CPU
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")


def evaluate_model(model, dataloader):
    model.eval()
    device = torch.device("cpu")  # Use CPU
    model.to(device)

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

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


def main():
    model = load_model_for_cifar10()
    fine_tune_model(model, train_loader, epochs=3)

    # Evaluate the model on the CIFAR-10 validation subset
    accuracy, f1, recall, cm = evaluate_model(model, validation_loader)


if __name__ == "__main__":
    main()
