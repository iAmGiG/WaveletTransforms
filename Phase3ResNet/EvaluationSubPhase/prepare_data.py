from datasets import load_from_disk
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Load the dataset from disk
dataset = load_from_disk('imagenet-1k-dataset')

# Load the processor for the ResNet model
processor = AutoImageProcessor.from_pretrained('microsoft/resnet-18')

# Define transformations
transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Apply transformations


def preprocess(example):
    example['pixel_values'] = [transform(Image.open(
        image).convert('RGB')) for image in example['image']]
    return example


test_dataset = dataset['test'].map(
    preprocess, batched=True, remove_columns=['image'])

# Custom Dataset class


class ImageNetDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['pixel_values']
        label = item['label']
        return image, label


# Create DataLoader
test_loader = DataLoader(ImageNetDataset(test_dataset),
                         batch_size=32, shuffle=False)

# Save DataLoader
torch.save(test_loader, 'test_loader.pth')
