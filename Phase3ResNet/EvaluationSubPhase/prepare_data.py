from datasets import load_from_disk
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
from torch.utils.data import DataLoader
from PIL import Image
from utility import ImageNetDataset

# Load the small test dataset from disk
dataset = load_from_disk('imagenet-1k-small-test')

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
    # example['image'] is already a PIL Image
    image = example['image'].convert('RGB')
    example['pixel_values'] = transform(image)
    return example


# Apply preprocessing to the test dataset
test_dataset = dataset.map(preprocess, batched=False, remove_columns=['image'])

# Create DataLoader
test_loader = DataLoader(ImageNetDataset(test_dataset),
                         batch_size=32, shuffle=False)

# Save DataLoader
torch.save(test_loader, 'test_loader.pth')
