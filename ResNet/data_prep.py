import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, img))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def prepare_and_save_data(test_dir, save_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = CustomImageDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, images in enumerate(test_loader):
        torch.save(images, os.path.join(save_dir, f'batch_{i}.pt'))

# Run this only once to preprocess and save the data
prepare_and_save_data('imagenet-1k/data/test_images', 'preprocessed_test_data')
