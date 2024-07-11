import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from imagenet1k.classes import IMAGENET2012_CLASSES
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import DataLoader
from pathlib import Path


class UnlabeledDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self.make_dataset(self.root)

    def make_dataset(self, directory):
        instances = []
        directory = Path(directory)
        for target in sorted(directory.rglob('*')):
            if target.is_file() and target.suffix.lower() in (".jpeg", ".jpg", ".JPEG", ".JPG"):
                instances.append(str(target))
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        image = read_image(path)
        if image.shape[0] == 1:  # Check if the image is grayscale
            image = image.repeat(3, 1, 1)  # Convert grayscale to RGB by repeating the channels
        image = to_pil_image(image)  # Convert tensor to PIL Image
        if self.transform:
            image = self.transform(image)
        return image, -1  # Dummy label

def prepare_test_dataloader(test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = UnlabeledDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

# Example usage
if __name__ == "__main__":
    test_loader = prepare_test_dataloader('imagenet1k/data/test_images')
    for images, labels in test_loader:
        print(f"Batch size: {images.size(0)}")
        break