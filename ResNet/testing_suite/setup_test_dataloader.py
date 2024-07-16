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
import torch
import re


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
            # Convert grayscale to RGB by repeating the channels
            image = image.repeat(3, 1, 1)
        image = to_pil_image(image)  # Convert tensor to PIL Image
        if self.transform:
            image = self.transform(image)
        return image, -1  # Dummy label


class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.class_to_idx = {v: k for k, v in IMAGENET2012_CLASSES.items()}
        self.samples = self.make_dataset(self.root)

    def make_dataset(self, directory):
        instances = []
        directory = Path(directory)
        print(f"Searching for images in: {directory}")

        # Check if the directory exists
        if not directory.exists():
            print(f"Directory does not exist: {directory}")
            return instances

        for image_path in directory.glob('*.JPEG'):
            print(f"Found image: {image_path}")
            # Extract the class ID from the filename (assuming ILSVRC2012_test_00000001.JPEG format)
            # Adjust this to correctly extract the ID part matching with IMAGENET2012_CLASSES keys
            filename = image_path.stem.split('_')[-1]  # Get the ID part
            # Assuming the filename ends with a numeric ID
            class_id = int(filename)
            if class_id in self.class_to_idx:
                label = self.class_to_idx[class_id]
                instances.append((str(image_path), label))
            else:
                print(f"Warning: No class found for image {image_path.name}")

        print(f"Total images found: {len(instances)}")
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = read_image(path)
        if image.shape[0] == 1:  # Check if the image is grayscale
            # Convert grayscale to RGB by repeating the channels
            image = image.repeat(3, 1, 1)
        image = to_pil_image(image)  # Convert tensor to PIL Image
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageNetValidationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.wnid_to_idx = {wnid: idx for idx,
                            (wnid, _) in enumerate(IMAGENET2012_CLASSES.items())}
        self.samples = self.make_dataset()

    def make_dataset(self):
        instances = []
        wnid_pattern = re.compile(r'n\d+')

        for img_path in self.root.glob('*.JPEG'):
            wnid_match = wnid_pattern.search(img_path.name)
            if wnid_match:
                wnid = wnid_match.group()
                if wnid in self.wnid_to_idx:
                    instances.append((str(img_path), self.wnid_to_idx[wnid]))
                else:
                    print(f"Warning: No class found for image {img_path.name}")
            else:
                print(f"Warning: Could not extract wnid from {img_path.name}")

        print(f"Total images found: {len(instances)}")
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = read_image(path)
        if image.shape[0] == 1:  # Check if the image is grayscale
            # Convert grayscale to RGB by repeating the channels
            image = image.repeat(3, 1, 1)
        image = to_pil_image(image)  # Convert tensor to PIL Image
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_name(self, idx):
        for wnid, (_, class_name) in zip(self.wnid_to_idx.keys(), IMAGENET2012_CLASSES.items()):
            if self.wnid_to_idx[wnid] == idx:
                return class_name
        return "Unknown"


def prepare_test_dataloader(test_dir, batch_size=32, model_preprocess=None, subset_size=None):
    if model_preprocess:
        transform = model_preprocess
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    test_dataset = ImageNetDataset(test_dir, transform=transform)

    # Print length of the dataset
    total_samples = len(test_dataset)
    print(f"Total number of samples in the dataset: {total_samples}")

    if subset_size:
        # Ensure subset size does not exceed dataset length
        if subset_size > total_samples:
            print(
                f"Requested subset size {subset_size} exceeds total samples {total_samples}. Using total samples instead.")
            subset_size = total_samples
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(subset_size))

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Debug prints to verify data and labels
    print(
        f"Number of samples in the test dataset (subset if applied): {len(test_dataset)}")
    if len(test_dataset) == 0:
        raise ValueError(
            "Test dataset is empty. Check the data path and contents.")

    for i, (images, labels) in enumerate(test_loader):
        print(f"Batch {i} - Number of images: {images.size(0)}")
        for j in range(min(3, images.size(0))):
            img = transforms.ToPILImage()(images[j]).convert("RGB")
            # Save the image to a file to check later
            img.save(f"/tmp/sample_{i}_{j}.png")
            class_name = list(IMAGENET2012_CLASSES.values())[labels[j].item()]
            print(f"Label: {labels[j].item()} - Class: {class_name}")
        if i >= 1:  # Print details for only the first batch
            break

    return test_loader


def prepare_validation_dataloader(val_dir, batch_size=32, model_preprocess=None, subset_size=None):
    if model_preprocess:
        transform = model_preprocess
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    val_dataset = ImageNetValidationDataset(val_dir, transform=transform)

    if subset_size and subset_size < len(val_dataset):
        val_subset = torch.utils.data.Subset(val_dataset, range(subset_size))
    else:
        val_subset = val_dataset

    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    return val_loader, val_dataset

# Test usage
# if __name__ == "__main__":
#     test_loader = prepare_test_dataloader('imagenet1k/data/test_images')
#     for images, labels in test_loader:
#         print(f"Batch size: {images.size(0)}")
#         break
