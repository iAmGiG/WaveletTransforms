import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import logging
from imagenet1k.classes import IMAGENET2012_CLASSES

# Create a reverse mapping from WordNet IDs to class indices
wnid_to_class_idx = {v: k for k, v in IMAGENET2012_CLASSES.items()}


class ImageNetFlatDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='validation'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(
            root_dir) if fname.endswith('.JPEG')]
        self.wnid_to_class_idx = {
            k: i for i, (k, v) in enumerate(IMAGENET2012_CLASSES.items())}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.split != 'test':
            # Extract synset ID from filename
            filename = os.path.basename(img_path)
            root, _ = os.path.splitext(filename)
            _, synset_id = root.rsplit("_", 1)
            label = self.wnid_to_class_idx[synset_id]
        else:
            label = -1  # For test set, we don't have labels

        return image, label

    def validate_dataset(self):
        valid_samples = 0
        invalid_samples = 0
        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            root, _ = os.path.splitext(filename)
            try:
                _, synset_id = root.rsplit("_", 1)
                if synset_id in self.wnid_to_class_idx:
                    valid_samples += 1
                else:
                    invalid_samples += 1
                    logging.warning(
                        f"Invalid synset ID found in file: {img_path}")
            except ValueError:
                invalid_samples += 1
                logging.warning(f"Unexpected filename format: {img_path}")

        logging.info(f"Valid samples: {valid_samples}")
        logging.info(f"Invalid samples: {invalid_samples}")
        return valid_samples, invalid_samples


def sanity_check(dataset, num_samples=5):
    for i in range(num_samples):
        image, label = dataset[i]
        synset_id = list(IMAGENET2012_CLASSES.keys())[label]
        class_name = IMAGENET2012_CLASSES[synset_id]
        logging.info(f"Sample {i}:")
        logging.info(f"  Image shape: {image.shape}")
        logging.info(f"  Label index: {label}")
        logging.info(f"  Synset ID: {synset_id}")
        logging.info(f"  Class name: {class_name}")


def prepare_validation_dataloader(val_dir, batch_size=32, subset_size=None):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    val_dataset = ImageNetFlatDataset(root_dir=val_dir, transform=transform)

    # Validate dataset
    valid_samples, invalid_samples = val_dataset.validate_dataset()
    if invalid_samples > 0:
        logging.warning(
            f"Found {invalid_samples} invalid samples in the dataset")

    # Perform sanity check
    sanity_check(val_dataset)

    if subset_size and subset_size < len(val_dataset):
        val_subset = torch.utils.data.Subset(val_dataset, range(subset_size))
    else:
        val_subset = val_dataset

    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    return val_loader, val_dataset
