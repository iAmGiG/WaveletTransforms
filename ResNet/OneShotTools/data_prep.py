from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_test_dataloader(test_dir, batch_size=32):
    """
    Prepares a DataLoader for the ImageNet test dataset.

    Args:
        test_dir (str): The directory containing the test images.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        DataLoader: A DataLoader for the test dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

# Prepare DataLoader for the extracted test images
test_loader = prepare_test_dataloader('./imagenet-1k/data/test_images')
