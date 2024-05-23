import os
from torch.utils.data import Dataset


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


def ensure_directories_exist():
    required_dirs = ['results', 'results/confusion_matrices']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
