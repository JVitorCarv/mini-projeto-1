# dataset.py
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import os


def seed_worker(worker_id):
    seed = 123 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class CatsDogsDataset(Dataset):
    def __init__(self, dataframe, root_path, transform=None):
        self.df = dataframe
        self.root = root_path
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.root, row["image"])
        image = Image.open(image_path).convert("RGB")
        label = int(row["labels"])
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.df)
