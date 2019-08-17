from torch import utils
from PIL import Image
import numpy as np
import json


class ImageDataset(utils.data.Dataset):

    def __init__(self, root, meta, transform=None):
        self.root = root
        self.transform = transform
        with open(meta) as file:
            self.meta = list(json.load(file).items())

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        path, label = self.meta[idx]
        path = f'{self.root}/{path}'
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
