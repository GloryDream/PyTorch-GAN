import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CelebaDataset40(Dataset):
    def __init__(self, root, transform=None):
        self.files = []
        labels = []
        self.transform = transform
        f = open(os.path.join(root, 'lists_40id.txt'), 'r')
        for line in f:
            self.files.append(os.path.join(root, line.strip()))
        f1 = open(os.path.join(root, 'labels_40id.txt'), 'r')
        for line in f1:
            labels.append(int(line.strip())-1)
        self.labels = labels

    def __getitem__(self, index):
        label = self.labels[index]
        img = Image.open(self.files[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.files)
