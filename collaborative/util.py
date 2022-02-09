import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import numpy as np



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class DatasetFormat(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label.numpy().tolist()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        image, label = self.dataset[item], self.label[item]
        return image, label