"""
Author: Camila Cristiano-Romero
Contact: ccristiano@bcamath.org / ccristiano001@ikasle.ehu.eus
License: MIT License
"""

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
