import random
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit

class DataModule(LightningDataModule):
    def __init__(self, train_dir, val_dir, transform, batch_size=32, num_workers=4, seed=42):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        ds_train = datasets.ImageFolder(self.train_dir, transform=self.transform)
        ds_val   = datasets.ImageFolder(self.val_dir,   transform=self.transform)
        dataset  = ConcatDataset([ds_train, ds_val])

        targets = []
        for ds in dataset.datasets:
            targets.extend([label for _, label in ds.samples])

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=self.seed)
        idx = list(range(len(targets)))
        train_idx, val_idx = next(splitter.split(idx, targets))

        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset   = Subset(dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
