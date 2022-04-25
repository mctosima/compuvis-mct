from typing import Optional, Tuple

import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class MNISTWL(LightningDataModule):
    def __init__(self, data_dir: str = "data/", batch_size: int = 100, num_workers: int = 4):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(
            root="data", train=True, transform=transforms.ToTensor(), download=True
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transforms.ToTensor()
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )
        return test_loader
