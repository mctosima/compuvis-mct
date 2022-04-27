from typing import Optional

import torchvision
import torchvision.datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class Flowers102(LightningDataModule):
    def __init__(
        self,
        datadir,
        batch_size=4,
        num_workers=8,
        pin_memory=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((227, 227)),
            ]
        )

        self.data_train = None
        self.data_test = None
        self.data_val = None

    @property
    def num_classes(self):
        return 102

    def prepare_data(self):
        # torchvision.datasets.Flowers102(self.hparams.datadir,split='train',download=True)
        # torchvision.datasets.Flowers102(self.hparams.datadir,split='test',download=True)
        # torchvision.datasets.Flowers102(self.hparams.datadir,split='val',download=True)
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_test and not self.data_val:
            self.data_train = torchvision.datasets.Flowers102(
                self.hparams.datadir,
                split="train",
                transform=self.transformation,
            )
            self.data_test = torchvision.datasets.Flowers102(
                self.hparams.datadir,
                split="test",
                transform=self.transformation,
            )
            self.data_val = torchvision.datasets.Flowers102(
                self.hparams.datadir,
                split="val",
                transform=self.transformation,
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        return test_loader
