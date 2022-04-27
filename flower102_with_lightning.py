import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from torchmetrics.classification.accuracy import Accuracy


class CNNFlower102(pl.LightningModule):
    def __init__(
        self,
        learning_rate=0.0001,
        batch_size=4,
        num_classes=102,
        num_workers=8,
        datadir="./data/flowers102/",
        pin_memory=True,
    ):

        super().__init__()

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((227, 227)),
            ]
        )

        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 102)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimz = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimz

    def train_dataloader(self):
        train_dataset = torchvision.datasets.Flowers102(
            self.hparams.datadir,
            split="train",
            transform=self.transforms,
            download=True,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = torchvision.datasets.Flowers102(
            self.hparams.datadir,
            split="val",
            transform=self.transforms,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

        return val_loader

    def test_dataloader(self):
        test_dataset = torchvision.datasets.Flowers102(
            self.hparams.datadir,
            split="test",
            transform=self.transforms,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

        return test_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y - 1)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y - 1)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y - 1)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "test_acc", (y_hat.argmax(dim=1) == y).float().mean(), on_step=True, on_epoch=False
        )
        return {"test_loss": loss}


if __name__ == "__main__":
    model = CNNFlower102()
    trainer = Trainer(max_epochs=1, fast_dev_run=False)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
