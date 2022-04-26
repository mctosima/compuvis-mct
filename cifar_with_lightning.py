import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer


class CNNCifar(pl.LightningModule):
    def __init__(
        self,
        learning_rate=0.0001,
        batch_size=4,
        num_classes=10,
        num_workers=8,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimz = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimz

    def train_dataloader(self):
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, transform=transformation, download=True
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )
        return train_loader

    def test_dataloader(self):
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, transform=transformation, download=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        return test_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "test_acc", (y_hat.argmax(dim=1) == y).float().mean(), on_step=True, on_epoch=True
        )
        return {"test_loss": loss}


if __name__ == "__main__":
    model = CNNCifar()
    trainer = Trainer(max_epochs=10, fast_dev_run=False)
    trainer.fit(model)
    trainer.test(model)
