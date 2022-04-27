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


class AlexNetCIFAR10(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        learning_rate=0.0001,
        batch_size=4,
        num_workers=8,
    ):

        super().__init__()
        self.in_size = 227
        self.in_channel = in_channels
        self.num_classes = num_classes
        self.save_hyperparameters()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.__init_bias()

    def __init_bias(self):
        for layer in self.fc_layer:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.conv_layer[4].bias, 1)
        nn.init.constant_(self.conv_layer[10].bias, 1)
        nn.init.constant_(self.conv_layer[12].bias, 1)

    def forward(self, x):
        assert x.shape[2] == self.in_size, "Input must be 227x227"
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def configure_optimizers(self):
        optimz = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimz

    def train_dataloader(self):
        transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((self.in_size, self.in_size)),
            ]
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
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((self.in_size, self.in_size)),
            ]
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
    model = AlexNetCIFAR10()
    trainer = Trainer(max_epochs=2, fast_dev_run=False)
    trainer.fit(model)
    trainer.test(model)
