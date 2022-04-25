import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.003
num_workers = 8


class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(-1, 28 * 28)

        # forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, on_step=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(-1, 28 * 28)

        # forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        test_acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", test_acc)
        return {"test_loss": loss, "test_acc": test_acc}

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True
        )
        return train_loader

    def test_dataloader(self):
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transforms.ToTensor()
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False
        )
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


if __name__ == "__main__":
    model = NeuralNet(input_size, hidden_size, num_classes)
    trainer = Trainer(max_epochs=num_epochs, fast_dev_run=False)
    trainer.fit(model)
    trainer.test(model)
