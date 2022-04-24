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
num_epochs = 2
batch_size = 100
learning_rate = 0.001


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
        return loss

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, num_workers=2, batch_size=batch_size, shuffle=True
        )
        return train_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


if __name__ == "__main__":
    model = NeuralNet(input_size, hidden_size, num_classes)
    trainer = Trainer(max_epochs=num_epochs)
    trainer.fit(model)
