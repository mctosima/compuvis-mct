import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMCT(nn.Module):
    def __init__(
        self,
        in_channel=3,
        num_classes=102,
        img_size=227,
    ):

        super().__init__()
        self.in_channel = in_channel
        self.img_size = img_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.in_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        _, self.channel, self.width, self.height = self.getOutputConv(
            torch.randn(1, self.in_channel, self.img_size, self.img_size)
        )
        self.fc1 = nn.Linear(self.channel * self.width * self.height, 120)
        self.fc2 = nn.Linear(120, self.num_classes)

    def getOutputConv(self, x):
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.size()
            return x

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.channel * self.width * self.height)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
