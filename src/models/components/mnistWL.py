import torch.nn.functional as F
from torch import nn


class NeuralNetWL(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out
