import torch
import torch.nn as nn
import torch.nn.functional as F



class StandardConv(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        img_size: int = 227,
        output_size: int = 102,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_size, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        _, self.channel, self.width, self.height = self.getOutputConv(
            torch.randn(1, input_size, img_size, img_size)
        )
        
        self.fc1 = nn.Linear(self.channel * self.width * self.height, 120)
        self.fc2 = nn.Linear(120, output_size)

        ''' Commented out for now
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )
        '''

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.channel * self.width * self.height)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def getOutputConv(self, x):
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.size()
            return x
