import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        self.fc1 = nn.Linear(128 * 17 * 17, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, 74, 74)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, 36, 36)
        x = self.pool(F.relu(self.conv3(x)))  # -> (128, 17, 17)
        x = x.view(-1, 128 * 17 * 17)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
