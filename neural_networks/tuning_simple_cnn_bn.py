import torch
import torch.nn as nn
import torch.nn.functional as F


class TuningSimpleCNNWithBN(nn.Module):
    def __init__(
        self, dropout=0.5, num_fc_layers=1, hidden_units=512, activation_fn=F.relu
    ):
        super(TuningSimpleCNNWithBN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = activation_fn
        self.num_fc_layers = num_fc_layers

        self.fc1 = nn.Linear(128 * 17 * 17, hidden_units)
        self.bn_fc1 = nn.BatchNorm1d(hidden_units)

        if num_fc_layers == 2:
            self.fc2 = nn.Linear(hidden_units, hidden_units)
            self.bn_fc2 = nn.BatchNorm1d(hidden_units)
            self.fc3 = nn.Linear(hidden_units, 1)
        else:
            self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.pool(self.activation_fn(self.bn1(self.conv1(x))))
        x = self.pool(self.activation_fn(self.bn2(self.conv2(x))))
        x = self.pool(self.activation_fn(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)

        x = self.dropout(self.activation_fn(self.bn_fc1(self.fc1(x))))

        if self.num_fc_layers == 2:
            x = self.dropout(self.activation_fn(self.bn_fc2(self.fc2(x))))
            x = torch.sigmoid(self.fc3(x))
        else:
            x = torch.sigmoid(self.fc2(x))

        return x
