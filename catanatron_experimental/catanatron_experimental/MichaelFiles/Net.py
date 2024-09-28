import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size=363, layer1=64, layer2=32):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(input_size, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.output = nn.Linear(64, 1)

        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 16)
        # self.output = nn.Linear(16, 1)

        self.fc1 = nn.Linear(input_size, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.output = nn.Linear(layer2, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)  # Optional dropout for regularization

    def forward(self, x):
        # x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        return x

