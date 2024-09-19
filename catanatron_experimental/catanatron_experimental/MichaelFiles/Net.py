import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size=477):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(input_size, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.output = nn.Linear(64, 1)

        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.output = nn.Linear(32, 1)

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)  # Optional dropout for regularization
        self.dropout2 = nn.Dropout(0.2)  # Optional dropout for regularization

    # def forward(self, x):
    #     x = torch.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     x = torch.relu(self.fc2(x))
    #     x = self.sigmoid(self.output(x))
    #     return x

    def forward(self, x):
        x = self.dropout2(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        # x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        return x
