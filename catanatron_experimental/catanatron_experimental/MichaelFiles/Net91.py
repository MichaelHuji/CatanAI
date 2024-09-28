import torch
import torch.nn as nn

class Net91(nn.Module):
    def __init__(self, input_size=91):
        super(Net91, self).__init__()
        self.output = nn.Linear(input_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.output(x))

