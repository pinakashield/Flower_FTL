import torch.nn as nn

class IntrusionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IntrusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output layer for attack classification
        )

    def forward(self, x):
        return self.net(x)


