
import torch.nn as nn

class IntrusionModel(nn.Module):
    def __init__(self, input_dim, num_classes, freeze_base=False):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, num_classes)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.base(x)
        return self.head(x)
