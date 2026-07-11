"""MLP architecture based on (Arjovsky et al., 2019)."""

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=2 * 14 * 14, hidden_dim=390):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(self.encoder(x))

    def features(self, x):
        return self.encoder(x)
