import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_CNN(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64):
        super(EEG_CNN, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, hidden_dim)

    def forward(self, x):
        # x shape: (B, C, T)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)

        x = self.fc(x)
        return x  # (B, hidden_dim)
