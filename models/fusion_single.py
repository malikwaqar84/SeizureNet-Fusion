import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadFusion(nn.Module):
    def __init__(self, eeg_dim, llm_dim, hidden_dim=128):
        super().__init__()

        self.proj = nn.Linear(eeg_dim + llm_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 2)

        self.out = nn.Sequential(
            nn.Linear(eeg_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, eeg_feat, llm_feat):

        # Concatenate
        x = torch.cat([eeg_feat, llm_feat], dim=1)

        # Projection
        h = F.relu(self.proj(x))

        # Attention weights
        alpha = torch.softmax(self.attn(h), dim=1)

        alpha_eeg = alpha[:, 0].unsqueeze(1)
        alpha_llm = alpha[:, 1].unsqueeze(1)

        # Fusion
        fused = alpha_eeg * eeg_feat + alpha_llm * llm_feat

        y = self.out(fused)

        return y, alpha
