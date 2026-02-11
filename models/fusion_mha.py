import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA_Fusion(nn.Module):
    def __init__(self, eeg_dim, llm_dim, hidden_dim=128, num_heads=4):
        super().__init__()

        self.eeg_fc = nn.Linear(eeg_dim, hidden_dim)
        self.llm_fc = nn.Linear(llm_dim, hidden_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, eeg_feat, llm_feat):

        eeg_p = F.relu(self.eeg_fc(eeg_feat))
        llm_p = F.relu(self.llm_fc(llm_feat))

        seq = torch.stack([eeg_p, llm_p], dim=1)

        att_out, attn_weights = self.mha(seq, seq, seq)

        fused = att_out.mean(dim=1)

        y = self.out(fused)

        return y, attn_weights
