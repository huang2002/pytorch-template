from torch import nn

import torch


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(28 * 28, 64)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.fc = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor):
        # x.shape: (batch_size=?, channels=1, width=28, height=28)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.view(attn_output.size(0), -1)
        x = self.fc(attn_output)
        return x
