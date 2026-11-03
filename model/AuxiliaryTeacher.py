"""Independent direct forecasting teacher; never replaces innovation modules."""
import torch
from torch import nn


class AxisAttention(nn.Module):
    def __init__(self, width=64, axis='time'):
        super().__init__()
        self.axis = axis
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, width))

    def forward(self, x):
        b, t, n, d = x.shape
        if self.axis == 'time':
            z = x.transpose(1, 2).reshape(b * n, t, d)
        else:
            z = x.reshape(b * t, n, d)
        q = self.norm1(z)
        z = z + self.attention(q, q, q, need_weights=False)[0]
        z = z + self.ff(self.norm2(z))
        if self.axis == 'time':
            return z.reshape(b, n, t, d).transpose(1, 2)
        return z.reshape(b, t, n, d)


class AuxiliaryTeacher(nn.Module):
    def __init__(self, nodes=307, layers=3):
        super().__init__()
        self.nodes = nodes
        self.flow = nn.Linear(3, 24)
        self.time = nn.Embedding(288, 8)
        self.week = nn.Embedding(7, 8)
        self.node = nn.Embedding(nodes, 8)
        self.adaptive = nn.Parameter(torch.empty(12, nodes, 16))
        nn.init.xavier_uniform_(self.adaptive)
        self.temporal = nn.ModuleList([AxisAttention(axis='time') for _ in range(layers)])
        self.spatial = nn.ModuleList([AxisAttention(axis='space') for _ in range(layers)])
        self.output = nn.Linear(12 * 64, 12)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, starts):
        b, t, n, _ = x.shape
        times = starts[:, None] + torch.arange(t, device=x.device)
        tod = self.time(times % 288)[:, :, None].expand(-1, -1, n, -1)
        dow = self.week(times // 288 % 7)[:, :, None].expand(-1, -1, n, -1)
        node = self.node.weight[None, None].expand(b, t, -1, -1)
        adaptive = self.adaptive[None].expand(b, -1, -1, -1)
        z = torch.cat([self.flow(x), tod, dow, node, adaptive], -1)
        for layer in self.temporal:
            z = layer(z)
        for layer in self.spatial:
            z = layer(z)
        residual = self.output(z.transpose(1, 2).reshape(b, n, -1)).transpose(1, 2)
        return residual[..., None] + x[:, -1, :, :1][:, None]
