"""Neural models: FedTGNN-SS encoder, GCN/GraphSAGE baselines, MLP, LR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class DynamicEdgeAttention(nn.Module):
    """a_ij = sigmoid(MLP([x_i || x_j])), gating the Gaussian edge weights."""

    def __init__(self, d, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x, edge_index):
        row, col = edge_index
        return torch.sigmoid(self.mlp(torch.cat([x[row], x[col]], 1))).squeeze(-1)


class FedTGNNEncoder(nn.Module):
    """Two-layer hybrid GCN/GraphSAGE encoder with learnable fusion
    alpha = sigmoid(a), dynamic edge attention and a temperature-scaled
    linear classifier. Embedding size = hidden // 2."""

    def __init__(self, d, hidden=128, dropout=0.4, edge_attention=True):
        super().__init__()
        self.edge_attention = edge_attention
        self.attn = DynamicEdgeAttention(d) if edge_attention else None
        self.gcn1, self.sage1 = GCNConv(d, hidden), SAGEConv(d, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.gcn2 = GCNConv(hidden, hidden // 2)
        self.sage2 = SAGEConv(hidden, hidden // 2)
        self.cls = nn.Linear(hidden // 2, 2)
        self.a = nn.Parameter(torch.tensor(0.0))
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        w = edge_weight
        if self.attn is not None:
            w = edge_weight * self.attn(x, edge_index)
        alpha = torch.sigmoid(self.a)
        h = alpha * self.gcn1(x, edge_index, w) + (1 - alpha) * self.sage1(x, edge_index)
        h = F.dropout(F.relu(self.bn1(h)), self.dropout, self.training)
        h = alpha * self.gcn2(h, edge_index, w) + (1 - alpha) * self.sage2(h, edge_index)
        logits = self.cls(h) / torch.clamp(self.temp, min=0.1)
        return logits, h


class GCN(nn.Module):
    def __init__(self, d, hidden=64, dropout=0.5):
        super().__init__()
        self.c1, self.c2, self.dropout = GCNConv(d, hidden), GCNConv(hidden, 2), dropout

    def forward(self, x, edge_index, edge_weight):
        h = F.dropout(F.relu(self.c1(x, edge_index, edge_weight)), self.dropout, self.training)
        return self.c2(h, edge_index, edge_weight), h


class SAGE(nn.Module):
    def __init__(self, d, hidden=64, dropout=0.5):
        super().__init__()
        self.c1, self.c2, self.dropout = SAGEConv(d, hidden), SAGEConv(hidden, 2), dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = F.dropout(F.relu(self.c1(x, edge_index)), self.dropout, self.training)
        return self.c2(h, edge_index), h


class MLP(nn.Module):
    def __init__(self, d, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


class LogReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 2)

    def forward(self, x):
        return self.lin(x)


def fedavg(states, sizes):
    """Size-weighted average of state dicts (integer buffers are averaged as
    floats and cast back on load)."""
    tot = float(sum(sizes))
    return {k: sum(s[k].float() * (n / tot) for s, n in zip(states, sizes))
            for k in states[0]}


def load_avg(model, avg):
    sd = model.state_dict()
    model.load_state_dict({k: avg[k].to(sd[k].dtype) for k in sd})


def n_params_bytes(model):
    return sum(p.numel() for p in model.state_dict().values()) * 4
