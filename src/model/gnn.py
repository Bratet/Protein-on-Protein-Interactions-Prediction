import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool as gmp


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = gmp(x, batch)  # Use the global_mean_pool function
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x