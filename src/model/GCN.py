import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define the GCN model
class GCNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_units, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x