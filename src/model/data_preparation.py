import torch
import numpy as np
from torch.utils.data import Dataset


class ProteinInteractionDataPreparation:
    
    def __init__(self, graph, adjacency_matrix, node_features, batch_size=64, train_ratio=0.8, val_ratio=0.1):
        self.graph = graph
        self.adjacency_matrix = adjacency_matrix
        self.node_features = node_features
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        self.train_edges, self.val_edges, self.test_edges = self.split_edges()
        
    def split_edges(self):
        edges = list(self.graph.edges(data=True))
        train_size = int(self.train_ratio * len(edges))
        val_size = int(self.val_ratio * len(edges))

        # Shuffle the edges and split them into train, val, and test sets
        np.random.shuffle(edges)
        train_edges = edges[:train_size]
        val_edges = edges[train_size:(train_size + val_size)]
        test_edges = edges[(train_size + val_size):]

        return train_edges, val_edges, test_edges
        
    def edges_to_tensor(self, edges):
        # Convert edge data (node indices and labels) to tensors
        node_list = list(self.graph.nodes())
        edge_indices = torch.tensor([(node_list.index(edge[0]), node_list.index(edge[1])) for edge in edges], dtype=torch.long).t()
        edge_labels = torch.tensor([edge[2]['label'] for edge in edges], dtype=torch.float)

        return edge_indices, edge_labels
    
    def get_data_loaders(self):
        # Convert node features to a tensor
        node_features_tensor = torch.tensor(self.node_features, dtype=torch.float)
        
        # Convert train, val, and test edge data to tensors
        train_edge_indices, train_edge_labels = self.edges_to_tensor(self.train_edges)
        val_edge_indices, val_edge_labels = self.edges_to_tensor(self.val_edges)
        test_edge_indices, test_edge_labels = self.edges_to_tensor(self.test_edges)

        # Transpose the edge indices tensors
        train_edge_indices = train_edge_indices.t()
        val_edge_indices = val_edge_indices.t()
        test_edge_indices = test_edge_indices.t()
        
         # Create PyTorch DataLoader objects for train, val, and test sets
        train_data = torch.utils.data.TensorDataset(train_edge_indices, train_edge_labels, torch.zeros(len(train_edge_indices), dtype=torch.long))
        val_data = torch.utils.data.TensorDataset(val_edge_indices, val_edge_labels, torch.zeros(len(val_edge_indices), dtype=torch.long))
        test_data = torch.utils.data.TensorDataset(test_edge_indices, test_edge_labels, torch.zeros(len(test_edge_indices), dtype=torch.long))
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        return node_features_tensor, train_loader, val_loader, test_loader
    
    def get_labels(self):
        train_edge_labels = [edge[2]['label'] for edge in self.train_edges]
        val_edge_labels = [edge[2]['label'] for edge in self.val_edges]
        test_edge_labels = [edge[2]['label'] for edge in self.test_edges]

        return train_edge_labels, val_edge_labels, test_edge_labels


def collate(samples):
    edge_indices, edge_labels = zip(*samples)

    # Concatenate edge indices and edge labels along the first dimension (batch dimension)
    edge_indices = torch.cat(edge_indices, dim=0)
    edge_labels = torch.cat(edge_labels, dim=0)

    # Create batch vector
    batch_size = len(samples)
    batch = torch.arange(batch_size).repeat_interleave(edge_indices.size(0) // batch_size)

    return edge_indices, edge_labels, batch

