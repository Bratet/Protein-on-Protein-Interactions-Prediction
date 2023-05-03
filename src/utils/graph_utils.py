import pandas as pd
import networkx as nx
import numpy as np
from feature_utils import extract_features

class ProteinInteractionGraph:
    
    def __init__(self):
        self.df = pd.read_csv("../dbs/HPRD/data.csv")
        self.df = self.df[['Interactor 1 RefSeq id', 'Interactor 2 RefSeq id']]
        
        self.protein_sequences = {}
        with open("../dbs/generated/sequences.fasta", "r") as file:
            for line in file:
                if line.startswith(">"):
                    protein_id = line.strip().split("|")[1]
                else:
                    self.protein_sequences[protein_id] = line.strip()
        
        self.graph, self.adjacency_matrix, self.node_degrees = self.generate_graph()
            
    def generate_graph(self):
        # Create a graph from the DataFrame using NetworkX
        G = nx.from_pandas_edgelist(self.df, 'Interactor 1 RefSeq id', 'Interactor 2 RefSeq id')

        # Optionally, you can add self-loops if needed for the GCN
        G.add_edges_from([(n, n) for n in G.nodes])

        # Create an adjacency matrix as a NumPy array
        adjacency_matrix = nx.adjacency_matrix(G).toarray()

        # Generate a feature matrix for the nodes (proteins)
        node_features = np.array([extract_features(self.protein_sequences[n]) for n in G.nodes])

        return G, adjacency_matrix, node_features
        
    def same_degree_distribution(self):
        degree_dict = dict(self.graph.degree())
        non_edges = list(nx.non_edges(self.graph))
        non_edges_sorted = sorted(non_edges, key=lambda x: degree_dict[x[0]] + degree_dict[x[1]], reverse=True)
        negative_interactions = non_edges_sorted[:len(self.df)]
        return negative_interactions

    def most_close(self):
        degree_dict = dict(self.graph.degree())
        non_edges = list(nx.non_edges(self.graph))
        mean_degree = np.mean(list(degree_dict.values()))
        non_edges_sorted = sorted(non_edges, key=lambda x: abs(degree_dict[x[0]] + degree_dict[x[1]] - mean_degree))
        negative_interactions = non_edges_sorted[:len(self.df)]
        return negative_interactions
        
    def most_distant(self):
        degree_dict = dict(self.graph.degree())
        non_edges = list(nx.non_edges(self.graph))
        mean_degree = np.mean(list(degree_dict.values()))
        non_edges_sorted = sorted(non_edges, key=lambda x: abs(degree_dict[x[0]] + degree_dict[x[1]] - mean_degree), reverse=True)
        negative_interactions = non_edges_sorted[:len(self.df)]
        return negative_interactions