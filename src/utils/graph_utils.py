import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from .feature_utils import extract_features

class ProteinInteractionGraph:
    
    def __init__(self):
        self.df = pd.read_csv("../dbs/HPRD/data.csv")
        self.df = self.df[['Interactor 1 RefSeq id', 'Interactor 2 RefSeq id']]
        # remove rows with self-interactions
        self.df = self.df[self.df['Interactor 1 RefSeq id'] != self.df['Interactor 2 RefSeq id']]
        
        
        self.protein_sequences = self.read_sequences_from_fasta("../dbs/generated/sequences.fasta")
        
        self.graph, self.adjacency_matrix, self.node_features = self.generate_graph()
        
    def read_sequences_from_fasta(self, fasta_file):
        protein_sequences = {}
        with open(fasta_file, "r") as file:
            for line in file:
                if line.startswith(">"):
                    protein_id = line.strip().split(">")[1]
                else:
                    protein_sequences[protein_id] = line.strip()
        return protein_sequences
            
    def generate_graph(self):
        # Create a graph from the DataFrame using NetworkX
        G = nx.from_pandas_edgelist(self.df, 'Interactor 1 RefSeq id', 'Interactor 2 RefSeq id')

        # Optionally, you can add self-loops if needed for the GCN
        G.add_edges_from([(n, n) for n in G.nodes])

        # Create an adjacency matrix as a NumPy array
        adjacency_matrix = nx.adjacency_matrix(G).toarray()

        # Generate a feature matrix for the nodes (proteins)
        node_features = np.array([extract_features(self.protein_sequences[n]) for n in G.nodes], dtype=np.float64)
        
        return G, adjacency_matrix, node_features
    
    def random_pairs(self):
        non_edges = list(nx.non_edges(self.graph))
        negative_interactions = np.random.choice(len(non_edges), len(self.df), replace=False)
        negative_interactions = [non_edges[i] for i in negative_interactions]
        return negative_interactions
        
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
    
    def visualize(self):
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, seed=42)

        # Draw nodes, labels, and edges with custom properties
        nx.draw(self.graph, pos, node_size=600, node_color='skyblue', with_labels=True, font_size=10, font_weight='bold', width=1.5)
        
        # Set the title
        plt.title("Protein Interaction Graph")
        
        # Show the graph
        plt.show()