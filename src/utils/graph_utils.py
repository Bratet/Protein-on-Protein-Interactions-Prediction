import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from .feature_utils import extract_features

class ProteinInteractionGraph:
    
    def __init__(self, negative_interaction_method='most_distant'):
        self.df = pd.read_csv("../data/data.csv")
        self.df = self.df[['Interactor 1 RefSeq id', 'Interactor 2 RefSeq id']]
        # remove rows with self-interactions
        self.df = self.df[self.df['Interactor 1 RefSeq id'] != self.df['Interactor 2 RefSeq id']]
        
        self.protein_sequences = self.read_sequences_from_fasta("../dbs/generated/sequences.fasta")
        
        self.graph, self.adjacency_matrix, self.node_features = self.generate_graph(negative_interaction_method)
        
        
    def generate_graph(self, negative_interaction_method='most_distant'):
        # Create an empty graph
        G = nx.Graph()

        # Add nodes with features
        for protein_id, sequence in self.protein_sequences.items():
            G.add_node(protein_id, features=extract_features(sequence))

        # Add positive interactions (edges) with label 1
        for idx, row in self.df.iterrows():
            G.add_edge(row['Interactor 1 RefSeq id'], row['Interactor 2 RefSeq id'], label=1)
            

        # Generate negative interactions using the specified method
        if negative_interaction_method == 'random_pairs':
            negative_interactions = self.random_pairs(G)
        elif negative_interaction_method == 'same_degree_distribution':
            negative_interactions = self.same_degree_distribution(G)
        elif negative_interaction_method == 'most_close':
            negative_interactions = self.most_close(G)
        else: # most_distant
            negative_interactions = self.most_distant(G)

        # Add negative interactions (edges) with label 0
        for pair in negative_interactions:
            G.add_edge(pair[0], pair[1], label=0)

        # Create an adjacency matrix as a NumPy array
        adjacency_matrix = nx.adjacency_matrix(G).toarray()

        # Generate a feature matrix for the nodes (proteins)
        node_features = np.array([G.nodes[n]['features'] for n in G.nodes], dtype=np.float64)
        
        return G, adjacency_matrix, node_features
        
    def read_sequences_from_fasta(self, fasta_file):
        protein_sequences = {}
        with open(fasta_file, "r") as file:
            for line in file:
                if line.startswith(">"):
                    protein_id = line.strip().split(">")[1]
                else:
                    protein_sequences[protein_id] = line.strip()
        return protein_sequences
    
    def random_pairs(self, positive_graph):
        non_edges = list(nx.non_edges(positive_graph))
        negative_interactions = np.random.choice(len(non_edges), len(self.df), replace=False)
        negative_interactions = [non_edges[i] for i in negative_interactions]
        return negative_interactions
        
    def same_degree_distribution(self, positive_graph):
        degree_dict = dict(positive_graph.degree())
        non_edges = list(nx.non_edges(positive_graph))
        non_edges_sorted = sorted(non_edges, key=lambda x: degree_dict[x[0]] + degree_dict[x[1]], reverse=True)
        negative_interactions = non_edges_sorted[:len(self.df)]
        return negative_interactions

    def most_close(self, positive_graph):
        degree_dict = dict(positive_graph.degree())
        non_edges = list(nx.non_edges(positive_graph))
        mean_degree = np.mean(list(degree_dict.values()))
        non_edges_sorted = sorted(non_edges, key=lambda x: abs(degree_dict[x[0]] + degree_dict[x[1]] - mean_degree))
        negative_interactions = non_edges_sorted[:len(self.df)]
        return negative_interactions
        
    def most_distant(self, positive_graph):
        degree_dict = dict(positive_graph.degree())
        non_edges = list(nx.non_edges(positive_graph))
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