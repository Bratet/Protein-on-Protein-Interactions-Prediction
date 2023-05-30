import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from .feature_utils import extract_features
import tensorflow_gnn as tfgnn

class ProteinInteractionGraph:
    
    def __init__(self, negative_interaction_method='most_distant'):
        self.df = pd.read_csv("../data/data.csv")
        self.df = self.df[['Interactor 1 RefSeq id', 'Interactor 2 RefSeq id']]
        # remove rows with self-interactions
        self.df = self.df[self.df['Interactor 1 RefSeq id'] != self.df['Interactor 2 RefSeq id']]
        
        self.protein_sequences = self.read_sequences_from_fasta("../dbs/generated/sequences.fasta")
        
        self.graph, self.adjacency_matrix, self.node_features = self.generate_graph(negative_interaction_method)
        
        self.node_mapping = {node: i for i, node in enumerate(self.graph.nodes)}
        
        
    def generate_graph(self, negative_interaction_method='most_distant'):
        # Create an empty graph
        G = nx.Graph()

        # Add nodes with features
        for i, (protein_id, sequence) in enumerate(self.protein_sequences.items()):
            basic_protein_properties, secondary_structure_content, other_properties, amino_acid_composition = extract_features(sequence)
            G.add_node(i, id=protein_id, basic_protein_properties=basic_protein_properties,
                       secondary_structure_content=secondary_structure_content,
                       other_properties=other_properties, amino_acid_composition=amino_acid_composition)

        # Add positive interactions (edges) with label 1
        id_to_index = {node[1]["id"]: node[0] for node in G.nodes(data=True)}  # Create the id to index mapping
        for idx, row in self.df.iterrows():
            G.add_edge(id_to_index[row['Interactor 1 RefSeq id']], id_to_index[row['Interactor 2 RefSeq id']], label=1)
            

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
        basic_protein_properties = np.array([G.nodes[n]['basic_protein_properties'] for n in G.nodes], dtype=np.float64)
        secondary_structure_content = np.array([G.nodes[n]['secondary_structure_content'] for n in G.nodes], dtype=np.float64)
        other_properties = np.array([G.nodes[n]['other_properties'] for n in G.nodes], dtype=np.float64)
        amino_acid_composition = np.array([G.nodes[n]['amino_acid_composition'] for n in G.nodes], dtype=np.float64)
        
        return G, adjacency_matrix, (basic_protein_properties, secondary_structure_content, other_properties, amino_acid_composition)
        
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
    
    def get_graph_tensor(self):
        
        basic_protein_properties, secondary_structure_content, other_properties, amino_acid_composition = self.node_features

        edge_sources = np.array([self.node_mapping[e[0]] for e in self.graph.edges], dtype=np.int32)
        edge_targets = np.array([self.node_mapping[e[1]] for e in self.graph.edges], dtype=np.int32)
        
        graph_tensor = tfgnn.GraphTensor.from_pieces(
            node_sets= {
                'Proteins': tfgnn.NodeSet.from_fields(
                    sizes = [len(self.graph.nodes)],
                    features = {
                        'basic_protein_properties': basic_protein_properties,
                        'secondary_structure_content': secondary_structure_content,
                        'other_properties': other_properties,
                        'amino_acid_composition': amino_acid_composition,
                    }
                )
            },
            edge_sets= {
                'Interactions': tfgnn.EdgeSet.from_fields(
                    sizes = [len(self.graph.edges)],
                    features = {
                        'labels': np.array([self.graph.edges[e]['label'] for e in self.graph.edges], dtype=np.int32),
                    },
                    adjacency = tfgnn.Adjacency.from_indices(
                        source = ("Proteins", edge_sources),
                        target = ("Proteins", edge_targets),
                )
            )
            }
        )
        
        
        return graph_tensor