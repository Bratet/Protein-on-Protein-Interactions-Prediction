import pandas as pd
import networkx as nx
import numpy as np
import tensorflow_gnn as tfgnn
from sklearn.model_selection import KFold
import json

class GraphToTensor:
    
    DATA_PATH = "../dbs/data2_no_self_interactions.csv"
    FEATURES_PATH = "../dbs/id_features_data2.json"
    
    def __init__(self, negative_interaction_method='most_distant'):
        self.df = self.load_data()
        self.features = self.load_features()
        self.graph, self.adjacency_matrix, self.node_features = self.generate_graph(negative_interaction_method)
        self.node_mapping = {node: i for i, node in enumerate(self.graph.nodes)}
        self.graph_tensor = self.get_graph_tensor(self.graph)

    def load_data(self):
        df = pd.read_csv(self.DATA_PATH)
        return df
    
    def load_features(self):
        with open(self.FEATURES_PATH) as json_file:
            data = json.load(json_file)
        return data

    def add_nodes_to_graph(self, G):
        for i, protein_id in enumerate(self.features.keys()):

            G.add_node(i, id=protein_id, 
                    basic_protein_properties=self.features[protein_id]['basic_protein_properties'],
                    secondary_structure_content=self.features[protein_id]['secondary_structure_content'],
                    other_properties=self.features[protein_id]['other_properties'],
                    amino_acid_composition=self.features[protein_id]['amino_acid_composition'],
                    pssm_sum=self.features[protein_id]['pssm_sum'],
                    S_vector=self.features[protein_id]['S_vector'],
                    )
                        

    def add_positive_interactions(self, G):
        id_to_index = {node[1]["id"]: node[0] for node in G.nodes(data=True)}  # Create the id to index mapping
        for idx, row in self.df.iterrows():
            G.add_edge(id_to_index[row['Interactor A']], id_to_index[row['Interactor B']], label=1)

    def add_negative_interactions(self, G, negative_interaction_method):
        if negative_interaction_method == 'random_pairs':
            negative_interactions = self.random_pairs(G)
        elif negative_interaction_method == 'same_degree_distribution':
            negative_interactions = self.same_degree_distribution(G)
        elif negative_interaction_method == 'most_close':
            negative_interactions = self.most_close(G)
        else:  # most_distant
            negative_interactions = self.most_distant(G)
        for pair in negative_interactions:
            G.add_edge(pair[0], pair[1], label=0)

    def generate_graph(self, negative_interaction_method='most_distant'):
        G = nx.Graph()
        self.add_nodes_to_graph(G)
        self.add_positive_interactions(G)
        self.add_negative_interactions(G, negative_interaction_method)
        adjacency_matrix = nx.adjacency_matrix(G).toarray()
        return G, adjacency_matrix, self.extract_node_features(G)

    def same_degree_distribution(self, graph):
        degree_dict = dict(graph.degree())
        non_edges = list(nx.non_edges(graph))
        non_edges_sorted = sorted(non_edges, key=lambda x: degree_dict[x[0]] + degree_dict[x[1]], reverse=True)
        negative_interactions = non_edges_sorted[:len(self.df)]
        return negative_interactions

    def calculate_mean_feature_vector(self, graph):
        all_features = np.array([self.features[graph.nodes[node]['id']]['amino_acid_composition'] for node in graph.nodes()])
        mean_vector = np.mean(all_features, axis=0)
        return mean_vector

    def most_close(self, graph):
        mean_vector = self.calculate_mean_feature_vector(graph)
        non_edges = list(nx.non_edges(graph))
        non_edges_sorted = sorted(non_edges, key=lambda x: np.linalg.norm(self.features[graph.nodes[x[0]]['id']]['amino_acid_composition'] - mean_vector + self.features[graph.nodes[x[1]]['id']]['amino_acid_composition'] - mean_vector))
        negative_interactions = non_edges_sorted[:len(self.df)]
        return negative_interactions

    def most_distant(self, positive_graph):
        mean_vector = self.calculate_mean_feature_vector(positive_graph)
        non_edges = list(nx.non_edges(positive_graph))
        non_edges_sorted = sorted(non_edges, key=lambda x: np.linalg.norm(self.features[positive_graph.nodes[x[0]]['id']]['amino_acid_composition'] - mean_vector + self.features[positive_graph.nodes[x[1]]['id']]['amino_acid_composition'] - mean_vector), reverse=True)
        negative_interactions = non_edges_sorted[:len(self.df)]
        return negative_interactions



    def get_graph_tensor(self, graph):
        edge_sources = np.array([self.node_mapping[e[0]] for e in graph.edges], dtype=np.int32)
        edge_targets = np.array([self.node_mapping[e[1]] for e in graph.edges], dtype=np.int32)
        graph_tensor = tfgnn.GraphTensor.from_pieces(
            node_sets=self.construct_node_sets(graph),
            edge_sets=self.construct_edge_sets(graph, edge_sources, edge_targets)
        )
        return graph_tensor
    
    def extract_node_features(self, G):
        features_list = ['basic_protein_properties', 'secondary_structure_content', 'other_properties', 'amino_acid_composition', 'pssm_sum', 'S_vector']
        node_features = {feat: np.array([G.nodes[n][feat] for n in G.nodes], dtype=np.float64) for feat in features_list}
        return node_features

    def construct_node_sets(self, graph):
        features_dict = {feat: self.node_features[feat] for feat in self.node_features}
        return {'Proteins': tfgnn.NodeSet.from_fields(sizes=[len(graph.nodes)], features=features_dict)}


    def construct_edge_sets(self, graph, edge_sources, edge_targets):
        return {'Interactions': tfgnn.EdgeSet.from_fields(
            sizes=[len(graph.edges)],
            features={'labels': np.array([graph.edges[e]['label'] for e in graph.edges], dtype=np.int32)},
            adjacency=tfgnn.Adjacency.from_indices(source=("Proteins", edge_sources),
                                                   target=("Proteins", edge_targets),
                                                   ))}

    def generate_graph_tensors_for_k_folds(self, k_folds=5):
        graph_splits = self.k_fold_split(k_folds)
        graph_tensor_splits = [(self.graph_to_graph_tensor(train), self.graph_to_graph_tensor(test)) for train, test in graph_splits]
        return graph_tensor_splits

    def k_fold_split(self, k_folds=5):
        # Ensure the graph has been generated
        if self.graph is None:
            raise ValueError("Graph has not been generated yet.")

        # Get all edges from the graph and their labels
        edges = list(self.graph.edges(data=True))
        edge_labels = [e[2]["label"] for e in edges]

        # Initialize KFold
        kf = KFold(n_splits=k_folds)

        graph_splits = []

        # Perform k-fold splitting
        for train_indices, test_indices in kf.split(edges):
            # Create training and testing graphs
            train_graph = nx.Graph()
            test_graph = nx.Graph()

            # Add nodes to the training and testing graphs
            for node in self.graph.nodes(data=True):
                train_graph.add_node(node[0], **node[1])
                test_graph.add_node(node[0], **node[1])

            # Add training edges to the training graph
            for index in train_indices:
                edge, label = edges[index], edge_labels[index]
                train_graph.add_edge(edge[0], edge[1], label=label)

            # Add testing edges to the testing graph
            for index in test_indices:
                edge, label = edges[index], edge_labels[index]
                if edge[0] in train_graph and edge[1] in train_graph:
                    test_graph.add_edge(edge[0], edge[1], label=label)

            graph_splits.append((train_graph, test_graph))

        return graph_splits

    def graph_to_graph_tensor(self, graph):
        # Ensure the graph has been generated
        if graph is None:
            raise ValueError("Graph has not been generated yet.")
        return self.get_graph_tensor(graph)




# import pandas as pd
# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
# from .feature_utils import extract_features
# import tensorflow_gnn as tfgnn

# class GraphToTensor:
    
#     def __init__(self, negative_interaction_method='most_distant'):
#         self.df = pd.read_csv("../data/data.csv")
#         self.df = self.df[['Interactor 1 RefSeq id', 'Interactor 2 RefSeq id']]
#         # remove rows with self-interactions
#         self.df = self.df[self.df['Interactor 1 RefSeq id'] != self.df['Interactor 2 RefSeq id']]
        
#         self.protein_sequences = self.read_sequences_from_fasta("../dbs/generated/sequences.fasta")
        
#         self.graph, self.adjacency_matrix, self.node_features = self.generate_graph(negative_interaction_method)
        
#         self.node_mapping = {node: i for i, node in enumerate(self.graph.nodes)}
        
        
#     def generate_graph(self, negative_interaction_method='most_distant'):
#         # Create an empty graph
#         G = nx.Graph()

#         # Add nodes with features
#         for i, (protein_id, sequence) in enumerate(self.protein_sequences.items()):
#             basic_protein_properties, secondary_structure_content, other_properties, amino_acid_composition = extract_features(sequence)
#             G.add_node(i, id=protein_id, basic_protein_properties=basic_protein_properties,
#                        secondary_structure_content=secondary_structure_content,
#                        other_properties=other_properties, amino_acid_composition=amino_acid_composition)

#         # Add positive interactions (edges) with label 1
#         id_to_index = {node[1]["id"]: node[0] for node in G.nodes(data=True)}  # Create the id to index mapping
#         for idx, row in self.df.iterrows():
#             G.add_edge(id_to_index[row['Interactor 1 RefSeq id']], id_to_index[row['Interactor 2 RefSeq id']], label=1)
            

#         # Generate negative interactions using the specified method
#         if negative_interaction_method == 'random_pairs':
#             negative_interactions = self.random_pairs(G)
#         elif negative_interaction_method == 'same_degree_distribution':
#             negative_interactions = self.same_degree_distribution(G)
#         elif negative_interaction_method == 'most_close':
#             negative_interactions = self.most_close(G)
#         else: # most_distant
#             negative_interactions = self.most_distant(G)

#         # Add negative interactions (edges) with label 0
#         for pair in negative_interactions:
#             G.add_edge(pair[0], pair[1], label=0)

#         # Create an adjacency matrix as a NumPy array
#         adjacency_matrix = nx.adjacency_matrix(G).toarray()

#         # Generate a feature matrix for the nodes (proteins)
#         basic_protein_properties = np.array([G.nodes[n]['basic_protein_properties'] for n in G.nodes], dtype=np.float64)
#         secondary_structure_content = np.array([G.nodes[n]['secondary_structure_content'] for n in G.nodes], dtype=np.float64)
#         other_properties = np.array([G.nodes[n]['other_properties'] for n in G.nodes], dtype=np.float64)
#         amino_acid_composition = np.array([G.nodes[n]['amino_acid_composition'] for n in G.nodes], dtype=np.float64)
        
#         return G, adjacency_matrix, (basic_protein_properties, secondary_structure_content, other_properties, amino_acid_composition)
        
#     def read_sequences_from_fasta(self, fasta_file):
#         protein_sequences = {}
#         with open(fasta_file, "r") as file:
#             for line in file:
#                 if line.startswith(">"):
#                     protein_id = line.strip().split(">")[1]
#                 else:
#                     protein_sequences[protein_id] = line.strip()
#         return protein_sequences
    
#     def random_pairs(self, positive_graph):
#         non_edges = list(nx.non_edges(positive_graph))
#         negative_interactions = np.random.choice(len(non_edges), len(self.df), replace=False)
#         negative_interactions = [non_edges[i] for i in negative_interactions]
#         return negative_interactions
        
#     def same_degree_distribution(self, positive_graph):
#         degree_dict = dict(positive_graph.degree())
#         non_edges = list(nx.non_edges(positive_graph))
#         non_edges_sorted = sorted(non_edges, key=lambda x: degree_dict[x[0]] + degree_dict[x[1]], reverse=True)
#         negative_interactions = non_edges_sorted[:len(self.df)]
#         return negative_interactions

#     def most_close(self, positive_graph):
#         degree_dict = dict(positive_graph.degree())
#         non_edges = list(nx.non_edges(positive_graph))
#         mean_degree = np.mean(list(degree_dict.values()))
#         non_edges_sorted = sorted(non_edges, key=lambda x: abs(degree_dict[x[0]] + degree_dict[x[1]] - mean_degree))
#         negative_interactions = non_edges_sorted[:len(self.df)]
#         return negative_interactions
        
#     def most_distant(self, positive_graph):
#         degree_dict = dict(positive_graph.degree())
#         non_edges = list(nx.non_edges(positive_graph))
#         mean_degree = np.mean(list(degree_dict.values()))
#         non_edges_sorted = sorted(non_edges, key=lambda x: abs(degree_dict[x[0]] + degree_dict[x[1]] - mean_degree), reverse=True)
#         negative_interactions = non_edges_sorted[:len(self.df)]
#         return negative_interactions
    
#     def get_graph_tensor(self, graph):
#         edge_sources = np.array([self.node_mapping[e[0]] for e in graph.edges], dtype=np.int32)
#         edge_targets = np.array([self.node_mapping[e[1]] for e in graph.edges], dtype=np.int32)
        
#         basic_protein_properties = np.array([graph.nodes[n]['basic_protein_properties'] for n in graph.nodes], dtype=np.float64)
#         secondary_structure_content = np.array([graph.nodes[n]['secondary_structure_content'] for n in graph.nodes], dtype=np.float64)
#         other_properties = np.array([graph.nodes[n]['other_properties'] for n in graph.nodes], dtype=np.float64)
#         amino_acid_composition = np.array([graph.nodes[n]['amino_acid_composition'] for n in graph.nodes], dtype=np.float64)
        
#         graph_tensor = tfgnn.GraphTensor.from_pieces(
#             node_sets= {
#                 'Proteins': tfgnn.NodeSet.from_fields(
#                     sizes = [len(graph.nodes)],
#                     features = {
#                         'basic_protein_properties': basic_protein_properties,
#                         'secondary_structure_content': secondary_structure_content,
#                         'other_properties': other_properties,
#                         'amino_acid_composition': amino_acid_composition,
#                     }
#                 )
#             },
#             edge_sets= {
#                 'Interactions': tfgnn.EdgeSet.from_fields(
#                     sizes = [len(graph.edges)],
#                     features = {
#                         'labels': np.array([graph.edges[e]['label'] for e in graph.edges], dtype=np.int32),
#                     },
#                     adjacency = tfgnn.Adjacency.from_indices(
#                         source = ("Proteins", edge_sources),
#                         target = ("Proteins", edge_targets),
#                 )
#             )
#             }
#         )
        
#         return graph_tensor

    
#     def split_graph(self, train_size=0.8):
#         # Ensure the graph has been generated
#         if self.graph is None:
#             raise ValueError("Graph has not been generated yet.")

#         # Get all edges from the graph and their labels
#         edges = list(self.graph.edges(data=True))
#         edge_labels = [e[2]["label"] for e in edges]

#         # Calculate the number of edges to be used for training
#         n_train = int(train_size * len(edges))

#         # Randomly shuffle edges and their labels
#         combined = list(zip(edges, edge_labels))
#         np.random.shuffle(combined)
#         shuffled_edges, shuffled_labels = zip(*combined)

#         # Split edges and their labels into training and testing sets
#         train_edges = shuffled_edges[:n_train]
#         train_labels = shuffled_labels[:n_train]
#         test_edges = shuffled_edges[n_train:]
#         test_labels = shuffled_labels[n_train:]

#         # Create training and testing graphs
#         train_graph = nx.Graph()
#         test_graph = nx.Graph()

#         # Add nodes to the training and testing graphs
#         for node in self.graph.nodes(data=True):
#             train_graph.add_node(node[0], **node[1])
#             test_graph.add_node(node[0], **node[1])

#         # Add training edges to the training graph
#         for edge, label in zip(train_edges, train_labels):
#             train_graph.add_edge(edge[0], edge[1], label=label)

#         # Add testing edges to the testing graph
#         for edge, label in zip(test_edges, test_labels):
#             if edge[0] in train_graph and edge[1] in train_graph:
#                 test_graph.add_edge(edge[0], edge[1], label=label)

#         return train_graph, test_graph
