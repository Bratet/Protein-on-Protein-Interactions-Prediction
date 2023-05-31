import tensorflow as tf
import tensorflow_gnn as tfgnn


def edge_batch_merge(graph):
    graph = graph.merge_batch_to_components()
    node_features = graph.node_sets['Proteins'].get_features_dict()
    edge_features = graph.edge_sets['Interactions'].get_features_dict()
    
    labels = edge_features.pop('labels')
    
    new_graph = graph.replace_features(
        node_sets = { 'Proteins': node_features },
        edge_sets = { 'Interactions': edge_features }
    )
    
    return new_graph, labels

def create_dataset(graph,function):
    dataset = tf.data.Dataset.from_tensors(graph)
    dataset = dataset.batch(32)
    return dataset.map(function)

def set_initial_node_state(node_set, node_set_name):
    features = [
        tf.keras.layers.Dense(32,activation="relu")(node_set['basic_protein_properties']    ),
        tf.keras.layers.Dense(32,activation="relu")(node_set['secondary_structure_content']),
        tf.keras.layers.Dense(32,activation="relu")(node_set['other_properties']),
        tf.keras.layers.Dense(32,activation="relu")(node_set['amino_acid_composition'])
    ]
    return tf.keras.layers.Concatenate()(features)

def set_initial_edge_state(edge_set, edge_set_name):
    
    return tfgnn.keras.layers.MakeEmptyFeature()(edge_set)