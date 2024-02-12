import tensorflow as tf
import tensorflow_gnn as tfgnn

def set_initial_edge_state(edge_set, edge_set_name):
    
    return tfgnn.keras.layers.MakeEmptyFeature()(edge_set)

def create_initial_node_state_fn(activation):
    def initial_node_state(node_set, node_set_name):
        features = [
            tf.keras.layers.Dense(32,activation=activation)(node_set['basic_protein_properties']),
            tf.keras.layers.Dense(32,activation=activation)(node_set['secondary_structure_content']),
            tf.keras.layers.Dense(32,activation=activation)(node_set['other_properties']),
            tf.keras.layers.Dense(32,activation=activation)(node_set['amino_acid_composition']),
            tf.keras.layers.Dense(32,activation=activation)(node_set['pssm_sum']),
            tf.keras.layers.Dense(32,activation=activation)(node_set['S_vector']),
        ]
        return tf.keras.layers.Concatenate()(features)
    return initial_node_state

def dense_batchnorm_layer(units, activation=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.Dropout(0.2)
    ])
    
def create_model(input_graph, graph_updates, activation='leaky_relu'):
    initial_node_state = create_initial_node_state_fn(activation)
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=initial_node_state,
        edge_sets_fn=set_initial_edge_state,
    )(input_graph)

    for i in range(graph_updates):
        graph = tfgnn.keras.layers.GraphUpdate(
            edge_sets = {'Interactions': tfgnn.keras.layers.EdgeSetUpdate(
                next_state = tfgnn.keras.layers.NextStateFromConcat(
                    dense_batchnorm_layer(64, activation=activation)))},
            node_sets = {
                'Proteins': tfgnn.keras.layers.NodeSetUpdate({
                    'Interactions': tfgnn.keras.layers.Pool(
                        tag=tfgnn.TARGET,
                        reduce_type="sum",
                        feature_name = tfgnn.HIDDEN_STATE)},
                    tfgnn.keras.layers.NextStateFromConcat(
                        dense_batchnorm_layer(64, activation=activation))),
            })(graph)

    logits = tf.keras.layers.Dense(1, activation='sigmoid')(graph.edge_sets['Interactions'][tfgnn.HIDDEN_STATE])

    model = tf.keras.Model(input_graph, logits)

    return model