import logging
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from texttable import Texttable
from tqdm import tqdm


# IMPORTANT
# This implementation is adapted from: https://github.com/benedekrozemberczki/AttentionWalk

def read_graph(graph_path):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    :return graph: graph.
    """
    print("\nTarget matrix creation started.\n")
    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist())
    return graph


def tab_printer(args):
    """
    Function to log the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    time.sleep(0.1)
    args = vars(args)
    keys = sorted(args.keys())
    bad_dict = ["__dict__", "__doc__", "__module__", "__weakref__"]
    [keys.remove(key) for key in bad_dict if key in keys]
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    logging.info("\n" + t.draw())
    time.sleep(0.1)


def feature_calculator(args, graph):
    """
    Calculating the feature tensor.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :return target_matrices: Target tensor.
    """
    ind = range(len(graph.nodes()))
    degs = [1.0 / graph.degree(node) for node in graph.nodes()]
    adjacency_matrix = sparse.csr_matrix(nx.adjacency_matrix(graph), dtype=np.float32)
    degs = sparse.csr_matrix(sparse.coo_matrix((degs, (ind, ind)), shape=adjacency_matrix.shape, dtype=np.float32))
    normalized_adjacency_matrix = degs.dot(adjacency_matrix)
    target_matrices = [normalized_adjacency_matrix.todense()]
    powered_A = normalized_adjacency_matrix
    if args.window_size > 1:
        for power in tqdm(range(args.window_size - 1), desc="Adjacency matrix powers"):
            powered_A = powered_A.dot(normalized_adjacency_matrix)
            to_add = powered_A.todense()
            target_matrices.append(to_add)
    target_matrices = np.array(target_matrices)
    return target_matrices


def adjacency_opposite_calculator(graph):
    """
    Creating no edge indicator matrix.
    :param graph: NetworkX object.
    :return adjacency_matrix_opposite: Indicator matrix.
    """
    adjacency_matrix = sparse.csr_matrix(nx.adjacency_matrix(graph), dtype=np.float32).todense()
    adjacency_matrix_opposite = np.ones(adjacency_matrix.shape) - adjacency_matrix
    return adjacency_matrix_opposite
