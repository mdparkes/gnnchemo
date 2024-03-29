

import itertools
import numpy as np
import os
import pandas as pd
import torch

from keggpathwaygraphs import biopathgraph as bpg
from torch import Tensor
from typing import Dict, List, Set, Tuple

from custom_data_types import NodeInfoDict, AdjacencyDict


def maybe_create_directories(*dirs) -> None:
    """Check if dirs exist and creates them if necessary"""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


def map_indices_to_names(nodes: NodeInfoDict) -> Dict[int, str]:
    """Create a dictionary of node names keyed by integer index values"""
    return dict(zip(range(len(nodes)), nodes.keys()))


def filter_datasets_and_graph_info(
        exprs_df: pd.DataFrame, node_info: NodeInfoDict, edge_info: AdjacencyDict, pathway_info: NodeInfoDict,
        min_nodes: int = 15
) -> Tuple[pd.DataFrame, NodeInfoDict, AdjacencyDict, NodeInfoDict]:
    """
    Given a gene expression DataFrame, a NodeInfoDict and AdjacencyDict for a graph whose nodes represent genes in a
    pathway, a NodeInfoDict whose nodes represent the pathways themselves, and a minimum number of nodes per
    pathway, this function removes genes (nodes) from the graph if they do not appear in exprs_df,
    and also removes genes from exprs_df that only appear in pathways with fewer than the specified minimum number of
    nodes. Those pathways are also removed from the pathway NodeInfoDict.

    When nodes are removed from `node_info`, edges that have the removed nodes as their source in edge_info are also
    removed. A new edge is formed between the target node and any node that had the removed node as its target. For
    example, if A -> B -> C <- D is a graph and node B is removed, the new graph becomes A -> C <- D.

    Note: When edge types are used in downstream applications, the edge removal may invalidate the graph. For
    example, consider A -[activates]-> B -[inhibits]-> C -[activates]-> D. Depending on the mechanism of inhibition
    of C by B, this may imply that A -[inhibits]-> D through the action of B, but if B and C are removed via remove_node
    the result would be A -[activates]-> D. This is not an issue when edge types are not considered in downstream
    applications.

    :param exprs_df: A DataFrame with gene expression counts
    :param node_info: A NodeInfoDict for a graph whose nodes represent genes
    :param edge_info: An AdjacencyDict for a graph whose nodes represent genes
    :param pathway_info: A NodeInfoDict for a graph whose nodes represent pathways
    :param min_nodes: The minimum number of nodes that a pathway must have in order to survive filtering
    :return: Returns the filtered input objects
    """
    # Remove genes from node_info and edge_info if they either don't appear in exprs_df. Create edges between parents
    # and children of the removed nodes in edge_info.
    for_removal = [gene for gene in node_info.keys() if gene not in exprs_df.columns]
    for gene in for_removal:
        node_info, edge_info = bpg.remove_node(gene, node_info, edge_info)
    # Restrict the members of each pathway to genes in node_info
    pathway_info = bpg.update_children(pathway_info, set(node_info.keys()))
    # Exclude pathways with < min_nodes nodes and create a set of nodes that appear in any of the remaining pathways
    # Returns empty dicts/DataFrame if all pathways are dropped
    new_pathway_info = dict()
    remaining_nodes = set()
    for path, info in pathway_info.items():
        children = info["children"]
        if len(children) < min_nodes:
            continue
        new_pathway_info[path] = info  # Append pathway to new_pathway_info if it has at least min_nodes
        remaining_nodes = remaining_nodes.union(children)  # Update remaining nodes
    # Form a list of nodes that are not in any of the remaining pathways
    if len(remaining_nodes) > 0:
        for_removal = [gene for gene in node_info.keys() if gene not in remaining_nodes]
        for gene in for_removal:
            node_info, edge_info = bpg.remove_node(gene, node_info, edge_info)
        # Restrict exprs_df to genes that remain in node_info. exprs_df has its columns (genes) ordered to match the
        # order of the keys in node_info
        all_features = exprs_df.columns.to_numpy()  # All genes in exprs_data
        sel_features = [list(all_features).index(gene) for gene in node_info.keys()]  # Indices of genes to keep
        feature_names = all_features[sel_features]  # Final genes to use
        exprs_df = exprs_df.loc[:, feature_names]  # Match the order of keys in node_info
    else:
        # There are no remaining nodes because all pathways were eliminated, so node_info, edge_info, and exprs_df
        # are emptied
        node_info = dict()
        edge_info = dict()
        exprs_df = exprs_df.loc[:, []]  # Empty DataFrame

    return exprs_df, node_info, edge_info, new_pathway_info


def make_assignment_matrix(
    sender_graph_info: NodeInfoDict,
    receiver_graph_info: NodeInfoDict,
    sender_index_name_map: Dict[int, str],
    receiver_index_name_map: Dict[int, str]
) -> Tensor:
    """
    During graph coarsening, node features in one graph are aggregated to serve as the initial node features for a
    coarsened graph with fewer nodes. Supposing that the sender graph (the graph to be coarsened) and the receiver
    graph (the coarsened graph) have already been defined, `make_assignment_matrix` returns a matrix that specifies
    the mapping of information from nodes in the sender graph to nodes in the receiver graph. For example,
    given a sender graph with nodes {A, B, C, D} and a receiver graph with nodes {P, Q}, the assignment matrix might
    specify that the features from node P derive their values from the features of nodes A and B, and Q's features
    derive their values from the features of C and D.

    :param sender_graph_info: A dictionary that defines the sender graph
    :param receiver_graph_info: A dictionary that defines the receiver and its nodes' relationships to sender nodes
    :param sender_index_name_map: A mapping of numeric indices to node names for the sender graph
    :param receiver_index_name_map: A mapping of numeric indices to node names for the receiver graph
    :return: An assignment matrix in Tensor form
    """
    n_rows = len(receiver_graph_info)
    n_cols = len(sender_graph_info)

    sender_name_list = list(sender_index_name_map.values())
    receiver_name_list = list(receiver_index_name_map.values())

    assignment_matrix = np.zeros(shape=(n_rows, n_cols), dtype=float)

    for i in range(n_rows):
        r_name = receiver_name_list[i]
        s_names = list(receiver_graph_info[r_name]["children"])  # Names of nodes in sender to receive info from
        j = [sender_name_list.index(val) for val in s_names]  # Indices of nodes in sender to receive info from
        assignment_matrix[i, j] = 1

    assignment_matrix = torch.tensor(assignment_matrix, dtype=torch.float32)

    return assignment_matrix


def list_source_and_target_indices(
        adjacency_dict: Dict[str, Set[str]], index_map: Dict[int, str]
) -> Tuple[List[int], List[int]]:
    """
    Returns a tuple with two equal-length lists of node indices, denoting an edge between the ith items of the
    lists. Edges are directed; an undirected edge is represented as two directed edges.

    :param adjacency_dict: A dict that indicates genes that are connected by an edge in KEGG pathways. The keys are
    KEGG gene ID strings, and the values are sets of KEGG gene ID strings that the keyed gene is connected with via
    an edge.
    :param index_map: A mapping from numeric node indices (zero-indexed) to KEGG gene ID strings.
    :return: A tuple of equal-length lists of nodes that are connected by an edge
    """
    source_idx = list()
    target_idx = list()
    index_map_val_list = list(index_map.values())  # For finding the node index that corresponds to the KEGG ID
    for idx, name in index_map.items():
        if name in adjacency_dict.keys():  # Not all nodes have outgoing edges
            # Add the node index to the source list once per node it shares an edge with
            updated_neighbors = set(neigh for neigh in adjacency_dict[name] if neigh in index_map.values())
            adjacency_dict[name] = updated_neighbors  # Only retain neighbors that are actually in the subgraph
            source_idx.extend(itertools.repeat(idx, len(adjacency_dict[name])))
            targets_to_add = list()
            for neigh in adjacency_dict[name]:  # iterates over a `set` object
                targets_to_add.append(index_map_val_list.index(neigh))
            target_idx.extend(targets_to_add)
    return source_idx, target_idx


def make_subgraph_info_dicts(
        nodes_dict: NodeInfoDict, adjacency_dict: AdjacencyDict, nodes: Set[str]
) -> Tuple[NodeInfoDict, AdjacencyDict]:
    """
    This function takes dictionaries of node and adjacency (edge) information that define a graph and returns a
    tuple of new dictionaries that define a subgraph on said graph. The subgraph contains nodes in the set passed to
    the `nodes` parameter. It is expected that `nodes_dict` is a dictionary keyed by node names. This function
    does not update information about the children of each node if that is one of the information fields; the
    children will have to be updated separately they are to include only those nodes that are in a subgraph that
    is immediately subordinate to the one being created in a graph coarsening hierarchy.

    :param nodes_dict: A nested dictionary of information about each node in the graph
    :param adjacency_dict: A dictionary giving the set of neighboring nodes that receive edges from the central node
    :param nodes: The nodes to use in the subgraph
    :return: A tuple whose first item is the subgraph's node information dictionary and whose second item is the
    subgraph's adjacency dictionary.
    """
    # If the adjacency dict has edge type information, it must be supplied as a tuple whose first item is a string
    # giving the node name and whose second item is a tuple of edge types.
    adjacency_has_edge_types = all(
        [isinstance(neighbor, tuple) for neighbor_set in adjacency_dict.values() for neighbor in neighbor_set]
    )
    nodes_dict_new = {k: nodes_dict[k] for k in nodes}
    adjacency_dict_new = dict()
    for k in nodes_dict_new.keys():
        try:
            neighbor_set = adjacency_dict[k]
        except KeyError:  # Not all nodes have neighbors
            continue
        adjacency_dict_new[k] = set()
        # Restrict neighbors to subsets of nodes present in the subgraph
        if adjacency_has_edge_types:
            for neighbor in neighbor_set:
                if neighbor[0] in nodes_dict_new.keys():
                    adjacency_dict_new[k].add(neighbor)
        else:
            for neighbor in neighbor_set:
                if neighbor in nodes_dict_new.keys():
                    adjacency_dict_new[k].add(neighbor)

    return nodes_dict_new, adjacency_dict_new
