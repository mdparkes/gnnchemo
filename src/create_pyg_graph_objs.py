"""
Script that creates PyTorch graph Data objects.
"""

import argparse
import numpy as np
import os
import pandas as pd
import pickle
import re
import torch

from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Dict

from custom_data_types import AdjacencyDict
from utilities import filter_datasets_and_graph_info, list_source_and_target_indices, map_indices_to_names


def make_edge_set_tensor(
        edge_set: str,
        adjacency_dict: AdjacencyDict,
        index_name_map: Dict[int, str],
        is_relational: bool) -> Tensor:
    """
    Creates an edge set Tensor for the specified edge set.

    :param edge_set: A string corresponding to one of the edge types in adjacency_dict
    :param adjacency_dict: An AdjacencyDict object listing target nodes and edge types for each source node
    :param index_name_map: A mapping of node indices in the GraphTensor node set keys/vals in adjacency_dict
    :param is_relational: True if graph is relational (multiple edge sets), False if graph is non-relational (single
    edge set)
    :return: An edge set Tensor for a specific edge set of a GraphTensor
    """
    edge_set_adjacency_dict = dict()
    # For each node in the graph, get targets connected to the source node by the specified edge type
    # Each key (source) is a GeneName, and each value (target set) is a Set[Tuple[GeneName, Tuple[EdgeTypeName]]
    # If the graph is non-relational (single edge set), ignore the edge types
    if is_relational:  # Relational graph with multiple edge sets
        for source, targets in adjacency_dict.items():
            edge_set_targets = set(t[0] for t in targets if edge_set in t[1])
            if len(edge_set_targets) > 0:
                edge_set_adjacency_dict[source] = edge_set_targets
    else:  # Non-relational graph with a single edge set
        for source, targets in adjacency_dict.items():
            edge_set_targets = set(t[0] for t in targets)
            if len(edge_set_targets) > 0:
                edge_set_adjacency_dict[source] = edge_set_targets

    source_indices, target_indices = list_source_and_target_indices(edge_set_adjacency_dict, index_name_map)
    edge_tns = torch.tensor([source_indices, target_indices], dtype=torch.long)
    return edge_tns


def main():
    # region Parse command line args
    parser = argparse.ArgumentParser(
        description="Creates PyG Data objects (graphs) from biopsy RNAseq data and writes them to disk."
    )
    parser.add_argument(
        "-e", "--exprs_file",
        help="The path to the csv file containing RNA expression data",
        default="data/tcga_exprs.csv"
    )
    parser.add_argument(
        "-d", "--drug_file",
        help="The path to the csv file containing the drug data",
        default="data/processed_drug_df.csv"
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="The path to the directory where the Data objects will be written. Graph Data objects will be written"
             "a subdirectory of output_dir, that is, ./[output_dir]/graphs, which will be created if it does not "
             "exist.",
        type=str
    )
    args = vars(parser.parse_args())
    # endregion Parse command line args

    data_dir = args["output_dir"]
    exprs_file = args["exprs_file"]  # File: Gene expression data
    drug_file = args["drug_file"]  # File: Drug response data
    graph_info_file = os.path.join(data_dir, "reactome_graph_directed.pkl")  # File: Reactome graph data
    feature_names_file = os.path.join(data_dir, "gnn_feature_names.pkl")  # File: KEGG IDs of genes used in graphs
    pathway_names_file = os.path.join(data_dir, "gnn_pathway_names.npy")  # File: Pathway IDs represented by graphs

    # Check for necessary files and directories
    if not os.path.exists(graph_info_file):
        raise FileNotFoundError("Run create_reactome_graph.py to create Reactome graph information objects.")

    # Load all pathway graphs' collective node and edge info
    with open(graph_info_file, "rb") as file_in:
        graph_info = pickle.load(file_in)

    # Load expression data and drug data
    exprs_data = pd.read_csv(exprs_file, index_col=0)
    drug_data = pd.read_csv(drug_file, index_col="aliquot_submitter_id")
    drug_data = drug_data.replace({True: 1, False: 0, "Positive response": 1, "Minimal response": 0})

    # Get a list of .pkl files containing node and edge info for each individual pathway
    node_info_dict_dir = os.path.join(data_dir, "Pathway", "pathways", "dicts")
    rx = re.compile(r".+_directed\.pkl")
    node_info_files = [f for f in sorted(os.listdir(node_info_dict_dir)) if rx.search(f)]
    node_info_files = [os.path.join(node_info_dict_dir, f) for f in node_info_files]

    # region Create the graph information dictionaries
    level_c_dict = graph_info[2][0].copy()
    # Create one graph object per pathway and save as a collection per patient
    tmp_data_dict = {}  # Accumulate graph objects for each patient
    feature_names = []  # Updated with feature names as individual pathway graphs are processed
    pathway_names = []  # Updated with pathway names as they are read
    drugs_administered = drug_data.iloc[:, 3:-1]  # Select binary columns indicating drug types administered
    drug_response = drug_data.loc[:, "measure_of_response"]
    assert(np.alltrue(drugs_administered.index == exprs_data.index))
    # Load individual pathway info dicts from disk
    rx = re.compile(r"[^/]+(?=_directed\.pkl)")  # matches the name of the reactome pathway
    # node_info_files was sorted to guarantee predictable order over pathways
    for file in tqdm(node_info_files, total=len(node_info_files), desc="Creating graph info objects"):
        with open(file, "rb") as file_in:
            level_d_dict, level_d_adj = pickle.load(file_in)
        path = rx.search(file)[0]
        # create a small single-pathway dict for faster filtering
        single_path_dict = {path: level_c_dict[path]}
        path_exprs_ss, level_d_dict, level_d_adj, _ = filter_datasets_and_graph_info(
            exprs_data, level_d_dict, level_d_adj, single_path_dict, min_nodes=15
        )
        if len(level_d_dict) == 0:
            # Skip graph creation for this pathway if it had fewer than 15 nodes with expression data
            continue
        all_features = path_exprs_ss.columns.to_list()  # All features in exprs_data
        features_used = [feat for feat in all_features if feat in level_d_dict.keys()]
        feature_names.append(features_used)  # Final features to use
        pathway_names.append(path)  # Record pathway name

        index_name_map = map_indices_to_names(level_d_dict)
        edge_index_tensor = make_edge_set_tensor("relation", level_d_adj, index_name_map, is_relational=False)
        for j, pt_id in enumerate(path_exprs_ss.index):
            node_tensor = torch.reshape(torch.tensor(path_exprs_ss.iloc[j].tolist()), shape=(-1, 1))
            graph_obj = Data(x=node_tensor.float(), edge_index=edge_index_tensor)
            # graph_obj["drugs_administered"] = torch.tensor(drugs_administered.iloc[j].to_list(), dtype=torch.float32)
            # graph_obj["drug_response"] = torch.tensor(drug_response.iat[j], dtype=torch.uint8)
            if pt_id not in tmp_data_dict.keys():
                tmp_data_dict[pt_id] = [graph_obj]
            else:
                tmp_data_dict[pt_id].append(graph_obj)

    with open(feature_names_file, "wb") as file_out:
        pickle.dump(feature_names, file_out)

    pathway_names = np.array(pathway_names)
    np.save(pathway_names_file, pathway_names)

    # write dataset to disk
    print("Writing graphs to disk", end="... ", flush=True)
    output_dir = os.path.join(data_dir, "graphs", "raw")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Records are Tuple[Tuple[List[graph], drugs], response]
    for j, pt_id, graphs in enumerate(tmp_data_dict.items()):
        drugs = torch.tensor(drugs_administered.loc[pt_id].tolist(), dtype=torch.float32)
        response = torch.tensor(drug_response.iat[j], dtype=torch.uint8)
        record = ((graphs, drugs), response)
        file_out = os.path.join(output_dir, f"{pt_id}.pt")
        torch.save(record, file_out)
    print("Done")
    # endregion Create the graph information dictionaries

    # Assign each combination of drug therapies and responses to a numbered stratum. Each combination of bits
    # observed in drugs_administered is represented as a barcode string. The observed unique combinations of drug
    # barcodes with responses are the strata for balancing that data splits.
    strata = pd.concat([drugs_administered, drug_response], axis=1).values.tolist()
    strata_strings = [''.join(str(bit) for bit in x) for x in strata]
    strata = list(set(strata_strings))
    strata_by_barcode = dict(zip(strata, range(len(strata))))
    stratum = [strata_by_barcode[key] for key in strata_strings]
    stratum = pd.Series(data=stratum, index=drugs_administered.index)
    # Stratified train_test_split() requires at least two members per stratum, but some strata have only one member.
    # Create a new stratum numbered -1 and reassign strata with only one member to this new stratum.
    counts = stratum.value_counts()
    singletons = counts[counts == 1].index.tolist()
    sel = stratum.isin(singletons)
    stratum.iloc[sel] = -1

    # Three-way train/val/test split stratified on drug response and drugs administered
    path_out = os.path.join(data_dir, "train_test_split_names.pkl")
    if not os.path.exists(path_out):
        print("Indexing training, validation, and test splits", end="... ", flush=True)
        bx_names = stratum.index.tolist()
        bx_names, test_names = train_test_split(bx_names, test_size=0.2, random_state=423, shuffle=True,
                                                stratify=stratum)
        stratum = stratum.loc[bx_names]
        train_names, val_names = train_test_split(bx_names, test_size=0.2, random_state=822, shuffle=True,
                                                  stratify=stratum)
        train_names = [f"{name}.pt" for name in train_names]
        val_names = [f"{name}.pt" for name in val_names]
        test_names = [f"{name}.pt" for name in test_names]
        with open(path_out, "wb") as file_out:
            pickle.dump([train_names, val_names, test_names], file_out)
        print("Done")


if __name__ == "__main__":
    main()
