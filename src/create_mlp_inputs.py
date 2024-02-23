"""
Write MLP surival model data to disk. The create_pyg_graph_objs.py script should be run before running this script
because this script needs to restrict the input tensors to features that are used in the inputs to GNN survival models.
"""

import argparse
import numpy as np
import os
import pandas as pd
import pickle
import torch

from keggpathwaygraphs import biopathgraph as bpg
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utilities import filter_datasets_and_graph_info, make_assignment_matrix, map_indices_to_names


def main():

    # region Parse command line args
    parser = argparse.ArgumentParser(
        description="Creates tensors for input to MLP models."
    )
    parser.add_argument(
        "-e", "--exprs_file",
        help="The path to the csv file containing RNA expression data",
        default="data/tcga_exprs.csv"

    )
    parser.add_argument(
        "-d", "--drug_file",
        help="The path to the csv file containing the clinical data",
        default="data/tcga_clin.csv"
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="The path to the directory where the Data objects will be written. For each cancer type passed as an "
             "argument to --cancer_types, graph Data objects will be written to ./[output_dir]/[cancer_type]/graphs.",
        type=str
    )
    args = vars(parser.parse_args())
    # endregion Parse command line args

    output_dir = args["output_dir"]
    exprs_file = args["exprs_file"]
    drug_file = args["drug_file"]
    graph_info_file = os.path.join(output_dir, "reactome_graph_directed.pkl")  # Reactome ptahway graph information
    feature_names_file = os.path.join(output_dir, "mlp_feature_names.pkl")  # KEGG IDs of genes input to MLP
    pathway_names_file = os.path.join(output_dir, "mlp_pathway_names.npy")  # Pathway IDs in MLP hidden layer 1
    assignment_matrix_file = os.path.join(output_dir, "mlp_mask.pt")  # MLP hidden layer 1 input mask

    # Load expression data and drug data
    exprs_data = pd.read_csv(exprs_file, index_col=0)
    drug_data = pd.read_csv(drug_file, index_col="aliquot_submitter_id")
    drug_data = drug_data.replace({True: 1, False: 0, "Positive response": 1, "Minimal response": 0})

    # Load graph info dicts (used for gene and pathway filtering prior to MLP input tensor construction)
    with open(graph_info_file, "rb") as file_in:
        graph_info = pickle.load(file_in)

    # Make working copies of the graph data
    level_c_dict = graph_info[2][0].copy()
    level_d_dict, level_d_adj = graph_info[3][0].copy(), graph_info[3][1].copy()
    # Filter data to include only relevant features and observations
    exprs_ss, level_d_dict, _, level_c_dict = filter_datasets_and_graph_info(
        exprs_data, level_d_dict, level_d_adj, level_c_dict, min_nodes=15
    )
    # Write feature names in order of appearance in the MLP input tensors
    feature_names = [[feat for feat in exprs_ss.columns.to_list() if feat in level_d_dict.keys()]]
    with open(feature_names_file, "wb") as file_out:
        pickle.dump(feature_names, file_out)
    # Write pathway names that correspond to the units of the MLP's first hidden layer
    pathway_names = list(level_c_dict.keys())
    np.save(pathway_names_file, np.array(pathway_names))
    # Create weight mask for sparse MLP
    level_c_dict = bpg.update_children(level_c_dict, set(level_d_dict.keys()))
    pooling_assignment = make_assignment_matrix(
        sender_graph_info=level_d_dict, sender_index_name_map=map_indices_to_names(level_d_dict),
        receiver_graph_info=level_c_dict, receiver_index_name_map=map_indices_to_names(level_c_dict)
    )
    torch.save(pooling_assignment, assignment_matrix_file)

    # Create input tensors
    drugs_administered = drug_data.iloc[:, 3:-1]  # Select binary columns indicating drug types administered
    drug_response = drug_data.loc[:, "measure_of_response"]

    path_out = os.path.join(output_dir, "mlp_inputs", "raw")
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    # Data are written as a tuple of tensors: Tuple[gene_expression, drugs_administered, drug_response]
    for j, pt_id in tqdm(enumerate(exprs_ss.index), total=len(exprs_ss), desc="Writing MLP input data to disk"):
        exprs_tensor = torch.tensor(exprs_ss.loc[pt_id].tolist(), dtype=torch.float32)
        drugs_tensor = torch.tensor(drugs_administered.loc[pt_id].tolist(), dtype=torch.float32)
        response_tensor = torch.tensor(drug_response.iat[j], dtype=torch.uint8)
        file_out = os.path.join(path_out, f"{pt_id}.pt")
        torch.save(obj=(exprs_tensor, drugs_tensor, response_tensor), f=file_out)

    # Three-way train/val/test split stratified on drug response and drugs administered
    path_out = os.path.join(output_dir, "train_test_split_names.pkl")
    if not os.path.exists(path_out):
        print("Indexing training, validation, and test folds", end="... ", flush=True)
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
        # Make splits
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
