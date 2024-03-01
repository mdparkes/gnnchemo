import argparse
import numpy as np
import os
import pickle
import re

import pandas as pd
import torch

from collections import OrderedDict
from filelock import FileLock
from torch_geometric.loader import DataLoader
from torch.utils.data import SequentialSampler

from dataset_definitions import CancerGraphDataset
from models import IndividualPathsMPNN, MLPOutputBlock
from utilities import maybe_create_directories


def load_dataset(graph_dir: str) -> CancerGraphDataset:
    graph_files = sorted(os.listdir(os.path.join(graph_dir, "raw")))  # Raw graph Data object files
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerGraphDataset(root=graph_dir, data_files=graph_files)
    return dataset


def mp_module_forward_pass(data_batch_list, mp_modules):
    """
    Perform a forward pass through the GNN message passing modules. Returns a Tensor of pathway scores and a list of
    graph node indices that were retained after each SAGPool operation.

    :param data_batch_list: A list of DataBatch or HeteroDataBatch objects, one per input graph, each representing a
    batch of biopsies.
    :param mp_modules: A ModuleList with the message passing module for each input graph
    :return: A pathway scores Tensor for a batch of biopsies and a list of tuples of indices of nodes retained by each
    SAGPool layer for each input graph with a batch of biopsies.
    """
    pathway_scores = list()  # Populated with a list of [n_pathways] shape [batch_size, 1] tensors of pathway scores
    nodes_retained_list = list()
    for i, graph in enumerate(data_batch_list):
        score, _, nodes_retained = mp_modules[i](graph.x, graph.edge_index, graph.batch)
        pathway_scores.append(score)
        nodes_retained_list.append(nodes_retained)
    pathway_scores = torch.cat(pathway_scores, dim=-1)  # shape [batch_size, n_pathways]
    return pathway_scores, nodes_retained_list


def main():
    # region Parse args
    parser = argparse.ArgumentParser(
        description="Export results from gnn models"
    )
    parser.add_argument(
        "-d", "--data_dir",
        help="The path of the directory where data necessary for training the models are located",
        type=str
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="The path of the directory where experimental results will be written",
        type=str
    )
    parser.add_argument(
        "--use_drug_input",
        help="If set, use drug types administered as auxiliary features for input into the MLP block that outputs the"
        "final prediction of treatment response",
        action="store_true"
    )
    parser.add_argument(
        "-b", "--batch_size",
        help="The mini-batch size for training and testing",
        type=int
    )
    args = vars(parser.parse_args())

    # # For interactive debugging
    # args = {
    #     "data_dir": "Users/pr3/Projects/gnnchemo/data",
    #     "output_dir": "Users/pr3/Projects/gnnchemo/test_expt",
    #     "use_drug_input": True,
    #     "batch_size": 48
    # }
    # endregion Parse args

    # region Define important values
    data_dir = args["data_dir"]  # e.g. ./data
    output_dir = args["output_dir"]  # e.g. ./experiment6
    model_type = "gnn"
    use_drug_input = True if args["use_drug_input"] else False
    batch_size = args["batch_size"]
    # endregion Define important values

    # region Directories
    graph_data_dir = os.path.join(data_dir, "graphs")
    model_dir = os.path.join(output_dir, "models", model_type)
    export_dir = os.path.join(output_dir, "exports", model_type)
    # Check for the existence of directories and create them if necessary
    maybe_create_directories(export_dir)
    # endregion Directories

    # region Files to read/write
    graph_data_files = sorted(os.listdir(os.path.join(data_dir, "graphs", "raw")))  # Raw graph Data object files
    model_file = os.path.join(model_dir, f"final_model-type={model_type}_drug-input={use_drug_input}.pt")
    pathway_names_file = os.path.join(data_dir, f"{model_type}_pathway_names.npy")
    # endregion Files to read/write

    # Load pathway names
    pathway_names = np.load(pathway_names_file, allow_pickle=True)

    # Load training and test datasets
    with open(os.path.join(data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        train_val_test_names = pickle.load(file_in)
    train_names, _, test_names = train_val_test_names
    # Get the indices of the current CV fold's raw Data files in the file list
    train_idx = [graph_data_files.index(name) for name in train_names]
    test_idx = [graph_data_files.index(name) for name in test_names]
    # Load dataset
    ds = load_dataset(graph_data_dir)
    train_dataset = ds[train_idx]
    test_dataset = ds[test_idx]

    # Initialize dataloaders
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)

    # Load saved intervals, input feature indices to use, and model weights for the current fold
    saved_model_objects = torch.load(model_file)
    mp_modules_state, model_state, _ = saved_model_objects

    # Edit the sparse layer state dict keys
    new_mp_modules_state = OrderedDict()
    for key, val in mp_modules_state.items():
        key = re.sub(r"\bmodule\.", "", key)
        new_mp_modules_state[key] = val
    mp_modules_state = new_mp_modules_state
    del new_mp_modules_state

    # Edit the neural network MTLR block state dict keys
    new_model_state = OrderedDict()
    for key, val in model_state.items():
        key = re.sub(r"\bmodule\.", "", key)
        new_model_state[key] = val
    model_state = new_model_state
    del new_model_state

    # Initialize models
    mp_modules = torch.nn.ModuleList()
    n_submodules = len(ds[0][0][0])  # Number of different pathways
    if use_drug_input:
        n_aux_feats = len(ds[0][0][1])  # Number of drug categories
        total_feats = n_submodules + n_aux_feats
    else:
        total_feats = n_submodules
    for i in range(n_submodules):
        num_nodes = int(ds[0][0][0][i].x.size(0))  # Number of nodes in the pathway graph
        mp_module = IndividualPathsMPNN(message_passing="graphsage", use_sagpool=True, ratio=0.7, num_nodes=num_nodes)
        mp_modules.append(mp_module)
    mp_modules.load_state_dict(mp_modules_state)

    model = MLPOutputBlock(in_features=total_feats)
    model.load_state_dict(model_state)

    # Forward passes
    # A forward pass through the model returns [batch_size, n_intervals] logits that express the log-probability
    # that the event occurred in each interval. To get the cumulative survival probability distribution,
    # exponentiate the logits,
    mp_modules.eval()
    model.eval()
    with torch.no_grad():
        # Training data
        train_pathway_scores = []
        train_predictions = []
        train_labels = []
        for (data_batch_list, aux_feat_tensor), targets in train_dataloader:
            current_batch_size = len(data_batch_list[0])
            targets = torch.reshape(targets, (current_batch_size, -1))
            pathway_scores, _ = mp_module_forward_pass(data_batch_list, mp_modules)
            # If drugs were used as input to the model, concatenate GNN latent representation with drug data
            if use_drug_input:
                feat_tensor = torch.cat([pathway_scores, aux_feat_tensor], dim=-1)
            else:
                feat_tensor = pathway_scores
            predictions = model(feat_tensor)

            train_pathway_scores.append(pathway_scores)
            train_predictions.append(predictions)
            train_labels.append(targets)

        # Test data
        test_pathway_scores = []
        test_predictions = []
        test_labels = []
        for (data_batch_list, aux_feat_tensor), targets in test_dataloader:
            current_batch_size = len(data_batch_list[0])
            targets = torch.reshape(targets, (current_batch_size, -1))
            pathway_scores, _ = mp_module_forward_pass(data_batch_list, mp_modules)
            # If drugs were used as input to the model, concatenate GNN latent representation with drug data
            if use_drug_input:
                feat_tensor = torch.cat([pathway_scores, aux_feat_tensor], dim=-1)
            else:
                feat_tensor = pathway_scores
            predictions = model(feat_tensor)

            test_pathway_scores.append(pathway_scores)
            test_predictions.append(predictions)
            test_labels.append(targets)
            
    # region Export data
    train_pathway_scores = torch.cat(train_pathway_scores, dim=0).numpy()
    test_pathway_scores = torch.cat(test_pathway_scores, dim=0).numpy()

    train_predictions = torch.cat(train_predictions, dim=0).numpy()
    test_predictions = torch.cat(test_predictions, dim=0).numpy()

    train_label_array = torch.cat(train_labels, dim=0).numpy()
    test_label_array = torch.cat(test_labels, dim=0).numpy()

    train_pathway_scores_df = pd.DataFrame(train_pathway_scores, columns=pathway_names, index=train_names)
    test_pathway_scores_df = pd.DataFrame(test_pathway_scores, columns=pathway_names, index=test_names)

    train_predictions_df = pd.DataFrame({
        "prediction": train_predictions.squeeze(),
        "label": train_label_array.squeeze()
    }, index=train_names)
    test_predictions_df = pd.DataFrame({
        "prediction": test_predictions.squeeze(),
        "label": test_label_array.squeeze()
    }, index=test_names)

    # Write data to csv files
    shared_prefix = f"model-type={model_type}_drug-input={use_drug_input}"

    file_path = os.path.join(export_dir, f"{shared_prefix}_train_pathway_scores_df.csv")
    train_pathway_scores_df.to_csv(file_path)

    file_path = os.path.join(export_dir, f"{shared_prefix}_test_pathway_scores_df.csv")
    test_pathway_scores_df.to_csv(file_path)

    file_path = os.path.join(export_dir, f"{shared_prefix}_train_predictions_df.csv")
    train_predictions_df.to_csv(file_path)

    file_path = os.path.join(export_dir, f"{shared_prefix}_test_predictions_df.csv")
    test_predictions_df.to_csv(file_path)
    # endregion Export data


if __name__ == "__main__":
    main()
