import argparse
import io
import numpy as np
import os
import pandas as pd
import pickle
import re
import torch

from collections import OrderedDict
from filelock import FileLock
from torch.utils.data import SequentialSampler, DataLoader

from dataset_definitions import CancerDataset
from models import SparseMLP, MLPOutputBlock
from utilities import maybe_create_directories


def load_dataset(root: str, files=None) -> CancerDataset:
    """
    Loads a dataset from serialized data on disk.

    :param root: The path to the directory where the data are stored. Should contain a "raw" subdirectory containing
    the serialized data files.
    :param files: A collection of files to use from the 'raw' subdirectory of `root`
    :return: A CancerDataset
    """

    # If merging all pathways into a single large graph, the standardization occurs over all genes in all pathways.
    # If feeding one pathway graph through the NN at a time, standardization is isolated to the pathway's genes.
    data_files = sorted(os.listdir(os.path.join(root, "raw")))  # Raw graph Data object files
    if files is not None:
        data_files = [file for file in data_files if file in files]
    # transformation = StandardizeFeatures(correction=1)  # No transform
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerDataset(root=root, data_files=data_files)  # No transform
    return dataset


def main():

    # region Parse args
    parser = argparse.ArgumentParser(
        description="Export results from MLP models"
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
    # endregion Parse args

    # region Define important values
    data_dir = args["data_dir"]  # e.g. ./data
    output_dir = args["output_dir"]  # e.g. ./experiment6
    model_type = "mlp"
    use_drug_input = True if args["use_drug_input"] else False
    batch_size = args["batch_size"]
    # endregion Define important values

    # region Directories
    input_data_dir = os.path.join(data_dir, "mlp_inputs")
    model_dir = os.path.join(output_dir, "models", model_type)
    export_dir = os.path.join(output_dir, "exports", model_type)
    # Check for the existence of directories and create them if necessary
    maybe_create_directories(export_dir)
    # endregion Directories

    # region Files to read
    model_file = os.path.join(model_dir, f"final_model-type={model_type}_drug-input={use_drug_input}.pt")
    pathway_names_file = os.path.join(data_dir, f"{model_type}_pathway_names.npy")
    mask_matrix_file = os.path.join(data_dir, "mlp_mask.pt")
    # endregion Files to read

    # region Load data
    # Load pathway name in the order they appear in the first hidden layer of the MLP
    pathway_names = np.load(pathway_names_file, allow_pickle=True)

    # Load weight mask for SparseMLP Module
    with open(mask_matrix_file, "rb") as file_in:
        buffer = io.BytesIO(file_in.read())
    mask = torch.load(buffer)

    # Load training and test datasets
    with open(os.path.join(data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        train_val_test_names = pickle.load(file_in)
    train_names, _, test_names = train_val_test_names
    # Create training, validation datasets
    train_dataset = load_dataset(input_data_dir, train_names)
    test_dataset = load_dataset(input_data_dir, test_names)
    # endregion Load data

    # Initialize dataloaders
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)

    # Load saved intervals, input feature indices to use, and model weights for the current fold
    saved_model_objects = torch.load(model_file)
    sparse_layer_state, model_state, _ = saved_model_objects

    # Edit the sparse layer state dict keys
    new_sparse_layer_state = OrderedDict()
    for key, val in sparse_layer_state.items():
        key = re.sub(r"\bmodule\.", "", key)
        new_sparse_layer_state[key] = val
    sparse_layer_state = new_sparse_layer_state
    del new_sparse_layer_state

    # Edit the neural network block state dict keys
    new_model_state = OrderedDict()
    for key, val in model_state.items():
        key = re.sub(r"\bmodule\.", "", key)
        new_model_state[key] = val
    model_state = new_model_state
    del new_model_state

    # Initialize models
    if use_drug_input:
        n_aux_feats = len(train_dataset[0][1])  # Number of drug categories
        total_feats = mask.shape[0] + n_aux_feats
    else:
        total_feats = mask.shape[0]
    sparse_layer = SparseMLP(mask)
    sparse_layer.load_state_dict(sparse_layer_state)
    model = MLPOutputBlock(in_features=total_feats)
    model.load_state_dict(model_state)

    # Forward passes
    sparse_layer.eval()
    model.eval()
    with torch.no_grad():
        # Training data
        train_pathway_scores = []
        train_predictions = []
        train_labels = []

        for feat_tensor, aux_feat_tensor, targets in train_dataloader:
            current_batch_size = feat_tensor.shape[0]
            targets = torch.reshape(targets, (current_batch_size, -1))
            pathway_scores = sparse_layer(feat_tensor)
            # If drugs were used as input to the model, concatenate first hidden layer's output with drug data
            if use_drug_input:
                feat_tensor = torch.cat([pathway_scores, aux_feat_tensor], dim=-1)
            else:
                feat_tensor = pathway_scores

            predictions = model(feat_tensor)

            # Append results to lists
            train_pathway_scores.append(pathway_scores)
            train_predictions.append(predictions)
            train_labels.append(targets)

        # Test data
        test_pathway_scores = []
        test_predictions = []
        test_labels = []
        for feat_tensor, aux_feat_tensor, targets in test_dataloader:
            current_batch_size = feat_tensor.shape[0]
            targets = torch.reshape(targets, (current_batch_size, -1))
            pathway_scores = sparse_layer(feat_tensor)
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
    if use_drug_input:
        print(f"Exporting results from {model_type} models with drug inputs")
    else:
        print(f"Exporting results from {model_type} models without drug inputs")

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
