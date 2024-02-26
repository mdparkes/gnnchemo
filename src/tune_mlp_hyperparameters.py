"""
Perform hyperparameter tuning using a fold of data that is exclusively reserved for this purpose.

Hyperparameter tuning is only performed with models that do not use feature selection. The tuned hyperparameters will
be shared with the models that use feature selection. For the purpose of tuning, MLP models can differ in
the type of database the pathways were derived from (KEGG/BRITE, Reactome) and whether the counterpart GNN was over
graphs that had only directed edges or both directed and undirected edges. If a GNN only used graphs with directed
edges, its node set only contains genes that participate in directed edges. If a GNN used both directed and undirected
edges, its node set contains genes that participate in either directed or undirected edges. In both styles of GNN,
genes that do not participate in any edge are absent from the node set. Since each MLP counterpart to a GNN must have
exactly the same genes as input, the MLP input depends on the directedness of the graphs passed through the GNN.
"""

import argparse
import io
import numpy as np
import os
import pickle
import tempfile
import torch

from filelock import FileLock
from matplotlib.ticker import MaxNLocator
from ray import tune, train
from ray.train import Checkpoint
from torch import Tensor
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import BatchSampler, Dataset, SubsetRandomSampler
from torch_geometric.loader import DataLoader
from typing import Any, Dict, Sequence

from dataset_definitions import CancerDataset
from models import MLPOutputBlock, SparseMLP
from utilities import maybe_create_directories


def load_dataset(root: str) -> CancerDataset:
    """
    Loads a dataset from serialized data on disk.

    :param root: The path to the directory where the data are stored. Should contain a "raw" subdirectory containing
    the serialized data files.
    :return: A CancerDataset
    """
    # If merging all pathways into a single large graph, the standardization occurs over all genes in all pathways.
    # If feeding one pathway graph through the NN at a time, standardization is isolated to the pathway's genes.
    data_files = sorted(os.listdir(os.path.join(root, "raw")))  # Raw graph Data object files
    # transformation = StandardizeFeatures(correction=1)  # No transform
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerDataset(root=root, data_files=data_files)  # No transform
    return dataset


def train_loop(
        config: Dict[str, Any],
        *,
        dataset: Dataset,
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        batch_size: int,
        pathway_mask: Tensor,
        use_aux_feats: bool
) -> None:

    epochs = 50

    # region Create dataloaders
    # Create samplers that partition stratified CV folds into disjoint random batches
    train_batch_sampler = BatchSampler(SubsetRandomSampler(train_indices), batch_size, drop_last=False)
    val_batch_sampler = BatchSampler(SubsetRandomSampler(val_indices), batch_size, drop_last=False)
    train_dataloader = DataLoader(dataset, batch_sampler=train_batch_sampler)
    val_dataloader = DataLoader(dataset, batch_sampler=val_batch_sampler)
    # endregion Create dataloaders

    # region Initialize models and optimizer
    if use_aux_feats:
        n_aux_feats = len(dataset[0][1])  # Number of drug categories
        total_feats = pathway_mask.shape[0] + n_aux_feats
    else:
        total_feats = pathway_mask.shape[0]

    sparse_layer = SparseMLP(pathway_mask)

    model = MLPOutputBlock(in_features=total_feats)

    optimizer = torch.optim.Adam([
        {"params": sparse_layer.parameters()},
        {"params": model.parameters()}
    ], lr=config["lr"], weight_decay=config["weight_decay"])
    # endregion Initialize models and optimizer

    # Check for and load checkpoint
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            ckpt = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch, sparse_layer_state, model_state, optimizer_state = ckpt
            sparse_layer.load_state_dict(sparse_layer_state)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    else:
        start_epoch = 0

    for t in range(start_epoch, epochs):

        # region Train
        sparse_layer.train()
        model.train()
        epoch_train_loss = 0.
        samples_processed = 0
        for loaded_data in train_dataloader:

            feature_tensor, aux_feat_tensor, targets = loaded_data
            current_batch_size = feature_tensor.shape[0]
            samples_processed += current_batch_size

            targets = torch.reshape(targets, (current_batch_size, -1))
            inputs = sparse_layer(feature_tensor)
            if use_aux_feats:
                inputs = torch.cat([inputs, aux_feat_tensor], dim=-1)
            predictions = model(inputs)
            predictions = torch.reshape(predictions, (current_batch_size, -1))
            # Calculate training loss for the batch
            # Positive responders represent ~ 76% of the dataset. Rescale the losses accordingly.
            pos_wt = torch.ones(size=[current_batch_size, 1], dtype=torch.float32)
            neg_wt = torch.full(size=[current_batch_size, 1], fill_value=759 / 242, dtype=torch.float32)
            rescaling_weights = torch.where(targets.bool(), pos_wt, neg_wt)
            rescaling_weights = rescaling_weights.reshape((current_batch_size, 1))
            loss = binary_cross_entropy(predictions, targets, weight=rescaling_weights, reduction="sum")
            epoch_train_loss += loss  # Running total of this epoch's training loss
            loss /= current_batch_size  # Per-patient loss for current batch
            # Optimize model weights w.r.t. per-patient batch loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= samples_processed  # Per-patient average training loss for this epoch
        # endregion Train

        # region Evaluate
        sparse_layer.eval()
        model.eval()
        epoch_val_loss = 0.
        samples_processed = 0
        with torch.no_grad():
            for loaded_data in val_dataloader:
                feature_tensor, aux_feat_tensor, targets = loaded_data
                current_batch_size = feature_tensor.shape[0]
                samples_processed += current_batch_size
                targets = torch.reshape(targets, (current_batch_size, -1))
                inputs = sparse_layer(feature_tensor)
                if use_aux_feats:
                    inputs = torch.cat([inputs, aux_feat_tensor], dim=-1)
                predictions = model(inputs)
                predictions = torch.reshape(predictions, (current_batch_size, -1))
                # Calculate training loss for the batch
                # Positive responders represent ~ 76% of the dataset. Rescale the losses accordingly.
                pos_wt = torch.ones(size=[current_batch_size, 1], dtype=torch.float32)
                neg_wt = torch.full(size=[current_batch_size, 1], fill_value=759 / 242, dtype=torch.float32)
                rescaling_weights = torch.where(targets.bool(), pos_wt, neg_wt)
                rescaling_weights = rescaling_weights.reshape((current_batch_size, 1))
                loss = binary_cross_entropy(predictions, targets, weight=rescaling_weights, reduction="sum")
                epoch_val_loss += loss  # Running total of this epoch's validation loss
        epoch_val_loss /= samples_processed
        # endregion Evaluate

        # Checkpointing
        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            torch.save(
                (t, sparse_layer.state_dict(), model.state_dict(), optimizer.state_dict()),
                os.path.join(tmp_ckpt_dir, "checkpoint.pt")
            )
            train.report(
                metrics={
                    "training_loss": float(epoch_train_loss),
                    "validation_loss": float(epoch_val_loss)
                },
                checkpoint=Checkpoint.from_directory(tmp_ckpt_dir)
            )


def main():

    # region Parse args
    parser = argparse.ArgumentParser(
        description="Tunes the hyperparameters of a sparse MLP model"
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
    #
    # # For interactive debugging
    # args = {
    #     "data_dir": "data",
    #     "output_dir": "test_expt",
    #     "use_drug_input": True,
    #     "batch_size": 48
    # }

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
    ckpt_dir = os.path.join(output_dir, "checkpoints", model_type)
    export_dir = os.path.join(output_dir, "exports", model_type)
    hp_dir = os.path.join(output_dir, "hyperparameters", model_type)
    # Check for the existence of directories and create them if necessary
    maybe_create_directories(model_dir, ckpt_dir, export_dir, hp_dir)
    # endregion Directories

    # region Files to read/write
    input_data_files = sorted(os.listdir(os.path.join(input_data_dir, "raw")))  # Raw graph Data object files
    # Weight mask for first hidden layer of MLP
    mask_matrix_file = os.path.join(data_dir, "mlp_mask.pt")
    feature_names_file = os.path.join(data_dir, f"{model_type}_feature_names.pkl")  # HSA/ENTREZ IDs of input genes
    pathway_names_file = os.path.join(data_dir, f"{model_type}_pathway_names.npy") # HSA IDs of pathways
    hp_file = os.path.join(hp_dir, f"{model_type}_hyperparameters.pkl")
    # endregion Files to read/write

    # Load weight mask for SparseMLP Module
    with open(mask_matrix_file, "rb") as file_in:
        buffer = io.BytesIO(file_in.read())
    mlp_mask_matrix = torch.load(buffer)

    # Load pathway names
    pathway_names = np.load(pathway_names_file, allow_pickle=True)
    assert(len(pathway_names) == mlp_mask_matrix.shape[0])

    # Get indices of train/validation/test biopsies in the dataset and load dataset
    with open(os.path.join(data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        train_test_names = pickle.load(file_in)
    train_names, val_names, _ = train_test_names
    train_idx = [input_data_files.index(name) for name in train_names]
    val_idx = [input_data_files.index(name) for name in val_names]
    ds = load_dataset(input_data_dir)

    # region Perform hyperparameter tuning
    # Hyperparameter options
    hp_dict = {
        "weight_decay": tune.grid_search([5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3]),
        "lr": tune.grid_search([1e-4, 1e-3])
    }
    train_model = tune.with_parameters(
        train_loop, dataset=ds, train_indices=train_idx, val_indices=val_idx, batch_size=batch_size,
        pathway_mask=mlp_mask_matrix, use_aux_feats=True
    )

    storage_path = os.path.abspath(ckpt_dir)
    expt_name = f"tuning_model-type={model_type}_drug-input={use_drug_input}"
    expt_storage_path = os.path.join(storage_path, expt_name)
    if tune.Tuner.can_restore(expt_storage_path):
        # Auto-resume experiment after fault occurred or restore completed tuning experiment
        tuner = tune.Tuner.restore(expt_storage_path, trainable=train_model, resume_errored=True)
    else:
        tuner = tune.Tuner(
            train_model,
            param_space=hp_dict,
            run_config=train.RunConfig(
                storage_path=storage_path,
                name=expt_name,
                failure_config=train.FailureConfig(max_failures=1),
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_score_attribute="validation_loss",
                    checkpoint_score_order="min",
                    num_to_keep=5  # Keep the five checkpoints with the lowest validation losses
                )
            ),
        )
    results = tuner.fit()
    best_result = results.get_best_result(metric="validation_loss", mode="min")
    best_hypers = best_result.config  # Best hyperparameters
    with open(hp_file, "wb") as file_out:
        pickle.dump(best_hypers, file_out)
    # endregion Perform hyperparameter tuning

    restored_tuner = tune.Tuner.restore(expt_storage_path, trainable=train_model)
    result_grid = restored_tuner.get_results()

    ax = None
    for result in result_grid:
        label = f"weight decay={result.config['weight_decay']:.1e}, learning rate={result.config['lr']:.1e}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", "validation_loss", label=label, figsize=(10, 8))
        else:
            result.metrics_dataframe.plot("training_iteration", "validation_loss", ax=ax, label=label, figsize=(10, 8))
    ax.set_title(f"GNN tuning results, batch size={batch_size}")
    ax.set_ylabel("Validation Loss (Binary cross entropy)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    box_pos = ax.get_position()
    ax.set_position([box_pos.x0, box_pos.y0, box_pos.width * 0.7, box_pos.height])
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_dir, f"{expt_name}_validation-loss.png"))

    # # Plots a new graph each time a trial is interrupted instead of overwriting the previous one. Good for debugging.
    # i = 1
    # figure_saved = False
    # while not figure_saved:
    #     file_out = os.path.join(cancer_out_dir, f"{expt_name}_{i}.png")
    #     if os.path.exists(file_out):
    #         i += 1
    #         continue
    #     fig.savefig(file_out)
    #     figure_saved = True


if __name__ == "__main__":
    main()