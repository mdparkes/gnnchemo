"""
Perform hyperparameter tuning using a fold of data that is exclusively reserved for this purpose.

Hyperparameter tuning is only performed with models that do not use feature selection. The tuned hyperparameters will
be shared with the models that use feature selection. For the purpose of tuning, GNN models can differ in
the type of database the graphs were derived from (KEGG/BRITE, Reactome) and whether the edges are strictly
directed or both directed and undirected (the latter only applies to reactome graphs).
"""

import argparse
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
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch_geometric.loader import DataLoader
from typing import Any, Dict, Sequence

from dataset_definitions import CancerGraphDataset
from models import MLPOutputBlock, IndividualPathsMPNN
from utilities import maybe_create_directories


def backtrace_sagpool_selections(*node_indices, n_original_nodes: int, batch_size: int, ptr: Tensor) -> Tensor:
    """
    Backtraces through a series of indices of unmasked node indices to identify the original indices of the
    nodes that survived through all the SAGPool layers

    :param node_indices: Tensors of indices of unmasked nodes returned as "perm" by each SAGPool layer. The indices
    should be supplied in order of increasing layer depth in the model. In other words, the indices returned by the
    earliest SAGPool layer should be first, and the indices returned by the final SAGPool layer should be last.
    :param n_original_nodes: The original number of nodes in the input graph. If the input to the GNN was a batch
    of graphs, this should be the total number of nodes across all graphs in the batch: batch_size * n_graph_nodes.
    :param batch_size: The number of graphs in the input batch
    :param ptr: The indices of the first nodes of each batch in the input tensor. The size should be batch_size + 1,
    and the final element of ptr should be n_original_nodes.
    :return: A Tensor of unmasked node indices with respect to the original input graph's nodes, starting from zero,
    which can be mapped to human-readable gene IDs.
    """
    original_indices = torch.arange(n_original_nodes, requires_grad=False)
    for indices in node_indices:
        original_indices = original_indices[indices]
    # The pathway graphs in each DataBatch are structurally identical, but since the node features differ
    # the SAGPool layer may return different node selections for each biopsy in the batch.
    pooled_pathway_size = original_indices.size(0) // batch_size  # Assumes all graphs have the same number of nodes
    offset = torch.repeat_interleave(ptr[:-1], pooled_pathway_size)
    # Get the indices of nodes that were retained from the original graph for each input in the batch
    original_indices = original_indices - offset  # Indices of retained nodes
    original_indices = torch.reshape(original_indices, (batch_size, -1))  # Rows are graphs in the batch
    original_indices, _ = torch.sort(original_indices, 1)  # Retained node indices sorted in ascending order
    return original_indices


def gnn_forward_pass(data_batch_list, mp_modules, model, aux_features=None):
    """
    Perform a forward pass through the GNN modules and model.

    :param data_batch_list: A list of DataBatch or HeteroDataBatch objects, one per input graph, each representing a
    batch of biopsies.
    :param mp_modules: A ModuleList with the message passing module for each input graph
    :param aux_features: A tensor of auxiliary features to use as input. If supplied, these will be concatenated with
    the results of forward passes of graphs through mp_modules. The concatenation is used as input to the final MLP
    block that outputs predictions.
    :param model: The neural network that takes a vector of graph scores as input and returns a target probability
    :return: Predictions for a batch of biopsies and a list of tuples of indices of nodes retained by each SAGPool
    layer for each input graph with a batch of biopsies.
    """
    pathway_scores = list()  # Populated with a list of [n_pathways] shape [batch_size, 1] tensors of pathway scores
    nodes_retained_list = list()
    for i, graph in enumerate(data_batch_list):
        score, _, nodes_retained = mp_modules[i](graph.x, graph.edge_index, graph.batch)
        pathway_scores.append(score)
        nodes_retained_list.append(nodes_retained)
    inputs = torch.cat(pathway_scores, dim=-1)  # shape [batch_size, n_pathways]
    if aux_features is not None:
        inputs = torch.cat([inputs, aux_features], dim=-1)  # shape [batch_size, n_pathways + n_aux_features]
    predictions = model(inputs)
    return predictions, nodes_retained_list


def train_loop(
        config: Dict[str, Any],
        *,
        dataset: CancerGraphDataset,
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        batch_size: int,
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
    n_submodules = len(dataset[0][0][0])  # Number of different pathways
    if use_aux_feats:
        n_aux_feats = len(dataset[0][0][1])  # Number of drug categories
        total_feats = n_submodules + n_aux_feats
    else:
        total_feats = n_submodules

    mp_modules = torch.nn.ModuleList()
    for i in range(n_submodules):
        num_nodes = int(dataset[0][0][0][i].x.size(0))  # Number of nodes in the pathway graph
        mp_mod = IndividualPathsMPNN(message_passing="graphsage", use_sagpool=True, ratio=0.7, num_nodes=num_nodes)
        mp_modules.append(mp_mod)

    model = MLPOutputBlock(in_features=total_feats)

    optimizer = torch.optim.Adam([
        {"params": mp_modules.parameters()},
        {"params": model.parameters()}
    ], lr=config["lr"], weight_decay=config["weight_decay"])
    # endregion Initialize models and optimizer

    # Check for and load checkpoint
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            ckpt = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch, mp_modules_state, model_state, optimizer_state = ckpt
            mp_modules.load_state_dict(mp_modules_state)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    else:
        start_epoch = 0

    train_losses = []  # Append epoch losses
    val_losses = []  # Append epoch losses

    for t in range(start_epoch, epochs):

        # region Train
        mp_modules.train()
        model.train()
        epoch_train_loss = 0.
        samples_processed = 0
        for loaded_data in train_dataloader:
            (data_batch_list, aux_feat_tensor), targets = loaded_data
            current_batch_size = len(data_batch_list[0])
            samples_processed += current_batch_size
            # data_batch_list is a list of m DataBatch objects, where m is the number of graphs fed through the GNN
            # for a single patient. Each DataBatch object represents a batch of a particular graph.
            # aux_feat_tensor is a [current_batch_size, n_aux_feat] tensor of auxiliary features (drugs administered).
            # targets is a [current_batch_size, ] tensor of prediction target labels
            targets = torch.reshape(targets, (current_batch_size, -1))
            if use_aux_feats:
                predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model, aux_feat_tensor)
            else:
                predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model)
            predictions = torch.reshape(predictions, (current_batch_size, -1))
            # Calculate training loss for the batch
            # Positive responders represent ~ 76% of the dataset. Rescale the losses accordingly.
            pos_wt = torch.ones(size=[current_batch_size], dtype=torch.float32)
            neg_wt = torch.full(size=[current_batch_size], fill_value=759 / 242, dtype=torch.float32)
            rescaling_weights = torch.where(targets.bool(), pos_wt, neg_wt)
            loss = binary_cross_entropy(predictions, targets, weight=rescaling_weights, reduction="sum")
            epoch_train_loss += loss  # Running total of this epoch's training loss
            loss /= current_batch_size  # Per-patient loss for the current batch
            # Optimize model weights w.r.t. per-patient loss for the current batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train_loss /= samples_processed  # Per-patient average training loss for this epoch
        train_losses.append(float(epoch_train_loss))
        # endregion Train

        # region Evaluate
        mp_modules.eval()
        model.eval()
        epoch_val_loss = 0.
        samples_processed = 0
        with torch.no_grad():
            for loaded_data in val_dataloader:
                (data_batch_list, aux_feat_tensor), targets = loaded_data
                current_batch_size = len(data_batch_list[0])
                samples_processed += current_batch_size
                targets = torch.reshape(targets, (current_batch_size, -1))
                if use_aux_feats:
                    predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model, aux_feat_tensor)
                else:
                    predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model)
                predictions = torch.reshape(predictions, (current_batch_size, -1))
                pos_wt = torch.ones(size=[current_batch_size], dtype=torch.float32)
                neg_wt = torch.full(size=[current_batch_size], fill_value=759 / 242, dtype=torch.float32)
                rescaling_weights = torch.where(targets.bool(), pos_wt, neg_wt)
                loss = binary_cross_entropy(predictions, targets, weight=rescaling_weights, reduction="sum")
                epoch_val_loss += loss
        epoch_val_loss /= samples_processed
        val_losses.append(float(epoch_val_loss))
        # endregion Evaluate

        # Checkpointing
        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            torch.save(
                (t, mp_modules.state_dict(), model.state_dict(), optimizer.state_dict()),
                os.path.join(tmp_ckpt_dir, "checkpoint.pt")
            )
            train.report(
                metrics={
                    "training_loss": float(epoch_train_loss),
                    "validation_loss": float(epoch_val_loss)
                },
                checkpoint=Checkpoint.from_directory(tmp_ckpt_dir)
            )


def load_dataset(graph_dir: str) -> CancerGraphDataset:
    # If merging all pathways into a single large graph, the standardization occurs over all genes in all pathways.
    # If feeding one pathway graph through the NN at a time, standardization is isolated to the pathway's genes.
    graph_files = sorted(os.listdir(os.path.join(graph_dir, "raw")))  # Raw graph Data object files
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerGraphDataset(root=graph_dir, data_files=graph_files)
    return dataset


def main():
    # region Parse args
    parser = argparse.ArgumentParser(
        description="GNN hyperparameter tuning"
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

    # # For interactive debugging
    # args = {
    #     "data_dir": "data",
    #     "output_dir": "test_expt",
    #     "use_drug_input": True,
    #     "batch_size": 48,
    # }

    # region Define important values
    data_dir = args["data_dir"]  # e.g. ./data
    output_dir = args["output_dir"]  # e.g. ./experiment6
    model_type = "gnn"
    use_drug_input = True if args["use_drug_input"] else False  # If False, only use pathway scores from GNNs as input
    batch_size = args["batch_size"]
    # endregion Define important values

    # region Directories
    graph_data_dir = os.path.join(data_dir, "graphs")
    model_dir = os.path.join(output_dir, "models", model_type)
    ckpt_dir = os.path.join(output_dir, "checkpoints", model_type)
    export_dir = os.path.join(output_dir, "exports", model_type)
    hp_dir = os.path.join(output_dir, "hyperparameters", model_type)
    # Check for the existence of directories and create them if necessary
    maybe_create_directories(model_dir, ckpt_dir, export_dir, hp_dir)
    # endregion Directories

    # region Files to read/write
    graph_data_files = sorted(os.listdir(os.path.join(data_dir, "graphs", "raw")))  # Raw graph Data object files
    feature_names_file = os.path.join(data_dir, "gnn_feature_names.pkl")  # HSA/ENTREZ IDs of input genes
    hp_file = os.path.join(hp_dir, f"{model_type}_hyperparameters.pkl")
    # endregion Files to read/write

    # Load lists that name the biopsies in each cross validation partition
    with open(os.path.join(data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        # A list of one tuple per CV fold: (train_names, test_names)
        train_val_test_names = pickle.load(file_in)
    # Get the names of the raw Data files for biopsies the current CV fold
    train_names, val_names, _ = train_val_test_names
    # Get the indices of the current CV fold's raw Data files in the file list
    train_idx = [graph_data_files.index(name) for name in train_names]
    val_idx = [graph_data_files.index(name) for name in val_names]
    # Load dataset of n_patients items. Each item is Tuple[Tuple[List[graph], drugs], response].
    ds = load_dataset(graph_data_dir)

    # region Perform hyperparameter tuning
    hp_dict = {
        "weight_decay": tune.grid_search([5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3]),
        "lr": tune.grid_search([1e-4, 1e-3])
    }

    train_model = tune.with_parameters(
        train_loop,
        dataset=ds,
        train_indices=train_idx,
        val_indices=val_idx,
        batch_size=batch_size,
        use_drug_input=use_drug_input
    )
    #  Adjust `resources={"cpu": x}` based on available cores to optimize performance. This is the number of
    #  cores that each trial will use. For example, if 12 cores are available and `resources={"cpu": 2}`, six trials
    #  can be run concurrently using two cores each.
    n_cpu = 4  # Allocate at least 64GB to /dev/shm and use default workers for dataloader
    train_model = tune.with_resources(train_model, resources={"cpu": n_cpu})

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

    restored_tuner = tune.Tuner.restore(expt_storage_path, trainable=train_model)  # Restores most recent
    result_grid = restored_tuner.get_results()  # List of tuning trials
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
    #     file_out = os.path.join(output_dir, f"{expt_name}_{i}.png")
    #     if os.path.exists(file_out):
    #         i += 1
    #         continue
    #     fig.savefig(file_out)
    #     figure_saved = True


if __name__ == "__main__":
    main()
