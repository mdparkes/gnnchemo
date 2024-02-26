"""
Note:
    Uses sagpool selections from GNNs with non-relational unmerged graphs -- hardcoded behaviour.

"""

import argparse
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import tempfile
import torch
import torchmetrics

from filelock import FileLock
from matplotlib.ticker import MaxNLocator
from ray import tune, train, init
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
from torch import Tensor
from torch.nn.functional import binary_cross_entropy
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader
from typing import Any, Dict, Optional, Sequence, Union

from dataset_definitions import CancerDataset
from models import MLPOutputBlock, SparseMLP
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


def train_loop(
        config: Dict[str, Any],
        *,
        data_dir: str,
        train_names: Sequence[str],
        val_names: Sequence[str],
        worker_batch_size: int,
        mask: Tensor,
        feature_indices: Optional[Union[Sequence[int], None]],
        use_aux_feats: bool
) -> None:
    # worker_batch_size is the size of the batch subset handled by the worker. The global batch size is calculated as
    # the number of workers times the worker batch size. For example, if the global batch size is 50 and there are 5
    # workers, worker_batch_size should be 10.

    epochs = 100

    # region Dataset and DataLoaders
    # Create training and validation fold Dataset objects
    train_dataset = load_dataset(data_dir, train_names)
    val_dataset = load_dataset(data_dir, val_names)
    # Create samplers that partition stratified CV folds into disjoint random batches
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=423, drop_last=False)
    val_sampler = DistributedSampler(val_dataset,  shuffle=True, seed=423, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=worker_batch_size, sampler=train_sampler)
    train_dataloader = prepare_data_loader(train_dataloader)
    val_dataloader = DataLoader(val_dataset, batch_size=worker_batch_size, sampler=val_sampler)
    val_dataloader = prepare_data_loader(val_dataloader)
    # endregion Dataset and DataLoaders

    # Restrict the mask to input genes (columns) that will be used by the sparse MLP
    if feature_indices is not None:
        mask = mask[:, feature_indices]

    # region Initialize models and optimizer
    if use_aux_feats:
        n_aux_feats = len(train_dataset[0][1])  # Number of drug categories
        total_feats = mask.shape[0] + n_aux_feats
    else:
        total_feats = mask.shape[0]

    sparse_layer = SparseMLP(mask)
    sparse_layer = DistributedDataParallel(sparse_layer)
    sparse_layer = prepare_model(sparse_layer)

    model = MLPOutputBlock(in_features=total_feats)
    model = DistributedDataParallel(model)
    model = prepare_model(model)

    optimizer = torch.optim.Adam([
        {"params": sparse_layer.parameters()},
        {"params": model.parameters()}
    ], lr=config["lr"], weight_decay=config["weight_decay"])
    # endregion Initialize models and optimizer

    # Check for and load checkpoint
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            ckpt = torch.load(os.path.join(checkpoint_dir, f"checkpoint_fold.pt"))
            start_epoch, sparse_layer_state, model_state, optimizer_state = ckpt
            sparse_layer.load_state_dict(sparse_layer_state)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    else:
        start_epoch = 0

    # MeanMetric objects will aggregate the losses across all workers in the DDP process
    mean_train_loss = torchmetrics.MeanMetric()
    mean_valid_loss = torchmetrics.MeanMetric()

    for t in range(start_epoch, epochs):
        # In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating the
        # DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same
        # ordering will always be used.
        train_sampler.set_epoch(t)
        val_sampler.set_epoch(t)

        # region Train
        sparse_layer.train()
        model.train()
        epoch_train_loss = 0.
        n_batches = len(train_dataloader)
        for loaded_data in train_dataloader:
            feature_tensor, aux_feat_tensor, targets = loaded_data
            if feature_indices is not None:
                # Only use the selected genes at the specified indices
                feature_tensor = feature_tensor[:, feature_indices]
            current_batch_size = feature_tensor.shape[0]
            targets = torch.reshape(targets, (current_batch_size, -1))
            inputs = sparse_layer(feature_tensor)
            if use_aux_feats:
                inputs = torch.cat([inputs, aux_feat_tensor], dim=-1)
            predictions = model(inputs)
            predictions = torch.reshape(predictions, (current_batch_size, -1))
            pos_wt = torch.ones(size=[current_batch_size, 1], dtype=torch.float32)
            neg_wt = torch.full(size=[current_batch_size, 1], fill_value=759 / 242, dtype=torch.float32)
            rescaling_weights = torch.where(targets.bool(), pos_wt, neg_wt)
            rescaling_weights = rescaling_weights.reshape((current_batch_size, 1))
            wt = current_batch_size / worker_batch_size  # Down-weights the per-patient losses from undersized batches
            loss = wt * binary_cross_entropy(predictions, targets, weight=rescaling_weights, reduction="mean")
            epoch_train_loss += loss  # Running total of this epoch's mean per-patient training minibatch losses

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= n_batches  # Per-patient average training loss for this epoch
        mean_train_loss(epoch_train_loss)  # Save the worker's epoch training loss in the aggregator
        aggregated_train_loss = mean_train_loss.compute().item()  # Aggregate mean loss across workers
        mean_train_loss.reset()  # Reset for next epoch
        # endregion Train

        # region Evaluate
        sparse_layer.eval()
        model.eval()
        epoch_val_loss = 0.
        n_batches = len(val_dataloader)
        with torch.no_grad():
            for loaded_data in val_dataloader:
                feature_tensor, aux_feat_tensor, targets = loaded_data
                if feature_indices is not None:
                    # Only use the selected genes at the specified indices
                    feature_tensor = feature_tensor[:, feature_indices]
                current_batch_size = feature_tensor.shape[0]
                targets = torch.reshape(targets, (current_batch_size, -1))
                inputs = sparse_layer(feature_tensor)
                if use_aux_feats:
                    inputs = torch.cat([inputs, aux_feat_tensor], dim=-1)
                predictions = model(inputs)
                predictions = torch.reshape(predictions, (current_batch_size, -1))
                pos_wt = torch.ones(size=[current_batch_size, 1], dtype=torch.float32)
                neg_wt = torch.full(size=[current_batch_size, 1], fill_value=759 / 242, dtype=torch.float32)
                rescaling_weights = torch.where(targets.bool(), pos_wt, neg_wt)
                rescaling_weights = rescaling_weights.reshape((current_batch_size, 1))
                wt = current_batch_size / worker_batch_size  # Down-weights the per-patient losses from undersized batches
                loss = wt * binary_cross_entropy(predictions, targets, weight=rescaling_weights, reduction="mean")
                epoch_val_loss += loss  # Running total of this epoch's mean per-patient validation minibatch losses

        epoch_val_loss /= n_batches  # Per-patient average validation loss for this epoch
        mean_valid_loss(epoch_val_loss)  # Save the worker's epoch validation loss in the aggregator
        aggregated_val_loss = mean_valid_loss.compute().item()  # Aggregate mean loss across workers
        mean_valid_loss.reset()  # Reset for next epoch
        # endregion Evaluate

        # Checkpointing
        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            torch.save(
                (t, sparse_layer.state_dict(), model.state_dict(), optimizer.state_dict()),
                os.path.join(tmp_ckpt_dir, "checkpoint.pt")
            )
            metrics = {
                "training_loss": float(aggregated_train_loss),
                "validation_loss": float(aggregated_val_loss)
            }
            train.report(
                metrics=metrics,
                checkpoint=Checkpoint.from_directory(tmp_ckpt_dir)
            )
        # Report metrics from worker 0
        if train.get_context().get_world_rank() == 0:
            print(f"Epoch {t:>3d} -- Training loss: {metrics['training_loss']:>4f}, Validation loss: "
                  f"{metrics['validation_loss']:>4f}")


def main():
    # region Parse args
    parser = argparse.ArgumentParser(
        description="Train an MLP model of survival with cross-validation"
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
    parser.add_argument(
        "-nw", "--num_workers",
        help="The number of workers for distributed training. Should evenly divide batch_size.",
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
    #     "num_workers": 8,
    # }

    # region Define important values
    data_dir = args["data_dir"]  # e.g. ./data
    output_dir = args["output_dir"]  # e.g. ./experiment6
    model_type = "mlp"
    use_drug_input = True if args["use_drug_input"] else False
    batch_size = args["batch_size"]
    num_workers = args["num_workers"]
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
    pathway_names_file = os.path.join(data_dir, f"{model_type}_pathway_names.npy")  # HSA IDs of pathways
    hp_file = os.path.join(hp_dir, f"{model_type}_hyperparameters.pkl")
    # endregion Files to read/write

    # Load feature names in the order they appear in the unfiltered input to the SparseMLP
    with open(feature_names_file, "rb") as file_in:
        all_feature_names = pickle.load(file_in)

    # Load weight mask for SparseMLP Module
    with open(mask_matrix_file, "rb") as file_in:
        buffer = io.BytesIO(file_in.read())
    mlp_mask_matrix = torch.load(buffer)

    # Load pathway names
    pathway_names = np.load(pathway_names_file, allow_pickle=True)
    assert (len(pathway_names) == mlp_mask_matrix.shape[0])

    # Load tuned MLP hyperparameters
    with open(hp_file, "rb") as file_in:
        hp_dict = pickle.load(file_in)

    # Get indices of train/validation/test biopsies in the dataset and load dataset
    with open(os.path.join(data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        train_val_test_names = pickle.load(file_in)
    train_names, _, test_names = train_val_test_names
    train_idx = [input_data_files.index(name) for name in train_names]
    test_idx = [input_data_files.index(name) for name in test_names]
    ds = load_dataset(input_data_dir)

    metrics_df_list = []

    # Features to use
    features_used_idx = None  # When this is passed to train_loop `features` param, all features will be used

    storage_path = os.path.abspath(ckpt_dir)
    expt_name = f"final_model-type={model_type}_drug-input={use_drug_input}"
    expt_storage_path = os.path.join(storage_path, expt_name)

    init(log_to_driver=False, ignore_reinit_error=True)  # should suppress info messages to stdout but allow logging

    worker_bsize = int(batch_size / num_workers)
    train_model = tune.with_parameters(
        train_loop, data_dir=input_data_dir, train_names=train_names, val_names=test_names,
        worker_batch_size=worker_bsize, mask=mlp_mask_matrix, feature_indices=features_used_idx,
        use_aux_feats=use_drug_input
    )

    if TorchTrainer.can_restore(expt_storage_path):
        # Auto-resume training if fault occurred
        trainer = TorchTrainer.restore(
            expt_storage_path, train_loop_per_worker=train_model, train_loop_config=hp_dict
        )
    else:
        # Begin training
        trainer = TorchTrainer(
            train_model,
            train_loop_config=hp_dict,
            scaling_config=train.ScalingConfig(num_workers=num_workers-1, use_gpu=False),
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
    results = trainer.fit()
    # Save best model
    best_ckpt = results.get_best_checkpoint(metric="validation_loss", mode="min")
    model_file = os.path.join(model_dir, f"final_model-type={model_type}_drug-input={use_drug_input}.pt")
    with best_ckpt.as_directory() as checkpoint_dir:
        ckpt = torch.load(os.path.join(checkpoint_dir, f"checkpoint.pt"))
        _, mp_modules_state, model_state, optimizer_state = ckpt
    torch.save((mp_modules_state, model_state, optimizer_state), model_file)
    metrics_df_list.append(results.metrics_dataframe)

    # Plot results
    fig, ax = plt.subplots(figsize=(4, 4))
    plt_df = metrics_df_list[0]
    sns.lineplot(x="training_iteration", y="validation_loss", label="Test Dataset", data=plt_df,
                 linestyle="solid", ax=ax)
    sns.lineplot(x="training_iteration", y="training_loss", label="Training Dataset", data=plt_df,
                 linestyle="dashed", ax=ax)
    ax.set_title(f"MLP train and test set losses")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss (Binary Cross Entropy)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 1))
    box_pos = ax.get_position()
    ax.set_position([box_pos.x0, box_pos.y0, box_pos.width * 0.66, box_pos.height])
    fig.savefig(os.path.join(output_dir, f"final_model-type={model_type}_drug-input={use_drug_input}_loss.png"))


if __name__ == "__main__":
    main()
