import argparse
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch

from ray.air.result import Result


def main():
    # region Parse args
    parser = argparse.ArgumentParser(
        description="Load a Result object from a training trial's latest checkpoint and extract best model"
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
    args = vars(parser.parse_args())
    # endregion Parse args

    # region Define important values
    output_dir = os.path.abspath(args["output_dir"])  # e.g. ./experiment6
    model_type = "gnn"
    use_drug_input = args["use_drug_input"]
    # endregion Define important values

    # region Directories
    ckpt_dir = os.path.join(output_dir, "checkpoints", model_type)
    model_dir = os.path.join(output_dir, "models", model_type)
    # endregion Directories

    metrics_df_list = []  # Store the results' metrics dataframes in this list

    storage_path = os.path.abspath(ckpt_dir)
    expt_name = (f"final_model-type={model_type}_drug-input={use_drug_input}")
    expt_storage_path = os.path.join(storage_path, expt_name)
    if use_drug_input:
        trial_dir = "TorchTrainer_ad753_00000_0_2024-03-07_05-08-04"
    else:
        trial_dir = "TorchTrainer_19c2e_00000_0_2024-03-07_11-44-48"
    trial_dir = os.path.join(expt_storage_path, trial_dir)
    results = Result.from_path(path=trial_dir)

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
    ax.set_title("GNN train and test set losses")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss (Binary Cross Entropy)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 1))
    box_pos = ax.get_position()
    ax.set_position([box_pos.x0, box_pos.y0, box_pos.width * 0.66, box_pos.height])
    fig.savefig(os.path.join(output_dir, f"final_model-type={model_type}_drug-input={use_drug_input}_loss.png"))


if __name__ == "__main__":
    main()