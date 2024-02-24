from collections import OrderedDict
import torch
from torch import Tensor
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.nn.conv import SAGEConv, GATv2Conv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.pool import SAGPooling
from typing import Optional


class SparseMLP(torch.nn.Module):
    """
    An MLP with sparse connections from the input (genes) to the first hidden layer (pathways). Input genes are only
    connected to pathway units in the first hidden layer if they participate in that pathway.

    If a pathway is not connected to any inputs (i.e. none of the members of the pathway are among the inputs),
    the output of the corresponding unit in the hidden pathway layer derives entirely from the bias term.
    Conceptually, the bias can account for "unknown contributions" of the pathway despite receiving no input in the
    form of gene expression values from the previous layer.
    """
    def __init__(self, pathway_mask: Tensor) -> None:
        super().__init__()
        self.n_pathways = pathway_mask.shape[0]
        self.n_genes = pathway_mask.shape[1]
        self.pathway = torch.nn.Linear(in_features=self.n_genes, out_features=self.n_pathways, bias=True)
        self.pathway_weight_mask = pathway_mask

    def forward(self, x):
        self.pathway.weight.data = self.pathway.weight.data.mul(self.pathway_weight_mask)
        x = self.pathway(x)
        x = torch.nn.Tanh()(x)  # Liang et al. used ReLU here, but tanh is used to activate the pathway scores in GNN
        return x


class IndividualPathsMPNN(torch.nn.Module):

    def __init__(self, message_passing: str, num_nodes: int, use_sagpool: bool, ratio: Optional[float] = None) -> None:
        if message_passing.lower() not in ["gatv2", "graphsage"]:
            raise ValueError(f"Argument to \"message_passing\" should be \"gatv2\" or \"graphsage\", "
                             f"got message_passing=\"{message_passing}\"")

        super().__init__()
        self._message_passing_type = message_passing.lower()
        self._use_sagpool = use_sagpool
        self._ratio = ratio
        self._num_nodes = num_nodes

        self.input_transform = torch.nn.Linear(in_features=1, out_features=8, bias=True)

        if self._message_passing_type == "graphsage":
            self.conv1 = SAGEConv(in_channels=8, out_channels=8, project=False)
            self.conv2 = SAGEConv(in_channels=8, out_channels=8, project=False)
            self.conv3 = SAGEConv(in_channels=8, out_channels=8, project=False)
        else:  # GATv2
            self.conv1 = GATv2Conv(in_channels=8, out_channels=8, add_self_loops=False)
            self.conv2 = GATv2Conv(in_channels=8, out_channels=8, add_self_loops=False)
            self.conv3 = GATv2Conv(in_channels=8, out_channels=8, add_self_loops=False)

        self.graphnorm2 = GraphNorm(8)
        self.graphnorm3 = GraphNorm(8)

        if self._use_sagpool:
            self.sagpool1 = SAGPooling(in_channels=8, ratio=self._ratio)
            self.sagpool2 = SAGPooling(in_channels=8, ratio=self._ratio)
            self.sagpool3 = SAGPooling(in_channels=8, ratio=self._ratio)

        self.aggregate1 = Set2Set(8, 3)
        self.aggregate2 = Set2Set(8, 3)
        self.aggregate3 = Set2Set(8, 3)

        # The output transform used by Liang et al. was a small MLP; here we perform a simple linear transformation
        self.output_transform = torch.nn.Linear(in_features=48, out_features=1, bias=True)

    def forward(self, x, edge_index, batch):

        x = torch.tanh(self.input_transform(x))

        # GNN Block 1
        x = torch.tanh(self.conv1(x, edge_index))
        if self._use_sagpool:
            x, edge_index, _, batch, perm1, _ = self.sagpool1(x, edge_index, batch=batch)
        else:
            perm1 = None
        x1 = self.aggregate1(x, batch)

        # GNN Block 2
        x = self.graphnorm2(x, batch)
        x = torch.tanh(self.conv2(x, edge_index))
        if self._use_sagpool:
            x, edge_index, _, batch, perm2, _ = self.sagpool2(x, edge_index, batch=batch)
        else:
            perm2 = None
        x2 = self.aggregate2(x, batch)

        # GNN Block 3
        x = self.graphnorm3(x, batch)
        x = torch.tanh(self.conv3(x, edge_index))
        if self._use_sagpool:
            x, edge_index, _, batch, perm3, _ = self.sagpool3(x, edge_index, batch=batch)
        else:
            perm3 = None
        x3 = self.aggregate3(x, batch)

        # Concatenation
        x = torch.cat([x1, x2, x3], dim=-1)
        # Transform to scalar
        x = torch.tanh(self.output_transform(x))

        return x, batch, (perm1, perm2, perm3)


class NeuralNetworkMTLR(torch.nn.Module):

    def __init__(self, in_features: int) -> None:

        super().__init__()
        self.nn_module = torch.nn.Sequential(OrderedDict([
            ("linear1", torch.nn.Linear(in_features=in_features, out_features=256, bias=True)),
            ("dropout1", torch.nn.Dropout(0.4)),
            ("relu1", torch.nn.ReLU()),
            ("linear2", torch.nn.Linear(in_features=256, out_features=128, bias=True)),
            ("relu2", torch.nn.ReLU()),
            ("linear3", torch.nn.Linear(in_features=128, out_features=64, bias=True)),
            ("relu3", torch.nn.ReLU()),
            ("linear4", torch.nn.Linear(in_features=64, out_features=32, bias=True)),
            ("relu4", torch.nn.ReLU()),
            ("linear5", torch.nn.Linear(in_features=32, out_features=1, bias=True)),
            ("output", torch.nn.Sigmoid())
        ]))

    def forward(self, x):
        return self.nn_module(x)
