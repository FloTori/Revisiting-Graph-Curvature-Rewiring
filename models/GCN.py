from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F

class GCN_Toppingetal(torch.nn.Module):
    def __init__(
        self, dataset, hidden: List[int] = [64], dropout: float = 0.5
    ):
        super().__init__() 

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data,device):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #x = x.to(device = device)
        #edge_index = edge_index.to(device = device)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, num_features,num_classes,hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features,hidden_channels)
        self.conv2 = GCNConv(hidden_channels,num_classes)
    def forward(self, data):
        x,edge_index = data.x,data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.4144)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GCN_Graph(torch.nn.Module):
    def __init__(
        self, dataset, hidden: List[int] = [64], dropout: float = 0.5):
        super().__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)
        self.num_layers = len(layers)
        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = x.float()
        for i, layer in enumerate(self.layers):

            x_new = layer(x, edge_index)
            if i != self.num_layers - 1:
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)

            x = x_new 

        x = global_mean_pool(x, batch)

        return x
