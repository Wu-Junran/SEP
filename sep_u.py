#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.pool import ASAPooling
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn.conv import MessagePassing


class SEPooling(MessagePassing):
    def __init__(self, nn: Callable, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, size=size)
        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class SEP_U_synthetic(torch.nn.Module):
    def __init__(self, args):
        super(SEP_U_synthetic, self).__init__()
        self.args = args
        self.num_features = args.num_features  # input_dim
        self.nhid = args.hidden_dim  # hidden dim
        self.convs = self.get_convs()
        self.sepools = self.get_sepool()

    def get_convs(self):
        convs = nn.ModuleList()
        _input_dim = self.num_features
        _output_dim = self.nhid
        for _ in range(self.args.num_blocks*2 + 1):
            if _ == self.args.num_blocks*2:
                _output_dim = self.num_features
            if self.args.conv == 'GCN':
                conv = GCNConv(_input_dim, _output_dim)
            elif self.args.conv == 'GIN':
                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(_input_dim, _output_dim),
                        nn.ReLU(),
                        nn.Linear(_output_dim, _output_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(_output_dim),
                ), train_eps=False)
            convs.append(conv)
            _input_dim = self.nhid
        return convs

    def get_sepool(self):
        pools = nn.ModuleList()
        for _ in range(2):
            pool = SEPooling(
                nn.Sequential(
                    nn.Linear(self.nhid, self.nhid),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.nhid),
                ))
            pools.append(pool)
        return pools

    def forward(self, data, S_edge_index):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        s_edge_index = torch.LongTensor(S_edge_index).to(self.args.device)
        size = [len(set(S_edge_index.numpy()[1])), len(set(S_edge_index.numpy()[0]))]
        # encoder
        for _ in range(self.args.num_blocks):
            x = F.relu(self.convs[_](x, edge_index))
        # sep
        x = F.relu(self.sepools[0](x, s_edge_index, size=size))
        # sep-u
        x = F.relu(self.sepools[1](x, s_edge_index[[1, 0]], size=(size[1], size[0])))
        # decoder
        for _ in range(self.args.num_blocks, self.args.num_blocks*2):
            x = F.relu(self.convs[_](x, edge_index))
        x = self.convs[-1](x, edge_index)
        return x
