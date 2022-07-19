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
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.conv import MessagePassing


class SEPooling(MessagePassing):
    def __init__(self, nn: Callable=None, **kwargs):
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


class SEP_U(torch.nn.Module):
    def __init__(self, args):
        super(SEP_U, self).__init__()
        self.args = args
        self.num_features = args.num_features  # input_dim
        self.nhid = args.hidden_dim  # hidden dim
        self.num_classes = args.num_classes  # output dim
        self.d1 = args.conv_dropout
        self.d2 = args.pooling_dropout
        self.convs = self.get_convs()
        self.sepools = self.get_sepool()
        self.classifier = self.get_classifier()

    def __process_layer_edgeIndex(self, batch, layer=0):
        if layer == 0:
            return batch['data'].edge_index
        return batch['layer_data']['layer_edgeMat'][layer].to(self.args.device)

    def __process_sep_edgeIndex(self, batch, layer=1):
        return batch['layer_data']['interLayer_edgeMat'][layer].to(self.args.device)

    def __process_sep_size(self, batch, layer=1):
        return [batch['layer_data']['node_size'][layer-1], batch['layer_data']['node_size'][layer]]

    def get_convs(self):
        convs = nn.ModuleList()
        _input_dim = self.num_features
        _output_dim = self.nhid
        for _ in range(self.args.num_blocks*2 + 1):
            if _ > self.args.num_blocks:
                _input_dim = 2 * self.nhid
            if _ == self.args.num_blocks*2:
                if self.args.link_input:
                    _input_dim = 2 * self.nhid + self.num_features
                _output_dim = self.num_classes
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
        for _ in range(self.args.num_blocks*2):
            pool = SEPooling(
                nn.Sequential(
                    nn.Linear(self.nhid, self.nhid),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.nhid),
                ))
            pools.append(pool)
        return pools

    def get_classifier(self):
        return nn.Linear(self.nhid, self.num_classes)

    def forward(self, batch):
        x = batch['data'].x
        xs = []
        # down sampling
        for _ in range(self.args.num_blocks):
            # mp
            edge_index = self.__process_layer_edgeIndex(batch, _)
            x = F.dropout(F.relu(self.convs[_](x, edge_index)), self.d1, training=self.training)
            xs.append(x)
            # sep
            edge_index = self.__process_sep_edgeIndex(batch, _+1)
            size = self.__process_sep_size(batch, _+1)
            x = F.dropout(F.relu(self.sepools[_](x, edge_index, size=size)), self.d2, training=self.training)

        # up sampling
        for _ in range(self.args.num_blocks, 0, -1):
            # mp
            edge_index = self.__process_layer_edgeIndex(batch, _)
            x = F.dropout(F.relu(self.convs[self.args.num_blocks*2-_](x, edge_index)), self.d1, training=self.training)
            # sep_u
            edge_index = self.__process_sep_edgeIndex(batch, _)
            size = self.__process_sep_size(batch, _)
            size.reverse()
            x = F.dropout(F.relu(self.sepools[self.args.num_blocks*2-_](x, edge_index[[1, 0]], size=size)), self.d2, training=self.training)
            x = torch.cat([x, xs[_-1]], dim=1)

        # last conv
        edge_index = self.__process_layer_edgeIndex(batch, 0)
        # link input
        if self.args.link_input:
            x = torch.cat([x, batch['data'].x], dim=1)
        x = self.convs[-1](x, edge_index)

        # For Classification
        # x = self.classifier(x)
        return F.log_softmax(x, dim=-1)
