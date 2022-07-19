#!/usr/bin/env python
# encoding: utf-8
import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import trange
from pygsp import graphs
import matplotlib.pyplot as plt
from sep_u import SEP_U_synthetic
from SEPG.utils import PartitionTree
from torch_geometric.data import Data
from SEPN.prepare_nodeData import update_node


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        # Random Seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.args = args
        ckpt_dir = os.path.join('./checkpoints/synthetic-%s' % self.args.data)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.ckpt = os.path.join(ckpt_dir, "best_model.pth")

        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)
            self.args.device = 'cuda:{}'.format(args.gpu)
        else:
            self.args.device = 'cpu'
        self.data = self.load_data()

    def extract_cluster_assignment(self, T, tree_depth=2):
        layer_idx = [0]
        for layer in range(tree_depth+1):
            layer_nodes = [i for i, n in T.items() if n['depth']==layer]
            layer_idx.append(layer_idx[-1] + len(layer_nodes))
        interLayerEdges = [[] for i in range(tree_depth+1)]
        # edges among layers
        for i, n in T.items():
            if n['depth'] == 0:
                continue
            n_idx = n['ID'] - layer_idx[n['depth']]
            c_base = layer_idx[n['depth']-1]
            interLayerEdges[n['depth']].extend([(n_idx, c-c_base) for c in n['children']])
        return torch.LongTensor(interLayerEdges[1]).T

    def load_data(self):
        if self.args.data == 'ring':
            G = graphs.Ring(N=200)
        elif self.args.data == 'grid':
            G = graphs.Grid2d(N1=30, N2=30)
        X = G.coords.astype(np.float32)
        A = G.W

        # cluster assignment
        tree_depth = 5 if self.args.data == 'ring' else 6
        if not os.path.exists('datasets'):
            os.makedirs('datasets')
        S_path = './datasets/%s-%s.pt' % (self.args.data, tree_depth)
        if os.path.exists(S_path):
            self.S_edge_index = torch.load(S_path)
        else:
            undirected_adj = np.array(A.todense())
            y = PartitionTree(adj_matrix=undirected_adj)
            y.build_coding_tree(tree_depth)
            T = update_node(y.tree_node)
            self.S_edge_index = self.extract_cluster_assignment(T, tree_depth)
            torch.save(self.S_edge_index, S_path)

        coo = A.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        X = torch.FloatTensor(X)
        edge_index = torch.LongTensor(indices)
        edge_weight = torch.FloatTensor(values)
        batch = torch.LongTensor([0 for _ in range(X.shape[0])])

        data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, batch=batch)
        self.args.num_features = 2
        self.args.num_classes = 2
        self.args.avg_num_nodes = A.shape[0]
        if self.use_cuda:
            data.to(self.args.device)
        return data

    def load_model(self):
        model = SEP_U_synthetic(self.args)
        if self.use_cuda:
            model.to(self.args.device)
        return model

    def train(self):
        self.model = self.load_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.MSELoss()
        self.best_loss = 1e6
        self.patience = self.args.patience

        epoch_iter = trange(0, self.args.num_epochs, desc='[EPOCH]', position=0)
        for epoch in epoch_iter:
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.data, self.S_edge_index)
            target = self.data.x

            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()

            desc = f"[Train] Train Loss {loss.item()}"
            epoch_iter.set_description(desc)
            epoch_iter.refresh()

            if loss.item() < self.best_loss:
                torch.save(self.model.state_dict(), self.ckpt)
                self.patience = self.args.patience
                self.best_loss = loss.item()
            else:
                self.patience -= 1
                if self.patience == 0:
                    break

        # Load Best Model
        self.model = self.load_model()
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model = self.model.to(self.args.device)
        loss, target, out = self.eval()
        print("Loss: %s" % loss)
        self.draw(target, out)

    def eval(self):
        self.model.eval()
        out = self.model(self.data, self.S_edge_index)
        target = self.data.x
        loss = self.criterion(out, target)
        return loss.item(), target.detach().cpu(), out.detach().cpu()

    def draw(self, target, out):
        plt.figure(figsize=(4, 4))
        pad = 0.1
        x_min, x_max = target[:, 0].min() - pad, target[:, 0].max() + pad
        y_min, y_max = target[:, 1].min() - pad, target[:, 1].max() + pad
        colors = target[:, 0] + target[:, 1]
        plt.scatter(*out[:, :2].T, c=colors, s=8, zorder=2)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if self.args.data == 'ring':
            plt.axvline(0, c='k', alpha=0.2)
            plt.axhline(0, c='k', alpha=0.2)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig("./figs/%s-%s.pdf" % (self.args.model, self.args.data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEP')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('-d', '--data', type=str, default="ring",
                        choices=["ring", "grid"])
    parser.add_argument('--model', type=str, default="SEP-U",
                        choices=["SEP-U"])
    parser.add_argument('--conv', default='GCN', type=str,
                        choices=['GCN', 'GIN'],
                        help='message-passing function type')
    parser.add_argument('--num-blocks', default=2, type=int)
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden size')
    parser.add_argument('--batch_size', default=1, type=int, help='train batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument('--num-epochs', default=10000, type=int, help='train epochs number')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--patience', type=int, default=1000, help='patience for earlystopping')
    parser.add_argument("--ln", action='store_true')
    parser.add_argument("--cluster", action='store_true')

    args = parser.parse_args()
    args.cluster = True
    args.ln = True
    trainer = Trainer(args)
    trainer.train()

