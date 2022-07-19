#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import time
import torch
import pickle
import random
import argparse
import numpy as np
from sep_u import SEP_U
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.datasets import Planetoid


PWD = os.path.dirname(os.path.realpath(__file__))


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

        if torch.cuda.is_available():
            self.args.device = 'cuda:{}'.format(args.gpu)
        else:
            self.args.device = 'cpu'
        self.load_data()
        self.patience = 0
        self.best_loss_epoch = 0
        self.best_acc_epoch = 0
        self.best_loss = 1e9
        self.best_loss_acc = -1e9
        self.best_acc = -1e9
        self.best_acc_loss = 1e9

    def load_data(self):
        t_path = os.path.join(PWD, 'trees', '%s_%s.pickle' % (self.args.dataset, self.args.tree_depth))
        with open(t_path, 'rb') as fp:
            self.layer_data = pickle.load(fp)
        dataset = Planetoid(root=os.path.join(PWD, 'data'), name=self.args.dataset)
        data = dataset[0]
        if self.args.global_degree:
            max_degree = degree(data.edge_index[0], dtype=torch.long).max()
            T.OneHotDegree(max_degree).__call__(data)
        self.args.num_features = data.num_features
        self.args.num_classes = dataset.num_classes
        self.data = data

    def load_model(self):
        model = SEP_U(self.args).to(self.args.device)
        return model

    def organize_val_log(self, val_loss, val_acc, epoch):
        if val_loss < self.best_loss:
            self.best_loss_acc = val_acc
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            self.patience = 0
        else:
            self.patience += 1

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_acc_loss = val_loss
            self.best_acc_epoch = epoch

    def eval(self, batch):
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch)
        pred = out.argmax(dim=1)
        # val
        mask = batch['data'].val_mask
        correct = (pred[mask] == batch['data'].y[mask]).sum()
        val_acc = int(correct) / int(mask.sum())
        val_loss = F.nll_loss(out[mask], batch['data'].y[mask], reduction='sum').item()
        # test
        mask = batch['data'].test_mask
        correct = (pred[mask] == batch['data'].y[mask]).sum()
        test_acc = int(correct) / int(mask.sum())
        test_loss = F.nll_loss(out[mask], batch['data'].y[mask], reduction='sum').item()
        return val_acc, val_loss, test_acc, test_loss

    def train(self):
        # Load Model & Optimizer
        self.model = self.load_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2rate)

        val_accs = []
        val_losses = []
        test_accs = []
        # training
        for epoch in range(self.args.num_epochs):
            self.model.train()

            batch = {'data': self.data.to(self.args.device), 'layer_data': self.layer_data}
            self.optimizer.zero_grad()
            out = self.model(batch)
            train_mask = batch['data'].train_mask
            loss = F.nll_loss(out[train_mask], batch['data'].y[train_mask])
            loss.backward()
            self.optimizer.step()

            # Validation
            val_acc, val_loss, test_acc, test_loss = self.eval(batch)
            self.organize_val_log(val_loss, val_acc, epoch)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            test_accs.append(test_acc)
            if self.patience > self.args.patience:
                break

        val_accs = np.array(val_accs)
        val_losses = np.array(val_losses)
        test_accs = np.array(test_accs)
        print("%.4f\t%.4f\t%s" %
              (test_accs[val_losses.argmin()],
               test_accs.max(),
               epoch), flush=True)
        return test_accs[val_losses.argmin()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEP arguments')
    parser.add_argument('-d', '--dataset', type=str, default="Cora", choices=['Cora', 'Pubmed', 'Citeseer'],
                        help='name of dataset (default: Cora)')
    parser.add_argument('-c', '--conv', default='GCN', type=str, choices=['GCN', 'GIN'],
                        help='message-passing function type')
    parser.add_argument('-nb', '--num_blocks', default=2, type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument('-k', '--tree_depth', type=int, default=3,
                        help='the depth of coding tree (=num_blocks+1)')
    parser.add_argument('-lr', '--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=128,
                        help='number of hidden units (default: 128)', choices=[16, 32, 128, 256])
    parser.add_argument('-cd', '--conv_dropout', type=float, default=0.5,
                        help='layer dropout (default: 0.5)', choices=[i/10 for i in range(10)])
    parser.add_argument('-pd', '--pooling_dropout', type=float, default=0.5,
                        help='layer dropout (default: 0.5)', choices=[i/10 for i in range(10)])
    parser.add_argument('-l2', '--l2rate', type=float, default=5e-4,
                        help='L2 penalty lambda, 0.02, 5e-4', choices=[5e-4, 0.02])
    parser.add_argument('-e', '--num_epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('-gd', "--global_degree", action='store_true')
    parser.add_argument("--link-input", action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for earlystopping')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    args = parser.parse_args()

    if args.dataset in ['Citeseer', 'Pubmed']:
        args.link_input = True
    print(args, flush=True)
    trainer = Trainer(args)
    test_acc = trainer.train()
