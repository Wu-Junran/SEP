#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import math
import time
import torch
import random
import argparse
import numpy as np
from sep import SEP
from data import load_tree
from functools import reduce
import torch.nn.functional as F
from transformers.optimization import get_cosine_schedule_with_warmup


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
        self.exp_name = self.set_experiment_name()

        if torch.cuda.is_available():
            self.args.device = 'cuda:{}'.format(args.gpu)
        else:
            self.args.device = 'cpu'
        self.load_data()

    def init_fold(self):
        self.patience = 0
        self.best_loss_epoch = 0
        self.best_acc_epoch = 0
        self.best_loss = 1e9
        self.best_loss_acc = -1e9
        self.best_acc = -1e9
        self.best_acc_loss = 1e9
        self.args.num_features = self.dataset[0]['node_features'].size(1)
        self.args.num_classes = len(set([t['label'] for t in self.dataset]))

    def load_data(self):
        self.dataset = load_tree(self.args.dataset, self.args.tree_depth)

    def load_tvt(self, fold_number, val_fold_number):
        data = self.args.dataset
        train_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/train_idx-%d.txt' % (data, fold_number), dtype=np.int32), dtype=torch.long)
        val_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/test_idx-%d.txt' % (data, val_fold_number), dtype=np.int32), dtype=torch.long)
        test_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/test_idx-%d.txt' % (data, fold_number), dtype=np.int32), dtype=torch.long)
        all_idxes = reduce(np.union1d, (train_idxes, val_idxes, test_idxes))
        assert len(all_idxes) == len(self.dataset)

        # bug: random.shuffle(tensor), this will cause replicated data
        train_idxes = np.setdiff1d(train_idxes, val_idxes)
        random.shuffle(train_idxes)
        train_set = [self.dataset[i] for i in train_idxes]
        val_set = [self.dataset[i] for i in val_idxes]
        test_set = [self.dataset[i] for i in test_idxes]
        return train_set, val_set, test_set

    def load_model(self):
        model = SEP(self.args).to(self.args.device)
        return model

    def organize_val_log(self, train_loss, val_loss, val_acc, fold_number, epoch):
        if val_loss < self.best_loss:
            torch.save(
                self.model.state_dict(),
                './checkpoints/{}/experiment-{}_fold-{}_seed-{}_best-model.pth'.format(self.log_folder_name, self.exp_name, fold_number, self.args.seed)
            )
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

    def organize_test_log(self, test_loss, test_acc, t_start, t_end, fold_number):
        self.overall_results['durations'].append(t_end - t_start)
        self.overall_results['val_loss'].append(self.best_loss)
        self.overall_results['val_acc'].append(self.best_acc)
        self.overall_results['test_loss'].append(test_loss)
        self.overall_results['test_acc'].append(test_acc)

    def eval(self, loader):
        self.model.eval()
        correct = 0.
        loss = 0.
        for i in range(0, len(loader), self.args.batch_size):
            batch = loader[i:i+self.args.batch_size]
            with torch.no_grad():
                out = self.model(batch)
            pred = out.max(dim=1)[1]
            y = torch.LongTensor([d['label'] for d in batch]).to(self.args.device)
            correct += pred.eq(y).sum().item()
            loss += F.nll_loss(out, y, reduction='sum').item()
        return correct / len(loader), loss / (i+1)

    def train(self):
        self.overall_results = {
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'durations': []
        }

        train_fold_iter = range(1, 11)
        val_fold_iter = list(range(1, 11))

        s_start = time.time()
        for fold_number in train_fold_iter:
            val_fold_number = val_fold_iter[fold_number-2]
            train_loader, val_loader, test_loader = self.load_tvt(fold_number, val_fold_number)

            self.init_fold()
            # Load Model & Optimizer
            self.model = self.load_model()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2rate)


            iter_per_epoch = 50 if args.dataset in i50_dataset \
                                else math.ceil(len(train_loader) / self.args.batch_size)
            if self.args.lr_schedule:
                self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.args.patience * iter_per_epoch, self.args.num_epochs * iter_per_epoch)

            t_start = time.time()
            # K-Fold Training, epoch default 500
            for epoch in range(self.args.num_epochs):
                self.model.train()
                total_loss = 0
                if args.dataset in i50_dataset:
                    e_iter = range(50)
                else:
                    e_iter = range(0, len(train_loader), self.args.batch_size)
                for _ in e_iter:
                    if args.dataset in i50_dataset:
                        indexes = list(range(len(train_loader)))
                        random.shuffle(indexes)
                        batch = [train_loader[i] for i in indexes[:self.args.batch_size]]
                    else:
                        batch = train_loader[_:_+self.args.batch_size]
                    self.optimizer.zero_grad()
                    out = self.model(batch)
                    y = torch.LongTensor([d['label'] for d in batch]).to(self.args.device)
                    loss = F.nll_loss(out, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                    total_loss += loss.item() * len(batch)
                    self.optimizer.step()

                    if self.args.lr_schedule:
                        self.scheduler.step()

                total_loss = total_loss / len(train_loader)

                # Validation
                val_acc, val_loss = self.eval(val_loader)
                self.organize_val_log(total_loss, val_loss, val_acc, fold_number, epoch)
                if self.patience > self.args.patience:
                    break

            t_end = time.time()
            checkpoint = torch.load('./checkpoints/{}/experiment-{}_fold-{}_seed-{}_best-model.pth'.format(self.log_folder_name, self.exp_name, fold_number, self.args.seed))
            self.model.load_state_dict(checkpoint)
            test_acc, test_loss = self.eval(test_loader)
            self.organize_test_log(test_loss, test_acc, t_start, t_end, fold_number)

        s_end = time.time()
        val_acc_mean = np.array(self.overall_results['val_acc']).mean()
        val_acc_std = np.array(self.overall_results['val_acc']).std()
        test_acc_mean = np.array(self.overall_results['test_acc']).mean()
        test_acc_std = np.array(self.overall_results['test_acc']).std()
        print("%.4f\t%.4f\t%.4f\t%.4f\t%.2f" % (test_acc_mean,
                                                test_acc_std,
                                                val_acc_mean,
                                                val_acc_std,
                                                (s_end-s_start)/60))
        sys.stdout.flush()

    def set_experiment_name(self):
        ts = time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime())
        self.log_folder_name = os.path.join(*[self.args.dataset, 'SEP'])
        if not(os.path.isdir('./checkpoints/{}'.format(self.log_folder_name))):
            os.makedirs(os.path.join('./checkpoints/{}'.format(self.log_folder_name)))
        exp_name = str()
        exp_name += "GNN={}_".format(self.args.conv)
        exp_name += "TD={}_".format(self.args.tree_depth)
        exp_name += "GP={}_".format(self.args.global_pooling)
        exp_name += "BS={}_".format(self.args.batch_size)
        exp_name += "HD={}_".format(self.args.hidden_dim)
        exp_name += "LR={}_".format(self.args.lr)
        exp_name += "WD={}_".format(self.args.l2rate)
        exp_name += "GN={}_".format(self.args.grad_norm)
        exp_name += "DO={}_".format(self.args.final_dropout)
        exp_name += "LS={}_".format(self.args.lr_schedule)
        exp_name += "TS={}".format(ts)
        return exp_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEP arguments')
    parser.add_argument('-d', '--dataset', type=str, default="IMDB-BINARY",
                        help='name of dataset')
    parser.add_argument('-c', '--conv', default='GCN', type=str,
                        choices=['GCN', 'GIN', 'GAT', 'Cheb', 'SAGE', 'GAT2', 'Transformer' ],
                        help='message-passing function type')
    parser.add_argument('--num-convs', default=3, type=int)
    parser.add_argument('--num-head', default=1, type=int)
    parser.add_argument('-k', '--tree_depth', type=int, default=3,
                        help='the depth of coding tree (default: 3)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('-e', '--num_epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('-lr', '--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.0005)')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('-fd', '--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('-l2', '--l2rate', type=float, default=0.0001,
                        help='L2 penalty lambda')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
    parser.add_argument("--lr-schedule", action='store_true')
    parser.add_argument("--link-input", action='store_true')
    parser.add_argument('-gp', '--global-pooling', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes: sum or average')
    args = parser.parse_args()

    linkIn_dataset  = ['IMDB-BINARY', 'IMDB-MULTI', 'COLLAB', 'DD']
    i50_dataset = ['IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'PROTEINS', 'PTC', 'DD']
    if args.dataset in linkIn_dataset:
        args.link_input = True

    line = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
            args.dataset,
            args.conv,
            args.tree_depth,
            args.global_pooling,
            args.hidden_dim,
            args.batch_size,
            args.final_dropout,
            args.num_head,
            args.lr_schedule)
    print(line, flush=True)
    trainer = Trainer(args)
    trainer.train()
