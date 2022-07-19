#!/usr/bin/env python
# encoding: utf-8

import os
import copy
import time
import torch
import pickle
import networkx as nx


PWD = os.path.dirname(os.path.realpath(__file__))


def get_layer_graph(tree, graph, tree_depth):
    layer_graph = [graph]
    for l in range(1, tree_depth):
        partition = {frozenset([tree[c].get('graphID', c) for c in n['children']]): i
                     for i, n in tree.items() if n['depth']==l}
        lg = nx.quotient_graph(layer_graph[-1], partition.keys(), relabel=False)
        lg = nx.relabel_nodes(lg, partition)
        layer_graph.append(lg)
    return layer_graph


def extract_tree(graph, tree_depth):
    leaf_size = len(graph['G'])
    tree = {'label': graph['label'],
            'node_size': [0] * (tree_depth+1),
            'edges': [[] for i in range(tree_depth+1)],
            'node_features': [0] * leaf_size,  # 叶子结点的特征
            'node_degrees': [0] * leaf_size,
            }
    old_tree = copy.deepcopy(graph['tree'])
    # tree layer mask
    layer_idx = [0]
    for layer in range(tree_depth+1):
        layer_nodes = [i for i, n in old_tree.items() if n['depth']==layer]
        layer_idx.append(layer_idx[-1] + len(layer_nodes))
        tree['node_size'][layer] = len(layer_nodes)

    for i, n in old_tree.items():
        # edge
        if n['depth'] > 0:
            n_idx = n['ID'] - layer_idx[n['depth']]
            c_base = layer_idx[n['depth']-1]
            tree['edges'][n['depth']].extend([(n_idx, c-c_base) for c in n['children']])
            continue
        # leaf: node feature
        graphID = n.get('graphID', n['ID'])
        nid = n['ID']
        tree['node_features'][nid] = int(graph['G'].nodes[graphID].get('tag', 0))
        tree['node_degrees'][nid] = graph['G'].degree[graphID]

    # for gin
    layer_graphs = get_layer_graph(old_tree, graph['G'], tree_depth)
    layer_edgeMat = []
    for l in range(tree_depth):
        g = layer_graphs[l]
        nmap = {n: n-layer_idx[l] for n in g.nodes}
        g = nx.relabel_nodes(g, nmap)
        edges = [[n1, n2] for n1, n2 in g.edges]
        edges.extend([[n2, n1] for n1, n2 in edges])
        edge_mat = torch.LongTensor(edges).transpose(0, 1)
        layer_edgeMat.append(edge_mat)
    tree['graph_mats'] = layer_edgeMat
    return tree


def integrate_label(trees):
    labels = [t['label'] for t in trees]
    labels = list(set(labels))
    labels.sort()
    for t in trees:
        t['label'] = labels.index(t['label'])
    return trees


def one_hot_features(trees):
    label_set = list(set(sum([t['node_features'] for t in trees], [])))
    label_set.sort()
    for t in trees:
        leaf_size = t['node_size'][0]
        node_features = torch.zeros(leaf_size, len(label_set))
        node_features[range(leaf_size), [label_set.index(d) for d in t['node_features']]] = 1
        t['node_features'] = node_features


def add_additional_features(trees, fField):
    fset = list(set(sum([t[fField] for t in trees], [])))
    fset.sort()
    for t in trees:
        leaf_size = t['node_size'][0]
        features = torch.zeros(leaf_size, len(fset))
        features[range(leaf_size), [fset.index(f) for f in t[fField]]] = 1
        t['node_features'] = torch.cat([t['node_features'], features], dim=1)


def load_tree(dataset, tree_depth=2):
    t_path = os.path.join(PWD, 'trees', '%s_%s.pickle' % (dataset, tree_depth))
    with open(t_path, 'rb') as fp:
        graphs = pickle.load(fp)
    trees = [extract_tree(g, tree_depth) for g in graphs]
    integrate_label(trees)
    one_hot_features(trees)
    add_additional_features(trees, 'node_degrees')
    return trees


if __name__ == '__main__':
    start = time.time()
    load_tree('IMDB-BINARY', 3)
    end = time.time()
    print((end-start)/60.)
