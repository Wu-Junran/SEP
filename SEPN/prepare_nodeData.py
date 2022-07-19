#!/usr/bin/env python
# encoding: utf-8
# author:  WuJunran
# email:   wu_junran@buaa.edu.cn
# date:    2021-01-15 16:30:35

import os
import sys
import copy
import torch
import pickle
import numpy as np
import networkx as nx
from .codingTree import PartitionTree
from torch_geometric.datasets import Planetoid


PWD = os.path.dirname(os.path.realpath(__file__))


def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return nx.to_numpy_array(graph, nodelist=nodes)


def trans_to_tree(adj, k=2):
    undirected_adj = np.array(adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_coding_tree(k)
    return y.tree_node


def update_depth(tree):
    # set leaf depth
    wait_update = [k for k, v in tree.items() if v.children is None]
    while wait_update:
        for nid in wait_update:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])


def update_node(tree):
    update_depth(tree)
    d_id = [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h+1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h-1, c)) for c in n.children]
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
    return new_tree


def trans_graph_tree(G, tree_depth):
    # G: networkx.Graph
    # tree_depth: 2-n, not including the leaf layer
    # return dict: {nodeID: {parent: nodeID, children: [], depth: int}}
    adj_mat = trans_to_adj(G)
    tree = trans_to_tree(adj_mat, tree_depth)
    return update_node(tree)


def get_layer_graph(tree, graph, tree_depth):
    layer_graph = [graph]
    for l in range(1, tree_depth):
        partition = {frozenset([tree[c].get('graphID', c) for c in n['children']]): i
                     for i, n in tree.items() if n['depth']==l}
        lg = nx.quotient_graph(layer_graph[-1], partition.keys(), relabel=False)
        lg = nx.relabel_nodes(lg, partition)
        layer_graph.append(lg)
    return layer_graph


def extract_layer_data(T, G, tree_depth):
    node_size = [0] * (tree_depth+1)
    # node size and layer index base
    layer_idx = [0]
    for layer in range(tree_depth+1):
        layer_nodes = [i for i, n in T.items() if n['depth']==layer]
        layer_idx.append(layer_idx[-1] + len(layer_nodes))
        node_size[layer] = len(layer_nodes)

    interLayerEdges = [[] for i in range(tree_depth+1)]
    # edges among layers
    for i, n in T.items():
        if n['depth'] == 0:
            continue
        n_idx = n['ID'] - layer_idx[n['depth']]
        c_base = layer_idx[n['depth']-1]
        interLayerEdges[n['depth']].extend([(n_idx, c-c_base) for c in n['children']])
    interLayer_edgeMat = [torch.LongTensor(es).T for es in interLayerEdges]

    # for gnn
    layer_graphs = get_layer_graph(T, G, tree_depth)
    layer_edgeMat = []
    for l in range(tree_depth):
        g = layer_graphs[l]
        nmap = {n: n-layer_idx[l] for n in g.nodes}
        g = nx.relabel_nodes(g, nmap)
        edges = [[n1, n2] for n1, n2 in g.edges]
        edges.extend([[n2, n1] for n1, n2 in edges])
        edge_mat = torch.LongTensor(edges).T
        layer_edgeMat.append(edge_mat)

    return {'node_size': node_size,
            'interLayer_edgeMat': interLayer_edgeMat,
            'layer_edgeMat': layer_edgeMat
            }


def load_tree(dataname, tree_depth=2):
    if not os.path.exists('data'):
        os.makedirs('data')
    dataset = Planetoid(root=os.path.join(PWD, 'data'), name=dataname)
    data = dataset[0]
    edges = data.edge_index.T.tolist()
    G = nx.Graph()
    G.add_nodes_from(range(data.x.size(0)))
    G.add_edges_from(edges)

    T = trans_graph_tree(G, tree_depth)
    layer_data = extract_layer_data(T, G, tree_depth)
    if not os.path.exists('trees'):
        os.makedirs('trees')
    with open(os.path.join(PWD, 'trees', '%s_%s.pickle' % (dataname, tree_depth)), 'wb') as fp:
        pickle.dump(layer_data, fp)


if __name__ == '__main__':
    dataset = sys.argv[1]
    for depth in range(2, 7):
        print(dataset, depth)
        load_tree(dataset, depth)
