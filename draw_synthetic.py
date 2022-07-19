import sys
import numpy as np
from pygsp import graphs
import matplotlib.pyplot as plt


dataset = sys.argv[1]
# LOAD DATASET
if dataset == 'ring':
    G = graphs.Ring(N=200)
elif dataset == 'grid':
    G = graphs.Grid2d(N1=30, N2=30)
X = G.coords.astype(np.float32)


def draw_original():
    plt.figure(figsize=(4, 4))
    pad = 0.1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    colors = X[:, 0] + X[:, 1]
    plt.scatter(*X[:, :2].T, c=colors, s=8, zorder=2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if dataset == 'ring':
        plt.axvline(0, c='k', alpha=0.2)
        plt.axhline(0, c='k', alpha=0.2)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("figs/origin-%s.pdf" % dataset)


if __name__ == '__main__':
    draw_original()
