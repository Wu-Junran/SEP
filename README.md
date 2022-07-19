# Structural Entropy Guided Graph Hierarchical Pooling

<img src="./figs/sep-framework.pdf" width="400" height="200">

This code reproduces the experimental results obtained with the SEP layer as presented in the ICML 2022 paper:

[ICML](https://proceedings.mlr.press/v162/wu22b/wu22b.html) 
[arXiv](https://arxiv.org/abs/2206.13510) 

Junran Wu, Xueyuan Chen, Ke Xu, Shangzhe Li


## Dependencies

* Python 3.7
* PyTorch 1.8
* PyTorch Geometric 2.0.1

## Graph Reconstruction - Local Structure Damage
A graph reconstruction experiment, which quantifies the structural information retained by pooling layer, is conducted to directly reveal the damage caused by previous hierarchical pooling methods to graph's local structures.

<div style="display:inline-block;text-align:center">
    <img src="./figs/origin-ring.pdf" width="110" height="110">
    <div>Original Ring</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/topk-ring.pdf" width="110" height="110">
    <div>TopKPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/sag-ring.pdf" width="110" height="110">
    <div>SAGPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/asap-ring.pdf" width="110" height="110">
    <div>ASAPPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/DiffPool-ring.png" width="112" height="112">
    <div>DiffPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/minCutPool-ring.pdf" width="110" height="110">
    <div>minCutPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/SEP-U-ring.pdf" width="110" height="110">
    <div>SEP</div>
</div>


<div style="display:inline-block;text-align:center">
    <img src="./figs/origin-grid.pdf" width="110" height="110">
    <div>Original Grid</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/topk-grid.pdf" width="110" height="110">
    <div>TopKPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/sag-grid.pdf" width="110" height="110">
    <div>SAGPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/asap-grid.pdf" width="110" height="110">
    <div>ASAPPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/DiffPool-grid.png" width="112" height="112">
    <div>DiffPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/minCutPool-grid.pdf" width="110" height="110">
    <div>minCutPool</div>
</div>
<div style="display:inline-block;text-align:center">
    <img src="./figs/SEP-U-grid.pdf" width="110" height="110">
    <div>SEP</div>
</div>

Run ```python trainer_sepu_synthetic.py -d ring/grid``` to train an autoencoder and compute the reconstructed graph. It is possible to switch between the `ring` and `grid` graphs. Results are provided in terms of the Mean Squared Error.


## Classification

To run the proposed model in the paper, SEP-G and SEP-N, see the corresponding folder:

* Graph Classification: [SEP-G](https://github.com/Wu-Junran/SEP/tree/master/SEPG)
* Node Classification: [SEP-N](https://github.com/Wu-Junran/SEP/tree/master/SEPN)



## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work. </br>

```BibTex
@inproceedings{wu2022structural,
  title={Structural Entropy Guided Graph Hierarchical Pooling},
  author={Wu, Junran and Chen, Xueyuan and Xu, Ke and Li, Shangzhe},
  booktitle={International Conference on Machine Learning},
  pages={24017--24030},
  year={2022},
  organization={PMLR}
}
```
