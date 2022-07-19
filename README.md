# Structural Entropy Guided Graph Hierarchical Pooling

<p align="middle">
<img src="./figs/sep-framework.jpg" width="400" height="200">
</p>
    
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
    
Original |  TopKPool | SAGPool | ASAPPool | DiffPool | minCutPool | SEP
:---:|:---:|:---:|:---:|:---:|:---:|:---:
<img src="./figs/origin-ring.jpg" width="110" height="110"> |  <img src="./figs/topk-ring.jpg" width="110" height="110"> | <img src="./figs/sag-ring.jpg" width="110" height="110"> | <img src="./figs/asap-ring.jpg" width="110" height="110"> | <img src="./figs/DiffPool-ring.png" width="110" height="110"> | <img src="./figs/minCutPool-ring.jpg" width="110" height="110"> | <img src="./figs/SEP-U-ring.jpg" width="110" height="110">
<img src="./figs/origin-grid.jpg" width="110" height="110"> |  <img src="./figs/topk-grid.jpg" width="110" height="110"> | <img src="./figs/sag-grid.jpg" width="110" height="110"> | <img src="./figs/asap-grid.jpg" width="110" height="110"> | <img src="./figs/DiffPool-grid.png" width="110" height="110"> | <img src="./figs/minCutPool-grid.jpg" width="110" height="110"> | <img src="./figs/SEP-U-grid.jpg" width="110" height="110">

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
