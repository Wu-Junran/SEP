# SEP-N

## Dependencies

* Python 3.7.11
* PyTorch 1.8.0
* PyTorch Geometric 2.0.1

## Run

To run the proposed model for node classification in the paper (Section 4.3), run following commands:

* First, transform the graph to its corresponding coding tree.

```python
python prepare_nodeData.py Cora
```

* Second, parameter tuning, including 

```
  --dataset {Cora,Pubmed,Citeseer} 
    name of dataset
  --num_blocks {1,2,3,4,5}
  --tree_depth {2,3,4,5,6}
    the depth of coding tree (=num_blocks+1)
  --hidden_dim {16,32,128,256}
    number of hidden units
  --conv_dropout {0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
    conv layer dropout
  --pooling_dropout {0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
    sep layer dropout
  --l2rate {0.0005,0.02}
    L2 penalty lambda
```

e.g.,

```python
python trainer_sepu_args.py --dataset Cora --num_blocks 2 --tree_depth 3 --hidden_dim 128 --conv_dropout 0 --pooling_dropout 0 --l2rate 0.02
```

