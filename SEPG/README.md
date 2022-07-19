# SEP-G

## Dependencies

* Python 3.7.11
* PyTorch 1.8.0
* PyTorch Geometric 2.0.1

## Run

To run the proposed model for graph classification in the paper (Section 4.2), run following commands:

* First, transform the graph to its corresponding coding tree.

```python
python trans_graph.py
```

* Second, parameter tuning, including 

```
  --dataset {IMDB-BINARY, IMDB-MULTI, COLLAB, MUTAG, PROTEINS, DD, NCI1}
    name of dataset
  --hidden_dim {64,128}
    number of hidden units
  --batch_size BATCH_SIZE {32, 128}
    input batch size for training (default: 32)
  --final_dropout {0, 0.5}
    final layer dropout
  --lr-schedule {True, False}
    just adopted from GMT
  --num-head {1, 2, 4}
    for SEP variants, the main results do not need
```

e.g.,

```python
python trainer_sep_args.py --dataset IMDB-BINARY --hidden_dim 64 --batch_size 128 --final_dropout 0 --lr_schedule
```

