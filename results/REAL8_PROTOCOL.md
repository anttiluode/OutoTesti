# REAL8 protocol — locked replication of REAL7

REAL7 passed on `roneneldan/TinyStories-1M`.

REAL8 changes only the trained checkpoint:

`roneneldan/TinyStories-Instruct-1M`

It has the same GPT-Neo size/shape used by REAL7 (hidden size 64, 16 heads,
8 layers) and was trained on the TinyStories-Instruct dataset.

## No protocol changes

Use the exact REAL7 object and thresholds:

- each attention head is its gauge-invariant 4-D row subspace in R^64;
- normalized Grassmann chordal distance between the 16 head subspaces;
- Q tree frozen and tested on K, and K tree frozen and tested on Q;
- 128 leaf-label shuffles per transfer;
- 128 random topologies carrying the source branch-length multiset;
- a fresh random-initialized copy of the exact Instruct architecture;
- direct Q/K head-distance correlation with label permutation.

Pass criteria remain:

```text
median label-shuffle gain              >= .01
positive label gain                    >= .75
median random-topology gain            >= .01
positive random gain                   >= .75

trained - init median label gain       >= .01
trained label gain > init              >= .75 paired transfers

median Q/K distance correlation        >= .20
permutation p<.05                      >= 4 / 8 layers
```

## Stopping line

If all locked criteria pass, REAL7 is replicated and the next experiment must
attack **tree-specificity**, not scale.

If they fail, do not average the two models together or relax thresholds.
Call REAL7 model-specific.
