# REAL8 — REAL7 replicates unchanged on TinyStories-Instruct-1M

REAL8 changed only the trained checkpoint. The metric, controls and thresholds
were locked before the run.

Target: `roneneldan/TinyStories-Instruct-1M`, the same 64-hidden / 16-head /
8-layer GPT-Neo shape used in REAL7.

## Result

Across the 16 Q<->K head-subspace transfers:

```text
                               TRAINED        RANDOM INIT

median label-shuffle gain       +0.0108          -0.0001
positive label gain              14 / 16           7 / 16

median random-topology gain     +0.3604          +0.3843

median Q/K distance corr        +0.515           -0.026
permutation p<.05 layers          6 / 8            0 / 8
```

Training specificity:

```text
trained - init median label gain      +0.0109
trained label gain > init             14 / 16
```

Classification:

```text
REAL7_HEAD_SUBSPACE_HIERARCHY_REPLICATES
```

## Layer pattern

```text
layer    Q/K distance correlation    tree label gain (Q->K / K->Q)

0              -0.071                -0.0024 / -0.0010
1              +0.628                +0.0174 / +0.0190
2              +0.170                +0.0055 / +0.0032
3              +0.812                +0.0454 / +0.0401
4              +0.324                +0.0042 / +0.0070
5              +0.482                +0.0098 / +0.0094
6              +0.548                +0.0118 / +0.0122
7              +0.964                +0.1311 / +0.1238
```

Again layer 0 is negative. The strongest layer differs from REAL7, which is
consistent with replication of the aggregate phenomenon rather than replication
of one exact learned tree.

## What is now supported

Across two separately trained 1M checkpoints with the same architecture:

- Q and K organize the 16 attention-head input subspaces into correlated
  gauge-invariant relational geometries;
- a Q-derived labeled tree transfers modestly but consistently to K, and vice
  versa;
- the labeled transfer is absent at random initialization.

## Remaining attacker

REAL7/8 establish shared relational structure, but not that a **tree** is the
best or uniquely appropriate compressed representation.

REAL9 compares source-derived predictors at fixed target evaluation:

```text
raw source distance metric       uncompressed upper bound
equal-star tree                  16 scalar leaf radii
neighbor-joining tree            29 branch lengths + topology
2-D classical MDS                32 nominal coordinates
```

All predictors are created from Q alone when predicting K (and vice versa).
The target gets only one global scale parameter.

## Stopping line

If the tree does not beat matched-size MDS, rename the surviving result from
`hierarchy` to `shared gauge-invariant head-subspace geometry`.

If the tree wins across both trained models and not at random initialization,
then a hierarchical compression claim is justified.
