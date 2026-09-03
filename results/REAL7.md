# REAL7 — a gauge-invariant hierarchy survives at the attention-head subspace level

REAL6 showed that individual Q/K coordinates are not the stable object. What
survived the exact gauge was each head's 4-D row subspace in the 64-D residual
stream.

REAL7 represented all 16 heads by those gauge-invariant subspaces and measured
pairwise Grassmann chordal distance.

For every layer, a tree inferred from Q head-subspace distances was frozen and
tested on K head-subspace distances, then the direction was reversed.

## Result

Across 16 Q<->K transfers:

```text
                               TRAINED        RANDOM INIT

median label-shuffle gain       +0.0131          -0.0001
positive label gain              14 / 16           7 / 16

median random-topology gain     +0.3642          +0.3825
positive random gain             16 / 16          16 / 16

median Q/K distance corr        +0.448           -0.026
permutation p<.05 layers          6 / 8            0 / 8
```

Training specificity:

```text
trained - init median label gain      +0.0132
trained label gain > init             14 / 16
```

Classification:

```text
GAUGE_INVARIANT_HEAD_SUBSPACE_HIERARCHY_PRESENT
```

## Layer pattern

```text
layer    Q/K distance correlation    tree label gain (Q->K / K->Q)

0              -0.094                -0.0000 / -0.0016
1              +0.427                +0.0160 / +0.0134
2              +0.698                +0.0246 / +0.0261
3              +0.468                +0.0127 / +0.0117
4              +0.211                +0.0018 / +0.0023
5              +0.902                +0.0596 / +0.0592
6              +0.388                +0.0107 / +0.0113
7              +0.855                +0.0539 / +0.0470
```

Layer 0 is a clear negative and layer 4 is weak. The effect is not universal
across depth.

## What the result means

The result is invariant to the within-head Q/K basis rotations that preserve
attention scores. The leaves are attention **heads**, not arbitrary raw weight
coordinates.

A head leaf represents the 4-D residual-stream input subspace read by that head.
The positive label-shuffle gain says the learned head identities matter: a Q
hierarchy predicts the corresponding K head-subspace geometry better than the
same tree with head labels permuted.

The direct Q/K distance correlation supports the same conclusion without any
tree fitter.

## What it does not mean

The large random-topology gain is not training-specific; random initialization
also shows it. Therefore the mere fact that a neighbor-joining tree beats a
random topology is not evidence of learning.

The training-specific evidence is the **labeled Q/K cross-transfer** and direct
distance-matrix alignment.

This does not establish that tree geometry is the unique or best compression of
that shared relational structure. A generic metric/low-dimensional explanation
may still beat the hierarchy.

## Next stopping line

Replicate unchanged on a separately trained small model before changing the
metric, thresholds, or model size.

The first replication target is `roneneldan/TinyStories-Instruct-1M`: the same
64-hidden / 16-head / 8-layer GPT-Neo architecture trained on the
TinyStories-Instruct dataset.

If the replication fails, treat REAL7 as model-specific. If it passes, the next
attacker is tree-specificity: compare the tree compression against simpler
representations of the same shared Q/K head-subspace geometry.
