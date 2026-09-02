# REAL2 protocol — two ways the v0.1 negative could still hide structure

REAL1 killed the naive radial-tree reconstruction of raw TinyStories-1M weights.
REAL2 stays at d=64 and asks two narrower questions before any scaling.

## A. Is the induced channel geometry itself unusually tree-like?

For each trained square projection W, compute cosine distance between normalized rows.
Measure the additive-tree four-point gap on 4096 fixed sampled quartets.

The attacker preserves the **exact singular values** of W but independently
randomizes its left and right singular-vector orientations. This keeps dimension
and spectrum while destroying learned channel orientation.

64 null matrices are generated per trained matrix.

Lower four-point gap is more tree-like. The aggregate geometry signal is
preregistered as:

```text
median tree-likeness z >= 1
AND
at least 8 / 32 matrices have empirical lower-tail p < .05
```

This is deliberately stricter than pointing at one favorable layer.

## B. Does a genuine graph Green operator make inferred topology matter?

REAL1 used a radial kernel exp(-alpha d_tree). REAL2 instead builds a weighted
tree Laplacian from the neighbor-joining branch lengths:

```text
g_edge ~ 1 / branch_length
L_tree = weighted graph Laplacian
K_green = [(L_tree + leak I)^-1]_leaf,leaf
W_hat = diag(a) K_green diag(b)
```

Leak is searched on a fixed logarithmic grid. Signed diagonal input/output
wrappers are fit by alternating least squares.

The crucial attacker is stronger than REAL1's random tree: every random binary
topology receives **exactly the inferred tree's branch-length multiset**, shuffled
onto the random topology. Thus an advantage cannot be blamed on a more convenient
length distribution.

Four random topologies are used per matrix. The preregistered aggregate Green
signal is:

```text
median(random-topology error - inferred-topology error) >= 0.005
AND
inferred topology wins on at least 75% of the 32 matrices
```

Matched-nominal-budget SVD and top-|weight| sparse attackers remain in the table.

## Stopping line

If neither A nor B is topology-sensitive, do not scale to TinyStories-8M or
DistilGPT2. The first real model will have told us that this black-box tree idea
is not carrying useful signal in its present forms.

If A survives but B fails, the result is still interesting but narrower:
learned channel **geometry** may be tree-like even though the raw signed weight
matrix is not translatable into a single graph-generated mixing operator.
