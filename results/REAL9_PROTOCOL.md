# REAL9 protocol — is the replicated shared head geometry actually tree-specific?

REAL7 and REAL8 replicated a gauge-invariant Q/K head-subspace relationship.
REAL9 attacks the word **hierarchy**.

For each layer and direction Q->K / K->Q, build every representation from the
source head-subspace distance matrix only. The target supplies one nonnegative
global scale and nothing else.

## Source-only representations

```text
RAW
  120 pairwise source distances
  uncompressed ceiling; not a complexity competitor

STAR
  D_ij = r_i + r_j
  16 nonnegative leaf radii

NJ TREE
  neighbor-joining on the source metric
  29 branch lengths + discrete topology

MDS2
  classical 2-D MDS of the source metric
  16 x 2 = 32 nominal scalar coordinates
```

The MDS attacker is intentionally close to the tree's scalar budget, and the
tree's discrete topology cost is not charged. This favors the tree if anything.

## Data

Run all 16 transfers on both trained models:

- `roneneldan/TinyStories-1M`
- `roneneldan/TinyStories-Instruct-1M`

and repeat on fresh random-initialized copies of both architectures.

## Preregistered tree-specific pass

Pooled across the 32 trained transfers:

```text
median (MDS2 error - tree error)        >= .01
tree beats MDS2                        >= 75%
median tree-vs-MDS2 gain in each model >= .005

median (star error - tree error)        >= .01
tree beats star                        >= 75%
```

Training specificity additionally requires:

```text
trained - init median tree-vs-MDS2 gain >= .005
trained tree advantage > init           >= 75% paired transfers
```

## Stopping line

If this fails, keep the replicated result but rename it:

> shared gauge-invariant Q/K head-subspace geometry

not hierarchy.

Do not add a more flexible tree family after seeing the answer.
