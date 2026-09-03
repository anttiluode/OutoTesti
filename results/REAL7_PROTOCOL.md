# REAL7 protocol — gauge-invariant hierarchy among attention-head subspaces

REAL6 showed that random within-head Q/K coordinate rotations weaken but do not
erase the raw Q/K tree signal. Those rotations preserve each head's 4-D row
subspace in the 64-D residual-stream input space.

REAL7 tests that exact gauge-invariant survivor and nothing below it.

## Object

For every Q or K projection, split its 64 output rows into 16 heads x 4 rows.
For each head, take the 4-D row span in R^64.

The distance between heads i,j is normalized Grassmann chordal distance:

```text
d(i,j)^2 = [k - ||U_i^T U_j||_F^2] / k
k = 4
```

where U_i and U_j are orthonormal bases of the head row subspaces.

This distance is exactly invariant to any within-head orthogonal basis rotation.

## Cross-transfer

For each of 8 layers:

```text
Q head-subspace distances -> neighbor-joining tree -> freeze
                                        |
                                        +-> test on K head-subspace distances

K head-subspace distances -> neighbor-joining tree -> freeze
                                        |
                                        +-> test on Q head-subspace distances
```

Target evaluation fits only one global distance scale. No topology or branch
length is re-fit on the target metric.

That gives 16 transfers.

## Attackers

Each transfer gets 128 controls:

1. same exact source tree and branch lengths, but the 16 head labels are permuted;
2. random binary topology carrying the exact source branch-length multiset;
3. the full assay repeated in a fresh random-initialized copy of the exact model.

A supporting non-tree measurement also compares the upper-triangle Q/K head
distance matrices directly, with head-label permutation as its null.

## Preregistered pass

Tree transfer:

```text
median target-error gain >= .01 vs label shuffle
positive on >=75% of 16 transfers
median gain >= .01 vs random topology
positive on >=75% of 16 transfers
```

Training specificity:

```text
trained - init median label gain >= .01
trained label gain > init on >=75% paired transfers
```

Supporting Q/K distance alignment:

```text
median QK distance correlation >= .2
permutation p<.05 in >=4 / 8 layers
```

## Stopping line

If REAL7 fails, stop the TinyStories tree chase completely.

If it passes, the claim is narrow and gauge-safe:

> training organizes the 16 attention-head input subspaces into a reusable
> Q/K hierarchical geometry.

That would still not imply dendrites, cables, or literal graph-derived weights.
