# Real 1 — TinyStories-1M

First real trained-model audit. Model: `roneneldan/TinyStories-1M`.

32 square attention projections were tested: Q/K/V/output across 8 transformer layers. Each matrix is 64 x 64.

## Result

The v0.1 tree projection does **not** earn a hidden-geometry claim.

```text
median relative Frobenius error

inferred tree            0.90993
random tree median       0.91027
matched-budget SVD       0.91029   (rank 1)
top-|weight| sparse      0.78884

median |tree-random| gap 0.00010
median |tree-SVD| gap    0.00007

tree beats random        21 / 32  (by mostly tiny margins)
tree beats SVD           17 / 32  (by mostly tiny margins)
tree beats sparse         3 / 32
```

The inferred topology is therefore doing almost no useful work in this operator family. The fit behaves essentially like a rank-1 structured approximation.

## Functional replacement

Every approximation was also inserted back into the trained language model one matrix at a time on the same fixed text.

Across the 32 interventions:

```text
median mean-logit KL

tree                     0.4404
SVD                      0.4432
sparse                   0.3601

median argmax agreement

tree                     0.6818
SVD                      0.6818
sparse                   0.6818
```

Matrix reconstruction and behavioral damage are not identical. There are individual matrices where the sparse approximation has much lower Frobenius error yet damages logits more than tree/SVD, and vice versa. That is worth retaining as a separate diagnostic, but it does not rescue the tree topology claim.

## Family pattern

Q/K projections are substantially easier for all low-complexity approximations than V/output projections. Median tree errors are approximately:

```text
Q       0.822
K       0.800
V       0.957
OUT     0.953
```

This is a property of the tested approximation family, not evidence that Q/K are tree-routed.

## Four-point diagnostic

The learned row-signature distances have p95 normalized four-point gaps between about 0.090 and 0.128 (median 0.104). This receipt does not interpret those numbers as tree-likeness because a spectrum/dimension-matched random-distance baseline has not yet been preregistered and run.

## What died

The original quick story — infer a neighbor-joining tree from a trained weight matrix, compile the tree back, and discover a clean circuit diagram — does not survive this first real model.

More specifically, the v0.1 family

`diag(a) exp(-alpha d_tree) diag(b)`

is too restrictive and its inferred topology adds essentially nothing over random topology.

## What remains

Two questions remain legitimate:

1. Does a different graph-generated operator family (for example a true Green/resolvent operator rather than a radial tree kernel) retain topology-sensitive signal?
2. Is the induced **channel geometry** itself unusually tree-like relative to matched random/spectral controls, even though raw signed `W` is not reconstructible by a single tree kernel?

Do not scale to TinyStories-8M until one of those questions produces a topology-sensitive result on d=64.

Raw matrix reconstruction values are committed in `results/tinystories_1m/matrix_audit.csv`. The full behavior JSON was produced as the `tinystories-1m-audit` artifact in GitHub Actions run 33658650213.
