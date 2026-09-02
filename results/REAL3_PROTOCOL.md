# REAL3 protocol — does Q/K topology transfer out of sample?

REAL2 found its strongest topology-sensitive signal in Q and K. That could still
be circular because each tree was inferred from the same matrix it helped fit.

REAL3 removes that circularity.

## Why Q <-> K is the clean transfer

Query and key output coordinates are explicitly paired by the attention dot
product. Therefore leaf label d in Q corresponds to leaf label d in K.

For each of the 8 layers:

```text
infer tree from Q rows -> freeze it -> fit only leak/wrappers to K
infer tree from K rows -> freeze it -> fit only leak/wrappers to Q
```

That gives 16 out-of-sample topology transfers.

## Attackers

Each transfer is compared against two topology controls:

1. a random binary topology carrying the source tree's exact branch-length multiset;
2. the **same exact source topology and branch lengths**, but with leaf identities
   randomly permuted.

The second attacker is particularly important: if the leaf labeling discovered
from Q is meaningful for K, shuffling those coordinate identities should hurt.

## Training control

The entire 16-transfer assay is repeated on a freshly random-initialized model
created from the exact same TinyStories-1M configuration.

This distinguishes a structure produced by training from an estimator bias or
architectural initialization artifact.

## Preregistered pass

Trained Q/K transfer must satisfy:

```text
median gain >= .005 versus random topology
positive gain in >=75% of 16 transfers

AND

median gain >= .005 versus shuffled leaf labels
positive gain in >=75% of 16 transfers
```

To call it training-specific:

```text
trained median label gain - random-init median label gain >= .005
AND
trained label gain > init label gain on >=75% of paired transfers
```

## Stopping line

Only a training-specific out-of-sample transfer earns the phrase
`reusable learned channel geometry` and any move to a larger model.
