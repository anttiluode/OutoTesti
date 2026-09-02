# REAL5 protocol — is REAL4 just low-rank geometry?

REAL4 found a large training-dependent held-out topology effect in the actual
head score operator M_h, but every M_h has rank at most 4. Random initialization
also showed positive held-out tree gain.

REAL5 asks the decisive confound question without fitting a tree operator.

## Exact-spectrum null

For each of all 128 trained head score matrices:

1. preserve its **exact singular values**;
2. independently randomize left and right singular-vector orientations;
3. generate 32 such null matrices.

This preserves rank, singular-value concentration and total scale. It destroys
only the learned orientation/relational geometry.

## Fitter-free held-out quartet test

For the real head and every null:

1. split the 64 columns into 32 train / 32 test columns;
2. compute row cosine distances separately in train and test halves;
3. sample 4096 four-point quartets;
4. on the train half, choose which of the three four-point pair sums is smallest;
5. ask whether the **same quartet split** is selected in the held-out half.

Repeat over four column splits and take median agreement.

No neighbor-joining tree, branch-length fit, Green operator, diagonal wrapper or
other model is optimized in this test.

## Preregistered pass

Relative to each head's exact-spectrum orientation null:

```text
trained median stability z >= 1
empirical p<.05 on >=25% of 128 heads
median agreement gain >= .02
positive agreement gain on >=75% of heads
```

Training-specificity additionally requires:

```text
trained median gain - random-init median gain >= .02
trained gain > init gain on >=75% of paired heads
```

## Stopping line

If REAL5 fails, close the tree-topology interpretation of REAL4: its held-out
effect is adequately explained by low-rank/spectral geometry.

If REAL5 passes, do **not** jump to a larger model. First replicate on a second
independently trained small model/checkpoint.
