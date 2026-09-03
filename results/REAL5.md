# REAL5 — the REAL4 held-out effect is spectral geometry, not tree topology

REAL4 found a large training-dependent held-out relational-geometry effect in
the actual gauge-invariant attention score operators M_h. Every M_h has rank at
most 4, so REAL5 preserved each head's **exact singular values** and randomized
only its left/right orientation.

The measurement was fitter-free: infer four-point quartet splits from half the
columns and ask whether the same split recurs in the held-out half.

## Result

Across all 128 trained heads:

```text
median observed quartet agreement     0.9011
median exact-spectrum null agreement  0.8997
median agreement gain                -0.0027
median stability z                   -0.155
empirical p < .05                     0 / 128
```

Random initialization:

```text
median agreement gain                -0.0008
median stability z                   -0.025
```

Training did not improve the spectrum-controlled quantity:

```text
trained - init median gain           -0.0019
paired trained gain > init            60 / 128   (0.469)
```

Classification:

```text
HELDOUT_TREE_EFFECT_EXPLAINED_BY_SPECTRAL_GEOMETRY
```

## Meaning

The large REAL4 held-out result was real, but it was not tree-specific. Once
rank and singular-value concentration are held exactly fixed, learned quartet
topology is indistinguishable from randomized orientation.

So the gauge-invariant functional attention score operator does **not** support
the hidden-tree interpretation in TinyStories-1M.

## What remains to close

REAL2/REAL3 found strong tree-like geometry in W_Q and W_K separately. Those
matrices are not functionally identifiable: each attention head admits a joint
orthogonal Q/K coordinate rotation that leaves all attention scores unchanged.

REAL6 is therefore a gauge attack, not another rescue attempt:

```text
W_Q,h' = R_h^T W_Q,h
W_K,h' = R_h^T W_K,h

M_h' = W_Q,h'^T W_K,h' = M_h
```

If the raw Q/K tree scores change under these exactly function-preserving
rotations, the remaining apparent geometry is a parameterization artifact.

## Stopping line

After REAL6, stop the TinyStories tree chase. Either raw Q/K geometry is
gauge-dependent and the hypothesis closes, or it is unexpectedly invariant and
deserves a new explanation. Do not add another flexible tree fitter.
