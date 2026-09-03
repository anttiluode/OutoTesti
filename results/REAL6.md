# REAL6 — raw Q/K tree signal weakens under gauge, but does not disappear

REAL6 attacked the remaining positive REAL2 result with an exact function-preserving
Q/K gauge.

For each head:

```text
W_Q,h' = R_h^T W_Q,h
W_K,h' = R_h^T W_K,h
```

with orthogonal R_h. This preserves every head score operator exactly in real
arithmetic.

## Functional invariance passed

```text
max relative error in W_Q,h^T W_K,h   1.198e-15
model relative logit error             8.435e-7
argmax token agreement                 1.000
```

So the transformed model is functionally equivalent to numerical precision.

## Raw Q/K tree geometry moves, but survives

Across all 8 layers x {Q,K} = 16 matrices:

```text
baseline median tree-likeness z        +3.803
median random-gauge z                  +1.694
median absolute z shift                 1.577

baseline significant vs spectrum null  11 / 16
median gauge significant fraction       0.562
baseline z > gauge-median z             0.625
```

The preregistered gauge-dependence criterion required baseline z to exceed the
gauge median on >=75% of matrices and the median gauge significance fraction to
fall to <=0.5. It does not pass.

Classification:

```text
RAW_QK_TREE_SIGNAL_SURVIVES_RANDOM_ORTHOGONAL_GAUGES
```

## Interpretation

This is not a rescue of the original 64-leaf circuit picture.

The gauge rotations freely mix the four Q/K coordinates **inside each head**.
Individual raw coordinate identities are therefore not functionally meaningful.

What the gauge leaves invariant is the 4-dimensional row subspace used by each
head in the 64-dimensional residual-stream input space.

The surviving tree-like signal may therefore live one level up:

```text
16 attention heads
each head = a 4-D input subspace
relative principal-angle / overlap geometry between head subspaces
```

That object is exactly invariant to the REAL6 gauge and has a direct functional
meaning: it describes which residual-stream directions different heads read.

## Next stopping line

REAL7 tests only this gauge-invariant head-subspace geometry.

Do not use a 64-leaf raw-weight tree again. Infer a 16-leaf tree from Q head
subspaces and ask whether it predicts the K head-subspace geometry in the same
layer, with leaf-label shuffles, same-length random topology, and random-init
controls.

If that cross-transfer fails, stop: REAL6's survivor is merely block/subspace
structure, not a reusable hierarchy.
