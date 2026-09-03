# REAL6 protocol — exact gauge attack on the remaining raw Q/K tree signal

REAL5 closed the tree interpretation of the actual functional attention-score
operator. The only apparently positive tree result left is the strong REAL2
geometry in W_Q and W_K separately.

Those matrices have an exact coordinate freedom.

For every attention head choose an orthogonal R_h and transform:

```text
W_Q,h' = R_h^T W_Q,h
W_K,h' = R_h^T W_K,h
```

Then:

```text
W_Q,h'^T W_K,h'
  = W_Q,h^T R_h R_h^T W_K,h
  = W_Q,h^T W_K,h
```

So every pre-softmax attention score is unchanged. The model computes the same
function, apart from floating-point roundoff.

Because the transformation is left-orthogonal, each full W_Q / W_K matrix also
keeps its **exact singular values**. Therefore each matrix can be compared to the
same exact-spectrum null before and after gauge rotation.

## Assay

For all 8 layers x {Q,K} = 16 raw matrices:

1. compute the baseline p95 four-point gap on 4096 fixed quartets;
2. generate 64 exact-spectrum randomized-orientation null matrices;
3. express baseline tree-likeness as z relative to that null;
4. apply 16 independent random head-wise Q/K gauges;
5. recompute z for every gauge using the unchanged spectrum null;
6. verify every head score operator is invariant;
7. mutate the actual model with one gauge and verify logits/argmax are invariant.

## Preregistered gauge-dependence criterion

Functional invariance must first pass:

```text
max relative error in W_Q,h^T W_K,h < 1e-12
model relative logit error             < 1e-5
argmax token agreement                  = 1.0
```

Then call the raw tree signal gauge-dependent if:

```text
median absolute change in tree z       >= 1.0
baseline z > gauge-median z on         >=75% of 16 matrices
median fraction of gauges still
significant vs same-spectrum null      <=0.5
```

## Stopping line

If this passes, stop. The raw Q/K tree geometry is a property of one convenient
parameter basis, not of the learned attention function. Combined with REAL5,
that closes the TinyStories hidden-tree hypothesis.

If the tree score unexpectedly survives exact function-preserving gauges, then
that invariance itself would need explaining before any scaling.
