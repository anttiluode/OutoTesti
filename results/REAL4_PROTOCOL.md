# REAL4 protocol — audit the thing attention actually uses

REAL3 found trained Q/K row geometry but no transferable labeled Q/K leaf circuit.
That motivates a change of object rather than another fitter.

## Gauge-invariant head score operator

For one attention head, PyTorch's linear convention gives:

```text
q = x W_Q,h^T
k = y W_K,h^T

q.k = x [W_Q,h^T W_K,h] y^T
```

Define:

```text
M_h = W_Q,h^T W_K,h / sqrt(head_dim)
```

`M_h` is the actual bilinear operator controlling that head's pre-softmax token-pair
score. Unlike W_Q or W_K separately, it is invariant to the ordinary joint Q/K
latent-coordinate rotations that leave attention scores unchanged.

TinyStories-1M has hidden size 64, 16 heads, and 8 layers, so REAL4 audits all
128 head score operators. No head selection is allowed.

## Attacker A — exact-spectrum orientation null

Run the same four-point channel-geometry audit as REAL2, preserving each M_h's
exact singular values while randomizing left/right singular-vector orientation.

Preregistered geometry pass:

```text
trained median tree-likeness z >= 1
AND
empirical p<.05 on at least 25% of all heads
```

## Attacker B — held-out columns

Same-matrix topology inference can always overfit. For every head, four times:

1. split the 64 columns of M_h into 32 train / 32 test columns;
2. infer the row tree from train columns only;
3. freeze that tree;
4. evaluate how well its leaf distances predict row geometry on held-out columns;
5. compare against the same tree with leaf labels shuffled and random topologies
   carrying the same branch-length multiset.

Only one global distance scale is fit on the held-out metric. No branch length or
topology is re-fit to test columns.

Preregistered held-out pass:

```text
median relative-error gain >= .01 versus label shuffle
median relative-error gain >= .01 versus random topology
positive gain on >=75% of heads for both
```

## Attacker C — random initialization

Repeat everything on a fresh random-initialized model from the exact same config.

Training-specific held-out signal requires:

```text
trained median label gain - init median label gain >= .01
AND
trained gain > init gain on >=75% of paired heads
```

## Stronger secondary test — Green translation

For each full M_h, also compile its inferred tree as the true Green family:

```text
K = [(L_tree + leak I)^-1]_leaves
M_hat = diag(a) K diag(b)
```

and compare with label-shuffled and same-length random topologies. This is not
required for a held-out geometry result, but it is required for the strongest
`HEAD_GREEN_GEOMETRY` classification.

## Stopping line

Only a training-specific held-out result on M_h earns scaling beyond the 1M model.
