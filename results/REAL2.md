# REAL2 — topology starts to matter, but only after using the right object

REAL1 was negative: the radial tree kernel was essentially topology-insensitive.
REAL2 stayed at d=64 and ran the two preregistered rescue tests.

## A. Spectrum-matched channel-geometry null

For each trained matrix, row-channel cosine geometry was compared against 64
null matrices with the **exact same singular values** but independently
randomized left/right singular-vector orientations.

Lower four-point gap means more additive-tree-like.

```text
median tree-likeness z              +1.119
mean tree-likeness z                +2.173
empirical p < .05                   15 / 32
```

The preregistered aggregate criterion was median z >= 1 and at least 8/32
p<.05. It passes.

The signal is not spread evenly across projection families:

```text
family     median geometry z     p<.05

Q              +3.329            6 / 8
K              +5.101            6 / 8
V              +0.512            2 / 8
OUT            +0.031            1 / 8
```

That Q/K versus V/OUT split was not part of the classification rule and should
be treated as an observed pattern requiring a new attacker, not as a finished
interpretation.

## B. True tree Green / resolvent operator

Instead of `exp(-alpha d_tree)`, REAL2 compiled the neighbor-joining tree into
a leaky weighted graph Green operator:

```text
g_edge ~ 1 / branch_length
L      = weighted tree Laplacian
K      = [(L + leak I)^-1]_leaves
W_hat  = diag(a) K diag(b)
```

Every random-topology attacker received **exactly the same branch-length
multiset** as the inferred tree. Only topology was randomized.

```text
median inferred Green error          0.91002
median random-topology error         0.92858
median topology gain                +0.02049
positive topology gain              32 / 32
gain > .005                         23 / 32
inferred beats all 4 random trees   32 / 32
```

The preregistered criterion was median gain >= .005 and positive on >=75% of
matrices. It passes comfortably.

Again, the family split is striking:

```text
family    median Green error    median topology gain

Q             0.82264                 +0.04235
K             0.80142                 +0.04939
V             0.95632                 +0.00427
OUT           0.95317                 +0.00636
```

Classification:

```text
CHANNEL_GEOMETRY_AND_GREEN_TOPOLOGY_SIGNAL
```

## What this does and does not say

This reverses only the **topology-insensitive** part of REAL1. The learned
topology now matters measurably when the tree is compiled as an actual network
resolvent rather than a radial distance kernel.

It does **not** mean a transformer projection literally is a tree cable. Raw
matrix reconstruction is still poor (about 0.91 median relative Frobenius
error), and the topology is inferred from the same matrix it is later used to
fit. That same-matrix circularity is the next thing to attack.

The most interesting unplanned observation is that Q/K carry most of both the
geometry and Green-topology signal, while V/output largely do not.

## Next stopping line

Do not scale yet.

REAL3 must test whether topology **transfers out of sample**. At minimum:

1. infer topology from Q and test it on K in the same layer, and vice versa;
2. compare with random topologies carrying the same branch-length multiset;
3. compare trained weights with a fresh random-initialized model of the exact
   same architecture.

If Q/K topology cannot transfer or distinguish training from initialization,
REAL2 is best understood as an in-sample geometry-fitting effect rather than
evidence that training discovered reusable graph structure.
