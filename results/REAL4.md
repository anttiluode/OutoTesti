# REAL4 — the gauge-invariant attention operator is stable, but not yet tree-specific

REAL4 moved from W_Q / W_K separately to the actual head-wise pre-softmax score
operator:

```text
M_h = W_Q,h^T W_K,h / sqrt(head_dim)
score_h(x,y) = x M_h y^T
```

TinyStories-1M has 8 layers x 16 heads = 128 operators, each 64 x 64 and rank
at most the 4-dimensional head size.

## Result

### Strict spectrum-matched four-point geometry

```text
TRAINED median z                    +0.595
TRAINED empirical p<.05             20 / 128

RANDOM INIT median z                +0.010
RANDOM INIT empirical p<.05          2 / 128
```

The preregistered geometry criterion required median z >=1 and p<.05 on >=25%
of heads. It fails.

### Held-out column topology

Trees inferred from only half of each M_h's columns were frozen and evaluated
against row geometry from the other half.

```text
                                  TRAINED       RANDOM INIT

median gain vs leaf shuffle       +0.2436         +0.1235
fraction label gain positive       128/128         128/128

median gain vs random topology    +0.2183         +0.1452
fraction random gain positive      128/128         128/128

trained - init label gain         +0.1201
trained gain > init gain           116/128
```

This is a large and extremely consistent training effect. However the fact that
random initialization also has positive held-out gain in every head shows the
test contains a generic low-rank/geometric component.

### Green translation

```text
                                  TRAINED       RANDOM INIT

median Green error                 0.5928          0.8030
median label-shuffle gain         +0.00384        +0.00273
median random-topology gain       +0.00261        +0.01233
```

The stronger Green criterion fails. Training makes M_h much easier to
approximate in absolute terms, but the inferred labeled tree does not acquire a
robust Green-operator advantage.

Classification:

```text
NO_ROBUST_GAUGE_INVARIANT_TREE_GEOMETRY
```

## What remains interesting

The preregistered classification is negative, but one sub-result is too large to
ignore: training nearly doubles held-out leaf-topology stability and does so in
90.6% of paired heads.

The obvious remaining confound is **singular spectrum / low rank**. Each head
score matrix has rank <=4, and training can change how concentrated that spectrum
is. A tree inferred from half the columns may generalize simply because all
columns share the same low-dimensional subspace.

## Next stopping line

REAL5 must preserve every M_h's exact singular values while randomizing
orientation, then test held-out **quartet topology agreement** directly.

If trained heads do not exceed that exact-spectrum null, the REAL4 held-out
effect is a low-rank geometry effect rather than evidence for tree topology.
