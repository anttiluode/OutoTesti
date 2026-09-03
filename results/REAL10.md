# REAL10 — weight-space head geometry predicts actual attention behavior

REAL9 killed the tree-specific interpretation but left a replicated, gauge-safe
Q/K attention-head subspace geometry. REAL10 asked whether that weight-only
geometry has behavioral meaning.

Two separately trained models were tested:

- `roneneldan/TinyStories-1M`
- `roneneldan/TinyStories-Instruct-1M`

For each layer:

```text
D_weight = (D_Q + D_K) / 2
```

where D_Q and D_K are the 16-head Grassmann chordal-distance matrices from
the gauge-invariant Q/K row subspaces.

Behavior was measured from the actual per-head causal attention-probability maps
on 24 fixed story-like texts.

## Reliability gate

Behavioral head-distance matrices were independently estimated from two halves
of the text corpus.

```text
pooled trained median split-half reliability   +0.963
```

All trained model-layers comfortably exceeded the preregistered reliability
gate.

## Weight geometry predicts behavioral geometry

Across 16 trained model-layers:

```text
median weight-behavior correlation       +0.385
positive correlation                      15 / 16
head-label permutation p<.05              13 / 16
```

Random initialization:

```text
median weight-behavior correlation       -0.003
```

Training specificity:

```text
trained - init median correlation        +0.387
trained correlation > init               15 / 16
```

Classification:

```text
HEAD_SUBSPACE_GEOMETRY_PREDICTS_ATTENTION_BEHAVIOR
```

## Per-model pattern

`TinyStories-1M` trained layers:

```text
layer   reliability   weight->behavior r
0          .962             -.222
1          .989             +.243
2          .981             +.482
3          .987             +.298
4          .945             +.183
5          .975             +.255
6          .964             +.442
7          .907             +.407
```

`TinyStories-Instruct-1M` trained layers:

```text
layer   reliability   weight->behavior r
0          .985             +.429
1          .956             +.557
2          .984             +.234
3          .994             +.511
4          .913             +.326
5          .950             +.497
6          .962             +.362
7          .961             +.539
```

The single strong negative is layer 0 of TinyStories-1M. The result is not a
claim that every layer obeys the relationship.

## What survived the entire chase

Not a hidden tree.

Not literal graph-derived transformer weights.

The surviving object is:

> **the relative geometry of the residual-stream subspaces read by attention
> heads, computed from Q/K weights, is gauge invariant and predicts similarity
> of their observed attention behavior.**

This survived:

- raw-weight tree reconstruction attackers;
- random topology;
- low-rank / exact-spectrum controls;
- exact function-preserving Q/K gauge rotations;
- random initialization;
- replication on a separately trained checkpoint;
- a tree-specificity attack;
- behavioral split-half reliability;
- behavioral head-label permutation.

## Product extraction

The research branch stops here.

Extract a small `head-geometry` command that can:

1. load a compatible Hugging Face attention model;
2. compute gauge-invariant Q/K head-subspace geometry from weights alone;
3. report Q/K alignment by layer;
4. optionally run user-supplied texts and test whether weight geometry predicts
   observed head attention behavior;
5. emit JSON/CSV without any tree fitting.

That tool should retain the scientific limits: it measures relational head
geometry, not semantics, causality, importance, or a hidden graph.
