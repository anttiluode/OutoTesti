# REAL9 — the replicated head-subspace geometry is not tree-specific

REAL7 and REAL8 replicated a gauge-invariant Q/K head-subspace relationship.
REAL9 attacked whether a tree is a particularly good compressed description of
that relationship.

Every representation was constructed from the source metric only. The target
received one global scale parameter.

## Pooled trained result across both models

32 Q<->K transfers total:

```text
raw source metric error          0.0338
star error                       0.0349
neighbor-joining tree error      0.0327
2-D classical MDS error          0.4349

tree vs MDS2 median gain        +0.3921
tree beats MDS2                  32 / 32

tree vs star median gain        +0.0014
```

The 2-D MDS attacker is a poor representation for these head-subspace metrics.
That is not a training effect:

```text
tree vs MDS2 median gain

TRAINED                         +0.3921
RANDOM INIT                     +0.3997

trained - init                 -0.0076
paired trained advantage better 15 / 32
```

The star is the more informative attacker. With only 16 leaf radii it performs
almost as well as the 29-edge neighbor-joining tree on the trained targets.

Classification:

```text
SHARED_HEAD_GEOMETRY_NOT_TREE_SPECIFIC
```

## Interpretation

The replicated training-specific result from REAL7/8 survives, but the tree
language does not.

What is supported is:

> training co-organizes the 16 Q and K attention-head input subspaces into a
> shared, gauge-invariant relational geometry.

The geometry is visible directly as Q/K head-distance correlation and as labeled
cross-transfer. It does not need a tree interpretation.

The near-star quality also explains why random topologies and low-dimensional
tree-like summaries behaved oddly in earlier gates: random head subspaces in a
high-dimensional ambient space are already close to equidistant.

## What died

- hidden 64-leaf tree circuit: killed by gauge/functional tests;
- tree topology inside the gauge-invariant head score operator: killed by the
  exact-spectrum quartet null;
- tree as a uniquely useful compression of head-subspace geometry: killed here.

## What remains useful

The head-subspace geometry itself is:

- gauge invariant;
- cheap to compute from trained weights;
- replicated on two separately trained 1M checkpoints;
- absent at random initialization;
- aligned between Q and K by the actual head labels.

That is now a plausible interpretability diagnostic.

## Next stopping line

Stop fitting graphs.

REAL10 asks whether pairwise distances in this weight-space head geometry predict
pairwise **behavioral similarity of the same heads on actual text**.

If they do, extract a small reusable `head-geometry` audit tool.

If they do not, keep REAL7/8 as a parameter-organization observation and stop.
