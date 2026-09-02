# OutoTesti

**Weird test.**

This repo asks one narrow question born from the `Operaattori` / `OperaattoriJako` chase:

> **When an ordinary trained model learns a square weight matrix, how much of that matrix can be explained post hoc by a compact tree-generated operator?**

The model is **not** trained with a tree constraint. We inspect it only after training.

This is an instrument, not a claim that transformer weights literally are dendritic cables.

## Why this exists

`Operaattori` ended up with a useful separation:

> structure compiles transport; local state supplies nonlinear computation.

That suggests a black-box diagnostic. Given a trained matrix `W`, try to project it onto a graph-generated family:

```text
trained W
   |
   +--> row/channel geometry
   |         |
   |         +--> distance matrix
   |         +--> four-point violation
   |         +--> neighbor-joining topology
   |
   +--> inferred weighted tree
             |
             +--> radial tree kernel K
             |
             +--> diag(a) K diag(b)
                       |
                       +--> W_tree
```

Then compare the residual with generic compression at a similar nominal parameter budget.

## Important correction to the original idea

We do **not** symmetrize a signed transformer matrix and pretend it is a tree metric.

Raw learned weights are signed and directional. Instead, v0.1:

1. treats each row as a learned channel signature;
2. builds a symmetric cosine-derived distance between row signatures;
3. measures the additive-tree four-point condition on that diagnostic distance;
4. infers a topology with neighbor joining;
5. compiles a positive radial kernel from the inferred additive tree;
6. wraps it in signed input/output diagonal gains: `W_tree = diag(a) K_tree diag(b)`.

The residual is called **non-graph residual**, not "computation". That stronger interpretation would require functional evidence.

## Attackers first

Every audited matrix gets four comparisons:

- inferred tree-generated operator;
- random tree topology;
- matched-nominal-budget truncated SVD;
- matched-nominal-budget top-|weight| sparse matrix.

The tree's reported parameter budget counts branch lengths, the two diagonal gain vectors, and one kernel scale. **It does not count the discrete topology encoding cost.** Therefore this is not yet a formal compression-bit-rate benchmark.

## Sanity gate

Before touching a real model:

```bash
python -m pip install -e .[dev]
pytest -q
python experiments/sanity.py
```

The sanity test requires:

- an exact additive tree metric to satisfy the four-point condition;
- a matrix generated from a known tree kernel to be recovered when given its true topology;
- a random dense matrix to produce finite attacker results.

The inferred topology is deliberately a separate problem: row geometry is not guaranteed to reveal the generating tree even when the matrix itself was made from one.

## Real Test 1 — TinyStories-1M

TinyStories-1M is a useful first real victim because its square attention projection matrices are only 64 x 64.

Install the optional model dependencies:

```bash
python -m pip install -e .[models]
```

Run the offline matrix audit:

```bash
python experiments/audit_tinystories.py --model roneneldan/TinyStories-1M --device cpu
```

On an RTX 3060:

```bash
python experiments/audit_tinystories.py --model roneneldan/TinyStories-1M --device cuda --behavior
```

`--behavior` goes beyond matrix reconstruction. For each individual Q/K/V/output projection it temporarily puts the tree, SVD, and sparse approximations **back into the real model**, runs the same text, and records:

- mean logit KL divergence;
- next-token argmax agreement;
- relative logit error.

The original parameter is restored after each intervention.

Outputs:

```text
results/tinystories_1m/matrix_audit.csv
results/tinystories_1m/audit.json
```

## What would be interesting?

Not merely "the inferred tree looks pretty."

A useful result would look like one of these:

```text
A. tree ~= SVD in matrix error, but damages behavior much less
B. some layer families repeatedly prefer tree over random-tree/sparse controls
C. training produces matrices substantially closer to the tree manifold than matched random initialization
D. nothing is tree-like; SVD wins everywhere
```

D is a perfectly good result. It says the graph-generated family is the wrong instrument for those learned matrices.

## First real receipt — TinyStories-1M says **not yet**

The full 32-matrix Q/K/V/output audit has now run on the actually trained
TinyStories-1M model, including one-at-a-time replacement back into the model.

```text
median relative matrix error

inferred tree            0.90993
random tree              0.91027
matched-budget SVD       0.91029
top-|weight| sparse      0.78884

median |tree-random|     0.00010
median |tree-SVD|        0.00007
```

The inferred topology is therefore doing essentially no work in v0.1. The
tree fit behaves almost like the rank-1 SVD attacker, while sparse weights
usually reconstruct the matrix substantially better.

The functional replacement experiment is also mixed rather than rescuing the
tree:

```text
median mean-logit KL

tree                     0.4404
SVD                      0.4432
sparse                   0.3601

median next-token argmax agreement
tree / SVD / sparse      0.6818 / 0.6818 / 0.6818
```

Individual matrices do show cases where Frobenius error and behavioral damage
disagree. That is worth keeping as a diagnostic. It is **not** evidence for
hidden tree geometry.

See [the full receipt](results/REAL1.md) and the
[32-matrix table](results/tinystories_1m/matrix_audit.csv).

## Revised ladder

```text
SANITY
known tree / random dense                    PASS

REAL 1
TinyStories-1M, radial tree kernel           NEGATIVE
                                             topology adds ~nothing

NEXT
stay at d=64:
  A. test tree-likeness of induced channel geometry against
     spectrum/dimension-matched random controls
  B. test a genuinely graph-resolvent / Green-operator family

ONLY IF one becomes topology-sensitive:
TinyStories-8M -> DistilGPT2
```

Do not scale merely because the code works.

## Current stopping line

The quick story

> "trained transformer matrix -> neighbor-joining tree -> circuit diagram"

is **not supported** by the first real model.

The v0.1 family
`diag(a) exp(-alpha d_tree) diag(b)` is too restrictive and is effectively
topology-insensitive on TinyStories-1M.

The repo earns a next gate only if a new test can distinguish **inferred
topology from random topology** at d=64. Until then, call the residual
"non-graph residual", not computation, routing, or hidden geometry.
