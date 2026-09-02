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

## Planned ladder

```text
SANITY
known tree / random dense

REAL 1
TinyStories-1M        d=64

REAL 2
TinyStories-8M        d=256

REAL 3
DistilGPT2            d=768
```

Do not scale until the 64-dimensional test tells us whether the diagnostic has any signal.

## Stopping line

A small tree residual by itself does **not** mean a learned layer "is routing" or "learned geometry."

The first claim this repo can earn is much narrower:

> a particular trained matrix is unusually well approximated by this graph-generated operator family relative to explicit low-rank, sparse, and random-topology attackers, and replacing the matrix by that approximation causes measured model damage.

Only after that would it be worth interpreting the inferred topology.
