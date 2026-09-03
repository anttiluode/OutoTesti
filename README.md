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

## REAL2 — inferred topology finally becomes measurable

The preregistered d=64 rescue used two harder tests.

First, row-channel geometry was compared with 64 null matrices per projection
that preserve the **exact singular spectrum** while randomizing singular-vector
orientation:

```text
median tree-likeness z          +1.119
empirical p<.05                 15 / 32
```

Second, the radial kernel was replaced by a true leaky tree Green operator:

```text
K = [(L_tree + leak I)^-1]_leaves
W_hat = diag(a) K diag(b)
```

Random topology controls received exactly the inferred tree's branch-length
multiset.

```text
median inferred Green error       0.91002
median random-topology error      0.92858
median topology gain             +0.02049
positive gain                    32 / 32
inferred beats all random trees  32 / 32
```

Both preregistered criteria pass.

The unexpected split is concentrated in Q/K:

```text
family   median geometry z   median Green topology gain

Q             +3.329                  +0.04235
K             +5.101                  +0.04939
V             +0.512                  +0.00427
OUT           +0.031                  +0.00636
```

This is **not yet hidden-tree evidence**: topology was inferred from the same
matrix it helped fit, and raw reconstruction remains poor.

See [REAL2](results/REAL2.md).

## REAL3 — the learned structure transfers unlabeled, not as a labeled circuit

Q-derived trees were frozen and used to fit K, and K-derived trees were used to
fit Q. The entire experiment was repeated in a fresh random-initialized copy of
the exact architecture.

```text
                               TRAINED       RANDOM INIT

median Q/K geometry z           +5.060          -0.237
median gain vs random topology  +0.04693        +0.00120
beats all random topologies     16 / 16          7 / 16
```

So training produces a large structural effect.

But permuting the leaf identities on the **same exact source tree** costs
essentially nothing:

```text
trained median label-shuffle gain   -0.00036
positive label gain                  6 / 16
```

Therefore REAL3 does **not** support a reusable labeled Q/K tree circuit.

See [REAL3](results/REAL3.md).

## REAL4 — gauge-invariant score operators: strong stability, strict tree test fails

The actual head score operator was audited:

```text
M_h = W_Q,h^T W_K,h / sqrt(head_dim)
```

There are 128 heads (8 layers x 16 heads), with head dimension 4.

```text
                                  TRAINED       RANDOM INIT

four-point median z                +0.595          +0.010
four-point p<.05                   20/128           2/128

held-out leaf-label gain           +0.2436         +0.1235
held-out random-topology gain      +0.2183         +0.1452
trained heldout gain > init         116/128

Green label-shuffle gain           +0.00384        +0.00273
Green random-topology gain         +0.00261        +0.01233
```

So training creates a very large held-out relational-geometry effect, but the
preregistered strict tree criterion and Green translation both fail.

See [REAL4](results/REAL4.md).

## REAL5 — low rank explains the apparent functional tree

The fitter-free quartet test preserved every head score operator's **exact
singular values** while randomizing only orientation.

```text
trained observed quartet agreement      0.9011
exact-spectrum null                     0.8997
median agreement gain                  -0.0027
median z                               -0.155
p < .05                                 0 / 128

random-init median gain                -0.0008
trained - init gain                    -0.0019
```

Classification:

`HELDOUT_TREE_EFFECT_EXPLAINED_BY_SPECTRAL_GEOMETRY`

So the large REAL4 effect was real low-dimensional/spectral structure, not
stable tree topology.

See [REAL5](results/REAL5.md).

## REAL6 — exact Q/K gauge weakens the signal, but does not erase it

Independent orthogonal rotations were applied inside every head to Q and K
together. Attention scores and model behavior were preserved:

```text
max head-score relative error      1.198e-15
relative logit error               8.435e-7
argmax agreement                   1.000
```

Yet the raw Q/K tree diagnostic only partially collapsed:

```text
baseline median z                  +3.803
random-gauge median z              +1.694
median absolute z shift             1.577
baseline significant               11 / 16
median gauge significant fraction   0.562
```

The preregistered gauge-kill criterion therefore fails.

See [REAL6](results/REAL6.md).

## Current stopping line

Individual Q/K coordinates are still non-identifiable. What survived the gauge
is each head's **4-D row subspace** in the 64-D residual-stream space.

REAL7 moves up one level:

> represent each of the 16 heads by its gauge-invariant row subspace; infer a
> 16-leaf hierarchy from Q head-subspace geometry and test whether it predicts K
> head-subspace geometry in the same layer, then reverse Q/K.

Use leaf-label shuffles, same-length random topologies, and a fresh random-init
model. If that fails, stop the tree chase completely.
