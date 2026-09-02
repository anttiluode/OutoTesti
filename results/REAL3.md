# REAL3 — training creates Q/K tree-like structure, but the leaf labels do not transfer

REAL3 attacked REAL2's same-matrix circularity by transferring topology between
independently learned Q and K projections in each layer.

Because Q and K output coordinates are paired by the attention dot product, the
first test froze a tree inferred from Q and fit only leak/diagonal wrappers to K,
then reversed the direction.

## Trained model versus random initialization

Across the 16 Q<->K transfers:

```text
                              TRAINED        RANDOM INIT

median Q/K geometry z          +5.060          -0.237
geometry empirical p<.05       11 / 16          2 / 16

median gain vs random topology +0.04693        +0.00120
positive random-topology gain  16 / 16         13 / 16
beats all 4 random topologies  16 / 16          7 / 16
```

So training clearly creates something that is absent or vastly weaker at fresh
initialization. A tree inferred from one trained Q/K matrix carries an
**unlabeled structural advantage** when fitting its partner.

However the decisive labeled-topology attacker fails:

```text
median gain vs leaf-label shuffle

TRAINED                         -0.00036
RANDOM INIT                     -0.00001

trained positive label gain      6 / 16
trained beats all label shuffles 2 / 16
```

Classification:

```text
NO_ROBUST_QK_TOPOLOGY_TRANSFER
```

## Interpretation

REAL3 rejects the strong claim that Q and K share a reusable **labeled tree
circuit** over their current latent coordinates.

But it does not reduce REAL2 to random fitting. Three things remain real:

1. trained Q/K row geometry is much more additive-tree-like than a
   singular-spectrum-matched orientation null;
2. the effect is absent at random initialization;
3. the unlabeled inferred tree shape/length structure transfers between Q and K
   far better than a random topology with the same lengths.

The leaf-label failure says the useful signal is not tied to individual Q/K
latent coordinate identities in the simple way REAL3 assumed.

## Why the next object changes

Auditing W_Q and W_K separately is basis-dependent. What attention actually uses
for a head is the bilinear score operator

```text
M_h = W_Q,h^T W_K,h
```

because a token pair contributes `x M_h y^T` to that head's pre-softmax score.

`M_h` is invariant to the ordinary joint Q/K latent-coordinate gauge changes
that leave attention scores unchanged. It is therefore a much better target for
the question 'did this head learn graph-like routing geometry?'

## Stopping line

Do not scale. REAL4 must audit the 32 actual head-wise bilinear attention
operators, with spectrum-matched geometry nulls, same-length random trees,
leaf-label shuffles, and the random-initialized architecture.
