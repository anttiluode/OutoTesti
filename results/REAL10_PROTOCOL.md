# REAL10 protocol — does weight-space head geometry predict head behavior?

REAL7/8 replicated a gauge-invariant Q/K head-subspace relationship. REAL9
killed the tree-specific interpretation.

REAL10 asks whether the surviving weight-space geometry is functionally useful.

## Weight geometry

For each layer, compute Q and K 16-head Grassmann chordal distance matrices and
use their mean:

```text
D_weight = (D_Q + D_K) / 2
```

This is invariant to the exact within-head Q/K gauge attacked in REAL6.

## Behavioral geometry

Run the actual model on 24 fixed story-like texts. For every layer and head:

- collect the causal attention-probability map;
- flatten valid token-pair entries across the corpus;
- center each head vector;
- compute pairwise centered-cosine distance between heads.

This gives `D_behavior`, a 16 x 16 matrix for each layer.

## Reliability gate

Split the corpus into two 12-text halves and independently construct
`D_behavior_A` and `D_behavior_B`.

Require:

```text
median split-half distance correlation >= .50
reliability > .30 in                >=75% of 16 trained model-layers
```

If the behavioral metric itself is not reliable, no interpretability claim is
allowed.

## Primary prediction test

For every layer of both trained models, correlate the 120 upper-triangle entries
of `D_weight` with `D_behavior`.

Use 512 head-label permutations per layer.

Pass requires pooled across 16 trained model-layers:

```text
median weight-behavior correlation >= .20
positive correlation               >=75%
permutation p<.05                   >=8 / 16 layers
```

## Training control

Repeat on fresh random-initialized copies using the same text.

Training-specificity requires:

```text
trained - init median correlation >= .15
trained correlation > init         >=75% paired model-layers
```

## Stopping line

If all three gates pass, extract the surviving mechanism as a small reusable
`head-geometry` interpretability audit.

If not, stop: the replicated geometry is a parameter-organization observation
without demonstrated behavioral meaning.
