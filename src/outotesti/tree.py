from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Tree:
    n_leaves: int
    edges: tuple[tuple[int, int, float], ...]

    @property
    def n_nodes(self) -> int:
        if not self.edges:
            return self.n_leaves
        return 1 + max(max(u, v) for u, v, _ in self.edges)


def neighbor_joining(D: np.ndarray) -> Tree:
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("D must be square")
    if n < 2:
        return Tree(n, ())

    active = list(range(n))
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = float(max(D[i, j], 0.0))

    def get(a: int, b: int) -> float:
        if a == b:
            return 0.0
        return dist[(min(a, b), max(a, b))]

    edges = []
    next_node = n
    while len(active) > 2:
        m = len(active)
        row_sum = {i: sum(get(i, j) for j in active if j != i) for i in active}
        best = None
        pair = None
        for ai, i in enumerate(active[:-1]):
            for j in active[ai + 1:]:
                q = (m - 2) * get(i, j) - row_sum[i] - row_sum[j]
                key = (q, i, j)
                if best is None or key < best:
                    best = key
                    pair = (i, j)

        i, j = pair
        dij = get(i, j)
        delta = (row_sum[i] - row_sum[j]) / (m - 2)
        li = max(0.0, 0.5 * (dij + delta))
        lj = max(0.0, dij - li)

        u = next_node
        next_node += 1
        edges.append((u, i, li))
        edges.append((u, j, lj))

        others = [k for k in active if k not in (i, j)]
        for k in others:
            duk = max(0.0, 0.5 * (get(i, k) + get(j, k) - dij))
            dist[(min(u, k), max(u, k))] = duk
        active = others + [u]

    a, b = active
    edges.append((a, b, max(0.0, get(a, b))))
    return Tree(n, tuple(edges))


def leaf_distance_matrix(tree: Tree) -> np.ndarray:
    adj = [[] for _ in range(tree.n_nodes)]
    for u, v, w in tree.edges:
        w = float(max(w, 0.0))
        adj[u].append((v, w))
        adj[v].append((u, w))

    D = np.zeros((tree.n_leaves, tree.n_leaves), dtype=float)
    for source in range(tree.n_leaves):
        stack = [(source, -1, 0.0)]
        while stack:
            node, parent, d = stack.pop()
            if node < tree.n_leaves:
                D[source, node] = d
            for nxt, w in adj[node]:
                if nxt != parent:
                    stack.append((nxt, node, d + w))
    return D


def random_binary_tree(n_leaves: int, rng: np.random.Generator) -> Tree:
    if n_leaves < 2:
        return Tree(n_leaves, ())
    active = list(range(n_leaves))
    edges = []
    next_node = n_leaves
    while len(active) > 2:
        pick = rng.choice(len(active), size=2, replace=False)
        ia, ib = sorted(pick.tolist(), reverse=True)
        a = active.pop(ia)
        b = active.pop(ib)
        u = next_node
        next_node += 1
        edges.append((u, a, float(rng.uniform(0.2, 1.0))))
        edges.append((u, b, float(rng.uniform(0.2, 1.0))))
        active.append(u)
    a, b = active
    edges.append((a, b, float(rng.uniform(0.2, 1.0))))
    return Tree(n_leaves, tuple(edges))
