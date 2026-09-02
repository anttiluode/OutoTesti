from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .audit import audit_matrix, jsonable_audit


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit a square trained matrix against a tree-generated operator family.")
    ap.add_argument("matrix", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outotesti_audit.json"))
    ap.add_argument("--random-trees", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    W = np.load(args.matrix)
    result = audit_matrix(W, random_tree_controls=args.random_trees, seed=args.seed)
    payload = jsonable_audit(result)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("OutoTesti — trained matrix -> tree-generated operator")
    print(f"shape:                 {tuple(W.shape)}")
    print(f"four-point p95 gap:    {payload['channel_four_point']['p95_gap']:.4f}")
    print(f"tree error:            {payload['tree']['error']:.4f}")
    print(f"random-tree median:    {payload['random_tree']['median_error']:.4f}")
    print(f"SVD error:             {payload['svd']['error']:.4f} (rank {payload['svd']['rank']})")
    print(f"sparse error:          {payload['sparse']['error']:.4f}")
    print(f"tree nominal budget:   {payload['tree']['parameter_budget']}")
