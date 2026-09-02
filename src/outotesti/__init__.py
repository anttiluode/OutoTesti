"""OutoTesti: project trained matrices onto tree-generated operator families."""

from .audit import audit_matrix
from .metrics import channel_distance, four_point_score
from .projection import fit_tree_kernel, matched_budget_svd, matched_budget_sparse

__all__ = [
    "audit_matrix",
    "channel_distance",
    "four_point_score",
    "fit_tree_kernel",
    "matched_budget_svd",
    "matched_budget_sparse",
]
