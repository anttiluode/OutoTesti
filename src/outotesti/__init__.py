"""OutoTesti: project trained matrices onto tree-generated operator families."""

from .audit import audit_matrix
from .geometry import geometry_null_audit
from .green import fit_green_operator
from .metrics import channel_distance, four_point_score
from .projection import fit_tree_kernel, matched_budget_svd, matched_budget_sparse

__all__ = [
    "audit_matrix",
    "geometry_null_audit",
    "fit_green_operator",
    "channel_distance",
    "four_point_score",
    "fit_tree_kernel",
    "matched_budget_svd",
    "matched_budget_sparse",
]
