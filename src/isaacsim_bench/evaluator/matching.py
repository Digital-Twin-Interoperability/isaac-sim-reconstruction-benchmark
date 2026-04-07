from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from isaacsim_bench.schemas.prediction import PredictedComponent
from isaacsim_bench.schemas.scene import ComponentEntry
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry


@dataclass
class MatchResult:
    matched_pairs: list[tuple[int, int]] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)


def match_components(
    gt_components: list[ComponentEntry],
    pred_components: list[PredictedComponent],
    mode: Literal["exact", "family"] = "exact",
    registry: TaxonomyRegistry | None = None,
) -> MatchResult:
    """Bipartite matching between GT and predicted components.

    mode="exact": match only if asset_id is identical.
    mode="family": match if same category (requires registry).
    """
    n_gt = len(gt_components)
    n_pred = len(pred_components)

    if n_gt == 0 and n_pred == 0:
        return MatchResult()
    if n_gt == 0:
        return MatchResult(unmatched_pred=list(range(n_pred)))
    if n_pred == 0:
        return MatchResult(unmatched_gt=list(range(n_gt)))

    # Build cost matrix: 0 = match, 1 = mismatch
    cost = np.ones((n_gt, n_pred), dtype=float)
    for i, gt in enumerate(gt_components):
        for j, pred in enumerate(pred_components):
            if mode == "exact":
                if gt.asset_id == pred.asset_id:
                    cost[i, j] = 0.0
            elif mode == "family":
                if registry is not None:
                    gt_cat = registry.get_category(gt.asset_id)
                    pred_cat = registry.get_category(pred.asset_id)
                    if gt_cat is not None and gt_cat == pred_cat:
                        cost[i, j] = 0.0
                # Fallback: compare family strings directly
                elif gt.family == pred.family:
                    cost[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs = []
    unmatched_gt = set(range(n_gt))
    unmatched_pred = set(range(n_pred))

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 0.5:  # matched
            matched_pairs.append((r, c))
            unmatched_gt.discard(r)
            unmatched_pred.discard(c)

    return MatchResult(
        matched_pairs=matched_pairs,
        unmatched_gt=sorted(unmatched_gt),
        unmatched_pred=sorted(unmatched_pred),
    )
