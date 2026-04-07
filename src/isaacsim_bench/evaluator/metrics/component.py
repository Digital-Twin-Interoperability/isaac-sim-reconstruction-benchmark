from __future__ import annotations

from typing import Literal

from isaacsim_bench.evaluator.matching import MatchResult, match_components
from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry


def compute_component_metrics(
    gt_scenes: list[SceneJSON],
    pred_scenes: list[PredictionJSON],
    mode: Literal["exact", "family"] = "exact",
    registry: TaxonomyRegistry | None = None,
) -> dict:
    """Component-level P/R/F1 over primary components."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    count_correct = 0
    total_scenes = 0

    all_matches: list[MatchResult] = []

    for gt_scene, pred_scene in zip(gt_scenes, pred_scenes):
        primary_gt = [c for c in gt_scene.components if c.evaluation_role == "primary"]
        pred_comps = pred_scene.components

        match = match_components(primary_gt, pred_comps, mode=mode, registry=registry)
        all_matches.append(match)

        tp = len(match.matched_pairs)
        fp = len(match.unmatched_pred)
        fn = len(match.unmatched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if len(primary_gt) == len(pred_comps) == tp:
            count_correct += 1
        total_scenes += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_count_accuracy": count_correct / total_scenes if total_scenes > 0 else 0.0,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "_matches": all_matches,
    }
