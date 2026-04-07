from __future__ import annotations

from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry


def _recall_at_k(hits: list[bool], k: int) -> float:
    """Fraction of queries where the correct answer appeared in top-k."""
    if not hits:
        return 0.0
    # For single-retrieval, each component is one query with one candidate
    return sum(hits[:k]) / len(hits) if hits else 0.0


def compute_retrieval_metrics(
    gt_scenes: list[SceneJSON],
    pred_scenes: list[PredictionJSON],
    registry: TaxonomyRegistry,
) -> dict:
    """Per-component retrieval metrics, split by match regime.

    Since each predicted component corresponds to a single retrieval candidate
    (not a ranked list), Recall@1 == accuracy and Recall@5 == Recall@1.
    MRR == Recall@1 in this single-candidate setting.
    """
    exact_hits: list[bool] = []
    proxy_hits: list[bool] = []

    for gt_scene, pred_scene in zip(gt_scenes, pred_scenes):
        pred_by_name = {c.name: c for c in pred_scene.components}
        for gt_comp in gt_scene.components:
            if gt_comp.evaluation_role != "primary":
                continue
            pred_comp = pred_by_name.get(gt_comp.name)
            if pred_comp is None:
                if gt_comp.match_regime == "exact_match":
                    exact_hits.append(False)
                elif gt_comp.match_regime == "proxy_match":
                    proxy_hits.append(False)
                continue

            if gt_comp.match_regime == "exact_match":
                exact_hits.append(pred_comp.asset_id == gt_comp.asset_id)
            elif gt_comp.match_regime == "proxy_match":
                gt_cat = registry.get_category(gt_comp.asset_id)
                pred_cat = registry.get_category(pred_comp.asset_id)
                proxy_hits.append(gt_cat is not None and gt_cat == pred_cat)

    def safe_mean(vals: list[bool]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "exact_recall_at_1": safe_mean(exact_hits),
        "exact_recall_at_5": safe_mean(exact_hits),  # same in single-candidate
        "exact_count": len(exact_hits),
        "proxy_family_recall_at_1": safe_mean(proxy_hits),
        "proxy_family_recall_at_5": safe_mean(proxy_hits),
        "proxy_count": len(proxy_hits),
        "mrr": safe_mean(exact_hits),  # single-candidate
    }
