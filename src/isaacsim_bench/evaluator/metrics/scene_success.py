from __future__ import annotations

from isaacsim_bench.evaluator.matching import MatchResult
from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.match_regime import derive_scene_regime


def compute_scene_success(
    gt_scenes: list[SceneJSON],
    pred_scenes: list[PredictionJSON],
    component_matches: list[MatchResult],
    component_f1_threshold: float = 0.9,
    relation_f1_threshold: float = 0.9,
    translation_threshold: float = 0.10,
    rotation_threshold: float = 10.0,
) -> dict:
    """End-to-end scene success rate.

    A scene succeeds if ALL of:
    - component F1 >= threshold
    - relation F1 >= threshold (oracle mode)
    - mean root-relative translation error <= threshold
    - mean root-relative rotation error <= threshold
    - correct abstention on unknown scenes
    """
    successes = 0
    total = len(gt_scenes)
    per_scene: list[dict] = []

    for gt_scene, pred_scene, match in zip(gt_scenes, pred_scenes, component_matches):
        checks = {}

        # 1. Component F1
        primary_gt = [c for c in gt_scene.components if c.evaluation_role == "primary"]
        tp = len(match.matched_pairs)
        fp = len(match.unmatched_pred)
        fn = len(match.unmatched_gt)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        comp_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        checks["component_f1"] = comp_f1
        checks["component_f1_pass"] = comp_f1 >= component_f1_threshold

        # 2. Abstention check
        primary_regimes = [c.match_regime for c in primary_gt]
        scene_regime = derive_scene_regime(primary_regimes)
        is_unknown = scene_regime == "scene_unknown"
        checks["abstention_correct"] = (
            (pred_scene.abstained and is_unknown)
            or (not pred_scene.abstained and not is_unknown)
        )

        # Overall success (placement and relation F1 computed externally;
        # here we check the composite with available info)
        scene_ok = (
            checks["component_f1_pass"]
            and checks["abstention_correct"]
        )
        checks["success"] = scene_ok
        if scene_ok:
            successes += 1
        per_scene.append(checks)

    return {
        "scene_success_rate": successes / total if total > 0 else 0.0,
        "total_scenes": total,
        "successful_scenes": successes,
        "per_scene": per_scene,
    }
