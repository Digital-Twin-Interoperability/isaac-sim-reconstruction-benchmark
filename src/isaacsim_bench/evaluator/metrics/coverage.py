from __future__ import annotations

from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.match_regime import derive_scene_regime


def _prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_coverage_metrics(
    gt_scenes: list[SceneJSON],
    pred_scenes: list[PredictionJSON],
) -> dict:
    # Component-level unknown abstention
    comp_tp = comp_fp = comp_fn = 0
    false_accept = 0
    unknown_total = 0

    for gt_scene, pred_scene in zip(gt_scenes, pred_scenes):
        pred_by_name = {c.name: c for c in pred_scene.components}
        for gt_comp in gt_scene.components:
            if gt_comp.evaluation_role != "primary":
                continue
            if gt_comp.match_regime == "unknown":
                unknown_total += 1
                pred_comp = pred_by_name.get(gt_comp.name)
                if pred_comp is None or pred_scene.abstained:
                    comp_tp += 1  # correctly abstained
                else:
                    comp_fn += 1  # should have abstained but didn't
                    false_accept += 1
            else:
                pred_comp = pred_by_name.get(gt_comp.name)
                if pred_comp is None or pred_scene.abstained:
                    comp_fp += 1  # abstained when shouldn't have

    # Scene-level unknown abstention
    scene_tp = scene_fp = scene_fn = 0
    for gt_scene, pred_scene in zip(gt_scenes, pred_scenes):
        primary_regimes = [
            c.match_regime
            for c in gt_scene.components
            if c.evaluation_role == "primary"
        ]
        scene_regime = derive_scene_regime(primary_regimes)
        is_unknown = scene_regime == "scene_unknown"

        if is_unknown:
            if pred_scene.abstained:
                scene_tp += 1
            else:
                scene_fn += 1
        else:
            if pred_scene.abstained:
                scene_fp += 1

    false_accept_rate = false_accept / unknown_total if unknown_total > 0 else 0.0

    return {
        "component_level": _prf1(comp_tp, comp_fp, comp_fn),
        "scene_level": _prf1(scene_tp, scene_fp, scene_fn),
        "false_accept_rate": false_accept_rate,
        "unknown_component_count": unknown_total,
    }
