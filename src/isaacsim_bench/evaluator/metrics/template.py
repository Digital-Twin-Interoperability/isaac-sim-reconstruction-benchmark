from __future__ import annotations

from collections import Counter

from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.scene import SceneJSON


def compute_template_metrics(
    gt_scenes: list[SceneJSON],
    pred_scenes: list[PredictionJSON],
) -> dict:
    family_correct = 0
    template_correct = 0
    total = len(gt_scenes)

    # For macro F1
    family_labels: list[str] = []
    family_preds: list[str] = []
    template_labels: list[str] = []
    template_preds: list[str] = []

    for gt, pred in zip(gt_scenes, pred_scenes):
        family_labels.append(gt.family)
        family_preds.append(pred.predicted_family)
        template_labels.append(gt.template_id)
        template_preds.append(pred.predicted_template)

        if gt.family == pred.predicted_family:
            family_correct += 1
        if gt.template_id == pred.predicted_template:
            template_correct += 1

    return {
        "family_accuracy": family_correct / total if total > 0 else 0.0,
        "template_accuracy": template_correct / total if total > 0 else 0.0,
        "template_macro_f1": _macro_f1(template_labels, template_preds),
        "scene_count": total,
    }


def _macro_f1(labels: list[str], preds: list[str]) -> float:
    classes = set(labels) | set(preds)
    if not classes:
        return 0.0

    f1s = []
    for cls in classes:
        tp = sum(1 for l, p in zip(labels, preds) if l == cls and p == cls)
        fp = sum(1 for l, p in zip(labels, preds) if l != cls and p == cls)
        fn = sum(1 for l, p in zip(labels, preds) if l == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s) if f1s else 0.0
