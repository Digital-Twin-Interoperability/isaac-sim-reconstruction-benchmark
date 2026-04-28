from __future__ import annotations

from isaacsim_bench.evaluator.matching import MatchResult
from isaacsim_bench.schemas.anchors import AnchorRegistry
from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.scene import SceneJSON


def _canonical_anchor(
    registry: AnchorRegistry | None,
    asset_id: str | None,
    anchor: str,
) -> str:
    """Resolve ``anchor`` (raw or alias) to its raw name via the registry.

    Returns the input unchanged when the registry has no entry for the
    asset or no record of the anchor.  This is a string-only fallback —
    so unresolvable anchors keep their original spelling and miscompare
    only when they should (no silent equivalence).
    """
    if registry is None or asset_id is None:
        return anchor
    rec = registry.get_record(asset_id)
    if rec is None:
        return anchor
    if anchor in rec.anchors:
        return anchor
    return rec.aliases.get(anchor, anchor)


def _relation_key(rel_type: str, from_node: str, to_node: str, from_anchor: str, to_anchor: str) -> tuple:
    return (rel_type, from_node, to_node, from_anchor, to_anchor)


def compute_relation_metrics(
    gt_scenes: list[SceneJSON],
    pred_scenes: list[PredictionJSON],
    component_matches: list[MatchResult],
    anchor_registry: AnchorRegistry | None = None,
) -> dict:
    """Cascaded relation scoring in strict and oracle modes.

    ``anchor_registry`` is used only for anchor-name canonicalization at
    scoring time.  Both GT and prediction relations may carry alias names
    (``out``, ``curve_entry``, ``right_end``); the registry resolves both
    sides to the same raw key (``origin`` / ``anchorpoint``) before the
    string comparison.  Without it, ``"out"`` vs ``"anchorpoint"`` would
    register as wrong even when they refer to the same authored anchor.
    """
    strict_tp = strict_fp = strict_fn = 0
    strict_anchor_correct = strict_anchor_total = 0
    oracle_tp = oracle_fp = oracle_fn = 0
    oracle_anchor_correct = oracle_anchor_total = 0

    for gt_scene, pred_scene, match in zip(gt_scenes, pred_scenes, component_matches):
        primary_gt = [c for c in gt_scene.components if c.evaluation_role == "primary"]

        # Build name mapping from matched component pairs
        gt_to_pred_name: dict[str, str] = {}
        for gt_idx, pred_idx in match.matched_pairs:
            gt_to_pred_name[primary_gt[gt_idx].name] = pred_scene.components[pred_idx].name

        gt_asset_by_name = {c.name: c.asset_id for c in gt_scene.components}
        pred_asset_by_name = {c.name: c.asset_id for c in pred_scene.components}

        def _gt_canonical(node: str, anchor: str) -> str:
            return _canonical_anchor(
                anchor_registry, gt_asset_by_name.get(node), anchor,
            )

        def _pred_canonical(node: str, anchor: str) -> str:
            return _canonical_anchor(
                anchor_registry, pred_asset_by_name.get(node), anchor,
            )

        # GT relations (only between primary components)
        primary_names = {c.name for c in primary_gt}
        gt_rels = [
            r for r in gt_scene.relations
            if r.from_node in primary_names and r.to_node in primary_names
        ]

        # Strict mode: relation TP only if both endpoints were matched
        pred_rel_set: dict[tuple, tuple] = {}
        for r in pred_scene.relations:
            key = (r.type, r.from_node, r.to_node)
            pred_rel_set[key] = (
                _pred_canonical(r.from_node, r.from_anchor),
                _pred_canonical(r.to_node, r.to_anchor),
            )

        for gt_rel in gt_rels:
            from_mapped = gt_to_pred_name.get(gt_rel.from_node)
            to_mapped = gt_to_pred_name.get(gt_rel.to_node)

            if from_mapped is None or to_mapped is None:
                strict_fn += 1
                continue

            key = (gt_rel.type, from_mapped, to_mapped)
            if key in pred_rel_set:
                strict_tp += 1
                strict_anchor_total += 1
                pred_anchors = pred_rel_set.pop(key)
                gt_anchors = (
                    _gt_canonical(gt_rel.from_node, gt_rel.from_anchor),
                    _gt_canonical(gt_rel.to_node, gt_rel.to_anchor),
                )
                if pred_anchors == gt_anchors:
                    strict_anchor_correct += 1
            else:
                strict_fn += 1

        strict_fp += len(pred_rel_set)

        # Oracle mode: assume perfect matching (GT names == pred names)
        pred_rel_set_oracle: dict[tuple, tuple] = {}
        for r in pred_scene.relations:
            key = (r.type, r.from_node, r.to_node)
            pred_rel_set_oracle[key] = (
                _pred_canonical(r.from_node, r.from_anchor),
                _pred_canonical(r.to_node, r.to_anchor),
            )

        for gt_rel in gt_rels:
            key = (gt_rel.type, gt_rel.from_node, gt_rel.to_node)
            if key in pred_rel_set_oracle:
                oracle_tp += 1
                oracle_anchor_total += 1
                pred_anchors = pred_rel_set_oracle.pop(key)
                gt_anchors = (
                    _gt_canonical(gt_rel.from_node, gt_rel.from_anchor),
                    _gt_canonical(gt_rel.to_node, gt_rel.to_anchor),
                )
                if pred_anchors == gt_anchors:
                    oracle_anchor_correct += 1
            else:
                oracle_fn += 1

        oracle_fp += len(pred_rel_set_oracle)

    def _prf1(tp: int, fp: int, fn: int) -> dict:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1}

    return {
        "strict": {
            **_prf1(strict_tp, strict_fp, strict_fn),
            "anchor_accuracy": (
                strict_anchor_correct / strict_anchor_total
                if strict_anchor_total > 0
                else 0.0
            ),
        },
        "oracle_components": {
            **_prf1(oracle_tp, oracle_fp, oracle_fn),
            "anchor_accuracy": (
                oracle_anchor_correct / oracle_anchor_total
                if oracle_anchor_total > 0
                else 0.0
            ),
        },
    }
