from __future__ import annotations

import math

import numpy as np

from isaacsim_bench.evaluator.matching import MatchResult
from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.scene import SceneJSON


def _quat_angular_distance_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angular distance (degrees) between two xyzw quaternions."""
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _root_relative_pose(
    components: list, root_name: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Express each component pose relative to root node.

    Returns dict of name -> (relative_translate, orientation_xyzw).
    """
    root_translate = None
    root_quat = None
    for c in components:
        if c.name == root_name:
            root_translate = np.array(c.translate)
            root_quat = np.array(c.orientation_xyzw)
            break

    if root_translate is None:
        return {}

    result = {}
    for c in components:
        rel_t = np.array(c.translate) - root_translate
        quat = np.array(c.orientation_xyzw)
        result[c.name] = (rel_t, quat)
    return result


def compute_placement_metrics(
    gt_scenes: list[SceneJSON],
    pred_scenes: list[PredictionJSON],
    component_matches: list[MatchResult],
    translation_threshold: float = 0.10,
    rotation_threshold: float = 10.0,
) -> dict:
    """Root-relative placement error for matched primary components."""
    all_trans_errors: list[float] = []
    all_rot_errors: list[float] = []

    for gt_scene, pred_scene, match in zip(gt_scenes, pred_scenes, component_matches):
        primary_gt = [c for c in gt_scene.components if c.evaluation_role == "primary"]

        gt_poses = _root_relative_pose(primary_gt, gt_scene.root_node)
        pred_poses = _root_relative_pose(pred_scene.components, gt_scene.root_node)

        for gt_idx, pred_idx in match.matched_pairs:
            gt_name = primary_gt[gt_idx].name
            pred_name = pred_scene.components[pred_idx].name

            gt_pose = gt_poses.get(gt_name)
            pred_pose = pred_poses.get(pred_name)

            if gt_pose is None or pred_pose is None:
                continue

            trans_err = float(np.linalg.norm(gt_pose[0] - pred_pose[0]))
            rot_err = _quat_angular_distance_deg(gt_pose[1], pred_pose[1])

            all_trans_errors.append(trans_err)
            all_rot_errors.append(rot_err)

    if not all_trans_errors:
        return {
            "mean_translation_error_m": 0.0,
            "median_translation_error_m": 0.0,
            "mean_rotation_error_deg": 0.0,
            "median_rotation_error_deg": 0.0,
            "translation_pass_rate": 0.0,
            "rotation_pass_rate": 0.0,
            "matched_component_count": 0,
        }

    trans_arr = np.array(all_trans_errors)
    rot_arr = np.array(all_rot_errors)

    return {
        "mean_translation_error_m": float(np.mean(trans_arr)),
        "median_translation_error_m": float(np.median(trans_arr)),
        "mean_rotation_error_deg": float(np.mean(rot_arr)),
        "median_rotation_error_deg": float(np.median(rot_arr)),
        "translation_pass_rate": float(np.mean(trans_arr <= translation_threshold)),
        "rotation_pass_rate": float(np.mean(rot_arr <= rotation_threshold)),
        "matched_component_count": len(all_trans_errors),
    }
