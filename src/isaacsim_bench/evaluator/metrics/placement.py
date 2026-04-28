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


def _quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (xyzw format)."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in xyzw format."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _rotate_vec_by_quat(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a unit quaternion (xyzw format)."""
    v_quat = np.array([v[0], v[1], v[2], 0.0])
    rotated = _quat_mul(_quat_mul(q, v_quat), _quat_inv(q))
    return rotated[:3]


def _root_relative_pose(
    components: list, root_name: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Express each component pose relative to root node.

    Translations are rotated into the root's local coordinate frame,
    making the result invariant to the root's global position and
    orientation.  This means a correctly-assembled scene scores well
    regardless of where it is placed in world space.

    Returns dict of name -> (relative_translate, relative_orientation_xyzw).
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

    root_quat_inv = _quat_inv(root_quat)
    result = {}
    for c in components:
        # Translate into root's frame, then rotate into root's local axes
        world_offset = np.array(c.translate) - root_translate
        rel_t = _rotate_vec_by_quat(root_quat_inv, world_offset)
        rel_q = _quat_mul(root_quat_inv, np.array(c.orientation_xyzw))
        result[c.name] = (rel_t, rel_q)
    return result


def _procrustes_align(
    gt_pts: np.ndarray, pred_pts: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Align pred_pts to gt_pts via Procrustes (rotation + translation).

    Returns (aligned_pred_pts, rmse).  No scaling — only rigid transform.
    """
    gt_c = gt_pts.mean(axis=0)
    pred_c = pred_pts.mean(axis=0)
    gt_centered = gt_pts - gt_c
    pred_centered = pred_pts - pred_c

    H = pred_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    S[2, 2] = np.sign(d)
    R = Vt.T @ S @ U.T

    aligned = (R @ pred_centered.T).T + gt_c
    rmse = float(np.sqrt(np.mean(np.sum((aligned - gt_pts) ** 2, axis=1))))
    return aligned, rmse


def _pairwise_distance_error(
    gt_pts: np.ndarray, pred_pts: np.ndarray,
) -> float:
    """Mean absolute error of pairwise distances (rigid-transform invariant)."""
    n = len(gt_pts)
    if n < 2:
        return 0.0
    errors = []
    for i in range(n):
        for j in range(i + 1, n):
            gt_d = float(np.linalg.norm(gt_pts[i] - gt_pts[j]))
            pred_d = float(np.linalg.norm(pred_pts[i] - pred_pts[j]))
            errors.append(abs(gt_d - pred_d))
    return float(np.mean(errors))


def compute_placement_metrics(
    gt_scenes: list[SceneJSON],
    pred_scenes: list[PredictionJSON],
    component_matches: list[MatchResult],
    translation_threshold: float = 0.5,
    rotation_threshold: float = 15.0,
) -> dict:
    """Placement metrics for matched primary components.

    Three complementary metrics:
      1. **Root-relative** — poses in root's local frame (translation + rotation invariant).
      2. **Procrustes-aligned** — optimal rigid alignment, then per-component residual.
      3. **Pairwise distance** — MAE of inter-component distances (fully rigid-invariant).
    """
    all_trans_errors: list[float] = []
    all_rot_errors: list[float] = []
    all_procrustes_rmse: list[float] = []
    all_pairwise_err: list[float] = []

    for gt_scene, pred_scene, match in zip(gt_scenes, pred_scenes, component_matches):
        primary_gt = [c for c in gt_scene.components if c.evaluation_role == "primary"]

        if not match.matched_pairs:
            continue

        # Resolve predicted root via matched pairs
        gt_root_idx = next(
            (i for i, c in enumerate(primary_gt) if c.name == gt_scene.root_node),
            None,
        )
        pred_root_name = None
        if gt_root_idx is not None:
            for gi, pi in match.matched_pairs:
                if gi == gt_root_idx:
                    pred_root_name = pred_scene.components[pi].name
                    break

        gt_poses = _root_relative_pose(primary_gt, gt_scene.root_node)
        pred_poses = _root_relative_pose(
            pred_scene.components,
            pred_root_name if pred_root_name else gt_scene.root_node,
        )

        # Collect matched positions for Procrustes & pairwise metrics
        gt_pts_list: list[np.ndarray] = []
        pred_pts_list: list[np.ndarray] = []

        for gt_idx, pred_idx in match.matched_pairs:
            gt_comp = primary_gt[gt_idx]
            pred_comp = pred_scene.components[pred_idx]

            gt_pts_list.append(np.array(gt_comp.translate))
            pred_pts_list.append(np.array(pred_comp.translate))

            # Root-relative errors
            gt_pose = gt_poses.get(gt_comp.name)
            pred_pose = pred_poses.get(pred_comp.name)
            if gt_pose is not None and pred_pose is not None:
                all_trans_errors.append(
                    float(np.linalg.norm(gt_pose[0] - pred_pose[0]))
                )
                all_rot_errors.append(
                    _quat_angular_distance_deg(gt_pose[1], pred_pose[1])
                )

        # Procrustes alignment
        if len(gt_pts_list) >= 2:
            gt_pts = np.array(gt_pts_list)
            pred_pts = np.array(pred_pts_list)
            _, rmse = _procrustes_align(gt_pts, pred_pts)
            all_procrustes_rmse.append(rmse)
            all_pairwise_err.append(_pairwise_distance_error(gt_pts, pred_pts))

    if not all_trans_errors:
        return {
            "root_relative": {
                "mean_translation_error_m": 0.0,
                "median_translation_error_m": 0.0,
                "mean_rotation_error_deg": 0.0,
                "median_rotation_error_deg": 0.0,
                "translation_pass_rate": 0.0,
                "rotation_pass_rate": 0.0,
            },
            "procrustes_rmse_m": 0.0,
            "pairwise_distance_mae_m": 0.0,
            "matched_component_count": 0,
        }

    trans_arr = np.array(all_trans_errors)
    rot_arr = np.array(all_rot_errors)

    return {
        "root_relative": {
            "mean_translation_error_m": float(np.mean(trans_arr)),
            "median_translation_error_m": float(np.median(trans_arr)),
            "mean_rotation_error_deg": float(np.mean(rot_arr)),
            "median_rotation_error_deg": float(np.median(rot_arr)),
            "translation_pass_rate": float(np.mean(trans_arr <= translation_threshold)),
            "rotation_pass_rate": float(np.mean(rot_arr <= rotation_threshold)),
        },
        "procrustes_rmse_m": float(np.mean(all_procrustes_rmse)) if all_procrustes_rmse else 0.0,
        "pairwise_distance_mae_m": float(np.mean(all_pairwise_err)) if all_pairwise_err else 0.0,
        "matched_component_count": len(all_trans_errors),
    }
