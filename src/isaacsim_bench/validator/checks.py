from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.match_regime import derive_component_regime
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry


@dataclass
class CheckResult:
    passed: bool
    severity: Literal["error", "warning"]
    message: str


# ---------------------------------------------------------------------------
# Check 1: Transform sanity (AABB overlap)
# ---------------------------------------------------------------------------

def check_transform_sanity(
    scene: SceneJSON,
    default_extent: float = 0.5,
    overlap_warn: float = 0.3,
    overlap_reject: float = 0.8,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    comps = scene.components
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a_pos = np.array(comps[i].translate)
            b_pos = np.array(comps[j].translate)
            # Simplified AABB: cube of side default_extent centred at translate
            half = default_extent / 2.0
            a_min, a_max = a_pos - half, a_pos + half
            b_min, b_max = b_pos - half, b_pos + half

            inter_min = np.maximum(a_min, b_min)
            inter_max = np.minimum(a_max, b_max)
            inter_sides = np.maximum(inter_max - inter_min, 0.0)
            inter_vol = float(np.prod(inter_sides))

            box_vol = default_extent**3
            union_vol = 2 * box_vol - inter_vol
            iou = inter_vol / union_vol if union_vol > 0 else 0.0

            if iou > overlap_reject:
                results.append(
                    CheckResult(
                        passed=False,
                        severity="error",
                        message=(
                            f"Components '{comps[i].name}' and '{comps[j].name}' "
                            f"overlap with IoU={iou:.3f} > reject threshold {overlap_reject}"
                        ),
                    )
                )
            elif iou > overlap_warn:
                results.append(
                    CheckResult(
                        passed=True,
                        severity="warning",
                        message=(
                            f"Components '{comps[i].name}' and '{comps[j].name}' "
                            f"overlap with IoU={iou:.3f} > warn threshold {overlap_warn}"
                        ),
                    )
                )
    return results


# ---------------------------------------------------------------------------
# Check 2: Relation graph connectivity (BFS, primary components only)
# ---------------------------------------------------------------------------

def check_relation_connectivity(scene: SceneJSON) -> list[CheckResult]:
    primary_names = {c.name for c in scene.components if c.evaluation_role == "primary"}
    if len(primary_names) <= 1:
        return []

    adj: dict[str, set[str]] = {n: set() for n in primary_names}
    for rel in scene.relations:
        if rel.from_node in primary_names and rel.to_node in primary_names:
            adj[rel.from_node].add(rel.to_node)
            adj[rel.to_node].add(rel.from_node)

    # BFS from any primary node
    start = next(iter(primary_names))
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj.get(node, set()) - visited)

    unreachable = primary_names - visited
    if unreachable:
        return [
            CheckResult(
                passed=False,
                severity="error",
                message=(
                    f"Primary component graph is disconnected. "
                    f"Unreachable from '{start}': {sorted(unreachable)}"
                ),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Check 3: Camera validity (pinhole projection)
# ---------------------------------------------------------------------------

def check_camera_validity(
    scene: SceneJSON, min_coverage: float = 0.8
) -> list[CheckResult]:
    results: list[CheckResult] = []
    cam = scene.camera
    cam_pos = np.array(cam.position)
    cam_target = np.array(cam.target)
    forward = cam_target - cam_pos

    if np.linalg.norm(forward) < 1e-8:
        return [
            CheckResult(
                passed=False,
                severity="error",
                message="Degenerate camera: position equals target.",
            )
        ]

    forward = forward / np.linalg.norm(forward)
    half_fov = math.radians(cam.fov_deg / 2.0)

    in_frustum = 0
    total = len(scene.components)
    if total == 0:
        return []

    for comp in scene.components:
        to_comp = np.array(comp.translate) - cam_pos
        dist = np.linalg.norm(to_comp)
        if dist < 1e-8:
            in_frustum += 1
            continue
        cos_angle = float(np.dot(forward, to_comp / dist))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        if cos_angle >= math.cos(half_fov):
            in_frustum += 1

    coverage = in_frustum / total
    if coverage < min_coverage:
        results.append(
            CheckResult(
                passed=False,
                severity="error",
                message=(
                    f"Camera covers only {coverage:.0%} of components "
                    f"({in_frustum}/{total}), below threshold {min_coverage:.0%}."
                ),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Check 4: Asset existence (all asset_ids in world pool)
# ---------------------------------------------------------------------------

def check_asset_existence(
    scene: SceneJSON, registry: TaxonomyRegistry
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for comp in scene.components:
        if not registry.is_in_world_pool(comp.asset_id):
            results.append(
                CheckResult(
                    passed=False,
                    severity="error",
                    message=(
                        f"Component '{comp.name}': asset_id '{comp.asset_id}' "
                        f"not found in world pool."
                    ),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Check 5: Match regime consistency
# ---------------------------------------------------------------------------

def check_match_regime_consistency(
    scene: SceneJSON, registry: TaxonomyRegistry
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for comp in scene.components:
        expected = derive_component_regime(registry, comp.asset_id)
        if comp.match_regime != expected:
            results.append(
                CheckResult(
                    passed=False,
                    severity="error",
                    message=(
                        f"Component '{comp.name}': match_regime is "
                        f"'{comp.match_regime}' but should be '{expected}' "
                        f"(asset_id='{comp.asset_id}')."
                    ),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Check 6: File completeness
# ---------------------------------------------------------------------------

_REQUIRED_FILES = ["rgb.png", "scene.json"]
_DEPTH_OPTIONS = ["depth.npy", "depth.exr"]
_OPTIONAL_WITH_ONE_REQUIRED = ["segmentation.png"]


def check_file_completeness(sample_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    for fname in _REQUIRED_FILES:
        if not (sample_dir / fname).exists():
            results.append(
                CheckResult(
                    passed=False, severity="error", message=f"Missing required file: {fname}"
                )
            )

    if not any((sample_dir / d).exists() for d in _DEPTH_OPTIONS):
        results.append(
            CheckResult(
                passed=False,
                severity="error",
                message=f"Missing depth file: need one of {_DEPTH_OPTIONS}",
            )
        )

    for fname in _OPTIONAL_WITH_ONE_REQUIRED:
        if not (sample_dir / fname).exists():
            results.append(
                CheckResult(
                    passed=False, severity="error", message=f"Missing required file: {fname}"
                )
            )

    return results
