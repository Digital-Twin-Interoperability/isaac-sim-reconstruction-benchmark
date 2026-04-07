#!/usr/bin/env python3
"""Demo: take a slice of the warehouse GT, perturb it, and run the evaluator.

Produces a comparison report showing how the evaluator scores a prediction
with moved objects, wrong assets, and missing components.

Usage:
    uv run python scripts/demo_evaluation.py
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from isaacsim_bench.evaluator.runner import EvaluatorRunner
from isaacsim_bench.schemas.prediction import (
    PredictedComponent,
    PredictedRelation,
    PredictionJSON,
)
from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

random.seed(42)


def make_demo_scene() -> tuple[SceneJSON, PredictionJSON, TaxonomyRegistry]:
    """Build a small GT scene from the warehouse and a perturbed prediction."""

    # Load full warehouse GT
    gt_full = SceneJSON.model_validate_json(
        (DATA_DIR / "gt_full_warehouse.json").read_text()
    )

    # Load taxonomy + pools
    registry = TaxonomyRegistry.load(
        DATA_DIR / "asset_taxonomy.json",
        DATA_DIR / "world_pool.json",
        DATA_DIR / "retrieval_pool.json",
    )

    # Pick a diverse subset: 5 rack, 5 box, 3 pallet, 2 barrel, 2 bottle,
    # 1 forklift, 1 crate, 1 safety = 20 primary components
    primary = [c for c in gt_full.components if c.evaluation_role == "primary"]
    by_family: dict[str, list] = {}
    for c in primary:
        by_family.setdefault(c.family, []).append(c)

    picks = {
        "rack": 5,
        "box": 5,
        "pallet": 3,
        "barrel": 2,
        "bottle": 2,
        "vehicle": 1,
        "crate": 1,
        "safety": 1,
    }

    selected = []
    for fam, count in picks.items():
        pool = by_family.get(fam, [])
        selected.extend(pool[:count])

    # Build the small GT scene
    gt_scene = SceneJSON(
        sample_id="warehouse_demo",
        benchmark_tier="closed_world",
        family="warehouse",
        template_id="usd_import",
        root_node=selected[0].name,
        template_params={},
        camera=gt_full.camera,
        components=selected,
        relations=[],
        taxonomy_version=gt_full.taxonomy_version,
        generator_version="demo",
    )

    # --- Create a perturbed prediction ---
    # Simulate a reconstruction pipeline that:
    #  1. Gets most objects right (14/20)
    #  2. Moves a few objects slightly (small translation error)
    #  3. Moves one object far (large translation error)
    #  4. Rotates one object wrong
    #  5. Swaps one asset_id for a wrong one
    #  6. Misses 2 objects entirely
    #  7. Hallucinates 1 extra object

    pred_components: list[PredictedComponent] = []

    for i, gt_comp in enumerate(selected):
        # Skip 2 objects (index 15, 18) — simulates missing detection
        if i in (15, 18):
            continue

        t = list(gt_comp.translate)
        q = list(gt_comp.orientation_xyzw)
        aid = gt_comp.asset_id

        if i == 3:
            # Move slightly: +0.05m in x (within 0.10m threshold)
            t[0] += 0.05
        elif i == 7:
            # Move moderately: +0.15m in y (exceeds 0.10m threshold)
            t[1] += 0.15
        elif i == 10:
            # Move far: +0.5m in x and z
            t[0] += 0.5
            t[2] += 0.5
        elif i == 12:
            # Rotate wrong: 45 degrees around z-axis
            angle = math.radians(45)
            q = [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]
        elif i == 5:
            # Wrong asset: swap to a different variant
            aid = "CardBox_A1_01"  # a box variant, wrong for this rack

        pred_components.append(
            PredictedComponent(
                name=gt_comp.name,
                asset_id=aid,
                family=gt_comp.family,
                translate=t,
                orientation_xyzw=q,
                confidence=random.uniform(0.7, 0.99),
            )
        )

    # Hallucinate one extra object
    pred_components.append(
        PredictedComponent(
            name="Hallucinated_Box_999",
            asset_id="CardBox_A1_01",
            family="box",
            translate=[5.0, 5.0, 0.0],
            orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            confidence=0.55,
        )
    )

    pred_scene = PredictionJSON(
        sample_id="warehouse_demo",
        components=pred_components,
        relations=[],
        abstained=False,
        latency_seconds=2.34,
    )

    return gt_scene, pred_scene, registry


def main() -> None:
    gt_scene, pred_scene, registry = make_demo_scene()

    runner = EvaluatorRunner()
    report = runner.evaluate([gt_scene], [pred_scene], registry)

    result = report.to_dict()

    out_path = DATA_DIR / "demo_evaluation_report.json"
    out_path.write_text(json.dumps(result, indent=2))

    # Print human-readable summary
    print("=" * 70)
    print("  EVALUATION REPORT: Warehouse Demo Scene")
    print("=" * 70)

    gt_primary = [c for c in gt_scene.components if c.evaluation_role == "primary"]
    print(f"\n  Ground Truth:  {len(gt_primary)} primary components")
    print(f"  Prediction:    {len(pred_scene.components)} components")
    print()

    # Component metrics
    comp = result["component"]
    print("--- Component Matching (exact asset_id) ---")
    print(f"  Precision:          {comp['precision']:.3f}")
    print(f"  Recall:             {comp['recall']:.3f}")
    print(f"  F1:                 {comp['f1']:.3f}")
    print(f"  True positives:     {comp['total_tp']}")
    print(f"  False positives:    {comp['total_fp']}  (hallucinated or wrong asset)")
    print(f"  False negatives:    {comp['total_fn']}  (missed detections)")
    print(f"  Count accuracy:     {comp['exact_count_accuracy']:.3f}")
    print()

    # Retrieval metrics
    ret = result["retrieval"]
    print("--- Retrieval ---")
    print(f"  Exact Recall@1:     {ret['exact_recall_at_1']:.3f}  ({ret['exact_count']} components)")
    print(f"  MRR:                {ret['mrr']:.3f}")
    print()

    # Placement metrics
    plc = result["placement"]
    print("--- Placement (root-relative) ---")
    print(f"  Mean translation error:    {plc['mean_translation_error_m']:.4f} m")
    print(f"  Median translation error:  {plc['median_translation_error_m']:.4f} m")
    print(f"  Mean rotation error:       {plc['mean_rotation_error_deg']:.2f} deg")
    print(f"  Median rotation error:     {plc['median_rotation_error_deg']:.2f} deg")
    print(f"  Translation pass rate:     {plc['translation_pass_rate']:.3f}  (<= 0.10m)")
    print(f"  Rotation pass rate:        {plc['rotation_pass_rate']:.3f}  (<= 10 deg)")
    print(f"  Matched components:        {plc['matched_component_count']}")
    print()


    # Relation metrics
    rel = result["relation"]
    print("--- Relations ---")
    print(f"  (No relations in USD-extracted GT — all zeros expected)")
    print()

    # Scene success
    ss = result["scene_success"]
    status = "PASS" if ss["scene_success_rate"] == 1.0 else "FAIL"
    print("--- Scene Success ---")
    print(f"  Result:             {status}")
    print(f"  Success rate:       {ss['scene_success_rate']:.3f}")
    print(f"  Scenes:             {ss['successful_scenes']}/{ss['total_scenes']}")
    print(f"  (Requires: component F1 >= 0.9, placement within thresholds)")
    print()

    # What went wrong
    print("--- Perturbation Summary ---")
    print("  - 2 objects dropped (missed by detector)")
    print("  - 1 object moved +0.05m (within threshold)")
    print("  - 1 object moved +0.15m (exceeds threshold)")
    print("  - 1 object moved +0.71m (far off)")
    print("  - 1 object rotated 45 deg (exceeds threshold)")
    print("  - 1 object given wrong asset_id")
    print("  - 1 hallucinated object added")
    print()

    print(f"Full report saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
