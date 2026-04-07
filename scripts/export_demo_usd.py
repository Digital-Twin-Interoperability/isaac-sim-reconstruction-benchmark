#!/usr/bin/env python3
"""Export the demo GT and perturbed prediction as USD files.

Usage:
    uv run python scripts/export_demo_usd.py

Produces:
    samples/demo/gt_scene.usd         — ground truth (20 objects from warehouse)
    samples/demo/pred_scene.usd       — perturbed prediction
    samples/demo/gt_scene.json        — GT as SceneJSON
    samples/demo/pred_scene.json      — prediction as PredictionJSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pxr import Gf, Sdf, Usd, UsdGeom

from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "demo"

# Nucleus CDN root — Isaac Sim resolves these via its HTTP asset resolver
NUCLEUS_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/"
)


def _build_variant_usd_index(taxonomy: AssetTaxonomy) -> dict[str, str]:
    """Map variant_id -> full usd_path."""
    index: dict[str, str] = {}
    for cat in taxonomy.categories:
        for var in cat.variants:
            if var.usd_path:
                index[var.variant_id] = var.usd_path
    return index


def _scene_to_usd(
    components: list[dict],
    usd_index: dict[str, str],
    output_path: Path,
    asset_root: str = "",
) -> None:
    """Write a list of components (name, asset_id, translate, orientation_xyzw) to USD."""
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Root")

    for comp in components:
        prim_path = f"/Root/{comp['name']}"
        xform = UsdGeom.Xform.Define(stage, prim_path)
        prim = stage.GetPrimAtPath(prim_path)

        # Add reference to the actual asset if we have its USD path
        usd_path = usd_index.get(comp["asset_id"], "")
        if usd_path:
            ref_path = f"{asset_root}{usd_path}" if asset_root else usd_path
            prim.GetReferences().AddReference(ref_path)

        # Set transform
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()

        tx, ty, tz = comp["translate"]
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))

        qx, qy, qz, qw = comp["orientation_xyzw"]
        orient_op = xformable.AddOrientOp()
        # References may define orient as Quatf; try double first, fall back to float
        try:
            orient_op.Set(Gf.Quatd(qw, qx, qy, qz))
        except Exception:
            orient_op.Set(Gf.Quatf(qw, qx, qy, qz))

    stage.GetRootLayer().Save()


def main() -> None:
    # Reuse the demo_evaluation scene builder
    from demo_evaluation import make_demo_scene

    gt_scene, pred_scene, _ = make_demo_scene()
    taxonomy = AssetTaxonomy.model_validate_json(
        (DATA_DIR / "asset_taxonomy.json").read_text()
    )
    usd_index = _build_variant_usd_index(taxonomy)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export GT
    gt_components = [
        {
            "name": c.name,
            "asset_id": c.asset_id,
            "translate": c.translate,
            "orientation_xyzw": c.orientation_xyzw,
        }
        for c in gt_scene.components
    ]
    gt_usd_path = OUT_DIR / "gt_scene.usd"
    _scene_to_usd(gt_components, usd_index, gt_usd_path, asset_root=NUCLEUS_ASSET_ROOT)

    # Export prediction
    pred_components = [
        {
            "name": c.name,
            "asset_id": c.asset_id,
            "translate": c.translate,
            "orientation_xyzw": c.orientation_xyzw,
        }
        for c in pred_scene.components
    ]
    pred_usd_path = OUT_DIR / "pred_scene.usd"
    _scene_to_usd(pred_components, usd_index, pred_usd_path, asset_root=NUCLEUS_ASSET_ROOT)

    # Also save the JSON versions
    (OUT_DIR / "gt_scene.json").write_text(
        gt_scene.model_dump_json(indent=2, by_alias=True)
    )
    (OUT_DIR / "pred_scene.json").write_text(
        pred_scene.model_dump_json(indent=2)
    )

    print(f"Exported to {OUT_DIR}/:")
    print(f"  gt_scene.usd    — {len(gt_components)} components")
    print(f"  pred_scene.usd  — {len(pred_components)} components")
    print(f"  gt_scene.json")
    print(f"  pred_scene.json")
    print()
    print("Perturbations in pred_scene:")
    print("  - 2 objects missing (SM_BottlePlasticA_01_1429, SM_WetFloorSign_1515)")
    print("  - 1 object shifted +0.05m (within threshold)")
    print("  - 1 object shifted +0.15m (over threshold)")
    print("  - 1 object shifted +0.71m (far off)")
    print("  - 1 object rotated 45 deg")
    print("  - 1 wrong asset (rack -> box)")
    print("  - 1 hallucinated object")
    print()
    print("Open in Isaac Sim to visualize. Asset references resolve")
    print("when the Isaac Sim Nucleus asset server is accessible.")


if __name__ == "__main__":
    main()
