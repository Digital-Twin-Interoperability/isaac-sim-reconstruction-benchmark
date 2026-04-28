#!/usr/bin/env python3
"""Generate sample scenes from templates and export as JSON + USD.

Usage:
    uv run python scripts/generate_scenes.py

Produces scenes in samples/<scene_name>/ with:
    scene.json   — SceneJSON ground truth
    scene.usd    — USD file viewable in Isaac Sim
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pxr import Gf, Usd, UsdGeom

from isaacsim_bench.generator.isaac_sim import IsaacSimSceneGenerator
from isaacsim_bench.generator.templates import TEMPLATES
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

NUCLEUS_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/"
)

# Scene configs: (name, template_id, params)
SCENE_CONFIGS = [
    (
        "u_conveyor_default",
        "u_conveyor",
        {},
    ),
    (
        "u_conveyor_wide",
        "u_conveyor",
        {"segment_length": 3.5, "straight_asset": "ConveyorBelt_A02", "curve_asset": "ConveyorBelt_A10"},
    ),
    (
        "shelf_row_3",
        "shelf_row",
        {"count": 3, "spacing": 2.0},
    ),
    (
        "shelf_row_6",
        "shelf_row",
        {"count": 6, "spacing": 1.8},
    ),
    (
        "pallet_grid_2x3",
        "pallet_grid",
        {"rows": 2, "cols": 3, "spacing": 1.2},
    ),
    (
        "pallet_grid_3x4",
        "pallet_grid",
        {"rows": 3, "cols": 4, "spacing": 1.0},
    ),
]


def _build_variant_usd_index(taxonomy: AssetTaxonomy) -> dict[str, str]:
    index: dict[str, str] = {}
    for cat in taxonomy.categories:
        for var in cat.variants:
            if var.usd_path:
                index[var.variant_id] = var.usd_path
    return index


def _scene_to_usd(
    scene_json,
    usd_index: dict[str, str],
    output_path: Path,
) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/Root")

    for comp in scene_json.components:
        prim_path = f"/Root/{comp.name}"
        UsdGeom.Xform.Define(stage, prim_path)
        prim = stage.GetPrimAtPath(prim_path)

        usd_path = usd_index.get(comp.asset_id, "")
        if usd_path:
            prim.GetReferences().AddReference(f"{NUCLEUS_ASSET_ROOT}{usd_path}")

        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()

        tx, ty, tz = comp.translate
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))

        qx, qy, qz, qw = comp.orientation_xyzw
        orient_op = xformable.AddOrientOp()
        try:
            orient_op.Set(Gf.Quatd(qw, qx, qy, qz))
        except Exception:
            orient_op.Set(Gf.Quatf(qw, qx, qy, qz))

    stage.GetRootLayer().Save()


def main() -> None:
    registry = TaxonomyRegistry.load(
        DATA_DIR / "asset_taxonomy.json",
        DATA_DIR / "world_pool.json",
        DATA_DIR / "retrieval_pool.json",
    )
    taxonomy = AssetTaxonomy.model_validate_json(
        (DATA_DIR / "asset_taxonomy.json").read_text()
    )
    usd_index = _build_variant_usd_index(taxonomy)

    generator = IsaacSimSceneGenerator()

    print(f"Generating {len(SCENE_CONFIGS)} scenes...\n")

    for name, template_id, params in SCENE_CONFIGS:
        scene = generator.generate_scene(template_id, params, registry)
        out_dir = SAMPLES_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON
        json_path = out_dir / "scene.json"
        json_path.write_text(scene.model_dump_json(indent=2, by_alias=True))

        # Write USD
        usd_path = out_dir / "scene.usd"
        _scene_to_usd(scene, usd_index, usd_path)

        n_comps = len(scene.components)
        n_rels = len(scene.relations)
        print(f"  {name:25s}  {template_id:15s}  {n_comps} components, {n_rels} relations")

    print(f"\nAll scenes written to {SAMPLES_DIR}/")
    print("Open the .usd files in Isaac Sim to visualize.")


if __name__ == "__main__":
    main()
