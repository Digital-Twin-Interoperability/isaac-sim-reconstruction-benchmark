#!/usr/bin/env python3
"""Precompute axis-aligned bbox metadata for every asset in the retrieval pool.

For each ``variant_id`` in ``data/retrieval_pool.json``:
  * create an empty stage,
  * reference the variant's USD,
  * step the SimulationApp until referenced layers resolve,
  * compute the world-bound aligned range,
  * record ``extent_xyz`` (size), ``bbox_min`` and ``bbox_max`` in meters.

Output is ``data/asset_extents.json`` — a per-variant dict::

    {
      "ConveyorBelt_A14": {
        "extent_xyz": [2.57, 4.93, 1.17],
        "bbox_min":   [-1.28, 0.0, -0.58],
        "bbox_max":   [ 1.28, 4.93, 0.58]
      },
      ...
    }

The agentic loop loads it (when present) and surfaces both extents and
bbox-min/max via ``get_asset_info`` so the agent can reason about where the
asset's geometry sits relative to its local origin (``translate=(0,0,0)``).
Without bbox-min/max the agent only knows asset size, not where its
"connection points" land — which is what lets it chain modular pieces
end-to-end correctly.

**Incremental by default.** Loads any existing ``asset_extents.json``,
computes the set of variant_ids missing from it, and only boots Isaac Sim
to process the missing ones.  If everything is covered, exits before
booting (which is the expensive part).  Pass ``--force`` to recompute
every entry.

Run on the pod with Isaac Sim's bundled python:

    /isaac-sim/python.sh scripts/compute_asset_extents.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

NUCLEUS_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POOL_PATH = DATA_DIR / "retrieval_pool.json"
TAX_PATH = DATA_DIR / "asset_taxonomy.json"
OUT_PATH = DATA_DIR / "asset_extents.json"

LOAD_FRAMES = 60


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute bbox metadata for retrieval-pool assets",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute every variant, even those already in "
            "data/asset_extents.json (default: skip already-computed)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    pool_ids: list[str] = json.loads(POOL_PATH.read_text())["asset_ids"]

    existing: dict[str, dict[str, list[float]]] = {}
    if OUT_PATH.exists() and not args.force:
        try:
            existing = json.loads(OUT_PATH.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [warn] failed to read {OUT_PATH.name} ({e}); recomputing all")
            existing = {}

    missing = [vid for vid in pool_ids if vid not in existing]
    if not missing:
        print(
            f"All {len(pool_ids)} pool variant(s) already in "
            f"{OUT_PATH.name}; nothing to do.  Pass --force to recompute.",
        )
        return

    print(
        f"Pool has {len(pool_ids)} variant(s); "
        f"{len(existing)} already computed, {len(missing)} missing.  "
        "Booting Isaac Sim to fill the gap...",
    )

    # Heavy import — only reached when there's actually work to do.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    # Imports valid only once SimulationApp has booted.
    from pxr import Usd, UsdGeom

    from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

    taxonomy = AssetTaxonomy.model_validate_json(TAX_PATH.read_text())

    variant_index: dict[str, str] = {}
    for cat in taxonomy.categories:
        for var in cat.variants:
            if var.usd_path:
                variant_index[var.variant_id] = var.usd_path

    extents = dict(existing)  # start from existing so we never drop entries

    for vid in missing:
        usd_rel = variant_index.get(vid)
        if not usd_rel:
            print(f"  [warn] {vid}: no usd_path in taxonomy")
            continue

        usd_url = f"{NUCLEUS_ASSET_ROOT}{usd_rel}"
        print(f"  computing {vid} <- {usd_url}")

        # Read the referenced asset's authored metersPerUnit so we can
        # convert its bbox into meters.  USD references do NOT auto-scale
        # across unit mismatches: an asset authored in cm (mpu=0.01)
        # referenced into our mpu=1.0 stage yields a bbox in centimetres.
        # Without this conversion the IsaacLab packing-table family came
        # back as e.g. extent=[247, 76, 99] while CardBoxA was in m — the
        # mixed units silently break agent placement reasoning.
        try:
            ref_stage = Usd.Stage.Open(usd_url)
            mpu = UsdGeom.GetStageMetersPerUnit(ref_stage) if ref_stage else 1.0
        except Exception as e:
            print(f"    [warn] {vid}: could not read metersPerUnit ({e}); assuming 1.0")
            mpu = 1.0
        if mpu <= 0:
            mpu = 1.0

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/Asset")
        prim = stage.GetPrimAtPath("/Asset")
        prim.GetReferences().AddReference(usd_url)

        for _ in range(LOAD_FRAMES):
            app.update()

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
        )
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        if rng.IsEmpty():
            print(f"    [warn] empty bbox for {vid}; skipping")
            continue
        bmin = rng.GetMin()
        bmax = rng.GetMax()
        size = bmax - bmin
        # Multiply by the asset's native metersPerUnit to land in metres.
        entry = {
            "extent_xyz": [round(float(size[0]) * mpu, 4),
                           round(float(size[1]) * mpu, 4),
                           round(float(size[2]) * mpu, 4)],
            "bbox_min": [round(float(bmin[0]) * mpu, 4),
                         round(float(bmin[1]) * mpu, 4),
                         round(float(bmin[2]) * mpu, 4)],
            "bbox_max": [round(float(bmax[0]) * mpu, 4),
                         round(float(bmax[1]) * mpu, 4),
                         round(float(bmax[2]) * mpu, 4)],
        }
        if mpu != 1.0:
            print(f"    [info] asset mpu={mpu}, scaled bbox to metres")
        extents[vid] = entry
        print(
            f"    extent={entry['extent_xyz']} "
            f"min={entry['bbox_min']} max={entry['bbox_max']}",
        )

        # Persist as we go — boot is the expensive part, but bad assets can
        # still hang on a single ComputeWorldBound, so don't lose progress.
        OUT_PATH.write_text(json.dumps(extents, indent=2))

    OUT_PATH.write_text(json.dumps(extents, indent=2))
    print(
        f"\nWrote bbox metadata for {len(extents)} assets to {OUT_PATH} "
        f"({len(missing)} newly computed).",
    )
    app.close()


if __name__ == "__main__":
    main()
