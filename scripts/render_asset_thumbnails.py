#!/usr/bin/env python3
"""Render thumbnail previews for all assets in the taxonomy.

Produces one PNG per variant_id, suitable for building CLIP embeddings
for asset retrieval.

Run from Isaac Sim's Python environment (Windows):

    D:\\isaac-sim-standalone-5.1.0-windows-x86_64\\python.bat ^
        scripts/render_asset_thumbnails.py --headless

Produces output in data/asset_thumbnails/<variant_id>.png

Supports resuming — already-rendered PNGs are skipped.
Filter by family with --families, or render specific IDs with --assets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# ---- Parse args before Isaac Sim bootstrap ----
parser = argparse.ArgumentParser(description="Render asset thumbnail previews")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--output-dir",
    default="data/asset_thumbnails",
    help="Output directory for thumbnails",
)
parser.add_argument(
    "--resolution",
    default="512x512",
    help="Render resolution WxH (default: 512x512)",
)
parser.add_argument(
    "--assets",
    nargs="*",
    default=None,
    help="Render only these variant IDs.",
)
parser.add_argument(
    "--families",
    nargs="*",
    default=None,
    help="Render only assets from these families (e.g. conveyor rack pallet box).",
)
parser.add_argument(
    "--pool-only",
    action="store_true",
    help="Render only assets in the world/retrieval pools.",
)
parser.add_argument(
    "--skip-existing",
    action="store_true",
    default=True,
    help="Skip assets that already have a rendered PNG (default: True).",
)
parser.add_argument(
    "--no-skip-existing",
    action="store_false",
    dest="skip_existing",
    help="Re-render all assets even if PNG already exists.",
)
parser.add_argument(
    "--warm-up",
    type=int,
    default=30,
    help="Frames to warm up renderer at start (default: 30)",
)
parser.add_argument(
    "--settle-frames",
    type=int,
    default=20,
    help="Frames to settle per asset before capture (default: 20)",
)
args = parser.parse_args()

# ---- Isaac Sim bootstrap ----
print("[1/4] Starting SimulationApp ...")
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.kit.app
import omni.usd
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

import omni.replicator.core as rep

# ---- Constants ----
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NUCLEUS_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/"
)


def load_taxonomy() -> dict:
    return json.loads((DATA_DIR / "asset_taxonomy.json").read_text())


def load_pool_asset_ids() -> set[str]:
    wp = json.loads((DATA_DIR / "world_pool.json").read_text())
    rp = json.loads((DATA_DIR / "retrieval_pool.json").read_text())
    return set(wp["asset_ids"] + rp["asset_ids"])


def collect_assets(
    taxonomy: dict,
    asset_ids: list[str] | None,
    families: list[str] | None,
    pool_only: bool,
) -> list[tuple[str, str, str]]:
    """Return list of (variant_id, usd_path, family) to render."""
    pool_ids = load_pool_asset_ids() if pool_only else None
    family_set = set(families) if families else None
    asset_set = set(asset_ids) if asset_ids else None

    results = []
    for cat in taxonomy["categories"]:
        fam = cat["family"]
        if family_set and fam not in family_set:
            continue
        for var in cat["variants"]:
            vid = var["variant_id"]
            usd_path = var.get("usd_path")
            if not usd_path:
                continue
            if asset_set and vid not in asset_set:
                continue
            if pool_ids is not None and vid not in pool_ids:
                continue
            results.append((vid, usd_path, fam))

    return results


def compute_bbox(stage) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    """Compute bounding box center and size for /World/Asset."""
    prim = stage.GetPrimAtPath("/World/Asset")
    if not prim:
        return Gf.Vec3d(0, 0, 0), Gf.Vec3d(1, 1, 1)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    box = bbox.ComputeAlignedRange()
    lo = Gf.Vec3d(box.GetMin())
    hi = Gf.Vec3d(box.GetMax())
    center = (lo + hi) * 0.5
    size = hi - lo
    return center, size


def camera_for_bbox(
    center: Gf.Vec3d,
    size: Gf.Vec3d,
    fov_deg: float = 50.0,
) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    """Compute camera eye/target to frame the bounding box from a 3/4 angle."""
    diag = math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)
    if diag < 0.01:
        diag = 1.0

    half_fov_rad = math.radians(fov_deg) / 2.0
    dist = (diag * 0.7) / math.tan(half_fov_rad)
    dist = max(dist, 1.0)

    # 35° elevation, 45° azimuth
    elev_rad = math.radians(35)
    azim_rad = math.radians(45)
    dx = dist * math.cos(elev_rad) * math.cos(azim_rad)
    dy = dist * math.cos(elev_rad) * math.sin(azim_rad)
    dz = dist * math.sin(elev_rad)

    eye = Gf.Vec3d(center[0] + dx, center[1] + dy, center[2] + dz)
    target = Gf.Vec3d(center[0], center[1], center[2])
    return eye, target


def create_or_move_camera(stage, cam_path, position, target, fov_deg):
    prim = stage.GetPrimAtPath(cam_path)
    if not prim:
        UsdGeom.Camera.Define(stage, cam_path)
        prim = stage.GetPrimAtPath(cam_path)

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    eye = Gf.Vec3d(*position)
    center = Gf.Vec3d(*target)
    up = Gf.Vec3d(0, 0, 1)

    look_at = Gf.Matrix4d()
    look_at.SetLookAt(eye, center, up)
    xformable.AddTransformOp().Set(look_at.GetInverse())

    cam = UsdGeom.Camera(prim)
    horiz_aperture = cam.GetHorizontalApertureAttr().Get() or 20.955
    focal_length = horiz_aperture / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cam.GetFocalLengthAttr().Set(focal_length)


def setup_stage(stage):
    """Clean stage with lighting."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")

    # Dome light for even ambient illumination
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.GetIntensityAttr().Set(500.0)

    # Distant light for directional shadows
    dist_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    dist_light.GetIntensityAttr().Set(3000.0)
    xf = UsdGeom.Xformable(dist_light.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    # Ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.GetPointsAttr().Set(
        [Gf.Vec3f(-50, -50, 0), Gf.Vec3f(50, -50, 0),
         Gf.Vec3f(50, 50, 0), Gf.Vec3f(-50, 50, 0)]
    )
    ground.GetFaceVertexCountsAttr().Set([4])
    ground.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])


def swap_asset(stage, usd_url: str):
    """Remove old asset prim and add a new reference, reusing the same stage."""
    asset_path = "/World/Asset"
    old = stage.GetPrimAtPath(asset_path)
    if old:
        stage.RemovePrim(asset_path)

    prim = UsdGeom.Xform.Define(stage, asset_path).GetPrim()
    prim.GetReferences().AddReference(usd_url)


def main():
    w, h = (int(x) for x in args.resolution.split("x"))
    resolution = (w, h)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect assets to render ----
    taxonomy = load_taxonomy()
    assets = collect_assets(taxonomy, args.assets, args.families, args.pool_only)

    # Skip already-rendered
    if args.skip_existing:
        before = len(assets)
        assets = [
            (vid, usd, fam)
            for vid, usd, fam in assets
            if not (output_dir / f"{vid}.png").exists()
        ]
        skipped = before - len(assets)
        if skipped:
            print(f"  Skipping {skipped} already-rendered assets")

    print(f"[2/4] Will render {len(assets)} asset thumbnails at {w}x{h}")
    if not assets:
        print("  Nothing to render.")
        simulation_app.close()
        return

    # ---- Set up one persistent stage ----
    print("[3/4] Setting up stage and renderer ...")
    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(5):
        simulation_app.update()

    stage = ctx.get_stage()
    setup_stage(stage)

    # Camera + render product created once, reused for all assets
    cam_path = "/World/ThumbnailCam"
    # Place camera at a default position; will be repositioned per asset
    create_or_move_camera(
        stage, cam_path,
        position=(3, 3, 3), target=(0, 0, 0), fov_deg=50.0,
    )

    for _ in range(args.warm_up):
        simulation_app.update()

    rp = rep.create.render_product(cam_path, resolution)
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_ann.attach([rp])

    # Let the render pipeline initialise
    for _ in range(10):
        simulation_app.update()

    # ---- Render each asset ----
    print(f"[4/4] Rendering {len(assets)} thumbnails ...")
    rendered = []
    failed = []
    t_start = time.time()

    for idx, (variant_id, usd_path, family) in enumerate(assets):
        full_url = f"{NUCLEUS_ASSET_ROOT}{usd_path}"
        elapsed = time.time() - t_start
        rate = (idx / elapsed) if elapsed > 0 and idx > 0 else 0
        eta = ((len(assets) - idx) / rate) if rate > 0 else 0
        print(
            f"\n  [{idx + 1}/{len(assets)}] {variant_id}  "
            f"({family})  [{rate:.1f} assets/s, ETA {eta / 60:.0f}m]"
        )

        try:
            # Swap the asset on the existing stage
            swap_asset(stage, full_url)

            # Let the new asset load
            for _ in range(args.settle_frames):
                simulation_app.update()

            # Recompute camera framing for this asset
            center, size = compute_bbox(stage)
            eye, target = camera_for_bbox(center, size)
            create_or_move_camera(stage, cam_path, eye, target, fov_deg=50.0)

            # Settle and render
            for _ in range(args.settle_frames):
                simulation_app.update()
            rep.orchestrator.step()
            for _ in range(5):
                simulation_app.update()

            # Capture
            rgb_data = rgb_ann.get_data()
            if rgb_data is not None and rgb_data.ndim >= 3:
                rgb_data = rgb_data[:, :, :3].astype(np.uint8)
                out_path = output_dir / f"{variant_id}.png"
                Image.fromarray(rgb_data).save(out_path)
                rendered.append(variant_id)
            else:
                print(f"    WARNING: invalid rgb data")
                failed.append(variant_id)

        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            failed.append(variant_id)

        # Periodic index save (every 50 assets)
        if (idx + 1) % 50 == 0:
            _save_index(output_dir, rendered, failed)

    # ---- Final index ----
    _save_index(output_dir, rendered, failed)

    elapsed = time.time() - t_start
    print(f"\nDone. {len(rendered)}/{len(assets)} rendered, "
          f"{len(failed)} failed in {elapsed / 60:.1f}m")
    if failed:
        print(f"Failed IDs: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    # ---- Shutdown ----
    try:
        rgb_ann.detach([rp])
    except Exception:
        pass
    try:
        rp.destroy()
    except Exception:
        pass
    try:
        rep.orchestrator.stop()
    except Exception:
        pass
    for _ in range(5):
        simulation_app.update()
    simulation_app.close()


def _save_index(output_dir: Path, rendered: list[str], failed: list[str]):
    """Write thumbnail_index.json with current progress."""
    index = {
        "total_rendered": len(rendered),
        "total_failed": len(failed),
        "rendered": rendered,
        "failed": failed,
    }
    (output_dir / "thumbnail_index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
