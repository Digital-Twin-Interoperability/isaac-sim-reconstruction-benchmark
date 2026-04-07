#!/usr/bin/env python3
"""Render focused sub-scene views from full_warehouse.usd.

Run from Isaac Sim's Python environment (Windows):

    D:\\isaac-sim-standalone-5.1.0-windows-x86_64\\python.bat ^
        samples/render_warehouse_views.py --headless

Produces output in samples/renders/<view_name>/:
    rgb.png, depth.npy, segmentation.png, scene_info.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# ---- Parse args before Isaac Sim bootstrap ----
parser = argparse.ArgumentParser(description="Render focused warehouse views")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--usd",
    default="full_warehouse.usd",
    help="Path to the warehouse USD file",
)
parser.add_argument(
    "--output-dir",
    default="samples/renders",
    help="Output root directory",
)
parser.add_argument(
    "--resolution",
    default="1024x1024",
    help="Render resolution WxH",
)
parser.add_argument(
    "--views",
    nargs="*",
    default=None,
    help="Render only these views (by name). Omit to render all.",
)
parser.add_argument(
    "--warm-up",
    type=int,
    default=60,
    help="Number of app.update() frames to wait for stage to load (default 60)",
)
parser.add_argument(
    "--no-segmentation",
    action="store_true",
    help="Skip semantic segmentation (faster, avoids syntheticdata crashes)",
)
args = parser.parse_args()

# ---- Isaac Sim bootstrap ----
print("[1/6] Starting SimulationApp ...")
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.kit.app
import omni.usd
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

# Replicator import — deferred until we know stage is loaded
rep = None


def _ensure_rep():
    global rep
    if rep is None:
        import omni.replicator.core as _rep
        rep = _rep


# ---------------------------------------------------------------------------
# View definitions
# ---------------------------------------------------------------------------

VIEWS = [
    {
        "name": "rack_aisle_perspective",
        "description": "Perspective view down the left rack aisle",
        "camera": {
            "position": (-14.5, 8.0, 4.0),
            "target": (-15.4, 22.0, 2.5),
            "fov_deg": 60.0,
        },
    },
    {
        "name": "shelf_closeup",
        "description": "Close-up of a rack bay with pallets and boxes on shelves",
        "camera": {
            "position": (-13.0, 15.0, 3.0),
            "target": (-15.4, 15.0, 2.5),
            "fov_deg": 50.0,
        },
    },
    {
        "name": "floor_area_overview",
        "description": "Open floor area between rack sections",
        "camera": {
            "position": (-8.0, 10.0, 5.0),
            "target": (-10.0, 17.0, 0.5),
            "fov_deg": 65.0,
        },
    },
    {
        "name": "right_aisle_perspective",
        "description": "Perspective view down the right rack aisle with boxes and pallets",
        "camera": {
            "position": (2.0, 9.0, 3.5),
            "target": (2.0, 22.0, 2.0),
            "fov_deg": 60.0,
        },
    },
    {
        "name": "forklift_closeup",
        "description": "Close-up of the forklift from the open floor area",
        "camera": {
            "position": (0.5, 8.5, 2.5),
            "target": (3.2, 10.8, 0.5),
            "fov_deg": 55.0,
        },
    },
]

# Semantic labels
LABEL_MAP = {
    "RackShelf": "rack_shelf",
    "RackFrame": "rack_frame",
    "Palette": "pallet",
    "Pallet": "pallet",
    "CardBox": "box",
    "FuseBox": "box",
    "Box_": "box",
    "Crate": "crate",
    "Barrel": "barrel",
    "Bottle": "bottle",
    "floor": "floor",
    "Floor": "floor",
    "Wall": "wall",
    "Ceiling": "ceiling",
    "Lamp": "light_fixture",
    "Sign": "sign",
    "forklift": "forklift",
    "TrafficCone": "traffic_cone",
    "FireExtinguisher": "fire_extinguisher",
    "Bracket": "bracket",
    "Beam": "beam",
    "Pillar": "pillar",
    "Rackshield": "rack_shield",
    "RackPile": "rack_pile",
    "RectLight": "rect_light",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_camera(stage, cam_path, position, target, fov_deg):
    cam = UsdGeom.Camera.Define(stage, cam_path)
    prim = cam.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    eye = Gf.Vec3d(*position)
    center = Gf.Vec3d(*target)
    up = Gf.Vec3d(0, 0, 1)

    look_at = Gf.Matrix4d()
    look_at.SetLookAt(eye, center, up)
    xformable.AddTransformOp().Set(look_at.GetInverse())

    horiz_aperture = cam.GetHorizontalApertureAttr().Get() or 20.955
    focal_length = horiz_aperture / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cam.GetFocalLengthAttr().Set(focal_length)
    return cam_path


def wait_for_stage(num_frames: int):
    """Pump the app event loop so assets load and materials compile."""
    print(f"  Warming up renderer ({num_frames} frames) ...")
    for i in range(num_frames):
        simulation_app.update()
        if (i + 1) % 20 == 0:
            print(f"    frame {i + 1}/{num_frames}")
    print("  Warm-up complete.")


def tag_semantics(stage):
    root = stage.GetPrimAtPath("/Root")
    if not root:
        print("  WARNING: /Root not found")
        return 0
    tagged = 0
    for prim in root.GetChildren():
        name = prim.GetName()
        for pattern, label in LABEL_MAP.items():
            if pattern.lower() in name.lower():
                try:
                    from isaacsim.core.utils.semantics import add_labels
                    add_labels(prim, labels=[label], instance_name="class")
                except Exception as e:
                    # Fallback: write USD attribute directly
                    try:
                        prim.CreateAttribute(
                            "semantics:Semantics:class:params:semanticType",
                            Sdf.ValueTypeNames.String,
                        ).Set("class")
                        prim.CreateAttribute(
                            "semantics:Semantics:class:params:semanticData",
                            Sdf.ValueTypeNames.String,
                        ).Set(label)
                    except Exception:
                        pass
                tagged += 1
                break
    return tagged


# ---------------------------------------------------------------------------
# Render one view using Replicator BasicWriter
# ---------------------------------------------------------------------------


def render_view_with_writer(stage, view, output_dir, resolution):
    """Use rep.WriterRegistry / BasicWriter for more stable rendering."""
    _ensure_rep()
    name = view["name"]
    cam_cfg = view["camera"]
    view_dir = output_dir / name
    view_dir.mkdir(parents=True, exist_ok=True)

    cam_path = f"/RenderCameras/{name}"
    create_camera(stage, cam_path, cam_cfg["position"], cam_cfg["target"], cam_cfg["fov_deg"])

    print(f"  [{name}] Creating render product ...")
    rp = rep.create.render_product(cam_path, resolution)

    # Let the render product initialize
    for _ in range(5):
        simulation_app.update()

    print(f"  [{name}] Setting up BasicWriter ...")
    writer = rep.WriterRegistry.get("BasicWriter")
    writer_params = {
        "output_dir": str(view_dir),
        "rgb": True,
        "distance_to_camera": True,
    }
    if not args.no_segmentation:
        writer_params["semantic_segmentation"] = True
        writer_params["colorize_semantic_segmentation"] = False
    writer.initialize(**writer_params)
    writer.attach([rp])

    # Render several frames to let the scene converge
    print(f"  [{name}] Rendering ...")
    for _ in range(10):
        simulation_app.update()
    rep.orchestrator.step()
    for _ in range(5):
        simulation_app.update()

    # Detach and destroy
    writer.detach()
    rp.destroy()
    for _ in range(3):
        simulation_app.update()

    print(f"  [{name}] Output -> {view_dir}")

    meta = {
        "name": name,
        "description": view["description"],
        "camera": cam_cfg,
        "resolution": list(resolution),
        "output_dir": str(view_dir),
    }
    (view_dir / "scene_info.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Render one view using annotators directly (fallback)
# ---------------------------------------------------------------------------


def move_camera(stage, cam_path, position, target, fov_deg):
    """Reposition an existing camera prim."""
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


def save_view(name, rgb_ann, depth_ann, seg_ann, view, view_dir):
    """Read annotator data and save to disk. Returns metadata dict."""
    cam_cfg = view["camera"]

    rgb_data = rgb_ann.get_data()
    if rgb_data is not None and rgb_data.ndim >= 3:
        rgb_data = rgb_data[:, :, :3].astype(np.uint8)
        Image.fromarray(rgb_data).save(view_dir / "rgb.png")
        print(f"  [{name}] Saved rgb.png ({rgb_data.shape})")
    else:
        print(f"  [{name}] WARNING: rgb data invalid (shape={getattr(rgb_data, 'shape', None)})")

    depth_data = depth_ann.get_data()
    if depth_data is not None and depth_data.ndim >= 2:
        depth_data = depth_data.astype(np.float32)
        np.save(view_dir / "depth.npy", depth_data)
        print(f"  [{name}] Saved depth.npy ({depth_data.shape})")

    if seg_ann is not None:
        seg_raw = seg_ann.get_data()
        if seg_raw is not None and "data" in seg_raw:
            seg_data = seg_raw["data"].astype(np.int32)

            # Save raw IDs as .npy for evaluation
            np.save(view_dir / "segmentation.npy", seg_data)

            # Save label mapping if available
            id_to_labels = {}
            if "info" in seg_raw and "idToLabels" in seg_raw["info"]:
                id_to_labels = seg_raw["info"]["idToLabels"]
                (view_dir / "segmentation_labels.json").write_text(
                    json.dumps(id_to_labels, indent=2)
                )
                print(f"  [{name}] Found {len(id_to_labels)} semantic labels")

            # Colorized visualization — assign distinct colors per ID
            unique_ids = np.unique(seg_data)
            print(f"  [{name}] Segmentation IDs: {unique_ids[:20]}{'...' if len(unique_ids) > 20 else ''}")
            rng = np.random.RandomState(42)
            color_lut = np.zeros((seg_data.max() + 1, 3), dtype=np.uint8)
            color_lut[1:] = rng.randint(40, 255, size=(len(color_lut) - 1, 3))
            # background (0) stays black
            seg_colored = color_lut[np.clip(seg_data, 0, len(color_lut) - 1)]
            Image.fromarray(seg_colored).save(view_dir / "segmentation.png")
            print(f"  [{name}] Saved segmentation.png (colorized, {seg_data.shape})")

    meta = {
        "name": name,
        "description": view["description"],
        "camera": cam_cfg,
        "resolution": list(args.resolution.split("x")),
        "output_dir": str(view_dir),
    }
    (view_dir / "scene_info.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    w, h = (int(x) for x in args.resolution.split("x"))
    resolution = (w, h)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Open stage ----
    usd_path = str(Path(args.usd).resolve())
    print(f"[2/6] Opening stage: {usd_path}")
    result = omni.usd.get_context().open_stage(usd_path)
    if not result:
        print("  ERROR: Failed to open stage")
        simulation_app.close()
        sys.exit(1)

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/Root")
    n_children = len(list(root.GetChildren())) if root else 0
    print(f"  Stage opened. /Root has {n_children} children.")

    # ---- Warm up ----
    print("[3/6] Warming up ...")
    wait_for_stage(args.warm_up)

    # ---- Tag semantics ----
    if not args.no_segmentation:
        print("[4/6] Tagging semantic labels ...")
        n = tag_semantics(stage)
        print(f"  Tagged {n} prims")
        for _ in range(5):
            simulation_app.update()
    else:
        print("[4/6] Skipping semantic tagging (--no-segmentation)")

    # ---- Select views ----
    views_to_render = VIEWS
    if args.views:
        names = set(args.views)
        views_to_render = [v for v in VIEWS if v["name"] in names]
        if not views_to_render:
            print(f"ERROR: no views matched {args.views}")
            print(f"Available: {[v['name'] for v in VIEWS]}")
            simulation_app.close()
            sys.exit(1)

    # ---- Create ONE camera + ONE render product (reuse across all views) ----
    _ensure_rep()
    cam_path = "/RenderCameras/BenchCam"
    first_cam = views_to_render[0]["camera"]
    create_camera(stage, cam_path, first_cam["position"], first_cam["target"], first_cam["fov_deg"])

    print(f"[5/6] Rendering {len(views_to_render)} view(s) at {w}x{h} ...")

    rp = rep.create.render_product(cam_path, resolution)
    for _ in range(10):
        simulation_app.update()

    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_ann.attach([rp])
    depth_ann.attach([rp])

    seg_ann = None
    if not args.no_segmentation:
        seg_ann = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False}
        )
        seg_ann.attach([rp])

    # Initial settle
    for _ in range(10):
        simulation_app.update()

    # ---- Render each view by moving the camera ----
    all_meta = []
    for view in views_to_render:
        name = view["name"]
        cam_cfg = view["camera"]
        view_dir = output_dir / name
        view_dir.mkdir(parents=True, exist_ok=True)

        try:
            move_camera(stage, cam_path, cam_cfg["position"], cam_cfg["target"], cam_cfg["fov_deg"])

            # Let the renderer converge with the new camera position
            for _ in range(10):
                simulation_app.update()
            rep.orchestrator.step()
            for _ in range(5):
                simulation_app.update()

            meta = save_view(name, rgb_ann, depth_ann, seg_ann, view, view_dir)
            all_meta.append(meta)
        except Exception as e:
            import traceback
            print(f"  [{name}] FAILED: {e}")
            traceback.print_exc()

    # ---- Cleanup (best-effort) ----
    for ann in [rgb_ann, depth_ann, seg_ann]:
        if ann is not None:
            try:
                ann.detach([rp])
            except Exception:
                pass
    try:
        rp.destroy()
    except Exception:
        pass

    # ---- Summary ----
    print(f"[6/6] Writing summary ...")
    summary_path = output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(all_meta, indent=2))
    print(f"\nDone. {len(all_meta)} views rendered. Summary: {summary_path}")

    # ---- Shutdown ----
    try:
        rep.orchestrator.stop()
    except Exception:
        pass
    for _ in range(5):
        simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
