#!/usr/bin/env python3
"""Render preview images of a composed scene from multiple camera angles.

Produces PNGs that can be used as input for multi-agent scene reconstruction.

Run from Isaac Sim's Python environment.

    # Linux
    /path/to/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \\
        scripts/render_scene.py samples/u_conveyor_default --headless

    # Windows
    D:\\isaac-sim-standalone-5.1.0-windows-x86_64\\python.bat ^
        scripts/render_scene.py samples/u_conveyor_default --headless

Produces output in samples/u_conveyor_default/renders/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the src package is importable when running from the repo root under
# Isaac Sim's bundled Python.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from isaacsim_bench.renderer.session import (
    IsaacRenderSession,
    Viewpoint,
    auto_orbit_viewpoints,
    compute_scene_bbox,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scene preview images")
    parser.add_argument(
        "scene_dir",
        help="Path to scene directory containing scene.usd and scene.json",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--resolution",
        default="1024x1024",
        help="Render resolution WxH (default: 1024x1024)",
    )
    parser.add_argument(
        "--warm-up",
        type=int,
        default=60,
        help="Frames to warm up renderer (default: 60)",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=40,
        help="Frames to settle per viewpoint before capture (default: 40)",
    )
    return parser.parse_args()


def _viewpoint_progress(idx: int, total: int, vp: Viewpoint, status: str) -> None:
    if status == "start":
        print(f"  [{idx}/{total}] {vp.name}")
    elif status == "done":
        print(f"    -> {vp.name}.png")
    elif status == "failed":
        print(f"    WARNING: {vp.name} failed")


def main() -> None:
    args = _parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    scene_usd = scene_dir / "scene.usd"
    if not scene_usd.exists():
        print(f"ERROR: {scene_usd} not found")
        sys.exit(1)

    try:
        w, h = (int(x) for x in args.resolution.split("x"))
    except ValueError:
        print(f"ERROR: invalid --resolution {args.resolution!r}, expected WxH")
        sys.exit(1)

    scene_config_path = scene_dir / "scene.json"
    scene_config: dict = (
        json.loads(scene_config_path.read_text())
        if scene_config_path.exists() else {}
    )

    output_dir = scene_dir / "renders"

    session = IsaacRenderSession(
        headless=args.headless,
        resolution=(w, h),
        warm_up_frames=args.warm_up,
        settle_frames=args.settle_frames,
    )

    print("[1/4] Starting SimulationApp ...")
    session.boot()

    try:
        print(f"[2/4] Opening {scene_usd} ...")
        stage = session.open_stage(scene_usd)

        print("[3/4] Warming up renderer ...")
        session.prepare_render_product()

        center, size = compute_scene_bbox(stage)
        print(f"  Scene center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"  Scene size:   ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")

        viewpoints = auto_orbit_viewpoints(center, size, scene_config.get("camera"))

        print(f"[4/4] Rendering {len(viewpoints)} viewpoints at {w}x{h} ...")
        result = session.render_viewpoints(
            viewpoints, output_dir, progress=_viewpoint_progress,
        )

        index = {
            "scene_id": scene_config.get("sample_id", scene_dir.name),
            "resolution": f"{w}x{h}",
            "images": [f"{name}.png" for name in result.rendered],
            "total_rendered": len(result.rendered),
            "total_failed": len(result.failed),
        }
        (output_dir / "render_index.json").write_text(json.dumps(index, indent=2))

        print(
            f"\nDone. {len(result.rendered)}/{len(viewpoints)} rendered, "
            f"{len(result.failed)} failed"
        )
        print(f"Output: {output_dir}/")
    finally:
        session.close()


if __name__ == "__main__":
    main()
