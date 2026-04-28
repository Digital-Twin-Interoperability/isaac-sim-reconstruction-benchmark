"""Reusable Isaac Sim rendering session.

The :class:`IsaacRenderSession` wraps ``SimulationApp`` lifecycle, stage
loading, camera orbit, and RGB capture so that both the standalone render
script (``scripts/render_scene.py``) and the agentic scene-reconstruction
loop can share the same rendering machinery.

Importing this subpackage does **not** boot Isaac Sim — that happens lazily
on :meth:`IsaacRenderSession.boot`.  So it is safe to import from pure
uv-Python contexts.  Actually booting or rendering requires the process to
be running under ``isaac-sim-standalone-*/python.sh`` (or ``python.bat``).
"""

from isaacsim_bench.renderer.session import (
    IsaacRenderSession,
    RenderResult,
    Viewpoint,
    add_scene_lighting,
    auto_orbit_viewpoints,
    compute_scene_bbox,
    create_or_move_camera,
)

__all__ = [
    "IsaacRenderSession",
    "RenderResult",
    "Viewpoint",
    "add_scene_lighting",
    "auto_orbit_viewpoints",
    "compute_scene_bbox",
    "create_or_move_camera",
]
