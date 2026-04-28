"""Render tool for the agentic loop.

Kept in a separate module so ``agentic.py`` stays pure-Python when the
caller disables rendering; importing this module must only happen under
Isaac Sim's ``python.sh``.

The render tool materialises the current :class:`PredictionJSON` to a temp
USD file, opens it in an :class:`IsaacRenderSession`, renders a chosen set
of viewpoints, and returns PNG paths that get fed back to the driver as
inline images.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from isaacsim_bench.agents.agentic_tools import AgentState, ToolSpec
from isaacsim_bench.agents.composer import ComposerAgent
from isaacsim_bench.agents.vlm import ToolResult
from isaacsim_bench.renderer.session import IsaacRenderSession, Viewpoint

logger = logging.getLogger(__name__)

# By default the render tool returns a small, diverse set of views per call
# to keep context costs reasonable.  The model can override via `viewpoints`.
DEFAULT_RENDER_VIEWPOINTS = ["scene_default", "front_high", "top_down"]
# `scene_default` mirrors the input camera (read by the reconstructor from
# scene.json) so the agent can compare directly against the reference's
# scene_default.png.  It is only generated when state.scene_camera is set;
# otherwise that viewpoint silently falls through to no-op.
ALLOWED_VIEWPOINTS = {
    "scene_default",
    "front_high", "front_right", "right", "back_right", "back",
    "back_left", "left", "front_left", "top_down", "low_angle",
}
DEFAULT_RENDER_RESOLUTION = (768, 768)


def make_default_renderer(
    *,
    headless: bool = True,
    resolution: tuple[int, int] = DEFAULT_RENDER_RESOLUTION,
) -> IsaacRenderSession:
    """Create an un-booted session.  Boot is lazy on first render."""
    return IsaacRenderSession(headless=headless, resolution=resolution)


def _scene_hash(state: AgentState) -> str:
    """Stable hash of the current scene prediction for render caching."""
    parts = []
    for c in sorted(state.prediction.components, key=lambda x: x.name):
        pos = tuple(round(v, 3) for v in c.translate)
        quat = tuple(round(v, 3) for v in c.orientation_xyzw)
        parts.append(f"{c.name}|{c.asset_id}|{pos}|{quat}")
    digest = hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]
    return digest


def _materialize_usd(state: AgentState, out_path: Path) -> None:
    """Write current PredictionJSON to a USD file for the renderer to load."""
    ComposerAgent.export_usd(state.prediction, out_path, state.taxonomy)


def _render_viewpoints_from_prediction(
    state: AgentState, viewpoints: list[str],
) -> tuple[dict[str, Path], int]:
    """Render *viewpoints* of the current prediction.

    Returns ``(name → png_path, freshly_rendered_count)``.  The second
    element lets the caller decide whether to charge the call against the
    render budget — a fully-cached call should be free.
    """
    session: IsaacRenderSession = state.renderer
    if session is None:
        raise RuntimeError("No renderer attached to AgentState")

    scene_key = _scene_hash(state)
    cached = state.render_cache.get(scene_key)
    if cached is not None:
        cached_by_name = {p.stem: p for p in cached}
        need = [v for v in viewpoints if v not in cached_by_name]
        if not need:
            return {v: cached_by_name[v] for v in viewpoints}, 0
    else:
        cached_by_name = {}
        need = list(viewpoints)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"agentic_render_{scene_key}_"))
    usd_path = tmp_dir / "scene.usd"

    # Boot Isaac Sim BEFORE materialising the USD: ``ComposerAgent.export_usd``
    # imports ``pxr`` lazily, and Isaac Sim's bundled pxr only becomes
    # importable once SimulationApp has started.  Doing it the other way
    # round causes any pip-installed usd-core to win and Kit's extensions
    # then crash on namespace mismatch.
    session.boot()
    _materialize_usd(state, usd_path)
    session.open_stage(usd_path)
    session.prepare_render_product()

    from isaacsim_bench.renderer.session import (
        auto_orbit_viewpoints,
        compute_scene_bbox,
    )

    center, size = compute_scene_bbox(session.stage)
    # Pass the input camera through so `scene_default` is rendered at the
    # same pose as the reference's scene_default.png — making compare-to-
    # reference apples-to-apples.
    all_vps = {
        vp.name: vp
        for vp in auto_orbit_viewpoints(center, size, state.scene_camera)
    }

    to_render: list[Viewpoint] = []
    for name in need:
        vp = all_vps.get(name)
        if vp is None:
            logger.warning(
                "Viewpoint %r not available "
                "(scene_default needs state.scene_camera set) — skipping",
                name,
            )
            continue
        to_render.append(vp)

    if not to_render:
        return (
            {v: cached_by_name[v] for v in viewpoints if v in cached_by_name},
            0,
        )

    result = session.render_viewpoints(to_render, tmp_dir)

    # Update cache and return
    all_paths = list(cached_by_name.values()) + [
        result.output_paths[n] for n in result.rendered
    ]
    state.render_cache[scene_key] = all_paths

    combined = {**cached_by_name, **result.output_paths}
    return (
        {v: combined[v] for v in viewpoints if v in combined},
        len(result.rendered),
    )


def handle_render(args: dict, state: AgentState) -> ToolResult:
    if not state.prediction.components:
        return ToolResult(
            tool_use_id="",
            text="Scene is empty — add at least one component before rendering.",
            is_error=True,
        )

    requested = args.get("viewpoints") or DEFAULT_RENDER_VIEWPOINTS
    if not isinstance(requested, list) or not all(isinstance(v, str) for v in requested):
        return ToolResult(
            tool_use_id="",
            text="viewpoints must be a list of strings.",
            is_error=True,
        )
    unknown = [v for v in requested if v not in ALLOWED_VIEWPOINTS]
    if unknown:
        return ToolResult(
            tool_use_id="",
            text=(
                f"Unknown viewpoint names {unknown}. "
                f"Allowed: {sorted(ALLOWED_VIEWPOINTS)}."
            ),
            is_error=True,
        )

    try:
        rendered, fresh_count = _render_viewpoints_from_prediction(state, requested)
    except Exception as exc:
        logger.exception("Render tool failed")
        return ToolResult(
            tool_use_id="",
            text=f"Render failed: {exc}",
            is_error=True,
        )

    # Only count against the render budget when the renderer actually did
    # work.  Fully-cached calls are free so the agent can re-inspect old
    # renders without burning budget.
    if fresh_count > 0:
        state.render_count += 1

    # Any successful render — including fully-cached ones — clears the
    # "scene modified since last render" flag.  Cached renders are valid
    # because the cache key is the scene hash, so a cache hit means the
    # current scene has been rendered before.
    if rendered:
        state.scene_modified_since_render = False

    image_paths = [rendered[v] for v in requested if v in rendered]
    failed = [v for v in requested if v not in rendered]

    if fresh_count > 0:
        budget_line = (
            f"Rendered {fresh_count} new viewpoint(s); "
            f"render {state.render_count} of budget."
        )
    else:
        budget_line = "All requested viewpoints served from cache (no budget used)."

    text_lines = [
        f"Returned {len(image_paths)}/{len(requested)} viewpoints. {budget_line}",
    ]
    if failed:
        text_lines.append(f"  Skipped viewpoints: {failed}")
    text_lines.append("  Images are attached; compare them to the references.")

    return ToolResult(
        tool_use_id="",
        text="\n".join(text_lines),
        image_paths=image_paths,
    )


def build_render_tool() -> ToolSpec:
    return ToolSpec(
        name="render",
        description=(
            "Render the current scene-under-construction from one or more "
            "camera viewpoints and return the images for inspection. "
            "Use this to verify placement and compare to the reference images."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "viewpoints": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": sorted(ALLOWED_VIEWPOINTS),
                    },
                    "description": (
                        "Which camera angles to render.  Default is "
                        "[front_high, right, top_down].  More views cost more "
                        "tokens; pick only what you need."
                    ),
                },
            },
        },
        handler=handle_render,
    )
