"""Agentic scene reconstructor — tool-calling loop on top of :class:`VLMClient`.

Sits alongside the fixed-pipeline :class:`~isaacsim_bench.agents.orchestrator.SceneReconstructor`
as the "agentic baseline".  A single VLM driver decides which tools to call
(search, place components, modify, render, submit) and iterates until it
invokes ``submit_prediction`` or hits the turn/token/render budget.

The loop itself never consumes ground truth — termination is always the
driver's decision or a budget, so the same loop can run in real-world
deployments without GT available.  Benchmark-side evaluation happens
downstream from the returned :class:`PredictionJSON`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from isaacsim_bench.agents.agentic_tools import (
    AgentState,
    ToolSpec,
    default_tool_specs,
)
from isaacsim_bench.agents.vlm import ToolCall, ToolResult, TurnResult, VLMClient
from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

DEFAULT_MAX_TURNS = 40
DEFAULT_MAX_RENDERS = 10
DEFAULT_MAX_TOTAL_TOKENS = 500_000


def build_system_prompt(*, render_enabled: bool) -> str:
    """Compose the system prompt from the tools that are actually wired in.

    The fixed-pipeline reconstructor never sees this — it's only for the
    agentic loop.  Render-related guidance is omitted when the render tool
    is disabled to avoid teaching the model about a tool that doesn't exist.
    """
    lines = [
        "You are a scene reconstruction agent for Isaac Sim.",
        "",
        "You will be given reference images of a warehouse-style scene and a",
        "catalog of assets.  Your task is to rebuild the scene by placing",
        "assets at the correct positions and orientations so that a "
        "re-rendering of your prediction would match the references.",
        "",
        "## How to work",
        "",
        "1. **Inventory first — before any placement.** Look at every reference",
        "   image and write a numbered list of *every distinct physical object*",
        "   you can see, with a rough description and rough location.  This is",
        "   your **starting hypothesis** — update it whenever a render reveals",
        "   you missed something.  The dominant CLIP match is almost never the",
        "   whole scene; secondary objects (a straight section beside a curved",
        "   one, a pallet under a shelf, etc.) are exactly what gets missed.",
        "   When in doubt, count up.  An object that looks like one continuous",
        "   piece in a render is often actually several modular assets placed",
        "   end-to-end — conveyors, shelving, racks all compose this way.",
        "2. Use `get_catalog_hits` for a fast visual shortlist, plus",
        "   `list_families` / `list_assets_in_family` to browse the taxonomy.",
        "   Treat the top-1 hit as a hint about ONE object, not as a description",
        "   of the whole scene.  Search separately for each inventory item.",
        "3. Confirm an asset's identity with `get_asset_info` before placing it.",
        "   The response includes two things you should look for first:",
        "     - `anchors:` a list of named connection points in the asset's",
        "       local frame (e.g. `origin`, `anchorpoint`, plus aliases like",
        "       `in`, `out`, `curve_entry`, `curve_exit`, `right_end`).  When",
        "       present, these are authored attach points — use them.",
        "     - `bbox_local: min={...} max={...}` — the asset's geometry in",
        "       its own local frame.  Use this only when the asset has no",
        "       anchors (pallets, racks, boxes).",
        "4. **For modular assets (anything with anchors), chain via",
        "   `add_aligned_component`.**  This places a new piece *and* snaps",
        "   one of its anchors to an existing component's anchor in a single",
        "   edit — no need to invent a placeholder pose.  Direction:",
        "   `fixed --[attach]--> new`.",
        "   **Anchors mate tip-to-tail.** Each join must pair one `in` face",
        "   with one `out` face — never `in`↔`in` or `out`↔`out`. `in` is",
        "   the entry face, `out` is the exit face, so flow goes through one",
        "   piece's `out` into the next piece's `in`. The same applies to the",
        "   raw names `origin`↔`anchorpoint` (`in` aliases `origin`; `out`",
        "   aliases `anchorpoint`): pair `origin` on one side with",
        "   `anchorpoint` on the other, never the same name on both sides.",
        "   Example for a U-curve (curve in the middle, two straights):",
        "     - place the curve first: `add_component(name='curve', ...,",
        "       position=(0,0,0))`;",
        "     - straight feeding INTO the curve: `add_aligned_component(",
        "       name='in_straight', ..., fixed_component='curve',",
        "       fixed_anchor='in', moving_anchor='out')` — the upstream",
        "       piece's `out` mates with the curve's `in`;",
        "     - straight feeding OUT of the curve: `add_aligned_component(",
        "       name='out_straight', ..., fixed_component='curve',",
        "       fixed_anchor='out', moving_anchor='in')` — the curve's",
        "       `out` mates with the downstream piece's `in`.",
        "   Both arms come out parallel, on opposite faces of the curve.",
        "   The `facing` parameter defaults to `same_frame`, which is the",
        "   correct mate for Isaac's anchor convention (the anchorpoint",
        "   already encodes the destination pose).  Use `opposed_frame` only",
        "   when you've actually verified back-to-back facing is needed.",
        "   Use `align_components` to re-snap an existing pair after a",
        "   `modify_component`, and `get_component_anchors` to read out a",
        "   component's anchor poses in WORLD coordinates when planning.",
        "5. **For non-modular assets (no anchors — pallets, racks, boxes),",
        "   fall back to `add_component`** with explicit positions in meters",
        "   (Z-up, right-handed) and quaternions `[x, y, z, w]`.  Compute",
        "   placement from each asset's `bbox_local` — the goal is",
        "   `componentA.bbox_max == componentB.translate + componentB.bbox_min`",
        "   along the shared axis (after rotation).",
    ]
    if render_enabled:
        lines += [
            "6. **Edit one thing, then render.** Each scene edit —",
            "   `add_component`, `add_aligned_component`, `align_components`,",
            "   `modify_component`, `remove_component` — must be followed by",
            "   a `render` before the next edit.  Back-to-back edits without",
            "   an intervening render are rejected.  The point is so you can",
            "   tell, after each change, whether you made the scene better",
            "   or worse.  **After every render, explicitly list inventory",
            "   items still unaccounted for** before doing anything else.",
            "   **Add before swap.** If a render reproduces the dominant",
            "   shape but is missing peripheral structural pieces (end",
            "   supports, abutting sections, posts, an extra row), the",
            "   reference is much more likely *several components placed",
            "   adjacent* than *one different variant* — try",
            "   `add_aligned_component` (or `add_component`) for the missing",
            "   piece before reaching for `modify_component`.  Use",
            "   `modify_component` / `remove_component` for genuine",
            "   asset-identity or pose mistakes.  Renders that don't change",
            "   the scene are served from cache for free — don't hesitate to",
            "   re-inspect.",
            "   **For positioning, also call `plot_top_down`** alongside",
            "   render.  It returns a 2D schematic with a 1-meter grid and",
            "   labeled axes, so you can read each component's position in",
            "   coordinates instead of eyeballing a perspective image.  Pair",
            "   it with the reference `top_down.png` to compare layouts",
            "   directly.  Free and always available — does not satisfy the",
            "   render-after-edit rule, but is the best tool for picking the",
            "   *next* coordinates.",
            "7. When every inventory item is placed and the rendered output",
            "   matches the references, call `submit_prediction` with",
            "   `expected_components` equal to your *final* inventory count.",
        ]
    else:
        lines += [
            "6. Inspect placements via `list_components` and adjust them with",
            "   `modify_component` / `remove_component` based on the references.",
            "   (No render tool is available in this run.)",
            "7. When every inventory item is placed, call `submit_prediction`",
            "   with `expected_components` equal to your inventory count.",
        ]
    lines += [
        "",
        "## Constraints",
        "",
        "- Only use `asset_id` values that are in the retrieval pool.",
        "  Attempts to place an asset outside the pool will be rejected.",
        "- Use unique, descriptive `name` values for each component (e.g.",
        "  `conveyor_left`, `box_on_shelf_1`).",
        "- You have budgets for turns" + (
            ", rendering," if render_enabled else ""
        ) + " and tokens.  Be efficient, but do not submit a half-built scene",
        "  just to terminate; finish the inventory.",
        "- `submit_prediction` is gated.  It will reject the call if your",
        "  inventory count doesn't match the placed-component count" + (
            ", or if you've changed the scene since the last render."
            if render_enabled else "."
        ),
        "  If after honest effort you cannot find a usable asset for an",
        "  inventory item, set `acknowledge_unmatched=true` and explain in",
        "  `notes` — that's the proper escape hatch.",
        "- Every turn must end with a tool call — even if you are done,",
        "  call `submit_prediction` rather than only outputting text.",
    ]
    return "\n".join(lines)


# Kept as a module-level convenience for callers that don't toggle render.
SYSTEM_PROMPT = build_system_prompt(render_enabled=True)


@dataclass
class AgenticRunResult:
    prediction: PredictionJSON
    submitted: bool
    turns_used: int
    renders_used: int
    stop_reason: str
    notes: str = ""
    tool_call_log: list[dict] = field(default_factory=list)


def run_agentic_loop(
    vlm: VLMClient,
    state: AgentState,
    tools: list[ToolSpec],
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_renders: int = DEFAULT_MAX_RENDERS,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
    system_prompt: str = SYSTEM_PROMPT,
) -> AgenticRunResult:
    """Drive the loop.

    *state* must already have input_image_paths (and optionally catalog_hits)
    populated.  Tool handlers read/write *state.prediction* in place.
    """
    tool_map = {t.name: t for t in tools}
    tool_schemas = [t.to_anthropic_schema() for t in tools]

    messages: list[dict] = [_build_initial_user_message(vlm, state)]
    tool_call_log: list[dict] = []

    stop_reason = "unknown"
    no_tool_streak = 0  # consecutive turns where the model only produced text

    for turn in range(1, max_turns + 1):
        state.turn = turn
        logger.info("=== Agentic turn %d/%d ===", turn, max_turns)

        if vlm.usage.total_tokens >= max_total_tokens:
            stop_reason = "token_budget"
            logger.warning(
                "Token budget reached: %d tokens used", vlm.usage.total_tokens,
            )
            break

        # If the model just monologued last turn, force it to actually use
        # a tool this turn — otherwise the loop can stall on text alone.
        tool_choice = "any" if no_tool_streak >= 1 else "auto"

        try:
            result: TurnResult = vlm.run_turn(
                system=system_prompt,
                messages=messages,
                tools=tool_schemas,
                max_tokens=4096,
                tool_choice=tool_choice,
            )
        except Exception:
            logger.exception("VLM turn %d failed irrecoverably", turn)
            stop_reason = "vlm_error"
            break

        messages.append(result.assistant_message)

        if result.text.strip():
            logger.info("Driver: %s", result.text.strip()[:300])

        if not result.tool_calls:
            no_tool_streak += 1
            if no_tool_streak >= 2:
                # Even after a forced tool_choice="any", the model didn't
                # call anything — give up rather than spin forever.
                stop_reason = "no_tool_calls"
                logger.warning(
                    "Driver produced text-only on %d consecutive turns — ending loop.",
                    no_tool_streak,
                )
                break
            # Nudge the model and retry next turn with tool_choice="any".
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        "Please proceed by calling a tool — search, place a "
                        "component, render (if available), or call "
                        "`submit_prediction` if you are done."
                    ),
                }],
            })
            continue
        no_tool_streak = 0

        tool_results: list[ToolResult] = []
        for call in result.tool_calls:
            tr = _dispatch_tool_call(
                call, tool_map, state, max_renders=max_renders,
            )
            tr.tool_use_id = call.id
            tool_results.append(tr)
            tool_call_log.append({
                "turn": turn,
                "tool": call.name,
                "arguments": call.arguments,
                "is_error": tr.is_error,
                "text_preview": tr.text[:200],
                "images": len(tr.image_paths),
            })
            logger.info(
                "  tool=%s ok=%s text=%s",
                call.name, not tr.is_error, tr.text[:120].replace("\n", " | "),
            )

        vlm.append_tool_results(messages, tool_results)

        if state.submitted:
            stop_reason = "submitted"
            logger.info("Driver submitted prediction — ending loop.")
            break
    else:
        stop_reason = "turn_budget"
        logger.warning("Turn budget reached (%d turns)", max_turns)

    return AgenticRunResult(
        prediction=state.prediction,
        submitted=state.submitted,
        turns_used=state.turn,
        renders_used=state.render_count,
        stop_reason=stop_reason,
        notes=state.submit_notes,
        tool_call_log=tool_call_log,
    )


def _dispatch_tool_call(
    call: ToolCall,
    tool_map: dict[str, ToolSpec],
    state: AgentState,
    *,
    max_renders: int,
) -> ToolResult:
    spec = tool_map.get(call.name)
    if spec is None:
        return ToolResult(
            tool_use_id=call.id,
            text=f"Unknown tool {call.name!r}.",
            is_error=True,
        )

    # Render budget is enforced before dispatch, not inside the handler.
    if call.name == "render" and state.render_count >= max_renders:
        return ToolResult(
            tool_use_id=call.id,
            text=(
                f"Render budget exhausted ({max_renders} renders used). "
                "Work from the renders you already have, or submit."
            ),
            is_error=True,
        )

    try:
        return spec.handler(call.arguments, state)
    except Exception as exc:
        logger.exception("Tool %r handler raised", call.name)
        return ToolResult(
            tool_use_id=call.id,
            text=f"Tool {call.name} raised: {exc}",
            is_error=True,
        )


def _build_initial_user_message(vlm: VLMClient, state: AgentState) -> dict:
    """Show the agent every reference image upfront — the agent has no other
    way to fetch additional ones aside from the optional `view_reference`
    tool, and silently dropping reference views was a real source of
    reconstruction errors."""
    images = list(state.input_image_paths)
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"## Reference images ({len(images)} provided)\n"
                "These are the views of the scene you must reconstruct."
            ),
        },
    ]
    for p in images:
        content.append(vlm.encode_image(p))
        content.append({"type": "text", "text": f"_{p.name}_"})

    pool_summary = _pool_summary(state)
    content.append({
        "type": "text",
        "text": (
            f"\n## Retrieval pool\n{pool_summary}\n\n"
            "Use the tools below to search for assets, place them, and (if a "
            "render tool is available) compare against the references.  "
            "Start by inspecting the images and calling `get_catalog_hits` "
            "for a visual shortlist."
        ),
    })
    return {"role": "user", "content": content}


def _pool_summary(state: AgentState) -> str:
    fams: dict[str, int] = {}
    for vid in state.retrieval_pool_ids:
        cat = state._variant_to_category.get(vid)
        if cat:
            fams[cat.family] = fams.get(cat.family, 0) + 1
    lines = [f"{len(state.retrieval_pool_ids)} assets across {len(fams)} families:"]
    for fam, n in sorted(fams.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {fam}: {n}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# High-level reconstructor
# ----------------------------------------------------------------------------

class AgenticReconstructor:
    """Agentic counterpart to :class:`SceneReconstructor`.

    Drives a tool-calling loop instead of a fixed pipeline.  Returns a
    :class:`PredictionJSON` with identical shape, so downstream evaluation
    works unchanged.
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        taxonomy_path: Path | None = None,
        retrieval_pool_path: Path | None = None,
        thumbnail_dir: Path | None = None,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_renders: int = DEFAULT_MAX_RENDERS,
        max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
        enable_render: bool = False,
        clip_top_k: int = 15,
        renderer_factory=None,
    ) -> None:
        self.vlm = VLMClient(model=model, provider=provider)
        self.max_turns = max_turns
        self.max_renders = max_renders
        self.max_total_tokens = max_total_tokens
        self.enable_render = enable_render
        self.clip_top_k = clip_top_k
        self.renderer_factory = renderer_factory

        tax_path = taxonomy_path or DATA_DIR / "asset_taxonomy.json"
        pool_path = retrieval_pool_path or DATA_DIR / "retrieval_pool.json"
        thumb_dir = thumbnail_dir or DATA_DIR / "asset_thumbnails"

        self.taxonomy = AssetTaxonomy.model_validate_json(tax_path.read_text())
        pool_data = json.loads(pool_path.read_text())
        self.retrieval_pool_ids: set[str] = set(pool_data["asset_ids"])
        self.thumbnail_dir = thumb_dir if thumb_dir.exists() else None

        # Optional precomputed asset bboxes (data/asset_extents.json).  When
        # present, ``get_asset_info`` surfaces dimensions to the agent so it
        # can chain modular pieces end-to-end with arithmetic instead of
        # eyeballing positions from renders.
        extents_path = DATA_DIR / "asset_extents.json"
        self.asset_extents: dict[str, list[float]] = {}
        if extents_path.exists():
            try:
                self.asset_extents = json.loads(extents_path.read_text())
                logger.info(
                    "Loaded extents for %d assets from %s",
                    len(self.asset_extents), extents_path.name,
                )
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    "Failed to parse %s; agent will run without extent info",
                    extents_path,
                )

        # Optional anchor registry — extracted /World/Anchorpoint poses
        # (data/asset_anchors.json) overlaid by hand-authored aliases and
        # validity flags (data/asset_anchors_overrides.json).  When present,
        # `align_components` does deterministic SE(3) chaining instead of
        # asking the agent to do quaternion math.
        from isaacsim_bench.schemas.anchors import (
            AnchorRegistry,
            load_anchor_registry,
        )

        anchors_path = DATA_DIR / "asset_anchors.json"
        overrides_path = DATA_DIR / "asset_anchors_overrides.json"
        self.asset_anchors: AnchorRegistry | None = None
        if anchors_path.exists():
            try:
                self.asset_anchors = load_anchor_registry(
                    anchors_path,
                    overrides_path if overrides_path.exists() else None,
                )
                logger.info(
                    "Loaded anchors for %d assets from %s",
                    len(self.asset_anchors.asset_ids), anchors_path.name,
                )
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Failed to parse %s; agent will run without anchors: %s",
                    anchors_path, e,
                )

        self.clip_catalog = None
        if self.thumbnail_dir:
            try:
                from isaacsim_bench.agents.clip_catalog import CLIPCatalog

                self.clip_catalog = CLIPCatalog(
                    thumbnail_dir=self.thumbnail_dir,
                    taxonomy_path=tax_path,
                )
            except ImportError:
                logger.warning(
                    "CLIP catalog unavailable; agent will rely on taxonomy browsing.",
                )

    def reconstruct(
        self,
        image_dir: Path,
        scene_id: str | None = None,
    ) -> tuple[PredictionJSON, AgenticRunResult]:
        image_paths = _collect_valid_images(image_dir)
        sid = scene_id or image_dir.parent.name

        # Optional: pick up the input camera so render(scene_default) is at
        # the same pose as the reference's `scene_default.png`.  Only the
        # camera field is used — components/relations are not consulted.
        scene_camera = _try_load_input_camera(image_dir)

        catalog_hits: list = []
        if self.clip_catalog:
            logger.info("Running CLIP catalog on %d images …", len(image_paths))
            catalog_hits = self.clip_catalog.query(
                image_paths,
                top_k=self.clip_top_k,
                must_include_ids=self.retrieval_pool_ids,
            )

        prediction = PredictionJSON(sample_id=sid, components=[], relations=[])
        state = AgentState(
            prediction=prediction,
            taxonomy=self.taxonomy,
            retrieval_pool_ids=self.retrieval_pool_ids,
            input_image_paths=image_paths,
            catalog_hits=catalog_hits,
            scene_camera=scene_camera,
            asset_extents=self.asset_extents,
            asset_anchors=self.asset_anchors,
        )

        tools = default_tool_specs()
        if self.enable_render:
            from isaacsim_bench.agents.agentic_render import (
                build_render_tool,
                make_default_renderer,
            )

            factory = self.renderer_factory or make_default_renderer
            state.renderer = factory()
            tools.append(build_render_tool())

        system_prompt = build_system_prompt(render_enabled=self.enable_render)

        t0 = time.time()
        result = run_agentic_loop(
            self.vlm, state, tools,
            max_turns=self.max_turns,
            max_renders=self.max_renders,
            max_total_tokens=self.max_total_tokens,
            system_prompt=system_prompt,
        )
        elapsed = time.time() - t0
        prediction.latency_seconds = elapsed

        # NB: the renderer is intentionally NOT closed here.  Kit's shutdown
        # path can hard-exit the Python process (sys.exit / os._exit), which
        # would skip any prediction-saving the caller wants to do.  The
        # caller is expected to call :meth:`close` *after* saving artifacts.
        self._state = state

        logger.info(
            "Agentic run: stop=%s turns=%d renders=%d tokens=%d time=%.1fs",
            result.stop_reason, result.turns_used,
            result.renders_used, self.vlm.usage.total_tokens, elapsed,
        )
        return prediction, result

    def close(self) -> None:
        """Shut down the renderer if one was created.  Best-effort; safe to
        call even when render tool was disabled or already closed."""
        state = getattr(self, "_state", None)
        if state is None or state.renderer is None:
            return
        try:
            state.renderer.close()
        except Exception:
            logger.warning("Failed to close renderer cleanly", exc_info=True)
        state.renderer = None


def _try_load_input_camera(image_dir: Path) -> dict | None:
    """Return the input rendering camera if a sibling ``scene.json`` is present.

    Only ``scene.json["camera"]`` is read — never components or relations —
    so this does not couple the loop to ground truth.  It exists so that
    when the render tool is enabled the agent's self-render of
    ``scene_default`` matches the pose of the reference ``scene_default.png``.
    """
    candidates = [
        image_dir / "scene.json",
        image_dir.parent / "scene.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cam = data.get("camera")
        if isinstance(cam, dict) and "position" in cam and "target" in cam:
            logger.info("Loaded input camera from %s", path)
            return {
                "position": list(cam["position"]),
                "target": list(cam["target"]),
                "fov_deg": float(cam.get("fov_deg", 60.0)),
            }
    return None


def _collect_valid_images(image_dir: Path) -> list[Path]:
    MIN_IMAGE_BYTES = 10_000
    all_images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not all_images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    valid = [p for p in all_images if p.stat().st_size >= MIN_IMAGE_BYTES]
    if not valid:
        raise FileNotFoundError(
            f"All {len(all_images)} images in {image_dir} appear blank",
        )
    if len(valid) < len(all_images):
        logger.info(
            "Filtered %d blank images (< %d bytes)",
            len(all_images) - len(valid), MIN_IMAGE_BYTES,
        )
    return valid
