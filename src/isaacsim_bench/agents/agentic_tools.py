"""Tools for the agentic scene-reconstruction loop.

Each tool is a :class:`ToolSpec` — name, JSON-schema, and a handler function
that mutates :class:`AgentState` and returns a :class:`ToolResult` (text +
optional inline images).

The tools in this module are provider-agnostic; they do not touch Isaac Sim.
The render tool lives separately and is wired in only when running under
``python.sh``.
"""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pydantic import ValidationError

from isaacsim_bench.agents.vlm import ToolResult
from isaacsim_bench.geom.transforms import (
    align_pose,
    pose_compose,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from isaacsim_bench.schemas.anchors import AnchorRegistry
from isaacsim_bench.schemas.prediction import (
    PredictedComponent,
    PredictedRelation,
    PredictionJSON,
)
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# State + tool typing
# ----------------------------------------------------------------------------

@dataclass
class InventoryItem:
    """One distinct physical object the agent has counted in the references.

    ``matched_components`` is a list (1:N) so an inventory item like
    "row of pallets" can be tied to multiple placed components.  An item
    is considered satisfied for the submit gate iff this list is non-empty.
    """

    item_id: str
    description: str
    rough_xy: tuple[float, float] | None = None
    matched_components: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def matched(self) -> bool:
        return bool(self.matched_components)


@dataclass
class AgentState:
    """Mutable state shared across tool calls in one agentic run."""

    prediction: PredictionJSON
    taxonomy: AssetTaxonomy
    retrieval_pool_ids: set[str]
    input_image_paths: list[Path] = field(default_factory=list)
    catalog_hits: list[Any] = field(default_factory=list)  # list[CatalogHit]

    # Inventory baseline — the agent's structured count of distinct objects
    # in the reference images.  Locked by ``set_inventory`` (one-shot); after
    # that, edits to the inventory must go through update / add / mark.
    # The hard gate on add/modify/align edit handlers refuses to run until
    # ``inventory_locked`` is True, so the agent can't drift into placement
    # without first committing to a count.
    inventory: list[InventoryItem] = field(default_factory=list)
    inventory_locked: bool = False

    # Optional input-camera config (position/target/fov_deg).  Populated
    # by the reconstructor from a sibling ``scene.json``; consumed only by
    # the render tool to align ``scene_default`` self-renders with the
    # reference ``scene_default.png``.
    scene_camera: dict[str, Any] | None = None

    # Optional bbox metadata per asset, derived from each variant's USD::
    #
    #     {variant_id: {extent_xyz, bbox_min, bbox_max}}
    #
    # Both ``extent_xyz`` (size) and the local-frame bbox-min/max are
    # surfaced by ``get_asset_info``.  Knowing the bbox-min/max — not just
    # the size — is what lets the agent compute "where does this asset's
    # other end land in scene coordinates", which is essential for chaining
    # modular pieces (a curve + two straights into a U) end-to-end.
    asset_extents: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    # Optional anchor registry (per-asset connection points).  When present,
    # ``get_component_anchors`` and ``align_components`` use this for
    # deterministic rigid-body chaining.  Loaded from
    # ``data/asset_anchors.json`` overlaid by ``asset_anchors_overrides.json``.
    asset_anchors: AnchorRegistry | None = None

    # Render tool state (populated externally when wired)
    renderer: Any = None  # IsaacRenderSession | None
    render_count: int = 0
    render_cache: dict[str, list[Path]] = field(default_factory=dict)

    # Submit-gate bookkeeping: flips True on any add/modify/remove and back
    # to False once a successful render has been produced for the new state.
    # Initial value is False so an empty scene cannot be submitted (the
    # placed-vs-expected check fails) but isn't blocked solely by this flag.
    scene_modified_since_render: bool = False

    # Termination
    submitted: bool = False
    submit_notes: str = ""

    # Bookkeeping
    turn: int = 0

    # Cached indexes (built in __post_init__)
    _variant_to_category: dict[str, Any] = field(default_factory=dict, repr=False)
    _family_to_categories: dict[str, list[Any]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for cat in self.taxonomy.categories:
            self._family_to_categories.setdefault(cat.family, []).append(cat)
            for var in cat.variants:
                self._variant_to_category[var.variant_id] = cat


ToolHandler = Callable[[dict[str, Any], AgentState], ToolResult]


@dataclass
class ToolSpec:
    """A tool the agent can invoke in the loop."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ----------------------------------------------------------------------------
# Helpers shared by handlers
# ----------------------------------------------------------------------------

def _ok(text: str, *, images: list[Path] | None = None) -> ToolResult:
    return ToolResult(tool_use_id="", text=text, image_paths=images or [])


def _err(msg: str) -> ToolResult:
    return ToolResult(tool_use_id="", text=msg, is_error=True)


def _check_edit_serialization(state: AgentState, op: str) -> ToolResult | None:
    """Block back-to-back edits without an intervening render.

    Each component edit (add / modify / remove) must be followed by a
    `render` before the next edit, so the agent can see whether its last
    change made things better or worse.  Without this, modifies tend to
    batch — agent moves two components, renders once, can no longer
    attribute which move helped.

    Skipped when the render tool isn't wired in (state.renderer is None) —
    in that mode there's no rendered feedback to wait for.
    """
    if state.renderer is None:
        return None
    if not state.scene_modified_since_render:
        return None
    return _err(
        f"Edit blocked: you {op}d a component since the last render.  "
        "Call `render` first to see the effect of your previous change "
        "before making another edit — that's how you tell whether your "
        "last move improved the scene or made it worse.",
    )


def _component_index(state: AgentState) -> dict[str, int]:
    return {c.name: i for i, c in enumerate(state.prediction.components)}


def _inventory_index(state: AgentState) -> dict[str, InventoryItem]:
    return {item.item_id: item for item in state.inventory}


def _check_inventory_locked(state: AgentState, op: str) -> ToolResult | None:
    """Hard gate: refuse edits until ``set_inventory`` has been called.

    The agent must commit to a count of distinct objects in the references
    before any placement work, so the submit gate has a baseline to check
    against.  Without this, the count drifts silently turn-to-turn.
    """
    if state.inventory_locked:
        return None
    return _err(
        f"Cannot {op}: inventory not set.  Call `set_inventory` first with "
        "a numbered list of every distinct physical object you can see in "
        "the reference images, then come back to placement.",
    )


def _format_inventory_status(state: AgentState) -> list[str]:
    """Render a 'matched X / total Y' rollup with per-item status lines.

    Used by ``list_components`` and ``render`` so the agent never has to
    re-derive the inventory status from prose.  Returns text lines (no
    leading newline); caller decides where to splice them in.
    """
    if not state.inventory_locked:
        return [
            "Inventory: not set.  Call `set_inventory` first to record what "
            "you see in the reference images.",
        ]
    if not state.inventory:
        return ["Inventory: locked but empty (0 items)."]
    total = len(state.inventory)
    matched = sum(1 for item in state.inventory if item.matched)
    lines = [f"Inventory status: {matched}/{total} matched"]
    for item in state.inventory:
        if item.matched:
            comps = ", ".join(item.matched_components)
            lines.append(f"  ✓ {item.item_id}  → {comps}")
        else:
            xy = ""
            if item.rough_xy is not None:
                rx, ry = item.rough_xy
                xy = f"  rough_xy=({rx:.2f}, {ry:.2f})"
            lines.append(f"  ✗ {item.item_id}  unmatched  ({item.description}){xy}")
    return lines


def _format_component(c: PredictedComponent) -> str:
    x, y, z = (round(v, 3) for v in c.translate)
    qx, qy, qz, qw = (round(v, 3) for v in c.orientation_xyzw)
    return (
        f"- {c.name}: asset={c.asset_id} family={c.family} "
        f"pos=({x}, {y}, {z}) quat=({qx}, {qy}, {qz}, {qw}) "
        f"conf={c.confidence:.2f}"
    )


# ----------------------------------------------------------------------------
# Inventory handlers
# ----------------------------------------------------------------------------

def _coerce_rough_xy(value: Any) -> tuple[float, float] | None | str:
    """Parse an optional rough_xy field.

    Returns the tuple, ``None`` if absent, or an error string if malformed.
    Errors are returned (not raised) so handlers can wrap them in ToolResult.
    """
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        return "rough_xy must be a [x, y] list of 2 numbers (meters)."
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return "rough_xy entries must be numbers."


def handle_set_inventory(args: dict, state: AgentState) -> ToolResult:
    """Record the agent's inventory of distinct objects — one-shot.

    Once committed, the inventory is locked: edits go through
    ``update_inventory_item`` (revise existing) or ``add_inventory_item``
    (explicit post-lock addition with a reason).  This is the trigger that
    unlocks all the scene-edit handlers; before this call they refuse to run.
    """
    if state.inventory_locked:
        return _err(
            "Inventory is already locked.  Use `update_inventory_item` to "
            "revise existing entries, or `add_inventory_item` to add a new "
            "one with a reason.",
        )
    items_arg = args.get("items")
    if not isinstance(items_arg, list) or not items_arg:
        return _err(
            "items must be a non-empty list of objects with item_id "
            "and description fields.",
        )

    parsed: list[InventoryItem] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(items_arg):
        if not isinstance(raw, dict):
            return _err(f"items[{i}] must be an object.")
        item_id = (raw.get("item_id") or "").strip()
        description = (raw.get("description") or "").strip()
        if not item_id:
            return _err(f"items[{i}].item_id is required and must be non-empty.")
        if not description:
            return _err(f"items[{i}].description is required and must be non-empty.")
        if item_id in seen_ids:
            return _err(f"Duplicate item_id {item_id!r} at items[{i}].")
        seen_ids.add(item_id)
        rough_xy = _coerce_rough_xy(raw.get("rough_xy"))
        if isinstance(rough_xy, str):
            return _err(f"items[{i}]: {rough_xy}")
        notes = (raw.get("notes") or "").strip()
        parsed.append(InventoryItem(
            item_id=item_id,
            description=description,
            rough_xy=rough_xy,
            notes=notes,
        ))

    state.inventory = parsed
    state.inventory_locked = True
    lines = [
        f"Inventory locked with {len(parsed)} item(s).  You can now place "
        "components.  After placing each, call `mark_matched` to tie the "
        "component to its inventory item.",
    ]
    for item in parsed:
        lines.append(f"  - {item.item_id}: {item.description}")
    return _ok("\n".join(lines))


def handle_mark_matched(args: dict, state: AgentState) -> ToolResult:
    """Tie a placed component to an inventory item (1:N, no auto-match)."""
    if not state.inventory_locked:
        return _err(
            "Inventory not set.  Call `set_inventory` before mark_matched.",
        )
    item_id = (args.get("item_id") or "").strip()
    component = (args.get("component") or "").strip()
    if not item_id or not component:
        return _err("Both item_id and component are required.")

    inv_idx = _inventory_index(state)
    if item_id not in inv_idx:
        return _err(
            f"Unknown item_id {item_id!r}.  Existing items: "
            f"{sorted(inv_idx.keys())}",
        )
    if component not in _component_index(state):
        return _err(
            f"No component named {component!r}.  Place it first with "
            "`add_component` or `add_aligned_component`.",
        )
    item = inv_idx[item_id]
    if component in item.matched_components:
        return _ok(
            f"{component!r} is already matched to inventory item "
            f"{item_id!r} — no change.",
        )
    item.matched_components.append(component)
    return _ok(
        f"Matched component {component!r} to inventory item {item_id!r}. "
        f"{item_id!r} now has {len(item.matched_components)} matched "
        "component(s).",
    )


def handle_unmark_matched(args: dict, state: AgentState) -> ToolResult:
    """Remove a component link from an inventory item.

    With ``component`` omitted, clears all matches for the item — useful
    when the agent decides to redo placement for that item from scratch.
    """
    if not state.inventory_locked:
        return _err("Inventory not set.")
    item_id = (args.get("item_id") or "").strip()
    if not item_id:
        return _err("item_id is required.")
    inv_idx = _inventory_index(state)
    if item_id not in inv_idx:
        return _err(f"Unknown item_id {item_id!r}.")
    item = inv_idx[item_id]
    component = (args.get("component") or "").strip()
    if not component:
        n = len(item.matched_components)
        item.matched_components.clear()
        return _ok(f"Cleared {n} match(es) from inventory item {item_id!r}.")
    if component not in item.matched_components:
        return _err(
            f"Component {component!r} is not matched to {item_id!r}.",
        )
    item.matched_components.remove(component)
    return _ok(
        f"Unmatched {component!r} from inventory item {item_id!r}.",
    )


def handle_update_inventory_item(args: dict, state: AgentState) -> ToolResult:
    """Revise description / rough_xy / notes of an existing inventory item."""
    if not state.inventory_locked:
        return _err("Inventory not set.")
    item_id = (args.get("item_id") or "").strip()
    if not item_id:
        return _err("item_id is required.")
    inv_idx = _inventory_index(state)
    if item_id not in inv_idx:
        return _err(f"Unknown item_id {item_id!r}.")
    item = inv_idx[item_id]

    changed: list[str] = []
    if (description := args.get("description")) is not None:
        description = description.strip()
        if not description:
            return _err("description cannot be empty.")
        item.description = description
        changed.append(f"description={description!r}")
    if "rough_xy" in args:
        rough_xy = _coerce_rough_xy(args.get("rough_xy"))
        if isinstance(rough_xy, str):
            return _err(rough_xy)
        item.rough_xy = rough_xy
        changed.append(f"rough_xy={item.rough_xy}")
    if (notes := args.get("notes")) is not None:
        item.notes = notes.strip()
        changed.append(f"notes={item.notes!r}")

    if not changed:
        return _err("No fields to update provided.")
    return _ok(f"Updated inventory item {item_id!r}: " + ", ".join(changed))


def handle_add_inventory_item(args: dict, state: AgentState) -> ToolResult:
    """Add a new inventory item AFTER the initial lock.

    Requires ``reason`` so post-lock additions are auditable — the whole
    point of locking is to surface drift; silent additions defeat that.
    """
    if not state.inventory_locked:
        return _err(
            "Inventory not set.  Use `set_inventory` for the initial baseline.",
        )
    item_id = (args.get("item_id") or "").strip()
    description = (args.get("description") or "").strip()
    reason = (args.get("reason") or "").strip()
    if not item_id:
        return _err("item_id is required.")
    if not description:
        return _err("description is required.")
    if not reason:
        return _err(
            "reason is required for post-lock additions — explain why this "
            "item was missed in the initial inventory.  Adding without a "
            "reason defeats the purpose of locking the count.",
        )
    inv_idx = _inventory_index(state)
    if item_id in inv_idx:
        return _err(
            f"Inventory item {item_id!r} already exists.  Use "
            "`update_inventory_item` to revise it.",
        )

    rough_xy = _coerce_rough_xy(args.get("rough_xy"))
    if isinstance(rough_xy, str):
        return _err(rough_xy)

    state.inventory.append(InventoryItem(
        item_id=item_id,
        description=description,
        rough_xy=rough_xy,
        notes=f"[post-lock] {reason}",
    ))
    return _ok(
        f"Added inventory item {item_id!r} (post-lock).  Reason recorded: "
        f"{reason!r}.  Inventory now has {len(state.inventory)} item(s).",
    )


# ----------------------------------------------------------------------------
# Tool handlers
# ----------------------------------------------------------------------------

def handle_list_families(args: dict, state: AgentState) -> ToolResult:
    families = sorted(state._family_to_categories.keys())
    pool_families = set()
    for vid in state.retrieval_pool_ids:
        cat = state._variant_to_category.get(vid)
        if cat:
            pool_families.add(cat.family)
    lines = [f"{len(families)} families in taxonomy:"]
    for fam in families:
        marker = "*" if fam in pool_families else " "
        n = sum(len(c.variants) for c in state._family_to_categories[fam])
        lines.append(f"  {marker} {fam} ({n} variants)")
    lines.append("  * = family has at least one asset in the retrieval pool.")
    return _ok("\n".join(lines))


def handle_list_assets_in_family(args: dict, state: AgentState) -> ToolResult:
    family = args.get("family", "").strip()
    pool_only = bool(args.get("pool_only", True))
    cats = state._family_to_categories.get(family)
    if not cats:
        return _err(
            f"Unknown family {family!r}. Use list_families to see valid values.",
        )
    lines = [f"Family {family!r}:"]
    for cat in cats:
        lines.append(f"  category {cat.category_id} — {cat.name}")
        for var in cat.variants:
            in_pool = var.variant_id in state.retrieval_pool_ids
            if pool_only and not in_pool:
                continue
            marker = "*" if in_pool else " "
            lines.append(f"    {marker} {var.variant_id:30s}  {var.name}")
    if pool_only:
        lines.append("  (pool_only=True — set pool_only=False to see all)")
    return _ok("\n".join(lines))


def handle_get_asset_info(args: dict, state: AgentState) -> ToolResult:
    asset_id = args.get("asset_id", "").strip()
    cat = state._variant_to_category.get(asset_id)
    if not cat:
        return _err(f"Unknown asset_id {asset_id!r}.")
    var = next((v for v in cat.variants if v.variant_id == asset_id), None)
    if not var:
        return _err(f"asset_id {asset_id!r} not found in category.")
    in_pool = asset_id in state.retrieval_pool_ids
    lines = [
        f"asset_id: {asset_id}",
        f"name: {var.name}",
        f"family: {cat.family}",
        f"category: {cat.category_id} ({cat.name})",
        f"in_retrieval_pool: {in_pool}",
    ]
    if var.usd_path:
        lines.append(f"usd_path: {var.usd_path}")
    if var.semantic_class:
        lines.append(f"semantic_class: {var.semantic_class}")
    bbox = state.asset_extents.get(asset_id)
    if bbox:
        ext = bbox.get("extent_xyz")
        bmin = bbox.get("bbox_min")
        bmax = bbox.get("bbox_max")
        if ext and len(ext) == 3:
            ex, ey, ez = (round(float(v), 3) for v in ext)
            lines.append(f"extent_xyz: [{ex}, {ey}, {ez}] meters  (size)")
        if bmin and bmax and len(bmin) == 3 and len(bmax) == 3:
            lo = [round(float(v), 3) for v in bmin]
            hi = [round(float(v), 3) for v in bmax]
            lines.append(
                f"bbox_local: min={lo}  max={hi}  "
                "— this asset's geometry, in its own local frame.  When you "
                "place it with translate=(0,0,0), the asset extends from "
                "min to max.  Critical for chaining: if min_y=0 and max_y=4.93 "
                "then placing a neighbour at translate=(0, 4.93, 0) abuts the "
                "two assets at the y=4.93 plane.",
            )

    # Anchors: print canonical raw rows + alias metadata.  Use these instead
    # of bbox arithmetic for any modular asset that has them — align_components
    # snaps deterministically.
    if state.asset_anchors is not None and asset_id in state.asset_anchors:
        rec = state.asset_anchors.get_record(asset_id)
        if rec is not None and rec.anchors:
            aliases_by_target: dict[str, list[str]] = {}
            for alias, raw in rec.aliases.items():
                aliases_by_target.setdefault(raw, []).append(alias)
            lines.append("anchors (in asset's local frame):")
            for raw_name, pose in rec.anchors.items():
                pos = [round(float(v), 4) for v in pose.position]
                q = [round(float(v), 4) for v in pose.orient_wxyz]
                aliases = aliases_by_target.get(raw_name, [])
                tag = f"  ({', '.join(sorted(aliases))})" if aliases else ""
                valid_tag = "" if pose.valid else "  [INVALID — do not use]"
                lines.append(
                    f"  {raw_name}: pos={pos} orient_wxyz={q}{tag}{valid_tag}",
                )
            lines.append(
                "  Use `align_components` to chain modular pieces by "
                "anchor — it computes the rigid transform deterministically.",
            )
    return _ok("\n".join(lines))


def handle_get_catalog_hits(args: dict, state: AgentState) -> ToolResult:
    top_k = int(args.get("top_k", 15))
    if not state.catalog_hits:
        return _ok(
            "No CLIP catalog hits available (CLIP catalog unavailable or not "
            "precomputed). Use list_families / list_assets_in_family instead.",
        )
    hits = state.catalog_hits[:top_k]
    lines = [
        f"Top {len(hits)} CLIP visual matches for the reference images:",
        "  (higher score = more visually similar)",
    ]
    for h in hits:
        lines.append(
            f"  {h.variant_id:30s}  family={h.family:10s}  "
            f"score={h.score:.3f}  ({h.category_name})"
        )
    return _ok("\n".join(lines))


def handle_list_components(args: dict, state: AgentState) -> ToolResult:
    comps = state.prediction.components
    if not comps:
        lines = ["Scene is empty — no components placed yet."]
        lines.extend(_format_inventory_status(state))
        return _ok("\n".join(lines))
    lines = [f"Current scene ({len(comps)} components):"]
    for c in comps:
        lines.append(_format_component(c))
    if state.prediction.relations:
        lines.append(f"Relations ({len(state.prediction.relations)}):")
        for r in state.prediction.relations:
            lines.append(
                f"  {r.from_node} --[{r.type}]--> {r.to_node}  "
                f"({r.from_anchor} -> {r.to_anchor})"
            )
    lines.append("")
    lines.extend(_format_inventory_status(state))
    return _ok("\n".join(lines))


def _validate_item_id_for_add(
    state: AgentState, item_id: str | None,
) -> tuple[str | None, ToolResult | None]:
    """Pre-validate an optional ``item_id`` before doing an add.

    Returns ``(normalized_item_id, None)`` on success (or when item_id is
    absent), or ``(None, ToolResult)`` to short-circuit with an error.  Done
    upfront so a doomed add doesn't leave a stranded component before the
    auto-match step fails.
    """
    if item_id is None or not str(item_id).strip():
        return None, None
    iid = str(item_id).strip()
    if iid not in _inventory_index(state):
        return None, _err(
            f"item_id {iid!r} is not in the inventory. Existing items: "
            f"{sorted(_inventory_index(state).keys())}",
        )
    return iid, None


def handle_add_component(args: dict, state: AgentState) -> ToolResult:
    if (gate := _check_inventory_locked(state, "add a component")) is not None:
        return gate
    # NOTE: add ops are intentionally exempt from the render-after-edit gate.
    # The gate is for modify/align/remove where attribution of "did my change
    # help?" matters per-step; adds with confident pose can batch and let the
    # agent render once at the end.  See system prompt step 7.
    name = args.get("name", "").strip()
    asset_id = args.get("asset_id", "").strip()
    position = args.get("position")
    orientation = args.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])
    confidence = float(args.get("confidence", 0.8))
    item_id, err = _validate_item_id_for_add(state, args.get("item_id"))
    if err is not None:
        return err

    if not name:
        return _err("name is required and must be non-empty.")
    if name in _component_index(state):
        return _err(
            f"Component named {name!r} already exists. Use modify_component "
            "to change it, or pick a different name.",
        )

    cat = state._variant_to_category.get(asset_id)
    if not cat:
        return _err(
            f"Unknown asset_id {asset_id!r}. Use search tools to find valid IDs.",
        )
    if asset_id not in state.retrieval_pool_ids:
        return _err(
            f"asset_id {asset_id!r} is not in the retrieval pool. "
            "Only pool assets are allowed.",
        )

    if not isinstance(position, list) or len(position) != 3:
        return _err("position must be a [x, y, z] list of 3 floats (meters).")
    if not isinstance(orientation, list) or len(orientation) != 4:
        return _err(
            "orientation_xyzw must be a [x, y, z, w] quaternion list of 4 floats.",
        )

    try:
        comp = PredictedComponent(
            name=name,
            asset_id=asset_id,
            family=cat.family,
            translate=[float(v) for v in position],
            orientation_xyzw=[float(v) for v in orientation],
            confidence=confidence,
        )
    except ValidationError as e:
        return _err(f"Invalid component: {e}")

    state.prediction.components.append(comp)
    state.scene_modified_since_render = True

    msg = (
        f"Added component {name!r} (asset={asset_id}). "
        f"Scene now has {len(state.prediction.components)} components."
    )
    if item_id is not None:
        _inventory_index(state)[item_id].matched_components.append(name)
        msg += f"  Auto-matched to inventory item {item_id!r}."
    return _ok(msg)


def handle_modify_component(args: dict, state: AgentState) -> ToolResult:
    if (gate := _check_inventory_locked(state, "modify a component")) is not None:
        return gate
    if (gate := _check_edit_serialization(state, "modified")) is not None:
        return gate
    name = args.get("name", "").strip()
    idx_map = _component_index(state)
    if name not in idx_map:
        return _err(f"No component named {name!r}.")
    comp = state.prediction.components[idx_map[name]]

    changed: list[str] = []

    if (asset_id := args.get("asset_id")) is not None:
        asset_id = asset_id.strip()
        cat = state._variant_to_category.get(asset_id)
        if not cat:
            return _err(f"Unknown asset_id {asset_id!r}.")
        if asset_id not in state.retrieval_pool_ids:
            return _err(f"asset_id {asset_id!r} is not in the retrieval pool.")
        comp.asset_id = asset_id
        comp.family = cat.family
        changed.append(f"asset_id={asset_id}")

    if (position := args.get("position")) is not None:
        if not isinstance(position, list) or len(position) != 3:
            return _err("position must be a [x, y, z] list of 3 floats.")
        comp.translate = [float(v) for v in position]
        changed.append(f"position={comp.translate}")

    if (orientation := args.get("orientation_xyzw")) is not None:
        if not isinstance(orientation, list) or len(orientation) != 4:
            return _err("orientation_xyzw must be a 4-float list.")
        comp.orientation_xyzw = [float(v) for v in orientation]
        changed.append(f"orientation={comp.orientation_xyzw}")

    if (confidence := args.get("confidence")) is not None:
        comp.confidence = float(confidence)
        changed.append(f"confidence={comp.confidence:.2f}")

    if not changed:
        return _err("No fields to modify provided.")
    state.scene_modified_since_render = True
    return _ok(f"Modified {name!r}: " + ", ".join(changed))


def handle_remove_component(args: dict, state: AgentState) -> ToolResult:
    # NOTE: removes — like adds — are exempt from the render-after-edit
    # gate.  A remove is a deterministic-intent action; per-step
    # attribution ("did this delete help?") is not the question.  Modify
    # and align still receive the gate (they're "did this change make
    # things better?" style).  Submit still requires a clean
    # render-after-edit before terminating.
    name = args.get("name", "").strip()
    idx_map = _component_index(state)
    if name not in idx_map:
        return _err(f"No component named {name!r}.")
    del state.prediction.components[idx_map[name]]
    # Drop relations referencing this component
    state.prediction.relations = [
        r for r in state.prediction.relations
        if r.from_node != name and r.to_node != name
    ]
    # Auto-unmatch from any inventory item — leaving stale references would
    # let the submit gate think the item is satisfied even though its only
    # backing component just got deleted.
    affected: list[str] = []
    for item in state.inventory:
        if name in item.matched_components:
            item.matched_components.remove(name)
            affected.append(item.item_id)
    state.scene_modified_since_render = True
    msg = (
        f"Removed {name!r}. Scene now has "
        f"{len(state.prediction.components)} components."
    )
    if affected:
        msg += (
            f"  Auto-unmatched from inventory item(s): {affected}.  "
            "Re-match a replacement or these items will block submit."
        )
    return _ok(msg)


def handle_add_relation(args: dict, state: AgentState) -> ToolResult:
    names = _component_index(state)
    rel_type = args.get("type", "").strip()
    from_node = args.get("from_node", "").strip()
    to_node = args.get("to_node", "").strip()
    from_anchor = args.get("from_anchor", "").strip()
    to_anchor = args.get("to_anchor", "").strip()

    if from_node not in names:
        return _err(f"from_node {from_node!r} does not exist.")
    if to_node not in names:
        return _err(f"to_node {to_node!r} does not exist.")
    if not rel_type:
        return _err("type is required.")

    try:
        state.prediction.relations.append(PredictedRelation(
            type=rel_type,
            from_node=from_node,
            to_node=to_node,
            from_anchor=from_anchor,
            to_anchor=to_anchor,
        ))
    except ValidationError as e:
        return _err(f"Invalid relation: {e}")
    return _ok(
        f"Added relation {from_node} --[{rel_type}]--> {to_node} "
        f"({from_anchor} -> {to_anchor}).",
    )


def handle_list_references(args: dict, state: AgentState) -> ToolResult:
    paths = state.input_image_paths
    if not paths:
        return _ok("No reference images attached to this run.")
    lines = [f"{len(paths)} reference images available:"]
    for p in paths:
        lines.append(f"  - {p.name}")
    lines.append(
        "Call `view_reference` with one of these names to re-attach an "
        "image to the conversation (useful if it was compacted out).",
    )
    return _ok("\n".join(lines))


def handle_view_reference(args: dict, state: AgentState) -> ToolResult:
    name = args.get("name", "").strip()
    if not name:
        return _err("name is required.")
    match = next((p for p in state.input_image_paths if p.name == name), None)
    if match is None:
        avail = [p.name for p in state.input_image_paths]
        return _err(f"No reference image named {name!r}. Available: {avail}")
    return _ok(f"Re-attaching reference image {name}.", images=[match])


def _quat_z_angle_rad(quat_xyzw: list[float]) -> float:
    """Extract the rotation angle about Z from an [x, y, z, w] quaternion.

    Top-down placement only cares about the Z component of the rotation —
    other axes tilt the asset out of the floor plane and are uncommon in
    warehouse scenes.
    """
    qz, qw = quat_xyzw[2], quat_xyzw[3]
    return 2.0 * math.atan2(qz, qw)


def _render_top_down_schematic(state: AgentState, image_px: int = 1024) -> Path:
    """Draw a 2D top-down schematic of the current scene to a temp PNG.

    Always available, free of any render budget — this is just PIL drawing
    based on ``state.prediction`` + ``state.asset_extents``.  Each component
    is drawn as a footprint rectangle (XY extent rotated by the Z component
    of its quaternion) at its translate, with a grid marked every 1 m and
    every 5 m, axis labels in meters, and the component name burned in.
    """
    margin_m = 2.0
    if state.prediction.components:
        xs: list[float] = []
        ys: list[float] = []
        for c in state.prediction.components:
            bbox = state.asset_extents.get(c.asset_id) or {}
            ext = bbox.get("extent_xyz") or [1.0, 1.0, 1.0]
            # Use the asset's full diagonal so the autoscale never crops a
            # rotated bbox.
            half_diag = math.hypot(ext[0], ext[1]) / 2.0
            cx, cy, _ = c.translate
            xs.extend([cx - half_diag, cx + half_diag])
            ys.extend([cy - half_diag, cy + half_diag])
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
    else:
        x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0

    cx_m = (x_min + x_max) / 2.0
    cy_m = (y_min + y_max) / 2.0
    span = max(x_max - x_min, y_max - y_min, 4.0) + 2 * margin_m
    px_per_m = image_px / span

    def world_to_px(x: float, y: float) -> tuple[int, int]:
        # Image y grows downward; world y grows upward.
        u = (x - cx_m) * px_per_m + image_px / 2.0
        v = image_px / 2.0 - (y - cy_m) * px_per_m
        return int(u), int(v)

    img = Image.new("RGB", (image_px, image_px), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Determine integer-meter range we need to draw gridlines for.
    half_span = span / 2.0
    m_lo = int(math.floor(cx_m - half_span)) - 1
    m_hi = int(math.ceil(cx_m + half_span)) + 1
    n_lo = int(math.floor(cy_m - half_span)) - 1
    n_hi = int(math.ceil(cy_m + half_span)) + 1

    # 1 m gridlines (light), 5 m gridlines (slightly darker).
    for m in range(m_lo, m_hi + 1):
        u, _ = world_to_px(m, 0)
        if 0 <= u <= image_px:
            color = (175, 175, 175) if m % 5 == 0 else (220, 220, 220)
            draw.line([(u, 0), (u, image_px)], fill=color, width=1)
    for n in range(n_lo, n_hi + 1):
        _, v = world_to_px(0, n)
        if 0 <= v <= image_px:
            color = (175, 175, 175) if n % 5 == 0 else (220, 220, 220)
            draw.line([(0, v), (image_px, v)], fill=color, width=1)

    # Origin axes: X red, Y green.
    ox, oy = world_to_px(0, 0)
    if 0 <= oy <= image_px:
        draw.line([(0, oy), (image_px, oy)], fill=(220, 50, 50), width=2)
    if 0 <= ox <= image_px:
        draw.line([(ox, 0), (ox, image_px)], fill=(50, 180, 50), width=2)

    # Numeric labels along axes (every meter; bigger every 5 m).
    for m in range(m_lo, m_hi + 1):
        u, _ = world_to_px(m, 0)
        if 0 <= u <= image_px and m != 0:
            label = f"{m}"
            v = oy + 4 if 0 <= oy <= image_px else 4
            draw.text((u + 2, v), label, fill=(120, 120, 120))
    for n in range(n_lo, n_hi + 1):
        _, v = world_to_px(0, n)
        if 0 <= v <= image_px and n != 0:
            label = f"{n}"
            u = ox + 4 if 0 <= ox <= image_px else 4
            draw.text((u, v + 2), label, fill=(120, 120, 120))

    # Components: rotated footprint rectangle + name + origin dot.
    # Use bbox_min/max (in the asset's local frame, before rotation) when
    # available, so the rectangle reflects the asset's true footprint
    # relative to its origin — not just an extent-centered approximation.
    for comp in state.prediction.components:
        bbox = state.asset_extents.get(comp.asset_id) or {}
        bmin = bbox.get("bbox_min")
        bmax = bbox.get("bbox_max")
        if bmin and bmax and len(bmin) >= 2 and len(bmax) >= 2:
            x0, y0 = float(bmin[0]), float(bmin[1])
            x1, y1 = float(bmax[0]), float(bmax[1])
        else:
            ext = bbox.get("extent_xyz") if bbox else None
            sx, sy = (float(ext[0]), float(ext[1])) if ext else (1.0, 1.0)
            x0, y0, x1, y1 = -sx / 2.0, -sy / 2.0, sx / 2.0, sy / 2.0
        cx, cy, _ = comp.translate
        theta = _quat_z_angle_rad(comp.orientation_xyzw)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        local = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        world = [
            (cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t)
            for dx, dy in local
        ]
        pixels = [world_to_px(x, y) for x, y in world]
        # Outline only — fill draws over the grid and hides labels.
        draw.polygon(pixels, outline=(30, 80, 200), width=2)

        ux, vy = world_to_px(cx, cy)
        draw.ellipse([ux - 3, vy - 3, ux + 3, vy + 3], fill=(200, 0, 0))
        draw.text((ux + 5, vy - 12), comp.name, fill=(30, 30, 30))

    # Title / scale annotation in the top-left corner.
    draw.text(
        (8, 8),
        f"top-down schematic — 1 grid square = 1 m  (axes: X red, Y green)",
        fill=(40, 40, 40),
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="schematic_"))
    out_path = tmp_dir / "top_down_schematic.png"
    img.save(out_path)
    return out_path


def handle_plot_top_down(args: dict, state: AgentState) -> ToolResult:
    """Return a 2D top-down schematic of the placed components.

    Free (no render budget consumed) and always available — does not require
    the render tool to be wired in.  The schematic shows component bboxes
    with labels in scene coordinates so the agent can read offsets directly
    instead of guessing position deltas from a perspective render.
    """
    if not state.prediction.components:
        return _err(
            "No components to plot.  Add at least one with add_component "
            "before requesting a top-down schematic.",
        )
    out = _render_top_down_schematic(state)
    n = len(state.prediction.components)
    return _ok(
        f"Top-down schematic of the current scene ({n} components). "
        "Read positions directly from the gridlines (1 grid square = 1 m). "
        "Compare against the reference top-down view to see whether each "
        "component sits in the right place.",
        images=[out],
    )


def _component_world_pose(
    comp: PredictedComponent,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Read a component's world pose in (pos, quat_wxyz) form."""
    pos = (float(comp.translate[0]), float(comp.translate[1]), float(comp.translate[2]))
    quat_xyzw = (
        float(comp.orientation_xyzw[0]),
        float(comp.orientation_xyzw[1]),
        float(comp.orientation_xyzw[2]),
        float(comp.orientation_xyzw[3]),
    )
    return pos, xyzw_to_wxyz(quat_xyzw)


def handle_get_component_anchors(args: dict, state: AgentState) -> ToolResult:
    """Return world-frame anchor poses for a placed component.

    Canonical raw anchors only (no alias rows); aliases shown as metadata.
    Invalid anchors are still returned with ``valid=false`` so the agent can
    see why they're unusable, but ``align_components`` rejects them.
    """
    name = args.get("component", "").strip()
    if not name:
        return _err("`component` is required.")
    idx = _component_index(state)
    if name not in idx:
        return _err(f"No component named {name!r}.")
    comp = state.prediction.components[idx[name]]

    if state.asset_anchors is None or comp.asset_id not in state.asset_anchors:
        return _err(
            f"No anchor metadata for asset {comp.asset_id!r}.  This asset "
            "isn't in the anchor registry — fall back to bbox/extent reasoning.",
        )
    rec = state.asset_anchors.get_record(comp.asset_id)
    assert rec is not None  # guarded above

    comp_pos, comp_quat = _component_world_pose(comp)
    aliases_by_target: dict[str, list[str]] = {}
    for alias, raw in rec.aliases.items():
        aliases_by_target.setdefault(raw, []).append(alias)

    lines = [
        f"Anchors for component {name!r} (asset={comp.asset_id}) in WORLD frame:",
    ]
    for raw_name, pose in rec.anchors.items():
        local_pos = (
            float(pose.position[0]),
            float(pose.position[1]),
            float(pose.position[2]),
        )
        local_quat = (
            float(pose.orient_wxyz[0]),
            float(pose.orient_wxyz[1]),
            float(pose.orient_wxyz[2]),
            float(pose.orient_wxyz[3]),
        )
        world_pos, world_quat_wxyz = pose_compose(
            comp_pos, comp_quat, local_pos, local_quat,
        )
        world_quat_xyzw = wxyz_to_xyzw(world_quat_wxyz)
        aliases = sorted(aliases_by_target.get(raw_name, []))
        alias_tag = f"  aliases={aliases}" if aliases else ""
        valid_tag = "" if pose.valid else "  [INVALID — align_components will reject]"
        lines.append(
            f"  {raw_name}: pos={[round(v, 4) for v in world_pos]} "
            f"orient_xyzw={[round(v, 4) for v in world_quat_xyzw]}"
            f"{alias_tag}{valid_tag}",
        )
    return _ok("\n".join(lines))


def _resolve_align_anchors(
    state: AgentState,
    fixed: PredictedComponent,
    moving_asset_id: str,
    fixed_anchor: str,
    moving_anchor: str,
) -> tuple[Any, Any] | ToolResult:
    """Resolve and validate both anchors; return ToolResult on any error."""
    if state.asset_anchors is None:
        return _err(
            "No anchor registry loaded — alignment is unavailable. "
            "Place the component with explicit pose instead.",
        )
    fa = state.asset_anchors.resolve(fixed.asset_id, fixed_anchor)
    ma = state.asset_anchors.resolve(moving_asset_id, moving_anchor)
    if fa is None:
        names = state.asset_anchors.list_names(fixed.asset_id)
        return _err(
            f"Unknown anchor {fixed_anchor!r} on asset {fixed.asset_id!r}. "
            f"Available: {names}",
        )
    if ma is None:
        names = state.asset_anchors.list_names(moving_asset_id)
        return _err(
            f"Unknown anchor {moving_anchor!r} on asset {moving_asset_id!r}. "
            f"Available: {names}",
        )
    if not fa.valid:
        return _err(
            f"fixed_anchor {fixed_anchor!r} on {fixed.asset_id!r} is marked "
            "INVALID — its authored pose is degenerate. Pick a different "
            "anchor or place this asset with explicit pose.",
        )
    if not ma.valid:
        return _err(
            f"moving_anchor {moving_anchor!r} on {moving_asset_id!r} is "
            "marked INVALID — pick a different anchor or place explicitly.",
        )
    # Tip-to-tail check: same canonical anchor on both sides means the
    # two pieces' identical faces collide at the join, which collapses
    # them on top of each other.  `origin` and `anchorpoint` are the two
    # opposite faces of an asset's bed; mating must pair opposites
    # (origin↔anchorpoint, equivalently in↔out).
    fixed_rec = state.asset_anchors.get_record(fixed.asset_id)
    moving_rec = state.asset_anchors.get_record(moving_asset_id)

    def _canon(rec: "AssetAnchors", name: str) -> str:  # noqa: F821
        return name if name in rec.anchors else rec.aliases.get(name, name)

    fa_canon = _canon(fixed_rec, fixed_anchor)
    ma_canon = _canon(moving_rec, moving_anchor)
    if fa_canon == ma_canon:
        return _err(
            f"Anchor mating {fixed_anchor!r} ↔ {moving_anchor!r} resolves "
            f"to the same canonical anchor {fa_canon!r} on both sides. "
            "Anchors mate tip-to-tail: pair `origin` with `anchorpoint` "
            "(equivalently `in` with `out`), never the same name on "
            "both sides — `origin` and `anchorpoint` are the two opposite "
            "faces of the asset's bed, so same-name mating collapses the "
            "pieces onto the same face and they overlap.",
        )
    return fa, ma


def _compute_aligned_pose(
    fixed: PredictedComponent,
    fa: "AnchorPose",  # noqa: F821 — runtime type only
    ma: "AnchorPose",  # noqa: F821
    facing: str,
) -> tuple[list[float], list[float]]:
    """Return (translate, orientation_xyzw) for the moving component."""
    fixed_pos, fixed_quat = _component_world_pose(fixed)
    fa_local_pos = (
        float(fa.position[0]), float(fa.position[1]), float(fa.position[2]),
    )
    fa_local_quat = (
        float(fa.orient_wxyz[0]), float(fa.orient_wxyz[1]),
        float(fa.orient_wxyz[2]), float(fa.orient_wxyz[3]),
    )
    ma_local_pos = (
        float(ma.position[0]), float(ma.position[1]), float(ma.position[2]),
    )
    ma_local_quat = (
        float(ma.orient_wxyz[0]), float(ma.orient_wxyz[1]),
        float(ma.orient_wxyz[2]), float(ma.orient_wxyz[3]),
    )
    new_pos, new_quat_wxyz = align_pose(
        fixed_pos, fixed_quat,
        fa_local_pos, fa_local_quat,
        ma_local_pos, ma_local_quat,
        facing=facing,
    )
    new_quat_xyzw = wxyz_to_xyzw(new_quat_wxyz)
    return (
        [round(float(v), 6) for v in new_pos],
        [round(float(v), 6) for v in new_quat_xyzw],
    )


def _upsert_relation(
    state: AgentState,
    relation_type: str,
    fixed_name: str,
    moving_name: str,
    fixed_anchor: str,
    moving_anchor: str,
) -> str:
    """Add or update an attach-style relation; return a status word."""
    existing = next(
        (
            r for r in state.prediction.relations
            if r.type == relation_type
            and r.from_node == fixed_name
            and r.to_node == moving_name
        ),
        None,
    )
    if existing is not None:
        existing.from_anchor = fixed_anchor
        existing.to_anchor = moving_anchor
        return "updated existing relation"
    state.prediction.relations.append(PredictedRelation(
        type=relation_type,
        from_node=fixed_name,
        to_node=moving_name,
        from_anchor=fixed_anchor,
        to_anchor=moving_anchor,
    ))
    return "added relation"


def handle_align_components(args: dict, state: AgentState) -> ToolResult:
    """Snap one component to another's anchor with a full SE(3) transform.

    Sets the moving component's translate + orientation_xyzw so that its
    anchor frame coincides with the fixed component's anchor frame (with
    optional ``Rz(pi)`` mate for ``opposed_frame``).  Also records or updates
    a relation ``fixed --[type]--> moving`` with the anchor names.

    Invalid anchors are rejected loudly — silent fallback to bbox math is
    exactly what we're trying to eliminate.
    """
    if (gate := _check_inventory_locked(state, "align components")) is not None:
        return gate
    if (gate := _check_edit_serialization(state, "aligned")) is not None:
        return gate

    fixed_name = args.get("fixed_component", "").strip()
    moving_name = args.get("moving_component", "").strip()
    fixed_anchor = args.get("fixed_anchor", "").strip()
    moving_anchor = args.get("moving_anchor", "").strip()
    facing = args.get("facing", "same_frame")
    relation_type = args.get("relation_type", "attach").strip() or "attach"

    if not fixed_name or not moving_name:
        return _err("fixed_component and moving_component are required.")
    if not fixed_anchor or not moving_anchor:
        return _err("fixed_anchor and moving_anchor are required.")
    if fixed_name == moving_name:
        return _err("fixed_component and moving_component must differ.")
    if facing not in ("same_frame", "opposed_frame"):
        return _err(
            f"facing must be 'same_frame' or 'opposed_frame', got {facing!r}.",
        )

    idx = _component_index(state)
    if fixed_name not in idx:
        return _err(f"No component named {fixed_name!r}.")
    if moving_name not in idx:
        return _err(f"No component named {moving_name!r}.")

    fixed = state.prediction.components[idx[fixed_name]]
    moving = state.prediction.components[idx[moving_name]]

    resolved = _resolve_align_anchors(
        state, fixed, moving.asset_id, fixed_anchor, moving_anchor,
    )
    if isinstance(resolved, ToolResult):
        return resolved
    fa, ma = resolved

    new_translate, new_orient = _compute_aligned_pose(fixed, fa, ma, facing)
    moving.translate = new_translate
    moving.orientation_xyzw = new_orient
    state.scene_modified_since_render = True

    rel_msg = _upsert_relation(
        state, relation_type, fixed_name, moving_name, fixed_anchor, moving_anchor,
    )

    pos_str = [round(v, 4) for v in moving.translate]
    quat_str = [round(v, 4) for v in moving.orientation_xyzw]
    return _ok(
        f"Aligned {moving_name!r}.{moving_anchor} -> "
        f"{fixed_name!r}.{fixed_anchor} (facing={facing}); "
        f"{rel_msg} {fixed_name} --[{relation_type}]--> {moving_name}. "
        f"New {moving_name} pose: translate={pos_str} orient_xyzw={quat_str}.",
    )


def handle_add_aligned_component(args: dict, state: AgentState) -> ToolResult:
    """Add a new component snapped to an existing one's anchor — atomically.

    Combines ``add_component`` + ``align_components`` so the agent never has
    to render a known-wrong placeholder pose just to clear the edit gate.
    Validates name/asset/pool/anchors first, then commits in one shot.

    Direction: ``fixed_component --[relation_type]--> new_component`` with
    ``from_anchor=fixed_anchor``, ``to_anchor=moving_anchor`` (matches
    ``align_components``).  Counts as one edit.
    """
    if (gate := _check_inventory_locked(
        state, "add an aligned component",
    )) is not None:
        return gate
    # NOTE: like add_component, this is exempt from the render-after-edit gate.
    # Anchor-based placement is geometrically determined — there's no
    # attribution question to answer per-edit.

    name = args.get("name", "").strip()
    asset_id = args.get("asset_id", "").strip()
    fixed_name = args.get("fixed_component", "").strip()
    fixed_anchor = args.get("fixed_anchor", "").strip()
    moving_anchor = args.get("moving_anchor", "").strip()
    facing = args.get("facing", "same_frame")
    relation_type = args.get("relation_type", "attach").strip() or "attach"
    confidence = float(args.get("confidence", 0.8))
    item_id, err = _validate_item_id_for_add(state, args.get("item_id"))
    if err is not None:
        return err

    if not name:
        return _err("name is required and must be non-empty.")
    if not asset_id:
        return _err("asset_id is required.")
    if not fixed_name:
        return _err("fixed_component is required.")
    if not fixed_anchor or not moving_anchor:
        return _err("fixed_anchor and moving_anchor are required.")
    if facing not in ("same_frame", "opposed_frame"):
        return _err(
            f"facing must be 'same_frame' or 'opposed_frame', got {facing!r}.",
        )

    idx = _component_index(state)
    if name in idx:
        return _err(
            f"Component named {name!r} already exists. Use modify_component "
            "or align_components instead, or pick a different name.",
        )
    if fixed_name not in idx:
        return _err(f"No component named {fixed_name!r}.")
    if name == fixed_name:
        return _err("name and fixed_component must differ.")

    cat = state._variant_to_category.get(asset_id)
    if not cat:
        return _err(f"Unknown asset_id {asset_id!r}.")
    if asset_id not in state.retrieval_pool_ids:
        return _err(
            f"asset_id {asset_id!r} is not in the retrieval pool. "
            "Only pool assets are allowed.",
        )

    fixed = state.prediction.components[idx[fixed_name]]
    resolved = _resolve_align_anchors(
        state, fixed, asset_id, fixed_anchor, moving_anchor,
    )
    if isinstance(resolved, ToolResult):
        return resolved
    fa, ma = resolved

    # Compute pose first against a hypothetical identity-pose moving comp;
    # the math doesn't actually need the component to exist yet, just the
    # asset-local anchor pose.  Once the pose is known, append.
    # We do this by running the alignment math directly here against a
    # synthesized PredictedComponent for the moving piece — at identity.
    placeholder = PredictedComponent(
        name=name, asset_id=asset_id, family=cat.family,
        translate=[0.0, 0.0, 0.0], orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        confidence=confidence,
    )
    new_translate, new_orient = _compute_aligned_pose(fixed, fa, ma, facing)
    placeholder.translate = new_translate
    placeholder.orientation_xyzw = new_orient

    state.prediction.components.append(placeholder)
    state.scene_modified_since_render = True

    rel_msg = _upsert_relation(
        state, relation_type, fixed_name, name, fixed_anchor, moving_anchor,
    )

    pos_str = [round(v, 4) for v in placeholder.translate]
    quat_str = [round(v, 4) for v in placeholder.orientation_xyzw]
    msg = (
        f"Added {name!r} (asset={asset_id}) and aligned it: "
        f"{name!r}.{moving_anchor} -> {fixed_name!r}.{fixed_anchor} "
        f"(facing={facing}); {rel_msg} {fixed_name} --[{relation_type}]--> "
        f"{name}.  New {name} pose: translate={pos_str} orient_xyzw={quat_str}. "
        f"Scene now has {len(state.prediction.components)} components."
    )
    if item_id is not None:
        _inventory_index(state)[item_id].matched_components.append(name)
        msg += f"  Auto-matched to inventory item {item_id!r}."
    return _ok(msg)


def handle_submit_prediction(args: dict, state: AgentState) -> ToolResult:
    """Finalise the prediction — gated on inventory completeness.

    Two structural gates beyond the render-after-edit rule:
      1. Every inventory item must have ≥1 matched component, or be in
         ``acknowledge_unmatched`` with notes.
      2. Every placed component must be matched to some inventory item,
         or be in ``acknowledge_extras`` with notes.  This catches
         under-counting: when the agent treats a modular assembly as a
         single inventory item, the extra pieces show up unmatched here.
    """
    notes = (args.get("notes") or "").strip()
    ack_arg = args.get("acknowledge_unmatched") or []
    if not isinstance(ack_arg, list) or not all(isinstance(s, str) for s in ack_arg):
        return _err(
            "acknowledge_unmatched must be a list of inventory item_id "
            "strings (the items you couldn't match).",
        )
    ack_set = {s.strip() for s in ack_arg if s.strip()}

    extras_arg = args.get("acknowledge_extras") or []
    if not isinstance(extras_arg, list) or not all(isinstance(s, str) for s in extras_arg):
        return _err(
            "acknowledge_extras must be a list of component name strings "
            "(placed components you intentionally left unmatched to any "
            "inventory item).",
        )
    extras_ack_set = {s.strip() for s in extras_arg if s.strip()}

    if not state.inventory_locked:
        return _err(
            "Cannot submit: inventory not set.  Call `set_inventory` first "
            "with the list of distinct objects you can see in the references, "
            "then place components and `mark_matched` each one before "
            "submitting.",
        )
    if not state.inventory:
        return _err(
            "Cannot submit: inventory is locked but empty.  At minimum the "
            "references must contain one distinct object.",
        )

    # Render-after-edit gate (only meaningful when the render tool exists).
    if state.renderer is not None and state.scene_modified_since_render:
        return _err(
            "You changed the scene since the last render.  Call `render` "
            "first to verify the new state matches the references, then "
            "re-submit.",
        )

    inv_idx = _inventory_index(state)

    # Validate acknowledge_unmatched references real items.
    bad_ack = [s for s in ack_set if s not in inv_idx]
    if bad_ack:
        return _err(
            f"acknowledge_unmatched references unknown inventory item_id(s): "
            f"{bad_ack}.  Valid ids: {sorted(inv_idx.keys())}",
        )

    # Find unmatched items not covered by acknowledge_unmatched.
    unmatched = [item.item_id for item in state.inventory if not item.matched]
    blocking = [iid for iid in unmatched if iid not in ack_set]
    if blocking:
        return _err(
            f"Cannot submit: {len(blocking)} inventory item(s) are still "
            f"unmatched and not acknowledged: {blocking}.  Either "
            "`mark_matched` a placed component to each, or — if you've "
            "genuinely tried and cannot place one — pass its item_id in "
            "`acknowledge_unmatched` along with a `notes` explanation.",
        )

    # Find placed components not matched to any inventory item.  Most common
    # cause: agent under-counted by lumping multiple modular pieces into one
    # inventory item (e.g. a U-conveyor as one item, then placing curve+2
    # straights but only marking one of them).  Block by default; allow
    # override via acknowledge_extras with notes.
    placed = len(state.prediction.components)
    matched_components = {
        c
        for item in state.inventory
        for c in item.matched_components
    }
    extras = [
        c.name for c in state.prediction.components
        if c.name not in matched_components
    ]
    blocking_extras = [n for n in extras if n not in extras_ack_set]
    if blocking_extras:
        suggestion = ""
        # If component count is much larger than inventory item count, this
        # is almost certainly under-counting.  Surface the hint.
        if len(state.inventory) > 0 and placed > 2 * len(state.inventory):
            suggestion = (
                "  Component count ({} placed) is much larger than inventory "
                "count ({} items) — you likely under-counted: a modular "
                "assembly such as a U-conveyor or a shelf row should be "
                "ONE inventory item PER piece, not one item for the whole "
                "assembly.  Use `add_inventory_item` to add the missing "
                "items, then `mark_matched` each component to its own item."
            ).format(placed, len(state.inventory))
        return _err(
            f"Cannot submit: {len(blocking_extras)} placed component(s) "
            f"are not matched to any inventory item: {blocking_extras}.  "
            "Either `mark_matched` each one (adding new inventory items via "
            "`add_inventory_item` if your initial inventory was too coarse), "
            "or — if you intentionally placed scaffolding that shouldn't be "
            "tied to an inventory item — pass the component names in "
            "`acknowledge_extras` with a `notes` explanation." + suggestion,
        )

    # If anything is acknowledged (either side), require notes.
    if (ack_set or extras_ack_set) and not notes:
        return _err(
            "acknowledge_unmatched / acknowledge_extras is non-empty, so "
            "`notes` is required — explain the rationale.",
        )

    state.submitted = True
    state.submit_notes = notes
    matched_items = sum(1 for item in state.inventory if item.matched)
    msg = (
        f"Submitted prediction with {placed} components covering "
        f"{matched_items}/{len(state.inventory)} inventory item(s)."
    )
    if ack_set:
        msg += f"  acknowledged_unmatched={sorted(ack_set)}"
    if extras_ack_set:
        msg += f"  acknowledged_extras={sorted(extras_ack_set)}"
    msg += "  The loop will terminate after this turn."
    return _ok(msg)


# ----------------------------------------------------------------------------
# Tool specs (Anthropic-style schemas)
# ----------------------------------------------------------------------------

_POSITION_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
    "description": "[x, y, z] in meters (Z-up).",
}
_ORIENT_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
    "description": "Quaternion [x, y, z, w] in scalar-last form.",
}


def default_tool_specs() -> list[ToolSpec]:
    """The full non-render tool set for the agentic loop."""
    _INVENTORY_ITEM_SCHEMA = {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": (
                    "Short unique slug for this object (e.g. 'u_curve', "
                    "'left_straight', 'pallet_back_1').  Stays stable across "
                    "the run; you'll use it to mark_matched / acknowledge."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "One-line description: what the object looks like and "
                    "where in the scene it sits."
                ),
            },
            "rough_xy": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Optional rough (x, y) guess in meters — purely a hint "
                    "for the agent's own future reference."
                ),
            },
            "notes": {"type": "string"},
        },
        "required": ["item_id", "description"],
    }
    return [
        ToolSpec(
            name="set_inventory",
            description=(
                "Commit your inventory of distinct physical objects in the "
                "reference images.  REQUIRED before any placement: scene-edit "
                "tools (add_component, modify_component, align_components, "
                "add_aligned_component) refuse to run until this is called.  "
                "One-shot — once locked, use `update_inventory_item` or "
                "`add_inventory_item` to revise."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": _INVENTORY_ITEM_SCHEMA,
                    },
                },
                "required": ["items"],
            },
            handler=handle_set_inventory,
        ),
        ToolSpec(
            name="mark_matched",
            description=(
                "Tie a placed component to an inventory item (1:N — one "
                "inventory item can have multiple matched components, e.g. "
                "a 'row of pallets' item covering 3 placed pallets).  Call "
                "after placing each component."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "component": {"type": "string"},
                },
                "required": ["item_id", "component"],
            },
            handler=handle_mark_matched,
        ),
        ToolSpec(
            name="unmark_matched",
            description=(
                "Remove a component link from an inventory item.  Omit "
                "`component` to clear all matches for that item — useful "
                "when redoing placement from scratch."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "component": {"type": "string"},
                },
                "required": ["item_id"],
            },
            handler=handle_unmark_matched,
        ),
        ToolSpec(
            name="update_inventory_item",
            description=(
                "Revise the description / rough_xy / notes of an existing "
                "inventory item without re-committing the whole inventory."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "description": {"type": "string"},
                    "rough_xy": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "notes": {"type": "string"},
                },
                "required": ["item_id"],
            },
            handler=handle_update_inventory_item,
        ),
        ToolSpec(
            name="add_inventory_item",
            description=(
                "Add a NEW inventory item after the initial lock.  Requires "
                "a `reason` — use only when a render reveals an object you "
                "missed in your initial inventory pass.  Silent additions "
                "defeat the purpose of locking."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "description": {"type": "string"},
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why this item was missed in the initial "
                            "inventory.  Will be recorded in audit trail."
                        ),
                    },
                    "rough_xy": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "required": ["item_id", "description", "reason"],
            },
            handler=handle_add_inventory_item,
        ),
        ToolSpec(
            name="list_families",
            description=(
                "List all asset families in the taxonomy and which have "
                "entries in the retrieval pool (pool entries are the only "
                "assets usable for placement)."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=handle_list_families,
        ),
        ToolSpec(
            name="list_assets_in_family",
            description=(
                "List all asset variant_ids in a given family.  Prefer "
                "pool_only=true (default) to only see placeable assets."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "family": {"type": "string"},
                    "pool_only": {"type": "boolean", "default": True},
                },
                "required": ["family"],
            },
            handler=handle_list_assets_in_family,
        ),
        ToolSpec(
            name="get_asset_info",
            description="Fetch metadata for one asset variant_id.",
            input_schema={
                "type": "object",
                "properties": {"asset_id": {"type": "string"}},
                "required": ["asset_id"],
            },
            handler=handle_get_asset_info,
        ),
        ToolSpec(
            name="get_catalog_hits",
            description=(
                "Return top-k CLIP visual matches between the reference "
                "images and the asset catalog.  Useful as a starting shortlist."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "default": 15, "minimum": 1},
                },
            },
            handler=handle_get_catalog_hits,
        ),
        ToolSpec(
            name="list_components",
            description=(
                "Return the current scene under construction — components, "
                "their asset_ids, positions, orientations, and relations."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=handle_list_components,
        ),
        ToolSpec(
            name="add_component",
            description=(
                "Place a new component in the scene.  asset_id must be in "
                "the retrieval pool; names must be unique.  Pass `item_id` "
                "to auto-match the new component to an inventory item in "
                "the same call — preferred over add + separate mark_matched. "
                "Adds do NOT require a render between consecutive adds; you "
                "can batch placements and render once at the end."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "asset_id": {"type": "string"},
                    "position": _POSITION_SCHEMA,
                    "orientation_xyzw": _ORIENT_SCHEMA,
                    "confidence": {"type": "number", "default": 0.8},
                    "item_id": {
                        "type": "string",
                        "description": (
                            "Optional inventory item_id.  When provided, "
                            "the new component is mark_matched to this "
                            "item atomically.  Saves one tool call per "
                            "placement.  Must already exist in the "
                            "inventory (use add_inventory_item first if "
                            "you discovered an extra item)."
                        ),
                    },
                },
                "required": ["name", "asset_id", "position"],
            },
            handler=handle_add_component,
        ),
        ToolSpec(
            name="modify_component",
            description=(
                "Patch one or more fields of an existing component.  Only "
                "the fields you provide are changed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "asset_id": {"type": "string"},
                    "position": _POSITION_SCHEMA,
                    "orientation_xyzw": _ORIENT_SCHEMA,
                    "confidence": {"type": "number"},
                },
                "required": ["name"],
            },
            handler=handle_modify_component,
        ),
        ToolSpec(
            name="remove_component",
            description="Delete a component and any relations referencing it.",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=handle_remove_component,
        ),
        ToolSpec(
            name="add_relation",
            description=(
                "Add a spatial relation between two placed components "
                "(e.g. on_top_of, adjacent_to).  Both nodes must already exist."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "from_node": {"type": "string"},
                    "to_node": {"type": "string"},
                    "from_anchor": {"type": "string", "default": ""},
                    "to_anchor": {"type": "string", "default": ""},
                },
                "required": ["type", "from_node", "to_node"],
            },
            handler=handle_add_relation,
        ),
        ToolSpec(
            name="list_references",
            description=(
                "List the file names of all reference images provided to "
                "this run.  Use to discover which views you can re-fetch."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=handle_list_references,
        ),
        ToolSpec(
            name="view_reference",
            description=(
                "Re-attach a specific reference image to the conversation "
                "by file name.  Useful for inspecting a view that was not "
                "in the initial batch or has been compacted out of context."
            ),
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=handle_view_reference,
        ),
        ToolSpec(
            name="add_aligned_component",
            description=(
                "Add a new component AND snap it to an existing one's anchor "
                "in a single edit.  Preferred path for chaining modular "
                "assets — you don't have to invent a placeholder pose just "
                "to call align afterward.  Direction: fixed --[type]--> new "
                "with from_anchor=fixed_anchor, to_anchor=moving_anchor.  "
                "Adds do NOT require a render between consecutive adds — "
                "batch a chain of pieces and render once at the end.  Pass "
                "`item_id` to also auto-match the new component to an "
                "inventory item in the same call."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "asset_id": {"type": "string"},
                    "fixed_component": {"type": "string"},
                    "fixed_anchor": {"type": "string"},
                    "moving_anchor": {"type": "string"},
                    "facing": {
                        "type": "string",
                        "enum": ["same_frame", "opposed_frame"],
                        "default": "same_frame",
                    },
                    "relation_type": {
                        "type": "string",
                        "default": "attach",
                    },
                    "confidence": {"type": "number", "default": 0.8},
                    "item_id": {
                        "type": "string",
                        "description": (
                            "Optional inventory item_id.  When provided, "
                            "the new component is mark_matched to this "
                            "item atomically.  Saves one tool call per "
                            "placement."
                        ),
                    },
                },
                "required": [
                    "name", "asset_id", "fixed_component",
                    "fixed_anchor", "moving_anchor",
                ],
            },
            handler=handle_add_aligned_component,
        ),
        ToolSpec(
            name="get_component_anchors",
            description=(
                "Return WORLD-frame anchor poses for a placed component. "
                "Each row gives a position and orientation_xyzw you can "
                "compare against another component's anchors to plan a "
                "snap.  Canonical raw anchor names only; aliases shown as "
                "metadata.  Invalid anchors are returned with a flag — "
                "align_components will reject them."
            ),
            input_schema={
                "type": "object",
                "properties": {"component": {"type": "string"}},
                "required": ["component"],
            },
            handler=handle_get_component_anchors,
        ),
        ToolSpec(
            name="align_components",
            description=(
                "Snap one component's anchor to another's with a "
                "deterministic rigid SE(3) transform.  Sets the moving "
                "component's translate + orientation_xyzw and adds/updates "
                "a relation `fixed --[type]--> moving` carrying the anchor "
                "names.  Prefer this over manual position math for any "
                "modular asset (conveyors).  Counts as a scene edit — must "
                "be followed by a render before the next edit."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "fixed_component": {"type": "string"},
                    "fixed_anchor": {
                        "type": "string",
                        "description": (
                            "Anchor name on the fixed component's asset. "
                            "Use get_asset_info or get_component_anchors to "
                            "see available names (raw + aliases)."
                        ),
                    },
                    "moving_component": {"type": "string"},
                    "moving_anchor": {"type": "string"},
                    "facing": {
                        "type": "string",
                        "enum": ["same_frame", "opposed_frame"],
                        "default": "same_frame",
                        "description": (
                            "same_frame: moving anchor frame matches fixed "
                            "anchor frame (the standard for conveyor chaining "
                            "— Isaac's Anchorpoint already encodes the "
                            "destination pose).  opposed_frame: insert Rz(pi) "
                            "so anchors face into each other (use only for "
                            "pieces that mate back-to-back)."
                        ),
                    },
                    "relation_type": {
                        "type": "string",
                        "default": "attach",
                        "description": (
                            "Relation type stored on the resulting "
                            "PredictedRelation.  Default 'attach' is the "
                            "modular-chain semantic; override only for "
                            "non-attach mating (e.g. 'stack')."
                        ),
                    },
                },
                "required": [
                    "fixed_component", "fixed_anchor",
                    "moving_component", "moving_anchor",
                ],
            },
            handler=handle_align_components,
        ),
        ToolSpec(
            name="plot_top_down",
            description=(
                "Render a 2D top-down schematic of the currently placed "
                "components — gridded in meters, with each component drawn "
                "as a labelled bbox rectangle.  Free (no render budget) and "
                "always available.  Use this to read off component positions "
                "in scene coordinates and to compare the layout against the "
                "reference top-down view."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=handle_plot_top_down,
        ),
        ToolSpec(
            name="submit_prediction",
            description=(
                "Finalise the scene reconstruction.  Gated: inventory must "
                "be locked (via set_inventory), every inventory item must "
                "have at least one matched component (via mark_matched) OR "
                "be listed in `acknowledge_unmatched`, every placed "
                "component must be matched to some inventory item OR be "
                "listed in `acknowledge_extras`, and — when the render "
                "tool is wired in — the current scene must have been "
                "rendered since the last edit."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "acknowledge_unmatched": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of inventory item_ids you couldn't place "
                            "after honest effort.  These items will not "
                            "block submit but `notes` becomes required."
                        ),
                    },
                    "acknowledge_extras": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of placed component names you "
                            "intentionally left unmatched to any inventory "
                            "item (e.g. scaffolding).  Required to bypass "
                            "the extras gate.  `notes` becomes required."
                        ),
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "Summary of decisions.  Required when "
                            "acknowledge_unmatched or acknowledge_extras "
                            "is non-empty; should explain the rationale."
                        ),
                    },
                },
            },
            handler=handle_submit_prediction,
        ),
    ]
