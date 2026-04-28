"""Perception agent — analyse scene images with a visual asset catalog.

When CLIP candidates are provided, the VLM sees the scene images alongside
thumbnail previews of each candidate asset, making it much easier to
identify individual components and count instances.

Supports optional critic feedback for re-perception during refinement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaacsim_bench.agents.vlm import VLMClient

if TYPE_CHECKING:
    from isaacsim_bench.agents.clip_catalog import CatalogHit
    from isaacsim_bench.agents.critic import CriticResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class DetectedObject:
    name: str
    asset_id: str  # variant_id from catalog, or "unknown"
    family: str
    sub_type: str
    visual_description: str


@dataclass
class Connection:
    from_object: str
    to_object: str
    connection_type: str
    description: str = ""


@dataclass
class PerceptionResult:
    scene_description: str
    objects: list[DetectedObject]
    spatial_layout: str
    connections: list[Connection]


# ------------------------------------------------------------------
# Prompt & tool schema
# ------------------------------------------------------------------

_SYSTEM = """\
You are an expert industrial-scene analyst.  You decompose rendered 3D \
scenes into individual catalog parts.

DOMAIN KNOWLEDGE — how industrial scenes are assembled:

  Conveyors:
    Conveyor lines are NEVER one piece.  They are assembled from modular \
  segments connected end-to-end:
    • Straight conveyor belts — flat belt sections, typically 2-3 m long.
    • Curve conveyors — 90° or 180° turns that redirect the line.
    • Roller conveyors — use exposed rollers instead of a flat belt.
    A U-shaped conveyor = 2 straight + 1 × 180° curve = 3 parts.
    An L-shaped conveyor = 2 straight + 1 × 90° curve = 3 parts.
    Segments connect at their ends — look for joints/seams between pieces.

  Storage & logistics:
    • Shelf/rack rows = N shelf units side-by-side.
    • Pallet areas = M × N individual pallets in a grid.
    • Boxes, crates, barrels — count each instance separately.

  Robots & vehicles:
    • Manipulator arms (mounted on a base or pedestal)
    • Mobile robots (wheeled/tracked platforms)
    • Forklifts, dollies, pushcarts
    Each robot is typically ONE object (do not decompose into joints).

  Environment & safety:
    • Traffic cones, wet-floor signs, fire extinguishers
    • Wall-mounted items: fuse boxes, signage, brackets
    • Barrels (plastic drums) — often scattered around warehouses

You will be shown:
1. Multiple camera views of the scene.
2. A visual catalog of parts (thumbnails).  Each thumbnail = ONE segment.

Your task:
  Look at each catalog thumbnail.  Then look at the scene and count how \
  many instances of that segment you can find.  Where sections repeat \
  (e.g. the two parallel runs of a U-shape conveyor), report each as a \
  separate object.

  Report physical connections (attached end-to-end, stacked, adjacent) \
  between pieces."""

_SYSTEM_NO_CATALOG = """\
You are an expert industrial-scene analyst.  You study rendered images of \
warehouse / factory environments and identify every physical object.

Rules:
1. Count every distinct physical instance separately.
2. Distinguish sub-types (e.g. straight vs curved conveyor, plastic vs \
   wooden pallet, shelf unit vs rack frame).
3. An assembled structure is made from MULTIPLE separate pieces — count each \
   piece individually (e.g. a conveyor line is multiple segments).
4. Robots and vehicles are single objects — do NOT decompose into parts.
5. Note all physical connections between objects."""

_TOOL = {
    "name": "report_scene_analysis",
    "description": "Report the structured analysis of the industrial scene",
    "input_schema": {
        "type": "object",
        "properties": {
            "scene_description": {
                "type": "string",
                "description": "1-2 sentence overview of the scene",
            },
            "objects": {
                "type": "array",
                "description": "Every distinct physical object instance",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Unique instance name, e.g. "
                                "'straight_conveyor_1', 'straight_conveyor_2'"
                            ),
                        },
                        "asset_id": {
                            "type": "string",
                            "description": (
                                "The variant_id from the catalog that best "
                                "matches this object, or 'unknown' if no "
                                "catalog match"
                            ),
                        },
                        "family": {
                            "type": "string",
                            "description": (
                                "Asset family: conveyor, rack, pallet, box, "
                                "barrel, crate, container, robot, vehicle, "
                                "safety, signage, prop, other"
                            ),
                        },
                        "sub_type": {
                            "type": "string",
                            "description": (
                                "Specific type: straight, curve_90, curve_180, "
                                "roller, shelf_unit, frame, pile, etc."
                            ),
                        },
                        "visual_description": {
                            "type": "string",
                            "description": "Colour, size, distinguishing features",
                        },
                    },
                    "required": [
                        "name", "asset_id", "family",
                        "sub_type", "visual_description",
                    ],
                },
            },
            "spatial_layout": {
                "type": "string",
                "description": (
                    "Overall arrangement: pattern, approximate dimensions, "
                    "relative positions"
                ),
            },
            "connections": {
                "type": "array",
                "description": "Physical connections / adjacencies",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_object": {"type": "string"},
                        "to_object": {"type": "string"},
                        "connection_type": {
                            "type": "string",
                            "enum": ["attached", "adjacent", "stacked", "nearby"],
                        },
                        "description": {
                            "type": "string",
                            "description": "How they connect",
                        },
                    },
                    "required": ["from_object", "to_object", "connection_type"],
                },
            },
        },
        "required": ["scene_description", "objects", "spatial_layout", "connections"],
    },
}


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class PerceptionAgent:
    """Analyse scene images with optional CLIP-guided visual catalog."""

    def __init__(self, vlm: VLMClient) -> None:
        self.vlm = vlm

    def analyze(
        self,
        image_paths: list[Path],
        catalog_hits: list[CatalogHit] | None = None,
        critic_feedback: CriticResult | None = None,
    ) -> PerceptionResult:
        selected = self.vlm.select_images(image_paths)
        logger.info("Perception: sending %d images", len(selected))

        content: list[dict] = []

        # --- Scene images ---
        content.append({"type": "text", "text": "## Scene Images\n"})
        for p in selected:
            content.append(self.vlm.encode_image(p))

        # --- Visual catalog (if available) ---
        if catalog_hits:
            # Sort by CLIP score descending so the VLM sees best matches first
            sorted_hits = sorted(catalog_hits, key=lambda h: h.score, reverse=True)

            content.append({
                "type": "text",
                "text": (
                    "\n## Asset Catalog — individual parts "
                    "(sorted by visual similarity to this scene)\n"
                    "Each thumbnail below is ONE individual part.  "
                    "The scene is ASSEMBLED from these parts.  "
                    "A single assembled structure uses MULTIPLE parts.\n"
                    "The **CLIP score** measures visual similarity between "
                    "this asset thumbnail and the scene images — higher = "
                    "more visually similar.\n"
                ),
            })
            for hit in sorted_hits:
                if hit.thumbnail_path.exists():
                    content.append({
                        "type": "text",
                        "text": (
                            f"\n**{hit.variant_id}** — {hit.category_name} "
                            f"(family: {hit.family}, "
                            f"CLIP similarity: {hit.score:.3f})"
                        ),
                    })
                    content.append(self.vlm.encode_image(hit.thumbnail_path))

            instructions = (
                "\n---\n"
                "INSTRUCTIONS:\n"
                "The scene is built ENTIRELY from the catalog parts above.\n\n"
                "For each catalog thumbnail, count how many times that "
                "exact part appears in the scene.  Report each instance "
                "as a separate object with a unique name.\n\n"
                "IMPORTANT: When multiple catalog assets look similar (e.g. "
                "several straight conveyor variants), prefer the one with "
                "the HIGHER CLIP similarity score — it is the better visual "
                "match.  Only override this if you can clearly see a specific "
                "visual difference (colour, texture, rollers vs belt) that "
                "matches a lower-scored asset better.\n\n"
                "EXAMPLE: If the scene shows a U-shaped conveyor, it is "
                "assembled from separate pieces — likely 2 × straight "
                "conveyor + 1 × 180° curve.  Report all 3 as separate "
                "objects and note how they connect end-to-end.\n\n"
                "Use the variant_id from the catalog as the asset_id."
            )
            system = _SYSTEM
        else:
            instructions = (
                "\nIdentify every distinct physical object.  "
                "Count each instance separately.  "
                "Note all physical connections."
            )
            system = _SYSTEM_NO_CATALOG

        # --- Critic feedback (for re-perception after structural errors) ---
        if critic_feedback and critic_feedback.issues:
            feedback_lines = []
            for issue in critic_feedback.issues:
                feedback_lines.append(
                    f"  - [{issue.issue_type}] {issue.component_name}: "
                    f"{issue.description} — suggestion: {issue.suggestion}"
                )
            instructions += (
                "\n\n## Previous attempt feedback\n"
                "A critic reviewed a previous analysis and found these "
                "errors. Pay close attention to these issues:\n"
                + "\n".join(feedback_lines)
                + f"\n\nCritic reasoning: {critic_feedback.reasoning}\n"
                "Fix the issues above in your new analysis."
            )

        content.append({"type": "text", "text": instructions})

        raw = self.vlm.query_with_tool(
            system=system, user_content=content, tool=_TOOL,
        )

        objects = [
            DetectedObject(
                name=o["name"],
                asset_id=o.get("asset_id", "unknown"),
                family=o["family"],
                sub_type=o["sub_type"],
                visual_description=o["visual_description"],
            )
            for o in raw.get("objects", [])
        ]
        connections = [
            Connection(
                from_object=c["from_object"],
                to_object=c["to_object"],
                connection_type=c["connection_type"],
                description=c.get("description", ""),
            )
            for c in raw.get("connections", [])
        ]

        return PerceptionResult(
            scene_description=raw.get("scene_description", ""),
            objects=objects,
            spatial_layout=raw.get("spatial_layout", ""),
            connections=connections,
        )
