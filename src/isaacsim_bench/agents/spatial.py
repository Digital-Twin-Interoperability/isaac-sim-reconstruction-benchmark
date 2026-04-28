"""Spatial agent — estimate 3D positions, orientations, and relations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaacsim_bench.agents.perception import PerceptionResult
from isaacsim_bench.agents.retrieval import AssetMatch
from isaacsim_bench.agents.vlm import VLMClient

if TYPE_CHECKING:
    from isaacsim_bench.agents.critic import CriticResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class ComponentPose:
    name: str
    position: list[float]  # [x, y, z] metres
    heading_deg: float  # rotation about Z


@dataclass
class InferredRelation:
    type: str  # attach, adjacent
    from_node: str
    to_node: str
    from_anchor: str
    to_anchor: str


@dataclass
class LayoutEstimate:
    poses: list[ComponentPose]
    relations: list[InferredRelation]


# ------------------------------------------------------------------
# Prompt & tool schema
# ------------------------------------------------------------------

_SYSTEM = """\
You are a 3D scene-layout estimation expert.  Given multiple views of an \
industrial scene together with a list of identified objects, you estimate \
their 3D positions and orientations.

Coordinate conventions
  • Right-handed, Z-up
  • Unit: metres
  • Place the origin near the centre of the scene

Orientation
  • Express as a *heading* angle in degrees (rotation about Z).
    0° → facing +X,  90° → facing +Y,  180° → facing −X,  270° → facing −Y.

Typical dimensions (use to calibrate your estimates)
  Conveyors:
    • Straight conveyor belt  ≈ 2.5 m long × 0.6 m wide × 0.8 m tall
    • 180° curve conveyor     ≈ 1.5 m radius × 0.8 m tall
    • 90°  curve conveyor     ≈ 1.0 m radius × 0.8 m tall
    • Roller conveyor         ≈ 2.5 m long × 0.6 m wide × 0.8 m tall
  Storage:
    • Shelf / rack unit       ≈ 1.2 m wide × 0.5 m deep × 2.0 m tall
    • Rack frame              ≈ 1.2 m wide × 0.5 m deep × 2.0 m tall
    • Standard pallet         ≈ 1.2 × 1.0 × 0.15 m
    • Cardboard box           ≈ 0.3–0.6 m per side
    • Plastic crate           ≈ 0.4 × 0.3 × 0.2 m
    • Plastic barrel          ≈ 0.6 m diameter × 0.9 m tall
  Safety & environment:
    • Traffic cone             ≈ 0.3 × 0.3 × 0.7 m
    • Wet-floor sign           ≈ 0.3 × 0.02 × 0.6 m
    • Fire extinguisher        ≈ 0.15 m diameter × 0.5 m tall
  Vehicles:
    • Forklift                 ≈ 2.5 × 1.2 × 2.0 m
    • Dolly / pushcart         ≈ 1.0 × 0.6 × 1.0 m
  Robots:
    • Manipulator arm (base)   ≈ 0.3 m diameter base × 0.5–1.5 m reach
    • Mobile robot             ≈ 0.6 × 0.4 × 0.3 m

Constraints — check your output against these:
  • Attached objects (type "attach") MUST be close (< 0.5 m gap).
  • No two objects should overlap (centres must be at least half the sum \
    of their widths apart).
  • Objects should rest on the ground (z ≈ 0) unless stacked.

Relations & anchors
  • type "attach": objects physically connected end-to-end.
  • type "adjacent": objects placed side-by-side without a physical joint.
  • Conveyor anchors: straight → "left_end" / "right_end";
    curve → "curve_entry" / "curve_exit".
  • Shelf anchors: "left_side" / "right_side".
  • Pallet anchors: "left_side" / "right_side" / "front_side" / "back_side".
  • Robot anchors: "base" / "end_effector".
  • Generic: "centre"."""

_TOOL = {
    "name": "report_layout",
    "description": "Report estimated 3D layout of scene components",
    "input_schema": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Must match an object name from the detection list",
                        },
                        "position": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "[x, y, z] in metres",
                        },
                        "heading_deg": {
                            "type": "number",
                            "description": "Rotation about Z in degrees",
                        },
                    },
                    "required": ["name", "position", "heading_deg"],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["attach", "adjacent"],
                        },
                        "from_node": {"type": "string"},
                        "to_node": {"type": "string"},
                        "from_anchor": {"type": "string"},
                        "to_anchor": {"type": "string"},
                    },
                    "required": [
                        "type", "from_node", "to_node",
                        "from_anchor", "to_anchor",
                    ],
                },
            },
        },
        "required": ["components", "relations"],
    },
}


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class SpatialAgent:
    """Estimate 3D poses and relations from scene images + detected objects."""

    def __init__(self, vlm: VLMClient) -> None:
        self.vlm = vlm

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_layout(raw: dict) -> LayoutEstimate:
        """Parse the VLM tool output into a LayoutEstimate."""
        poses = [
            ComponentPose(
                name=c["name"],
                position=c["position"],
                heading_deg=c["heading_deg"],
            )
            for c in raw.get("components", [])
        ]
        relations = [
            InferredRelation(
                type=r["type"],
                from_node=r["from_node"],
                to_node=r["to_node"],
                from_anchor=r["from_anchor"],
                to_anchor=r["to_anchor"],
            )
            for r in raw.get("relations", [])
        ]
        return LayoutEstimate(poses=poses, relations=relations)

    def _build_objects_context(
        self,
        perception: PerceptionResult,
        matches: list[AssetMatch],
    ) -> str:
        """Format object list + connections for the VLM context."""
        obj_lines = []
        for m in matches:
            det = next((o for o in perception.objects if o.name == m.name), None)
            sub = det.sub_type if det else "unknown"
            desc = det.visual_description if det else ""
            line = f"  • {m.name}  →  {m.asset_id} ({m.family} / {sub})"
            if desc:
                line += f"  — {desc}"
            obj_lines.append(line)

        conn_lines = []
        for c in perception.connections:
            conn_lines.append(
                f"  • {c.from_object} ↔ {c.to_object}  ({c.connection_type}: {c.description})"
            )

        return (
            "## Detected objects and matched assets\n"
            + "\n".join(obj_lines)
            + "\n\n## Observed connections\n"
            + ("\n".join(conn_lines) if conn_lines else "  (none observed)")
            + "\n\n## Spatial layout (from perception)\n"
            + perception.spatial_layout
        )

    def _prepare_images(
        self, image_paths: list[Path],
    ) -> tuple[list[Path], list[dict]]:
        """Select images and encode them as content blocks."""
        selected = self.vlm.select_images(image_paths)
        content = [self.vlm.encode_image(p) for p in selected]
        return selected, content

    @staticmethod
    def _find_depth_map(image_paths: list[Path]) -> Path | None:
        """Look for a depth .npy file alongside the images."""
        for p in image_paths:
            candidate = p.parent / "depth.npy"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _depth_summary(depth_path: Path) -> str:
        """Produce a textual summary of depth statistics."""
        import numpy as np

        depth = np.load(depth_path)
        valid = depth[depth > 0]
        if len(valid) == 0:
            return "  Depth map: no valid depth values"
        return (
            f"  Depth map statistics (from rendered depth):\n"
            f"    min={valid.min():.2f}m  max={valid.max():.2f}m  "
            f"mean={valid.mean():.2f}m  median={float(np.median(valid)):.2f}m\n"
            f"    This gives you the approximate scale of the scene."
        )

    # ------------------------------------------------------------------
    # Initial estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        image_paths: list[Path],
        perception: PerceptionResult,
        matches: list[AssetMatch],
    ) -> LayoutEstimate:
        selected, content = self._prepare_images(image_paths)
        logger.info("Spatial: sending %d images", len(selected))

        context = self._build_objects_context(perception, matches)

        # Add depth info if available
        depth_path = self._find_depth_map(image_paths)
        if depth_path:
            try:
                context += "\n\n" + self._depth_summary(depth_path)
            except Exception:
                logger.debug("Could not load depth map", exc_info=True)

        context += (
            "\n\n---\n"
            "Using the images and context above, estimate the 3D position "
            "and heading for each object.  Also infer the structural "
            "relations (attach / adjacent) with appropriate anchor names.\n\n"
            "Remember: attached objects must be close (< 0.5 m gap), and "
            "no two objects should overlap."
        )
        content.append({"type": "text", "text": context})

        raw = self.vlm.query_with_tool(
            system=_SYSTEM, user_content=content, tool=_TOOL,
        )
        return self._parse_layout(raw)

    # ------------------------------------------------------------------
    # Refinement (with critic feedback)
    # ------------------------------------------------------------------

    def refine(
        self,
        image_paths: list[Path],
        perception: PerceptionResult,
        matches: list[AssetMatch],
        current_layout: LayoutEstimate,
        critic: CriticResult,
    ) -> LayoutEstimate:
        """Re-estimate layout incorporating critic feedback."""
        selected, content = self._prepare_images(image_paths)
        logger.info("Spatial (refine): sending %d images + critic feedback", len(selected))

        # Objects context
        obj_context = self._build_objects_context(perception, matches)

        # Current layout summary
        pose_lines = []
        for p in current_layout.poses:
            pose_lines.append(
                f"  • {p.name}  pos=({p.position[0]:.2f}, {p.position[1]:.2f}, "
                f"{p.position[2]:.2f})  heading={p.heading_deg:.0f}°"
            )

        rel_lines = []
        for r in current_layout.relations:
            rel_lines.append(
                f"  • {r.from_node} —[{r.type}]→ {r.to_node}  "
                f"({r.from_anchor} → {r.to_anchor})"
            )

        # Critic feedback
        issue_lines = []
        for issue in critic.issues:
            issue_lines.append(
                f"  • [{issue.issue_type}] {issue.component_name}: "
                f"{issue.description}\n"
                f"    Suggestion: {issue.suggestion}"
            )

        context = (
            obj_context
            + "\n\n## Current layout estimate (needs correction)\n"
            + "\n".join(pose_lines)
            + "\n\nRelations:\n"
            + ("\n".join(rel_lines) if rel_lines else "  (none)")
            + "\n\n## Critic feedback (score: "
            + f"{critic.score:.0f}/10)\n"
            + f"{critic.reasoning}\n\n"
            + "Issues to fix:\n"
            + ("\n".join(issue_lines) if issue_lines else "  (none)")
            + "\n\n---\n"
            "Correct the positions, orientations, and relations based on "
            "the critic feedback and the images.  Output the full updated "
            "layout for ALL components, not just the ones with issues.\n\n"
            "Remember: attached objects must be close (< 0.5 m gap), and "
            "no two objects should overlap."
        )
        content.append({"type": "text", "text": context})

        raw = self.vlm.query_with_tool(
            system=_SYSTEM, user_content=content, tool=_TOOL,
        )
        return self._parse_layout(raw)
