"""Critic agent — review a prediction against the source images and flag issues."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from isaacsim_bench.agents.perception import PerceptionResult
from isaacsim_bench.agents.vlm import VLMClient
from isaacsim_bench.schemas.prediction import PredictionJSON

logger = logging.getLogger(__name__)


def _quat_to_heading_deg(xyzw: list[float]) -> float:
    """Convert [x, y, z, w] quaternion to heading degrees about Z axis."""
    _, _, z, w = xyzw
    return math.degrees(2.0 * math.atan2(z, w)) % 360


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class Issue:
    component_name: str
    issue_type: str  # wrong_position, wrong_orientation, wrong_asset,
    #                   wrong_count, missing_relation, wrong_relation,
    #                   physical_impossibility
    description: str
    suggestion: str


@dataclass
class CriticResult:
    passed: bool
    score: float  # 0-10
    issues: list[Issue]
    reasoning: str


# ------------------------------------------------------------------
# Prompt & tool schema
# ------------------------------------------------------------------

_SYSTEM = """\
You are a rigorous quality-assurance reviewer for 3D scene reconstructions.

Given the original scene images and a proposed reconstruction (list of objects \
with positions, orientations, and relations), you must check for errors.

Check each of the following (in priority order):

1. **Object count** — does the prediction have the same number of objects \
   visible in the images?  Missing or hallucinated objects are critical errors.
2. **Asset identity** — does each predicted asset type match what is visible?  \
   (e.g. a straight conveyor should not be labelled as a curved one)
3. **Positions** — are the predicted positions plausible given the images?  \
   Objects should not overlap, float in mid-air, or be underground.  \
   Connected objects should be at physically consistent distances.
4. **Orientations** — do the heading angles match what the images show?  \
   (0° = facing +X, 90° = +Y, 180° = −X, 270° = −Y)
5. **Relations** — are all visible physical connections captured?  \
   Are the anchor names reasonable?
6. **Physical plausibility** — no interpenetration, no impossible gaps \
   between attached objects.  Attached objects should be < 0.5 m apart.

Scoring guide:
  10 = perfect reconstruction
  7-9 = minor position/orientation inaccuracies (< 0.5 m or < 15°)
  4-6 = some wrong assets or significant pose errors
  1-3 = major structural errors (wrong count, wrong layout)
  0   = completely wrong

Be strict.  If the layout topology is correct but positions are slightly off, \
that is still a pass (score >= 7).  If the topology is wrong (wrong connections, \
missing objects), that is a fail.

When suggesting corrections:
  • For positions, give approximate corrected [x, y, z] coordinates.
  • For orientations, give the corrected heading in degrees.
  • For wrong_count, state how many objects should be added or removed.
  • For wrong_asset, name the correct asset type."""

_TOOL = {
    "name": "review_prediction",
    "description": "Report the structured review of the scene prediction",
    "input_schema": {
        "type": "object",
        "properties": {
            "overall_pass": {
                "type": "boolean",
                "description": "True if the prediction is acceptable (score >= 7)",
            },
            "score": {
                "type": "number",
                "description": "Quality score 0-10",
            },
            "issues": {
                "type": "array",
                "description": "List of issues found (empty if perfect)",
                "items": {
                    "type": "object",
                    "properties": {
                        "component_name": {
                            "type": "string",
                            "description": "Name of the affected component, or 'scene' for global issues",
                        },
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "wrong_count",
                                "wrong_asset",
                                "wrong_position",
                                "wrong_orientation",
                                "missing_relation",
                                "wrong_relation",
                                "physical_impossibility",
                            ],
                        },
                        "description": {
                            "type": "string",
                            "description": "What is wrong",
                        },
                        "suggestion": {
                            "type": "string",
                            "description": (
                                "Concrete fix suggestion.  For positions, give "
                                "approximate corrected [x, y, z].  For orientations, "
                                "give corrected heading in degrees."
                            ),
                        },
                    },
                    "required": [
                        "component_name",
                        "issue_type",
                        "description",
                        "suggestion",
                    ],
                },
            },
            "reasoning": {
                "type": "string",
                "description": "Overall assessment and reasoning",
            },
        },
        "required": ["overall_pass", "score", "issues", "reasoning"],
    },
}


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class CriticAgent:
    """Review a composed prediction against the original scene images."""

    def __init__(self, vlm: VLMClient) -> None:
        self.vlm = vlm

    def review(
        self,
        image_paths: list[Path],
        prediction: PredictionJSON,
        perception: PerceptionResult,
    ) -> CriticResult:
        selected = self.vlm.select_images(image_paths)
        logger.info("Critic: reviewing prediction against %d images", len(selected))

        content: list[dict] = [self.vlm.encode_image(p) for p in selected]

        # Build prediction summary — show heading degrees, not quaternions
        comp_lines = []
        for c in prediction.components:
            t = c.translate
            heading = _quat_to_heading_deg(c.orientation_xyzw)
            comp_lines.append(
                f"  • {c.name}: asset={c.asset_id}, family={c.family}, "
                f"pos=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}), "
                f"heading={heading:.0f}°, "
                f"conf={c.confidence:.2f}"
            )

        rel_lines = []
        for r in prediction.relations:
            rel_lines.append(
                f"  • {r.from_node} —[{r.type}]→ {r.to_node}  "
                f"({r.from_anchor} → {r.to_anchor})"
            )

        # Quantitative consistency checks (pre-computed for the VLM)
        consistency_notes = self._check_consistency(prediction)

        text = (
            "## Original perception\n"
            f"Scene: {perception.scene_description}\n"
            f"Layout: {perception.spatial_layout}\n"
            f"\n## Predicted reconstruction\n"
            f"Components ({len(prediction.components)}):\n"
            + "\n".join(comp_lines)
            + "\n\nRelations:\n"
            + ("\n".join(rel_lines) if rel_lines else "  (none)")
        )

        if consistency_notes:
            text += "\n\n## Automated consistency checks\n" + "\n".join(consistency_notes)

        text += (
            "\n\n---\n"
            "Review this prediction against the images above.  "
            "Flag any errors in object count, asset identity, position, "
            "orientation, relations, or physical plausibility.  "
            "For each issue, provide a concrete correction suggestion."
        )
        content.append({"type": "text", "text": text})

        raw = self.vlm.query_with_tool(
            system=_SYSTEM, user_content=content, tool=_TOOL,
        )

        issues = [
            Issue(
                component_name=i["component_name"],
                issue_type=i["issue_type"],
                description=i["description"],
                suggestion=i["suggestion"],
            )
            for i in raw.get("issues", [])
        ]

        score = raw.get("score", 0)
        passed = raw.get("overall_pass", False)
        reasoning = raw.get("reasoning", "")

        # Enforce scoring guide — VLMs are systematically too lenient.
        # The prompt says: wrong assets → 4-6, wrong count → 1-3.
        # If the VLM scored higher despite flagging these issues, cap it.
        score, passed, reasoning = self._calibrate_score(
            score, passed, issues, reasoning,
        )

        return CriticResult(
            passed=passed,
            score=score,
            issues=issues,
            reasoning=reasoning,
        )

    @staticmethod
    def _calibrate_score(
        score: float,
        passed: bool,
        issues: list[Issue],
        reasoning: str,
    ) -> tuple[float, bool, str]:
        """Enforce the scoring guide that VLMs tend to ignore.

        Caps:
          wrong_count → max 3  (major structural error)
          wrong_asset → max 6  (wrong assets = significant error)
        """
        issue_types = {i.issue_type for i in issues}
        cap = 10.0
        cap_reason = ""

        if "wrong_count" in issue_types:
            cap = min(cap, 3.0)
            cap_reason = "wrong_count detected (cap=3)"
        if "wrong_asset" in issue_types:
            cap = min(cap, 6.0)
            if not cap_reason:
                cap_reason = "wrong_asset detected (cap=6)"

        if score > cap:
            logger.info(
                "Critic calibration: VLM scored %.0f but %s → capping to %.0f",
                score, cap_reason, cap,
            )
            reasoning += (
                f"\n[Score calibrated: VLM gave {score:.0f} but {cap_reason},"
                f" adjusted to {cap:.0f}]"
            )
            score = cap
            passed = score >= 7

        return score, passed, reasoning

    @staticmethod
    def _check_consistency(prediction: PredictionJSON) -> list[str]:
        """Run quantitative sanity checks on the prediction."""
        notes: list[str] = []

        comps = prediction.components
        comp_map = {c.name: c for c in comps}

        # Check for overlapping positions (centres too close)
        for i, a in enumerate(comps):
            for b in comps[i + 1:]:
                dx = a.translate[0] - b.translate[0]
                dy = a.translate[1] - b.translate[1]
                dz = a.translate[2] - b.translate[2]
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                if dist < 0.1:
                    notes.append(
                        f"  ⚠ OVERLAP: {a.name} and {b.name} are only "
                        f"{dist:.2f}m apart (nearly coincident)"
                    )

        # Check attached objects are close enough
        for r in prediction.relations:
            if r.type != "attach":
                continue
            a = comp_map.get(r.from_node)
            b = comp_map.get(r.to_node)
            if a is None or b is None:
                notes.append(
                    f"  ⚠ DANGLING RELATION: {r.from_node} → {r.to_node} "
                    f"references unknown component"
                )
                continue
            dx = a.translate[0] - b.translate[0]
            dy = a.translate[1] - b.translate[1]
            dz = a.translate[2] - b.translate[2]
            dist = (dx**2 + dy**2 + dz**2) ** 0.5
            if dist > 5.0:
                notes.append(
                    f"  ⚠ ATTACH GAP: {a.name} ↔ {b.name} are {dist:.1f}m apart "
                    f"(attached objects should be < 3m)"
                )

        # Check for underground objects
        for c in comps:
            if c.translate[2] < -0.1:
                notes.append(
                    f"  ⚠ UNDERGROUND: {c.name} has z={c.translate[2]:.2f}m "
                    f"(below ground)"
                )

        # Check for floating objects
        for c in comps:
            if c.translate[2] > 3.0:
                notes.append(
                    f"  ⚠ FLOATING: {c.name} has z={c.translate[2]:.2f}m "
                    f"(unusually high)"
                )

        return notes
