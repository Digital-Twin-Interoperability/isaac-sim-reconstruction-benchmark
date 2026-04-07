"""Extract ground-truth SceneJSON from a USD stage.

Uses only ``pxr`` (OpenUSD / usd-core) — no Isaac Sim runtime required.
Each Xform prim under the scene root is mapped to a ComponentEntry by
matching its USD reference path against the asset taxonomy.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pxr import Sdf, Usd

from isaacsim_bench.schemas.scene import (
    CameraParams,
    ComponentEntry,
    RelationEntry,
    SceneJSON,
)
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

logger = logging.getLogger(__name__)

# Structural families are tagged as "distractor" in evaluation —
# they define the environment but aren't the objects being reconstructed.
_STRUCTURAL_FAMILIES = frozenset(
    {
        "floor",
        "ceiling",
        "wall",
        "beam",
        "pillar",
        "light_fixture",
        "decal",
    }
)

# Name patterns → semantic class, used when taxonomy match gives no class.
_LABEL_MAP: list[tuple[str, str]] = [
    ("RackShelf", "rack_shelf"),
    ("RackFrame", "rack_frame"),
    ("Palette", "pallet"),
    ("Pallet", "pallet"),
    ("CardBox", "box"),
    ("FuseBox", "box"),
    ("Box_", "box"),
    ("Crate", "crate"),
    ("Barel", "barrel"),
    ("Barrel", "barrel"),
    ("Bottle", "bottle"),
    ("floor", "floor"),
    ("Floor", "floor"),
    ("Wall", "wall"),
    ("Ceiling", "ceiling"),
    ("Lamp", "light_fixture"),
    ("Sign", "sign"),
    ("forklift", "forklift"),
    ("TrafficCone", "traffic_cone"),
    ("FireExtinguisher", "fire_extinguisher"),
    ("Bracket", "bracket"),
    ("Beam", "beam"),
    ("Pillar", "pillar"),
    ("Rackshield", "rack_shield"),
    ("RackPile", "rack_pile"),
    ("Decal", "decal"),
    ("Stripe", "decal"),
    ("Pushcart", "pushcart"),
    ("WetFloorSign", "sign"),
]


def _semantic_from_name(prim_name: str) -> str:
    """Fallback semantic class derived from prim name patterns."""
    for pattern, label in _LABEL_MAP:
        if pattern.lower() in prim_name.lower():
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Taxonomy reverse index
# ---------------------------------------------------------------------------


def _build_taxonomy_index(
    taxonomy: AssetTaxonomy,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Build lookup tables from taxonomy.

    Returns
    -------
    by_path : dict mapping full ``usd_path`` → variant info dict
    by_stem : dict mapping filename stem (no extension) → variant info dict
    """
    by_path: dict[str, dict] = {}
    by_stem: dict[str, dict] = {}

    for cat in taxonomy.categories:
        for var in cat.variants:
            if var.usd_path is None:
                continue
            info = {
                "variant_id": var.variant_id,
                "name": var.name,
                "category_id": cat.category_id,
                "family": cat.family,
                "semantic_class": var.semantic_class or cat.category_id,
            }
            by_path[var.usd_path] = info
            stem = var.usd_path.rsplit("/", 1)[-1].removesuffix(".usd")
            by_stem[stem] = info

    return by_path, by_stem


def _resolve_ref(
    ref_asset_path: str,
    by_path: dict[str, dict],
    by_stem: dict[str, dict],
) -> dict | None:
    """Match a USD reference URL against the taxonomy index."""
    # Extract Isaac-relative path from S3 URL
    marker = "/Assets/Isaac/5.1/"
    idx = ref_asset_path.find(marker)
    rel_path = ref_asset_path[idx + len(marker) :] if idx >= 0 else ref_asset_path

    # 1. Direct path match
    if rel_path in by_path:
        return by_path[rel_path]

    # 2. Stem match
    stem = rel_path.rsplit("/", 1)[-1].removesuffix(".usd")
    if stem in by_stem:
        return by_stem[stem]

    # 3. Strip trailing _NNN version suffix and retry
    stripped = re.sub(r"_\d{2,}$", "", stem)
    if stripped in by_stem:
        return by_stem[stripped]

    return None


# ---------------------------------------------------------------------------
# Prim extraction
# ---------------------------------------------------------------------------


def _extract_prim_pose(prim: Usd.Prim) -> tuple[list[float], list[float]]:
    """Return (translate, orientation_xyzw) for a prim."""
    t_attr = prim.GetAttribute("xformOp:translate")
    translate = list(t_attr.Get()) if t_attr and t_attr.Get() is not None else [0.0, 0.0, 0.0]

    o_attr = prim.GetAttribute("xformOp:orient")
    if o_attr and o_attr.Get() is not None:
        q = o_attr.Get()
        orientation = [
            float(q.GetImaginary()[0]),
            float(q.GetImaginary()[1]),
            float(q.GetImaginary()[2]),
            float(q.GetReal()),
        ]
    else:
        orientation = [0.0, 0.0, 0.0, 1.0]

    return [float(v) for v in translate], orientation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_ground_truth(
    usd_path: str | Path,
    taxonomy_path: str | Path,
    *,
    scene_root: str = "/Root",
    sample_id: str | None = None,
    include_structural: bool = True,
) -> SceneJSON:
    """Parse a USD file and produce a SceneJSON ground truth.

    Parameters
    ----------
    usd_path
        Path to the ``.usd`` / ``.usdc`` file.
    taxonomy_path
        Path to ``asset_taxonomy.json``.
    scene_root
        USD prim path of the scene root (default ``/Root``).
    sample_id
        Identifier for this scene sample.  Defaults to the USD filename stem.
    include_structural
        If True (default), structural prims (floor, wall, ceiling …) are
        included as ``distractor`` components.  Set False to keep only props.
    """
    usd_path = Path(usd_path)
    taxonomy = AssetTaxonomy.model_validate_json(Path(taxonomy_path).read_text())
    by_path, by_stem = _build_taxonomy_index(taxonomy)

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD stage: {usd_path}")

    root_prim = stage.GetPrimAtPath(scene_root)
    if not root_prim:
        raise ValueError(f"Scene root prim not found: {scene_root}")

    layer = stage.GetRootLayer()

    components: list[ComponentEntry] = []
    unmatched: list[str] = []

    for prim in root_prim.GetChildren():
        prim_type = prim.GetTypeName()

        # Skip lights and non-Xform prims
        if prim_type in ("RectLight", "DistantLight", "DomeLight", "SphereLight"):
            continue
        if prim_type not in ("Xform", ""):
            logger.debug("Skipping prim %s (type=%s)", prim.GetName(), prim_type)
            continue

        prim_name = prim.GetName()

        # Get reference path
        prim_spec = layer.GetPrimAtPath(prim.GetPath())
        ref_path = ""
        if prim_spec:
            for ref in prim_spec.referenceList.prependedItems:
                ref_path = ref.assetPath
                break

        # Resolve against taxonomy
        info = _resolve_ref(ref_path, by_path, by_stem) if ref_path else None

        if info is not None:
            variant_id = info["variant_id"]
            asset_name = info["name"]
            family = info["family"]
            semantic_class = info["semantic_class"]
        else:
            # Fallback: use prim name for classification
            variant_id = prim_name
            asset_name = prim_name
            semantic_class = _semantic_from_name(prim_name)
            family = semantic_class
            unmatched.append(prim_name)

        # Determine evaluation role
        is_structural = family in _STRUCTURAL_FAMILIES or semantic_class in _STRUCTURAL_FAMILIES
        if is_structural and not include_structural:
            continue
        role = "distractor" if is_structural else "primary"

        translate, orientation = _extract_prim_pose(prim)

        components.append(
            ComponentEntry(
                name=prim_name,
                asset_id=variant_id,
                asset_name=asset_name,
                family=family,
                evaluation_role=role,
                match_regime="exact_match" if info else "unknown",
                translate=translate,
                orientation_xyzw=orientation,
            )
        )

    if unmatched:
        logger.warning(
            "%d prims unmatched against taxonomy: %s",
            len(unmatched),
            unmatched[:10],
        )

    # Pick root_node: first primary component, or first overall
    primary = [c for c in components if c.evaluation_role == "primary"]
    root_node = (primary[0] if primary else components[0]).name if components else ""

    sid = sample_id or usd_path.stem

    n_primary = len([c for c in components if c.evaluation_role == "primary"])
    n_distractor = len([c for c in components if c.evaluation_role == "distractor"])
    logger.info(
        "Extracted %d components (%d primary, %d distractor) from %s",
        len(components),
        n_primary,
        n_distractor,
        usd_path.name,
    )

    return SceneJSON(
        sample_id=sid,
        benchmark_tier="closed_world",
        family="warehouse",
        template_id="usd_import",
        root_node=root_node,
        template_params={"source_usd": str(usd_path)},
        camera=CameraParams(
            position=[0.0, 0.0, 5.0],
            target=[0.0, 0.0, 0.0],
            fov_deg=60.0,
        ),
        components=components,
        relations=[],
        taxonomy_version=taxonomy.version,
        generator_version="gt_extract_v1",
    )
