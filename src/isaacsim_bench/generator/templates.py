"""Scene template definitions.

Each template is a callable that takes (params, registry) and returns
the components, relations, root_node, and camera needed to populate a SceneJSON.
"""

from __future__ import annotations

import math
from typing import Any

from isaacsim_bench.schemas.scene import CameraParams, ComponentEntry, RelationEntry
from isaacsim_bench.taxonomy.match_regime import derive_component_regime
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry

# ---------------------------------------------------------------------------
# Template result type
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class TemplateResult:
    family: str
    root_node: str
    components: list[ComponentEntry]
    relations: list[RelationEntry]
    camera: CameraParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quat_from_z_deg(deg: float) -> list[float]:
    """Return [x, y, z, w] quaternion for a rotation of *deg* about Z."""
    half = math.radians(deg) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _make_component(
    name: str,
    asset_id: str,
    family: str,
    registry: TaxonomyRegistry,
    translate: list[float],
    orientation_xyzw: list[float] | None = None,
    role: str = "primary",
) -> ComponentEntry:
    return ComponentEntry(
        name=name,
        asset_id=asset_id,
        asset_name=registry.get_variant(asset_id).name
        if registry.get_variant(asset_id)
        else asset_id,
        family=family,
        evaluation_role=role,
        match_regime=derive_component_regime(registry, asset_id),
        translate=translate,
        orientation_xyzw=orientation_xyzw or [0.0, 0.0, 0.0, 1.0],
    )


# ---------------------------------------------------------------------------
# u_conveyor — a U-shaped conveyor: straight → 180° curve → straight return
# ---------------------------------------------------------------------------


def build_u_conveyor(
    params: dict[str, Any], registry: TaxonomyRegistry
) -> TemplateResult:
    """Build a U-shaped conveyor scene.

    Params
    ------
    straight_asset : str, default "ConveyorBelt_A01"
        Asset ID for the straight segments.
    curve_asset : str, default "ConveyorBelt_A10"
        Asset ID for the 180° curve.
    return_asset : str, default "ConveyorBelt_A02"
        Asset ID for the return straight.
    segment_length : float, default 2.0
        Spacing between segments along the X axis.
    """
    straight_id = params.get("straight_asset", "ConveyorBelt_A01")
    curve_id = params.get("curve_asset", "ConveyorBelt_A10")
    return_id = params.get("return_asset", "ConveyorBelt_A02")
    seg_len = params.get("segment_length", 2.0)

    top = _make_component(
        "Straight_Top",
        straight_id,
        "conveyor_straight",
        registry,
        translate=[0.0, 0.0, 0.0],
    )
    curve = _make_component(
        "Curve_180",
        curve_id,
        "conveyor_curve",
        registry,
        translate=[seg_len, 0.0, 0.0],
        orientation_xyzw=_quat_from_z_deg(180.0),
    )
    bottom = _make_component(
        "Straight_Bottom",
        return_id,
        "conveyor_straight",
        registry,
        translate=[0.0, -seg_len, 0.0],
        orientation_xyzw=_quat_from_z_deg(180.0),
    )

    relations = [
        RelationEntry(
            type="attach",
            from_node="Straight_Top",
            to_node="Curve_180",
            from_anchor="right_end",
            to_anchor="curve_entry",
        ),
        RelationEntry(
            type="attach",
            from_node="Curve_180",
            to_node="Straight_Bottom",
            from_anchor="curve_exit",
            to_anchor="right_end",
        ),
    ]

    # Camera positioned above and to the side, looking at scene centre
    cx = seg_len / 2
    cy = -seg_len / 2
    camera = CameraParams(
        position=[cx + 5.0, cy + 5.0, 5.0],
        target=[cx, cy, 0.0],
        fov_deg=60.0,
    )

    return TemplateResult(
        family="conveyor",
        root_node="Straight_Top",
        components=[top, curve, bottom],
        relations=relations,
        camera=camera,
    )


# ---------------------------------------------------------------------------
# shelf_row — a row of N shelves side-by-side
# ---------------------------------------------------------------------------


def build_shelf_row(
    params: dict[str, Any], registry: TaxonomyRegistry
) -> TemplateResult:
    """Build a row of shelves.

    Params
    ------
    shelf_asset : str, default "Shelf_A01"
    count : int, default 3
    spacing : float, default 1.5
    """
    shelf_id = params.get("shelf_asset", "Shelf_A01")
    count = params.get("count", 3)
    spacing = params.get("spacing", 1.5)

    components: list[ComponentEntry] = []
    relations: list[RelationEntry] = []

    for i in range(count):
        name = f"Shelf_{i:02d}"
        comp = _make_component(
            name,
            shelf_id,
            "shelf_unit",
            registry,
            translate=[i * spacing, 0.0, 0.0],
        )
        components.append(comp)

        if i > 0:
            relations.append(
                RelationEntry(
                    type="adjacent",
                    from_node=f"Shelf_{i - 1:02d}",
                    to_node=name,
                    from_anchor="right_side",
                    to_anchor="left_side",
                )
            )

    cx = (count - 1) * spacing / 2
    camera = CameraParams(
        position=[cx, 5.0, 3.0],
        target=[cx, 0.0, 0.0],
        fov_deg=60.0,
    )

    return TemplateResult(
        family="shelf",
        root_node="Shelf_00",
        components=components,
        relations=relations,
        camera=camera,
    )


# ---------------------------------------------------------------------------
# pallet_grid — an M×N grid of pallets
# ---------------------------------------------------------------------------


def build_pallet_grid(
    params: dict[str, Any], registry: TaxonomyRegistry
) -> TemplateResult:
    """Build a grid of pallets.

    Params
    ------
    pallet_asset : str, default "Pallet_A01"
    rows : int, default 2
    cols : int, default 3
    spacing : float, default 1.2
    """
    pallet_id = params.get("pallet_asset", "Pallet_A01")
    rows = params.get("rows", 2)
    cols = params.get("cols", 3)
    spacing = params.get("spacing", 1.2)

    components: list[ComponentEntry] = []
    relations: list[RelationEntry] = []

    for r in range(rows):
        for c in range(cols):
            name = f"Pallet_r{r}_c{c}"
            comp = _make_component(
                name,
                pallet_id,
                "pallet_standard",
                registry,
                translate=[c * spacing, r * spacing, 0.0],
            )
            components.append(comp)

            # Row adjacency
            if c > 0:
                relations.append(
                    RelationEntry(
                        type="adjacent",
                        from_node=f"Pallet_r{r}_c{c - 1}",
                        to_node=name,
                        from_anchor="right_side",
                        to_anchor="left_side",
                    )
                )
            # Column adjacency
            if r > 0:
                relations.append(
                    RelationEntry(
                        type="adjacent",
                        from_node=f"Pallet_r{r - 1}_c{c}",
                        to_node=name,
                        from_anchor="front_side",
                        to_anchor="back_side",
                    )
                )

    cx = (cols - 1) * spacing / 2
    cy = (rows - 1) * spacing / 2
    camera = CameraParams(
        position=[cx, cy + 5.0, 5.0],
        target=[cx, cy, 0.0],
        fov_deg=60.0,
    )

    return TemplateResult(
        family="pallet",
        root_node="Pallet_r0_c0",
        components=components,
        relations=relations,
        camera=camera,
    )


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, Any] = {
    "u_conveyor": build_u_conveyor,
    "shelf_row": build_shelf_row,
    "pallet_grid": build_pallet_grid,
}
