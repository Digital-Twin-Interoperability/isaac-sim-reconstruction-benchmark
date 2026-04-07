import pytest

from isaacsim_bench.schemas import (
    AssetTaxonomy,
    Category,
    ComponentEntry,
    PoolDefinition,
    Variant,
)


@pytest.fixture()
def sample_taxonomy() -> AssetTaxonomy:
    return AssetTaxonomy(
        version="v1",
        categories=[
            Category(
                category_id="conveyor_straight",
                name="Straight Conveyor",
                family="conveyor",
                variants=[
                    Variant(variant_id="ConveyorBelt_A01", name="ConveyorBelt A01"),
                    Variant(variant_id="ConveyorBelt_A02", name="ConveyorBelt A02"),
                    Variant(variant_id="ConveyorBelt_A03", name="ConveyorBelt A03"),
                ],
            ),
            Category(
                category_id="conveyor_curve",
                name="Curve Conveyor",
                family="conveyor",
                variants=[
                    Variant(variant_id="ConveyorBelt_A10", name="ConveyorBelt A10"),
                    Variant(variant_id="ConveyorBelt_A11", name="ConveyorBelt A11"),
                ],
            ),
            Category(
                category_id="shelf_unit",
                name="Shelf Unit",
                family="shelf",
                variants=[
                    Variant(variant_id="Shelf_A01", name="Shelf A01"),
                    Variant(variant_id="Shelf_A02", name="Shelf A02"),
                ],
            ),
            Category(
                category_id="pallet_standard",
                name="Standard Pallet",
                family="pallet",
                variants=[
                    Variant(variant_id="Pallet_A01", name="Pallet A01"),
                    Variant(variant_id="Pallet_A02", name="Pallet A02"),
                ],
            ),
        ],
    )


@pytest.fixture()
def sample_world_pool() -> PoolDefinition:
    return PoolDefinition(
        version="v1",
        asset_ids=[
            "ConveyorBelt_A01",
            "ConveyorBelt_A02",
            "ConveyorBelt_A03",
            "ConveyorBelt_A10",
            "ConveyorBelt_A11",
            "Shelf_A01",
            "Shelf_A02",
            "Pallet_A01",
            "Pallet_A02",
        ],
    )


@pytest.fixture()
def sample_retrieval_pool() -> PoolDefinition:
    """Retrieval pool withholds A03, A11, Shelf_A02 to create proxy_match."""
    return PoolDefinition(
        version="v1",
        asset_ids=[
            "ConveyorBelt_A01",
            "ConveyorBelt_A02",
            "ConveyorBelt_A10",
            "Shelf_A01",
            "Pallet_A01",
            "Pallet_A02",
        ],
    )


def _make_component(
    name: str,
    asset_id: str,
    family: str,
    role: str = "primary",
    regime: str = "exact_match",
    translate: list[float] | None = None,
    orientation_xyzw: list[float] | None = None,
) -> ComponentEntry:
    return ComponentEntry(
        name=name,
        asset_id=asset_id,
        asset_name=asset_id.replace("_", " "),
        family=family,
        evaluation_role=role,
        match_regime=regime,
        translate=translate or [0.0, 0.0, 0.0],
        orientation_xyzw=orientation_xyzw or [0.0, 0.0, 0.0, 1.0],
    )


@pytest.fixture()
def sample_scene_dict() -> dict:
    """A minimal U-conveyor scene as a raw dict (for JSON round-trip tests)."""
    return {
        "sample_id": "conveyor_u_000001",
        "benchmark_tier": "closed_world",
        "family": "conveyor",
        "template_id": "u_conveyor",
        "root_node": "Straight_Top",
        "template_params": {"length": 3},
        "camera": {
            "position": [5.0, 5.0, 5.0],
            "target": [0.0, 0.0, 0.0],
            "fov_deg": 60.0,
        },
        "components": [
            {
                "name": "Straight_Top",
                "asset_id": "ConveyorBelt_A01",
                "asset_name": "ConveyorBelt A01",
                "family": "conveyor_straight",
                "evaluation_role": "primary",
                "match_regime": "exact_match",
                "translate": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "name": "Curve_180",
                "asset_id": "ConveyorBelt_A10",
                "asset_name": "ConveyorBelt A10",
                "family": "conveyor_curve",
                "evaluation_role": "primary",
                "match_regime": "exact_match",
                "translate": [2.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
            },
            {
                "name": "Straight_Bottom",
                "asset_id": "ConveyorBelt_A02",
                "asset_name": "ConveyorBelt A02",
                "family": "conveyor_straight",
                "evaluation_role": "primary",
                "match_regime": "exact_match",
                "translate": [0.0, -2.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
            },
        ],
        "relations": [
            {
                "type": "attach",
                "from": "Straight_Top",
                "to": "Curve_180",
                "from_anchor": "right_end",
                "to_anchor": "curve_entry",
            },
            {
                "type": "attach",
                "from": "Curve_180",
                "to": "Straight_Bottom",
                "from_anchor": "curve_exit",
                "to_anchor": "right_end",
            },
        ],
    }
