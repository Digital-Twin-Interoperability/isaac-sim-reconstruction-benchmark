"""Tests for ground-truth extraction from USD files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaacsim_bench.extractor.usd_parser import (
    _build_taxonomy_index,
    _resolve_ref,
    _semantic_from_name,
    extract_ground_truth,
)
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy, Category, Variant


# ---------------------------------------------------------------------------
# Taxonomy index tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_taxonomy() -> AssetTaxonomy:
    return AssetTaxonomy(
        version="test",
        categories=[
            Category(
                category_id="barrel_plastic",
                name="Plastic Barrel",
                family="barrel",
                variants=[
                    Variant(
                        variant_id="BarelPlastic_A_01",
                        name="BarelPlastic A 01",
                        usd_path="Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
                        semantic_class="barrel",
                    ),
                ],
            ),
            Category(
                category_id="floor_tile",
                name="Floor Tile",
                family="building",
                variants=[
                    Variant(
                        variant_id="floor02",
                        name="floor02",
                        usd_path="Isaac/Environments/Simple_Warehouse/Props/SM_floor02.usd",
                        semantic_class="floor",
                    ),
                ],
            ),
        ],
    )


class TestBuildTaxonomyIndex:
    def test_by_path_lookup(self, mini_taxonomy):
        by_path, _ = _build_taxonomy_index(mini_taxonomy)
        key = "Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd"
        assert key in by_path
        assert by_path[key]["variant_id"] == "BarelPlastic_A_01"
        assert by_path[key]["family"] == "barrel"

    def test_by_stem_lookup(self, mini_taxonomy):
        _, by_stem = _build_taxonomy_index(mini_taxonomy)
        assert "SM_BarelPlastic_A_01" in by_stem
        assert "SM_floor02" in by_stem


class TestResolveRef:
    def test_direct_path_match(self, mini_taxonomy):
        by_path, by_stem = _build_taxonomy_index(mini_taxonomy)
        url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd"
        info = _resolve_ref(url, by_path, by_stem)
        assert info is not None
        assert info["variant_id"] == "BarelPlastic_A_01"

    def test_suffix_stripped_match(self, mini_taxonomy):
        by_path, by_stem = _build_taxonomy_index(mini_taxonomy)
        # SM_floor02_218 → strips _218 → matches SM_floor02
        url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/SM_floor02_218.usd"
        info = _resolve_ref(url, by_path, by_stem)
        assert info is not None
        assert info["variant_id"] == "floor02"

    def test_no_match(self, mini_taxonomy):
        by_path, by_stem = _build_taxonomy_index(mini_taxonomy)
        info = _resolve_ref("https://example.com/NonExistent.usd", by_path, by_stem)
        assert info is None


class TestSemanticFromName:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("SM_RackShelf_01_135", "rack_shelf"),
            ("SM_CardBoxA_01_301", "box"),
            ("SM_floor27", "floor"),
            ("forklift", "forklift"),
            ("SM_BarelPlastic_A_01_1487", "barrel"),
            ("SM_CeilingA_6X14", "ceiling"),
            ("SM_TrafficCone_5", "traffic_cone"),
            ("GroundPlane", "unknown"),
        ],
    )
    def test_pattern_match(self, name, expected):
        assert _semantic_from_name(name) == expected


# ---------------------------------------------------------------------------
# Full extraction test (requires full_warehouse.usd)
# ---------------------------------------------------------------------------


_USD_PATH = Path("full_warehouse.usd")
_TAXONOMY_PATH = Path("data/asset_taxonomy.json")


@pytest.mark.skipif(
    not _USD_PATH.exists() or not _TAXONOMY_PATH.exists(),
    reason="Requires full_warehouse.usd and asset_taxonomy.json",
)
class TestExtractFullWarehouse:
    def test_extraction_produces_valid_scene(self):
        scene = extract_ground_truth(_USD_PATH, _TAXONOMY_PATH)
        assert scene.sample_id == "full_warehouse"
        assert scene.family == "warehouse"
        assert len(scene.components) > 3000

    def test_all_components_have_pose(self):
        scene = extract_ground_truth(_USD_PATH, _TAXONOMY_PATH)
        for comp in scene.components:
            assert len(comp.translate) == 3
            assert len(comp.orientation_xyzw) == 4

    def test_primary_and_distractor_roles(self):
        scene = extract_ground_truth(_USD_PATH, _TAXONOMY_PATH)
        primary = [c for c in scene.components if c.evaluation_role == "primary"]
        distractor = [c for c in scene.components if c.evaluation_role == "distractor"]
        assert len(primary) > 2000
        assert len(distractor) > 100

    def test_no_structural_flag(self):
        full = extract_ground_truth(_USD_PATH, _TAXONOMY_PATH, include_structural=True)
        props_only = extract_ground_truth(_USD_PATH, _TAXONOMY_PATH, include_structural=False)
        assert len(props_only.components) < len(full.components)
        for c in props_only.components:
            assert c.evaluation_role == "primary"

    def test_roundtrip_json(self):
        scene = extract_ground_truth(_USD_PATH, _TAXONOMY_PATH)
        json_str = scene.model_dump_json(indent=2, by_alias=True)
        parsed = json.loads(json_str)
        assert parsed["sample_id"] == "full_warehouse"
        assert len(parsed["components"]) == len(scene.components)
