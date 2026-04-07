import json

import pytest
from pydantic import ValidationError

from isaacsim_bench.schemas import (
    AssetTaxonomy,
    BenchmarkManifest,
    ComponentEntry,
    PoolDefinition,
    PredictionJSON,
    RelationEntry,
    SceneJSON,
    SplitConfig,
)


class TestSceneJSON:
    def test_round_trip(self, sample_scene_dict: dict):
        scene = SceneJSON.model_validate(sample_scene_dict)
        dumped = json.loads(scene.model_dump_json(by_alias=True))
        scene2 = SceneJSON.model_validate(dumped)
        assert scene2.sample_id == scene.sample_id
        assert len(scene2.components) == 3
        assert len(scene2.relations) == 2

    def test_relation_alias(self, sample_scene_dict: dict):
        scene = SceneJSON.model_validate(sample_scene_dict)
        rel = scene.relations[0]
        assert rel.from_node == "Straight_Top"
        assert rel.to_node == "Curve_180"
        dumped = json.loads(rel.model_dump_json(by_alias=True))
        assert "from" in dumped
        assert "to" in dumped
        assert "from_node" not in dumped

    def test_root_node_validation(self, sample_scene_dict: dict):
        sample_scene_dict["root_node"] = "NonExistent"
        with pytest.raises(ValidationError, match="root_node"):
            SceneJSON.model_validate(sample_scene_dict)

    def test_invalid_tier(self, sample_scene_dict: dict):
        sample_scene_dict["benchmark_tier"] = "invalid"
        with pytest.raises(ValidationError):
            SceneJSON.model_validate(sample_scene_dict)

    def test_invalid_role(self, sample_scene_dict: dict):
        sample_scene_dict["components"][0]["evaluation_role"] = "secondary"
        with pytest.raises(ValidationError):
            SceneJSON.model_validate(sample_scene_dict)

    def test_invalid_match_regime(self, sample_scene_dict: dict):
        sample_scene_dict["components"][0]["match_regime"] = "partial"
        with pytest.raises(ValidationError):
            SceneJSON.model_validate(sample_scene_dict)


class TestTaxonomy:
    def test_taxonomy_loads(self, sample_taxonomy: AssetTaxonomy):
        assert sample_taxonomy.version == "v1"
        assert len(sample_taxonomy.categories) == 4

    def test_taxonomy_round_trip(self, sample_taxonomy: AssetTaxonomy):
        dumped = json.loads(sample_taxonomy.model_dump_json())
        loaded = AssetTaxonomy.model_validate(dumped)
        assert len(loaded.categories) == len(sample_taxonomy.categories)


class TestPoolDefinition:
    def test_pool_loads(self, sample_world_pool: PoolDefinition):
        assert len(sample_world_pool.asset_ids) == 9

    def test_empty_pool(self):
        pool = PoolDefinition(version="v0", asset_ids=[])
        assert len(pool.asset_ids) == 0


class TestPredictionJSON:
    def test_minimal(self):
        pred = PredictionJSON(
            sample_id="test_001",
            components=[],
            relations=[],
        )
        assert pred.abstained is False
        assert pred.latency_seconds == 0.0


class TestBenchmarkManifest:
    def test_manifest(self):
        m = BenchmarkManifest(
            benchmark_version="v1",
            asset_taxonomy_version="v1",
            world_pool_version="v1",
            retrieval_pool_version="v1",
            generator_version="abc123",
            created_at="2026-04-05",
            sample_count=900,
        )
        assert m.sample_count == 900


class TestSplitConfig:
    def test_split_config(self):
        cfg = SplitConfig(
            assignments=[
                {"sample_id": "s001", "split": "train", "holdout_tags": []},
                {"sample_id": "s002", "split": "test", "holdout_tags": ["camera_angle"]},
            ]
        )
        assert len(cfg.assignments) == 2

    def test_invalid_split(self):
        with pytest.raises(ValidationError):
            SplitConfig(
                assignments=[
                    {"sample_id": "s001", "split": "holdout", "holdout_tags": []},
                ]
            )
