import json
from pathlib import Path

import numpy as np
import pytest

from isaacsim_bench.schemas import SceneJSON
from isaacsim_bench.taxonomy import TaxonomyRegistry
from isaacsim_bench.validator.checks import (
    check_asset_existence,
    check_camera_validity,
    check_file_completeness,
    check_match_regime_consistency,
    check_relation_connectivity,
    check_transform_sanity,
)
from isaacsim_bench.validator.runner import ValidatorRunner


@pytest.fixture()
def registry(sample_taxonomy, sample_world_pool, sample_retrieval_pool):
    return TaxonomyRegistry(sample_taxonomy, sample_world_pool, sample_retrieval_pool)


@pytest.fixture()
def scene(sample_scene_dict) -> SceneJSON:
    return SceneJSON.model_validate(sample_scene_dict)


# ---- Check 1: Transform sanity ----

class TestTransformSanity:
    def test_no_overlap(self, scene):
        results = check_transform_sanity(scene, default_extent=0.5)
        assert all(r.passed for r in results)

    def test_overlap_detected(self, sample_scene_dict):
        # Place two components at the same position
        sample_scene_dict["components"][1]["translate"] = [0.0, 0.0, 0.0]
        scene = SceneJSON.model_validate(sample_scene_dict)
        results = check_transform_sanity(scene, default_extent=0.5)
        errors = [r for r in results if not r.passed]
        assert len(errors) >= 1


# ---- Check 2: Relation connectivity ----

class TestRelationConnectivity:
    def test_connected(self, scene):
        results = check_relation_connectivity(scene)
        assert len(results) == 0

    def test_disconnected(self, sample_scene_dict):
        # Remove all relations -> disconnected
        sample_scene_dict["relations"] = []
        scene = SceneJSON.model_validate(sample_scene_dict)
        results = check_relation_connectivity(scene)
        assert len(results) == 1
        assert not results[0].passed


# ---- Check 3: Camera validity ----

class TestCameraValidity:
    def test_valid_camera(self, scene):
        results = check_camera_validity(scene)
        assert all(r.passed for r in results) or len(results) == 0

    def test_degenerate_camera(self, sample_scene_dict):
        sample_scene_dict["camera"]["target"] = sample_scene_dict["camera"]["position"]
        scene = SceneJSON.model_validate(sample_scene_dict)
        results = check_camera_validity(scene)
        assert any(not r.passed for r in results)


# ---- Check 4: Asset existence ----

class TestAssetExistence:
    def test_all_exist(self, scene, registry):
        results = check_asset_existence(scene, registry)
        assert len(results) == 0

    def test_missing_asset(self, sample_scene_dict, registry):
        sample_scene_dict["components"][0]["asset_id"] = "NonExistent_X99"
        scene = SceneJSON.model_validate(sample_scene_dict)
        results = check_asset_existence(scene, registry)
        assert len(results) == 1
        assert not results[0].passed


# ---- Check 5: Match regime consistency ----

class TestMatchRegimeConsistency:
    def test_consistent(self, scene, registry):
        results = check_match_regime_consistency(scene, registry)
        assert len(results) == 0

    def test_inconsistent(self, sample_scene_dict, registry):
        sample_scene_dict["components"][0]["match_regime"] = "proxy_match"
        scene = SceneJSON.model_validate(sample_scene_dict)
        results = check_match_regime_consistency(scene, registry)
        assert len(results) >= 1
        assert not results[0].passed


# ---- Check 6: File completeness ----

class TestFileCompleteness:
    def test_complete(self, tmp_path):
        (tmp_path / "rgb.png").write_bytes(b"fake")
        (tmp_path / "depth.npy").write_bytes(b"fake")
        (tmp_path / "segmentation.png").write_bytes(b"fake")
        (tmp_path / "scene.json").write_text("{}")
        results = check_file_completeness(tmp_path)
        assert all(r.passed for r in results)

    def test_missing_files(self, tmp_path):
        results = check_file_completeness(tmp_path)
        errors = [r for r in results if not r.passed]
        assert len(errors) >= 3  # rgb, depth, segmentation, scene.json


# ---- ValidatorRunner integration ----

class TestValidatorRunner:
    def test_validate_sample(self, tmp_path, sample_scene_dict, registry):
        (tmp_path / "rgb.png").write_bytes(b"fake")
        (tmp_path / "depth.npy").write_bytes(b"fake")
        (tmp_path / "segmentation.png").write_bytes(b"fake")
        (tmp_path / "scene.json").write_text(
            SceneJSON.model_validate(sample_scene_dict).model_dump_json(by_alias=True)
        )
        runner = ValidatorRunner()
        report = runner.validate_sample(tmp_path, registry)
        assert report.passed
        assert report.error_count == 0
