import json
from pathlib import Path

from click.testing import CliRunner

from isaacsim_bench.cli.main import cli
from isaacsim_bench.schemas.scene import SceneJSON


def _write_data_files(tmp_path: Path):
    """Write taxonomy, world pool, retrieval pool to tmp_path."""
    taxonomy = {
        "version": "v1",
        "categories": [
            {
                "category_id": "conveyor_straight",
                "name": "Straight Conveyor",
                "family": "conveyor",
                "variants": [
                    {"variant_id": "ConveyorBelt_A01", "name": "A01"},
                    {"variant_id": "ConveyorBelt_A02", "name": "A02"},
                ],
            }
        ],
    }
    world_pool = {"version": "v1", "asset_ids": ["ConveyorBelt_A01", "ConveyorBelt_A02"]}
    retrieval_pool = {"version": "v1", "asset_ids": ["ConveyorBelt_A01", "ConveyorBelt_A02"]}

    (tmp_path / "taxonomy.json").write_text(json.dumps(taxonomy))
    (tmp_path / "world_pool.json").write_text(json.dumps(world_pool))
    (tmp_path / "retrieval_pool.json").write_text(json.dumps(retrieval_pool))

    return (
        str(tmp_path / "taxonomy.json"),
        str(tmp_path / "world_pool.json"),
        str(tmp_path / "retrieval_pool.json"),
    )


def _write_sample(sample_dir: Path):
    """Write a complete valid sample."""
    scene_data = {
        "sample_id": "test_001",
        "benchmark_tier": "closed_world",
        "family": "conveyor",
        "template_id": "u_conveyor",
        "root_node": "A",
        "template_params": {},
        "camera": {"position": [5, 5, 5], "target": [0, 0, 0], "fov_deg": 60},
        "components": [
            {
                "name": "A",
                "asset_id": "ConveyorBelt_A01",
                "asset_name": "A01",
                "family": "conveyor_straight",
                "evaluation_role": "primary",
                "match_regime": "exact_match",
                "translate": [0, 0, 0],
                "orientation_xyzw": [0, 0, 0, 1],
            }
        ],
        "relations": [],
    }
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "scene.json").write_text(json.dumps(scene_data))
    (sample_dir / "rgb.png").write_bytes(b"fake")
    (sample_dir / "depth.npy").write_bytes(b"fake")
    (sample_dir / "segmentation.png").write_bytes(b"fake")


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Isaac Sim Benchmark CLI" in result.output

    def test_taxonomy_validate(self, tmp_path):
        tax_path, _, _ = _write_data_files(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["taxonomy", "validate", "--taxonomy", tax_path])
        assert result.exit_code == 0
        assert "Valid taxonomy" in result.output

    def test_taxonomy_stats(self, tmp_path):
        tax_path, _, _ = _write_data_files(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["taxonomy", "stats", "--taxonomy", tax_path])
        assert result.exit_code == 0
        assert "conveyor_straight" in result.output

    def test_validate_sample(self, tmp_path):
        tax_path, wp_path, rp_path = _write_data_files(tmp_path)
        sample_dir = tmp_path / "sample_001"
        _write_sample(sample_dir)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate-sample", str(sample_dir),
            "--taxonomy", tax_path,
            "--world-pool", wp_path,
            "--retrieval-pool", rp_path,
        ])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_validate_dataset(self, tmp_path):
        tax_path, wp_path, rp_path = _write_data_files(tmp_path)
        dataset_dir = tmp_path / "dataset"
        _write_sample(dataset_dir / "sample_001")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", str(dataset_dir),
            "--taxonomy", tax_path,
            "--world-pool", wp_path,
            "--retrieval-pool", rp_path,
        ])
        assert result.exit_code == 0
        assert "1 passed" in result.output
