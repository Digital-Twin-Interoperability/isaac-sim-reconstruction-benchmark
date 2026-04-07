from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry
from isaacsim_bench.validator.checks import (
    CheckResult,
    check_asset_existence,
    check_camera_validity,
    check_file_completeness,
    check_match_regime_consistency,
    check_relation_connectivity,
    check_transform_sanity,
)


@dataclass
class ValidationReport:
    sample_id: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed or r.severity == "warning" for r in self.results)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.passed and r.severity == "warning")


class ValidatorRunner:
    def validate_sample(
        self, sample_dir: Path, registry: TaxonomyRegistry
    ) -> ValidationReport:
        results: list[CheckResult] = []

        # Check 6: file completeness (runs even if scene.json is missing)
        results.extend(check_file_completeness(sample_dir))

        scene_path = sample_dir / "scene.json"
        if not scene_path.exists():
            return ValidationReport(sample_id=sample_dir.name, results=results)

        scene = SceneJSON.model_validate_json(scene_path.read_text())

        results.extend(check_transform_sanity(scene))
        results.extend(check_relation_connectivity(scene))
        results.extend(check_camera_validity(scene))
        results.extend(check_asset_existence(scene, registry))
        results.extend(check_match_regime_consistency(scene, registry))

        return ValidationReport(sample_id=scene.sample_id, results=results)

    def validate_dataset(
        self, dataset_dir: Path, registry: TaxonomyRegistry
    ) -> list[ValidationReport]:
        reports: list[ValidationReport] = []
        for sub in sorted(dataset_dir.iterdir()):
            if sub.is_dir():
                reports.append(self.validate_sample(sub, registry))
        return reports
