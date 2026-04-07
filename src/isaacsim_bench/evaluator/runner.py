from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaacsim_bench.evaluator.matching import MatchResult, match_components
from isaacsim_bench.evaluator.metrics.component import compute_component_metrics
from isaacsim_bench.evaluator.metrics.coverage import compute_coverage_metrics
from isaacsim_bench.evaluator.metrics.placement import compute_placement_metrics
from isaacsim_bench.evaluator.metrics.relation import compute_relation_metrics
from isaacsim_bench.evaluator.metrics.retrieval import compute_retrieval_metrics
from isaacsim_bench.evaluator.metrics.scene_success import compute_scene_success
from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry


@dataclass
class EvaluationReport:
    retrieval: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    component: dict = field(default_factory=dict)
    relation: dict = field(default_factory=dict)
    placement: dict = field(default_factory=dict)
    scene_success: dict = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval,
            "coverage": self.coverage,
            "component": {k: v for k, v in self.component.items() if k != "_matches"},
            "relation": self.relation,
            "placement": self.placement,
            "scene_success": {
                k: v for k, v in self.scene_success.items() if k != "per_scene"
            },
        }


class EvaluatorRunner:
    def evaluate(
        self,
        gt_scenes: list[SceneJSON],
        pred_scenes: list[PredictionJSON],
        registry: TaxonomyRegistry,
    ) -> EvaluationReport:
        # Component matching (needed by relation and placement)
        comp_metrics = compute_component_metrics(
            gt_scenes, pred_scenes, mode="exact", registry=registry
        )
        matches: list[MatchResult] = comp_metrics.pop("_matches")

        report = EvaluationReport(
            retrieval=compute_retrieval_metrics(gt_scenes, pred_scenes, registry),
            coverage=compute_coverage_metrics(gt_scenes, pred_scenes),
            component=comp_metrics,
            relation=compute_relation_metrics(gt_scenes, pred_scenes, matches),
            placement=compute_placement_metrics(gt_scenes, pred_scenes, matches),
            scene_success=compute_scene_success(gt_scenes, pred_scenes, matches),
        )
        return report

    def evaluate_from_dirs(
        self,
        gt_dir: Path,
        pred_dir: Path,
        registry: TaxonomyRegistry,
    ) -> EvaluationReport:
        gt_scenes: list[SceneJSON] = []
        pred_scenes: list[PredictionJSON] = []

        for scene_path in sorted(gt_dir.rglob("scene.json")):
            scene = SceneJSON.model_validate_json(scene_path.read_text())
            pred_path = pred_dir / f"{scene.sample_id}.json"
            if not pred_path.exists():
                # Try matching directory structure
                pred_path = pred_dir / scene_path.parent.name / "prediction.json"
            if pred_path.exists():
                pred = PredictionJSON.model_validate_json(pred_path.read_text())
                gt_scenes.append(scene)
                pred_scenes.append(pred)

        return self.evaluate(gt_scenes, pred_scenes, registry)
