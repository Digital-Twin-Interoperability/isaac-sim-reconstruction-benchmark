import pytest

from isaacsim_bench.evaluator.matching import MatchResult
from isaacsim_bench.evaluator.metrics.component import compute_component_metrics
from isaacsim_bench.evaluator.metrics.coverage import compute_coverage_metrics
from isaacsim_bench.evaluator.metrics.placement import compute_placement_metrics
from isaacsim_bench.evaluator.metrics.relation import compute_relation_metrics
from isaacsim_bench.evaluator.metrics.retrieval import compute_retrieval_metrics
from isaacsim_bench.evaluator.metrics.scene_success import compute_scene_success
from isaacsim_bench.evaluator.runner import EvaluatorRunner
from isaacsim_bench.schemas.prediction import (
    PredictedComponent,
    PredictedRelation,
    PredictionJSON,
)
from isaacsim_bench.schemas.scene import (
    CameraParams,
    ComponentEntry,
    RelationEntry,
    SceneJSON,
)
from isaacsim_bench.taxonomy import TaxonomyRegistry


def _scene(
    components: list[ComponentEntry],
    relations: list[RelationEntry] | None = None,
    family: str = "conveyor",
    template_id: str = "u_conveyor",
) -> SceneJSON:
    return SceneJSON(
        sample_id="test_001",
        benchmark_tier="closed_world",
        family=family,
        template_id=template_id,
        root_node=components[0].name,
        template_params={},
        camera=CameraParams(position=[5, 5, 5], target=[0, 0, 0], fov_deg=60),
        components=components,
        relations=relations or [],
    )


def _comp(name, asset_id, family="conveyor_straight", role="primary", regime="exact_match", t=None):
    return ComponentEntry(
        name=name, asset_id=asset_id, asset_name=asset_id,
        family=family, evaluation_role=role, match_regime=regime,
        translate=t or [0, 0, 0], orientation_xyzw=[0, 0, 0, 1],
    )


def _pred_comp(name, asset_id, family="conveyor_straight", t=None):
    return PredictedComponent(
        name=name, asset_id=asset_id, family=family,
        translate=t or [0, 0, 0], orientation_xyzw=[0, 0, 0, 1],
    )


def _rel(from_node, to_node, from_anchor="a", to_anchor="b"):
    return RelationEntry.model_validate({
        "type": "attach", "from": from_node, "to": to_node,
        "from_anchor": from_anchor, "to_anchor": to_anchor,
    })


def _pred_rel(from_node, to_node, from_anchor="a", to_anchor="b"):
    return PredictedRelation(
        type="attach", from_node=from_node, to_node=to_node,
        from_anchor=from_anchor, to_anchor=to_anchor,
    )


def _prediction(
    components=None, relations=None, abstained=False,
):
    return PredictionJSON(
        sample_id="test_001",
        components=components or [],
        relations=relations or [],
        abstained=abstained,
    )


# ---- Component metrics ----

class TestComponentMetrics:
    def test_perfect(self):
        gt = [_scene([_comp("a", "X1"), _comp("b", "X2")])]
        pred = [_prediction(components=[_pred_comp("a", "X1"), _pred_comp("b", "X2")])]
        m = compute_component_metrics(gt, pred, mode="exact")
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_partial(self):
        gt = [_scene([_comp("a", "X1"), _comp("b", "X2")])]
        pred = [_prediction(components=[_pred_comp("a", "X1")])]
        m = compute_component_metrics(gt, pred, mode="exact")
        assert m["recall"] == 0.5
        assert m["precision"] == 1.0


# ---- Relation metrics ----

class TestRelationMetrics:
    def test_perfect_oracle(self):
        gt_comps = [_comp("a", "X1", t=[0, 0, 0]), _comp("b", "X2", t=[1, 0, 0])]
        gt_rels = [_rel("a", "b")]
        gt = [_scene(gt_comps, gt_rels)]

        pred = [_prediction(
            components=[_pred_comp("a", "X1"), _pred_comp("b", "X2")],
            relations=[_pred_rel("a", "b")],
        )]

        matches = [MatchResult(matched_pairs=[(0, 0), (1, 1)])]
        m = compute_relation_metrics(gt, pred, matches)
        assert m["oracle_components"]["f1"] == 1.0
        assert m["strict"]["f1"] == 1.0

    def test_strict_penalizes_unmatched_components(self):
        gt_comps = [_comp("a", "X1"), _comp("b", "X2")]
        gt_rels = [_rel("a", "b")]
        gt = [_scene(gt_comps, gt_rels)]

        pred = [_prediction(
            components=[_pred_comp("a", "X1")],
            relations=[],
        )]

        # Only "a" matched, "b" unmatched -> relation "a->b" is FN in strict
        matches = [MatchResult(matched_pairs=[(0, 0)], unmatched_gt=[1])]
        m = compute_relation_metrics(gt, pred, matches)
        assert m["strict"]["recall"] == 0.0

    def test_anchor_accuracy(self):
        gt_comps = [_comp("a", "X1"), _comp("b", "X2")]
        gt = [_scene(gt_comps, [_rel("a", "b", "left", "right")])]
        pred = [_prediction(
            components=[_pred_comp("a", "X1"), _pred_comp("b", "X2")],
            relations=[_pred_rel("a", "b", "left", "wrong")],
        )]
        matches = [MatchResult(matched_pairs=[(0, 0), (1, 1)])]
        m = compute_relation_metrics(gt, pred, matches)
        assert m["strict"]["f1"] == 1.0  # relation matched
        assert m["strict"]["anchor_accuracy"] == 0.0  # but anchors wrong

    def test_anchor_canonicalization_resolves_aliases(self):
        from isaacsim_bench.schemas.anchors import (
            AnchorPose, AnchorRegistry, AssetAnchors,
        )
        # GT uses alias 'right_end' on X1; pred uses raw 'anchorpoint'.
        # Without canonicalization that's a string mismatch even though
        # they refer to the same authored anchor.
        registry = AnchorRegistry({
            "X1": AssetAnchors(
                anchors={
                    "origin": AnchorPose(position=[0, 0, 0]),
                    "anchorpoint": AnchorPose(position=[2, 0, 0]),
                },
                aliases={"right_end": "anchorpoint", "out": "anchorpoint"},
            ),
            "X2": AssetAnchors(
                anchors={
                    "origin": AnchorPose(position=[0, 0, 0]),
                },
                aliases={"in": "origin"},
            ),
        })
        gt_comps = [_comp("a", "X1"), _comp("b", "X2")]
        gt = [_scene(gt_comps, [_rel("a", "b", "right_end", "in")])]
        pred = [_prediction(
            components=[_pred_comp("a", "X1"), _pred_comp("b", "X2")],
            relations=[_pred_rel("a", "b", "anchorpoint", "origin")],
        )]
        matches = [MatchResult(matched_pairs=[(0, 0), (1, 1)])]

        # Without registry: anchor mismatch.
        m_strict = compute_relation_metrics(gt, pred, matches)
        assert m_strict["strict"]["anchor_accuracy"] == 0.0

        # With registry: aliases canonicalize to same raw name.
        m_canon = compute_relation_metrics(
            gt, pred, matches, anchor_registry=registry,
        )
        assert m_canon["strict"]["anchor_accuracy"] == 1.0
        assert m_canon["oracle_components"]["anchor_accuracy"] == 1.0

    def test_anchor_canonicalization_keeps_real_mismatches(self):
        """Different raw anchors should still score as wrong with a registry."""
        from isaacsim_bench.schemas.anchors import (
            AnchorPose, AnchorRegistry, AssetAnchors,
        )
        registry = AnchorRegistry({
            "X1": AssetAnchors(
                anchors={
                    "origin": AnchorPose(position=[0, 0, 0]),
                    "anchorpoint": AnchorPose(position=[2, 0, 0]),
                },
                aliases={"in": "origin", "out": "anchorpoint"},
            ),
            "X2": AssetAnchors(
                anchors={"origin": AnchorPose(position=[0, 0, 0])},
                aliases={},
            ),
        })
        gt_comps = [_comp("a", "X1"), _comp("b", "X2")]
        # GT 'out' (= anchorpoint) vs pred 'in' (= origin) — genuine mismatch.
        gt = [_scene(gt_comps, [_rel("a", "b", "out", "origin")])]
        pred = [_prediction(
            components=[_pred_comp("a", "X1"), _pred_comp("b", "X2")],
            relations=[_pred_rel("a", "b", "in", "origin")],
        )]
        matches = [MatchResult(matched_pairs=[(0, 0), (1, 1)])]
        m = compute_relation_metrics(
            gt, pred, matches, anchor_registry=registry,
        )
        assert m["strict"]["anchor_accuracy"] == 0.0


# ---- Placement metrics ----

class TestPlacementMetrics:
    def test_perfect_placement(self):
        gt_comps = [_comp("a", "X1", t=[0, 0, 0]), _comp("b", "X2", t=[1, 0, 0])]
        gt = [_scene(gt_comps)]
        pred = [_prediction(components=[
            _pred_comp("a", "X1", t=[0, 0, 0]),
            _pred_comp("b", "X2", t=[1, 0, 0]),
        ])]
        matches = [MatchResult(matched_pairs=[(0, 0), (1, 1)])]
        m = compute_placement_metrics(gt, pred, matches)
        assert m["root_relative"]["mean_translation_error_m"] == 0.0
        assert m["root_relative"]["translation_pass_rate"] == 1.0
        assert m["procrustes_rmse_m"] == 0.0
        assert m["pairwise_distance_mae_m"] == 0.0

    def test_shifted_placement(self):
        gt_comps = [_comp("a", "X1", t=[0, 0, 0]), _comp("b", "X2", t=[1, 0, 0])]
        gt = [_scene(gt_comps)]
        pred = [_prediction(components=[
            _pred_comp("a", "X1", t=[0, 0, 0]),
            _pred_comp("b", "X2", t=[1.5, 0, 0]),
        ])]
        matches = [MatchResult(matched_pairs=[(0, 0), (1, 1)])]
        m = compute_placement_metrics(gt, pred, matches)
        assert m["root_relative"]["mean_translation_error_m"] > 0
        assert m["pairwise_distance_mae_m"] > 0

    def test_global_shift_invariant(self):
        """A globally shifted prediction should have 0 pairwise error."""
        gt_comps = [_comp("a", "X1", t=[0, 0, 0]), _comp("b", "X2", t=[2, 0, 0])]
        gt = [_scene(gt_comps)]
        pred = [_prediction(components=[
            _pred_comp("a", "X1", t=[10, 5, 0]),
            _pred_comp("b", "X2", t=[12, 5, 0]),
        ])]
        matches = [MatchResult(matched_pairs=[(0, 0), (1, 1)])]
        m = compute_placement_metrics(gt, pred, matches)
        assert m["pairwise_distance_mae_m"] < 0.01
        assert m["procrustes_rmse_m"] < 0.01


# ---- Coverage metrics ----

class TestCoverageMetrics:
    def test_correct_abstention(self):
        gt_comps = [_comp("a", "X1", regime="unknown")]
        gt = [_scene(gt_comps)]
        pred = [_prediction(abstained=True)]
        m = compute_coverage_metrics(gt, pred)
        assert m["component_level"]["precision"] == 1.0
        assert m["scene_level"]["recall"] == 1.0

    def test_false_accept(self):
        gt_comps = [_comp("a", "X1", regime="unknown")]
        gt = [_scene(gt_comps)]
        pred = [_prediction(components=[_pred_comp("a", "X1")])]
        m = compute_coverage_metrics(gt, pred)
        assert m["false_accept_rate"] == 1.0


# ---- Scene success ----

class TestSceneSuccess:
    def test_perfect_scene(self):
        gt_comps = [_comp("a", "X1"), _comp("b", "X2")]
        gt = [_scene(gt_comps)]
        pred = [_prediction(components=[_pred_comp("a", "X1"), _pred_comp("b", "X2")])]
        matches = [MatchResult(matched_pairs=[(0, 0), (1, 1)])]
        m = compute_scene_success(gt, pred, matches)
        assert m["scene_success_rate"] == 1.0


# ---- EvaluatorRunner integration ----

class TestEvaluatorRunner:
    def test_full_pipeline(self, sample_taxonomy, sample_world_pool, sample_retrieval_pool):
        registry = TaxonomyRegistry(sample_taxonomy, sample_world_pool, sample_retrieval_pool)

        gt_comps = [
            _comp("a", "ConveyorBelt_A01", t=[0, 0, 0]),
            _comp("b", "ConveyorBelt_A02", t=[1, 0, 0]),
        ]
        gt = [_scene(gt_comps, [_rel("a", "b")])]
        pred = [_prediction(
            components=[
                _pred_comp("a", "ConveyorBelt_A01", t=[0, 0, 0]),
                _pred_comp("b", "ConveyorBelt_A02", t=[1, 0, 0]),
            ],
            relations=[_pred_rel("a", "b")],
        )]

        runner = EvaluatorRunner()
        report = runner.evaluate(gt, pred, registry)
        d = report.to_dict()

        assert d["component"]["f1"] == 1.0
        assert d["relation"]["strict"]["f1"] == 1.0
        assert d["placement"]["root_relative"]["mean_translation_error_m"] == 0.0
        assert d["scene_success"]["scene_success_rate"] == 1.0
