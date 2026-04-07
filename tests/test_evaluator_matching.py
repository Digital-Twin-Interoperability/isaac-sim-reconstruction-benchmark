import pytest

from isaacsim_bench.evaluator.matching import MatchResult, match_components
from isaacsim_bench.schemas.prediction import PredictedComponent
from isaacsim_bench.schemas.scene import ComponentEntry


def _gt(name: str, asset_id: str, family: str) -> ComponentEntry:
    return ComponentEntry(
        name=name,
        asset_id=asset_id,
        asset_name=asset_id,
        family=family,
        evaluation_role="primary",
        match_regime="exact_match",
        translate=[0, 0, 0],
        orientation_xyzw=[0, 0, 0, 1],
    )


def _pred(name: str, asset_id: str, family: str) -> PredictedComponent:
    return PredictedComponent(
        name=name,
        asset_id=asset_id,
        family=family,
        translate=[0, 0, 0],
        orientation_xyzw=[0, 0, 0, 1],
    )


class TestMatchComponents:
    def test_perfect_exact(self):
        gt = [_gt("a", "X1", "conveyor"), _gt("b", "X2", "shelf")]
        pred = [_pred("a", "X1", "conveyor"), _pred("b", "X2", "shelf")]
        result = match_components(gt, pred, mode="exact")
        assert len(result.matched_pairs) == 2
        assert len(result.unmatched_gt) == 0
        assert len(result.unmatched_pred) == 0

    def test_partial_match(self):
        gt = [_gt("a", "X1", "conveyor"), _gt("b", "X2", "shelf")]
        pred = [_pred("a", "X1", "conveyor"), _pred("c", "X3", "pallet")]
        result = match_components(gt, pred, mode="exact")
        assert len(result.matched_pairs) == 1
        assert len(result.unmatched_gt) == 1
        assert len(result.unmatched_pred) == 1

    def test_no_match(self):
        gt = [_gt("a", "X1", "conveyor")]
        pred = [_pred("b", "X2", "shelf")]
        result = match_components(gt, pred, mode="exact")
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_gt) == 1
        assert len(result.unmatched_pred) == 1

    def test_empty_inputs(self):
        result = match_components([], [], mode="exact")
        assert len(result.matched_pairs) == 0

    def test_family_mode(self):
        gt = [_gt("a", "X1", "conveyor_straight")]
        pred = [_pred("a", "X99", "conveyor_straight")]
        # Without registry, falls back to family string comparison
        result = match_components(gt, pred, mode="family")
        assert len(result.matched_pairs) == 1

    def test_extra_predictions(self):
        gt = [_gt("a", "X1", "conveyor")]
        pred = [_pred("a", "X1", "conveyor"), _pred("b", "X2", "shelf")]
        result = match_components(gt, pred, mode="exact")
        assert len(result.matched_pairs) == 1
        assert len(result.unmatched_pred) == 1
