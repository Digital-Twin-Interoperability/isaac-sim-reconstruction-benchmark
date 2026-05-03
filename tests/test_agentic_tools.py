"""Tests for the agentic tool handlers.

These tests don't instantiate a VLM or boot Isaac Sim — they exercise
validation logic, state mutation, and error paths for each tool in
isolation.
"""

from __future__ import annotations

import pytest

from pathlib import Path

from isaacsim_bench.agents.agentic_tools import (
    AgentState,
    ToolSpec,
    default_tool_specs,
    handle_add_component,
    handle_add_relation,
    handle_add_aligned_component,
    handle_align_components,
    handle_get_asset_info,
    handle_get_catalog_hits,
    handle_get_component_anchors,
    handle_list_assets_in_family,
    handle_list_components,
    handle_list_families,
    handle_list_references,
    handle_modify_component,
    handle_plot_top_down,
    handle_remove_component,
    handle_submit_prediction,
    handle_view_reference,
)
from isaacsim_bench.schemas.anchors import (
    AnchorPose,
    AnchorRegistry,
    AssetAnchors,
)
from isaacsim_bench.schemas.prediction import PredictionJSON


def _make_state(taxonomy, pool_ids) -> AgentState:
    return AgentState(
        prediction=PredictionJSON(sample_id="test", components=[], relations=[]),
        taxonomy=taxonomy,
        retrieval_pool_ids=set(pool_ids),
    )


@pytest.fixture()
def state(sample_taxonomy, sample_retrieval_pool):
    return _make_state(sample_taxonomy, sample_retrieval_pool.asset_ids)


class TestToolSpecs:
    def test_default_set_is_complete(self):
        specs = default_tool_specs()
        names = {s.name for s in specs}
        expected = {
            "list_families", "list_assets_in_family", "get_asset_info",
            "get_catalog_hits", "list_components", "add_component",
            "modify_component", "remove_component", "add_relation",
            "list_references", "view_reference",
            "plot_top_down",
            "get_component_anchors", "align_components",
            "add_aligned_component",
            "submit_prediction",
        }
        assert names == expected

    def test_specs_have_valid_anthropic_schemas(self):
        for spec in default_tool_specs():
            schema = spec.to_anthropic_schema()
            assert schema["name"] == spec.name
            assert isinstance(schema["input_schema"], dict)
            assert schema["input_schema"].get("type") == "object"


class TestInspectionTools:
    def test_list_families(self, state):
        r = handle_list_families({}, state)
        assert not r.is_error
        assert "conveyor" in r.text
        assert "shelf" in r.text
        assert "pallet" in r.text
        # Retrieval-pool marker is present
        assert "*" in r.text

    def test_list_assets_in_family_pool_only(self, state):
        r = handle_list_assets_in_family({"family": "conveyor"}, state)
        assert not r.is_error
        # A03 and A11 are withheld from the retrieval pool per conftest
        assert "ConveyorBelt_A01" in r.text
        assert "ConveyorBelt_A03" not in r.text

    def test_list_assets_in_family_all(self, state):
        r = handle_list_assets_in_family(
            {"family": "conveyor", "pool_only": False}, state,
        )
        assert "ConveyorBelt_A01" in r.text
        assert "ConveyorBelt_A03" in r.text  # non-pool included

    def test_list_assets_unknown_family(self, state):
        r = handle_list_assets_in_family({"family": "widgets"}, state)
        assert r.is_error
        assert "Unknown family" in r.text

    def test_get_asset_info(self, state):
        r = handle_get_asset_info({"asset_id": "ConveyorBelt_A01"}, state)
        assert not r.is_error
        assert "family: conveyor" in r.text
        assert "in_retrieval_pool: True" in r.text

    def test_get_asset_info_out_of_pool(self, state):
        r = handle_get_asset_info({"asset_id": "ConveyorBelt_A03"}, state)
        assert not r.is_error
        assert "in_retrieval_pool: False" in r.text

    def test_get_asset_info_unknown(self, state):
        r = handle_get_asset_info({"asset_id": "nonsense"}, state)
        assert r.is_error

    def test_get_catalog_hits_empty(self, state):
        r = handle_get_catalog_hits({"top_k": 5}, state)
        assert not r.is_error
        assert "No CLIP catalog hits" in r.text


class TestSceneEditing:
    def test_add_component_happy(self, state):
        r = handle_add_component({
            "name": "conv_top",
            "asset_id": "ConveyorBelt_A01",
            "position": [0.0, 0.0, 0.0],
        }, state)
        assert not r.is_error
        assert len(state.prediction.components) == 1
        c = state.prediction.components[0]
        assert c.name == "conv_top"
        assert c.asset_id == "ConveyorBelt_A01"
        assert c.family == "conveyor"

    def test_add_component_default_orientation(self, state):
        handle_add_component({
            "name": "a",
            "asset_id": "ConveyorBelt_A01",
            "position": [1.0, 2.0, 3.0],
        }, state)
        assert state.prediction.components[0].orientation_xyzw == [0.0, 0.0, 0.0, 1.0]

    def test_add_component_rejects_out_of_pool(self, state):
        r = handle_add_component({
            "name": "x",
            "asset_id": "ConveyorBelt_A03",  # withheld from pool
            "position": [0.0, 0.0, 0.0],
        }, state)
        assert r.is_error
        assert "retrieval pool" in r.text
        assert state.prediction.components == []

    def test_add_component_rejects_unknown_asset(self, state):
        r = handle_add_component({
            "name": "x",
            "asset_id": "NotAThing",
            "position": [0, 0, 0],
        }, state)
        assert r.is_error
        assert "Unknown asset_id" in r.text

    def test_add_component_rejects_duplicate_name(self, state):
        handle_add_component({
            "name": "dup",
            "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        r = handle_add_component({
            "name": "dup",
            "asset_id": "ConveyorBelt_A02",
            "position": [0, 0, 0],
        }, state)
        assert r.is_error
        assert "already exists" in r.text
        assert len(state.prediction.components) == 1

    def test_add_component_rejects_bad_position(self, state):
        r = handle_add_component({
            "name": "x",
            "asset_id": "ConveyorBelt_A01",
            "position": [1.0, 2.0],  # only 2 components
        }, state)
        assert r.is_error

    def test_add_component_rejects_bad_orientation(self, state):
        r = handle_add_component({
            "name": "x",
            "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
            "orientation_xyzw": [0.0, 0.0, 1.0],  # should be 4
        }, state)
        assert r.is_error

    def test_modify_component(self, state):
        handle_add_component({
            "name": "c",
            "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        r = handle_modify_component({
            "name": "c",
            "position": [1.0, 2.0, 3.0],
            "asset_id": "ConveyorBelt_A02",
        }, state)
        assert not r.is_error
        c = state.prediction.components[0]
        assert c.translate == [1.0, 2.0, 3.0]
        assert c.asset_id == "ConveyorBelt_A02"

    def test_modify_nonexistent_component(self, state):
        r = handle_modify_component({"name": "ghost", "position": [0, 0, 0]}, state)
        assert r.is_error

    def test_remove_component_drops_relations(self, state):
        handle_add_component({
            "name": "a",
            "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        handle_add_component({
            "name": "b",
            "asset_id": "ConveyorBelt_A02",
            "position": [1, 0, 0],
        }, state)
        handle_add_relation({
            "type": "adjacent_to",
            "from_node": "a", "to_node": "b",
            "from_anchor": "right", "to_anchor": "left",
        }, state)
        assert len(state.prediction.relations) == 1
        handle_remove_component({"name": "a"}, state)
        assert len(state.prediction.relations) == 0
        assert len(state.prediction.components) == 1

    def test_add_relation_rejects_missing_node(self, state):
        r = handle_add_relation({
            "type": "x", "from_node": "nope", "to_node": "also_nope",
        }, state)
        assert r.is_error


class TestListComponents:
    def test_empty_scene(self, state):
        r = handle_list_components({}, state)
        assert not r.is_error
        assert "empty" in r.text

    def test_populated_scene(self, state):
        handle_add_component({
            "name": "conv",
            "asset_id": "ConveyorBelt_A01",
            "position": [1.5, 0.0, 0.25],
        }, state)
        r = handle_list_components({}, state)
        assert not r.is_error
        assert "conv" in r.text
        assert "ConveyorBelt_A01" in r.text


class TestReferences:
    def test_list_references_empty(self, state):
        r = handle_list_references({}, state)
        assert not r.is_error
        assert "No reference images" in r.text

    def test_list_references_populated(self, state, tmp_path):
        a = tmp_path / "front.png"
        b = tmp_path / "scene_default.png"
        a.write_bytes(b"\x89PNG\r\n")
        b.write_bytes(b"\x89PNG\r\n")
        state.input_image_paths = [a, b]
        r = handle_list_references({}, state)
        assert "front.png" in r.text and "scene_default.png" in r.text

    def test_view_reference_attaches_image(self, state, tmp_path):
        a = tmp_path / "front.png"
        a.write_bytes(b"\x89PNG\r\n")
        state.input_image_paths = [a]
        r = handle_view_reference({"name": "front.png"}, state)
        assert not r.is_error
        assert r.image_paths == [a]

    def test_view_reference_unknown_name(self, state, tmp_path):
        a = tmp_path / "front.png"
        a.write_bytes(b"\x89PNG\r\n")
        state.input_image_paths = [a]
        r = handle_view_reference({"name": "missing.png"}, state)
        assert r.is_error


class TestSystemPrompt:
    def test_render_enabled_mentions_render_tool(self):
        from isaacsim_bench.agents.agentic import build_system_prompt

        with_render = build_system_prompt(render_enabled=True)
        without_render = build_system_prompt(render_enabled=False)
        assert "render" in with_render.lower()
        assert "no render tool" in without_render.lower()
        # Both prompts must require ending each turn with a tool call
        # to avoid the silent "text-only" loop exit.
        for prompt in (with_render, without_render):
            assert "submit_prediction" in prompt
            assert "every turn" in prompt.lower() or "tool call" in prompt.lower()


class TestSubmit:
    def _place_n(self, state, n: int) -> None:
        for i in range(n):
            handle_add_component({
                "name": f"c{i}",
                "asset_id": "ConveyorBelt_A01",
                "position": [float(i), 0, 0],
            }, state)
        # Pretend a render happened so the render-after-edit gate doesn't
        # fire.  In the unit-test fixture state.renderer is None anyway, so
        # the gate is bypassed; this just defensively documents intent.
        state.scene_modified_since_render = False

    def test_submit_requires_expected_components(self, state):
        self._place_n(state, 1)
        r = handle_submit_prediction({"notes": "done"}, state)
        assert r.is_error
        assert "expected_components" in r.text
        assert state.submitted is False

    def test_submit_rejects_count_mismatch(self, state):
        self._place_n(state, 1)
        r = handle_submit_prediction({"expected_components": 3}, state)
        assert r.is_error
        assert "Inventory mismatch" in r.text
        assert state.submitted is False

    def test_submit_happy_path(self, state):
        self._place_n(state, 2)
        r = handle_submit_prediction(
            {"expected_components": 2, "notes": "done"}, state,
        )
        assert not r.is_error
        assert state.submitted is True
        assert state.submit_notes == "done"

    def test_submit_acknowledge_unmatched_requires_notes(self, state):
        self._place_n(state, 1)
        r = handle_submit_prediction({
            "expected_components": 3,
            "acknowledge_unmatched": True,
        }, state)
        assert r.is_error
        assert "notes" in r.text.lower()
        assert state.submitted is False

    def test_submit_acknowledge_unmatched_succeeds(self, state):
        self._place_n(state, 1)
        r = handle_submit_prediction({
            "expected_components": 3,
            "acknowledge_unmatched": True,
            "notes": "two straights not in pool",
        }, state)
        assert not r.is_error
        assert state.submitted is True

    def test_submit_rejects_zero_expected(self, state):
        self._place_n(state, 1)
        r = handle_submit_prediction({"expected_components": 0}, state)
        assert r.is_error
        assert state.submitted is False

    def test_submit_rejects_when_scene_modified_since_render(self, state):
        # Simulate render tool wired in via a sentinel renderer object.
        state.renderer = object()
        self._place_n(state, 1)
        # Mark scene as modified after the synthetic render (place_n cleared
        # it).  This is what would happen if the agent edited then tried
        # to submit without re-rendering.
        state.scene_modified_since_render = True
        r = handle_submit_prediction({"expected_components": 1}, state)
        assert r.is_error
        assert "render" in r.text.lower()
        assert state.submitted is False

    def test_submit_render_gate_skipped_without_renderer(self, state):
        # state.renderer is None in the fixture — render gate must not
        # fire.  Mismatched flag should not block submit.
        state.renderer = None
        self._place_n(state, 1)
        state.scene_modified_since_render = True
        r = handle_submit_prediction({"expected_components": 1}, state)
        assert not r.is_error
        assert state.submitted is True


class TestPlotTopDown:
    def test_empty_scene_errors(self, state):
        r = handle_plot_top_down({}, state)
        assert r.is_error
        assert "No components" in r.text

    def test_returns_image_for_populated_scene(self, state):
        handle_add_component({
            "name": "c",
            "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        r = handle_plot_top_down({}, state)
        assert not r.is_error
        assert len(r.image_paths) == 1
        out = r.image_paths[0]
        assert out.exists()
        # Sanity: PNG header.
        assert out.read_bytes()[:4] == b"\x89PNG"

    def test_uses_extents_when_provided(self, state):
        # New shape: per-variant dict with extent_xyz / bbox_min / bbox_max.
        # Plotter prefers bbox_min/max, falls back to centered extent_xyz.
        state.asset_extents = {
            "ConveyorBelt_A01": {
                "extent_xyz": [4.0, 1.5, 1.0],
                "bbox_min": [0.0, -0.75, 0.0],
                "bbox_max": [4.0, 0.75, 1.0],
            },
        }
        handle_add_component({
            "name": "long_one",
            "asset_id": "ConveyorBelt_A01",
            "position": [2.0, 1.0, 0.0],
        }, state)
        r = handle_plot_top_down({}, state)
        assert not r.is_error
        # Just check the image was produced; visual correctness is human-checked.
        assert r.image_paths[0].exists()


class TestEditSerialization:
    """Edit-after-render gate: only fires when state.renderer is set."""

    def _enable_render_gate(self, state):
        # Sentinel renderer object — handlers only check `is not None`.
        state.renderer = object()

    def test_no_gate_when_renderer_disabled(self, state):
        state.renderer = None  # default in fixture, but be explicit
        handle_add_component({
            "name": "a", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        # Without renderer, agent can keep editing — fixture mode.
        r = handle_add_component({
            "name": "b", "asset_id": "ConveyorBelt_A02", "position": [1, 0, 0],
        }, state)
        assert not r.is_error
        assert len(state.prediction.components) == 2

    def test_blocks_back_to_back_add(self, state):
        self._enable_render_gate(state)
        handle_add_component({
            "name": "a", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        r = handle_add_component({
            "name": "b", "asset_id": "ConveyorBelt_A02", "position": [1, 0, 0],
        }, state)
        assert r.is_error
        assert "render" in r.text.lower()
        # Second add should not have landed.
        assert len(state.prediction.components) == 1

    def test_blocks_back_to_back_modify(self, state):
        self._enable_render_gate(state)
        handle_add_component({
            "name": "a", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        # Simulate a render between add and the first modify.
        state.scene_modified_since_render = False
        handle_modify_component({"name": "a", "position": [1, 0, 0]}, state)
        r = handle_modify_component({"name": "a", "position": [2, 0, 0]}, state)
        assert r.is_error
        # Position should reflect the FIRST modify only.
        assert state.prediction.components[0].translate == [1.0, 0.0, 0.0]

    def test_blocks_remove_after_modify(self, state):
        self._enable_render_gate(state)
        handle_add_component({
            "name": "a", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        state.scene_modified_since_render = False
        handle_modify_component({"name": "a", "position": [1, 0, 0]}, state)
        r = handle_remove_component({"name": "a"}, state)
        assert r.is_error
        # Component still present.
        assert len(state.prediction.components) == 1

    def test_render_clears_gate(self, state):
        self._enable_render_gate(state)
        handle_add_component({
            "name": "a", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        # Simulate a successful render — handler in agentic_render clears
        # the flag.  We mimic that here.
        state.scene_modified_since_render = False
        r = handle_add_component({
            "name": "b", "asset_id": "ConveyorBelt_A02", "position": [1, 0, 0],
        }, state)
        assert not r.is_error
        assert len(state.prediction.components) == 2


class TestSceneModifiedFlag:
    def test_add_sets_flag(self, state):
        assert state.scene_modified_since_render is False
        handle_add_component({
            "name": "c",
            "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        assert state.scene_modified_since_render is True

    def test_modify_sets_flag(self, state):
        handle_add_component({
            "name": "c",
            "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        state.scene_modified_since_render = False
        handle_modify_component({"name": "c", "position": [1, 2, 3]}, state)
        assert state.scene_modified_since_render is True

    def test_remove_sets_flag(self, state):
        handle_add_component({
            "name": "c",
            "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        state.scene_modified_since_render = False
        handle_remove_component({"name": "c"}, state)
        assert state.scene_modified_since_render is True


# Real extracted+overlaid anchor data for the conveyors used by the U-shape
# test; lets us exercise align_components against the same numbers the agent
# would see in production.
def _make_anchor_registry() -> AnchorRegistry:
    return AnchorRegistry({
        "ConveyorBelt_A01": AssetAnchors(
            anchors={
                "origin": AnchorPose(
                    position=[0.0, 0.0, 0.0], orient_wxyz=[1.0, 0.0, 0.0, 0.0],
                ),
                "anchorpoint": AnchorPose(
                    position=[2.000006, 0.0, 0.0],
                    orient_wxyz=[1.0, 0.0, 0.0, 0.0],
                ),
            },
            aliases={"in": "origin", "out": "anchorpoint"},
        ),
        "ConveyorBelt_A02": AssetAnchors(
            anchors={
                "origin": AnchorPose(
                    position=[0.0, 0.0, 0.0], orient_wxyz=[1.0, 0.0, 0.0, 0.0],
                ),
                "anchorpoint": AnchorPose(
                    position=[0.0, -3.91878, 0.0],
                    orient_wxyz=[0.0, 0.0, 0.0, 1.0],
                    valid=False,  # mark degenerate to test rejection path
                ),
            },
            aliases={"in": "origin"},
        ),
        "ConveyorBelt_A10": AssetAnchors(
            anchors={
                "origin": AnchorPose(
                    position=[0.0, 0.0, 0.0], orient_wxyz=[1.0, 0.0, 0.0, 0.0],
                ),
                "anchorpoint": AnchorPose(
                    position=[0.0, -3.91878, 0.0],
                    orient_wxyz=[0.0, 0.0, 0.0, 1.0],
                ),
            },
            aliases={
                "in": "origin", "out": "anchorpoint",
                "curve_exit": "origin", "curve_entry": "anchorpoint",
            },
        ),
    })


@pytest.fixture()
def state_with_anchors(state) -> AgentState:
    state.asset_anchors = _make_anchor_registry()
    return state


class TestGetComponentAnchors:
    def test_requires_component(self, state_with_anchors):
        r = handle_get_component_anchors({}, state_with_anchors)
        assert r.is_error and "component" in r.text

    def test_unknown_component(self, state_with_anchors):
        r = handle_get_component_anchors(
            {"component": "ghost"}, state_with_anchors,
        )
        assert r.is_error and "ghost" in r.text

    def test_no_registry_falls_back_loudly(self, state):
        handle_add_component({
            "name": "c", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        # No registry on plain `state`.
        r = handle_get_component_anchors({"component": "c"}, state)
        assert r.is_error
        assert "anchor metadata" in r.text

    def test_world_pose_at_origin_matches_local(self, state_with_anchors):
        handle_add_component({
            "name": "s1", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1],
        }, state_with_anchors)
        r = handle_get_component_anchors(
            {"component": "s1"}, state_with_anchors,
        )
        assert not r.is_error
        # 'out' alias resolves to anchorpoint at (2, 0, 0); world == local
        # when the component sits at origin with identity rotation.
        assert "2.0" in r.text and "0.0" in r.text
        assert "out" in r.text  # alias listed as metadata

    def test_world_pose_translates_with_component(self, state_with_anchors):
        handle_add_component({
            "name": "s1", "asset_id": "ConveyorBelt_A01",
            "position": [10, 0, 0], "orientation_xyzw": [0, 0, 0, 1],
        }, state_with_anchors)
        r = handle_get_component_anchors(
            {"component": "s1"}, state_with_anchors,
        )
        # Anchorpoint should have moved to (12, 0, 0) in world.
        assert "12.0" in r.text

    def test_invalid_anchor_is_shown_with_flag(self, state_with_anchors):
        # A02's anchorpoint is marked invalid in the fixture.
        handle_add_component({
            "name": "bad", "asset_id": "ConveyorBelt_A02", "position": [0, 0, 0],
        }, state_with_anchors)
        r = handle_get_component_anchors(
            {"component": "bad"}, state_with_anchors,
        )
        assert "INVALID" in r.text


class TestAlignComponents:
    def _add_two(self, state, fixed_id="ConveyorBelt_A01",
                 moving_id="ConveyorBelt_A01"):
        handle_add_component({
            "name": "fixed", "asset_id": fixed_id,
            "position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1],
        }, state)
        handle_add_component({
            "name": "moving", "asset_id": moving_id,
            "position": [99, 99, 99],  # deliberately wrong; align overwrites
            "orientation_xyzw": [0, 0, 0, 1],
        }, state)

    def test_a01_to_a01_chain(self, state_with_anchors):
        self._add_two(state_with_anchors)
        r = handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_component": "moving", "moving_anchor": "in",
        }, state_with_anchors)
        assert not r.is_error, r.text
        moving = state_with_anchors.prediction.components[1]
        assert moving.translate[0] == pytest.approx(2.000006, abs=1e-4)
        assert moving.translate[1] == pytest.approx(0.0, abs=1e-4)
        # No rotation: A01.out has identity orient.
        assert moving.orientation_xyzw[3] == pytest.approx(1.0, abs=1e-4)

    def test_a10_to_a01_preserves_authored_180(self, state_with_anchors):
        self._add_two(
            state_with_anchors,
            fixed_id="ConveyorBelt_A10", moving_id="ConveyorBelt_A01",
        )
        r = handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "curve_entry",
            "moving_component": "moving", "moving_anchor": "in",
        }, state_with_anchors)
        assert not r.is_error, r.text
        moving = state_with_anchors.prediction.components[1]
        # The U far-end position; quat is 180° about Z (z-component magnitude 1).
        assert moving.translate[1] == pytest.approx(-3.91878, abs=1e-3)
        assert abs(moving.orientation_xyzw[2]) == pytest.approx(1.0, abs=1e-4)

    def test_records_relation(self, state_with_anchors):
        self._add_two(state_with_anchors)
        handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_component": "moving", "moving_anchor": "in",
        }, state_with_anchors)
        rels = state_with_anchors.prediction.relations
        assert len(rels) == 1
        assert rels[0].type == "attach"
        assert rels[0].from_node == "fixed"
        assert rels[0].to_node == "moving"
        assert rels[0].from_anchor == "out"
        assert rels[0].to_anchor == "in"

    def test_dedupes_relation_on_re_align(self, state_with_anchors):
        self._add_two(state_with_anchors)
        # First align by 'out' -> 'in'.
        handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_component": "moving", "moving_anchor": "in",
        }, state_with_anchors)
        # Render to clear edit gate, then re-align with different anchors.
        state_with_anchors.scene_modified_since_render = False
        handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "anchorpoint",
            "moving_component": "moving", "moving_anchor": "origin",
        }, state_with_anchors)
        rels = state_with_anchors.prediction.relations
        assert len(rels) == 1  # not duplicated
        assert rels[0].from_anchor == "anchorpoint"
        assert rels[0].to_anchor == "origin"

    def test_rejects_invalid_anchor_loudly(self, state_with_anchors):
        # A02's anchorpoint is marked invalid.
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state_with_anchors)
        handle_add_component({
            "name": "moving", "asset_id": "ConveyorBelt_A02",
            "position": [0, 0, 0],
        }, state_with_anchors)
        r = handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_component": "moving", "moving_anchor": "anchorpoint",
        }, state_with_anchors)
        assert r.is_error
        assert "INVALID" in r.text

    def test_rejects_unknown_anchor(self, state_with_anchors):
        self._add_two(state_with_anchors)
        r = handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "ghost_anchor",
            "moving_component": "moving", "moving_anchor": "in",
        }, state_with_anchors)
        assert r.is_error and "ghost_anchor" in r.text

    def test_rejects_self_alignment(self, state_with_anchors):
        handle_add_component({
            "name": "lonely", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_with_anchors)
        r = handle_align_components({
            "fixed_component": "lonely", "fixed_anchor": "out",
            "moving_component": "lonely", "moving_anchor": "in",
        }, state_with_anchors)
        assert r.is_error and "differ" in r.text

    def test_rejects_same_canonical_anchor_on_both_sides(self, state_with_anchors):
        # origin↔origin would put both pieces' start-faces at the same
        # world point, overlapping them.  The tool should catch this.
        self._add_two(state_with_anchors)
        r = handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "origin",
            "moving_component": "moving", "moving_anchor": "origin",
        }, state_with_anchors)
        assert r.is_error
        assert "same canonical anchor" in r.text or "tip-to-tail" in r.text

    def test_rejects_alias_to_raw_same_canonical(self, state_with_anchors):
        # `in` aliases `origin`; mating `in`↔`origin` is the same bug
        # disguised by aliasing — the canonical-name check must catch it.
        self._add_two(state_with_anchors)
        r = handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "in",
            "moving_component": "moving", "moving_anchor": "origin",
        }, state_with_anchors)
        assert r.is_error
        assert "canonical" in r.text

    def test_no_registry_fails_loudly(self, state):
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        handle_add_component({
            "name": "moving", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        r = handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_component": "moving", "moving_anchor": "in",
        }, state)
        assert r.is_error and "anchor registry" in r.text

    def test_sets_scene_modified_flag(self, state_with_anchors):
        self._add_two(state_with_anchors)
        state_with_anchors.scene_modified_since_render = False
        handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_component": "moving", "moving_anchor": "in",
        }, state_with_anchors)
        assert state_with_anchors.scene_modified_since_render is True

    def test_opposed_frame_flips_orientation(self, state_with_anchors):
        self._add_two(state_with_anchors)
        r = handle_align_components({
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_component": "moving", "moving_anchor": "in",
            "facing": "opposed_frame",
        }, state_with_anchors)
        assert not r.is_error, r.text
        moving = state_with_anchors.prediction.components[1]
        # Position at fixed.out, but rotated 180° about Z.
        assert moving.translate[0] == pytest.approx(2.000006, abs=1e-4)
        assert abs(moving.orientation_xyzw[2]) == pytest.approx(1.0, abs=1e-4)


class TestAddAlignedComponent:
    def test_adds_and_aligns_in_one_call(self, state_with_anchors):
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1],
        }, state_with_anchors)
        # Need to render to clear the gate before the next edit.
        state_with_anchors.scene_modified_since_render = False

        r = handle_add_aligned_component({
            "name": "next", "asset_id": "ConveyorBelt_A01",
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_anchor": "in",
        }, state_with_anchors)
        assert not r.is_error, r.text
        assert len(state_with_anchors.prediction.components) == 2
        new = state_with_anchors.prediction.components[-1]
        assert new.name == "next"
        assert new.translate[0] == pytest.approx(2.000006, abs=1e-4)
        assert new.confidence == pytest.approx(0.8)
        assert state_with_anchors.scene_modified_since_render is True

    def test_appends_relation_in_correct_direction(self, state_with_anchors):
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_with_anchors)
        state_with_anchors.scene_modified_since_render = False
        handle_add_aligned_component({
            "name": "next", "asset_id": "ConveyorBelt_A01",
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_anchor": "in",
        }, state_with_anchors)
        rels = state_with_anchors.prediction.relations
        assert len(rels) == 1
        assert rels[0].from_node == "fixed"
        assert rels[0].to_node == "next"
        assert rels[0].from_anchor == "out"
        assert rels[0].to_anchor == "in"

    def test_blocks_on_existing_name(self, state_with_anchors):
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_with_anchors)
        state_with_anchors.scene_modified_since_render = False
        r = handle_add_aligned_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01",
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_anchor": "in",
        }, state_with_anchors)
        assert r.is_error and "already exists" in r.text

    def test_blocks_on_invalid_anchor(self, state_with_anchors):
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_with_anchors)
        state_with_anchors.scene_modified_since_render = False
        # A02's anchorpoint is invalid.
        r = handle_add_aligned_component({
            "name": "next", "asset_id": "ConveyorBelt_A02",
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_anchor": "anchorpoint",
        }, state_with_anchors)
        assert r.is_error and "INVALID" in r.text
        # Crucially: nothing should have been appended on the failure path.
        assert len(state_with_anchors.prediction.components) == 1

    def test_obeys_edit_gate_when_renderer_present(self, state_with_anchors):
        # Simulate a wired renderer to enable the gate.
        state_with_anchors.renderer = object()
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_with_anchors)
        # Don't clear the flag; gate must block.
        r = handle_add_aligned_component({
            "name": "next", "asset_id": "ConveyorBelt_A01",
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_anchor": "in",
        }, state_with_anchors)
        assert r.is_error and "render" in r.text


class TestAnchorsInGetAssetInfo:
    def test_prints_anchors_when_registry_present(self, state_with_anchors):
        r = handle_get_asset_info(
            {"asset_id": "ConveyorBelt_A10"}, state_with_anchors,
        )
        assert "anchors" in r.text
        assert "origin" in r.text and "anchorpoint" in r.text
        assert "curve_entry" in r.text  # alias listed

    def test_marks_invalid_anchor(self, state_with_anchors):
        r = handle_get_asset_info(
            {"asset_id": "ConveyorBelt_A02"}, state_with_anchors,
        )
        assert "INVALID" in r.text

    def test_no_anchor_section_without_registry(self, state):
        r = handle_get_asset_info({"asset_id": "ConveyorBelt_A01"}, state)
        assert "anchors" not in r.text.lower() or "anchor" not in r.text.lower()

