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
    InventoryItem,
    ToolSpec,
    default_tool_specs,
    handle_add_aligned_component,
    handle_add_component,
    handle_add_inventory_item,
    handle_add_relation,
    handle_align_components,
    handle_get_asset_info,
    handle_get_catalog_hits,
    handle_get_component_anchors,
    handle_list_assets_in_family,
    handle_list_components,
    handle_list_families,
    handle_list_references,
    handle_mark_matched,
    handle_modify_component,
    handle_plot_top_down,
    handle_remove_component,
    handle_set_inventory,
    handle_submit_prediction,
    handle_unmark_matched,
    handle_update_inventory_item,
    handle_view_reference,
)
from isaacsim_bench.schemas.anchors import (
    AnchorPose,
    AnchorRegistry,
    AssetAnchors,
)
from isaacsim_bench.schemas.prediction import PredictionJSON


def _make_state(taxonomy, pool_ids, *, lock_inventory: bool = True) -> AgentState:
    state = AgentState(
        prediction=PredictionJSON(sample_id="test", components=[], relations=[]),
        taxonomy=taxonomy,
        retrieval_pool_ids=set(pool_ids),
    )
    if lock_inventory:
        # Pre-lock with a generic inventory so existing edit tests don't have
        # to call set_inventory before every add/modify/align.  Tests that
        # exercise the inventory gate itself pass lock_inventory=False, and
        # TestSubmit rebuilds the inventory to match its needs.
        state.inventory = [
            InventoryItem(item_id="placeholder_1", description="generic"),
            InventoryItem(item_id="placeholder_2", description="generic"),
        ]
        state.inventory_locked = True
    return state


@pytest.fixture()
def state(sample_taxonomy, sample_retrieval_pool):
    return _make_state(sample_taxonomy, sample_retrieval_pool.asset_ids)


@pytest.fixture()
def state_unlocked(sample_taxonomy, sample_retrieval_pool):
    """State with no inventory committed — for testing the hard gate."""
    return _make_state(
        sample_taxonomy, sample_retrieval_pool.asset_ids, lock_inventory=False,
    )


class TestToolSpecs:
    def test_default_set_is_complete(self):
        specs = default_tool_specs()
        names = {s.name for s in specs}
        expected = {
            "set_inventory", "mark_matched", "unmark_matched",
            "update_inventory_item", "add_inventory_item",
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


class TestInventoryHardGate:
    """Edit handlers refuse to run before set_inventory is called."""

    def test_add_component_blocked_pre_lock(self, state_unlocked):
        r = handle_add_component({
            "name": "c", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_unlocked)
        assert r.is_error
        assert "inventory" in r.text.lower() and "set_inventory" in r.text
        assert state_unlocked.prediction.components == []

    def test_modify_component_blocked_pre_lock(self, state_unlocked):
        r = handle_modify_component(
            {"name": "ghost", "position": [1, 0, 0]}, state_unlocked,
        )
        assert r.is_error
        assert "inventory" in r.text.lower()

    def test_align_components_blocked_pre_lock(self, state_unlocked):
        r = handle_align_components({
            "fixed_component": "x", "fixed_anchor": "out",
            "moving_component": "y", "moving_anchor": "in",
        }, state_unlocked)
        assert r.is_error
        assert "inventory" in r.text.lower()

    def test_add_aligned_component_blocked_pre_lock(self, state_unlocked):
        r = handle_add_aligned_component({
            "name": "n", "asset_id": "ConveyorBelt_A01",
            "fixed_component": "f", "fixed_anchor": "out",
            "moving_anchor": "in",
        }, state_unlocked)
        assert r.is_error
        assert "inventory" in r.text.lower()

    def test_set_inventory_unlocks_edits(self, state_unlocked):
        r = handle_set_inventory({
            "items": [{"item_id": "c1", "description": "a conveyor"}],
        }, state_unlocked)
        assert not r.is_error, r.text
        assert state_unlocked.inventory_locked is True
        # Now edits work.
        r = handle_add_component({
            "name": "c", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_unlocked)
        assert not r.is_error, r.text
        assert len(state_unlocked.prediction.components) == 1


class TestInventoryTools:
    def test_set_inventory_happy(self, state_unlocked):
        r = handle_set_inventory({
            "items": [
                {"item_id": "u_curve", "description": "curve in middle",
                 "rough_xy": [0.0, 0.0]},
                {"item_id": "left_straight",
                 "description": "feeds curve from west"},
            ],
        }, state_unlocked)
        assert not r.is_error, r.text
        assert state_unlocked.inventory_locked
        ids = [it.item_id for it in state_unlocked.inventory]
        assert ids == ["u_curve", "left_straight"]
        assert state_unlocked.inventory[0].rough_xy == (0.0, 0.0)

    def test_set_inventory_rejects_empty(self, state_unlocked):
        r = handle_set_inventory({"items": []}, state_unlocked)
        assert r.is_error and not state_unlocked.inventory_locked

    def test_set_inventory_rejects_duplicate_ids(self, state_unlocked):
        r = handle_set_inventory({
            "items": [
                {"item_id": "x", "description": "a"},
                {"item_id": "x", "description": "b"},
            ],
        }, state_unlocked)
        assert r.is_error and "Duplicate" in r.text
        assert not state_unlocked.inventory_locked

    def test_set_inventory_rejects_missing_fields(self, state_unlocked):
        r = handle_set_inventory({
            "items": [{"item_id": "x"}],  # no description
        }, state_unlocked)
        assert r.is_error and not state_unlocked.inventory_locked

    def test_set_inventory_rejects_bad_rough_xy(self, state_unlocked):
        r = handle_set_inventory({
            "items": [
                {"item_id": "x", "description": "a", "rough_xy": [1.0]},
            ],
        }, state_unlocked)
        assert r.is_error and not state_unlocked.inventory_locked

    def test_set_inventory_one_shot(self, state):
        # Default fixture is already locked.
        r = handle_set_inventory({
            "items": [{"item_id": "y", "description": "another"}],
        }, state)
        assert r.is_error
        assert "already locked" in r.text.lower()

    def test_mark_matched_happy(self, state):
        handle_add_component({
            "name": "comp_a", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        r = handle_mark_matched(
            {"item_id": "placeholder_1", "component": "comp_a"}, state,
        )
        assert not r.is_error, r.text
        item = next(
            i for i in state.inventory if i.item_id == "placeholder_1"
        )
        assert item.matched_components == ["comp_a"]
        assert item.matched is True

    def test_mark_matched_one_to_many(self, state):
        # 1:N — same inventory item, multiple components.
        for i in range(3):
            handle_add_component({
                "name": f"box_{i}", "asset_id": "ConveyorBelt_A01",
                "position": [float(i), 0, 0],
            }, state)
            handle_mark_matched(
                {"item_id": "placeholder_1", "component": f"box_{i}"}, state,
            )
        item = next(
            i for i in state.inventory if i.item_id == "placeholder_1"
        )
        assert item.matched_components == ["box_0", "box_1", "box_2"]

    def test_mark_matched_idempotent(self, state):
        handle_add_component({
            "name": "c", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        handle_mark_matched(
            {"item_id": "placeholder_1", "component": "c"}, state,
        )
        r = handle_mark_matched(
            {"item_id": "placeholder_1", "component": "c"}, state,
        )
        assert not r.is_error
        assert "already matched" in r.text.lower()
        item = next(i for i in state.inventory if i.item_id == "placeholder_1")
        assert item.matched_components == ["c"]  # not duplicated

    def test_mark_matched_unknown_item(self, state):
        handle_add_component({
            "name": "c", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        r = handle_mark_matched(
            {"item_id": "ghost", "component": "c"}, state,
        )
        assert r.is_error and "ghost" in r.text

    def test_mark_matched_unknown_component(self, state):
        r = handle_mark_matched(
            {"item_id": "placeholder_1", "component": "ghost"}, state,
        )
        assert r.is_error and "ghost" in r.text

    def test_unmark_matched_specific(self, state):
        handle_add_component({
            "name": "a", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        handle_mark_matched(
            {"item_id": "placeholder_1", "component": "a"}, state,
        )
        r = handle_unmark_matched(
            {"item_id": "placeholder_1", "component": "a"}, state,
        )
        assert not r.is_error
        item = next(i for i in state.inventory if i.item_id == "placeholder_1")
        assert item.matched_components == []

    def test_unmark_matched_clear_all(self, state):
        for i in range(2):
            handle_add_component({
                "name": f"c{i}", "asset_id": "ConveyorBelt_A01",
                "position": [float(i), 0, 0],
            }, state)
            handle_mark_matched(
                {"item_id": "placeholder_1", "component": f"c{i}"}, state,
            )
        r = handle_unmark_matched({"item_id": "placeholder_1"}, state)
        assert not r.is_error
        item = next(i for i in state.inventory if i.item_id == "placeholder_1")
        assert item.matched_components == []

    def test_remove_component_auto_unmatches(self, state):
        handle_add_component({
            "name": "doomed", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        handle_mark_matched(
            {"item_id": "placeholder_1", "component": "doomed"}, state,
        )
        state.scene_modified_since_render = False
        r = handle_remove_component({"name": "doomed"}, state)
        assert not r.is_error
        assert "Auto-unmatched" in r.text
        assert "placeholder_1" in r.text
        item = next(i for i in state.inventory if i.item_id == "placeholder_1")
        assert item.matched_components == []

    def test_update_inventory_item(self, state):
        r = handle_update_inventory_item({
            "item_id": "placeholder_1",
            "description": "more specific",
            "rough_xy": [3.0, 2.0],
        }, state)
        assert not r.is_error
        item = next(i for i in state.inventory if i.item_id == "placeholder_1")
        assert item.description == "more specific"
        assert item.rough_xy == (3.0, 2.0)

    def test_update_inventory_item_unknown(self, state):
        r = handle_update_inventory_item(
            {"item_id": "ghost", "description": "x"}, state,
        )
        assert r.is_error

    def test_update_inventory_item_no_fields(self, state):
        r = handle_update_inventory_item({"item_id": "placeholder_1"}, state)
        assert r.is_error

    def test_add_inventory_item_post_lock(self, state):
        r = handle_add_inventory_item({
            "item_id": "missed_box",
            "description": "found this in turn-7 render",
            "reason": "render revealed an extra pallet I missed initially",
        }, state)
        assert not r.is_error, r.text
        ids = [i.item_id for i in state.inventory]
        assert "missed_box" in ids
        item = next(i for i in state.inventory if i.item_id == "missed_box")
        assert "post-lock" in item.notes
        assert "missed initially" in item.notes

    def test_add_inventory_item_requires_reason(self, state):
        r = handle_add_inventory_item({
            "item_id": "x", "description": "y",
        }, state)
        assert r.is_error
        assert "reason" in r.text.lower()

    def test_add_inventory_item_rejects_duplicate(self, state):
        r = handle_add_inventory_item({
            "item_id": "placeholder_1",
            "description": "duplicate of existing",
            "reason": "test",
        }, state)
        assert r.is_error and "already exists" in r.text

    def test_add_inventory_item_pre_lock(self, state_unlocked):
        r = handle_add_inventory_item({
            "item_id": "x", "description": "y", "reason": "z",
        }, state_unlocked)
        assert r.is_error
        assert "set_inventory" in r.text.lower()


class TestAddWithItemId:
    """add_component / add_aligned_component accept an optional item_id that
    auto-marks the new component, collapsing add + mark into one tool call."""

    def test_add_component_auto_matches(self, state):
        r = handle_add_component({
            "name": "c", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
            "item_id": "placeholder_1",
        }, state)
        assert not r.is_error, r.text
        assert "Auto-matched" in r.text
        item = next(i for i in state.inventory if i.item_id == "placeholder_1")
        assert item.matched_components == ["c"]

    def test_add_component_rejects_unknown_item_id(self, state):
        r = handle_add_component({
            "name": "c", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
            "item_id": "ghost_item",
        }, state)
        assert r.is_error
        assert "ghost_item" in r.text
        # And the component should NOT have been added — pre-validation.
        assert state.prediction.components == []

    def test_add_component_without_item_id_unchanged(self, state):
        r = handle_add_component({
            "name": "c", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        assert not r.is_error
        assert "Auto-matched" not in r.text
        item = next(i for i in state.inventory if i.item_id == "placeholder_1")
        assert item.matched_components == []

    def test_add_aligned_component_auto_matches(self, state_with_anchors):
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_with_anchors)
        r = handle_add_aligned_component({
            "name": "next", "asset_id": "ConveyorBelt_A01",
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_anchor": "in",
            "item_id": "placeholder_2",
        }, state_with_anchors)
        assert not r.is_error, r.text
        assert "Auto-matched" in r.text
        item = next(
            i for i in state_with_anchors.inventory
            if i.item_id == "placeholder_2"
        )
        assert item.matched_components == ["next"]

    def test_add_aligned_component_rejects_unknown_item_id(self, state_with_anchors):
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_with_anchors)
        r = handle_add_aligned_component({
            "name": "next", "asset_id": "ConveyorBelt_A01",
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_anchor": "in",
            "item_id": "ghost_item",
        }, state_with_anchors)
        assert r.is_error
        assert "ghost_item" in r.text
        # Pre-validation: alignment was NOT applied.
        assert len(state_with_anchors.prediction.components) == 1


class TestInventoryStatusInListComponents:
    def test_status_appears_when_unlocked(self, state_unlocked):
        r = handle_list_components({}, state_unlocked)
        assert not r.is_error
        assert "not set" in r.text.lower()
        assert "set_inventory" in r.text

    def test_status_appears_when_empty(self, state_unlocked):
        # Lock with empty inventory by mutating state directly
        # (set_inventory rejects empty, but we want to test the formatter).
        state_unlocked.inventory_locked = True
        r = handle_list_components({}, state_unlocked)
        assert "0 items" in r.text or "empty" in r.text

    def test_status_shows_matched_and_unmatched(self, state):
        handle_add_component({
            "name": "c", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        handle_mark_matched(
            {"item_id": "placeholder_1", "component": "c"}, state,
        )
        r = handle_list_components({}, state)
        assert "1/2 matched" in r.text
        assert "✓ placeholder_1" in r.text
        assert "✗ placeholder_2" in r.text


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
    """Submit gate is now driven by inventory matching, not an integer count."""

    def _place_and_match(self, state, item_ids: list[str]) -> None:
        """Place one component per inventory id and mark each matched."""
        for i, iid in enumerate(item_ids):
            handle_add_component({
                "name": f"c_{iid}",
                "asset_id": "ConveyorBelt_A01",
                "position": [float(i), 0, 0],
            }, state)
            handle_mark_matched(
                {"item_id": iid, "component": f"c_{iid}"}, state,
            )
        # Pretend a render happened so the render-after-edit gate doesn't
        # fire.  state.renderer is None in the fixture so the gate is
        # bypassed anyway; this defensively documents intent.
        state.scene_modified_since_render = False

    def test_submit_requires_inventory_locked(self, state_unlocked):
        # No set_inventory call — the gate must reject.
        r = handle_submit_prediction({}, state_unlocked)
        assert r.is_error
        assert "inventory" in r.text.lower()
        assert state_unlocked.submitted is False

    def test_submit_rejects_unmatched_items(self, state):
        # Default fixture inventory has placeholder_1, placeholder_2.
        # Place a component but mark NEITHER inventory item.
        handle_add_component({
            "name": "c0", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state)
        state.scene_modified_since_render = False
        r = handle_submit_prediction({}, state)
        assert r.is_error
        assert "unmatched" in r.text.lower()
        assert "placeholder_1" in r.text or "placeholder_2" in r.text
        assert state.submitted is False

    def test_submit_happy_path(self, state):
        self._place_and_match(state, ["placeholder_1", "placeholder_2"])
        r = handle_submit_prediction({"notes": "done"}, state)
        assert not r.is_error, r.text
        assert state.submitted is True
        assert state.submit_notes == "done"

    def test_submit_acknowledge_unmatched_requires_notes(self, state):
        r = handle_submit_prediction({
            "acknowledge_unmatched": ["placeholder_1", "placeholder_2"],
        }, state)
        assert r.is_error
        assert "notes" in r.text.lower()
        assert state.submitted is False

    def test_submit_acknowledge_unmatched_succeeds(self, state):
        r = handle_submit_prediction({
            "acknowledge_unmatched": ["placeholder_1", "placeholder_2"],
            "notes": "neither object in pool",
        }, state)
        assert not r.is_error, r.text
        assert state.submitted is True

    def test_submit_rejects_unknown_acknowledge_id(self, state):
        # Match one item, acknowledge a nonexistent id.
        self._place_and_match(state, ["placeholder_1"])
        r = handle_submit_prediction({
            "acknowledge_unmatched": ["ghost_item"],
            "notes": "...",
        }, state)
        assert r.is_error
        assert "ghost_item" in r.text
        assert state.submitted is False

    def test_submit_rejects_acknowledge_non_list(self, state):
        self._place_and_match(state, ["placeholder_1", "placeholder_2"])
        r = handle_submit_prediction({
            "acknowledge_unmatched": True,  # old-style boolean
        }, state)
        assert r.is_error
        assert "list" in r.text.lower()
        assert state.submitted is False

    def test_submit_rejects_locked_but_empty_inventory(self, state):
        state.inventory.clear()  # locked but empty
        r = handle_submit_prediction({}, state)
        assert r.is_error
        assert "empty" in r.text.lower()
        assert state.submitted is False

    def test_submit_rejects_when_scene_modified_since_render(self, state):
        # Simulate render tool wired in via a sentinel renderer object.
        state.renderer = object()
        self._place_and_match(state, ["placeholder_1", "placeholder_2"])
        # Mark scene as modified after the synthetic render.
        state.scene_modified_since_render = True
        r = handle_submit_prediction({}, state)
        assert r.is_error
        assert "render" in r.text.lower()
        assert state.submitted is False

    def test_submit_render_gate_skipped_without_renderer(self, state):
        state.renderer = None
        self._place_and_match(state, ["placeholder_1", "placeholder_2"])
        state.scene_modified_since_render = True
        r = handle_submit_prediction({}, state)
        assert not r.is_error
        assert state.submitted is True

    def test_submit_blocks_on_extra_unmatched_component(self, state):
        self._place_and_match(state, ["placeholder_1", "placeholder_2"])
        # Add an extra component, leave it unmatched to any inventory item.
        handle_add_component({
            "name": "extra_c",
            "asset_id": "ConveyorBelt_A02",
            "position": [10, 0, 0],
        }, state)
        state.scene_modified_since_render = False
        r = handle_submit_prediction({}, state)
        assert r.is_error
        assert "extra_c" in r.text
        assert "not matched" in r.text.lower()
        assert state.submitted is False

    def test_submit_acknowledge_extras_succeeds(self, state):
        self._place_and_match(state, ["placeholder_1", "placeholder_2"])
        handle_add_component({
            "name": "scaffold",
            "asset_id": "ConveyorBelt_A02",
            "position": [10, 0, 0],
        }, state)
        state.scene_modified_since_render = False
        r = handle_submit_prediction({
            "acknowledge_extras": ["scaffold"],
            "notes": "intentional scaffolding",
        }, state)
        assert not r.is_error, r.text
        assert state.submitted is True

    def test_submit_acknowledge_extras_requires_notes(self, state):
        self._place_and_match(state, ["placeholder_1", "placeholder_2"])
        handle_add_component({
            "name": "scaffold",
            "asset_id": "ConveyorBelt_A02",
            "position": [10, 0, 0],
        }, state)
        state.scene_modified_since_render = False
        r = handle_submit_prediction({
            "acknowledge_extras": ["scaffold"],
        }, state)
        assert r.is_error
        assert "notes" in r.text.lower()
        assert state.submitted is False

    def test_submit_extras_gate_surfaces_undercount_hint(self, state):
        # Lock with 1 item; place 3 components, mark only 1.  This is the
        # exact pattern from the v14 smoke test that motivated the gate.
        state.inventory = [InventoryItem(item_id="u_assembly", description="the whole U")]
        for i in range(3):
            handle_add_component({
                "name": f"p{i}", "asset_id": "ConveyorBelt_A01",
                "position": [float(i), 0, 0],
            }, state)
        handle_mark_matched(
            {"item_id": "u_assembly", "component": "p0"}, state,
        )
        state.scene_modified_since_render = False
        r = handle_submit_prediction({}, state)
        assert r.is_error
        # The dedicated under-counting hint should fire (placed=3, items=1).
        assert "under-counted" in r.text.lower()
        assert "add_inventory_item" in r.text
        assert state.submitted is False

    def test_submit_rejects_extras_non_list(self, state):
        self._place_and_match(state, ["placeholder_1", "placeholder_2"])
        r = handle_submit_prediction({
            "acknowledge_extras": "p1",  # str, not list
        }, state)
        assert r.is_error
        assert "list" in r.text.lower()
        assert state.submitted is False


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

    def test_allows_back_to_back_add(self, state):
        # Adds are intentionally exempt from the render-after-edit gate so
        # the agent can batch placements (one turn per item instead of three).
        self._enable_render_gate(state)
        handle_add_component({
            "name": "a", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        r = handle_add_component({
            "name": "b", "asset_id": "ConveyorBelt_A02", "position": [1, 0, 0],
        }, state)
        assert not r.is_error, r.text
        assert len(state.prediction.components) == 2

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

    def test_allows_remove_after_modify(self, state):
        # Remove is exempt from the render-after-edit gate (like add) —
        # a remove is a deterministic-intent action, not a "did this
        # help?" adjustment.  Only modify / align retain the gate.
        self._enable_render_gate(state)
        handle_add_component({
            "name": "a", "asset_id": "ConveyorBelt_A01", "position": [0, 0, 0],
        }, state)
        state.scene_modified_since_render = False
        handle_modify_component({"name": "a", "position": [1, 0, 0]}, state)
        # scene_modified_since_render is now True; remove should still go through.
        r = handle_remove_component({"name": "a"}, state)
        assert not r.is_error, r.text
        assert len(state.prediction.components) == 0

    def test_allows_back_to_back_remove(self, state):
        # Batch removes (like the v5 packing-table wholesale-redo case)
        # should all succeed without intervening renders.
        self._enable_render_gate(state)
        for i in range(3):
            handle_add_component({
                "name": f"c{i}",
                "asset_id": "ConveyorBelt_A01",
                "position": [float(i), 0, 0],
            }, state)
        state.scene_modified_since_render = False
        for i in range(3):
            r = handle_remove_component({"name": f"c{i}"}, state)
            assert not r.is_error, f"remove c{i}: {r.text}"
        assert state.prediction.components == []

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

    def test_bypasses_edit_gate_when_renderer_present(self, state_with_anchors):
        # add_aligned_component is intentionally exempt from the edit gate —
        # anchor-based placement is geometrically deterministic, no
        # per-edit attribution needed.  Gate still applies to modify/align.
        state_with_anchors.renderer = object()
        handle_add_component({
            "name": "fixed", "asset_id": "ConveyorBelt_A01",
            "position": [0, 0, 0],
        }, state_with_anchors)
        # Don't clear the flag — should still succeed.
        r = handle_add_aligned_component({
            "name": "next", "asset_id": "ConveyorBelt_A01",
            "fixed_component": "fixed", "fixed_anchor": "out",
            "moving_anchor": "in",
        }, state_with_anchors)
        assert not r.is_error, r.text
        assert len(state_with_anchors.prediction.components) == 2


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

