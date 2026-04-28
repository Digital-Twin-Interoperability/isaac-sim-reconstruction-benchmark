"""Anchor schema and loader for the modular-asset chaining workflow.

The ground truth lives in two files:

* ``data/asset_anchors.json`` — written by ``scripts/extract_anchors.py``
  from each asset's USD.  Never hand-edit; it gets regenerated.
* ``data/asset_anchors_overrides.json`` — hand-authored overlay used to
  flag degenerate anchors and add semantic aliases (``in``, ``out``,
  ``curve_entry``, ...).  Persists across re-extraction.

The override file is deep-merged on top of extraction, per anchor — so
flipping just ``valid: false`` without rewriting the pose is supported.
Resolution follows aliases to raw anchor names.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel, Field


class AnchorPose(BaseModel):
    """One anchor in an asset's local frame.

    ``orient_wxyz`` is USD's quaternion convention (real first).  Convert
    to ``orientation_xyzw`` at the ``PredictedComponent`` boundary.
    """

    position: list[float]
    orient_wxyz: list[float] = [1.0, 0.0, 0.0, 0.0]
    valid: bool = True


class AssetAnchors(BaseModel):
    anchors: dict[str, AnchorPose] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)


class AnchorRegistry:
    """In-memory anchor lookup by ``(asset_id, anchor_name_or_alias)``."""

    def __init__(self, records: dict[str, AssetAnchors]):
        self._records = records

    @property
    def asset_ids(self) -> list[str]:
        return list(self._records.keys())

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._records

    def get_record(self, asset_id: str) -> AssetAnchors | None:
        return self._records.get(asset_id)

    def resolve(self, asset_id: str, name: str) -> AnchorPose | None:
        rec = self._records.get(asset_id)
        if rec is None:
            return None
        if name in rec.anchors:
            return rec.anchors[name]
        raw = rec.aliases.get(name)
        if raw is not None and raw in rec.anchors:
            return rec.anchors[raw]
        return None

    def list_names(self, asset_id: str) -> list[str]:
        """All accepted names for the asset (raw + aliases), in stable order."""
        rec = self._records.get(asset_id)
        if rec is None:
            return []
        return list(rec.anchors.keys()) + list(rec.aliases.keys())


def _merge_anchor_dict(
    base: dict[str, dict],
    override: dict[str, dict],
) -> dict[str, dict]:
    """Per-anchor deep merge so an override can flip ``valid`` alone."""
    merged = deepcopy(base)
    for name, fields in override.items():
        if name in merged:
            merged[name].update(fields)
        else:
            merged[name] = deepcopy(fields)
    return merged


def _merge_records(
    base: dict[str, dict],
    override: dict[str, dict],
) -> dict[str, dict]:
    merged: dict[str, dict] = deepcopy(base)
    for asset_id, ov in override.items():
        if asset_id not in merged:
            merged[asset_id] = deepcopy(ov)
            continue
        cur = merged[asset_id]
        cur["anchors"] = _merge_anchor_dict(
            cur.get("anchors", {}),
            ov.get("anchors", {}),
        )
        # Aliases are flat; later wins on key conflict.
        cur["aliases"] = {**cur.get("aliases", {}), **ov.get("aliases", {})}
    return merged


def load_anchor_registry(
    extracted_path: Path,
    overrides_path: Path | None = None,
) -> AnchorRegistry:
    """Read extracted JSON and overlay overrides; return a validated registry.

    Missing files are not an error; the loader returns whatever it has so
    callers can run on partial data and surface "no anchors known" cleanly.
    """
    base: dict[str, dict] = {}
    if extracted_path.exists():
        base = json.loads(extracted_path.read_text())
    overrides: dict[str, dict] = {}
    if overrides_path is not None and overrides_path.exists():
        overrides = json.loads(overrides_path.read_text())
    merged = _merge_records(base, overrides)
    records = {aid: AssetAnchors.model_validate(rec) for aid, rec in merged.items()}
    return AnchorRegistry(records)
