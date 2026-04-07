from __future__ import annotations

from typing import Literal

from isaacsim_bench.taxonomy.registry import TaxonomyRegistry


def derive_component_regime(
    registry: TaxonomyRegistry, asset_id: str
) -> Literal["exact_match", "proxy_match", "unknown"]:
    if registry.is_in_retrieval_pool(asset_id):
        return "exact_match"
    if registry.has_proxy_in_retrieval_pool(asset_id):
        return "proxy_match"
    return "unknown"


def derive_scene_regime(
    primary_regimes: list[str],
) -> Literal["scene_exact", "scene_proxy", "scene_unknown"]:
    """Derive scene-level regime from primary component regimes only."""
    if not primary_regimes:
        return "scene_exact"
    if any(r == "unknown" for r in primary_regimes):
        return "scene_unknown"
    if any(r == "proxy_match" for r in primary_regimes):
        return "scene_proxy"
    return "scene_exact"
