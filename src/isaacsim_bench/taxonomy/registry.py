from __future__ import annotations

import json
from pathlib import Path

from isaacsim_bench.schemas.pools import PoolDefinition
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy, Variant


class TaxonomyRegistry:
    """Loads taxonomy + pools and provides O(1) lookups for match regime derivation."""

    def __init__(
        self,
        taxonomy: AssetTaxonomy,
        world_pool: PoolDefinition,
        retrieval_pool: PoolDefinition,
    ) -> None:
        self.taxonomy = taxonomy
        self.world_pool_ids = set(world_pool.asset_ids)
        self.retrieval_pool_ids = set(retrieval_pool.asset_ids)

        # variant_id -> category_id
        self._variant_to_category: dict[str, str] = {}
        # category_id -> set of variant_ids that are in the retrieval pool
        self._retrieval_by_category: dict[str, set[str]] = {}
        # variant_id -> Variant model (for usd_path / semantic_class lookups)
        self._variants: dict[str, Variant] = {}

        for cat in taxonomy.categories:
            self._retrieval_by_category[cat.category_id] = set()
            for var in cat.variants:
                self._variant_to_category[var.variant_id] = cat.category_id
                self._variants[var.variant_id] = var
                if var.variant_id in self.retrieval_pool_ids:
                    self._retrieval_by_category[cat.category_id].add(var.variant_id)

    @classmethod
    def load(
        cls,
        taxonomy_path: str | Path,
        world_pool_path: str | Path,
        retrieval_pool_path: str | Path,
    ) -> TaxonomyRegistry:
        taxonomy = AssetTaxonomy.model_validate_json(Path(taxonomy_path).read_text())
        world_pool = PoolDefinition.model_validate_json(
            Path(world_pool_path).read_text()
        )
        retrieval_pool = PoolDefinition.model_validate_json(
            Path(retrieval_pool_path).read_text()
        )
        return cls(taxonomy, world_pool, retrieval_pool)

    def get_category(self, asset_id: str) -> str | None:
        return self._variant_to_category.get(asset_id)

    def is_in_world_pool(self, asset_id: str) -> bool:
        return asset_id in self.world_pool_ids

    def is_in_retrieval_pool(self, asset_id: str) -> bool:
        return asset_id in self.retrieval_pool_ids

    def has_proxy_in_retrieval_pool(self, asset_id: str) -> bool:
        """True if a same-category variant (not asset_id itself) is in retrieval pool."""
        cat = self._variant_to_category.get(asset_id)
        if cat is None:
            return False
        retrieval_variants = self._retrieval_by_category.get(cat, set())
        return bool(retrieval_variants - {asset_id})

    def get_variant(self, asset_id: str) -> Variant | None:
        return self._variants.get(asset_id)

    def resolve_usd(self, asset_id: str, asset_root: str | Path = "") -> Path | None:
        """Resolve an asset_id to a full USD path.

        Returns asset_root / variant.usd_path, or None if the variant has no
        usd_path or the asset_id is unknown.
        """
        var = self._variants.get(asset_id)
        if var is None or var.usd_path is None:
            return None
        return Path(asset_root) / var.usd_path

    def get_semantic_class(self, asset_id: str) -> str | None:
        """Return the semantic class label for segmentation rendering."""
        var = self._variants.get(asset_id)
        if var is None:
            return None
        # Fall back to category_id if no explicit semantic_class
        return var.semantic_class or self._variant_to_category.get(asset_id)
