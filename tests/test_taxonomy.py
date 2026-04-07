from isaacsim_bench.schemas import AssetTaxonomy, PoolDefinition
from isaacsim_bench.taxonomy import (
    TaxonomyRegistry,
    derive_component_regime,
    derive_scene_regime,
)


def _make_registry(
    sample_taxonomy: AssetTaxonomy,
    sample_world_pool: PoolDefinition,
    sample_retrieval_pool: PoolDefinition,
) -> TaxonomyRegistry:
    return TaxonomyRegistry(sample_taxonomy, sample_world_pool, sample_retrieval_pool)


class TestTaxonomyRegistry:
    def test_get_category(self, sample_taxonomy, sample_world_pool, sample_retrieval_pool):
        reg = _make_registry(sample_taxonomy, sample_world_pool, sample_retrieval_pool)
        assert reg.get_category("ConveyorBelt_A01") == "conveyor_straight"
        assert reg.get_category("ConveyorBelt_A10") == "conveyor_curve"
        assert reg.get_category("NonExistent") is None

    def test_is_in_retrieval_pool(self, sample_taxonomy, sample_world_pool, sample_retrieval_pool):
        reg = _make_registry(sample_taxonomy, sample_world_pool, sample_retrieval_pool)
        assert reg.is_in_retrieval_pool("ConveyorBelt_A01") is True
        # A03 is withheld from retrieval pool
        assert reg.is_in_retrieval_pool("ConveyorBelt_A03") is False

    def test_has_proxy(self, sample_taxonomy, sample_world_pool, sample_retrieval_pool):
        reg = _make_registry(sample_taxonomy, sample_world_pool, sample_retrieval_pool)
        # A03 not in retrieval, but A01/A02 same category -> proxy exists
        assert reg.has_proxy_in_retrieval_pool("ConveyorBelt_A03") is True
        # A01 is in retrieval, but also has sibling A02 in retrieval
        assert reg.has_proxy_in_retrieval_pool("ConveyorBelt_A01") is True
        # unknown asset
        assert reg.has_proxy_in_retrieval_pool("Unknown_X") is False


class TestComponentRegime:
    def test_exact_match(self, sample_taxonomy, sample_world_pool, sample_retrieval_pool):
        reg = _make_registry(sample_taxonomy, sample_world_pool, sample_retrieval_pool)
        assert derive_component_regime(reg, "ConveyorBelt_A01") == "exact_match"

    def test_proxy_match(self, sample_taxonomy, sample_world_pool, sample_retrieval_pool):
        reg = _make_registry(sample_taxonomy, sample_world_pool, sample_retrieval_pool)
        # A03 withheld but A01/A02 in retrieval pool
        assert derive_component_regime(reg, "ConveyorBelt_A03") == "proxy_match"
        # A11 withheld but A10 in retrieval pool
        assert derive_component_regime(reg, "ConveyorBelt_A11") == "proxy_match"

    def test_unknown(self, sample_taxonomy, sample_world_pool, sample_retrieval_pool):
        reg = _make_registry(sample_taxonomy, sample_world_pool, sample_retrieval_pool)
        assert derive_component_regime(reg, "TotallyUnknown") == "unknown"


class TestSceneRegime:
    def test_scene_exact(self):
        assert derive_scene_regime(["exact_match", "exact_match"]) == "scene_exact"

    def test_scene_proxy(self):
        assert derive_scene_regime(["exact_match", "proxy_match"]) == "scene_proxy"

    def test_scene_unknown(self):
        assert derive_scene_regime(["exact_match", "unknown"]) == "scene_unknown"
        assert derive_scene_regime(["proxy_match", "unknown"]) == "scene_unknown"

    def test_empty(self):
        assert derive_scene_regime([]) == "scene_exact"
