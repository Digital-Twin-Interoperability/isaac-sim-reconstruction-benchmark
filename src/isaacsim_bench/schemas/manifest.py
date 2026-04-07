from __future__ import annotations

from pydantic import BaseModel


class BenchmarkManifest(BaseModel):
    benchmark_version: str
    asset_taxonomy_version: str
    world_pool_version: str
    retrieval_pool_version: str
    generator_version: str
    created_at: str
    sample_count: int
