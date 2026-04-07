from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraParams(BaseModel):
    position: list[float]
    target: list[float]
    fov_deg: float


class ComponentEntry(BaseModel):
    name: str
    asset_id: str
    asset_name: str
    family: str
    evaluation_role: Literal["primary", "distractor"]
    match_regime: Literal["exact_match", "proxy_match", "unknown"]
    translate: list[float]
    orientation_xyzw: list[float] = [0.0, 0.0, 0.0, 1.0]
    acceptable_proxy_family: str | None = None


class RelationEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: str
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    from_anchor: str
    to_anchor: str


class SceneJSON(BaseModel):
    sample_id: str
    benchmark_tier: Literal["closed_world", "coverage_mismatch"]
    family: str
    template_id: str
    root_node: str
    template_params: dict[str, Any]
    camera: CameraParams
    components: list[ComponentEntry]
    relations: list[RelationEntry]

    retrieval_pool_version: str | None = None
    world_pool_version: str | None = None
    taxonomy_version: str | None = None
    generator_version: str | None = None

    @model_validator(mode="after")
    def _root_node_in_components(self) -> SceneJSON:
        names = {c.name for c in self.components}
        if self.root_node not in names:
            raise ValueError(
                f"root_node '{self.root_node}' not found in components: {names}"
            )
        return self
