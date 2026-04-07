from __future__ import annotations

from pydantic import BaseModel


class PredictedComponent(BaseModel):
    name: str
    asset_id: str
    family: str
    translate: list[float]
    orientation_xyzw: list[float] = [0.0, 0.0, 0.0, 1.0]
    confidence: float = 1.0


class PredictedRelation(BaseModel):
    type: str
    from_node: str
    to_node: str
    from_anchor: str
    to_anchor: str


class PredictionJSON(BaseModel):
    sample_id: str
    predicted_family: str
    predicted_template: str
    components: list[PredictedComponent]
    relations: list[PredictedRelation]
    abstained: bool = False
    latency_seconds: float = 0.0
