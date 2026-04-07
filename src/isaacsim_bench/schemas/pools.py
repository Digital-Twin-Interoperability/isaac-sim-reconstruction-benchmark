from __future__ import annotations

from pydantic import BaseModel


class PoolDefinition(BaseModel):
    version: str
    asset_ids: list[str]
