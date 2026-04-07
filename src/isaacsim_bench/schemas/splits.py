from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SplitAssignment(BaseModel):
    sample_id: str
    split: Literal["train", "val", "test"]
    holdout_tags: list[str] = []


class SplitConfig(BaseModel):
    assignments: list[SplitAssignment]
