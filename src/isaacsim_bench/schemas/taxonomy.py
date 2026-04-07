from __future__ import annotations

from pydantic import BaseModel


class Variant(BaseModel):
    variant_id: str
    name: str
    usd_path: str | None = None
    semantic_class: str | None = None


class Category(BaseModel):
    category_id: str
    name: str
    family: str
    variants: list[Variant]


class AssetTaxonomy(BaseModel):
    version: str
    categories: list[Category]
