"""CLIP-based visual asset retrieval.

Given a query image, find the most similar asset category from the taxonomy
using CLIP text-image cross-modal matching.

Text descriptions are auto-generated from taxonomy metadata using prompt
ensembling (multiple templates per category).  This generalizes to any
taxonomy without hardcoded descriptions.

The text embedding index is cached to disk so subsequent queries skip encoding.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaacsim_bench.schemas.taxonomy import AssetTaxonomy, Category

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt ensemble templates (CLIP paper §3.1.4)
# ---------------------------------------------------------------------------

_TEMPLATES = [
    "a photo of a {}",
    "a rendered image of a {}",
    "a 3D model of a {}",
    "a {} in a simulation environment",
    "a close-up photo of a {}",
]

# Family-level context injected into some templates for disambiguation
_FAMILY_CONTEXT: dict[str, str] = {
    "barrel": "industrial barrel or drum",
    "bottle": "plastic bottle",
    "box": "box or container",
    "building": "building structural element",
    "container": "industrial storage container",
    "conveyor": "conveyor belt section",
    "crate": "plastic storage crate",
    "environment": "3D environment or scene",
    "isaaclab": "Isaac Lab simulation asset",
    "pallet": "shipping pallet",
    "person": "human figure",
    "prop": "object or prop",
    "rack": "warehouse shelving rack component",
    "robot": "robot",
    "safety": "safety equipment",
    "sample": "simulation sample asset",
    "sensor": "sensor device",
    "signage": "sign or label",
    "vehicle": "vehicle",
}


def _clean_name(raw: str) -> str:
    """Turn 'BarelPlastic_A_01' into 'plastic barrel variant A'."""
    # Split camelCase and underscores
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw).replace("_", " ").lower().split()
    # Drop trailing numeric-only tokens (variant numbers)
    while parts and parts[-1].isdigit():
        parts.pop()
    return " ".join(parts) if parts else raw.lower()


def _generate_descriptions(cat: Category) -> list[str]:
    """Auto-generate CLIP text descriptions from category metadata."""
    name = cat.name.lower()
    family = cat.family
    context = _FAMILY_CONTEXT.get(family, family)

    descriptions: list[str] = []

    # 1. Prompt templates with the category name
    for tmpl in _TEMPLATES:
        descriptions.append(tmpl.format(name))

    # 2. Family-contextualized description
    descriptions.append(f"a {context}")
    descriptions.append(f"a {name}, which is a type of {context}")

    # 3. Sample variant names (up to 3) — adds specificity for diverse categories
    seen: set[str] = set()
    for var in cat.variants[:5]:
        cleaned = _clean_name(var.name)
        if cleaned not in seen and cleaned != name:
            seen.add(cleaned)
            descriptions.append(f"a {cleaned}")
        if len(seen) >= 3:
            break

    return descriptions


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Single retrieval hit."""

    category_id: str
    category_name: str
    family: str
    score: float
    variant_ids: list[str]


# ---------------------------------------------------------------------------
# AssetRetriever
# ---------------------------------------------------------------------------


class AssetRetriever:
    """CLIP-based asset retrieval from the taxonomy catalog.

    Parameters
    ----------
    taxonomy_path
        Path to ``asset_taxonomy.json``.
    model_name
        OpenCLIP model name (default: ViT-B-32).
    pretrained
        Pretrained weights tag (default: laion2b_s34b_b79k).
    cache_dir
        Directory for cached text embeddings.  Defaults to
        ``<taxonomy_dir>/.clip_cache/``.
    device
        Torch device.  Auto-detects CUDA if available.
    """

    def __init__(
        self,
        taxonomy_path: str | Path,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        cache_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = AssetTaxonomy.model_validate_json(
            self.taxonomy_path.read_text()
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained

        self._cache_dir = (
            Path(cache_dir)
            if cache_dir
            else self.taxonomy_path.parent / ".clip_cache"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded
        self._model = None
        self._preprocess = None
        self._tokenizer = None

        # Built during _ensure_index()
        self._text_embeddings: np.ndarray | None = None  # (N_cats, D)
        self._category_meta: list[dict] | None = None  # parallel to rows

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._model.eval()
        logger.info(
            "Loaded CLIP model %s (%s) on %s",
            self.model_name,
            self.pretrained,
            self.device,
        )

    # ------------------------------------------------------------------
    # Text embedding index
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        tag = f"{self.model_name}_{self.pretrained}_{self.taxonomy.version}"
        safe = tag.replace("/", "_").replace(" ", "_")
        return self._cache_dir / f"text_embeddings_{safe}.npz"

    def _ensure_index(self) -> None:
        if self._text_embeddings is not None:
            return

        cache = self._cache_path()
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            self._text_embeddings = data["embeddings"]
            self._category_meta = json.loads(str(data["meta"]))
            logger.info(
                "Loaded cached text embeddings (%d categories) from %s",
                len(self._category_meta),
                cache,
            )
            return

        # Build from scratch
        self._ensure_model()
        logger.info("Building text embedding index for %d categories ...", len(self.taxonomy.categories))

        meta: list[dict] = []
        all_embeddings: list[np.ndarray] = []

        for cat in self.taxonomy.categories:
            descs = _generate_descriptions(cat)

            # Encode all descriptions, average them
            tokens = self._tokenizer(descs).to(self.device)
            with torch.no_grad():
                text_features = self._model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                avg_embedding = text_features.mean(dim=0)
                avg_embedding = avg_embedding / avg_embedding.norm()

            all_embeddings.append(avg_embedding.cpu().numpy())
            meta.append(
                {
                    "category_id": cat.category_id,
                    "name": cat.name,
                    "family": cat.family,
                    "variant_ids": [v.variant_id for v in cat.variants],
                }
            )

        self._text_embeddings = np.stack(all_embeddings)  # (N, D)
        self._category_meta = meta

        # Cache to disk
        np.savez(
            cache,
            embeddings=self._text_embeddings,
            meta=json.dumps(meta),
        )
        logger.info("Cached text embeddings to %s", cache)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_image(
        self,
        image: str | Path | np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most similar asset categories for a query image.

        Parameters
        ----------
        image
            File path to an image, or an (H, W, 3) uint8 numpy array.
        top_k
            Number of results to return.

        Returns
        -------
        List of RetrievalResult sorted by descending similarity score.
        """
        self._ensure_model()
        self._ensure_index()

        from PIL import Image

        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = Image.fromarray(image)

        img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self._model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        img_np = img_features.cpu().numpy()  # (1, D)
        scores = (img_np @ self._text_embeddings.T).squeeze(0)  # (N,)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            meta = self._category_meta[idx]
            results.append(
                RetrievalResult(
                    category_id=meta["category_id"],
                    category_name=meta["name"],
                    family=meta["family"],
                    score=float(scores[idx]),
                    variant_ids=meta["variant_ids"],
                )
            )

        return results

    def build_index(self) -> None:
        """Pre-build the text embedding index (useful for warming up)."""
        self._ensure_index()
