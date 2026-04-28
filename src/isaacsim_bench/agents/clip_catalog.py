"""CLIP visual catalog — image-to-image retrieval against rendered thumbnails.

Compares scene images against the rendered asset thumbnail library to
identify which assets are likely present in the scene.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

logger = logging.getLogger(__name__)


@dataclass
class CatalogHit:
    """A single CLIP retrieval hit from the thumbnail catalog."""

    variant_id: str
    category_id: str
    category_name: str
    family: str
    score: float
    thumbnail_path: Path


class CLIPCatalog:
    """Image-to-image CLIP retrieval using rendered asset thumbnails.

    Builds an embedding index over all available thumbnail PNGs, then
    scores scene images against that index to find candidate assets.
    """

    def __init__(
        self,
        thumbnail_dir: str | Path,
        taxonomy_path: str | Path,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        import torch

        self.thumbnail_dir = Path(thumbnail_dir)
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = AssetTaxonomy.model_validate_json(
            self.taxonomy_path.read_text()
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained

        self._cache_dir = self.taxonomy_path.parent / ".clip_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # variant_id → (category_id, category_name, family)
        self._variant_meta: dict[str, tuple[str, str, str]] = {}
        for cat in self.taxonomy.categories:
            for var in cat.variants:
                self._variant_meta[var.variant_id] = (
                    cat.category_id, cat.name, cat.family,
                )

        # Lazy
        self._model = None
        self._preprocess = None
        self._thumb_embeddings: np.ndarray | None = None  # (N, D)
        self._thumb_ids: list[str] | None = None  # parallel variant_ids

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        logger.info(
            "CLIP catalog: loaded %s (%s) on %s",
            self.model_name, self.pretrained, self.device,
        )

    def _cache_path(self) -> Path:
        tag = f"{self.model_name}_{self.pretrained}_thumbs"
        safe = tag.replace("/", "_").replace(" ", "_")
        return self._cache_dir / f"thumb_embeddings_{safe}.npz"

    def _ensure_index(self) -> None:
        if self._thumb_embeddings is not None:
            return

        cache = self._cache_path()
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            self._thumb_embeddings = data["embeddings"]
            self._thumb_ids = list(data["ids"])
            logger.info(
                "Loaded cached thumbnail embeddings (%d assets) from %s",
                len(self._thumb_ids), cache,
            )
            return

        self._ensure_model()
        self._build_index()

    def _build_index(self) -> None:
        import torch
        from PIL import Image

        thumb_paths = sorted(self.thumbnail_dir.glob("*.png"))
        logger.info("Building thumbnail CLIP index for %d images …", len(thumb_paths))

        ids: list[str] = []
        embeddings: list[np.ndarray] = []
        batch_size = 32

        for i in range(0, len(thumb_paths), batch_size):
            batch_paths = thumb_paths[i : i + batch_size]
            batch_imgs = []
            batch_ids = []

            for p in batch_paths:
                vid = p.stem
                if vid not in self._variant_meta:
                    continue
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(self._preprocess(img))
                    batch_ids.append(vid)
                except Exception:
                    continue

            if not batch_imgs:
                continue

            img_tensor = torch.stack(batch_imgs).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_image(img_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)

            embeddings.append(feats.cpu().numpy())
            ids.extend(batch_ids)

            if (i // batch_size) % 20 == 0:
                logger.info("  … encoded %d / %d thumbnails", len(ids), len(thumb_paths))

        self._thumb_embeddings = np.concatenate(embeddings, axis=0)
        self._thumb_ids = ids

        np.savez(
            self._cache_path(),
            embeddings=self._thumb_embeddings,
            ids=np.array(ids),
        )
        logger.info(
            "Cached %d thumbnail embeddings to %s",
            len(ids), self._cache_path(),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        scene_image_paths: list[Path],
        top_k: int = 20,
        must_include_ids: set[str] | None = None,
    ) -> list[CatalogHit]:
        """Find the top-k most similar assets to the scene images.

        Parameters
        ----------
        must_include_ids
            Variant IDs that are always included in results regardless of
            CLIP score (e.g. retrieval-pool assets).  They appear first,
            ranked by score, followed by CLIP-discovered extras up to
            *top_k* total.
        """
        import torch
        from PIL import Image

        self._ensure_model()
        self._ensure_index()

        must = must_include_ids or set()

        # Per-view max scoring: for each thumbnail take the best score
        # across all scene views (more discriminative than mean embedding).
        all_scores = []
        for p in scene_image_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = self._preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self._model.encode_image(tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                view_scores = (feat.cpu().numpy() @ self._thumb_embeddings.T).squeeze(0)
                all_scores.append(view_scores)
            except Exception:
                continue

        if not all_scores:
            return []

        # Max score per thumbnail across all views
        scores = np.max(np.stack(all_scores, axis=0), axis=0)

        # Build variant_id → score lookup
        vid_to_score: dict[str, float] = {}
        for i, vid in enumerate(self._thumb_ids):
            if vid not in vid_to_score or scores[i] > vid_to_score[vid]:
                vid_to_score[vid] = float(scores[i])

        # Collect must-include hits (retrieval pool assets) first
        must_hits: list[CatalogHit] = []
        seen_categories: set[str] = set()

        for vid in sorted(must, key=lambda v: vid_to_score.get(v, 0), reverse=True):
            meta = self._variant_meta.get(vid)
            if meta is None:
                continue
            thumb = self.thumbnail_dir / f"{vid}.png"
            if not thumb.exists():
                continue
            cat_id, cat_name, family = meta
            seen_categories.add(cat_id)
            must_hits.append(CatalogHit(
                variant_id=vid,
                category_id=cat_id,
                category_name=cat_name,
                family=family,
                score=vid_to_score.get(vid, 0.0),
                thumbnail_path=thumb,
            ))

        # Fill remaining slots with CLIP-discovered extras
        ranked = np.argsort(scores)[::-1]
        extra_hits: list[CatalogHit] = []

        for idx in ranked:
            if len(must_hits) + len(extra_hits) >= top_k:
                break
            vid = self._thumb_ids[idx]
            if vid in must:
                continue
            meta = self._variant_meta.get(vid)
            if meta is None:
                continue
            cat_id, cat_name, family = meta
            if cat_id in seen_categories:
                continue
            seen_categories.add(cat_id)
            extra_hits.append(CatalogHit(
                variant_id=vid,
                category_id=cat_id,
                category_name=cat_name,
                family=family,
                score=float(scores[idx]),
                thumbnail_path=self.thumbnail_dir / f"{vid}.png",
            ))

        return must_hits + extra_hits

    def build_index(self) -> None:
        """Pre-build the thumbnail embedding index (for warming up)."""
        self._ensure_index()
