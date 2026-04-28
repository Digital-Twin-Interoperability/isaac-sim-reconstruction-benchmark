"""Retrieval agent — confirm or refine asset_id assignments from perception.

When perception already provides asset_ids (from the CLIP catalog), this
agent validates them and fills in metadata.  When asset_ids are unknown,
it falls back to taxonomy keyword search.  When multiple candidates score
similarly, it uses VLM thumbnail comparison to disambiguate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from isaacsim_bench.agents.perception import DetectedObject
from isaacsim_bench.agents.vlm import VLMClient
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy, Category

logger = logging.getLogger(__name__)

# If multiple keyword candidates tie, use VLM to disambiguate
_VLM_DISAMBIG_MAX_CANDIDATES = 5
# Only trigger VLM disambiguation when top candidates are this close
_VLM_DISAMBIG_SCORE_MARGIN = 1


@dataclass
class AssetMatch:
    """A detected object matched to a specific taxonomy asset."""

    name: str
    asset_id: str
    asset_name: str
    category_id: str
    family: str
    confidence: float


_FAMILY_ALIASES: dict[str, str] = {
    "shelf": "rack",
    "shelving": "rack",
    "drum": "barrel",
    "cardboard_box": "box",
    "wooden_pallet": "pallet",
    "plastic_crate": "crate",
    "cone": "safety",
    "sign": "signage",
    "arm": "robot",
    "manipulator": "robot",
    "forklift": "vehicle",
    "cart": "vehicle",
    "dolly": "vehicle",
}

_SUBTYPE_KEYWORDS: dict[str, list[str]] = {
    "straight": ["straight", "belt"],
    "curve_90": ["curve", "90"],
    "curve_180": ["curve", "180"],
    "roller": ["roller"],
    "shelf_unit": ["shelf"],
    "frame": ["frame"],
    "pile": ["pile"],
    "wooden": ["palette", "pallet", "wood"],
    "plastic": ["plastic"],
    "traffic_cone": ["traffic", "cone"],
    "fire_extinguisher": ["fire", "extinguisher"],
    "wet_floor_sign": ["wet", "floor", "sign"],
}

_DISAMBIG_TOOL = {
    "name": "select_best_match",
    "description": "Select the best matching asset from the candidates",
    "input_schema": {
        "type": "object",
        "properties": {
            "selected_variant_id": {
                "type": "string",
                "description": "The variant_id of the best matching asset",
            },
            "reasoning": {
                "type": "string",
                "description": "Why this asset matches best",
            },
        },
        "required": ["selected_variant_id", "reasoning"],
    },
}


class RetrievalAgent:
    """Validate / resolve asset_ids for detected objects."""

    def __init__(
        self,
        vlm: VLMClient,
        taxonomy: AssetTaxonomy,
        retrieval_pool_ids: set[str],
        thumbnail_dir: Path | None = None,
    ) -> None:
        self.vlm = vlm
        self.taxonomy = taxonomy
        self.retrieval_pool_ids = retrieval_pool_ids
        self.thumbnail_dir = thumbnail_dir

        self._family_to_categories: dict[str, list[Category]] = {}
        self._variant_to_category: dict[str, Category] = {}
        for cat in taxonomy.categories:
            self._family_to_categories.setdefault(cat.family, []).append(cat)
            for var in cat.variants:
                self._variant_to_category[var.variant_id] = cat

    def match(
        self,
        detections: list[DetectedObject],
        scene_image_paths: list[Path],
        catalog_hits: list | None = None,
    ) -> list[AssetMatch]:
        # Build CLIP score lookup: variant_id → score
        clip_scores: dict[str, float] = {}
        if catalog_hits:
            for hit in catalog_hits:
                clip_scores[hit.variant_id] = hit.score

        # Collect low-confidence detections for potential VLM disambiguation
        matches: list[AssetMatch] = []
        needs_disambig: list[tuple[int, DetectedObject, list[tuple[str, str, str]]]] = []

        for det in detections:
            # If perception already assigned a valid asset_id, use it
            if det.asset_id and det.asset_id != "unknown":
                cat = self._variant_to_category.get(det.asset_id)
                if cat:
                    # Check: did CLIP rank a same-category variant higher?
                    chosen_clip = clip_scores.get(det.asset_id, 0.0)
                    better_alt = None
                    for var in cat.variants:
                        alt_clip = clip_scores.get(var.variant_id, 0.0)
                        if (
                            var.variant_id != det.asset_id
                            and alt_clip > chosen_clip + 0.005
                        ):
                            if better_alt is None or alt_clip > clip_scores.get(better_alt, 0.0):
                                better_alt = var.variant_id

                    if better_alt and self.thumbnail_dir and scene_image_paths:
                        # CLIP says a same-category variant is a better match.
                        # Queue for VLM disambiguation.
                        cands = [
                            (det.asset_id, det.asset_id, cat.category_id),
                            (better_alt, better_alt, cat.category_id),
                        ]
                        idx = len(matches)
                        matches.append(AssetMatch(
                            name=det.name,
                            asset_id=det.asset_id,
                            asset_name=next(
                                (v.name for v in cat.variants
                                 if v.variant_id == det.asset_id),
                                det.asset_id,
                            ),
                            category_id=cat.category_id,
                            family=cat.family,
                            confidence=0.85,
                        ))
                        needs_disambig.append((idx, det, cands))
                        logger.info(
                            "Retrieval: %s chose %s (CLIP %.3f) but %s scored higher (%.3f) — queuing disambiguation",
                            det.name, det.asset_id, chosen_clip, better_alt, clip_scores.get(better_alt, 0.0),
                        )
                        continue

                    matches.append(AssetMatch(
                        name=det.name,
                        asset_id=det.asset_id,
                        asset_name=next(
                            (v.name for v in cat.variants
                             if v.variant_id == det.asset_id),
                            det.asset_id,
                        ),
                        category_id=cat.category_id,
                        family=cat.family,
                        confidence=0.9,
                    ))
                    continue

            # Fallback: keyword search in taxonomy
            scored_cands = self._find_candidates_scored(det.family, det.sub_type)
            if scored_cands:
                top_score = scored_cands[0][0]
                # Check if disambiguation is needed: multiple candidates
                # within the score margin and thumbnails available
                tied = [c for c in scored_cands if c[0] >= top_score - _VLM_DISAMBIG_SCORE_MARGIN]
                if (
                    len(tied) > 1
                    and self.thumbnail_dir
                    and scene_image_paths
                ):
                    cand_list = [(vid, vname, cid) for _, vid, vname, cid in tied[:_VLM_DISAMBIG_MAX_CANDIDATES]]
                    idx = len(matches)
                    matches.append(AssetMatch(
                        name=det.name,
                        asset_id=cand_list[0][0],
                        asset_name=cand_list[0][1],
                        category_id=cand_list[0][2],
                        family=self._variant_to_category.get(cand_list[0][0], det).family
                        if isinstance(self._variant_to_category.get(cand_list[0][0]), Category)
                        else det.family,
                        confidence=0.5,
                    ))
                    needs_disambig.append((idx, det, cand_list))
                else:
                    _, vid, vname, cat_id = scored_cands[0]
                    cat = self._variant_to_category.get(vid)
                    matches.append(AssetMatch(
                        name=det.name, asset_id=vid, asset_name=vname,
                        category_id=cat_id,
                        family=cat.family if cat else det.family,
                        confidence=0.5,
                    ))
            else:
                matches.append(AssetMatch(
                    name=det.name, asset_id="unknown", asset_name="unknown",
                    category_id="unknown", family=det.family, confidence=0.0,
                ))

        # VLM disambiguation for ambiguous matches
        if needs_disambig:
            logger.info(
                "Retrieval: %d detections need VLM disambiguation",
                len(needs_disambig),
            )
            for idx, det, cands in needs_disambig:
                resolved = self._vlm_disambiguate(
                    det, cands, scene_image_paths,
                )
                if resolved:
                    matches[idx] = resolved

        return matches

    def _vlm_disambiguate(
        self,
        det: DetectedObject,
        candidates: list[tuple[str, str, str]],
        scene_image_paths: list[Path],
    ) -> AssetMatch | None:
        """Use VLM + thumbnails to pick the best candidate for a detection."""
        if not self.thumbnail_dir:
            return None

        # Check which candidates have thumbnails
        valid_cands = []
        for vid, vname, cat_id in candidates:
            thumb = self.thumbnail_dir / f"{vid}.png"
            if thumb.exists():
                valid_cands.append((vid, vname, cat_id, thumb))

        if len(valid_cands) < 2:
            return None

        selected = self.vlm.select_images(scene_image_paths, max_count=4)
        content: list[dict] = []

        # Show scene images
        content.append({"type": "text", "text": "## Scene images\n"})
        for p in selected:
            content.append(self.vlm.encode_image(p))

        # Describe the detection
        content.append({
            "type": "text",
            "text": (
                f"\n## Object to match: \"{det.name}\"\n"
                f"Family: {det.family}, sub-type: {det.sub_type}\n"
                f"Visual: {det.visual_description}\n\n"
                "## Candidate assets — which one matches best?\n"
            ),
        })

        # Show candidate thumbnails
        for vid, vname, cat_id, thumb in valid_cands:
            content.append({
                "type": "text",
                "text": f"\n**{vid}** — {vname} (category: {cat_id})",
            })
            content.append(self.vlm.encode_image(thumb))

        content.append({
            "type": "text",
            "text": (
                "\nSelect the candidate whose thumbnail best matches "
                "the object in the scene images."
            ),
        })

        try:
            raw = self.vlm.query_with_tool(
                system=(
                    "You are an asset matching expert. Given scene images and "
                    "candidate asset thumbnails, select the one that best matches "
                    "the described object in the scene."
                ),
                user_content=content,
                tool=_DISAMBIG_TOOL,
            )
            vid = raw["selected_variant_id"]
            cat = self._variant_to_category.get(vid)
            if cat:
                logger.info(
                    "VLM disambiguation: %s → %s (%s)",
                    det.name, vid, raw.get("reasoning", ""),
                )
                return AssetMatch(
                    name=det.name,
                    asset_id=vid,
                    asset_name=next(
                        (v.name for v in cat.variants if v.variant_id == vid),
                        vid,
                    ),
                    category_id=cat.category_id,
                    family=cat.family,
                    confidence=0.75,
                )
        except Exception:
            logger.warning(
                "VLM disambiguation failed for %s, using keyword match",
                det.name, exc_info=True,
            )
        return None

    def _resolve_family(self, family: str) -> str:
        fam = family.lower().strip()
        if fam in self._family_to_categories:
            return fam
        if fam in _FAMILY_ALIASES:
            return _FAMILY_ALIASES[fam]
        for tax_fam in self._family_to_categories:
            if fam in tax_fam or tax_fam in fam:
                return tax_fam
        return fam

    def _find_candidates_scored(
        self, family: str, sub_type: str,
    ) -> list[tuple[int, str, str, str]]:
        """Return scored candidates as (score, variant_id, name, category_id)."""
        resolved = self._resolve_family(family)
        categories = self._family_to_categories.get(resolved, [])

        keywords = _SUBTYPE_KEYWORDS.get(sub_type, [])
        if not keywords:
            keywords = sub_type.lower().replace("_", " ").split()

        scored_cats: list[tuple[int, Category]] = []
        for cat in categories:
            haystack = (cat.name + " " + cat.category_id).lower()
            score = sum(1 for kw in keywords if kw in haystack)
            scored_cats.append((score, cat))
        scored_cats.sort(key=lambda x: -x[0])

        results: list[tuple[int, str, str, str]] = []
        for cat_score, cat in scored_cats:
            pool, rest = [], []
            for var in cat.variants:
                entry = (cat_score, var.variant_id, var.name, cat.category_id)
                (pool if var.variant_id in self.retrieval_pool_ids else rest).append(entry)
            results.extend(pool)
            results.extend(rest)
        return results
