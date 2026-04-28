"""Orchestrator — coordinate the multi-agent scene reconstruction pipeline.

Pipeline:
    images → CLIP Catalog → Perception (with thumbnails) → Retrieval
           → Spatial → Composer → Critic loop
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from isaacsim_bench.agents.composer import ComposerAgent
from isaacsim_bench.agents.critic import CriticAgent
from isaacsim_bench.agents.perception import PerceptionAgent
from isaacsim_bench.agents.retrieval import RetrievalAgent
from isaacsim_bench.agents.spatial import SpatialAgent
from isaacsim_bench.agents.vlm import VLMClient
from isaacsim_bench.schemas.prediction import PredictionJSON
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

MAX_REFINEMENT_ROUNDS = 2

# Critic issue types that require re-running perception + retrieval
# (as opposed to just re-running spatial estimation)
_STRUCTURAL_ISSUES = {"wrong_count", "wrong_asset"}


class SceneReconstructor:
    """Coordinate CLIP → perception → retrieval → spatial → compose → critic."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        taxonomy_path: Path | None = None,
        retrieval_pool_path: Path | None = None,
        thumbnail_dir: Path | None = None,
        max_refinements: int = MAX_REFINEMENT_ROUNDS,
        clip_top_k: int = 15,
        save_intermediates: bool = False,
    ) -> None:
        self.vlm = VLMClient(model=model, provider=provider)
        self.max_refinements = max_refinements
        self.clip_top_k = clip_top_k
        self.save_intermediates = save_intermediates

        tax_path = taxonomy_path or DATA_DIR / "asset_taxonomy.json"
        pool_path = retrieval_pool_path or DATA_DIR / "retrieval_pool.json"
        thumb_dir = thumbnail_dir or DATA_DIR / "asset_thumbnails"

        self.taxonomy = AssetTaxonomy.model_validate_json(tax_path.read_text())

        pool_data = json.loads(pool_path.read_text())
        self.retrieval_pool_ids: set[str] = set(pool_data["asset_ids"])

        self.thumbnail_dir = thumb_dir if thumb_dir.exists() else None

        # CLIP catalog (optional — needs thumbnails + torch)
        self.clip_catalog = None
        if self.thumbnail_dir:
            try:
                from isaacsim_bench.agents.clip_catalog import CLIPCatalog

                self.clip_catalog = CLIPCatalog(
                    thumbnail_dir=self.thumbnail_dir,
                    taxonomy_path=tax_path,
                )
            except ImportError:
                logger.warning(
                    "CLIP catalog unavailable (missing torch/open_clip). "
                    "Falling back to VLM-only perception."
                )

        # Agents
        self.perception = PerceptionAgent(self.vlm)
        self.retrieval = RetrievalAgent(
            self.vlm, self.taxonomy, self.retrieval_pool_ids, self.thumbnail_dir,
        )
        self.spatial = SpatialAgent(self.vlm)
        self.composer = ComposerAgent()
        self.critic = CriticAgent(self.vlm)

    def reconstruct(
        self,
        image_dir: Path,
        scene_id: str | None = None,
    ) -> PredictionJSON:
        """Run the full pipeline and return a PredictionJSON."""
        all_images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not all_images:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Filter out blank / placeholder images (< 10 KB is almost certainly
        # an all-black 1024x1024 PNG placeholder from a failed render).
        MIN_IMAGE_BYTES = 10_000
        image_paths = [p for p in all_images if p.stat().st_size >= MIN_IMAGE_BYTES]
        skipped = len(all_images) - len(image_paths)
        if skipped:
            print(f"  Filtered out {skipped} blank images "
                  f"(< {MIN_IMAGE_BYTES // 1000} KB)")
        if not image_paths:
            raise FileNotFoundError(
                f"All {len(all_images)} images in {image_dir} appear blank"
            )

        sid = scene_id or image_dir.parent.name
        t0 = time.time()
        stage_times: dict[str, float] = {}

        # ---- Step 1: CLIP visual catalog ----
        t_stage = time.time()
        catalog_hits = None
        if self.clip_catalog:
            print(f"\n[1/6] CLIP catalog — scanning {len(image_paths)} images …")
            catalog_hits = self.clip_catalog.query(
                image_paths,
                top_k=self.clip_top_k,
                must_include_ids=self.retrieval_pool_ids,
            )
            print(f"  Top {len(catalog_hits)} candidate assets:")
            for hit in catalog_hits:
                print(
                    f"    {hit.variant_id:30s}  {hit.family:12s}  "
                    f"score={hit.score:.3f}  ({hit.category_name})"
                )
        else:
            print("\n[1/6] CLIP catalog — skipped (no thumbnails)")
        stage_times["clip"] = time.time() - t_stage

        # ---- Step 2: Perception (with visual catalog) ----
        t_stage = time.time()
        print(f"\n[2/6] Perception — analysing scene with catalog …")
        perception = self.perception.analyze(image_paths, catalog_hits)

        print(f"  Scene: {perception.scene_description}")
        print(f"  Objects ({len(perception.objects)}):")
        for obj in perception.objects:
            print(f"    - {obj.name}  asset={obj.asset_id}  ({obj.family}/{obj.sub_type})")
        print(f"  Connections: {len(perception.connections)}")
        stage_times["perception"] = time.time() - t_stage

        # ---- Step 3: Retrieval (validate / resolve asset IDs) ----
        t_stage = time.time()
        print(f"\n[3/6] Retrieval — validating asset matches …")
        matches = self.retrieval.match(
            perception.objects, image_paths, catalog_hits,
        )

        for m in matches:
            print(f"    {m.name}  ->  {m.asset_id}  (conf={m.confidence:.2f})")
        stage_times["retrieval"] = time.time() - t_stage

        # ---- Step 4: Spatial estimation ----
        t_stage = time.time()
        print(f"\n[4/6] Spatial — estimating layout …")
        layout = self.spatial.estimate(image_paths, perception, matches)
        self._print_layout(layout)
        stage_times["spatial"] = time.time() - t_stage

        # ---- Step 5: Compose ----
        print(f"\n[5/6] Composing prediction …")
        prediction = self.composer.compose(sid, matches, layout, 0.0)

        if self.save_intermediates:
            self._save_intermediate(image_dir.parent, "initial", prediction)

        # ---- Step 6: Critic loop ----
        for round_i in range(self.max_refinements + 1):
            t_stage = time.time()
            round_label = "initial" if round_i == 0 else f"round {round_i}"
            print(f"\n[6/6] Critic review ({round_label}) …")
            review = self.critic.review(image_paths, prediction, perception)

            print(f"  Score: {review.score:.0f}/10  "
                  f"{'PASS' if review.passed else 'FAIL'}")
            if review.issues:
                for issue in review.issues:
                    print(f"    [{issue.issue_type}] {issue.component_name}: "
                          f"{issue.description}")
            print(f"  Reasoning: {review.reasoning}")
            stage_times[f"critic_{round_label}"] = time.time() - t_stage

            if review.passed:
                print("  -> Accepted")
                break

            if round_i >= self.max_refinements:
                print(f"  -> Max refinements ({self.max_refinements}) reached, "
                      f"using best effort")
                break

            # Route critic feedback to the right agent(s)
            issue_types = {i.issue_type for i in review.issues}
            has_structural = bool(issue_types & _STRUCTURAL_ISSUES)

            if has_structural:
                # Structural errors (wrong count, wrong asset) require
                # re-running perception and retrieval, not just spatial.
                print(f"\n  Structural issues detected — "
                      f"re-running perception + retrieval (round {round_i + 1}) …")
                prev_asset_ids = sorted(m.asset_id for m in matches)

                t_stage = time.time()
                # Feed critic feedback to perception as additional context
                perception = self.perception.analyze(
                    image_paths, catalog_hits, critic_feedback=review,
                )
                stage_times[f"re_perception_{round_i + 1}"] = time.time() - t_stage

                print(f"  Re-perceived {len(perception.objects)} objects")

                t_stage = time.time()
                matches = self.retrieval.match(
                    perception.objects, image_paths, catalog_hits,
                )
                stage_times[f"re_retrieval_{round_i + 1}"] = time.time() - t_stage

                for m in matches:
                    print(f"    {m.name}  ->  {m.asset_id}  (conf={m.confidence:.2f})")

                # Convergence check: if re-perception produced the same
                # assets, the system is confident — stop iterating.
                new_asset_ids = sorted(m.asset_id for m in matches)
                if new_asset_ids == prev_asset_ids:
                    print(f"\n  Converged — same assets after re-perception, "
                          f"accepting current prediction.")
                    break

            t_stage = time.time()
            print(f"\n  Refining layout (round {round_i + 1}) …")
            layout = self.spatial.refine(
                image_paths, perception, matches, layout, review,
            )
            self._print_layout(layout)
            stage_times[f"re_spatial_{round_i + 1}"] = time.time() - t_stage

            prediction = self.composer.compose(sid, matches, layout, 0.0)

            if self.save_intermediates:
                self._save_intermediate(
                    image_dir.parent, f"round_{round_i + 1}", prediction,
                )

        # Final prediction with total elapsed time
        elapsed = time.time() - t0
        prediction.latency_seconds = elapsed

        # Print summary
        print(f"\n{'=' * 50}")
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Components: {len(prediction.components)}")
        print(f"  Relations:  {len(prediction.relations)}")
        print(f"  Stage timing:")
        for stage, dt in stage_times.items():
            print(f"    {stage:25s} {dt:6.1f}s")
        print(f"  VLM usage: {self.vlm.usage.summary()}")

        return prediction

    @staticmethod
    def _print_layout(layout) -> None:
        for pose in layout.poses:
            p = pose.position
            print(
                f"    {pose.name}  pos=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})  "
                f"heading={pose.heading_deg:.0f}deg"
            )
        print(f"  Relations: {len(layout.relations)}")
        for rel in layout.relations:
            print(
                f"    {rel.from_node} --[{rel.type}]--> {rel.to_node}  "
                f"({rel.from_anchor} -> {rel.to_anchor})"
            )

    @staticmethod
    def _save_intermediate(
        scene_dir: Path, label: str, prediction: PredictionJSON,
    ) -> None:
        """Save intermediate prediction for debugging."""
        out = scene_dir / f"prediction_{label}.json"
        out.write_text(prediction.model_dump_json(indent=2))
        logger.info("Saved intermediate prediction: %s", out)
