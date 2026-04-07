# isaacsim-bench

Synthetic benchmark for Isaac Sim asset retrieval and scene assembly.

Given a rendered image of an Isaac Sim scene, evaluate how well a pipeline can:
1. **Retrieve** the correct assets from a 6,321-variant catalog
2. **Infer** the scene template and component list
3. **Recover** spatial relations between components
4. **Place** components with accurate 6-DOF poses (translation + quaternion orientation)

## Project Structure

```
isaacsim-bench/
├── src/isaacsim_bench/
│   ├── schemas/          # Pydantic data models (SceneJSON, PredictionJSON, taxonomy, etc.)
│   ├── taxonomy/         # Asset taxonomy registry, match regime logic
│   ├── extractor/        # Ground-truth extraction from USD scenes
│   ├── retrieval/        # CLIP-based visual asset retrieval
│   ├── evaluator/        # Metrics: retrieval, coverage, component, relation, placement
│   ├── generator/        # Scene generation (templates + Isaac Sim renderer)
│   ├── validator/        # Data validation checks
│   └── cli/              # Click CLI entry point
├── scripts/
│   ├── discover_assets.py    # Crawl Isaac Sim to build asset_taxonomy.json
│   ├── extract_gt.py         # Extract ground truth from a USD file
│   └── retrieve_asset.py     # CLIP image-to-asset retrieval
├── data/
│   ├── asset_taxonomy.json   # 85 categories, 6,321 variants
│   ├── world_pool.json       # All assets available for scene generation
│   ├── retrieval_pool.json   # Subset available to retrieval algorithms
│   └── gt_full_warehouse.json # Extracted GT (3,139 components)
├── samples/                  # Sample scenes, predictions, and rendered views
└── tests/                    # 79 tests covering all modules
```

## Setup

Requires Python 3.10+ (Isaac Sim compatibility).

```bash
uv sync
```

## What Is Done

### Schemas (complete)
- `SceneJSON` — full scene description with 6-DOF poses (translation + quaternion `orientation_xyzw`)
- `PredictionJSON` — predicted scene output for evaluation
- `AssetTaxonomy` — two-level category/variant hierarchy with USD paths and semantic classes
- Pool definitions, benchmark manifest, split configuration

### Asset Taxonomy (complete)
- 85 categories across 18 families (barrel, bottle, box, building, container, conveyor, crate, environment, isaaclab, pallet, person, prop, rack, robot, safety, sample, sensor, signage, vehicle)
- 6,321 total variants with USD paths mapped to Isaac Sim 5.1 assets
- `TaxonomyRegistry` with reverse index lookups and match regime derivation (exact/proxy/unknown)

### Ground-Truth Extractor (complete)
- Parses USD scenes using `pxr` (no Isaac Sim runtime required)
- Resolves prim references against the taxonomy via 3-stage matching (path, stem, suffix-stripped)
- Extracts full 6-DOF transforms (`xformOp:translate` + `xformOp:orient`)
- Classifies components as primary vs. distractor (structural elements)
- Tested on `full_warehouse.usd`: 3,139 components extracted (2,727 primary, 412 distractor)

```bash
uv run python scripts/extract_gt.py full_warehouse.usd data/asset_taxonomy.json -o data/gt_full_warehouse.json
```

### CLIP-Based Visual Retrieval (complete)
- OpenCLIP ViT-B-32 with prompt ensembling (5 templates per category)
- Auto-generates text descriptions from taxonomy metadata (generalizes beyond warehouse)
- Cached text embedding index for fast subsequent queries
- Returns top-k most similar asset categories with similarity scores

```bash
uv run python scripts/retrieve_asset.py path/to/image.png --top-k 5
```

### Evaluator (complete)
- **Retrieval metrics** — Recall@1, Recall@5, MRR by match regime
- **Coverage metrics** — abstention P/R/F1, false-accept rate
- **Template metrics** — family/template accuracy, macro F1
- **Component metrics** — P/R/F1 over matched components, count accuracy
- **Relation metrics** — strict + oracle cascaded scoring
- **Placement metrics** — root-relative translation error (m) + quaternion angular distance (deg)
- **Scene success** — composite gate (family + template + component F1 >= 0.9 + placement thresholds)

### Validator (complete)
Six validation checks: transform sanity, relation connectivity, camera validity, asset existence, match regime consistency, file completeness.

### CLI (complete)
```bash
isaacsim-bench validate <dataset_dir>
isaacsim-bench evaluate <gt_dir> <pred_dir> --output report.json
isaacsim-bench taxonomy stats
```

### Generator (partial)
- Abstract `SceneGenerator` interface defined
- Scene templates for U-conveyor, parallel-conveyor, shelf-grid, and mixed-warehouse layouts
- Isaac Sim renderer stub (requires Isaac Sim runtime)

## What Needs To Be Done

### Scene Reconstruction Pipeline (the "player")
The benchmark infrastructure (the "scoreboard") is complete. The actual reconstruction pipeline — the system being evaluated — still needs to be built:

1. **Object Detection + Cropping** — detect individual objects in a rendered image and crop them for per-object retrieval (current CLIP retrieval operates on full images)
2. **Per-Object Asset Retrieval** — run CLIP retrieval on each detected crop to identify the specific asset variant
3. **Scene Graph Prediction** — infer spatial relations (on-top-of, next-to, etc.) between detected objects
4. **Pose Estimation** — predict 6-DOF placement for each component
5. **Scene Assembly** — combine retrieval + relations + placement into a complete `PredictionJSON`

### Dataset Generation
- Generate a diverse set of benchmark scenes (not just the single warehouse)
- Render RGB, depth, and segmentation images via Isaac Sim Replicator
- Cover multiple scene families (warehouse, conveyor lines, shelf layouts, mixed)
- Produce train/val/test splits with controlled holdout strategies

### Additional Improvements
- Fine-tune or adapt CLIP embeddings on the Isaac Sim asset domain
- Add support for mesh-level comparison metrics (IoU, Chamfer distance)
- Integrate with Isaac Sim's Replicator for automated rendering pipelines
- Leaderboard / reporting dashboard for tracking pipeline iterations

## Running Tests

```bash
uv run pytest
```

Note: some extractor tests require `full_warehouse.usd` and are skipped if the file is not present.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Note: NVIDIA Isaac Sim assets (SimReady) are subject to the
[NVIDIA Isaac Sim Additional Software and Materials License](https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses-isaac-sim.html).
This repository contains benchmark tooling only; no NVIDIA assets are redistributed.
