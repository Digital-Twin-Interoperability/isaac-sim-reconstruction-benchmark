# Implementation Plan: Isaac Sim Benchmark Core Infrastructure

## Context

The project needs a quantitative benchmark for an Isaac Sim asset retrieval and scene assembly pipeline. The repo is greenfield (only `proposal.md` exists). The old pipeline was naive/agent-loop-based; the priority is getting the dataset infrastructure right first. Isaac Sim is available locally, but the scope is **core infra**: taxonomy, schemas, data format, validator, and evaluator. The scene generator and baselines are deferred but interfaces will be defined.

---

## Project Structure

```
/mnt/d/dev/Isaac_Sim_Scene/
├── proposal.md
├── pyproject.toml
├── .python-version                  # 3.10 (Isaac Sim compat)
├── data/
│   ├── asset_taxonomy.json          # two-level category/variant taxonomy
│   ├── world_pool.json              # all assets available for scene generation
│   └── retrieval_pool.json          # subset available to the retrieval algorithm
├── src/
│   └── isaacsim_bench/
│       ├── __init__.py
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── taxonomy.py          # Pydantic: AssetTaxonomy, Category, Variant
│       │   ├── pools.py             # Pydantic: PoolDefinition
│       │   ├── scene.py             # Pydantic: SceneJSON, Component, Relation, Camera
│       │   ├── prediction.py        # Pydantic: PredictionJSON (evaluator input)
│       │   ├── manifest.py          # Pydantic: BenchmarkManifest
│       │   └── splits.py            # Pydantic: SplitConfig, holdout definitions
│       ├── taxonomy/
│       │   ├── __init__.py
│       │   ├── registry.py          # TaxonomyRegistry: load taxonomy + pools, lookups
│       │   └── match_regime.py      # derive_match_regime(), derive_scene_regime()
│       ├── validator/
│       │   ├── __init__.py
│       │   ├── checks.py            # 6 validation check functions
│       │   └── runner.py            # ValidatorRunner: orchestrate checks, produce report
│       ├── evaluator/
│       │   ├── __init__.py
│       │   ├── matching.py          # bipartite component matching (scipy)
│       │   ├── metrics/
│       │   │   ├── __init__.py
│       │   │   ├── retrieval.py     # Recall@K, MRR by match regime
│       │   │   ├── coverage.py      # abstention P/R/F1, false-accept rate
│       │   │   ├── template.py      # family/template accuracy, macro F1
│       │   │   ├── component.py     # component P/R/F1, count accuracy
│       │   │   ├── relation.py      # cascaded scoring: strict + oracle modes
│       │   │   ├── placement.py     # root-relative translation/rotation error
│       │   │   └── scene_success.py # end-to-end composite metric
│       │   └── runner.py            # EvaluatorRunner: run all metrics, produce report
│       ├── generator/
│       │   ├── __init__.py
│       │   └── base.py              # abstract SceneGenerator interface (stub)
│       └── cli/
│           ├── __init__.py
│           └── main.py              # Click CLI: validate, evaluate, taxonomy commands
└── tests/
    ├── conftest.py                  # shared fixtures (sample scene, taxonomy, pools)
    ├── test_schemas.py
    ├── test_taxonomy.py
    ├── test_validator.py
    ├── test_evaluator_matching.py
    ├── test_evaluator_metrics.py
    └── test_cli.py
```

---

## Implementation Steps

### Step 1: Project Scaffold

**Files:** `pyproject.toml`, `.python-version`, all `__init__.py` files

**pyproject.toml:**
- Build backend: `hatchling`
- Python: `>=3.10`
- Dependencies: `pydantic>=2.0`, `numpy`, `scipy`, `click`, `Pillow`
- Dev dependencies: `pytest`, `pytest-cov`
- Entry point: `[project.scripts] isaacsim-bench = "isaacsim_bench.cli.main:cli"`
- src layout: `[tool.hatch.build.targets.wheel] packages = ["src/isaacsim_bench"]`

**Verify:** `uv sync && uv run python -c "import isaacsim_bench"`

---

### Step 2: Pydantic Schemas

**Files:** All files under `src/isaacsim_bench/schemas/`

This is the data contract everything else depends on. Define models that mirror the proposal's JSON formats exactly.

**taxonomy.py:**
- `Variant`: `variant_id: str`, `name: str`
- `Category`: `category_id: str`, `name: str`, `family: str`, `variants: list[Variant]`
- `AssetTaxonomy`: `version: str`, `categories: list[Category]`

**pools.py:**
- `PoolDefinition`: `version: str`, `asset_ids: list[str]`

**scene.py** (most critical file):
- `CameraParams`: `position: list[float]`, `target: list[float]`, `fov_deg: float`
- `ComponentEntry`: `name: str`, `asset_id: str`, `asset_name: str`, `family: str`, `evaluation_role: Literal["primary", "distractor"]`, `match_regime: Literal["exact_match", "proxy_match", "unknown"]`, `translate: list[float]`, `rotate_z_deg: float`, optional `acceptable_proxy_family: str | None`
- `RelationEntry`: `type: str`, `from_node: str` (alias `"from"`), `to_node: str` (alias `"to"`), `from_anchor: str`, `to_anchor: str` — use `model_config = ConfigDict(populate_by_name=True)` for JSON `"from"`/`"to"` keys
- `SceneJSON`: `sample_id: str`, `benchmark_tier: Literal["closed_world", "coverage_mismatch"]`, `family: str`, `template_id: str`, `root_node: str`, `template_params: dict`, `camera: CameraParams`, `components: list[ComponentEntry]`, `relations: list[RelationEntry]`, version metadata fields
- Validator: `root_node` must reference a component name

**prediction.py:**
- `PredictedComponent`: `name: str`, `asset_id: str`, `family: str`, `translate: list[float]`, `rotate_z_deg: float`, `confidence: float`
- `PredictedRelation`: `type: str`, `from_node: str`, `to_node: str`, `from_anchor: str`, `to_anchor: str`
- `PredictionJSON`: `sample_id: str`, `predicted_family: str`, `predicted_template: str`, `components: list[PredictedComponent]`, `relations: list[PredictedRelation]`, `abstained: bool`, `latency_seconds: float`

**manifest.py:**
- `BenchmarkManifest`: `benchmark_version: str`, `asset_taxonomy_version: str`, `world_pool_version: str`, `retrieval_pool_version: str`, `generator_version: str`, `created_at: str`, `sample_count: int`

**splits.py:**
- `SplitAssignment`: `sample_id: str`, `split: Literal["train", "val", "test"]`, `holdout_tags: list[str]`
- `SplitConfig`: `assignments: list[SplitAssignment]`

**Verify:** `uv run pytest tests/test_schemas.py` — round-trip JSON serialization, validation rejects bad data, `from`/`to` alias works

---

### Step 3: Taxonomy and Match Regime Logic

**Files:** `src/isaacsim_bench/taxonomy/registry.py`, `match_regime.py`, `data/*.json`

**registry.py:**
- `TaxonomyRegistry`:
  - `__init__(taxonomy: AssetTaxonomy, world_pool: PoolDefinition, retrieval_pool: PoolDefinition)`
  - Builds reverse index: `variant_id -> category_id`
  - Builds retrieval category cache: `category_id -> set[variant_id in retrieval pool]`
  - `get_category(asset_id: str) -> str | None`
  - `is_in_retrieval_pool(asset_id: str) -> bool`
  - `has_proxy_in_retrieval_pool(asset_id: str) -> bool` — same category, different variant in retrieval pool
  - `load(taxonomy_path, world_pool_path, retrieval_pool_path) -> TaxonomyRegistry` classmethod

**match_regime.py:**
- `derive_component_regime(registry, asset_id) -> Literal["exact_match", "proxy_match", "unknown"]`
  - In retrieval pool → `exact_match`
  - Not in pool but same-category variant exists in pool → `proxy_match`
  - Otherwise → `unknown`
- `derive_scene_regime(component_regimes: list[str]) -> Literal["scene_exact", "scene_proxy", "scene_unknown"]`
  - All primary `exact_match` → `scene_exact`
  - No primary `unknown`, at least one `proxy_match` → `scene_proxy`
  - Any primary `unknown` → `scene_unknown`

**data/asset_taxonomy.json:** skeleton with ~5 categories across 3 families:
- `conveyor_straight`, `conveyor_curve` (conveyor family)
- `shelf_unit` (shelf family)
- `pallet_standard`, `pallet_box` (pallet family)
- ~2-4 variants per category (use real Isaac Sim/SimReady asset names where possible)

**data/world_pool.json:** all variants from taxonomy
**data/retrieval_pool.json:** world pool minus ~3-4 withheld variants (creates proxy_match scenarios)

**Verify:** `uv run pytest tests/test_taxonomy.py` — correct regime derivation for exact/proxy/unknown cases, scene-level regime derivation

---

### Step 4: Data Validator

**Files:** `src/isaacsim_bench/validator/checks.py`, `runner.py`

**checks.py** — six independent functions, each returns a `CheckResult(passed: bool, severity: Literal["error", "warning"], message: str)`:

1. `check_transform_sanity(scene: SceneJSON, overlap_warn=0.3, overlap_reject=0.8) -> list[CheckResult]`
   - Simplified AABB overlap using a configurable default extent per component (no USD mesh loading)
   - IoU > warn threshold → warning, > reject threshold → error

2. `check_relation_connectivity(scene: SceneJSON) -> list[CheckResult]`
   - Build undirected graph from relations over primary components
   - BFS to check connectivity (no networkx dep)
   - Disconnected primary subgraph → error

3. `check_camera_validity(scene: SceneJSON, min_coverage=0.8) -> list[CheckResult]`
   - Simple pinhole projection of component positions
   - < min_coverage of components in frustum → error
   - Degenerate camera (position == target) → error

4. `check_asset_existence(scene: SceneJSON, registry: TaxonomyRegistry) -> list[CheckResult]`
   - Every `asset_id` must resolve in the world pool
   - Missing → error

5. `check_match_regime_consistency(scene: SceneJSON, registry: TaxonomyRegistry) -> list[CheckResult]`
   - Re-derive regime from pools + taxonomy for each component
   - Mismatch with stored label → error

6. `check_file_completeness(sample_dir: Path) -> list[CheckResult]`
   - Must contain `rgb.png`, (`depth.npy` or `depth.exr`), `segmentation.png`, `scene.json`
   - Missing file → error

**runner.py:**
- `ValidatorRunner`:
  - `validate_sample(sample_dir: Path, registry: TaxonomyRegistry) -> ValidationReport`
  - `validate_dataset(dataset_dir: Path, registry: TaxonomyRegistry) -> list[ValidationReport]`
- `ValidationReport`: `sample_id: str`, `results: list[CheckResult]`, `passed: bool` (no errors)

**Verify:** `uv run pytest tests/test_validator.py` — each check tested with passing and failing fixtures

---

### Step 5: Evaluator

**Files:** All files under `src/isaacsim_bench/evaluator/`

This is the largest step. Build bottom-up: matching → individual metrics → runner.

**matching.py** (foundation):
- `match_components(gt_components, pred_components, mode: Literal["exact", "family"]) -> MatchResult`
  - Cost matrix: 0 for match, 1 for mismatch
  - `scipy.optimize.linear_sum_assignment` for optimal bipartite matching
  - Returns: `matched_pairs: list[tuple[int, int]]`, `unmatched_gt: list[int]`, `unmatched_pred: list[int]`
- `MatchResult` dataclass

**metrics/retrieval.py:**
- `compute_retrieval_metrics(gt_scenes, pred_scenes, registry) -> dict`
- Recall@1, Recall@5, MRR — computed separately for `exact_match`, `proxy_match` components
- For `exact_match`: correct if predicted asset_id == gt asset_id
- For `proxy_match`: correct if predicted asset is in same category

**metrics/coverage.py:**
- `compute_coverage_metrics(gt_scenes, pred_scenes) -> dict`
- Component-level: abstention P/R/F1 on `unknown` components
- Scene-level: abstention P/R/F1 on `scene_unknown` scenes
- False-accept rate: fraction of `unknown` components given a match
- Proxy acceptance accuracy: fraction of `proxy_match` correctly matched at family level

**metrics/template.py:**
- `compute_template_metrics(gt_scenes, pred_scenes) -> dict`
- Family classification accuracy, template classification accuracy, macro F1

**metrics/component.py:**
- `compute_component_metrics(gt_scenes, pred_scenes, mode) -> dict`
- P/R/F1 over matched primary components, exact count accuracy
- Two modes: `exact` (asset_id match) and `family` (category match)

**metrics/relation.py** (most complex):
- `compute_relation_metrics(gt_scenes, pred_scenes, component_matches) -> dict`
- **Strict mode**: relation is TP only if both endpoints matched in component pass AND relation type + anchors match
- **Oracle mode**: assume perfect component matching, isolate relation prediction quality
- Returns P/R/F1 and anchor-pair accuracy in both modes

**metrics/placement.py:**
- `compute_placement_metrics(gt_scenes, pred_scenes, component_matches) -> dict`
- Express all poses relative to root node before comparison
- Root-relative translation error (meters), rotation error (degrees)
- Mean and median across components and scenes
- Pass rate at thresholds: translation <= 0.10m, rotation <= 10 deg

**metrics/scene_success.py:**
- `compute_scene_success(all_metrics, gt_scenes, pred_scenes) -> dict`
- A scene succeeds if: correct family/template, component F1 >= 0.9, relation F1 >= 0.9, mean translation error <= 0.10m, mean rotation error <= 10 deg, correct abstention on `unknown`
- Returns scene success rate

**runner.py:**
- `EvaluatorRunner`:
  - `evaluate(gt_dir: Path, pred_dir: Path, registry: TaxonomyRegistry) -> EvaluationReport`
  - Loads all scene.json + prediction.json pairs
  - Runs matching → all metric groups → scene success
  - `EvaluationReport`: per-metric-group dicts + scene success + per-sample details

**Verify:** `uv run pytest tests/test_evaluator_matching.py tests/test_evaluator_metrics.py`
- Hand-crafted GT + prediction pairs covering: perfect match, partial match, proxy match, unknown abstention, relation cascading, placement offset

---

### Step 6: CLI

**Files:** `src/isaacsim_bench/cli/main.py`

Click-based CLI with three command groups:

```
isaacsim-bench validate <dataset_dir> --taxonomy data/asset_taxonomy.json --world-pool data/world_pool.json --retrieval-pool data/retrieval_pool.json
isaacsim-bench validate-sample <sample_dir> ...

isaacsim-bench evaluate <gt_dir> <pred_dir> --taxonomy ... --output report.json
isaacsim-bench evaluate-sample <gt_scene.json> <pred.json> ...

isaacsim-bench taxonomy stats --taxonomy data/asset_taxonomy.json
isaacsim-bench taxonomy derive-regimes <scene.json> --taxonomy ... --world-pool ... --retrieval-pool ...
isaacsim-bench taxonomy validate --taxonomy ...
```

**Verify:** `uv run isaacsim-bench --help`, `uv run pytest tests/test_cli.py`

---

### Step 7: Generator Interface (Stub)

**Files:** `src/isaacsim_bench/generator/base.py`

- `SceneGenerator(ABC)`:
  - `generate_scene(template_id, params, registry) -> SceneJSON` (abstract)
  - `render_scene(scene, output_dir) -> None` (abstract)
  - `generate_batch(configs, output_dir) -> None` (abstract, default loops over generate + render)
- No Isaac Sim imports — just the interface contract
- Docstrings specify that implementations will use `omni.isaac.core`, Replicator, etc.

**Verify:** Can subclass without errors

---

## Dependency Graph

```
Step 1 (scaffold)
  └─> Step 2 (schemas) ── BLOCKS EVERYTHING ELSE
        ├─> Step 3 (taxonomy)
        │     ├─> Step 4 (validator) ── needs registry for checks 4,5
        │     └─> Step 5 (evaluator) ── needs registry for retrieval metrics
        ├─> Step 5 (evaluator) ── needs schemas for matching
        └─> Step 7 (generator stub) ── needs schemas
Step 6 (CLI) ── needs validator + evaluator
```

Steps 4, 5, 7 can be parallelized after step 3.

---

## Verification: End-to-End Smoke Test

After all steps, create a `tests/fixtures/` directory with:
- 3 hand-written `scene.json` files (one conveyor, one shelf, one pallet)
- 3 matching `prediction.json` files (one perfect, one partial, one with abstention)
- Minimal `rgb.png`, `depth.npy`, `segmentation.png` placeholders

Run:
```bash
uv run isaacsim-bench taxonomy validate --taxonomy data/asset_taxonomy.json
uv run isaacsim-bench validate tests/fixtures/sample_001/
uv run isaacsim-bench evaluate tests/fixtures/ tests/fixtures_pred/ --taxonomy data/asset_taxonomy.json --world-pool data/world_pool.json --retrieval-pool data/retrieval_pool.json --output report.json
```

All commands should complete without errors and produce meaningful output.
