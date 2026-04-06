# Proposal: Synthetic Benchmark for Isaac Sim Asset Retrieval and Assembly

## Why This Exists

The current project has a clear pipeline:

`image -> retrieval -> template / scene graph proposal -> deterministic layout solve -> USD export`

What it does not yet have is a quantitative benchmark. There is no single natural end-to-end loss for this problem, because the output is structured:

- asset identity
- component count
- topology / relations
- 3D placement
- final rendered scene consistency

Instead of forcing one weak scalar loss, the better approach is to create a synthetic benchmark with exact ground truth and evaluate each stage of the pipeline with task-appropriate metrics.

## Core Thesis

A synthetic dataset built from Isaac Sim / SimReady assets is a good idea if it is used first as a **benchmark**, and only second as a **training set**.

The practical generation path should be:

- compose scenes from Isaac Sim / SimReady assets in Omniverse / Isaac Sim
- render RGB, depth, and segmentation images
- export exact metadata for assets, transforms, and relations

That gives a strong controlled benchmark. The important refinement is that the benchmark should not stop at a pure closed-world setting, because real scenes often differ from what is available in the retrieval library.

The benchmark should answer:

1. Can the system retrieve the correct asset or asset family?
2. Can it infer the correct template or scene type?
3. Can it recover the correct component list and relations?
4. Can it place the components with low geometric error?
5. Does the final assembled scene match the rendered ground truth?
6. Can it correctly abstain or fall back when no suitable library match exists?

## Recommended Benchmark Tiers

The benchmark should have three tiers.

### Tier 1: Closed-World Synthetic

- compose scenes entirely from assets that also exist in the retrieval library
- render them in Omniverse / Isaac Sim
- use this as the best-case upper bound and pipeline sanity check

### Tier 2: Coverage-Mismatch Synthetic

- still compose and render scenes in Omniverse / Isaac Sim
- but separate the **world asset pool** from the **retrieval asset pool**
- this simulates the fact that real scenes often contain near-matches or unsupported objects

This tier should include three match regimes:

- `exact_match`: the rendered asset is in the retrieval pool
- `proxy_match`: the rendered asset is withheld, but a same-family proxy is acceptable
- `unknown`: the rendered object is outside the retrieval pool and should trigger abstention or fallback

#### Proxy Taxonomy

A `proxy_match` requires a concrete definition of "acceptable proxy." The benchmark must define a two-level asset taxonomy before generation begins:

- **Category**: coarse functional class (e.g., `conveyor_straight`, `conveyor_curve`, `shelf_unit`, `pallet`)
- **Variant**: specific asset within that category (e.g., `ConveyorBelt_A08`, `ConveyorBelt_A12`)

A proxy is acceptable if and only if the predicted asset shares the same **category** as the ground-truth asset. This rule is deterministic and does not require human judgment at evaluation time.

The taxonomy must be defined as a machine-readable file (`asset_taxonomy.json`) that maps every asset ID to its category. Any asset not present in the taxonomy is automatically `unknown`. This file is versioned alongside the benchmark.

### Tier 3: Real Holdout

- a small curated set of real photos
- used only for final validation
- not used for early threshold tuning or prompt iteration

#### Real Holdout Annotation Protocol

Each real holdout image must have a manually created `scene.json` following the same schema as synthetic samples. Annotation should include:

- template ID and family
- component list with asset category labels and `primary` vs `distractor` roles
- relations between components
- approximate camera parameters (estimated, not exact)

Real holdout annotations are intentionally weaker than synthetic labels. Therefore, the real holdout should only be used for:

- family / template metrics
- category-level component metrics
- relation / topology metrics
- fallback / abstention metrics

Do **not** use the real holdout for:

- exact-asset retrieval metrics
- anchor-pair accuracy unless anchors are explicitly annotated
- placement metrics that require exact global or root-relative coordinates

Expected annotation effort: roughly 15-20 minutes per image for a trained annotator. Budget accordingly for 50-100 images. Annotations should be reviewed by a second annotator for consistency. Inter-annotator disagreements should be resolved before the holdout is used for evaluation.

## Scope for Phase 1

Keep the first benchmark narrow and aligned with the codebase that already exists.

Target scene families:

- conveyors
- shelf rows
- pallet grids

These map directly onto the current template system:

- `u_conveyor`
- `shelf_row`
- `pallet_grid`

Phase 1 assumptions:

- single RGB input
- static scene
- known Isaac / SimReady asset library
- no articulated robots
- no deformables
- benchmark includes both closed-world and coverage-mismatch synthetic scenes
- a small real holdout is reserved for final validation only

## Dataset Design

Each sample should be generated in Omniverse / Isaac Sim from a scene template plus randomized assets, layout parameters, rendering conditions, and distractors. The renderer should output images, and the generator should export exact scene metadata.

### Automatic Generation Workflow

The benchmark should assume that scene generation is automated, not manual.

Recommended workflow:

- prototype scene composition once in the Omniverse / Isaac Sim UI
- move the finalized logic into a standalone Python generator
- run batch generation headlessly
- save both rendered outputs and custom metadata per scene

In practice, the generation stack should use:

- Omniverse / Isaac Sim for scene composition
- standalone Python for batch control
- Replicator for rendering and standard annotations
- custom JSON export for scene graph labels, root node, and match regime labels

This is important because the benchmark needs outputs beyond standard perception annotations. In addition to RGB, depth, and segmentation, the generator must save structured labels such as:

- template ID
- root node
- component list
- component evaluation role: `primary` or `distractor`
- relations
- `exact_match`, `proxy_match`, or `unknown`

### World Pool vs Retrieval Pool

To model real-world mismatch, define two asset sets:

- **world pool**: assets allowed to appear in generated scenes
- **retrieval pool**: assets the algorithm is allowed to retrieve

This is the cleanest way to simulate missing coverage without giving up exact synthetic labels.

Recommended label regimes:

- `exact_match`: rendered asset exists in the retrieval pool
- `proxy_match`: rendered asset is not in the retrieval pool, but a same-family proxy exists
- `unknown`: rendered asset is not in the retrieval pool and no acceptable proxy should be returned

Match regime labels are derived automatically from `asset_taxonomy.json` and the pool definitions. If the retrieval pool or taxonomy changes, all match regime labels must be regenerated. See **Data Versioning** below.

### Scene Factors to Randomize

- asset variant within family
- asset availability regime: `exact_match`, `proxy_match`, `unknown`
- number of components
- template parameters
- camera pose
- focal length / field of view
- lighting intensity and direction
- floor and wall materials
- clutter objects near the main target
- partial occlusion
- image resolution
- background complexity

### Scene Families

#### 1. Conveyor Scenes

Initial subtypes:

- straight run
- U-shape
- L-shape
- merge / T-junction

Ground-truth labels:

- template ID
- ordered component list
- connection graph
- world transforms
- anchor pairs used for attachment

#### 2. Shelf Scenes

Initial subtypes:

- single shelf
- shelf row
- two parallel rows
- shelf plus nearby pallets / boxes

Ground-truth labels:

- number of bays
- shelf asset IDs
- row grouping
- relative adjacency

#### 3. Pallet Scenes

Initial subtypes:

- line
- rectangular grid
- irregular floor arrangement
- pallet plus box stacks

Ground-truth labels:

- row / column layout
- pallet count
- support relations

## Per-Sample Outputs

Each generated sample should save:

- `rgb.png`
- `depth.npy` or `depth.exr`
- `segmentation.png`
- `scene.json`
- optional multi-view render for debugging

Suggested `scene.json` fields:

```json
{
  "sample_id": "conveyor_u_000123",
  "benchmark_tier": "closed_world",
  "family": "conveyor",
  "template_id": "u_conveyor",
  "root_node": "Straight_Top",
  "template_params": {
    "straight_asset": "ConveyorBelt A08",
    "curve_asset": "ConveyorBelt A14"
  },
  "camera": {
    "position": [0.0, 0.0, 0.0],
    "target": [0.0, 0.0, 0.0],
    "fov_deg": 60.0
  },
  "components": [
    {
      "name": "Straight_Top",
      "asset_id": "isaac:Conveyors/ConveyorBelt_A08",
      "asset_name": "ConveyorBelt A08",
      "family": "conveyor_straight",
      "evaluation_role": "primary",
      "match_regime": "exact_match",
      "translate": [0.0, 0.0, 0.0],
      "rotate_z_deg": 0.0
    }
  ],
  "relations": [
    {
      "type": "attach",
      "from": "Straight_Top",
      "to": "Curve_180",
      "from_anchor": "right_end",
      "to_anchor": "curve_entry"
    }
  ]
}
```

Additional recommended metadata:

- `retrieval_pool_version`
- `world_pool_version`
- `acceptable_proxy_family` for `proxy_match` components
- `fallback_expected: true_or_false` for `unknown` components or scenes
- `taxonomy_version`: version of `asset_taxonomy.json` used to assign match regimes
- `generator_version`: version or git hash of the generation script

### Primary Components vs Distractors

Evaluation must distinguish between:

- **primary** components: objects that define the scene's target structure and are scored in all main metrics
- **distractor** components: clutter or context objects that may be visible but do not define the target assembly

This matters for fallback logic. A scene should not be labeled `unknown` just because a distractor object is unsupported.

Match regimes are stored per component, but the benchmark also derives a **scene-level regime** from the primary components only:

- `scene_exact`: all primary components are `exact_match`
- `scene_proxy`: no primary component is `unknown`, and at least one primary component is `proxy_match`
- `scene_unknown`: at least one primary component is `unknown`

Coverage / fallback metrics should be reported at both levels:

- **component-level**: primary components only
- **scene-level**: using the derived regime above

## Data Validation

Every generated sample must pass automated validation before entering the benchmark. The validation step runs after generation and before any evaluation.

### Required Checks

1. **Transform sanity**: no two components overlap by more than a configurable threshold (e.g., bounding-box IoU > 0.3 flags a warning, > 0.8 is rejected)
2. **Relation graph connectivity**: the relation graph must be connected for scenes that should form a single assembly (conveyors, shelf rows); disconnected components are flagged
3. **Camera validity**: the camera frustum must contain at least 80% of the scene's bounding box; degenerate cameras (behind the scene, inside a mesh) are rejected
4. **Asset existence**: every `asset_id` in `scene.json` must resolve to a valid USD reference in the world pool
5. **Match regime consistency**: match regime labels are re-derived from the pool definitions and `asset_taxonomy.json`; any mismatch with the stored label is flagged
6. **File completeness**: every sample directory must contain `rgb.png`, `depth.npy` or `depth.exr`, `segmentation.png`, and `scene.json`

### Audit Protocol

In addition to automated checks, manually inspect a random 5% sample (minimum 30 scenes) before the benchmark is used for evaluation. Check for:

- visual plausibility of rendered images
- correct correspondence between `scene.json` labels and visible objects
- reasonable camera framing

Log all validation results. Rejected samples should be regenerated, not patched.

## Dataset Splits

Do not use only one random split. Within each synthetic tier, use train, val, and test splits. In addition, keep the real holdout fully separate.

### Split A: Standard IID

- train, val, test sampled from the same generator distribution
- use for early debugging

### Split B: Hard Generalization

Hold out specific slices to test structural understanding vs memorization. The holdout conditions must be defined before generation, not chosen post hoc.

Concrete holdout definitions for Phase 1:

- **Template parameter holdout**: for conveyors, hold out U-shape scenes with > 4 straight segments. For shelves, hold out rows with > 5 bays. For pallets, hold out grids larger than 3x3. These are the high-count parameter ranges.
- **Camera angle holdout**: hold out all camera elevations below 15 degrees (near-ground-level views) and above 75 degrees (near-top-down views). Train only on the 15-75 degree range.
- **Asset variant holdout**: for each category with >= 4 variants, hold out one specific variant (the last alphabetically) from all training scenes. It may appear only in test scenes.
- **Clutter holdout**: hold out scenes with > 3 distractor objects. Train scenes have 0-3 distractors; test scenes may have 4+.

Each holdout condition produces a separate test subset. Report metrics on each subset independently.

### Split C: Synthetic-to-Real

Keep a small real-image benchmark completely separate.

Use it only for final evaluation of transfer:

- real conveyor photos
- real shelf photos
- real pallet photos

This split matters because synthetic performance alone does not prove real-world usefulness.

## Metrics

The benchmark should report metrics at multiple levels.

### 1. Retrieval Metrics

Evaluate the candidate generation stage separately by match regime.

- exact asset `Recall@1` on `exact_match` (synthetic only)
- exact asset `Recall@5` on `exact_match` (synthetic only)
- family `Recall@1` on `proxy_match`
- family `Recall@5` on `proxy_match`
- mean reciprocal rank

For ambiguous scenes, family retrieval is often more important than exact asset retrieval. On the real holdout, report category / family retrieval only unless exact asset labels are manually available.

### 2. Coverage / Fallback Metrics

Evaluate whether the system handles unsupported or partial-coverage scenes correctly.

- component-level unknown abstention precision
- component-level unknown abstention recall
- component-level unknown abstention F1
- scene-level unknown abstention precision
- scene-level unknown abstention recall
- scene-level unknown abstention F1
- false-accept rate on `unknown`
- proxy acceptance accuracy on `proxy_match`
- confidence calibration for accept vs abstain

By default, compute these metrics over **primary components**. Distractors may be reported separately, but they should not dominate the main fallback score.

### 3. Routing / Template Metrics

Evaluate whether the system picks the right scene family or template.

- family classification accuracy
- template classification accuracy
- macro F1 across template classes

### 4. Component Metrics

Evaluate predicted component inventory.

- component precision
- component recall
- component F1
- exact count accuracy

Matching rule:

- exact asset match
- or family-level proxy match for a relaxed score on `proxy_match`

Primary results should be reported over primary components only. Optionally report a second set of metrics including distractors for clutter sensitivity analysis.

### 5. Relation / Topology Metrics

Evaluate scene graph correctness.

- relation precision
- relation recall
- relation F1
- anchor-pair accuracy for `attach` (synthetic only, unless anchors are manually annotated on the real holdout)

#### Cascaded Scoring

Relation metrics must account for upstream component matching errors. The evaluation uses a two-pass approach:

1. **Component matching pass**: first, establish a bipartite matching between predicted and ground-truth components (using exact asset match or category-level match as appropriate for the tier). Unmatched predicted components are false positives; unmatched ground-truth components are false negatives.
2. **Relation scoring pass**: a predicted relation is a true positive only if (a) both its source and target nodes were successfully matched to ground-truth components in pass 1, and (b) the relation type and anchors match the corresponding ground-truth relation.

This prevents a single retrieval error from cascading into multiple false-negative relations and inflating the relation error count.

Report relation metrics in two modes:

- **strict**: using the cascaded scoring above
- **oracle-components**: relation metrics computed assuming perfect component matching (i.e., using ground-truth component assignments), to isolate relation prediction quality from retrieval quality

### 6. Placement Metrics

Evaluate geometric quality after solving in a **root-relative canonical frame**, not in global coordinates. This section is **synthetic-only for Phase 1**.

- root-relative translation error in meters
- root-relative rotation error in degrees
- mean anchor gap
- overlap rate

Placement evaluation procedure:

1. Read `root_node` from `scene.json`.
2. Express every component pose relative to the root node pose.
3. Compare predicted vs ground-truth relative transforms for matched primary components.
4. Report mean and median errors across components and scenes.

This avoids penalizing scenes that are structurally correct but globally shifted or rotated. Global registration is explicitly out of scope for Phase 1.

Suggested pass thresholds:

- translation error <= `0.10 m`
- rotation error <= `10 deg`
- anchor gap <= `0.05 m`

These thresholds apply to root-relative errors, not absolute world-frame errors.

### 7. Render Consistency Metrics (Phase 1.5)

Evaluate final predicted scene against ground-truth rendering. These metrics require re-rendering the predicted scene from the exact ground-truth camera, which depends on a working USD export + Omniverse render pipeline in the evaluation loop. **Defer to Phase 1.5** — do not block the initial benchmark on render-loop stability.

- silhouette IoU
- depth MAE
- mask IoU per component class
- image similarity on a rendered overlay

**Phase 1 substitute**: use placement metrics (section 6) as the geometric quality proxy. Add render consistency once the USD export pipeline is stable and headless re-rendering is reliable.

For Phase 1.5, silhouette IoU plus depth MAE is the minimum.

### 8. End-to-End Scene Success

Define a strict scene success rate:

A scene counts as success only if all are true:

- correct family or template
- primary-component F1 >= `0.9`
- relation F1 >= `0.9`
- mean root-relative translation error <= `0.10 m`
- mean root-relative rotation error <= `10 deg`
- abstains correctly when the scene is labeled `unknown`

This gives you a single headline number without losing diagnostic detail.

## Benchmark Protocol

### Phase 1 Protocol: Synthetic Tiers

For each test image:

1. Run candidate retrieval.
2. Decide whether each primary component should be matched, proxied, or rejected, and derive the scene-level regime from the primary components.
3. Record retrieval and fallback metrics.
4. Run proposal / scene graph stage.
5. Compare predicted components and relations to ground truth.
6. Run layout solver.
7. Compare predicted transforms to ground truth in the root-relative frame.

This protocol aligns with the current architecture and makes failure sources visible without requiring a stable re-render loop.

### Phase 1 Protocol: Real Holdout

For each real holdout image:

1. Run candidate retrieval.
2. Record category / family retrieval metrics.
3. Record fallback / abstention metrics.
4. Run proposal / scene graph stage.
5. Compare predicted components and relations to the manually annotated holdout labels.

Do not compute exact-asset metrics, anchor metrics, or placement metrics on the real holdout unless those labels have been explicitly added.

### Phase 1.5 Extension

Once the USD export and headless re-render loop is stable:

8. Render predicted scene from the same camera.
9. Compare render outputs.

## Baselines to Compare

The benchmark becomes useful only if it supports ablations.

Minimum baselines, ordered by priority:

| # | Baseline | Priority | Purpose |
|---|----------|----------|---------|
| 1 | Full end-to-end pipeline | P0 | Primary system measurement |
| 2 | Image-only retrieval, closed-world tier | P0 | Retrieval lower bound |
| 3 | Image-only retrieval, coverage-mismatch tier | P0 | Mismatch stress test |
| 4 | Image + text hint retrieval | P0 | Measures value of human input |
| 5 | Abstention / fallback enabled vs disabled | P0 | Validates coverage-mismatch design |
| 6 | Ground-truth components + predicted relations | P1 | Isolates relation prediction quality |
| 7 | Template prior enabled vs disabled | P1 | Measures template conditioning value |
| 8 | Full retrieval pool vs restricted retrieval pool | P1 | Pool size sensitivity |
| 9 | Ground-truth relations + solver only | P2 | Solver-only sanity check |

P0 baselines should be included in every benchmark report. P1 baselines should be included when diagnosing specific subproblems. P2 baselines are useful but can be deferred.

Each baseline report should include wall-clock latency per scene (mean and p95) alongside accuracy metrics. For a retrieval system, knowing that image + text hint is 3x slower but only 2% better is operationally relevant.

This directly answers research questions like:

- does human description help candidate retrieval, and at what latency cost?
- does template conditioning improve scene graph accuracy?
- is the main bottleneck retrieval, topology prediction, or geometry?
- how often does the system force a wrong match when it should abstain?

## Recommended Dataset Scale

Start small enough to finish.

### Phase 1

- `200` conveyor scenes in the closed-world tier
- `200` shelf scenes in the closed-world tier
- `200` pallet scenes in the closed-world tier
- `100` conveyor scenes in the coverage-mismatch tier
- `100` shelf scenes in the coverage-mismatch tier
- `100` pallet scenes in the coverage-mismatch tier
- `50` to `100` real holdout images

Total: about `900` synthetic scenes plus a small real holdout

This is enough to debug metrics and infrastructure.

### Phase 2

- `3,000` to `10,000` scenes
- more clutter
- harder camera variation
- mixed scenes with multiple object families

## Implementation Plan

### Step 1: Asset Taxonomy and Pool Definition

Define:

- `asset_taxonomy.json`: two-level taxonomy (category, variant) for all known assets
- world asset pool
- retrieval asset pool
- rules for `exact_match`, `proxy_match`, and `unknown` derived from taxonomy + pools

The taxonomy must be defined before any scene generation begins, since match regime labels depend on it.

### Step 2: Scene Generator

Build an Omniverse / Isaac Sim scene composer that instantiates templates and emits:

- asset identities
- transforms
- relations
- camera parameters
- root node

Generation should first be validated in the UI, then moved into a standalone headless Python workflow for batch dataset creation.

### Step 3: Renderer

For each generated scene, export:

- RGB
- depth
- segmentation
- metadata JSON

Use Replicator where possible for rendering and standard annotation capture, and write custom metadata export for benchmark-specific labels.

### Step 4: Data Validator

Build an automated validation script that runs the checks described in the **Data Validation** section. This must run on every generated batch before the data enters the benchmark. Rejected samples are logged and regenerated.

### Step 5: Evaluator

Add an evaluation script that consumes:

- prediction JSON
- ground-truth JSON

and computes all benchmark metrics, including cascaded relation scoring in both strict and oracle-components modes. The evaluator should also record per-baseline wall-clock latency.

### Step 6: Baselines

Run the current pipeline on the synthetic test split and report:

- retrieval metrics
- fallback metrics
- topology metrics (strict and oracle-components)
- placement metrics
- scene success rate
- latency per scene (mean and p95)

### Step 7: Render Consistency (Phase 1.5)

Once the USD export and headless re-rendering pipeline is stable, add:

- silhouette IoU
- depth MAE
- render comparison tooling

## Data Versioning

The benchmark must be reproducible across pool and taxonomy changes. The versioning policy is:

### Immutable Benchmark Versions

Each benchmark release is a frozen snapshot:

- `benchmark_version`: monotonically increasing identifier (e.g., `v1`, `v2`)
- `asset_taxonomy_version`: hash or tag of `asset_taxonomy.json` at generation time
- `world_pool_version`: hash or tag of the world pool definition
- `retrieval_pool_version`: hash or tag of the retrieval pool definition
- `generator_version`: git hash of the generation script

These are recorded in every `scene.json` and in a top-level `benchmark_manifest.json`.

### When to Create a New Version

A new benchmark version is required when any of these change:

- an asset is added to or removed from the world or retrieval pool
- the asset taxonomy is modified (categories added, renamed, or merged)
- the generation script logic changes in a way that affects scene composition or labeling
- validation criteria change

Minor rendering changes (e.g., lighting tweaks) do not require a new version, but should be documented.

### Backward Compatibility

Old benchmark versions are never modified in place. If a pool changes, create a new benchmark version and re-derive all match regime labels. Results from different benchmark versions must not be compared directly — always report the benchmark version alongside metrics.

## Risks

### 1. Synthetic Bias

If rendering is too clean, the benchmark will overestimate performance.

Mitigation:

- random textures
- lighting variation
- clutter
- occlusion
- moderate camera noise

### 2. Closed-World Optimism

If the benchmark only renders assets that also exist in the retrieval pool, performance will look better than it really is.

Mitigation:

- separate world and retrieval pools
- include `proxy_match` and `unknown` regimes
- vary render style
- avoid using identical thumbnails as training views
- evaluate both exact asset and family-level retrieval

### 3. Missing Fallback Evaluation

If the benchmark has no unsupported objects, the system may look good simply because it always returns some asset.

Mitigation:

- include `unknown` cases
- report abstention metrics
- allow explicit fallback as a valid output

### 4. Unrealistic Evaluation Target

If the dataset only contains perfect template scenes, the system may look strong but fail on natural factory photos.

Mitigation:

- add distractors
- add non-template clutter
- keep a real-image benchmark

## Expected Research Value

This benchmark would give the project a clear quantitative story:

- not "we optimized a loss"
- but "we measured scene reconstruction quality across retrieval, topology, and geometry"

That is a much stronger framing for this kind of system.

## Recommendation

Proceed with a synthetic benchmark built by composing scenes in Omniverse / Isaac Sim and rendering them, but do not position it as a single closed-world benchmark.

Position it as:

**a controlled evaluation framework for structured scene reconstruction**

not merely as:

**a synthetic training dataset**

and structure it around:

- a closed-world tier
- a coverage-mismatch tier
- a small real holdout

That framing is more defensible and better aligned with the current pipeline.

## Suggested Next Deliverables

1. `asset_taxonomy.json` — two-level taxonomy for all known assets (blocks everything else)
2. World-pool and retrieval-pool definitions derived from the taxonomy
3. Dataset schema, folder format, and `benchmark_manifest.json` template
4. A minimal Omniverse scene generator for conveyors, shelves, and pallets
5. Automated data validation script
6. An evaluator script with cascaded relation scoring and latency tracking
7. A first report on P0 baselines: `image-only` vs `image + human hint`, with latency
8. Real holdout annotation protocol and initial batch of annotated images


