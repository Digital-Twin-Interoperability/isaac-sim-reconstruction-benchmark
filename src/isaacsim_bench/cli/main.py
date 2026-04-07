from __future__ import annotations

import json
from pathlib import Path

import click

from isaacsim_bench.taxonomy.registry import TaxonomyRegistry


def _load_registry(taxonomy: str, world_pool: str, retrieval_pool: str) -> TaxonomyRegistry:
    return TaxonomyRegistry.load(taxonomy, world_pool, retrieval_pool)


@click.group()
def cli():
    """Isaac Sim Benchmark CLI."""
    pass


# ---------------------------------------------------------------------------
# Validate commands
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--taxonomy", required=True, type=click.Path(exists=True))
@click.option("--world-pool", required=True, type=click.Path(exists=True))
@click.option("--retrieval-pool", required=True, type=click.Path(exists=True))
def validate(dataset_dir: str, taxonomy: str, world_pool: str, retrieval_pool: str):
    """Validate all samples in a dataset directory."""
    from isaacsim_bench.validator.runner import ValidatorRunner

    registry = _load_registry(taxonomy, world_pool, retrieval_pool)
    runner = ValidatorRunner()
    reports = runner.validate_dataset(Path(dataset_dir), registry)

    passed = sum(1 for r in reports if r.passed)
    failed = len(reports) - passed
    click.echo(f"Validated {len(reports)} samples: {passed} passed, {failed} failed")

    for report in reports:
        if not report.passed:
            click.echo(f"\n  FAIL: {report.sample_id} ({report.error_count} errors)")
            for result in report.results:
                if not result.passed:
                    click.echo(f"    [{result.severity}] {result.message}")


@cli.command("validate-sample")
@click.argument("sample_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--taxonomy", required=True, type=click.Path(exists=True))
@click.option("--world-pool", required=True, type=click.Path(exists=True))
@click.option("--retrieval-pool", required=True, type=click.Path(exists=True))
def validate_sample(sample_dir: str, taxonomy: str, world_pool: str, retrieval_pool: str):
    """Validate a single sample directory."""
    from isaacsim_bench.validator.runner import ValidatorRunner

    registry = _load_registry(taxonomy, world_pool, retrieval_pool)
    runner = ValidatorRunner()
    report = runner.validate_sample(Path(sample_dir), registry)

    if report.passed:
        click.echo(f"PASS: {report.sample_id}")
    else:
        click.echo(f"FAIL: {report.sample_id} ({report.error_count} errors)")
        for result in report.results:
            if not result.passed:
                click.echo(f"  [{result.severity}] {result.message}")


# ---------------------------------------------------------------------------
# Evaluate commands
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("gt_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("pred_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--taxonomy", required=True, type=click.Path(exists=True))
@click.option("--world-pool", required=True, type=click.Path(exists=True))
@click.option("--retrieval-pool", required=True, type=click.Path(exists=True))
@click.option("--pred-render-dir", type=click.Path(exists=True, file_okay=False), default=None, help="Dir with re-rendered prediction scenes for render-consistency metrics")
@click.option("--output", "-o", type=click.Path(), default=None, help="Save report to JSON")
def evaluate(gt_dir: str, pred_dir: str, taxonomy: str, world_pool: str, retrieval_pool: str, pred_render_dir: str | None, output: str | None):
    """Evaluate predictions against ground truth."""
    from isaacsim_bench.evaluator.runner import EvaluatorRunner

    registry = _load_registry(taxonomy, world_pool, retrieval_pool)
    runner = EvaluatorRunner()
    report = runner.evaluate_from_dirs(
        Path(gt_dir),
        Path(pred_dir),
        registry,
        pred_render_dir=Path(pred_render_dir) if pred_render_dir else None,
    )
    report_dict = report.to_dict()

    click.echo(json.dumps(report_dict, indent=2))

    if output:
        Path(output).write_text(json.dumps(report_dict, indent=2))
        click.echo(f"\nReport saved to {output}")


# ---------------------------------------------------------------------------
# Taxonomy commands
# ---------------------------------------------------------------------------

@cli.group()
def taxonomy():
    """Taxonomy management commands."""
    pass


@taxonomy.command("validate")
@click.option("--taxonomy", required=True, type=click.Path(exists=True))
def taxonomy_validate(taxonomy: str):
    """Validate taxonomy JSON file."""
    from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

    tax = AssetTaxonomy.model_validate_json(Path(taxonomy).read_text())
    total_variants = sum(len(c.variants) for c in tax.categories)
    click.echo(f"Valid taxonomy v{tax.version}: {len(tax.categories)} categories, {total_variants} variants")


@taxonomy.command("stats")
@click.option("--taxonomy", required=True, type=click.Path(exists=True))
def taxonomy_stats(taxonomy: str):
    """Print taxonomy statistics."""
    from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

    tax = AssetTaxonomy.model_validate_json(Path(taxonomy).read_text())
    click.echo(f"Taxonomy v{tax.version}")
    click.echo(f"  Categories: {len(tax.categories)}")

    families: dict[str, int] = {}
    for cat in tax.categories:
        families[cat.family] = families.get(cat.family, 0) + len(cat.variants)
        click.echo(f"  {cat.category_id} ({cat.family}): {len(cat.variants)} variants")

    click.echo(f"\n  Families: {dict(families)}")


# ---------------------------------------------------------------------------
# Generate commands
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--taxonomy", required=True, type=click.Path(exists=True))
@click.option("--world-pool", required=True, type=click.Path(exists=True))
@click.option("--retrieval-pool", required=True, type=click.Path(exists=True))
@click.option("--asset-root", default="", help="Root path prepended to USD asset paths")
@click.option("--resolution", default="1024x1024", help="Render resolution WxH")
def generate(
    config: str,
    output_dir: str,
    taxonomy: str,
    world_pool: str,
    retrieval_pool: str,
    asset_root: str,
    resolution: str,
):
    """Generate and render scenes from a batch config JSON.

    CONFIG is a JSON file containing a list of objects, each with
    'template_id' and optional 'params' keys.

    Requires Isaac Sim runtime (omni.usd, omni.replicator).
    """
    # Lazy import — only fails if Isaac Sim is not available
    try:
        from isaacsim_bench.generator.isaac_sim import IsaacSimSceneGenerator
    except ImportError as exc:
        raise click.ClickException(
            f"Isaac Sim runtime not available: {exc}\n"
            "Run this command from within an Isaac Sim Python environment."
        ) from exc

    registry = _load_registry(taxonomy, world_pool, retrieval_pool)

    configs = json.loads(Path(config).read_text())
    if not isinstance(configs, list):
        raise click.ClickException("Config JSON must be a list of scene configs")

    w, h = (int(x) for x in resolution.split("x"))

    generator = IsaacSimSceneGenerator(asset_root=asset_root, resolution=(w, h))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = generator.generate_batch(configs, out, registry)
    click.echo(f"Generated {len(results)} scenes in {out}")
    for scene, artifact in results:
        click.echo(f"  {scene.sample_id}: {artifact.rgb_path}")


@taxonomy.command("derive-regimes")
@click.argument("scene_json", type=click.Path(exists=True))
@click.option("--taxonomy", required=True, type=click.Path(exists=True))
@click.option("--world-pool", required=True, type=click.Path(exists=True))
@click.option("--retrieval-pool", required=True, type=click.Path(exists=True))
def taxonomy_derive_regimes(scene_json: str, taxonomy: str, world_pool: str, retrieval_pool: str):
    """Derive match regimes for components in a scene.json."""
    from isaacsim_bench.schemas.scene import SceneJSON
    from isaacsim_bench.taxonomy.match_regime import (
        derive_component_regime,
        derive_scene_regime,
    )

    registry = _load_registry(taxonomy, world_pool, retrieval_pool)
    scene = SceneJSON.model_validate_json(Path(scene_json).read_text())

    primary_regimes = []
    for comp in scene.components:
        regime = derive_component_regime(registry, comp.asset_id)
        marker = " *MISMATCH*" if regime != comp.match_regime else ""
        click.echo(f"  {comp.name} ({comp.asset_id}): {regime}{marker}")
        if comp.evaluation_role == "primary":
            primary_regimes.append(regime)

    scene_regime = derive_scene_regime(primary_regimes)
    click.echo(f"\n  Scene regime: {scene_regime}")
