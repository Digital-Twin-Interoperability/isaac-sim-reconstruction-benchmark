#!/usr/bin/env python3
"""Reconstruct a scene from rendered images using the multi-agent pipeline.

Usage:
    uv run python scripts/reconstruct_scene.py samples/u_conveyor_default/renders

Outputs prediction.json in the scene directory and optionally evaluates
it against the ground-truth scene.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the src package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Force line-buffered stdout/stderr.  When running under Isaac Sim's
# python.sh with output redirected to a file, Kit's shutdown can hard-exit
# the process before the default block-buffered streams flush — that loses
# our progress prints (and silently masks where execution stopped).
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a scene from images via multi-agent pipeline",
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing rendered scene images (PNG/JPG)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "VLM model name (default: auto-detect).  "
            "Examples: gpt-5.4, gpt-4o, claude-sonnet-4-6"
        ),
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic"],
        help="VLM provider (default: auto-detect from API key)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for prediction.json (default: <scene_dir>/prediction.json)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate prediction against ground-truth scene.json",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Scene ID for the prediction (default: inferred from directory name)",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate predictions from each refinement round",
    )
    parser.add_argument(
        "--agent",
        choices=["pipeline", "agentic"],
        default="pipeline",
        help=(
            "Reconstruction agent: 'pipeline' (fixed perception→retrieval→"
            "spatial→critic chain) or 'agentic' (tool-calling loop with "
            "optional mid-loop rendering).  Default: pipeline."
        ),
    )
    parser.add_argument(
        "--enable-render-tool",
        action="store_true",
        help=(
            "Enable the mid-loop render tool for --agent agentic. Requires "
            "running this script under Isaac Sim's python.sh so SimulationApp "
            "can boot."
        ),
    )
    parser.add_argument(
        "--max-turns", type=int, default=40,
        help="Max tool-calling turns for --agent agentic (default 40).",
    )
    parser.add_argument(
        "--max-renders", type=int, default=10,
        help="Max render tool invocations (default 10).",
    )
    parser.add_argument(
        "--max-tokens-budget", type=int, default=500_000,
        help="Cumulative token budget across the loop (default 500_000).",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    if not image_dir.is_dir():
        print(f"ERROR: {image_dir} is not a directory")
        sys.exit(1)

    # Infer scene directory (parent of renders/)
    scene_dir = image_dir.parent if image_dir.name == "renders" else image_dir

    print("=" * 60)
    print(f"  Scene Reconstruction — agent={args.agent}")
    print("=" * 60)
    print(f"  Images:   {image_dir}")
    print(f"  Provider: {args.provider or 'auto-detect'}")
    print(f"  Model:    {args.model or 'auto-detect'}")

    if args.agent == "pipeline":
        from isaacsim_bench.agents.orchestrator import SceneReconstructor

        reconstructor = SceneReconstructor(
            model=args.model, provider=args.provider,
            save_intermediates=args.save_intermediates,
        )
        prediction = reconstructor.reconstruct(image_dir, scene_id=args.scene_id)
        taxonomy = reconstructor.taxonomy
    else:
        from isaacsim_bench.agents.agentic import AgenticReconstructor

        reconstructor = AgenticReconstructor(
            model=args.model, provider=args.provider,
            max_turns=args.max_turns,
            max_renders=args.max_renders,
            max_total_tokens=args.max_tokens_budget,
            enable_render=args.enable_render_tool,
        )
        prediction, run = reconstructor.reconstruct(
            image_dir, scene_id=args.scene_id,
        )
        taxonomy = reconstructor.taxonomy
        print(
            f"\n  Agentic run: stop={run.stop_reason} "
            f"turns={run.turns_used} renders={run.renders_used} "
            f"submitted={run.submitted}",
        )
        if run.notes:
            print(f"  Driver notes: {run.notes}")
        # Save the tool-call trace alongside the prediction for debugging.
        trace_path = scene_dir / "agentic_trace.json"
        trace_path.write_text(json.dumps(run.tool_call_log, indent=2))
        print(f"  Tool trace saved to {trace_path}")

    # Save prediction JSON + USD.  IMPORTANT for the agentic path: write
    # before any renderer close, because Kit's shutdown can hard-exit the
    # Python process and skip these.
    out_path = Path(args.output) if args.output else scene_dir / "prediction.json"
    out_path.write_text(prediction.model_dump_json(indent=2))
    print(f"\n  Prediction saved to {out_path}")

    usd_path = out_path.with_suffix(".usd")
    from isaacsim_bench.agents.composer import ComposerAgent
    ComposerAgent.export_usd(prediction, usd_path, taxonomy)
    print(f"  USD exported to {usd_path}")

    # Now safe to tear down the renderer (agentic path only — pipeline
    # reconstructor doesn't have one to close).
    if args.agent == "agentic":
        try:
            reconstructor.close()
        except Exception:
            pass

    # Optional evaluation
    gt_path = scene_dir / "scene.json"
    if args.evaluate and gt_path.exists():
        print("\n" + "=" * 60)
        print("  Evaluation vs Ground Truth")
        print("=" * 60)

        from isaacsim_bench.evaluator.runner import EvaluatorRunner
        from isaacsim_bench.schemas.scene import SceneJSON
        from isaacsim_bench.taxonomy.registry import TaxonomyRegistry

        data_dir = Path(__file__).resolve().parent.parent / "data"
        registry = TaxonomyRegistry.load(
            data_dir / "asset_taxonomy.json",
            data_dir / "world_pool.json",
            data_dir / "retrieval_pool.json",
        )

        gt = SceneJSON.model_validate_json(gt_path.read_text())
        runner = EvaluatorRunner()

        # Run exact-match evaluation
        print("\n  --- Exact Match ---")
        report_exact = runner.evaluate([gt], [prediction], registry, match_mode="exact")
        exact_dict = report_exact.to_dict()
        print(json.dumps(exact_dict, indent=2))

        # Run family-match evaluation (same category = match)
        print("\n  --- Family Match ---")
        report_family = runner.evaluate([gt], [prediction], registry, match_mode="family")
        family_dict = report_family.to_dict()
        print(json.dumps(family_dict, indent=2))

        # Save combined report
        combined = {"exact": exact_dict, "family": family_dict}
        report_path = scene_dir / "evaluation_report.json"
        report_path.write_text(json.dumps(combined, indent=2))
        print(f"\n  Report saved to {report_path}")
    elif args.evaluate:
        print(f"\n  WARNING: --evaluate requested but {gt_path} not found")


if __name__ == "__main__":
    main()
