#!/usr/bin/env python3
"""Extract ground-truth SceneJSON from a USD scene file.

Usage:
    uv run python scripts/extract_gt.py full_warehouse.usd
    uv run python scripts/extract_gt.py full_warehouse.usd -o data/gt_scene.json
    uv run python scripts/extract_gt.py full_warehouse.usd --no-structural
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from isaacsim_bench.extractor import extract_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ground-truth SceneJSON from a USD file"
    )
    parser.add_argument("usd", help="Path to the USD scene file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: data/gt_<stem>.json)",
    )
    parser.add_argument(
        "--taxonomy",
        default="data/asset_taxonomy.json",
        help="Path to asset_taxonomy.json",
    )
    parser.add_argument(
        "--root",
        default="/Root",
        help="USD prim path of the scene root (default: /Root)",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Override the sample_id (default: USD filename stem)",
    )
    parser.add_argument(
        "--no-structural",
        action="store_true",
        help="Exclude structural elements (floor, wall, ceiling, etc.)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    scene = extract_ground_truth(
        usd_path=args.usd,
        taxonomy_path=args.taxonomy,
        scene_root=args.root,
        sample_id=args.sample_id,
        include_structural=not args.no_structural,
    )

    output = Path(args.output) if args.output else Path(f"data/gt_{scene.sample_id}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(scene.model_dump_json(indent=2, by_alias=True))

    n_primary = sum(1 for c in scene.components if c.evaluation_role == "primary")
    n_distractor = sum(1 for c in scene.components if c.evaluation_role == "distractor")
    print(f"Written {output}")
    print(f"  {len(scene.components)} components ({n_primary} primary, {n_distractor} distractor)")

    # Summary by family
    from collections import Counter

    fam_counts = Counter(c.family for c in scene.components)
    for fam, count in fam_counts.most_common():
        role = "distractor" if fam in {"floor", "ceiling", "wall", "beam", "pillar", "light_fixture", "decal"} else "primary"
        print(f"    {count:4d}  {fam} ({role})")


if __name__ == "__main__":
    main()
