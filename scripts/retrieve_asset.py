#!/usr/bin/env python3
"""Retrieve the most similar asset from the catalog for a query image.

Usage:
    uv run python scripts/retrieve_asset.py path/to/image.png
    uv run python scripts/retrieve_asset.py path/to/image.png --top-k 10
    uv run python scripts/retrieve_asset.py path/to/image.png --taxonomy data/asset_taxonomy.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from isaacsim_bench.retrieval import AssetRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP-based asset retrieval")
    parser.add_argument("image", help="Path to a query image")
    parser.add_argument(
        "--taxonomy",
        default="data/asset_taxonomy.json",
        help="Path to asset_taxonomy.json",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--model", default="ViT-B-32", help="OpenCLIP model name"
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b79k",
        help="Pretrained weights tag",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    retriever = AssetRetriever(
        taxonomy_path=args.taxonomy,
        model_name=args.model,
        pretrained=args.pretrained,
    )

    results = retriever.query_image(args.image, top_k=args.top_k)

    print(f"\nTop {args.top_k} matches for: {args.image}\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.category_name} ({r.family})")
        print(f"     category_id: {r.category_id}")
        print(f"     variants: {len(r.variant_ids)} — {r.variant_ids[:3]}{'...' if len(r.variant_ids) > 3 else ''}")
        print()


if __name__ == "__main__":
    main()
