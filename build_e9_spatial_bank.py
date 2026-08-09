#!/usr/bin/env python3
"""Build an E9 spatial bank from a documented streaming dense source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.e9_spatial_bank import build_e9_spatial_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument(
        "--expected_dino_source_commit",
        default=None,
        help=(
            "optional; if omitted, the value from the source manifest's "
            "source_commit is trusted directly instead of being "
            "cross-checked against an independently supplied expectation"
        ),
    )
    parser.add_argument(
        "--expected_dino_checkpoint_sha256",
        default=None,
        help=(
            "optional; if omitted, the value from the source manifest's "
            "extraction_config.backbone_weights_sha256 is trusted directly "
            "instead of being cross-checked against an independently "
            "supplied expectation"
        ),
    )
    parser.add_argument("--shard_rows", type=int, default=128)
    parser.add_argument("--max_images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_dirty_source", action="store_true")
    args = parser.parse_args()
    result = build_e9_spatial_bank(
        args.source,
        args.output,
        split=args.split,
        expected_dino_source_commit=args.expected_dino_source_commit,
        expected_dino_checkpoint_sha256=args.expected_dino_checkpoint_sha256,
        shard_rows=args.shard_rows,
        max_images=args.max_images,
        overwrite=args.overwrite,
        allow_dirty_source=args.allow_dirty_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
