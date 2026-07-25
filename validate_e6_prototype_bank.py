#!/usr/bin/env python3
"""Validate an E6 RGTP prototype bank without loading any backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.e6_prototype_bank import validate_prototype_bank


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an E6 prototype bank.")
    parser.add_argument("bank_path", type=Path)
    parser.add_argument(
        "--allow_pilot",
        "--allow-pilot",
        action="store_true",
        help="Explicitly permit validation of a pilot bank.",
    )
    parser.add_argument(
        "--require_complete",
        "--require-complete",
        action="store_true",
        help="Reject a bank whose metadata.complete is false.",
    )
    parser.add_argument(
        "--allow_dirty_source",
        "--allow-dirty-source",
        action="store_true",
        help="Explicitly permit inspection of a dirty-source development pilot.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Expected E3 projection YAML; verifies its name and SHA256.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Expected E3 checkpoint; verifies both its name and SHA256.",
    )
    parser.add_argument(
        "--source_features",
        "--source-features",
        type=Path,
        default=None,
        help="Expected source archive; verifies metadata.source_feature_sha256.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.bank_path.is_file():
        raise FileNotFoundError(f"E6 prototype bank does not exist: {args.bank_path}")
    bank = torch.load(args.bank_path, map_location="cpu", weights_only=False)
    summary = validate_prototype_bank(
        bank,
        allow_pilot=args.allow_pilot,
        allow_dirty_source=args.allow_dirty_source,
        require_complete=args.require_complete,
        expected_config_path=args.config,
        expected_checkpoint_path=args.checkpoint,
        expected_source_features_path=args.source_features,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
