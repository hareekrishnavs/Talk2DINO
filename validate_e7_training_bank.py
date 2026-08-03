#!/usr/bin/env python3
"""Validate an E7 query-target bank and print its compact identity."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.e7_training_bank import validate_e7_training_bank


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--split", choices=("train", "val"))
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source_features", type=Path)
    parser.add_argument("--allow_pilot", action="store_true")
    parser.add_argument("--allow_dirty_source", action="store_true")
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.bank.is_file():
        raise FileNotFoundError(f"E7 training bank does not exist: {args.bank}")
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    summary = validate_e7_training_bank(
        bank,
        allow_pilot=args.allow_pilot,
        allow_dirty_source=args.allow_dirty_source,
        require_complete=not args.allow_incomplete,
        expected_split=args.split,
        expected_config_path=args.model_config,
        expected_checkpoint_path=args.checkpoint,
        expected_source_features_path=args.source_features,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
