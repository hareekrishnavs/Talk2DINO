#!/usr/bin/env python3
"""Validate a closed E9 spatial bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.e9_spatial_bank import validate_e9_spatial_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--allow_nonproduction", action="store_true")
    parser.add_argument("--verify_source_artifacts", action="store_true")
    args = parser.parse_args()
    result = validate_e9_spatial_bank(
        args.bank,
        require_production=not args.allow_nonproduction,
        verify_source_artifacts=args.verify_source_artifacts,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
