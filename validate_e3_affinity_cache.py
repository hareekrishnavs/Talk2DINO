#!/usr/bin/env python3
"""Read-only validation for an E3 affinity-oracle cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.e3_affinity_oracle import cache_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    print(json.dumps(cache_summary(args.cache), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
