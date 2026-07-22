import argparse
import json
from pathlib import Path

from src.dense_features import validate_dense_dataset, validate_dense_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only validation for E5 dense-feature shards")
    parser.add_argument("path", help="A finalized .tar shard or directory containing manifest.json")
    parser.add_argument("--expected_images", type=int, default=None)
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Fail unless this is a complete non-pilot full extraction",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        result = validate_dense_dataset(path, require_complete=args.require_complete)
    else:
        if args.require_complete:
            parser.error("--require_complete requires a dataset directory with a manifest")
        result = validate_dense_shard(path, args.expected_images)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
