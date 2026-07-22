import argparse
import json
from pathlib import Path

from src.dense_features import validate_dense_dataset, validate_dense_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only validation for E5 dense-feature shards")
    parser.add_argument("path", help="A finalized .tar shard or directory containing manifest.json")
    parser.add_argument("--expected_images", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        result = validate_dense_dataset(path)
    else:
        result = validate_dense_shard(path, args.expected_images)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
