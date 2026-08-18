"""Deterministic, type-preserving identities for configuration values."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


TYPED_CONFIGURATION_ENCODING = "talk2dino-typed-configuration-v1"


class TypedConfigurationError(ValueError):
    """Raised when a value cannot participate in a typed configuration."""


def clone_configuration(value: Any, *, path: str = "$") -> Any:
    """Recursively copy supported values without retaining mutable aliases."""
    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise TypedConfigurationError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypedConfigurationError(
                    f"{path} contains a non-string mapping key {key!r}"
                )
            result[key] = clone_configuration(item, path=f"{path}.{key}")
        return result
    if type(value) is list:
        return [
            clone_configuration(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is tuple:
        return tuple(
            clone_configuration(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypedConfigurationError(
        f"{path} has unsupported configuration type {type(value).__name__}"
    )


def deep_merge_configuration(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge mappings recursively while owning every mutable result value."""
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        raise TypedConfigurationError("configuration merge inputs must be mappings")
    merged = clone_configuration(base)
    for key, value in override.items():
        if type(key) is not str:
            raise TypedConfigurationError(
                f"configuration merge contains a non-string key {key!r}"
            )
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge_configuration(merged[key], value)
        else:
            merged[key] = clone_configuration(value, path=f"$.{key}")
    return merged


def _length_prefixed(tag: bytes, payload: bytes) -> bytes:
    return tag + str(len(payload)).encode("ascii") + b":" + payload


def typed_canonical_bytes(value: Any, *, path: str = "$") -> bytes:
    """Encode supported values without collapsing any Python scalar/container type.

    Mapping keys are UTF-8 sorted. Integers use exact decimal spelling and floats
    use ``float.hex()``, preserving signed zero and the exact IEEE-754 value.
    """
    if value is None:
        return b"N"
    if type(value) is bool:
        return b"B1" if value else b"B0"
    if type(value) is int:
        return _length_prefixed(b"I", str(value).encode("ascii"))
    if type(value) is float:
        if not math.isfinite(value):
            raise TypedConfigurationError(f"{path} contains a non-finite float")
        return _length_prefixed(b"F", value.hex().encode("ascii"))
    if type(value) is str:
        return _length_prefixed(b"S", value.encode("utf-8"))
    if type(value) is list:
        payload = b"".join(
            _length_prefixed(
                b"E", typed_canonical_bytes(item, path=f"{path}[{index}]")
            )
            for index, item in enumerate(value)
        )
        return _length_prefixed(b"L", payload)
    if type(value) is tuple:
        payload = b"".join(
            _length_prefixed(
                b"E", typed_canonical_bytes(item, path=f"{path}[{index}]")
            )
            for index, item in enumerate(value)
        )
        return _length_prefixed(b"T", payload)
    if isinstance(value, Mapping):
        items: list[tuple[bytes, str, Any]] = []
        for key, item in value.items():
            if type(key) is not str:
                raise TypedConfigurationError(
                    f"{path} contains a non-string mapping key {key!r}"
                )
            items.append((key.encode("utf-8"), key, item))
        items.sort(key=lambda entry: entry[0])
        payload = b"".join(
            _length_prefixed(b"K", encoded_key)
            + _length_prefixed(
                b"V", typed_canonical_bytes(item, path=f"{path}.{key}")
            )
            for encoded_key, key, item in items
        )
        return _length_prefixed(b"M", payload)
    raise TypedConfigurationError(
        f"{path} has unsupported configuration type {type(value).__name__}"
    )


def typed_configuration_sha256(value: Any) -> str:
    payload = (
        TYPED_CONFIGURATION_ENCODING.encode("ascii")
        + b"\0"
        + typed_canonical_bytes(value)
    )
    return hashlib.sha256(payload).hexdigest()


def first_typed_difference(expected: Any, observed: Any, *, path: str = "$") -> str | None:
    """Return the first deterministic structural/type difference path."""
    if type(expected) is not type(observed):
        return (
            f"{path}: expected type {type(expected).__name__}, "
            f"observed {type(observed).__name__}"
        )
    if isinstance(expected, Mapping):
        expected_keys = sorted(expected, key=lambda key: str(key).encode("utf-8"))
        observed_keys = sorted(observed, key=lambda key: str(key).encode("utf-8"))
        if expected_keys != observed_keys:
            missing = [key for key in expected_keys if key not in observed]
            unknown = [key for key in observed_keys if key not in expected]
            return f"{path}: missing keys={missing!r}, unknown keys={unknown!r}"
        for key in expected_keys:
            difference = first_typed_difference(
                expected[key], observed[key], path=f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if type(expected) in (list, tuple):
        if len(expected) != len(observed):
            return f"{path}: expected length {len(expected)}, observed {len(observed)}"
        for index, (expected_item, observed_item) in enumerate(zip(expected, observed)):
            difference = first_typed_difference(
                expected_item, observed_item, path=f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if expected != observed or (
        type(expected) is float
        and expected == 0.0
        and math.copysign(1.0, expected) != math.copysign(1.0, observed)
    ):
        return f"{path}: expected {expected!r}, observed {observed!r}"
    return None


def raw_file_identity(path: Path) -> dict[str, str]:
    data = Path(path).read_bytes()
    blob_header = f"blob {len(data)}\0".encode("ascii")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob": hashlib.sha1(blob_header + data).hexdigest(),
    }


__all__ = [
    "TYPED_CONFIGURATION_ENCODING",
    "TypedConfigurationError",
    "clone_configuration",
    "deep_merge_configuration",
    "first_typed_difference",
    "raw_file_identity",
    "typed_canonical_bytes",
    "typed_configuration_sha256",
]
