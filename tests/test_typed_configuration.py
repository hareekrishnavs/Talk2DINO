import pytest

from src.typed_configuration import (
    TypedConfigurationError,
    clone_configuration,
    deep_merge_configuration,
    first_typed_difference,
    typed_canonical_bytes,
    typed_configuration_sha256,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (True, 1),
        (1, 1.0),
        ([1, 2], (1, 2)),
        (0.0, -0.0),
        (None, "null"),
        ({"value": 1}, {"value": 1.0}),
    ],
)
def test_typed_encoding_distinguishes_python_types_and_signed_zero(left, right):
    assert typed_canonical_bytes(left) != typed_canonical_bytes(right)
    assert typed_configuration_sha256(left) != typed_configuration_sha256(right)


def test_typed_mapping_order_is_deterministic():
    left = {"z": [1, 2.0], "a": (False, None)}
    right = {"a": (False, None), "z": [1, 2.0]}
    assert typed_canonical_bytes(left) == typed_canonical_bytes(right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_float_is_rejected(value):
    with pytest.raises(TypedConfigurationError, match="non-finite"):
        typed_configuration_sha256({"value": value})


@pytest.mark.parametrize("value", [{1: "wrong"}, {"value": object()}])
def test_unsupported_configuration_values_are_rejected(value):
    with pytest.raises(TypedConfigurationError):
        typed_configuration_sha256(value)


def test_clone_and_merge_do_not_retain_deep_mutable_aliases():
    nested = {"mapping": {"items": [{"value": 1}]}}
    cloned = clone_configuration(nested)
    merged = deep_merge_configuration(nested, {"extra": [2]})
    nested["mapping"]["items"][0]["value"] = 9
    assert cloned["mapping"]["items"][0]["value"] == 1
    assert merged["mapping"]["items"][0]["value"] == 1
    merged["extra"].append(3)
    assert first_typed_difference(cloned, merged) is not None
