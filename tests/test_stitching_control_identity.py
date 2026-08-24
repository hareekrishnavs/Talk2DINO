"""CPU-only coverage for src.stitching_control_identity: valid identity
loading, missing/unknown-field rejection, exact-type rejection, altered
variant/Hann/sigmoid/dtype rejection, and parent-identity mismatch
detection. Never initializes CUDA, never loads the model or dataset."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.stitching_control_identity import (  # noqa: E402
    CANONICAL_VARIANT_NAMES,
    StitchingControlIdentityError,
    load_identity,
    validate_static_configuration,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_stitching_control_suite.toml"

pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the stitching-control identity")


def _raw() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _write_toml_manually(path: Path, data: dict) -> None:
    """Minimal, dependency-free TOML writer sufficient for this identity's
    shape (nested tables of str/int/float/bool/list-of-str/list-of-int).
    Used only because tomli_w may not be installed in this environment;
    never used to duplicate scientific values -- only to re-serialize a
    mutated copy of the real identity dict for negative-path tests."""
    lines: list[str] = []

    def fmt_value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(fmt_value(x) for x in v) + "]"
        raise TypeError(f"unsupported TOML value type: {type(v)}")

    def write_table(prefix, table):
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subtables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {fmt_value(v)}")
        lines.append("")
        for k, v in subtables.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            write_table(new_prefix, v)

    write_table("", data)
    path.write_text("\n".join(lines))


def _mutated_load(tmp_path, mutate):
    data = _raw()
    mutate(data)
    path = Path(tmp_path) / "identity.toml"
    _write_toml_manually(path, data)
    return load_identity(path, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Valid identity
# ---------------------------------------------------------------------------


def test_valid_identity_loads():
    identity = load_identity(repo_root=ROOT)
    assert identity["identity"]["name"] == "e12-stitching-control-suite"
    assert tuple(identity["variants"]["names"]) == CANONICAL_VARIANT_NAMES


def test_static_configuration_validates_full_parent_chain():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["identity_name"] == "e12-stitching-control-suite"
    assert result["power_evaluation_identity"] == "e12-k11-k12-power-evaluation"
    assert result["matched_identity"] == "e12-matched-k11-k12-t320"
    assert result["pilot20_image_count"] == 20
    assert result["pilot100_image_count"] == 100
    assert result["full5000_image_count"] == 5000


# ---------------------------------------------------------------------------
# Missing/unknown fields
# ---------------------------------------------------------------------------


def test_missing_top_level_section_rejected(tmp_path):
    def mutate(d):
        del d["hann"]
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_unknown_top_level_field_rejected(tmp_path):
    def mutate(d):
        d["unexpected_section"] = {"x": 1}
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_unknown_field_in_known_section_rejected(tmp_path):
    def mutate(d):
        d["propagation"]["extra_field"] = "surprise"
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_missing_field_in_known_section_rejected(tmp_path):
    def mutate(d):
        del d["propagation"]["alpha"]
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Exact-type rejection
# ---------------------------------------------------------------------------


def test_alpha_as_int_rejected(tmp_path):
    def mutate(d):
        d["propagation"]["alpha"] = 1  # int, not exact float
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_k_as_bool_rejected(tmp_path):
    def mutate(d):
        d["propagation"]["k"] = True
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_align_corners_as_int_rejected(tmp_path):
    def mutate(d):
        d["geometry"]["align_corners"] = 1
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_class_count_as_string_rejected(tmp_path):
    def mutate(d):
        d["dataset"]["classes"] = "171"
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Altered variants
# ---------------------------------------------------------------------------


def test_extra_variant_name_rejected(tmp_path):
    def mutate(d):
        d["variants"]["names"] = list(CANONICAL_VARIANT_NAMES) + ["majority_vote"]
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_reordered_variants_rejected(tmp_path):
    def mutate(d):
        d["variants"]["order"] = list(reversed(CANONICAL_VARIANT_NAMES))
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_wrong_uniform_anchor_rejected(tmp_path):
    def mutate(d):
        d["variants"]["uniform_identity_anchor"] = "hann_probability"
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_unsupported_alternative_variant_name_rejected(tmp_path):
    def mutate(d):
        d["variants"]["names"][0] = "uniform_probability_v2"
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Altered Hann formula
# ---------------------------------------------------------------------------


def test_altered_hann_formula_string_rejected(tmp_path):
    def mutate(d):
        d["hann"]["formula"] = "h_N(x) = 1.0"
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_nonzero_hann_epsilon_rejected(tmp_path):
    def mutate(d):
        d["hann"]["epsilon"] = 1e-6
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_hann_not_pixel_centred_rejected(tmp_path):
    def mutate(d):
        d["hann"]["pixel_centred"] = False
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Altered sigmoid stage
# ---------------------------------------------------------------------------


def test_uniform_probability_wrong_sigmoid_stage_rejected(tmp_path):
    def mutate(d):
        d["variants"]["uniform_probability"]["sigmoid_stage"] = "after_stitch"
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_uniform_score_wrong_sigmoid_stage_rejected(tmp_path):
    def mutate(d):
        d["variants"]["uniform_score"]["sigmoid_stage"] = "before_interpolation"
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_hann_probability_reproduces_existing_evaluator_flipped_rejected(tmp_path):
    def mutate(d):
        d["variants"]["hann_probability"]["reproduces_existing_evaluator"] = True
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Altered accumulator dtype
# ---------------------------------------------------------------------------


def test_numerator_dtype_float16_rejected(tmp_path):
    def mutate(d):
        d["accumulation"]["numerator_dtype"] = "float16"
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_denominator_dtype_float64_rejected(tmp_path):
    def mutate(d):
        d["accumulation"]["denominator_dtype"] = "float64"
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_accumulator_storage_shared_between_variants_rejected(tmp_path):
    def mutate(d):
        d["accumulation"]["accumulator_storage_shared_between_variants"] = True
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


def test_in_place_source_mutation_true_rejected(tmp_path):
    def mutate(d):
        d["accumulation"]["in_place_source_mutation"] = True
    with pytest.raises(StitchingControlIdentityError):
        _mutated_load(tmp_path, mutate)


# ---------------------------------------------------------------------------
# Parent-identity mismatch
# ---------------------------------------------------------------------------


def test_wrong_power_evaluation_identity_sha256_rejected(tmp_path):
    def mutate(d):
        d["parent_identity"]["power_evaluation_identity_sha256"] = "0" * 64
    identity = _mutated_load(tmp_path, mutate)
    from src.stitching_control_identity import _validate_parent_identity

    with pytest.raises(StitchingControlIdentityError):
        _validate_parent_identity(ROOT, identity)


def test_wrong_matched_identity_sha256_rejected(tmp_path):
    def mutate(d):
        d["parent_identity"]["matched_identity_sha256"] = "1" * 64
    identity = _mutated_load(tmp_path, mutate)
    from src.stitching_control_identity import _validate_parent_identity

    with pytest.raises(StitchingControlIdentityError):
        _validate_parent_identity(ROOT, identity)


def test_wrong_required_ancestor_commit_fails_git_ancestry_check(tmp_path):
    def mutate(d):
        d["identity"]["required_ancestor_commit"] = "f" * 40
    identity = _mutated_load(tmp_path, mutate)
    from src.stitching_control_identity import _check_git_ancestry

    with pytest.raises(StitchingControlIdentityError):
        _check_git_ancestry(ROOT, identity["identity"]["required_ancestor_commit"], label="test")


# ---------------------------------------------------------------------------
# k/alpha/steps/affinity values must come from the identity, never a bare
# Python literal duplicated elsewhere
# ---------------------------------------------------------------------------


def test_propagation_values_are_registered_not_hardcoded():
    identity = load_identity(repo_root=ROOT)
    assert identity["propagation"]["k"] == 12
    assert identity["propagation"]["alpha"] == 0.98
    assert identity["propagation"]["steps"] == 320
    assert identity["propagation"]["affinity_power"] == 3.0
