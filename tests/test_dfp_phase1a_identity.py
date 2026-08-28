"""Static/mutation tests for the DFP Phase 1A protocol identity. CPU-only,
no torch/CUDA. Tests load the real committed identity and mutate one
field at a time -- never duplicate the entire scientific contract as an
independent fixture."""

from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1a_identity as dfp_module  # noqa: E402
from src.dfp_phase1a_identity import (  # noqa: E402
    DfpPhase1aIdentityError,
    load_identity,
    validate_bridge_checkpoint_binding,
    validate_static_configuration,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml"


def _load_raw() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value)}")


def _dump_toml(document: dict) -> str:
    lines: list[str] = [f'format_version = {_fmt(document["format_version"])}']
    for section, body in document.items():
        if section == "format_version":
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in body.items():
            lines.append(f"{key} = {_fmt(value)}")
    return "\n".join(lines) + "\n"


def _write_and_load(tmp_path: Path, document: dict):
    path = tmp_path / "identity.toml"
    path.write_text(_dump_toml(document), encoding="utf-8")
    return load_identity(path, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_real_identity_loads():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    assert identity["identity"]["name"] == "e13-dfp-phase1a-protocol"


def test_static_configuration_validates_against_real_repo():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["coco_stuff_images"] == 5000
    assert result["ade20k_images"] == 2000


def test_bridge_checkpoint_binding_accepts_real_checkpoint():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    observed = validate_bridge_checkpoint_binding(ROOT, identity)
    assert observed == identity["bridge_checkpoint"]["sha256"]


def test_bridge_checkpoint_binding_rejects_wrong_hash():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    tampered = copy.deepcopy(identity)
    tampered["bridge_checkpoint"]["sha256"] = "f" * 64
    with pytest.raises(DfpPhase1aIdentityError, match="SHA256 mismatch"):
        validate_bridge_checkpoint_binding(ROOT, tampered)


def test_no_private_path_in_identity_file():
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    for fragment in ("/scratch/", "/project/", "/home/"):
        assert fragment not in text


# ---------------------------------------------------------------------------
# Parent hash / path / schema mutations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "matched_k11_k12_identity_sha256", "e3_identity_sha256", "rwr_identity_sha256",
        "ade20k_source_identity_sha256", "ade20k_evaluator_identity_sha256",
    ],
)
def test_wrong_parent_identity_hash_rejected(tmp_path, field):
    document = _load_raw()
    document["parent_identities"][field] = "f" * 64
    identity = _write_and_load(tmp_path, document)
    with pytest.raises(DfpPhase1aIdentityError, match="SHA256 mismatch"):
        validate_static_configuration(repo_root=ROOT, identity_path=tmp_path / "identity.toml", check_git=False)


def test_unsafe_absolute_parent_path_rejected(tmp_path):
    document = _load_raw()
    document["parent_identities"]["matched_k11_k12_identity_path"] = "/etc/passwd"
    with pytest.raises(DfpPhase1aIdentityError, match="safe repository-relative path"):
        _write_and_load(tmp_path, document)


def test_unsafe_traversal_parent_path_rejected(tmp_path):
    document = _load_raw()
    document["parent_identities"]["matched_k11_k12_identity_path"] = "../../etc/passwd"
    with pytest.raises(DfpPhase1aIdentityError, match="safe repository-relative path"):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Scientific field mutations: graph/propagation/affinity/tie-policy drift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("graph", "maximum_rank", 11),
        ("graph", "affinity_function", "cosine_only"),
        ("graph", "affinity_power", 2.0),
        ("graph", "canonical_k_variant", 11),
        ("graph", "graph_symmetrization", True),
        ("propagation", "alpha", 0.9),
        ("propagation", "steps", 160),
        ("propagation", "early_stopping", True),
        ("propagation", "fallback_solver", "cgls"),
        ("propagation", "primary_sensitivity_horizon", 16),
        ("score_interpolation_stitching", "crop", [512, 512]),
        ("score_interpolation_stitching", "stride", [256, 256]),
        ("score_interpolation_stitching", "align_corners", False),
        ("score_interpolation_stitching", "averaging", "hann"),
        ("score_interpolation_stitching", "hann", True),
        ("dispute_protocol", "tie_rule", "highest_index_wins"),
        ("dispute_protocol", "random_tie_break_allowed", True),
        ("dispute_categories", "third_label_and_ignore_excluded_from_negative_class", False),
        ("metric_contract", "reported_metric_unit", "fraction_0_1"),
        ("metric_contract", "difference_unit", "raw_ratio"),
        ("boundary_iou", "implementation_source_status", "bound"),
        ("gate_contract", "oracle_unary_gate_threshold_percentage_points", 1.0),
        ("gate_contract", "recover_gate_threshold_percentage_points", 0.5),
        ("run_modes", "coco_stuff_500_required_unique_count", 400),
        ("run_modes", "ade20k_500_required_unique_count", 600),
        ("structural_components", "primary_connectivity", 4),
        ("structural_components", "tie_rule", "highest_class_index"),
        ("structural_component_area", "primary_microcomponent_threshold_patch_equivalents", 2.0),
        ("numerical_validation_policy", "tolerance_invented_in_this_identity", True),
        ("numerical_validation_policy", "conservation_summary_accumulation_dtype", "float32"),
        ("subset_selection_policy", "required_unique_count", 250),
        ("subset_selection_policy", "concrete_manifest_ids_bound_in_this_identity", True),
    ],
)
def test_scientific_field_mutation_rejected(tmp_path, section, key, value):
    document = _load_raw()
    document[section][key] = value
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Horizons: reordered, missing, duplicated, wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "horizons",
    [
        [320, 64, 32, 16, 0],      # reordered
        [0, 16, 32, 320],          # missing (64)
        [0, 16, 16, 32, 64, 320],  # duplicated
        [0, 8, 32, 64, 320],       # wrong value
        [0, 16, 32, 64],           # missing final horizon
    ],
)
def test_horizons_mutation_rejected(tmp_path, horizons):
    document = _load_raw()
    document["propagation"]["horizons"] = horizons
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_horizon_persistence_disagreeing_with_propagation_rejected(tmp_path):
    document = _load_raw()
    document["horizon_persistence"]["primary_horizon"] = 16
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Oracle-definition drift
# ---------------------------------------------------------------------------


def test_oracle_definition_missing_field_rejected(tmp_path):
    document = _load_raw()
    del document["oracle_definitions"]["g_recover_definition"]
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_oracle_definition_whitespace_only_rejected(tmp_path):
    document = _load_raw()
    document["oracle_definitions"]["g_recover_definition"] = "   "
    with pytest.raises(DfpPhase1aIdentityError, match="whitespace"):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Gates passing on different datasets incorrectly represented as a joint
# pass -- this is a RUNTIME/analysis-level concern (Phase 1A execution),
# not a schema field this static identity can represent; the identity
# instead freezes gate_relation as an explicit, non-empty string requiring
# same-dataset evaluation, and cross_dataset_pass_requires_independent_
# evaluation=true so a future analysis script cannot silently treat two
# different datasets' passes as a joint claim.
# ---------------------------------------------------------------------------


def test_cross_dataset_independent_evaluation_flag_cannot_be_disabled(tmp_path):
    document = _load_raw()
    document["gate_contract"]["cross_dataset_pass_requires_independent_evaluation"] = False
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Wrong counts / run modes
# ---------------------------------------------------------------------------


def test_run_mode_names_wrong_order_rejected(tmp_path):
    document = _load_raw()
    document["run_modes"]["run_mode_names"] = ["ade20k_500", "coco_stuff_500"]
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_run_mode_names_extra_entry_rejected(tmp_path):
    document = _load_raw()
    document["run_modes"]["run_mode_names"] = ["coco_stuff_500", "ade20k_500", "voc2012_500"]
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Structural-component connectivity / microcomponent threshold / sensitivity
# ---------------------------------------------------------------------------


def test_sensitivity_connectivity_wrong_rejected(tmp_path):
    document = _load_raw()
    document["structural_components"]["sensitivity_connectivity"] = 8
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_microcomponent_sensitivity_list_wrong_rejected(tmp_path):
    document = _load_raw()
    document["structural_component_area"]["sensitivity_thresholds_patch_equivalents"] = [1.0, 2.0, 4.0]
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Exact-type adversarial matrix
# ---------------------------------------------------------------------------


def test_int_as_float_for_maximum_rank_rejected(tmp_path):
    document = _load_raw()
    document["graph"]["maximum_rank"] = 12.0
    with pytest.raises(DfpPhase1aIdentityError, match="exact"):
        _write_and_load(tmp_path, document)


def test_bool_as_int_for_directed_rejected(tmp_path):
    document = _load_raw()
    document["graph"]["directed"] = 1
    with pytest.raises(DfpPhase1aIdentityError, match="exact boolean"):
        _write_and_load(tmp_path, document)


def test_numeric_string_for_alpha_rejected(tmp_path):
    document = _load_raw()
    document["propagation"]["alpha"] = "0.98"
    with pytest.raises(DfpPhase1aIdentityError, match="exact"):
        _write_and_load(tmp_path, document)


def test_int_for_float_field_rejected(tmp_path):
    document = _load_raw()
    document["graph"]["affinity_power"] = 3
    with pytest.raises(DfpPhase1aIdentityError, match="exact"):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Duplicate TOML keys / whitespace-only strings / non-finite values
# ---------------------------------------------------------------------------


def test_duplicate_toml_key_rejected(tmp_path):
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    # inject a duplicate key inside [identity]
    injected = text.replace(
        'name = "e13-dfp-phase1a-protocol"',
        'name = "e13-dfp-phase1a-protocol"\nname = "duplicate-name"',
        1,
    )
    path = tmp_path / "identity.toml"
    path.write_text(injected, encoding="utf-8")
    with pytest.raises(DfpPhase1aIdentityError):
        load_identity(path, repo_root=ROOT)


def test_whitespace_only_description_rejected(tmp_path):
    document = _load_raw()
    document["identity"]["description"] = "   \t  "
    with pytest.raises(DfpPhase1aIdentityError, match="whitespace"):
        _write_and_load(tmp_path, document)


def test_nan_rejected_for_alpha(tmp_path):
    document = _load_raw()
    document["propagation"]["alpha"] = float("nan")
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_infinity_rejected_for_gate_threshold(tmp_path):
    document = _load_raw()
    document["gate_contract"]["oracle_unary_gate_threshold_percentage_points"] = float("inf")
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Schema: unknown/missing fields, unexpected top-level keys
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected(tmp_path):
    document = _load_raw()
    document["unexpected_section"] = {"x": 1}
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_section_rejected(tmp_path):
    document = _load_raw()
    del document["gate_contract"]
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_unknown_field_within_section_rejected(tmp_path):
    document = _load_raw()
    document["graph"]["extra_unexpected_field"] = 1
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_missing_field_within_section_rejected(tmp_path):
    document = _load_raw()
    del document["graph"]["affinity_power"]
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Case sensitivity / exact schema name
# ---------------------------------------------------------------------------


def test_wrong_case_format_version_rejected(tmp_path):
    document = _load_raw()
    document["format_version"] = "Talk2Dino-DFP-Phase1A-Protocol-Identity-V1"
    document["identity"]["schema_version"] = document["format_version"]
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_uppercase_sha256_rejected(tmp_path):
    document = _load_raw()
    document["bridge_checkpoint"]["sha256"] = document["bridge_checkpoint"]["sha256"].upper()
    with pytest.raises(DfpPhase1aIdentityError, match="lowercase"):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------


def test_load_identity_does_not_mutate_source_file():
    before = IDENTITY_PATH.read_bytes()
    load_identity(IDENTITY_PATH, repo_root=ROOT)
    after = IDENTITY_PATH.read_bytes()
    assert before == after


def test_validate_static_configuration_does_not_mutate_source_file():
    before = IDENTITY_PATH.read_bytes()
    validate_static_configuration(repo_root=ROOT, check_git=True)
    after = IDENTITY_PATH.read_bytes()
    assert before == after


# ---------------------------------------------------------------------------
# Canonical scientific-string pinning: coverage guard + confirmed-defect
# regressions + exhaustive mutation matrix.
#
# The "expected" side of the coverage guard is derived from
# `SECTION_KEYS` (the loader's independently declared schema) plus a live
# type-introspection of the real identity's string-typed fields -- never
# from `CANONICAL_SCIENTIFIC_STRINGS` itself, so this cannot be tautological.
# ---------------------------------------------------------------------------


def _set_document_value(document: dict, section: str, field: str, value) -> None:
    if section == "format_version":
        document["format_version"] = value
    else:
        document[section][field] = value


def test_every_scientific_string_field_is_canonically_pinned_or_relationally_governed():
    """Every string-typed field declared in SECTION_KEYS (plus the
    top-level format_version) must be covered by exactly one of:
    CANONICAL_SCIENTIFIC_STRINGS (exact-value pinned) or
    _RELATIONALLY_GOVERNED_STRING_FIELDS (pinned by a stronger independent
    mechanism -- a live SHA-256 file hash, git ancestry, or a hardcoded
    module constant). A field newly added to the TOML/schema without being
    classified into one of these two registries fails this test; a field
    double-classified, or a registry entry pointing at a field that no
    longer exists, also fails it."""
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    governed = dfp_module._RELATIONALLY_GOVERNED_STRING_FIELDS
    pinned = set(dfp_module.CANONICAL_SCIENTIFIC_STRINGS)

    all_string_fields = {("format_version", "format_version")}
    for section, keys in dfp_module.SECTION_KEYS.items():
        block = identity[section]
        for field in keys:
            if type(block[field]) is str:
                all_string_fields.add((section, field))

    unaccounted = all_string_fields - governed - pinned
    assert not unaccounted, f"scientific string fields missing an exact canonical pin: {sorted(unaccounted)}"
    overlap = governed & pinned
    assert not overlap, f"fields double-classified as both relationally governed and canonically pinned: {sorted(overlap)}"
    stale = pinned - all_string_fields
    assert not stale, f"CANONICAL_SCIENTIFIC_STRINGS references fields that do not exist in the schema: {sorted(stale)}"
    stale_governed = governed - all_string_fields
    assert not stale_governed, f"_RELATIONALLY_GOVERNED_STRING_FIELDS references fields that do not exist: {sorted(stale_governed)}"


# ---------------------------------------------------------------------------
# Confirmed-defect regressions: these four mutations were independently
# reproduced as INCORRECTLY ACCEPTED before this repair (nonempty-string
# check only, no exact-value pin). They must now be rejected.
# ---------------------------------------------------------------------------


def test_confirmed_defect_image_global_q_w_definition_rejected(tmp_path):
    document = _load_raw()
    document["query_compute_projection"]["q_w_definition"] = (
        "Q_image = { a(x), b(x) : x is a stitched T=320 dispute in the whole image }"
    )
    with pytest.raises(DfpPhase1aIdentityError, match="q_w_definition disagrees"):
        _write_and_load(tmp_path, document)


def test_confirmed_defect_harmful_beneficial_category_swap_rejected(tmp_path):
    document = _load_raw()
    document["dispute_categories"]["harmful_washout"] = "GT(x) == b(x) and GT(x) != a(x)"
    with pytest.raises(DfpPhase1aIdentityError, match="harmful_washout disagrees"):
        _write_and_load(tmp_path, document)


def test_confirmed_defect_g_recover_sign_reversal_rejected(tmp_path):
    document = _load_raw()
    document["oracle_definitions"]["g_recover_definition"] = "G_recover = metric(P) - metric(O_P)"
    with pytest.raises(DfpPhase1aIdentityError, match="g_recover_definition disagrees"):
        _write_and_load(tmp_path, document)


def test_confirmed_defect_whitespace_padded_q_w_definition_rejected(tmp_path):
    document = _load_raw()
    canonical = document["query_compute_projection"]["q_w_definition"]
    document["query_compute_projection"]["q_w_definition"] = f"  {canonical}  "
    with pytest.raises(DfpPhase1aIdentityError, match="q_w_definition disagrees"):
        _write_and_load(tmp_path, document)


# ---------------------------------------------------------------------------
# Exhaustive mutation matrix over every canonically-pinned scientific
# string field: prefixed space, suffixed space, a one-character mutation,
# a case mutation, and a wholesale substitution with a different nonempty
# scientific-looking sentence. All must be rejected, with no stripping,
# normalization, or case-folding applied before comparison.
# ---------------------------------------------------------------------------


def _mutation_variants(original: str) -> dict[str, str]:
    mid = len(original) // 2
    one_char = (original[:mid] + ("X" if original[mid : mid + 1] != "X" else "Y") + original[mid + 1 :]) if original else "X"
    return {
        "prefix_space": " " + original,
        "suffix_space": original + " ",
        "one_char_mutation": one_char,
        "case_mutation": original.swapcase(),
        "different_sentence": "This is a completely different but nonempty scientific sentence that must never validate.",
    }


_CANONICAL_MUTATION_CASES: list[tuple[str, str, str, str]] = []
for (_section, _field), _canonical in dfp_module.CANONICAL_SCIENTIFIC_STRINGS.items():
    for _kind, _mutated in _mutation_variants(_canonical).items():
        if _mutated != _canonical:
            _CANONICAL_MUTATION_CASES.append((_section, _field, _kind, _mutated))


@pytest.mark.parametrize(
    "section,field,kind,mutated",
    _CANONICAL_MUTATION_CASES,
    ids=[f"{s}.{f}:{k}" for s, f, k, _ in _CANONICAL_MUTATION_CASES],
)
def test_canonical_scientific_string_mutation_rejected(tmp_path, section, field, kind, mutated):
    document = _load_raw()
    _set_document_value(document, section, field, mutated)
    with pytest.raises(DfpPhase1aIdentityError):
        _write_and_load(tmp_path, document)


def test_canonical_string_accepted_value_returned_byte_for_byte_unchanged():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    for (section, field), canonical in dfp_module.CANONICAL_SCIENTIFIC_STRINGS.items():
        actual = identity["format_version"] if section == "format_version" else identity[section][field]
        assert actual == canonical
        assert type(actual) is str
