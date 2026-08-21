"""Tests for the matched k=11 vs k=12 finite-step (T=320) connectivity
identity. Every canonical scientific value (alpha, steps, historical
metrics, window count, tolerances) is obtained from ``load_identity()`` --
this file never hardcodes one of its own."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from src.matched_k11_k12_identity import (
    GRAPH_CONTRACT_KEYS,
    IDENTITY_RELATIVE_PATH,
    MatchedK11K12Error,
    PER_IMAGE_STATISTICS_ARTIFACT_KEYS,
    PER_IMAGE_STATS_SCHEMA_NAME,
    PER_PHASE_RUNTIME_KEYS,
    PROPAGATION_CONTRACT_KEYS,
    PROVENANCE_KEYS,
    RESULT_SCHEMA_NAME,
    RUNTIME_TELEMETRY_KEYS,
    STITCHING_CONTRACT_KEYS,
    SUPPORTED_COMPUTE_DTYPE,
    SUPPORTED_METRIC_UNIT,
    SUPPORTED_OUTPUT_DTYPE,
    TOP_RESULT_KEYS,
    VARIANT_K_VALUES,
    VARIANT_KEYS,
    VARIANT_RECORD_KEYS,
    _validate_relational_equality,
    load_identity,
    parse_structured_result,
    recompute_metrics,
    repository_root,
    validate_historical_provenance,
    validate_per_image_statistics_additive_aggregation,
    validate_per_image_statistics_artifact_metadata,
    validate_per_image_statistics_row,
    validate_static_configuration,
    validate_window_count_reference,
    verify_record,
    verify_result,
)
from src.e3_evaluation_identity import load_identity as _load_e3_identity_direct
from src.rwr_reproduction_identity import load_identity as _load_rwr_identity_direct


ROOT = Path(__file__).parents[1]
IDENTITY_PATH = ROOT / IDENTITY_RELATIVE_PATH


def _identity():
    return load_identity(repo_root=ROOT)


def _identity_sha256() -> str:
    return hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest()


def _json_like(value):
    """Mirror how strict JSON parsing (``parse_float=Decimal``) would have
    represented this TOML-sourced value inside a structured result record."""
    if type(value) is float:
        return Decimal(str(value))
    return value


def _contract_slice(identity, section, keys):
    return {key: _json_like(identity[section][key]) for key in keys}


def build_valid_variant(identity, key, *, aAcc, mIoU, mAcc, intersection, union, predicted, gt, fallback=5):
    return {
        "k": VARIANT_K_VALUES[key],
        "aAcc": Decimal(f"{aAcc:.12f}"),
        "mIoU": Decimal(f"{mIoU:.12f}"),
        "mAcc": Decimal(f"{mAcc:.12f}"),
        "metric_unit": identity["metrics"]["unit"],
        "metric_source": identity["metrics"]["precision_source"],
        "intersection": intersection,
        "union": union,
        "predicted_pixels": predicted,
        "ground_truth_pixels": gt,
        "score_or_label_digest": "a" * 64,
        "completed_image_count": identity["dataset"]["images"],
        "completed_window_count": identity["window_count_reference"]["value"],
        "min_steps": identity["propagation"]["steps"],
        "max_steps": identity["propagation"]["steps"],
        "fallback_row_count": fallback,
        "error_count": 0,
        "non_finite_count": 0,
    }


def build_valid_record(identity, *, k11_bias=1):
    """``k11_bias`` shifts k11's intersection counts relative to k12's;
    0 gives a null delta, positive/negative give a positive/negative delta."""
    class_count = identity["dataset"]["classes"]
    image_count = identity["dataset"]["images"]
    window_count = identity["window_count_reference"]["value"]

    intersection_k12 = [10] * class_count
    union_k12 = [20] * class_count
    predicted_k12 = [15] * class_count
    gt_k12 = [18] * class_count
    intersection_k11 = [10 + k11_bias] * class_count
    union_k11 = [20] * class_count
    predicted_k11 = [15] * class_count
    gt_k11 = list(gt_k12)

    aAcc12, mIoU12, mAcc12 = recompute_metrics(intersection_k12, union_k12, predicted_k12, gt_k12)
    aAcc11, mIoU11, mAcc11 = recompute_metrics(intersection_k11, union_k11, predicted_k11, gt_k11)

    fallback_counts = {"k11": 5, "k12": 5}
    return {
        "schema": RESULT_SCHEMA_NAME,
        "identity": identity["identity"]["name"],
        "identity_sha256": _identity_sha256(),
        "git_commit": "a" * 40,
        "complete": True,
        "final": True,
        "dataset_identity": identity["dataset"]["evaluation_split_identity"],
        "image_count": image_count,
        "unique_image_count": image_count,
        "window_count": window_count,
        "image_order_digest": "b" * 64,
        "score_stage": identity["snapshot"]["raw_score_stage"],
        "graph_contract": _contract_slice(identity, "graph", GRAPH_CONTRACT_KEYS),
        "propagation_contract": _contract_slice(identity, "propagation", PROPAGATION_CONTRACT_KEYS),
        "stitching_contract": _contract_slice(identity, "stitching", STITCHING_CONTRACT_KEYS),
        "variant_results": {
            "k11": build_valid_variant(
                identity, "k11", aAcc=aAcc11, mIoU=mIoU11, mAcc=mAcc11,
                intersection=intersection_k11, union=union_k11, predicted=predicted_k11, gt=gt_k11,
                fallback=fallback_counts["k11"],
            ),
            "k12": build_valid_variant(
                identity, "k12", aAcc=aAcc12, mIoU=mIoU12, mAcc=mAcc12,
                intersection=intersection_k12, union=union_k12, predicted=predicted_k12, gt=gt_k12,
                fallback=fallback_counts["k12"],
            ),
        },
        "paired_delta": Decimal(f"{mIoU11 - mIoU12:.12f}"),
        "runtime_telemetry": {
            "backbone_forward_count": window_count,
            "snapshot_count": window_count,
            "affinity_build_count": window_count,
            "top12_selection_count": window_count,
            "k11_propagation_count": window_count,
            "k12_propagation_count": window_count,
            "tie_row_count": 0,
            "k11_prefix_mismatch_count": 0,
            "fallback_row_counts": fallback_counts,
            "early_termination": False,
            "solver_fallback_used": False,
            "cgls_call_count": 0,
            "second_backbone_pass_count": 0,
            "per_phase_runtime_seconds": {name: Decimal("1.0") for name in PER_PHASE_RUNTIME_KEYS},
            "peak_gpu_memory_bytes": 1000,
            "interrupted_resumed": None,
        },
        "per_image_statistics_artifact": {
            "path": "artifacts/e12_per_image_stats.json",
            "sha256": "c" * 64,
            "image_count": image_count,
            "class_count": class_count,
            "schema": PER_IMAGE_STATS_SCHEMA_NAME,
            "image_id_digest": "d" * 64,
            "additive_aggregation_verified": True,
        },
        "provenance": {
            "source_git_commit": "a" * 40,
            "source_git_branch": "e12-connectivity-analysis",
            "source_git_dirty": False,
            "elapsed_seconds": Decimal("1.0"),
            "gpu_model": "synthetic",
            "torch_version": "2.0",
            "cuda_version": "11.8",
        },
    }


def build_valid_historical_artifacts(identity):
    artifacts = {}
    for section, prefix in (("historical_reference", "source"), ("window_count_reference", "source")):
        commit = identity[section][f"{prefix}_commit"]
        path = identity[section][f"{prefix}_path"]
        artifacts[(commit, path)] = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    return artifacts


def _reader(artifacts):
    def read(commit, path):
        try:
            return artifacts[(commit, path)]
        except KeyError as error:
            raise MatchedK11K12Error("missing injected historical artifact") from error
    return read


# ---------------------------------------------------------------------------
# 1-4: TOML loading, schema, unknown keys, exact-type matrix
# ---------------------------------------------------------------------------


def test_identity_loads():
    identity = _identity()
    assert identity["identity"]["name"] == "e12-matched-k11-k12-t320"
    assert identity["propagation"]["method"] == "finite_power_iteration"


def test_identity_top_level_schema_is_closed(tmp_path):
    raw = IDENTITY_PATH.read_text()
    marker = 'format_version = "talk2dino-matched-k11-k12-identity-v1"\n'
    assert raw.startswith(marker)
    mutated_text = marker + 'extra_top_level_key = "x"\n' + raw[len(marker):]
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(mutated_text)
    with pytest.raises(MatchedK11K12Error, match="unexpected top-level schema"):
        load_identity(mutated, repo_root=ROOT)


@pytest.mark.parametrize(
    "section",
    [
        "identity", "parent_identities", "dataset", "geometry", "snapshot",
        "graph", "propagation", "execution", "stitching", "metrics",
        "historical_reference", "window_count_reference", "result_contract",
        "prohibited",
    ],
)
def test_unknown_key_rejected_in_every_section(tmp_path, section):
    raw = IDENTITY_PATH.read_text()
    # Anchored to a real table header on its own line -- the file's leading
    # comment block also mentions "[prohibited]" in prose, which a bare
    # substring search would match first.
    marker = f"\n[{section}]\n"
    index = raw.index(marker) + len(marker)
    mutated_text = raw[:index] + 'bogus_unknown_key = "x"\n' + raw[index:]
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(mutated_text)
    with pytest.raises(MatchedK11K12Error, match="unexpected schema"):
        load_identity(mutated, repo_root=ROOT)


EXACT_TYPE_MUTATIONS = [
    ('alpha = 0.98', 'alpha = "0.98"', "alpha"),
    ('steps = 320', 'steps = 320.0', "steps"),
    ('affinity_power = 3.0', 'affinity_power = 3', "affinity_power"),
    ('early_stopping = false', 'early_stopping = 0', "early_stopping"),
    ('directed = true', 'directed = 1', "directed"),
    ('images = 5000', 'images = "5000"', "images"),
    ('background_class = false', 'background_class = "false"', "background_class"),
    ('crop = [448, 448]', 'crop = [448.0, 448]', "crop"),
]


@pytest.mark.parametrize("old, new, match", EXACT_TYPE_MUTATIONS)
def test_exact_type_mutation_matrix(tmp_path, old, new, match):
    raw = IDENTITY_PATH.read_text()
    assert old in raw
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw.replace(old, new, 1))
    with pytest.raises(MatchedK11K12Error):
        load_identity(mutated, repo_root=ROOT)


def test_variants_must_be_exactly_11_and_12(tmp_path):
    raw = IDENTITY_PATH.read_text().replace("variants = [11, 12]", "variants = [10, 12]")
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    with pytest.raises(MatchedK11K12Error, match="variants must be exactly"):
        load_identity(mutated, repo_root=ROOT)


def test_prohibited_list_must_match_supported_vocabulary(tmp_path):
    raw = IDENTITY_PATH.read_text().replace('"pamr",', '"pamr", "unexpected_new_practice",')
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    with pytest.raises(MatchedK11K12Error, match="prohibited"):
        load_identity(mutated, repo_root=ROOT)


# ---------------------------------------------------------------------------
# 5: required ancestor validation
# ---------------------------------------------------------------------------


def test_ancestor_check_passes_for_real_repo():
    identity = _identity()
    validate_static_configuration(repo_root=ROOT, check_git=True)
    assert identity["identity"]["required_ancestor_commit"]


def test_ancestor_check_fails_for_non_ancestor_commit(tmp_path):
    raw = IDENTITY_PATH.read_text().replace(
        'required_ancestor_commit = "e03d1f501d039811ad4b9b4505e8e5b5ae2862eb"',
        'required_ancestor_commit = "' + ("0" * 40) + '"',
    )
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    with pytest.raises(MatchedK11K12Error, match="Git ancestry check failed"):
        validate_static_configuration(repo_root=ROOT, identity_path=mutated, check_git=True)


def test_ancestor_check_skipped_when_check_git_false(tmp_path):
    raw = IDENTITY_PATH.read_text().replace(
        'required_ancestor_commit = "e03d1f501d039811ad4b9b4505e8e5b5ae2862eb"',
        'required_ancestor_commit = "' + ("0" * 40) + '"',
    )
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    # Ancestry is skipped, but parent-identity and historical checks still run
    # and must pass against the real, unmodified repository.
    validate_static_configuration(repo_root=ROOT, identity_path=mutated, check_git=False)


# ---------------------------------------------------------------------------
# 6-7: parent identity hashes and relational comparison
# ---------------------------------------------------------------------------


def test_parent_e3_identity_hash_mismatch_rejected(tmp_path):
    raw = IDENTITY_PATH.read_text().replace(
        'e3_identity_sha256 = "db9cb8d407d6b5770d13c2b6546a33008206ae44ebcdc55f6b7384813eec477c"',
        'e3_identity_sha256 = "' + ("0" * 64) + '"',
    )
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    with pytest.raises(MatchedK11K12Error, match="E3 parent identity file SHA256 mismatch"):
        validate_static_configuration(repo_root=ROOT, identity_path=mutated)


def test_parent_rwr_identity_hash_mismatch_rejected(tmp_path):
    raw = IDENTITY_PATH.read_text().replace(
        'rwr_identity_sha256 = "da0eb69de586328b53aeea83a9a48cd089dd53666522e5d1d3d5237742dbd365"',
        'rwr_identity_sha256 = "' + ("0" * 64) + '"',
    )
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    with pytest.raises(MatchedK11K12Error, match="RWR parent identity file SHA256 mismatch"):
        validate_static_configuration(repo_root=ROOT, identity_path=mutated)


def test_parent_identity_name_mismatch_rejected(tmp_path):
    raw = IDENTITY_PATH.read_text().replace(
        'e3_identity_name = "e3-paired-soft-routing"',
        'e3_identity_name = "wrong-name"',
    )
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    with pytest.raises(MatchedK11K12Error, match="E3 parent identity name mismatch"):
        validate_static_configuration(repo_root=ROOT, identity_path=mutated)


def test_relational_crop_mismatch_rejected(tmp_path):
    # Change crop AND patch_grid together so internal geometry consistency
    # (crop == patch_size * patch_grid) still holds and load_identity's own
    # geometry check does not fire first; only the cross-identity relational
    # comparison against E3's crop should reject this.
    raw = IDENTITY_PATH.read_text()
    raw = raw.replace("crop = [448, 448]", "crop = [224, 224]", 1)
    raw = raw.replace("patch_grid = [32, 32]", "patch_grid = [16, 16]", 1)
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    with pytest.raises(MatchedK11K12Error, match="disagrees with parent identity"):
        validate_static_configuration(repo_root=ROOT, identity_path=mutated)


def test_relational_affinity_power_mismatch_rejected(tmp_path):
    raw = IDENTITY_PATH.read_text().replace("affinity_power = 3.0", "affinity_power = 2.0", 1)
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    with pytest.raises(MatchedK11K12Error, match="disagrees with parent identity"):
        validate_static_configuration(repo_root=ROOT, identity_path=mutated)


def test_propagation_method_must_differ_from_canonical_rwr_solver():
    # Structural guarantee: our finite-step method string can never equal
    # the canonical RWR solver method string, since load_identity requires
    # propagation.method == "finite_power_iteration" and the canonical RWR
    # identity's solver.method == "cgls".
    identity = _identity()
    from src.rwr_reproduction_identity import load_identity as load_rwr_identity
    rwr_identity = load_rwr_identity(ROOT / identity["parent_identities"]["rwr_identity_path"], repo_root=ROOT)
    assert identity["propagation"]["method"] != rwr_identity["solver"]["method"]


# ---------------------------------------------------------------------------
# 8-14: historical Git provenance
# ---------------------------------------------------------------------------


def test_historical_provenance_passes_with_real_git_artifacts():
    identity = _identity()
    artifacts = build_valid_historical_artifacts(identity)
    result = validate_historical_provenance(identity, artifact_reader=_reader(artifacts))
    assert result["source_sha256"] == identity["historical_reference"]["source_sha256"]


def test_window_count_reference_passes_with_real_git_artifact():
    identity = _identity()
    artifacts = build_valid_historical_artifacts(identity)
    count = validate_window_count_reference(identity, artifact_reader=_reader(artifacts))
    assert count == identity["window_count_reference"]["value"]


def test_historical_row_selection_is_typed_and_matches_identity():
    identity = _identity()
    artifacts = build_valid_historical_artifacts(identity)
    reader = _reader(artifacts)
    result = validate_historical_provenance(identity, artifact_reader=reader)
    assert result["row_alpha"] == identity["historical_reference"]["k12_alpha"]
    assert result["row_steps"] == identity["historical_reference"]["k12_steps"]


def _historical_payload(identity, artifacts):
    commit = identity["historical_reference"]["source_commit"]
    path = identity["historical_reference"]["source_path"]
    return json.loads(artifacts[(commit, path)])


def _provenance_with_tampered_artifact(identity, mutate_fn):
    """Tamper the historical artifact's *content* (e.g. its row values),
    then re-point the identity's own recorded hash/blob at the tampered
    bytes -- isolating the row-level cross-check (artifact content vs. the
    identity's declared historical_reference values) from the SHA256/blob
    integrity check, which is exercised separately."""
    identity = copy.deepcopy(identity)
    artifacts = build_valid_historical_artifacts(identity)
    key = (identity["historical_reference"]["source_commit"], identity["historical_reference"]["source_path"])
    payload = json.loads(artifacts[key])
    mutate_fn(payload)
    tampered = json.dumps(payload).encode("utf-8")
    artifacts[key] = tampered
    identity["historical_reference"]["source_sha256"] = hashlib.sha256(tampered).hexdigest()
    header = f"blob {len(tampered)}\0".encode("ascii")
    identity["historical_reference"]["source_blob"] = hashlib.sha1(header + tampered).hexdigest()
    return identity, artifacts


def test_wrong_historical_alpha_rejected():
    identity, artifacts = _provenance_with_tampered_artifact(
        _identity(), lambda payload: payload["payload"]["rows"][0].__setitem__("alpha", 0.5)
    )
    with pytest.raises(MatchedK11K12Error, match="historical row alpha"):
        validate_historical_provenance(identity, artifact_reader=_reader(artifacts))


def test_wrong_historical_steps_rejected():
    identity, artifacts = _provenance_with_tampered_artifact(
        _identity(), lambda payload: payload["payload"]["rows"][0].__setitem__("steps", 10)
    )
    with pytest.raises(MatchedK11K12Error, match="historical row steps"):
        validate_historical_provenance(identity, artifact_reader=_reader(artifacts))


def test_wrong_historical_k_rejected():
    identity, artifacts = _provenance_with_tampered_artifact(
        _identity(),
        lambda payload: payload["payload"]["kappa_k_baked_at_construction_time"].__setitem__("knn_k", False),
    )
    with pytest.raises(MatchedK11K12Error, match="k=12 was baked"):
        validate_historical_provenance(identity, artifact_reader=_reader(artifacts))


def test_historical_artifact_mutation_rejected():
    identity = copy.deepcopy(_identity())
    artifacts = build_valid_historical_artifacts(identity)
    key = (identity["historical_reference"]["source_commit"], identity["historical_reference"]["source_path"])
    artifacts[key] = artifacts[key] + b" "  # one appended byte changes the SHA256, identity is NOT updated
    with pytest.raises(MatchedK11K12Error, match="SHA256 mismatch"):
        validate_historical_provenance(identity, artifact_reader=_reader(artifacts))


def test_missing_git_object_rejected():
    identity = _identity()

    def failing_reader(commit, path):
        raise MatchedK11K12Error(f"cannot read committed artifact {commit}:{path}: not found")

    with pytest.raises(MatchedK11K12Error, match="not found"):
        validate_historical_provenance(identity, artifact_reader=failing_reader)


def test_path_traversal_rejected_in_historical_reference(tmp_path):
    raw = IDENTITY_PATH.read_text().replace(
        'source_path = "ablationAll/e10_adaptive_diffusion/results/global_sweep_ext_converged.json"',
        'source_path = "../outside/evil.json"',
    )
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw)
    with pytest.raises(MatchedK11K12Error, match="safe repository-relative path"):
        load_identity(mutated, repo_root=ROOT)


def test_path_traversal_rejected_in_per_image_statistics_artifact():
    identity = _identity()
    record = build_valid_record(identity)
    record["per_image_statistics_artifact"]["path"] = "../../etc/passwd"
    with pytest.raises(MatchedK11K12Error, match="safe repository-relative path"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


# ---------------------------------------------------------------------------
# 16-20: valid result records and paired delta
# ---------------------------------------------------------------------------


def test_valid_synthetic_result_record_passes():
    identity = _identity()
    record = build_valid_record(identity)
    message = verify_record(record, identity, identity_sha256=_identity_sha256())
    assert message.startswith("MATCHED K11/K12 RESULT PASS")


def test_valid_null_delta_passes():
    identity = _identity()
    record = build_valid_record(identity, k11_bias=0)
    message = verify_record(record, identity, identity_sha256=_identity_sha256())
    assert "paired_delta=0.000000000000" in message


def test_valid_positive_delta_passes():
    identity = _identity()
    record = build_valid_record(identity, k11_bias=2)
    verify_record(record, identity, identity_sha256=_identity_sha256())
    assert record["paired_delta"] > 0


def test_valid_negative_delta_passes():
    identity = _identity()
    record = build_valid_record(identity, k11_bias=-2)
    verify_record(record, identity, identity_sha256=_identity_sha256())
    assert record["paired_delta"] < 0


def test_inconsistent_supplied_delta_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["paired_delta"] = record["paired_delta"] + Decimal("1.0")
    with pytest.raises(MatchedK11K12Error, match="paired_delta is inconsistent"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


# ---------------------------------------------------------------------------
# 21-27: telemetry, contract, and metric-source rejections
# ---------------------------------------------------------------------------


def test_wrong_step_count_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["max_steps"] = identity["propagation"]["steps"] - 1
    with pytest.raises(MatchedK11K12Error, match="max_steps must equal"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_early_termination_telemetry_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["runtime_telemetry"]["early_termination"] = True
    with pytest.raises(MatchedK11K12Error, match="early_termination must be false"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_second_backbone_pass_telemetry_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["runtime_telemetry"]["second_backbone_pass_count"] = 1
    with pytest.raises(MatchedK11K12Error, match="second_backbone_pass_count must be 0"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_backbone_forward_count_twice_window_count_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["runtime_telemetry"]["backbone_forward_count"] = 2 * identity["window_count_reference"]["value"]
    with pytest.raises(MatchedK11K12Error, match="backbone_forward_count must equal window_count"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_independently_selected_k11_attestation_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["graph_contract"]["k11_construction_method"] = "independently_selected_k11_graph"
    with pytest.raises(MatchedK11K12Error, match="graph_contract.k11_construction_method mismatch"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_prefix_mismatch_count_greater_than_zero_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["runtime_telemetry"]["k11_prefix_mismatch_count"] = 1
    with pytest.raises(MatchedK11K12Error, match="k11_prefix_mismatch_count must be 0"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_wrong_metric_unit_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k11"]["metric_unit"] = "fraction_0_1"
    with pytest.raises(MatchedK11K12Error, match="metric_unit mismatch"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_rounded_metric_source_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k11"]["metric_source"] = "rounded_prettytable_display_value"
    with pytest.raises(MatchedK11K12Error, match="metric_source mismatch"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_rounded_only_metric_value_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["mIoU"] = Decimal("50.0")
    with pytest.raises(MatchedK11K12Error, match="rounded-only"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


# ---------------------------------------------------------------------------
# 28-33: aggregate arrays, recomputation, cross-variant invariants
# ---------------------------------------------------------------------------


def test_wrong_array_length_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["intersection"] = record["variant_results"]["k12"]["intersection"][:-1]
    with pytest.raises(MatchedK11K12Error, match="exact array of length 171"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_negative_confusion_count_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["intersection"][0] = -1
    with pytest.raises(MatchedK11K12Error, match="must be at least 0"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_non_integer_array_entry_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["intersection"][0] = 1.5
    with pytest.raises(MatchedK11K12Error, match="exact JSON integer"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_intersection_exceeding_union_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["intersection"][0] = record["variant_results"]["k12"]["union"][0] + 1
    with pytest.raises(MatchedK11K12Error, match="exceeds union"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_metrics_recomputed_from_arrays_reject_disagreement():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["mIoU"] = record["variant_results"]["k12"]["mIoU"] + Decimal("10.000000000000")
    with pytest.raises(MatchedK11K12Error, match="disagrees with recomputation"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_ground_truth_arrays_must_match_between_variants():
    identity = _identity()
    record = build_valid_record(identity)
    k11 = record["variant_results"]["k11"]
    gt = list(k11["ground_truth_pixels"])
    gt[0] += 1
    k11["ground_truth_pixels"] = gt
    # Recompute k11's own metrics from its (now-diverged) arrays so the
    # per-variant recomputation-consistency check still passes; only the
    # cross-variant GT-identity check should fail.
    aAcc, mIoU, mAcc = recompute_metrics(k11["intersection"], k11["union"], k11["predicted_pixels"], gt)
    k11["aAcc"] = Decimal(f"{aAcc:.12f}")
    k11["mIoU"] = Decimal(f"{mIoU:.12f}")
    k11["mAcc"] = Decimal(f"{mAcc:.12f}")
    record["paired_delta"] = k11["mIoU"] - record["variant_results"]["k12"]["mIoU"]
    with pytest.raises(MatchedK11K12Error, match="ground_truth_pixels arrays must be identical"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_completed_image_count_mismatch_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k11"]["completed_image_count"] -= 1
    with pytest.raises(MatchedK11K12Error, match="completed_image_count must equal"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_completed_window_count_mismatch_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["completed_window_count"] -= 1
    with pytest.raises(MatchedK11K12Error, match="completed_window_count must equal"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_missing_variant_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    del record["variant_results"]["k12"]
    with pytest.raises(MatchedK11K12Error, match="exactly the variants"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_extra_variant_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k13"] = copy.deepcopy(record["variant_results"]["k12"])
    with pytest.raises(MatchedK11K12Error, match="exactly the variants"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_reversed_variant_order_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"] = {
        "k12": record["variant_results"]["k12"],
        "k11": record["variant_results"]["k11"],
    }
    with pytest.raises(MatchedK11K12Error, match="exactly the variants"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_wrong_k_for_variant_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k11"]["k"] = 12
    with pytest.raises(MatchedK11K12Error, match="k11.k must equal 11"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_non_finite_count_nonzero_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["non_finite_count"] = 1
    with pytest.raises(MatchedK11K12Error, match="non_finite_count must be 0"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_error_count_nonzero_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k12"]["error_count"] = 1
    with pytest.raises(MatchedK11K12Error, match="error_count must be 0"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_fallback_row_count_must_match_between_variants():
    identity = _identity()
    record = build_valid_record(identity)
    record["variant_results"]["k11"]["fallback_row_count"] = 999
    with pytest.raises(MatchedK11K12Error, match="fallback_row_count must be identical"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_fallback_row_counts_telemetry_must_match_variant_results():
    identity = _identity()
    record = build_valid_record(identity)
    record["runtime_telemetry"]["fallback_row_counts"]["k11"] = 999
    with pytest.raises(MatchedK11K12Error, match="fallback_row_counts.k11 disagrees"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_incomplete_result_marked_complete_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["final"] = False
    with pytest.raises(MatchedK11K12Error, match="not final"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_missing_or_mismatched_identity_sha256_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["identity_sha256"] = "0" * 64
    with pytest.raises(MatchedK11K12Error, match="identity_sha256 does not match"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


def test_unknown_top_level_key_rejected():
    identity = _identity()
    record = build_valid_record(identity)
    record["bogus_extra_field"] = True
    with pytest.raises(MatchedK11K12Error, match="unexpected schema"):
        verify_record(record, identity, identity_sha256=_identity_sha256())


# ---------------------------------------------------------------------------
# 34-35: per-image sufficient-statistic contract
# ---------------------------------------------------------------------------


def test_per_image_statistics_row_schema():
    row = {
        "image_id": "000000001.jpg",
        "intersection": [1, 2, 3],
        "union": [2, 3, 4],
        "predicted_pixels": [1, 2, 3],
        "ground_truth_pixels": [2, 3, 4],
    }
    validated = validate_per_image_statistics_row(row, class_count=3)
    assert validated["image_id"] == "000000001.jpg"


def test_per_image_statistics_additive_aggregation_passes():
    rows = [
        {
            "image_id": "a", "intersection": [1, 0, 2], "union": [2, 1, 2],
            "predicted_pixels": [1, 0, 2], "ground_truth_pixels": [2, 1, 2],
        },
        {
            "image_id": "b", "intersection": [3, 1, 0], "union": [3, 2, 1],
            "predicted_pixels": [3, 1, 0], "ground_truth_pixels": [3, 2, 1],
        },
    ]
    validate_per_image_statistics_additive_aggregation(
        rows, class_count=3,
        dataset_intersection=[4, 1, 2], dataset_union=[5, 3, 3],
        dataset_predicted_pixels=[4, 1, 2], dataset_ground_truth_pixels=[5, 3, 3],
    )


def test_per_image_statistics_additive_aggregation_rejects_mismatch():
    rows = [
        {
            "image_id": "a", "intersection": [1, 0, 2], "union": [2, 1, 2],
            "predicted_pixels": [1, 0, 2], "ground_truth_pixels": [2, 1, 2],
        },
    ]
    with pytest.raises(MatchedK11K12Error, match="additive aggregation"):
        validate_per_image_statistics_additive_aggregation(
            rows, class_count=3,
            dataset_intersection=[999, 0, 0], dataset_union=[2, 1, 2],
            dataset_predicted_pixels=[1, 0, 2], dataset_ground_truth_pixels=[2, 1, 2],
        )


def test_per_image_statistics_artifact_metadata_schema():
    identity = _identity()
    metadata = {
        "path": "artifacts/e12_per_image_stats.json",
        "sha256": "c" * 64,
        "image_count": identity["dataset"]["images"],
        "class_count": identity["dataset"]["classes"],
        "schema": PER_IMAGE_STATS_SCHEMA_NAME,
        "image_id_digest": "d" * 64,
        "additive_aggregation_verified": True,
    }
    validate_per_image_statistics_artifact_metadata(metadata, identity)


def test_per_image_statistics_artifact_wrong_image_count_rejected():
    identity = _identity()
    metadata = {
        "path": "artifacts/e12_per_image_stats.json",
        "sha256": "c" * 64,
        "image_count": 42,
        "class_count": identity["dataset"]["classes"],
        "schema": PER_IMAGE_STATS_SCHEMA_NAME,
        "image_id_digest": "d" * 64,
        "additive_aggregation_verified": True,
    }
    with pytest.raises(MatchedK11K12Error, match="image_count must equal dataset.images"):
        validate_per_image_statistics_artifact_metadata(metadata, identity)


def test_no_per_image_miou_scalar_field_in_schema():
    forbidden_terms = {"per_image_miou", "per_image_mean_iou", "image_miou"}
    from src.matched_k11_k12_identity import PER_IMAGE_ROW_KEYS
    schema_terms = {name.lower() for name in PER_IMAGE_STATISTICS_ARTIFACT_KEYS | PER_IMAGE_ROW_KEYS}
    assert not (forbidden_terms & schema_terms)


# ---------------------------------------------------------------------------
# 36: immutability of inputs
# ---------------------------------------------------------------------------


def test_verify_record_does_not_mutate_the_input_record():
    identity = _identity()
    record = build_valid_record(identity)
    before = copy.deepcopy(record)
    verify_record(record, identity, identity_sha256=_identity_sha256())
    assert record == before


def test_load_identity_does_not_mutate_the_toml_file():
    before = IDENTITY_PATH.read_bytes()
    load_identity(repo_root=ROOT)
    after = IDENTITY_PATH.read_bytes()
    assert before == after


# ---------------------------------------------------------------------------
# 37-39: CLI missing-file / malformed-result handling
# ---------------------------------------------------------------------------


def test_cli_missing_result_file_clean_failure(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    completed = subprocess.run(
        [
            "python", str(ROOT / "verify_matched_k11_k12.py"), "verify-result",
            "--result", str(missing), "--repo-root", str(ROOT),
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert completed.stderr.startswith("MATCHED K11/K12 VERIFICATION FAIL:")
    assert "Traceback" not in completed.stderr
    assert "Traceback" not in completed.stdout


def test_cli_malformed_result_clean_failure(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not valid json")
    completed = subprocess.run(
        [
            "python", str(ROOT / "verify_matched_k11_k12.py"), "verify-result",
            "--result", str(malformed), "--repo-root", str(ROOT),
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert completed.stderr.startswith("MATCHED K11/K12 VERIFICATION FAIL:")
    assert "Traceback" not in completed.stderr


def test_cli_non_finite_json_result_clean_failure(tmp_path):
    non_finite = tmp_path / "nonfinite.json"
    non_finite.write_text('{"schema": "x", "value": NaN}')
    completed = subprocess.run(
        [
            "python", str(ROOT / "verify_matched_k11_k12.py"), "verify-result",
            "--result", str(non_finite), "--repo-root", str(ROOT),
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert "non-finite" in completed.stderr
    assert "Traceback" not in completed.stderr


# ---------------------------------------------------------------------------
# 40-41: preflights
# ---------------------------------------------------------------------------


def test_static_preflight_succeeds_against_the_real_repository():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["identity_name"] == "e12-matched-k11-k12-t320"
    assert result["e3_identity"] == "e3-paired-soft-routing"
    assert result["rwr_identity"] == "e3-canonical-directed-rwr"
    assert result["window_count"] == 11075 or result["window_count"] == _identity()["window_count_reference"]["value"]


def test_existing_e3_and_rwr_preflights_still_pass():
    e3 = subprocess.run(
        ["python", str(ROOT / "verify_e3_identity.py"), "preflight"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert e3.returncode == 0, e3.stderr
    assert "E3 PREFLIGHT PASS" in e3.stdout
    rwr = subprocess.run(
        ["python", str(ROOT / "verify_rwr_reproduction.py"), "preflight"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert rwr.returncode == 0, rwr.stderr
    assert "RWR PREFLIGHT PASS" in rwr.stdout


def test_cli_preflight_passes():
    completed = subprocess.run(
        ["python", str(ROOT / "verify_matched_k11_k12.py"), "preflight", "--repo-root", str(ROOT)],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "MATCHED K11/K12 PREFLIGHT PASS" in completed.stdout


# ---------------------------------------------------------------------------
# 42: TOML is the sole source of scientific constants
# ---------------------------------------------------------------------------


def test_canonical_scientific_literals_do_not_reappear_in_python():
    identity = _identity()
    # Note: the step count (320) is deliberately excluded from this set --
    # it is part of this experiment's own name/schema identifiers
    # ("e12_matched_k11_k12_t320", "...-t320-result-v1"), not a value the
    # module ever hardcodes for a validation comparison (every check
    # compares against identity["propagation"]["steps"], never a literal).
    forbidden = {
        str(identity["propagation"]["alpha"]),
        str(identity["graph"]["affinity_power"]),
        str(identity["window_count_reference"]["value"]),
        str(identity["historical_reference"]["k12_aAcc"]),
        str(identity["historical_reference"]["k12_mIoU"]),
        str(identity["historical_reference"]["k12_mAcc"]),
        identity["historical_reference"]["source_commit"],
        identity["historical_reference"]["source_sha256"],
        identity["window_count_reference"]["source_sha256"],
        identity["parent_identities"]["e3_identity_sha256"],
        identity["parent_identities"]["rwr_identity_sha256"],
        identity["identity"]["required_ancestor_commit"],
    }
    sources = [
        ROOT / "src/matched_k11_k12_identity.py",
        ROOT / "verify_matched_k11_k12.py",
    ]
    combined = "\n".join(path.read_text() for path in sources)
    offenders = [literal for literal in forbidden if literal in combined]
    assert offenders == [], f"forbidden scientific literals reappear in Python: {offenders}"


# ---------------------------------------------------------------------------
# 43: no production inference imports the new module
# ---------------------------------------------------------------------------


def test_no_production_inference_imports_the_new_module():
    production_files = [
        ROOT / "src/open_vocabulary_segmentation/main.py",
        ROOT / "src/open_vocabulary_segmentation/models/dinotext/dinotext.py",
        ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr/inference.py",
        ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py",
    ]
    for path in production_files:
        text = path.read_text()
        assert "matched_k11_k12_identity" not in text, f"{path} must not import the new identity module"


def test_module_imports_no_torch_mmcv_or_cv2_at_top_level():
    import ast
    source = (ROOT / "src/matched_k11_k12_identity.py").read_text()
    tree = ast.parse(source)
    forbidden_modules = {"torch", "cv2", "mmcv", "mmseg"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_modules
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_modules


def test_does_not_implement_an_evaluator_or_production_yaml():
    # This commit pre-registers the identity only; it must not add an
    # evaluator entry point or a new production evaluation YAML config.
    assert not (ROOT / "evaluate_matched_k11_k12.py").exists()
    config_dir = ROOT / "src/open_vocabulary_segmentation/configs/stuff"
    for path in config_dir.glob("*k11*k12*"):
        pytest.fail(f"unexpected production config artifact for this commit: {path}")


# ---------------------------------------------------------------------------
# Regression tests for the independent-verification findings:
#   Fix 1 -- the parent RWR top_k comparison must be relational, not a bare
#            hardcoded literal.
#   Fix 2 -- metrics.unit / propagation.compute_dtype / propagation.output_dtype
#            must be closed to exactly one supported value each.
# ---------------------------------------------------------------------------


def test_no_bare_hardcoded_parent_top_k_literal_in_source():
    source = (ROOT / "src/matched_k11_k12_identity.py").read_text()
    assert 'rwr_identity["rwr"]["top_k"] != 12' not in source
    assert 'rwr_identity["rwr"]["top_k"] != identity["graph"]["maximum_rank"]' in source


def test_parent_top_k_mismatch_is_relationally_rejected():
    # Calls the relational-equality function directly with a hand-mutated
    # RWR identity mapping (in-memory, no filesystem/git involved) --
    # isolates exactly the code path Fix 1 touched.
    identity = _identity()
    e3 = _load_e3_identity_direct(ROOT / identity["parent_identities"]["e3_identity_path"], repo_root=ROOT)
    rwr = copy.deepcopy(
        _load_rwr_identity_direct(ROOT / identity["parent_identities"]["rwr_identity_path"], repo_root=ROOT)
    )
    assert rwr["rwr"]["top_k"] == identity["graph"]["maximum_rank"]  # sanity: real values agree
    rwr["rwr"]["top_k"] = identity["graph"]["maximum_rank"] + 1
    with pytest.raises(MatchedK11K12Error, match="graph.maximum_rank"):
        _validate_relational_equality(identity, e3, rwr)


def test_parent_top_k_mismatch_error_does_not_mention_bare_twelve():
    identity = _identity()
    e3 = _load_e3_identity_direct(ROOT / identity["parent_identities"]["e3_identity_path"], repo_root=ROOT)
    rwr = copy.deepcopy(
        _load_rwr_identity_direct(ROOT / identity["parent_identities"]["rwr_identity_path"], repo_root=ROOT)
    )
    rwr["rwr"]["top_k"] = 99
    with pytest.raises(MatchedK11K12Error) as excinfo:
        _validate_relational_equality(identity, e3, rwr)
    # The diagnostic must name the relational field, not assert a bare "12".
    assert "graph.maximum_rank" in str(excinfo.value)


def test_supported_vocabulary_constants_match_the_real_identity():
    identity = _identity()
    assert identity["metrics"]["unit"] == SUPPORTED_METRIC_UNIT
    assert identity["propagation"]["compute_dtype"] == SUPPORTED_COMPUTE_DTYPE
    assert identity["propagation"]["output_dtype"] == SUPPORTED_OUTPUT_DTYPE


def _mutate_identity_toml(tmp_path, old, new, *, count=1):
    raw = IDENTITY_PATH.read_text()
    assert old in raw
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(raw.replace(old, new, count))
    return mutated


@pytest.mark.parametrize(
    "old, new, match",
    [
        ('unit = "percent_0_100"', 'unit = "percent"', r"metrics\.unit"),
        ('unit = "percent_0_100"', 'unit = "fraction_0_1"', r"metrics\.unit"),
        ('unit = "percent_0_100"', 'unit = "PERCENT_0_100"', r"metrics\.unit"),
        ('unit = "percent_0_100"', 'unit = "percent_0_100 "', r"metrics\.unit"),
        ('unit = "percent_0_100"', 'unit = " percent_0_100"', r"metrics\.unit"),
        ('compute_dtype = "float32"', 'compute_dtype = "float16"', r"propagation\.compute_dtype"),
        ('compute_dtype = "float32"', 'compute_dtype = "float64"', r"propagation\.compute_dtype"),
        ('compute_dtype = "float32"', 'compute_dtype = "fp32"', r"propagation\.compute_dtype"),
        ('compute_dtype = "float32"', 'compute_dtype = "torch.float32"', r"propagation\.compute_dtype"),
        ('output_dtype = "float32"', 'output_dtype = "float16"', r"propagation\.output_dtype"),
        ('output_dtype = "float32"', 'output_dtype = "float64"', r"propagation\.output_dtype"),
        ('output_dtype = "float32"', 'output_dtype = "fp32"', r"propagation\.output_dtype"),
        ('output_dtype = "float32"', 'output_dtype = "torch.float32"', r"propagation\.output_dtype"),
    ],
)
def test_closed_vocabulary_rejections(tmp_path, old, new, match):
    mutated = _mutate_identity_toml(tmp_path, old, new)
    with pytest.raises(MatchedK11K12Error, match=match):
        load_identity(mutated, repo_root=ROOT)


def test_metric_unit_numeric_value_rejected(tmp_path):
    mutated = _mutate_identity_toml(tmp_path, 'unit = "percent_0_100"', "unit = 100")
    with pytest.raises(MatchedK11K12Error, match=r"metrics\.unit"):
        load_identity(mutated, repo_root=ROOT)


def test_metric_unit_boolean_value_rejected(tmp_path):
    mutated = _mutate_identity_toml(tmp_path, 'unit = "percent_0_100"', "unit = true")
    with pytest.raises(MatchedK11K12Error, match=r"metrics\.unit"):
        load_identity(mutated, repo_root=ROOT)


def test_closed_vocabulary_diagnostics_name_field_expected_and_observed(tmp_path):
    mutated = _mutate_identity_toml(tmp_path, 'unit = "percent_0_100"', 'unit = "percent"')
    with pytest.raises(MatchedK11K12Error) as excinfo:
        load_identity(mutated, repo_root=ROOT)
    message = str(excinfo.value)
    assert "metrics.unit" in message
    assert "percent_0_100" in message
    assert "'percent'" in message


def test_exact_canonical_unit_and_dtype_values_pass(tmp_path):
    # Round-trip: rewriting the file with the *same* canonical values must
    # still load cleanly -- proves the check is exact-equality, not merely
    # "rejects everything."
    raw = IDENTITY_PATH.read_text()
    mutated = tmp_path / "roundtrip.toml"
    mutated.write_text(raw)
    identity = load_identity(mutated, repo_root=ROOT)
    assert identity["metrics"]["unit"] == SUPPORTED_METRIC_UNIT
    assert identity["propagation"]["compute_dtype"] == SUPPORTED_COMPUTE_DTYPE
    assert identity["propagation"]["output_dtype"] == SUPPORTED_OUTPUT_DTYPE


@pytest.mark.parametrize(
    "old, new",
    [
        ('unit = "percent_0_100"', 'unit = "percent"'),
        ('compute_dtype = "float32"', 'compute_dtype = "float64"'),
        ('output_dtype = "float32"', 'output_dtype = "float16"'),
    ],
)
def test_cli_preflight_rejects_closed_vocabulary_mutations_cleanly(tmp_path, old, new):
    mutated = _mutate_identity_toml(tmp_path, old, new)
    result = subprocess.run(
        [
            "python", str(ROOT / "verify_matched_k11_k12.py"), "--identity", str(mutated),
            "preflight", "--repo-root", str(ROOT),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert result.stderr.startswith("MATCHED K11/K12 VERIFICATION FAIL:")
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


def test_closed_vocabulary_mutation_probes_do_not_mutate_the_real_toml(tmp_path):
    before = IDENTITY_PATH.read_bytes()
    for old, new in (
        ('unit = "percent_0_100"', 'unit = "percent"'),
        ('compute_dtype = "float32"', 'compute_dtype = "float64"'),
        ('output_dtype = "float32"', 'output_dtype = "float16"'),
    ):
        mutated = _mutate_identity_toml(tmp_path, old, new)
        with pytest.raises(MatchedK11K12Error):
            load_identity(mutated, repo_root=ROOT)
    after = IDENTITY_PATH.read_bytes()
    assert before == after
