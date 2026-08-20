import copy
import json
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from src.rwr_reproduction_identity import (
    FULL_PRECISION_METRIC_SOURCE,
    RESULT_FORMAT_VERSION_V3,
    RESULT_KEYS_V3,
    RESULT_PREFIX,
    RESULT_FORMAT_VERSION,
    RWRReproductionError,
    SUPPORTED_CANONICAL_GRAPH_MODE,
    SUPPORTED_CANONICAL_SOLVER_METHOD,
    SUPPORTED_RESULT_FORMAT_VERSIONS,
    load_identity,
    validate_historical_provenance,
    validate_resolved_config_pair,
    validate_static_configuration,
    verify_record,
    verify_result,
)
from src.e3_evaluation_identity import (
    _deep_merge, _load_yaml_with_bases, load_identity as load_e3_identity,
)


ROOT = Path(__file__).parents[1]


def build_valid_record(identity):
    return {
        "format_version": RESULT_FORMAT_VERSION,
        "identity_name": identity["identity_name"],
        "image_count": identity["dataset"]["images"],
        "class_count": identity["dataset"]["classes"],
        "aAcc": identity["expected_metrics"]["rwr_aAcc"],
        "mIoU": identity["expected_metrics"]["rwr_mIoU"],
        "mAcc": identity["expected_metrics"]["rwr_mAcc"],
        "metrics_precision": "full",
        "gain_over_e3_miou": identity["expected_metrics"]["gain_mIoU"],
        "rwr_enabled": identity["rwr"]["enabled"],
        "alpha": identity["rwr"]["alpha"],
        "top_k": identity["rwr"]["top_k"],
        "affinity_power": identity["rwr"]["affinity_power"],
        "graph_mode": identity["rwr"]["graph_mode"],
        "solver": identity["solver"]["method"],
        "solver_rtol": identity["solver"]["rtol"],
        "solver_atol": identity["solver"]["atol"],
        "solver_max_iterations": identity["solver"]["max_iterations"],
        "crop": identity["evaluation"]["crop"],
        "stride": identity["evaluation"]["stride"],
        "pamr": identity["evaluation"]["pamr"],
        "background_class": identity["dataset"]["background_class"],
        "config_path": identity["canonical_config_path"],
        "config_sha256": identity["canonical_config_sha256"],
        "checkpoint_path": identity["checkpoint"]["path"],
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "source_e10_commit": identity["source_e10_commit"],
        "cache_source_commit": identity["cache_source_commit"],
        "cache_manifest_sha256": identity["cache_manifest_sha256"],
        "cache_manifest_evidence": identity["cache_manifest_evidence"],
        "cache_manifest_attestation_commit": identity[
            "cache_manifest_attestation_commit"
        ],
        "cache_manifest_attestation_path": identity[
            "cache_manifest_attestation_path"
        ],
        "cache_manifest_attestation_blob_sha256": identity[
            "cache_manifest_attestation_blob_sha256"
        ],
        "cache_manifest_archived": identity["cache_manifest_archived"],
        "historical_cache_used_by_current_run": identity[
            "historical_cache_used_by_current_run"
        ],
        "source_git_commit": "a" * 40,
        "source_git_branch": "e11-cover-dr-1",
        "source_git_dirty": False,
        "gpu_model": "synthetic GPU",
        "torch_version": "2.0",
        "cuda_version": "11.8",
        "elapsed_seconds": 12.5,
        "solver_summary": {
            "window_count": 11075,
            "converged_window_count": 11075,
            "total_iterations": 100,
            "minimum_iterations": 1,
            "maximum_iterations": 4,
            "total_restarts": 3,
            "nonzero_restart_windows": 2,
            "total_residual_replacements": 3,
            "total_fallback_rows": 0,
            "maximum_scaled_residual": 0.9,
        },
    }


def _record():
    return build_valid_record(load_identity(repo_root=ROOT))


def build_valid_record_v3(identity, *, parity_solver_summary=None):
    record = build_valid_record(identity)
    record["format_version"] = RESULT_FORMAT_VERSION_V3
    record["metric_source"] = FULL_PRECISION_METRIC_SOURCE
    record["parity_solver_summary"] = parity_solver_summary
    return record


def _record_v3(**kwargs):
    return build_valid_record_v3(load_identity(repo_root=ROOT), **kwargs)


def build_valid_resolved_config(identity):
    e3_identity = load_e3_identity(
        ROOT / identity["e3_identity_path"], repo_root=ROOT
    )
    e3_path = ROOT / e3_identity["evaluation"]["config_path"]
    rwr_path = ROOT / identity["canonical_config_path"]
    base_path = ROOT / e3_identity["evaluation"]["base_config_path"]
    return (
        _deep_merge(_load_yaml_with_bases(e3_path), _load_yaml_with_bases(base_path)),
        _deep_merge(_load_yaml_with_bases(rwr_path), _load_yaml_with_bases(base_path)),
    )


def build_valid_historical_artifacts(identity):
    artifacts = {}
    for prefix in ("metrics", "cache_evidence", "graph", "solver", "config", "eval_base"):
        commit = identity["historical"][f"{prefix}_commit"]
        path = identity["historical"][f"{prefix}_path"]
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
            raise RWRReproductionError("missing injected historical artifact") from error
    return read


def _replace_artifact(identity, artifacts, prefix, data):
    identity = copy.deepcopy(identity)
    artifacts = dict(artifacts)
    historical = identity["historical"]
    artifacts[(historical[f"{prefix}_commit"], historical[f"{prefix}_path"])] = data
    historical[f"{prefix}_sha256"] = hashlib.sha256(data).hexdigest()
    historical[f"{prefix}_blob"] = hashlib.sha1(
        f"blob {len(data)}\0".encode() + data
    ).hexdigest()
    if prefix == "cache_evidence":
        identity["cache_manifest_attestation_blob_sha256"] = historical[
            "cache_evidence_sha256"
        ]
    return identity, artifacts


def _write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=True) + "\n", encoding="utf-8")


def _write_capability_mutation(tmp_path, section, field, observed):
    identity = load_identity(repo_root=ROOT)
    source = (ROOT / "evaluation_identities/e3_canonical_directed_rwr.toml").read_text()
    expected = identity[section][field]
    old = f"{field} = {json.dumps(expected)}"
    new = f"{field} = {json.dumps(observed)}"
    assert source.count(old) == 1
    path = tmp_path / f"{section}-{field}-{str(observed)}.toml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return path, expected


def _copy_rwr_static_repository(destination):
    for relative in (
        "evaluation_identities",
        "src/open_vocabulary_segmentation/configs/stuff",
        "src/open_vocabulary_segmentation/segmentation/configs/_base_",
        "configs",
    ):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    constructor = ROOT / load_e3_identity(repo_root=ROOT)["model"]["constructor_path"]
    target = destination / constructor.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(constructor, target)
    return destination


def test_static_preflight_proves_e3_dependency_config_allowlist_and_e10_source():
    identity = load_identity(repo_root=ROOT)
    result = validate_static_configuration(repo_root=ROOT)
    assert result["identity_name"] == identity["identity_name"]
    assert result["e3_identity"] == load_e3_identity(repo_root=ROOT)["identity_name"]
    assert result["source_e10_commit"] == identity["source_e10_commit"]


def test_canonical_capabilities_are_centralized_and_identity_passes():
    identity = load_identity(repo_root=ROOT)
    assert identity["rwr"]["graph_mode"] == SUPPORTED_CANONICAL_GRAPH_MODE
    assert identity["solver"]["method"] == SUPPORTED_CANONICAL_SOLVER_METHOD
    assert validate_static_configuration(repo_root=ROOT)["identity_name"] == identity["identity_name"]


@pytest.mark.parametrize(
    ("section", "field", "observed", "capability"),
    [
        ("rwr", "graph_mode", "undirected_topk", SUPPORTED_CANONICAL_GRAPH_MODE),
        ("rwr", "graph_mode", "wrong", SUPPORTED_CANONICAL_GRAPH_MODE),
        ("rwr", "graph_mode", "", SUPPORTED_CANONICAL_GRAPH_MODE),
        ("rwr", "graph_mode", 1, SUPPORTED_CANONICAL_GRAPH_MODE),
        ("solver", "method", "fixed_point", SUPPORTED_CANONICAL_SOLVER_METHOD),
        ("solver", "method", "gmres", SUPPORTED_CANONICAL_SOLVER_METHOD),
        ("solver", "method", "wrong", SUPPORTED_CANONICAL_SOLVER_METHOD),
        ("solver", "method", "", SUPPORTED_CANONICAL_SOLVER_METHOD),
        ("solver", "method", 1, SUPPORTED_CANONICAL_SOLVER_METHOD),
    ],
)
def test_load_identity_rejects_unsupported_capabilities(
    tmp_path, section, field, observed, capability
):
    path, _expected = _write_capability_mutation(
        tmp_path, section, field, observed
    )
    with pytest.raises(RWRReproductionError) as error:
        load_identity(path, repo_root=ROOT)
    message = str(error.value)
    assert f"{section}.{field}" in message
    assert repr(capability) in message
    assert repr(observed) in message


@pytest.mark.parametrize(
    ("section", "field", "observed", "capability"),
    [
        ("rwr", "graph_mode", "undirected_topk", SUPPORTED_CANONICAL_GRAPH_MODE),
        ("rwr", "graph_mode", "wrong", SUPPORTED_CANONICAL_GRAPH_MODE),
        ("rwr", "graph_mode", "", SUPPORTED_CANONICAL_GRAPH_MODE),
        ("rwr", "graph_mode", 1, SUPPORTED_CANONICAL_GRAPH_MODE),
        ("solver", "method", "fixed_point", SUPPORTED_CANONICAL_SOLVER_METHOD),
        ("solver", "method", "gmres", SUPPORTED_CANONICAL_SOLVER_METHOD),
        ("solver", "method", "wrong", SUPPORTED_CANONICAL_SOLVER_METHOD),
        ("solver", "method", "", SUPPORTED_CANONICAL_SOLVER_METHOD),
        ("solver", "method", 1, SUPPORTED_CANONICAL_SOLVER_METHOD),
    ],
)
def test_static_preflight_rejects_unsupported_capabilities_before_models(
    tmp_path, section, field, observed, capability
):
    path, _expected = _write_capability_mutation(
        tmp_path, section, field, observed
    )
    with pytest.raises(RWRReproductionError) as error:
        validate_static_configuration(
            repo_root=ROOT,
            identity_path=path,
            check_git=False,
        )
    message = str(error.value)
    assert f"{section}.{field}" in message
    assert repr(capability) in message
    assert repr(observed) in message


def test_canonical_config_contains_only_the_approved_evaluate_rwr_delta():
    import yaml

    identity = load_identity(repo_root=ROOT)
    config = yaml.safe_load((ROOT / identity["canonical_config_path"]).read_text())
    assert set(config) == {"_base_", "evaluate"}
    assert set(config["evaluate"]) == {"rwr"}
    assert config["_base_"] == (
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    )


def test_exact_structured_json_and_log_results_pass(tmp_path):
    result = tmp_path / "result.json"
    log = tmp_path / "eval.log"
    _write_json(result, _record())
    log.write_text(
        "ordinary log\n" + RESULT_PREFIX + result.read_text().strip() + "\n",
        encoding="utf-8",
    )
    assert verify_result(result, source_kind="json").startswith(
        "RWR REPRODUCTION PASS"
    )
    assert verify_result(log, source_kind="log").startswith(
        "RWR REPRODUCTION PASS"
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("image_count", 4999, "image count"),
        ("class_count", 170, "class count"),
        ("alpha", 0.9, "alpha"),
        ("top_k", 11, "top_k"),
        ("affinity_power", 2.0, "affinity_power"),
        ("graph_mode", "symmetric", "graph_mode"),
        ("solver", "fixed_point", "solver"),
        ("solver_rtol", 0.1, "solver_rtol"),
        ("solver_atol", float("nan"), "non-finite"),
        ("solver_max_iterations", 0, "solver_max_iterations"),
        ("pamr", True, "pamr"),
    ],
)
def test_wrong_canonical_identity_fields_fail(tmp_path, field, value, match):
    record = _record()
    record[field] = value
    path = tmp_path / "wrong.json"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match=match):
        verify_result(path, source_kind="json")


def test_missing_and_duplicate_structured_log_blocks_fail(tmp_path):
    missing = tmp_path / "missing.log"
    missing.write_text("mIoU: 29.88\n", encoding="utf-8")
    with pytest.raises(RWRReproductionError, match="exactly one"):
        verify_result(missing, source_kind="log")
    duplicate = tmp_path / "duplicate.log"
    block = RESULT_PREFIX + json.dumps(_record())
    duplicate.write_text(block + "\n" + block + "\n", encoding="utf-8")
    with pytest.raises(RWRReproductionError, match="exactly one"):
        verify_result(duplicate, source_kind="log")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metrics_fail(tmp_path, value):
    record = _record()
    record["mIoU"] = value
    path = tmp_path / "nonfinite.json"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="non-finite"):
        verify_result(path, source_kind="json")


def test_fractional_metrics_fail(tmp_path):
    record = _record()
    record.update({"aAcc": 0.4852867057377273, "mIoU": 0.29877244374599126, "mAcc": 0.5413703466982515})
    path = tmp_path / "fractional.json"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="fractional"):
        verify_result(path, source_kind="json")


def test_outside_frozen_tolerance_fails(tmp_path):
    identity = load_identity(repo_root=ROOT)
    record = _record()
    record["mIoU"] = 29.8830001
    record["gain_over_e3_miou"] = (
        record["mIoU"] - identity["expected_metrics"]["e3_mIoU"]
    )
    path = tmp_path / "outside.json"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="mIoU mismatch"):
        verify_result(path, source_kind="json")


def test_rounded_only_result_cannot_masquerade_as_full_precision(tmp_path):
    identity = load_identity(repo_root=ROOT)
    record = _record()
    record["mIoU"] = identity["acceptance"]["rounded_mIoU"]
    record["gain_over_e3_miou"] = (
        record["mIoU"] - identity["expected_metrics"]["e3_mIoU"]
    )
    path = tmp_path / "rounded.json"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="rounded-only"):
        verify_result(path, source_kind="json")


def test_unknown_or_partial_schema_fails(tmp_path):
    record = _record()
    del record["mAcc"]
    path = tmp_path / "partial.json"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="schema"):
        verify_result(path, source_kind="json")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("image_count", 5000.0, "image_count.*integer"),
        ("class_count", True, "class_count.*integer"),
        ("rwr_enabled", 1, "rwr_enabled.*boolean"),
        ("alpha", 1, "alpha.*floating-point"),
        ("top_k", 12.0, "top_k.*integer"),
        ("crop", "448,448", "two-element list"),
        ("crop", [448.0, 448], "elements.*integers"),
        ("elapsed_seconds", 1, "elapsed_seconds.*floating-point"),
        ("source_git_dirty", 0, "dirty flag.*boolean"),
    ],
)
def test_structured_result_exact_type_matrix(tmp_path, field, value, match):
    record = _record()
    record[field] = value
    path = tmp_path / "wrong-type.json"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match=match):
        verify_result(path, source_kind="json")
    record = _record()
    record["unexpected"] = True
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="schema"):
        verify_result(path, source_kind="json")


def test_identity_nested_schema_is_closed(tmp_path):
    source = (ROOT / "evaluation_identities/e3_canonical_directed_rwr.toml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "identity.toml"
    path.write_text(
        source.replace("[rwr]\n", "[rwr]\nunknown_setting = true\n"),
        encoding="utf-8",
    )
    with pytest.raises(RWRReproductionError, match="rwr.*schema"):
        load_identity(path, repo_root=ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cfg: cfg["evaluate"].__setitem__("template", "wrong_template"),
        lambda cfg: cfg["model"].__setitem__("checkpoint_path", "weights/wrong.pth"),
        lambda cfg: cfg["model"].__setitem__("proj_name", "wrong_projection"),
        lambda cfg: cfg["model"].__setitem__("pre_trained", False),
        lambda cfg: cfg["model"].__setitem__("backbone_weights", "weights/wrong.pth"),
        lambda cfg: cfg["evaluate"].__setitem__("crop_size", [224, 224]),
        lambda cfg: cfg["evaluate"].__setitem__("stride", [112, 112]),
        lambda cfg: cfg["evaluate"].__setitem__("pamr", True),
        lambda cfg: cfg["data"].__setitem__("metainfo", {"classes": ["wrong"]}),
        lambda cfg: cfg["evaluate"].__setitem__("with_bg", True),
        lambda cfg: cfg["evaluate"].__setitem__("unknown_evaluation_key", 1),
    ],
)
def test_complete_resolved_config_rejects_every_non_rwr_mutation(mutation):
    identity = load_identity(repo_root=ROOT)
    e3, rwr = build_valid_resolved_config(identity)
    mutation(rwr)
    with pytest.raises(RWRReproductionError, match="outside evaluate.rwr"):
        validate_resolved_config_pair(e3, rwr, identity)


def test_three_authoritative_identity_mutations_fail(tmp_path):
    source = (ROOT / "evaluation_identities/e3_canonical_directed_rwr.toml").read_text()
    for index, (old, new) in enumerate((
        ('template = "sub_imagenet_template"', 'template = "wrong_template"'),
        ('path = "weights/vitb_mlp_infonce_paired_soft_routing_tau010.pth"', 'path = "weights/wrong.pth"'),
        ('projection_name = "vitb_mlp_infonce_paired_soft_routing_tau010"', 'projection_name = "wrong_projection"'),
    )):
        path = tmp_path / f"mutation-{index}.toml"
        path.write_text(source.replace(old, new), encoding="utf-8")
        with pytest.raises(RWRReproductionError):
            validate_static_configuration(repo_root=ROOT, identity_path=path)


def test_generated_record_tracks_authoritative_identity_changes():
    identity = load_identity(repo_root=ROOT)
    changed = copy.deepcopy(identity)
    changed["expected_metrics"]["rwr_mIoU"] += 0.001
    assert build_valid_record(changed)["mIoU"] != build_valid_record(identity)["mIoU"]


def test_committed_artifact_provenance_passes_without_documentation():
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    result = validate_historical_provenance(identity, artifact_reader=_reader(artifacts))
    assert result["metrics_sha256"] == identity["historical"]["metrics_sha256"]
    assert result["cache_manifest_evidence"] == "committed_attestation"
    assert result["cache_manifest_archived"] is False


def test_missing_historical_artifact_and_wrong_source_blob_fail():
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    key = (identity["historical"]["metrics_commit"], identity["historical"]["metrics_path"])
    del artifacts[key]
    with pytest.raises(RWRReproductionError, match="missing injected"):
        validate_historical_provenance(identity, artifact_reader=_reader(artifacts))


def test_missing_attestation_artifact_fails():
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    key = (
        identity["cache_manifest_attestation_commit"],
        identity["cache_manifest_attestation_path"],
    )
    del artifacts[key]
    with pytest.raises(RWRReproductionError, match="missing injected"):
        validate_historical_provenance(identity, artifact_reader=_reader(artifacts))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("cache_manifest_attestation_commit", "1" * 40, "attestation commit"),
        ("cache_manifest_attestation_path", "wrong/path.json", "attestation path"),
        ("cache_manifest_attestation_blob_sha256", "1" * 64, "blob SHA256"),
    ],
)
def test_wrong_attestation_identity_fails(field, value, match):
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    identity = copy.deepcopy(identity)
    identity[field] = value
    with pytest.raises(RWRReproductionError, match=match):
        validate_historical_provenance(identity, artifact_reader=_reader(artifacts))


def test_missing_and_wrong_historical_commits_fail_closed(tmp_path):
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)

    def missing_commit(commit, path):
        if commit == identity["source_e10_commit"]:
            raise RWRReproductionError("missing historical commit")
        return _reader(artifacts)(commit, path)

    with pytest.raises(RWRReproductionError, match="missing historical commit"):
        validate_historical_provenance(identity, artifact_reader=missing_commit)

    source = (ROOT / "evaluation_identities/e3_canonical_directed_rwr.toml").read_text()
    path = tmp_path / "wrong-commit.toml"
    path.write_text(
        source.replace(identity["source_e10_commit"], "1" * 40, 1),
        encoding="utf-8",
    )
    with pytest.raises(RWRReproductionError, match="attestation commit mismatch"):
        load_identity(path, repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    key = (identity["historical"]["graph_commit"], identity["historical"]["graph_path"])
    artifacts[key] += b"\nmutation"
    with pytest.raises(RWRReproductionError, match="graph SHA256"):
        validate_historical_provenance(identity, artifact_reader=_reader(artifacts))


@pytest.mark.parametrize(
    "case,match",
    [
        ("malformed", "cannot parse"),
        ("duplicate", "duplicate JSON key"),
        ("nonfinite", "non-finite"),
        ("wrong_schema", "unexpected schema"),
    ],
)
def test_malformed_duplicate_nonfinite_and_wrong_historical_metrics_fail(case, match):
    identity = load_identity(repo_root=ROOT)
    record_name = identity["historical"]["metrics_record"]
    if case == "malformed":
        payload = b"{"
    elif case == "duplicate":
        payload = f'{{"{record_name}":{{}},"{record_name}":{{}}}}'.encode()
    elif case == "nonfinite":
        payload = f'{{"{record_name}":{{"aAcc":NaN}}}}'.encode()
    else:
        payload = json.dumps(
            {record_name: {"aAcc": 1, "mIoU": 2, "mAcc": 3, "evaluated_images": 1}}
        ).encode()
    artifacts = build_valid_historical_artifacts(identity)
    changed_identity, changed_artifacts = _replace_artifact(identity, artifacts, "metrics", payload)
    with pytest.raises(RWRReproductionError, match=match):
        validate_historical_provenance(
            changed_identity, artifact_reader=_reader(changed_artifacts)
        )


def test_wrong_committed_cache_manifest_hash_fails():
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    key = (identity["historical"]["cache_evidence_commit"], identity["historical"]["cache_evidence_path"])
    value = json.loads(artifacts[key])
    value["cache_manifest_sha256"] = "0" * 64
    payload = json.dumps(value, allow_nan=False).encode()
    changed_identity, changed_artifacts = _replace_artifact(identity, artifacts, "cache_evidence", payload)
    with pytest.raises(RWRReproductionError, match="cache-manifest hash"):
        validate_historical_provenance(
            changed_identity, artifact_reader=_reader(changed_artifacts)
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"{", "cannot parse historical cache evidence"),
        (b'{"cache_manifest_sha256":"a","cache_manifest_sha256":"b"}', "duplicate JSON key"),
    ],
)
def test_malformed_and_duplicate_key_attestation_json_fail(payload, match):
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    changed_identity, changed_artifacts = _replace_artifact(
        identity, artifacts, "cache_evidence", payload
    )
    with pytest.raises(RWRReproductionError, match=match):
        validate_historical_provenance(
            changed_identity, artifact_reader=_reader(changed_artifacts)
        )


def test_unarchived_attestation_needs_no_manifest_bytes_and_forbids_cache_use():
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    requested = []

    def reader(commit, path):
        requested.append((commit, path))
        return _reader(artifacts)(commit, path)

    validate_historical_provenance(identity, artifact_reader=reader)
    assert set(requested) == set(artifacts)
    changed = copy.deepcopy(identity)
    changed["historical_cache_used_by_current_run"] = True
    with pytest.raises(RWRReproductionError, match="must not use"):
        validate_historical_provenance(changed, artifact_reader=reader)


def test_archived_manifest_requires_fields_and_direct_matching_bytes():
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    missing = copy.deepcopy(identity)
    missing["cache_manifest_archived"] = True
    with pytest.raises(RWRReproductionError, match="fields are missing"):
        validate_historical_provenance(missing, artifact_reader=_reader(artifacts))

    wrong = copy.deepcopy(identity)
    wrong["cache_manifest_archived"] = True
    archive_bytes = b'{"not":"the historical manifest"}'
    archive_commit = wrong["source_e10_commit"]
    archive_path = "historical/cache/manifest.json"
    wrong["archived_cache_manifest"] = {
        "commit": archive_commit,
        "path": archive_path,
        "blob": hashlib.sha1(
            f"blob {len(archive_bytes)}\0".encode() + archive_bytes
        ).hexdigest(),
        "sha256": wrong["cache_manifest_sha256"],
    }
    artifacts[(archive_commit, archive_path)] = archive_bytes
    with pytest.raises(RWRReproductionError, match="direct SHA256 mismatch"):
        validate_historical_provenance(wrong, artifact_reader=_reader(artifacts))


def test_runtime_record_labels_attestation_without_direct_verification():
    identity = load_identity(repo_root=ROOT)
    record = build_valid_record(identity)
    assert record["cache_manifest_evidence"] == "committed_attestation"
    assert record["cache_manifest_archived"] is False
    assert record["historical_cache_used_by_current_run"] is False
    assert not any("verified" in key for key in record)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("images = 5000", "images = 5000.0", "dataset images.*exact integer"),
        ("classes = 171", "classes = 171.0", "dataset classes.*exact integer"),
        ("background_class = false", "background_class = 0", "background_class.*boolean"),
        ("pamr = false", "pamr = 0", "evaluation pamr.*boolean"),
        ("enabled = true", "enabled = 1", "RWR enabled.*boolean"),
        ("crop = [448, 448]", "crop = [448.0, 448]", "crop elements.*integers"),
        ("top_k = 12", "top_k = 12.0", "top_k.*integer"),
        ("alpha = 0.98", "alpha = 1", "alpha.*exact float"),
        ("max_iterations = 5000", "max_iterations = true", "max_iterations.*integer"),
        ("rtol = 0.00001", 'rtol = "0.00001"', "rtol.*exact float"),
    ],
)
def test_rwr_toml_exact_type_matrix(tmp_path, old, new, match):
    source = (ROOT / "evaluation_identities/e3_canonical_directed_rwr.toml").read_text()
    assert source.count(old) >= 1
    path = tmp_path / "identity.toml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(RWRReproductionError, match=match):
        load_identity(path, repo_root=ROOT)


def test_resolved_pair_validation_preserves_both_inputs_on_success_and_failure():
    identity = load_identity(repo_root=ROOT)
    e3, rwr = build_valid_resolved_config(identity)
    e3_before = copy.deepcopy(e3)
    rwr_before = copy.deepcopy(rwr)
    validate_resolved_config_pair(e3, rwr, identity)
    assert e3 == e3_before
    assert rwr == rwr_before
    assert "rwr" in rwr["evaluate"]

    rwr["evaluate"]["unknown_nested"] = {"items": [{"value": 1}]}
    e3_before = copy.deepcopy(e3)
    rwr_before = copy.deepcopy(rwr)
    with pytest.raises(RWRReproductionError, match="outside evaluate.rwr"):
        validate_resolved_config_pair(e3, rwr, identity)
    assert e3 == e3_before
    assert rwr == rwr_before
    assert "rwr" in rwr["evaluate"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("evaluated_images", 5000.0, "evaluated_images.*exact integer"),
        ("peak_gpu_bytes", True, "peak_gpu_bytes.*exact integer"),
        ("runtime_seconds", 1, "runtime_seconds.*floating-point"),
    ],
)
def test_historical_metrics_exact_types_fail(field, value, match):
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    key = (identity["historical"]["metrics_commit"], identity["historical"]["metrics_path"])
    document = json.loads(artifacts[key])
    document[identity["historical"]["metrics_record"]][field] = value
    payload = json.dumps(document, allow_nan=False).encode()
    changed_identity, changed_artifacts = _replace_artifact(
        identity, artifacts, "metrics", payload
    )
    with pytest.raises(RWRReproductionError, match=match):
        validate_historical_provenance(
            changed_identity, artifact_reader=_reader(changed_artifacts)
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("git_dirty", 0, "git_dirty.*boolean"),
        ("payload.rows.0.evaluated_images", 5000.0, "evaluated_images.*exact integer"),
        ("payload.rows.0.alpha", 1, "alpha.*floating-point"),
    ],
)
def test_historical_attestation_exact_types_fail(field, value, match):
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    key = (
        identity["historical"]["cache_evidence_commit"],
        identity["historical"]["cache_evidence_path"],
    )
    document = json.loads(artifacts[key])
    target = document
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    target[parts[-1]] = value
    payload = json.dumps(document, allow_nan=False).encode()
    changed_identity, changed_artifacts = _replace_artifact(
        identity, artifacts, "cache_evidence", payload
    )
    with pytest.raises(RWRReproductionError, match=match):
        validate_historical_provenance(
            changed_identity, artifact_reader=_reader(changed_artifacts)
        )


@pytest.mark.parametrize(
    "hash_name",
    ["full_sha256", "without_rwr_sha256", "evaluation_sha256"],
)
def test_each_complete_rwr_hash_mismatch_is_rejected(tmp_path, hash_name):
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    source = (ROOT / "evaluation_identities/e3_canonical_directed_rwr.toml").read_text()
    current = identity["resolved_configuration"][hash_name]
    path = tmp_path / f"{hash_name}.toml"
    path.write_text(source.replace(current, "0" * 64, 1))
    with pytest.raises(RWRReproductionError, match="complete RWR configuration mismatch"):
        validate_static_configuration(
            repo_root=ROOT,
            identity_path=path,
            artifact_reader=_reader(artifacts),
            check_git=False,
        )


def test_rwr_leaf_source_content_mutation_is_rejected(tmp_path):
    repo = _copy_rwr_static_repository(tmp_path / "repo")
    identity = load_identity(repo_root=ROOT)
    artifacts = build_valid_historical_artifacts(identity)
    config = repo / identity["canonical_config_path"]
    config.write_text(config.read_text() + "\nunknown_nested:\n  value: true\n")
    with pytest.raises(
        RWRReproductionError,
        match="canonical RWR config SHA256 mismatch|outside evaluate.rwr|source mismatch",
    ):
        validate_static_configuration(
            repo_root=repo,
            artifact_reader=_reader(artifacts),
            check_git=False,
        )


def test_canonical_scientific_literals_do_not_reappear_in_rwr_python():
    identity = load_identity(repo_root=ROOT)
    forbidden = {
        identity["identity_name"],
        identity["source_e10_commit"],
        identity["cache_source_commit"],
        identity["cache_manifest_sha256"],
        identity["checkpoint"]["sha256"],
        *(str(value) for value in identity["expected_metrics"].values()),
    }
    sources = [
        ROOT / "src/rwr_reproduction_identity.py",
        ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr/inference.py",
        ROOT / "src/open_vocabulary_segmentation/main.py",
    ]
    combined = "\n".join(path.read_text() for path in sources)
    assert all(literal not in combined for literal in forbidden)


# ---------------------------------------------------------------------------
# v3 structured-result format (talk2dino-canonical-rwr-result-v3)
# ---------------------------------------------------------------------------


def test_v3_record_with_no_parity_solver_summary_passes(tmp_path):
    path = tmp_path / "v3.json"
    _write_json(path, _record_v3())
    assert verify_result(path, source_kind="json").startswith("RWR REPRODUCTION PASS")


def test_v3_record_with_a_valid_parity_solver_summary_passes(tmp_path):
    parity = dict(_record()["solver_summary"])  # same shape, different (still valid) values
    parity["window_count"] = 42
    parity["converged_window_count"] = 42
    path = tmp_path / "v3.json"
    _write_json(path, _record_v3(parity_solver_summary=parity))
    assert verify_result(path, source_kind="json").startswith("RWR REPRODUCTION PASS")


def test_v3_record_reports_format_version_v3_in_its_pass_message(tmp_path):
    path = tmp_path / "v3.json"
    identity = load_identity(repo_root=ROOT)
    _write_json(path, build_valid_record_v3(identity))
    message = verify_result(path, source_kind="json")
    assert message.startswith("RWR REPRODUCTION PASS")


def test_v2_record_still_uses_the_v2_verifier_and_passes(tmp_path):
    # Historical provenance: a v2-format_version record must still verify
    # successfully under the current code, dispatched to _verify_record_v2.
    path = tmp_path / "v2.json"
    _write_json(path, _record())
    assert verify_result(path, source_kind="json").startswith("RWR REPRODUCTION PASS")


def test_v2_and_v3_records_of_the_same_underlying_run_both_pass_independently(tmp_path):
    v2_path, v3_path = tmp_path / "v2.json", tmp_path / "v3.json"
    _write_json(v2_path, _record())
    _write_json(v3_path, _record_v3())
    assert verify_result(v2_path, source_kind="json").startswith("RWR REPRODUCTION PASS")
    assert verify_result(v3_path, source_kind="json").startswith("RWR REPRODUCTION PASS")


def test_unknown_format_version_is_rejected(tmp_path):
    path = tmp_path / "unknown.json"
    record = _record()
    record["format_version"] = "talk2dino-canonical-rwr-result-v99"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="unsupported structured result format_version"):
        verify_result(path, source_kind="json")


def test_supported_format_versions_are_exactly_v2_and_v3():
    assert set(SUPPORTED_RESULT_FORMAT_VERSIONS) == {RESULT_FORMAT_VERSION, RESULT_FORMAT_VERSION_V3}


def test_v3_record_rejects_unknown_metric_source(tmp_path):
    path = tmp_path / "bad_source.json"
    record = _record_v3()
    record["metric_source"] = "some_rounded_approximate_source"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="metric_source"):
        verify_result(path, source_kind="json")


def test_v3_record_rejects_a_v2_metric_source_string_too(tmp_path):
    path = tmp_path / "wrong_source.json"
    record = _record_v3()
    record["metric_source"] = "rounded_2dp_via_mmseg_dataset_evaluate_summary"
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="metric_source"):
        verify_result(path, source_kind="json")


def test_v3_record_rejects_missing_metric_source_key(tmp_path):
    path = tmp_path / "missing_source.json"
    record = _record_v3()
    del record["metric_source"]
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="unexpected schema"):
        verify_result(path, source_kind="json")


def test_v3_record_rejects_an_unknown_extra_field(tmp_path):
    path = tmp_path / "extra_field.json"
    record = _record_v3()
    record["not_a_real_field"] = 1
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="unexpected schema"):
        verify_result(path, source_kind="json")


def test_v3_record_rejects_rounded_metrics_claiming_full_precision(tmp_path):
    # Two decimal places is far short of the identity's own
    # minimum_metric_decimal_places requirement -- a value that LOOKS like
    # mmseg's rounded-to-2dp natural summary must never pass as full
    # precision just because metric_source claims it is.
    path = tmp_path / "rounded.json"
    record = _record_v3()
    record["mIoU"] = 29.88
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="rounded-only"):
        verify_result(path, source_kind="json")


@pytest.mark.parametrize("field", ["window_count", "converged_window_count", "total_iterations"])
def test_v3_record_rejects_wrong_type_in_parity_solver_summary(tmp_path, field):
    parity = dict(_record()["solver_summary"])
    parity[field] = float(parity[field])  # wrong exact type: float where an exact int is required
    path = tmp_path / "bad_parity_type.json"
    _write_json(path, _record_v3(parity_solver_summary=parity))
    with pytest.raises(RWRReproductionError):
        verify_result(path, source_kind="json")


def test_v3_record_rejects_parity_solver_summary_with_wrong_key_set(tmp_path):
    parity = dict(_record()["solver_summary"])
    del parity["maximum_scaled_residual"]
    path = tmp_path / "bad_parity_keys.json"
    _write_json(path, _record_v3(parity_solver_summary=parity))
    with pytest.raises(RWRReproductionError, match="schema mismatch"):
        verify_result(path, source_kind="json")


def test_v3_record_rejects_wrong_image_count(tmp_path):
    path = tmp_path / "wrong_images.json"
    record = _record_v3()
    record["image_count"] = 1
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="image count"):
        verify_result(path, source_kind="json")


def test_v3_record_rejects_wrong_class_count(tmp_path):
    path = tmp_path / "wrong_classes.json"
    record = _record_v3()
    record["class_count"] = 1
    _write_json(path, record)
    with pytest.raises(RWRReproductionError, match="class count"):
        verify_result(path, source_kind="json")


def test_v3_record_v2_verify_record_rejects_a_v3_shaped_record_key_set(tmp_path):
    # verify_record dispatches purely on format_version -- a record with
    # v3's extra keys but a v2 format_version tag is rejected as an
    # unexpected v2 schema (never silently accepted through the wrong path).
    record = _record_v3()
    record["format_version"] = RESULT_FORMAT_VERSION  # v2 tag, but v3 key set
    with pytest.raises(RWRReproductionError, match="unexpected schema"):
        verify_record(record_with_decimals(record), load_identity(repo_root=ROOT))


def record_with_decimals(record):
    from decimal import Decimal
    return json.loads(json.dumps(record), parse_float=Decimal)


def test_verify_record_requires_a_mapping():
    with pytest.raises(RWRReproductionError, match="mapping"):
        verify_record("not a mapping", load_identity(repo_root=ROOT))


def test_result_keys_v3_is_exactly_v2_keys_plus_metric_source_and_parity_summary():
    from src.rwr_reproduction_identity import RESULT_KEYS
    assert RESULT_KEYS_V3 == RESULT_KEYS | {"metric_source", "parity_solver_summary"}
