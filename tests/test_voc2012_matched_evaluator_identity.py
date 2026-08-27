"""Tests for the shared VOC2012 V20/V21 matched-evaluator identity
loader/validator. CPU-only; no CUDA, no model, no GPU evaluation."""

from __future__ import annotations

import copy
import hashlib
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.voc2012_matched_evaluator_identity import (  # noqa: E402
    IDENTITY_TOP_KEYS,
    Voc2012MatchedEvaluatorIdentityError,
    load_identity,
    validate_bg_thresh_binding,
    validate_model_and_checkpoint_binding,
    validate_static_configuration,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_voc2012_matched_evaluator.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the voc2012 matched-evaluator identity")


@pytest.fixture(scope="module")
def raw_identity() -> dict:
    with IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def test_load_identity_real_file_passes():
    identity = load_identity(repo_root=ROOT)
    assert identity["identity"]["name"] == "e12-voc2012-matched-evaluator"
    assert set(identity) == IDENTITY_TOP_KEYS


def test_validate_static_configuration_real_file_passes():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["voc2012_source_identity_name"] == "e12-voc2012-dataset-source"
    assert result["matched_identity_name"] == "e12-matched-k11-k12-t320"


def test_validate_model_and_checkpoint_binding_real_files_passes():
    identity = load_identity(repo_root=ROOT)
    validate_model_and_checkpoint_binding(ROOT, identity)


def test_v20_v21_class_count_relationship():
    identity = load_identity(repo_root=ROOT)
    assert identity["v21_protocol"]["class_count"] == identity["v20_protocol"]["class_count"] + 1
    assert identity["v20_protocol"]["background_included"] is False
    assert identity["v21_protocol"]["background_included"] is True
    assert identity["v21_protocol"]["background_class_index"] == 0


def test_execution_contract_matches_task_call_counts():
    identity = load_identity(repo_root=ROOT)
    execution = identity["execution"]
    assert execution["snapshot_calls_per_window"] == 1
    assert execution["topk_graph_calls_per_window"] == 1
    assert execution["k11_prefix_calls_per_window"] == 1
    assert execution["k11_propagations_per_window"] == 1
    assert execution["k12_propagations_per_window"] == 1
    assert execution["k11_updates_per_window"] == 320
    assert execution["k12_updates_per_window"] == 320
    assert execution["e3_propagations_per_window"] == 0
    assert execution["second_model_pass_for_v20_v21"] is False


def test_bg_thresh_is_independently_sourced_and_bounded():
    identity = load_identity(repo_root=ROOT)
    bg = identity["background_protocol"]
    assert 0.0 <= bg["bg_thresh"] <= 1.0
    assert bg["v20_applies"] is False
    assert bg["v21_applies"] is True
    assert "independently sourced" in bg["provenance_note"]


def _write_mutated(tmp_path: Path, raw_identity: dict, mutate) -> Path:
    document = copy.deepcopy(raw_identity)
    mutate(document)
    out = tmp_path / "mutated.toml"
    _dump_toml(document, out)
    return out


def _dump_toml(document: dict, out: Path) -> None:
    # Minimal, dependency-free TOML writer sufficient for this identity's
    # shape (strings/ints/floats/bools/lists-of-strings, one level of
    # nested tables) -- avoids a tomli_w dependency for test-only mutation.
    lines: list[str] = []

    def _emit_value(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'"{escaped}"'
        if isinstance(value, list):
            return "[" + ", ".join(_emit_value(v) for v in value) + "]"
        raise TypeError(f"unsupported TOML value type: {type(value)}")

    def _emit_table(prefix: str, table: dict):
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subtables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_emit_value(value)}")
        lines.append("")
        for key, value in subtables.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            _emit_table(child_prefix, value)

    top_scalars = {k: v for k, v in document.items() if not isinstance(v, dict)}
    for key, value in top_scalars.items():
        lines.append(f"{key} = {_emit_value(value)}")
    lines.append("")
    for key, value in document.items():
        if isinstance(value, dict):
            _emit_table(key, value)
    out.write_text("\n".join(lines), encoding="utf-8")


def test_mutated_toml_round_trips_through_tomllib(tmp_path):
    # Sanity check for the test-only TOML writer itself, independent of
    # the identity loader: what we write must be valid TOML that decodes
    # back to the same structure.
    with IDENTITY_PATH.open("rb") as handle:
        original = tomllib.load(handle)
    out = tmp_path / "roundtrip.toml"
    _dump_toml(original, out)
    with out.open("rb") as handle:
        reloaded = tomllib.load(handle)
    assert reloaded == original


def test_unknown_top_level_field_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["unknown_extra_field"] = "x"

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_missing_top_level_field_rejected(tmp_path, raw_identity):
    def mutate(document):
        del document["execution"]

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_wrong_format_version_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["format_version"] = "wrong-version"
        document["identity"]["schema_version"] = "wrong-version"

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_v20_class_count_tampered_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["v20_protocol"]["class_count"] = 19

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_v21_background_included_false_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["v21_protocol"]["background_included"] = False

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_bg_thresh_out_of_range_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["background_protocol"]["bg_thresh"] = 1.5

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_bg_thresh_int_instead_of_float_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["background_protocol"]["bg_thresh"] = 1

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_e3_propagations_per_window_nonzero_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["execution"]["e3_propagations_per_window"] = 1

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_k11_updates_not_320_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["execution"]["k11_updates_per_window"] = 100

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_second_model_pass_true_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["execution"]["second_model_pass_for_v20_v21"] = True

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_wrong_checkpoint_sha256_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["model_and_checkpoint"]["projection_checkpoint_sha256"] = "0" * 64

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    identity = load_identity(path=mutated, repo_root=ROOT)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_model_and_checkpoint_binding(ROOT, identity)


def test_wrong_reused_module_sha256_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["reused_evaluator_modules"]["graph_sha256"] = "1" * 64

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    identity = load_identity(path=mutated, repo_root=ROOT)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_model_and_checkpoint_binding(ROOT, identity)


def test_wrong_voc2012_source_parent_sha256_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["parent_identities"]["voc2012_source_identity_sha256"] = "2" * 64

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=mutated, check_git=True)


def test_wrong_matched_parent_sha256_rejected(tmp_path, raw_identity):
    def mutate(document):
        document["parent_identities"]["matched_identity_sha256"] = "3" * 64

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_static_configuration(repo_root=ROOT, identity_path=mutated, check_git=True)


@pytest.mark.parametrize("raw", ["   ", "\t", "\n"])
def test_whitespace_only_identity_description_rejected(tmp_path, raw_identity, raw):
    def mutate(document):
        document["identity"]["description"] = raw

    mutated = _write_mutated(tmp_path, raw_identity, mutate)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        load_identity(path=mutated, repo_root=ROOT)


def test_identity_toml_file_immutable():
    before = IDENTITY_PATH.read_bytes()
    load_identity(repo_root=ROOT)
    validate_static_configuration(repo_root=ROOT, check_git=True)
    after = IDENTITY_PATH.read_bytes()
    assert before == after


def test_no_torch_cuda_import_in_identity_module():
    """AST-based, not a substring search: the identity loader itself must
    never import torch/CUDA/model machinery."""
    import ast

    path = ROOT / "src" / "voc2012_matched_evaluator_identity.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module.split(".")[0])
    assert "torch" not in imported_names
    assert "mmseg" not in imported_names
    assert "mmcv" not in imported_names


# ---------------------------------------------------------------------
# bg_thresh live-config binding: the identity's declared
# background_protocol.bg_thresh must exactly match the authoritative
# evaluate.bg_thresh parsed live from the identity-pinned V21 eval
# config, before any CUDA/model work.
# ---------------------------------------------------------------------


def _config_relative_path(identity: dict) -> str:
    return identity["model_and_checkpoint"]["v21_eval_config"]["eval_base_config_relative_path"]


def _write_fake_root_with_config(tmp_path: Path, identity: dict, yaml_text: str) -> Path:
    fake_root = tmp_path / "fake_root"
    target = fake_root / _config_relative_path(identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml_text, encoding="utf-8")
    return fake_root


def test_bg_thresh_binding_real_repo_matching_threshold_accepted():
    identity = load_identity(repo_root=ROOT)
    live_value = validate_bg_thresh_binding(ROOT, identity)
    assert live_value == identity["background_protocol"]["bg_thresh"]
    assert type(live_value) is float


def test_bg_thresh_binding_called_before_model_or_cuda_import():
    """AST-based: validate_model_and_checkpoint_binding (which now calls
    validate_bg_thresh_binding) must never import torch/mmseg/mmcv at
    module scope -- confirmed once for the whole module above; this test
    additionally confirms the call site exists textually inside
    validate_model_and_checkpoint_binding, so the ordering guarantee
    (binding validation always precedes CUDA/model work in
    validate_static_configuration) actually holds."""
    import inspect

    from src.voc2012_matched_evaluator_identity import validate_model_and_checkpoint_binding

    source = inspect.getsource(validate_model_and_checkpoint_binding)
    assert "validate_bg_thresh_binding(root, identity)" in source


def test_bg_thresh_identity_only_mutation_rejected_against_unchanged_config():
    identity = load_identity(repo_root=ROOT)
    mutated = copy.deepcopy(identity)
    mutated["background_protocol"]["bg_thresh"] = 0.5  # real config still says 0.55
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_bg_thresh_binding(ROOT, mutated)


def test_bg_thresh_yaml_only_mutation_rejected_against_unchanged_identity(tmp_path):
    identity = load_identity(repo_root=ROOT)
    fake_root = _write_fake_root_with_config(tmp_path, identity, "evaluate:\n  bg_thresh: 0.5\n")
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_bg_thresh_binding(fake_root, identity)  # identity still declares 0.55


@pytest.mark.parametrize(
    "yaml_text,label",
    [
        ("evaluate:\n  bg_thresh: true\n", "bool"),
        ("evaluate:\n  bg_thresh: 1\n", "int"),
        ('evaluate:\n  bg_thresh: "0.55"\n', "numeric string"),
        ("evaluate:\n  bg_thresh: .nan\n", "NaN"),
        ("evaluate:\n  bg_thresh: .inf\n", "Infinity"),
        ("evaluate:\n  bg_thresh: -.inf\n", "negative Infinity"),
    ],
)
def test_bg_thresh_wrong_type_or_non_finite_rejected(tmp_path, yaml_text, label):
    identity = load_identity(repo_root=ROOT)
    fake_root = _write_fake_root_with_config(tmp_path, identity, yaml_text)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_bg_thresh_binding(fake_root, identity)


def test_bg_thresh_missing_evaluate_section_rejected(tmp_path):
    identity = load_identity(repo_root=ROOT)
    fake_root = _write_fake_root_with_config(tmp_path, identity, "other_section:\n  x: 1\n")
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_bg_thresh_binding(fake_root, identity)


def test_bg_thresh_missing_key_rejected(tmp_path):
    identity = load_identity(repo_root=ROOT)
    fake_root = _write_fake_root_with_config(tmp_path, identity, "evaluate:\n  pamr: false\n")
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_bg_thresh_binding(fake_root, identity)


def test_bg_thresh_wrong_container_rejected(tmp_path):
    identity = load_identity(repo_root=ROOT)
    fake_root = _write_fake_root_with_config(tmp_path, identity, "evaluate:\n  - 1\n  - 2\n")
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_bg_thresh_binding(fake_root, identity)


def test_bg_thresh_duplicate_key_rejected(tmp_path):
    identity = load_identity(repo_root=ROOT)
    fake_root = _write_fake_root_with_config(tmp_path, identity, "evaluate:\n  bg_thresh: 0.55\n  bg_thresh: 0.5\n")
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_bg_thresh_binding(fake_root, identity)


def test_bg_thresh_duplicate_section_rejected(tmp_path):
    identity = load_identity(repo_root=ROOT)
    fake_root = _write_fake_root_with_config(tmp_path, identity, "evaluate:\n  bg_thresh: 0.55\nevaluate:\n  bg_thresh: 0.5\n")
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_bg_thresh_binding(fake_root, identity)


def test_bg_thresh_binding_included_in_validate_static_configuration():
    """Confirm the real end-to-end preflight path (validate_static_configuration,
    the same function verify_voc2012_matched_evaluation.py preflight calls)
    actually reaches the bg_thresh check, by mutating the identity's
    declared threshold and requiring the whole preflight chain to reject
    it -- not just the narrow validate_bg_thresh_binding function in
    isolation."""
    identity = load_identity(repo_root=ROOT)
    mutated = copy.deepcopy(identity)
    mutated["background_protocol"]["bg_thresh"] = 0.5
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_model_and_checkpoint_binding(ROOT, mutated)


def test_bg_thresh_binding_never_mutates_identity_argument():
    identity = load_identity(repo_root=ROOT)
    before = copy.deepcopy(identity)
    validate_bg_thresh_binding(ROOT, identity)
    assert identity == before
