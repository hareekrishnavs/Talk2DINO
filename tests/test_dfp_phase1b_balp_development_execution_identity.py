"""Tests for the E23 BALP development-execution authorization identity
loader (`src.dfp_phase1b_balp_development_execution_identity`).
Exercises the real committed identity plus a battery of adversarial
mutations, verifying every one is rejected. No torch/model/dataset/CUDA
access -- pure TOML/JSON/filesystem/hash validation."""

from __future__ import annotations

import copy
import shutil
import tomllib
from pathlib import Path

import pytest

import src.dfp_phase1b_balp_development_execution_identity as e23

REPO_ROOT = Path(__file__).resolve().parent.parent
E = e23.DfpPhase1bBalpDevelopmentExecutionIdentityError


def _load_document() -> dict:
    with open(REPO_ROOT / e23.IDENTITY_RELATIVE_PATH, "rb") as fh:
        return tomllib.load(fh)


def _write_toml(document: dict, path: Path) -> None:
    # Minimal, dependency-free TOML writer sufficient for this document's
    # own shape (flat/nested string/int/bool/list-of-scalar values only,
    # no need for a third-party TOML writer in the test suite).
    def _dump_value(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, list):
            return "[" + ", ".join(_dump_value(v) for v in value) + "]"
        raise TypeError(f"unsupported value type for test TOML writer: {type(value)}")

    lines: list[str] = []

    def _walk(prefix: str, table: dict) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subtables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {_dump_value(v)}")
        lines.append("")
        for k, v in subtables.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            _walk(new_prefix, v)

    _walk("", document)
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(scope="module")
def valid_document():
    return _load_document()


def test_real_committed_identity_loads_and_validates():
    doc = e23.load_identity(repo_root=REPO_ROOT)
    assert doc["identity"]["name"] == "e23-dfp-phase1b-balp-development-execution"
    assert doc["population"]["development_image_count"] == 350
    assert doc["population"]["window_count"] == 769


def test_repository_gate_passes_at_head():
    import subprocess

    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    e23.verify_repository_gate(REPO_ROOT, head)  # must not raise


def test_repository_gate_rejects_unrelated_commit():
    with pytest.raises(E, match="not an ancestor"):
        e23.verify_repository_gate(REPO_ROOT, "0" * 40)


def _mutate_and_load(tmp_path, valid_document, mutator, *, match=None):
    doc = copy.deepcopy(valid_document)
    mutator(doc)
    target = tmp_path / "e23_mutated.toml"
    _write_toml(doc, target)
    with pytest.raises(E, match=match):
        e23.load_identity(target, repo_root=REPO_ROOT)


def test_rejects_missing_top_level_key(tmp_path, valid_document):
    _mutate_and_load(tmp_path, valid_document, lambda d: d.pop("precision"), match="wrong key set")


def test_rejects_extra_top_level_key(tmp_path, valid_document):
    _mutate_and_load(tmp_path, valid_document, lambda d: d.__setitem__("unexpected_extra_key", "x"), match="wrong key set")


def test_rejects_extra_dependency_binding_key(tmp_path, valid_document):
    def mutate(d):
        d["dependency_bindings"]["e13_protocol"]["extra_unexpected_field"] = "x"

    _mutate_and_load(tmp_path, valid_document, mutate, match="wrong key set")


def test_rejects_missing_dependency_binding(tmp_path, valid_document):
    _mutate_and_load(tmp_path, valid_document, lambda d: d["dependency_bindings"].pop("e13_protocol"), match="wrong key set")


def test_rejects_wrong_hash(tmp_path, valid_document):
    def mutate(d):
        d["dependency_bindings"]["e13_protocol"]["sha256"] = "0" * 64

    _mutate_and_load(tmp_path, valid_document, mutate, match="SHA-256 mismatch")


def test_rejects_path_traversal(tmp_path, valid_document):
    def mutate(d):
        d["dependency_bindings"]["e13_protocol"]["relative_path"] = "../etc/passwd"

    _mutate_and_load(tmp_path, valid_document, mutate, match="unsafe or unresolvable")


def test_rejects_absolute_dependency_path(tmp_path, valid_document):
    def mutate(d):
        d["dependency_bindings"]["e13_protocol"]["relative_path"] = "/etc/passwd"

    _mutate_and_load(tmp_path, valid_document, mutate, match="unsafe or unresolvable")


def test_rejects_gpu_count_bool_as_int(tmp_path, valid_document):
    def mutate(d):
        d["resource_contract"]["gpu_count"] = True

    _mutate_and_load(tmp_path, valid_document, mutate, match="must be an int")


def test_rejects_authorization_gpu_count_bool_as_int(tmp_path, valid_document):
    def mutate(d):
        d["authorization_boundary"]["gpu_count_authorized"] = True

    _mutate_and_load(tmp_path, valid_document, mutate, match="must be an int")


def test_rejects_development_image_count_349(tmp_path, valid_document):
    def mutate(d):
        d["population"]["development_image_count"] = 349

    _mutate_and_load(tmp_path, valid_document, mutate, match="development_image_count")


def test_rejects_development_image_count_351(tmp_path, valid_document):
    def mutate(d):
        d["population"]["development_image_count"] = 351

    _mutate_and_load(tmp_path, valid_document, mutate, match="development_image_count")


def test_rejects_candidate_count_15(tmp_path, valid_document):
    def mutate(d):
        d["population"]["candidate_count"] = 15

    _mutate_and_load(tmp_path, valid_document, mutate, match="candidate_count")


def test_rejects_candidate_count_17(tmp_path, valid_document):
    def mutate(d):
        d["population"]["candidate_count"] = 17

    _mutate_and_load(tmp_path, valid_document, mutate, match="candidate_count")


def test_rejects_gpu_count_0(tmp_path, valid_document):
    def mutate(d):
        d["resource_contract"]["gpu_count"] = 0
        d["authorization_boundary"]["gpu_count_authorized"] = 0

    _mutate_and_load(tmp_path, valid_document, mutate, match="gpu_count")


def test_rejects_gpu_count_2(tmp_path, valid_document):
    def mutate(d):
        d["resource_contract"]["gpu_count"] = 2
        d["authorization_boundary"]["gpu_count_authorized"] = 2

    _mutate_and_load(tmp_path, valid_document, mutate, match="gpu_count")


def test_rejects_wrong_gpu_model(tmp_path, valid_document):
    def mutate(d):
        d["resource_contract"]["gpu_model"] = "a100_80gb"

    _mutate_and_load(tmp_path, valid_document, mutate, match="gpu_model")


def test_rejects_wall_time_over_five_hours(tmp_path, valid_document):
    def mutate(d):
        d["resource_contract"]["max_wall_time"] = "06:00:00"

    _mutate_and_load(tmp_path, valid_document, mutate, match="max_wall_time")


def test_rejects_full_5000_execution_authorized_true(tmp_path, valid_document):
    def mutate(d):
        d["authorization_boundary"]["full_5000_image_evaluation_authorized"] = True

    _mutate_and_load(tmp_path, valid_document, mutate, match="full_5000_image_evaluation_authorized")


def test_rejects_confirmation_execution_authorized_true(tmp_path, valid_document):
    def mutate(d):
        d["authorization_boundary"]["confirmation_execution_authorized"] = True

    _mutate_and_load(tmp_path, valid_document, mutate, match="confirmation_execution_authorized")


def test_rejects_exact_masked_forward_required_true(tmp_path, valid_document):
    def mutate(d):
        d["authorization_boundary"]["exact_masked_forward_provenance_authorized"] = True

    _mutate_and_load(tmp_path, valid_document, mutate, match="exact_masked_forward_provenance_authorized")


def test_rejects_candidate_specific_recurrence_authorized_true(tmp_path, valid_document):
    def mutate(d):
        d["authorization_boundary"]["candidate_specific_recurrence_authorized"] = True

    _mutate_and_load(tmp_path, valid_document, mutate, match="candidate_specific_recurrence_authorized")


def test_rejects_overwrite_authorization_enabled(tmp_path, valid_document):
    def mutate(d):
        d["authorization_boundary"]["overwrite_of_completed_development_result_authorized"] = True

    _mutate_and_load(tmp_path, valid_document, mutate, match="overwrite_of_completed_development_result_authorized")


def test_rejects_real_execute_cli_authorized_true(tmp_path, valid_document):
    def mutate(d):
        d["authorization_boundary"]["real_execute_cli_implementation_authorized_by_this_identity"] = True

    _mutate_and_load(tmp_path, valid_document, mutate, match="real_execute_cli_implementation_authorized_by_this_identity")


def test_rejects_stale_runner_hash(tmp_path, valid_document):
    def mutate(d):
        d["dependency_bindings"]["balp_execution_runner"]["sha256"] = "1" * 64

    _mutate_and_load(tmp_path, valid_document, mutate, match="SHA-256 mismatch")


def test_rejects_window_universe_image_count_tampered(tmp_path, valid_document):
    def mutate(d):
        d["dependency_bindings"]["development_window_universe"]["image_count"] = 349

    _mutate_and_load(tmp_path, valid_document, mutate, match="image_count")


def test_rejects_window_universe_window_count_tampered(tmp_path, valid_document):
    def mutate(d):
        d["dependency_bindings"]["development_window_universe"]["window_count"] = 770

    _mutate_and_load(tmp_path, valid_document, mutate, match="window_count")


def test_rejects_window_universe_digest_tampered(tmp_path, valid_document):
    def mutate(d):
        d["dependency_bindings"]["development_window_universe"]["canonical_record_digest"] = "0" * 64

    _mutate_and_load(tmp_path, valid_document, mutate, match="canonical_record_digest")


def test_rejects_population_window_count_disagreeing_with_binding(tmp_path, valid_document):
    def mutate(d):
        d["population"]["window_count"] = 770

    _mutate_and_load(tmp_path, valid_document, mutate, match="window_count")


def test_rejects_accepted_modes_with_forbidden_mode(tmp_path, valid_document):
    def mutate(d):
        d["checkpoint_resume_contract"]["accepted_modes"] = ["fresh", "resume", "debug_subset"]

    _mutate_and_load(tmp_path, valid_document, mutate, match="accepted_modes")


def test_rejects_wrong_required_ancestor_commit(tmp_path, valid_document):
    def mutate(d):
        d["identity"]["required_ancestor_commit"] = "1" * 40

    _mutate_and_load(tmp_path, valid_document, mutate, match="required_ancestor_commit")


def test_rejects_wrong_successor_of(tmp_path, valid_document):
    def mutate(d):
        d["identity"]["successor_of"] = "e21-dfp-phase1b-provenance-features"

    _mutate_and_load(tmp_path, valid_document, mutate, match="successor_of")
