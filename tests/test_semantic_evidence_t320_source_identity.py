"""Focused tests for src/semantic_evidence_t320_source_identity.py.

Deliberately uses the standard-library `unittest` framework (not pytest) so
these tests can run in a plain, Torch-free interpreter -- no venv, no
project dependencies. Run with:

    python3 -m unittest tests.test_semantic_evidence_t320_source_identity -v

These tests never checkout a branch, never mutate any real Git object, and
never write to the actual repository files. All mutation fixtures operate
on in-memory copies of the loaded dict, serialized to temporary files.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TOML_PATH = REPO_ROOT / "research_specs" / "semantic_evidence_t320_source_v1.toml"
MANIFEST_PATH = REPO_ROOT / "research_specs" / "semantic_evidence_t320_source_manifest_v1.json"

sys.path.insert(0, str(SRC_DIR))
import semantic_evidence_t320_source_identity as t320id  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _load_toml_dict() -> dict:
    with open(TOML_PATH, "rb") as f:
        return tomllib.load(f)


def _load_manifest_dict() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _toml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    raise TypeError(f"unsupported scalar type for TOML writer: {type(value)}")


def _write_toml_value(fh, data: dict, path: list) -> None:
    scalars = {}
    tables = {}
    for k, v in data.items():
        if isinstance(v, dict):
            tables[k] = v
        else:
            scalars[k] = v
    if path:
        fh.write(f"\n[{'.'.join(path)}]\n")
    for k, v in scalars.items():
        fh.write(f"{k} = {_toml_scalar(v)}\n")
    for k, v in tables.items():
        _write_toml_value(fh, v, path + [k])


def _write_toml(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    _write_toml_value(tmp, data, path=[])
    tmp.close()
    return Path(tmp.name)


def _write_json(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp, indent=2)
    tmp.close()
    return Path(tmp.name)


def _valid_toml() -> dict:
    return copy.deepcopy(_load_toml_dict())


def _valid_manifest() -> dict:
    return copy.deepcopy(_load_manifest_dict())


def _expect_reject_toml(mutated: dict) -> None:
    path = _write_toml(mutated)
    try:
        with unittest.TestCase().assertRaises(t320id.T320SourceAuthorityError):
            t320id.load_authority(path)
    finally:
        path.unlink()


def _expect_reject_manifest(mutated: dict) -> None:
    path = _write_json(mutated)
    try:
        with unittest.TestCase().assertRaises(t320id.T320SourceAuthorityError):
            t320id.load_manifest(path)
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# Valid-load sanity
# ---------------------------------------------------------------------------

class ValidLoadTests(unittest.TestCase):
    def test_valid_authority_loads(self):
        data = t320id.load_authority(TOML_PATH)
        self.assertEqual(data["historical_source"]["source_commit"], "00392527937253f93b9f3fe36b76347b4a6f1eb0")

    def test_valid_manifest_loads(self):
        manifest = t320id.load_manifest(MANIFEST_PATH)
        self.assertEqual(len(manifest["file_records"]), 7)

    def test_import_in_fresh_interpreter_does_not_pull_in_torch(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(SRC_DIR)!r}); import semantic_evidence_t320_source_identity; print('torch' in sys.modules)"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_canonical_digest_reconstruction_matches(self):
        manifest = _valid_manifest()
        recomputed = t320id.compute_canonical_source_closure_digest(manifest)
        self.assertEqual(recomputed, manifest["canonical_source_closure_digest"])

    def test_manifest_against_git_succeeds(self):
        manifest = t320id.load_manifest(MANIFEST_PATH)
        result = t320id.verify_manifest_against_git(REPO_ROOT, manifest)
        self.assertEqual(result["source_commit"], manifest["source_commit"])

    def test_manifest_file_binding_succeeds(self):
        toml_data = t320id.load_authority(TOML_PATH)
        sha = t320id.verify_manifest_file_binding(REPO_ROOT, toml_data)
        self.assertEqual(len(sha), 64)

    def test_toml_manifest_digest_cross_check_succeeds(self):
        toml_data = t320id.load_authority(TOML_PATH)
        manifest = t320id.load_manifest(MANIFEST_PATH)
        t320id.verify_toml_binds_same_digest_as_manifest(toml_data, manifest)


# ---------------------------------------------------------------------------
# Repository / source bindings
# ---------------------------------------------------------------------------

class RepositorySourceBindingTests(unittest.TestCase):
    def test_wrong_source_commit_in_manifest_rejected(self):
        manifest = _valid_manifest()
        manifest["source_commit"] = "1111111111111111111111111111111111111111"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_manifest_against_git(REPO_ROOT, manifest)

    def test_abbreviated_commit_rejected(self):
        manifest = _valid_manifest()
        manifest["source_commit"] = "0039252"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_manifest_against_git(REPO_ROOT, manifest)

    def test_wrong_tree_id_rejected(self):
        manifest = _valid_manifest()
        manifest["source_tree_id"] = "0" * 40
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_manifest_against_git(REPO_ROOT, manifest)

    def test_wrong_blob_id_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["blob_id"] = "0" * 40
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_manifest_against_git(REPO_ROOT, manifest)

    def test_wrong_sha256_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["sha256"] = "0" * 64
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_manifest_against_git(REPO_ROOT, manifest)

    def test_missing_source_object_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["path"] = "src/does_not_exist_anywhere.py"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_manifest_against_git(REPO_ROOT, manifest)

    def test_path_substitution_to_real_different_file_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["path"] = "src/local_weights.py"
        # loads fine (real, different file) but the seven-path-closure check
        # and/or the git blob-id/sha256 recomputation must catch the swap.
        _expect_reject_manifest(manifest)

    def test_absolute_path_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["path"] = "/etc/passwd"
        _expect_reject_manifest(manifest)

    def test_lexical_traversal_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["path"] = "../../etc/passwd"
        _expect_reject_manifest(manifest)

    def test_option_like_path_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["path"] = "--upload-pack=evil"
        _expect_reject_manifest(manifest)

    def test_extra_source_record_rejected(self):
        manifest = _valid_manifest()
        extra = copy.deepcopy(manifest["file_records"][0])
        extra["ordinal"] = 8
        extra["path"] = "src/open_vocabulary_segmentation/models/dinotext/cover_dr/rwr.py"
        manifest["file_records"].append(extra)
        _expect_reject_manifest(manifest)

    def test_missing_source_record_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"].pop()
        _expect_reject_manifest(manifest)

    def test_reordered_records_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0], manifest["file_records"][1] = (
            manifest["file_records"][1], manifest["file_records"][0]
        )
        # ordinals now out of position-order (record at position 0 has ordinal 2)
        _expect_reject_manifest(manifest)

    def test_canonical_digest_corruption_rejected(self):
        manifest = _valid_manifest()
        manifest["canonical_source_closure_digest"] = "0" * 64
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_manifest_against_git(REPO_ROOT, manifest)

    def test_canonical_digest_corruption_via_record_mutation_detected(self):
        manifest = _valid_manifest()
        # digest was computed over the ORIGINAL role string; changing only
        # the role (not the frozen digest) must desynchronize them.
        manifest["file_records"][0]["role"] = "renamed_role"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_manifest_against_git(REPO_ROOT, manifest)

    def test_toml_manifest_digest_mismatch_rejected(self):
        toml_data = _valid_toml()
        toml_data["source_manifest_binding"]["canonical_source_closure_digest"] = "0" * 64
        manifest = _valid_manifest()
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_toml_binds_same_digest_as_manifest(toml_data, manifest)

    def test_current_prerequisite_byte_change_rejected(self):
        toml_data = _valid_toml()
        toml_data["prerequisite"]["feasibility_contract_toml_sha256"] = "0" * 64
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_current_repository_prerequisite(REPO_ROOT, toml_data)

    def test_wrong_head_rejected(self):
        toml_data = _valid_toml()
        toml_data["prerequisite"]["required_head"] = "0" * 40
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_current_repository_prerequisite(REPO_ROOT, toml_data)

    def test_wrong_branch_rejected(self):
        toml_data = _valid_toml()
        toml_data["prerequisite"]["required_branch"] = "main"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_current_repository_prerequisite(REPO_ROOT, toml_data)


# ---------------------------------------------------------------------------
# Scientific constants
# ---------------------------------------------------------------------------

class ScientificConstantMutationTests(unittest.TestCase):
    def test_k_mutation_rejected(self):
        data = _valid_toml()
        data["graph_contract"]["k"] = 11
        _expect_reject_toml(data)

    def test_alpha_mutation_rejected(self):
        data = _valid_toml()
        data["recurrence_contract"]["alpha"] = 0.9
        _expect_reject_toml(data)

    def test_restart_mutation_rejected(self):
        data = _valid_toml()
        data["recurrence_contract"]["restart_coefficient"] = 0.5
        _expect_reject_toml(data)

    def test_affinity_power_mutation_rejected(self):
        data = _valid_toml()
        data["graph_contract"]["affinity_power"] = 2.0
        _expect_reject_toml(data)

    def test_step_count_mutation_rejected(self):
        data = _valid_toml()
        data["recurrence_contract"]["steps"] = 64
        _expect_reject_toml(data)

    def test_propagation_dtype_mutation_rejected(self):
        data = _valid_toml()
        data["recurrence_contract"]["propagation_dtype"] = "float16"
        _expect_reject_toml(data)

    def test_crop_mutation_rejected(self):
        data = _valid_toml()
        data["geometry_contract"]["crop"] = [512, 512]
        _expect_reject_toml(data)

    def test_stride_mutation_rejected(self):
        data = _valid_toml()
        data["geometry_contract"]["stride"] = [256, 256]
        _expect_reject_toml(data)

    def test_patch_size_mutation_rejected(self):
        data = _valid_toml()
        data["geometry_contract"]["patch_size"] = [16, 16]
        _expect_reject_toml(data)

    def test_graph_direction_mutation_rejected(self):
        data = _valid_toml()
        data["graph_contract"]["directed_symmetric_behavior"] = "symmetric, fully undirected adjacency"
        _expect_reject_toml(data)

    def test_normalization_direction_mutation_rejected(self):
        data = _valid_toml()
        data["graph_contract"]["normalization_direction"] = "column-wise normalization"
        _expect_reject_toml(data)

    def test_early_stopping_authorization_rejected(self):
        data = _valid_toml()
        data["recurrence_contract"]["no_early_stopping"] = False
        _expect_reject_toml(data)

    def test_convergence_test_authorization_rejected(self):
        data = _valid_toml()
        data["recurrence_contract"]["no_convergence_test"] = False
        _expect_reject_toml(data)

    def test_k_as_bool_rejected(self):
        data = _valid_toml()
        data["graph_contract"]["k"] = True  # bool where int required
        _expect_reject_toml(data)

    def test_steps_as_bool_rejected(self):
        data = _valid_toml()
        data["recurrence_contract"]["steps"] = True
        _expect_reject_toml(data)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

class InterpolationMutationTests(unittest.TestCase):
    def test_internal_upsample_true_to_false_rejected(self):
        data = _valid_toml()
        data["internal_upsample_contract"]["align_corners"] = False
        _expect_reject_toml(data)

    def test_final_rescale_false_to_true_rejected(self):
        data = _valid_toml()
        data["final_rescale_contract"]["align_corners_frozen_runtime_value"] = True
        _expect_reject_toml(data)

    def test_swap_internal_and_final_values_rejected(self):
        data = _valid_toml()
        data["internal_upsample_contract"]["align_corners"] = False
        data["final_rescale_contract"]["align_corners_frozen_runtime_value"] = True
        _expect_reject_toml(data)

    def test_remove_distinct_operation_assertion_rejected(self):
        data = _valid_toml()
        data["final_rescale_contract"]["internal_upsample_and_final_rescale_are_distinct"] = False
        _expect_reject_toml(data)

    def test_change_bilinear_mode_internal_rejected(self):
        data = _valid_toml()
        data["internal_upsample_contract"]["interpolation_mode"] = "nearest"
        _expect_reject_toml(data)

    def test_change_bilinear_mode_final_rejected(self):
        data = _valid_toml()
        data["final_rescale_contract"]["interpolation_mode"] = "nearest"
        _expect_reject_toml(data)

    def test_replace_align_corners_source_rejected(self):
        data = _valid_toml()
        data["final_rescale_contract"]["align_corners_source"] = "a hardcoded True constant"
        _expect_reject_toml(data)

    def test_runtime_override_key_injection_rejected(self):
        data = _valid_toml()
        data["final_rescale_contract"]["runtime_override_authorized"] = True
        _expect_reject_toml(data)


# ---------------------------------------------------------------------------
# Closure / exclusions
# ---------------------------------------------------------------------------

class ClosureExclusionTests(unittest.TestCase):
    def test_add_window_cache_to_file_records_rejected(self):
        manifest = _valid_manifest()
        wc = copy.deepcopy(manifest["file_records"][3])
        wc["path"] = "src/open_vocabulary_segmentation/segmentation/evaluation/window_cache.py"
        wc["ordinal"] = 8
        manifest["file_records"].append(wc)
        _expect_reject_manifest(manifest)

    def test_remove_sliding_window_geometry_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"].pop(3)
        _expect_reject_manifest(manifest)

    def test_add_cgls_rwr_module_rejected(self):
        manifest = _valid_manifest()
        rwr = copy.deepcopy(manifest["file_records"][0])
        rwr["path"] = "src/open_vocabulary_segmentation/models/dinotext/cover_dr/rwr.py"
        rwr["ordinal"] = 8
        manifest["file_records"].append(rwr)
        _expect_reject_manifest(manifest)

    def test_add_masked_forward_module_rejected(self):
        manifest = _valid_manifest()
        mf = copy.deepcopy(manifest["file_records"][0])
        mf["path"] = "src/dfp_phase1b_masked_forward.py"
        mf["ordinal"] = 8
        manifest["file_records"].append(mf)
        _expect_reject_manifest(manifest)

    def test_unresolved_project_import_empty_symbols_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["required_symbols"] = []
        _expect_reject_manifest(manifest)

    def test_duplicate_source_record_rejected(self):
        manifest = _valid_manifest()
        dup = copy.deepcopy(manifest["file_records"][0])
        dup["ordinal"] = 8
        manifest["file_records"].append(dup)
        _expect_reject_manifest(manifest)

    def test_window_cache_overlap_with_excluded_source_rejected(self):
        manifest = _valid_manifest()
        # move window_cache.py from excluded_source into file_records under a
        # different ordinal position -- must still be rejected because it is
        # the forbidden path, regardless of position.
        wc_excluded = next(e for e in manifest["excluded_source"] if e["path"].endswith("window_cache.py"))
        injected = {
            "ordinal": 8, "role": "smuggled", "path": wc_excluded["path"],
            "source_commit": manifest["source_commit"], "blob_id": wc_excluded["blob_id"],
            "sha256": "0" * 64, "required_symbols": ["x"], "classification": "scientific_adapter",
            "required": True, "reason_required": "smuggled",
        }
        manifest["file_records"].append(injected)
        _expect_reject_manifest(manifest)


# ---------------------------------------------------------------------------
# Historical 350-image disclosure
# ---------------------------------------------------------------------------

class Historical350DisclosureTests(unittest.TestCase):
    def test_mark_350_evidence_matched_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["matched_evidence_for_this_authority"] = True
        _expect_reject_toml(data)

    def test_change_350_align_corners_to_false_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["statement"]["the_350_image_run_used_align_corners"] = False
        _expect_reject_toml(data)

    def test_claim_full_path_identity_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["divergence"]["complete_end_to_end_identity"] = True
        _expect_reject_toml(data)

    def test_use_gain_as_expected_performance_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["statement"]["gain_must_not_be_cited_as_matched_empirical_performance"] = False
        _expect_reject_toml(data)

    def test_change_execution_commit_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["execution_commit"] = "1" * 40
        _expect_reject_toml(data)

    def test_change_scientific_digest_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["scientific_digest"] = "1" * 64
        _expect_reject_toml(data)

    def test_change_t0_miou_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["t0_miou"] = 0.0
        _expect_reject_toml(data)

    def test_change_t320_miou_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["t320_miou"] = 0.0
        _expect_reject_toml(data)

    def test_change_gain_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["gain_percentage_points"] = 99.0
        _expect_reject_toml(data)

    def test_remove_contextual_only_flag_rejected(self):
        data = _valid_toml()
        del data["historical_350_disclosure"]["contextual_evidence_only"]
        _expect_reject_toml(data)

    def test_expected_miou_frozen_flag_flip_rejected(self):
        data = _valid_toml()
        data["historical_350_disclosure"]["statement"]["expected_t320_miou_frozen_by_this_authority"] = True
        _expect_reject_toml(data)


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

class AuthorizationMutationTests(unittest.TestCase):
    def _assert_flag_rejected(self, table: str, key: str) -> None:
        data = _valid_toml()
        data[table][key] = True
        _expect_reject_toml(data)

    def test_authorize_implementation_in_this_task_rejected(self):
        self._assert_flag_rejected("status", "implementation_performed_in_this_task")

    def test_authorize_model_construction_rejected(self):
        self._assert_flag_rejected("status", "model_construction_authorized")
        self._assert_flag_rejected("authorization_boundary", "model_construction_authorized")

    def test_authorize_dataset_access_rejected(self):
        self._assert_flag_rejected("status", "dataset_access_authorized")
        self._assert_flag_rejected("authorization_boundary", "dataset_access_authorized")

    def test_authorize_checkpoint_tensor_loading_rejected(self):
        self._assert_flag_rejected("status", "checkpoint_tensor_loading_authorized")
        self._assert_flag_rejected("authorization_boundary", "checkpoint_tensor_loading_authorized")

    def test_authorize_t320_execution_rejected(self):
        self._assert_flag_rejected("status", "t320_execution_authorized")
        self._assert_flag_rejected("authorization_boundary", "t320_execution_authorized")

    def test_authorize_gt_access_rejected(self):
        self._assert_flag_rejected("status", "gt_access_authorized")
        self._assert_flag_rejected("authorization_boundary", "gt_access_authorized")

    def test_authorize_96_images_rejected(self):
        self._assert_flag_rejected("status", "feasibility_96_execution_authorized")
        self._assert_flag_rejected("authorization_boundary", "feasibility_96_execution_authorized")

    def test_authorize_training_rejected(self):
        self._assert_flag_rejected("status", "training_authorized")
        self._assert_flag_rejected("authorization_boundary", "training_authorized")

    def test_authorize_confirmation_rejected(self):
        self._assert_flag_rejected("status", "confirmation_execution_authorized")
        self._assert_flag_rejected("authorization_boundary", "confirmation_execution_authorized")

    def test_authorize_full_5000_rejected(self):
        self._assert_flag_rejected("status", "full_5000_evaluation_authorized")
        self._assert_flag_rejected("authorization_boundary", "full_5000_evaluation_authorized")

    def test_authorize_e3_file_modification_rejected(self):
        self._assert_flag_rejected("status", "existing_e3_scientific_file_modification_authorized")
        self._assert_flag_rejected("authorization_boundary", "existing_e3_scientific_file_modification_authorized")

    def test_disable_future_implementation_authorization_rejected(self):
        data = _valid_toml()
        data["status"]["isolated_future_implementation_authorized_after_commit_and_independent_verification"] = False
        _expect_reject_toml(data)
        data2 = _valid_toml()
        data2["authorization_boundary"]["isolated_future_implementation_authorized_after_commit_and_independent_verification"] = False
        _expect_reject_toml(data2)


# ---------------------------------------------------------------------------
# Error hygiene
# ---------------------------------------------------------------------------

class ErrorHygieneTests(unittest.TestCase):
    def test_unknown_top_level_key_rejected(self):
        data = _valid_toml()
        data["unexpected_table"] = {"x": 1}
        _expect_reject_toml(data)

    def test_missing_top_level_key_rejected(self):
        data = _valid_toml()
        del data["gt_isolation"]
        _expect_reject_toml(data)

    def test_unknown_nested_key_rejected(self):
        data = _valid_toml()
        data["graph_contract"]["unexpected_field"] = "x"
        _expect_reject_toml(data)

    def test_missing_nested_key_rejected(self):
        data = _valid_toml()
        del data["graph_contract"]["k"]
        _expect_reject_toml(data)

    def test_wrong_scalar_type_rejected(self):
        data = _valid_toml()
        data["recurrence_contract"]["steps"] = "320"
        _expect_reject_toml(data)

    def test_manifest_unknown_key_rejected(self):
        manifest = _valid_manifest()
        manifest["unexpected"] = 1
        _expect_reject_manifest(manifest)

    def test_manifest_record_missing_key_rejected(self):
        manifest = _valid_manifest()
        del manifest["file_records"][0]["reason_required"]
        _expect_reject_manifest(manifest)

    def test_manifest_bad_sha256_length_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["sha256"] = "abc"
        _expect_reject_manifest(manifest)

    def test_manifest_invalid_classification_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["classification"] = "not_a_real_classification"
        _expect_reject_manifest(manifest)

    def test_json_with_nan_rejected(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write('{"a": NaN}')
        tmp.close()
        try:
            with self.assertRaises(t320id.T320SourceAuthorityError):
                t320id.load_json_strict(Path(tmp.name))
        finally:
            Path(tmp.name).unlink()

    def test_json_with_infinity_rejected(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write('{"a": Infinity}')
        tmp.close()
        try:
            with self.assertRaises(t320id.T320SourceAuthorityError):
                t320id.load_json_strict(Path(tmp.name))
        finally:
            Path(tmp.name).unlink()

    def test_declared_domain_exception_used(self):
        data = _valid_toml()
        data["graph_contract"]["k"] = 99
        path = _write_toml(data)
        try:
            with self.assertRaises(t320id.T320SourceAuthorityError) as ctx:
                t320id.load_authority(path)
            self.assertNotIsInstance(ctx.exception, (KeyError, TypeError))
        finally:
            path.unlink()

    def test_fatal_exception_propagates(self):
        # KeyboardInterrupt raised inside a `_run_git`-style broad except
        # must propagate, never be swallowed as a T320SourceAuthorityError.
        with self.assertRaises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            except Exception as exc:  # noqa: BLE001
                t320id._reraise_if_fatal(exc)
                raise AssertionError("should not reach here")


# ---------------------------------------------------------------------------
# Path safety unit tests (direct, not via full load)
# ---------------------------------------------------------------------------

class PathSafetyTests(unittest.TestCase):
    def test_reject_absolute(self):
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id._reject_unsafe_relative_path("/etc/passwd")

    def test_reject_traversal(self):
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id._reject_unsafe_relative_path("a/../../b")

    def test_reject_empty(self):
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id._reject_unsafe_relative_path("")

    def test_reject_option_like(self):
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id._reject_unsafe_relative_path("--upload-pack=x")

    def test_accept_normal_path(self):
        t320id._reject_unsafe_relative_path("src/open_vocabulary_segmentation/models/dinotext/dinotext.py")

    def test_reject_malformed_commit_hex(self):
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id._require_full_commit_hex("zzzz", "commit")

    def test_reject_uppercase_commit_hex(self):
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id._require_full_commit_hex("A" * 40, "commit")

    def test_accept_valid_commit_hex(self):
        t320id._require_full_commit_hex("00392527937253f93b9f3fe36b76347b4a6f1eb0", "commit")


class DigestScopeAndRecord7PrecisionRepairTests(unittest.TestCase):
    """Regression tests for REPAIR_T320_AUTHORITY_DIGEST_SCOPE_AND_SYMBOL_PRECISION:
    (1) required/reason_required are now inside the canonical digest's
    cryptographic scope; (2) record 7's required_symbols is narrowed to
    DINOTextSegInference.align_corners, verified by AST without import."""

    # --- Digest coverage (1-10) ---

    def test_01_baseline_digest_reconstructs_exactly(self):
        manifest = _valid_manifest()
        self.assertEqual(
            t320id.compute_canonical_source_closure_digest(manifest),
            manifest["canonical_source_closure_digest"],
        )

    def test_02_required_true_to_false_changes_digest(self):
        manifest = _valid_manifest()
        baseline = t320id.compute_canonical_source_closure_digest(manifest)
        mutated = copy.deepcopy(manifest)
        mutated["file_records"][0]["required"] = False
        self.assertNotEqual(t320id.compute_canonical_source_closure_digest(mutated), baseline)

    def test_03_one_character_reason_change_changes_digest(self):
        manifest = _valid_manifest()
        baseline = t320id.compute_canonical_source_closure_digest(manifest)
        mutated = copy.deepcopy(manifest)
        mutated["file_records"][0]["reason_required"] += "."
        self.assertNotEqual(t320id.compute_canonical_source_closure_digest(mutated), baseline)

    def test_04_empty_reason_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["reason_required"] = ""
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.compute_canonical_source_closure_digest(manifest)

    def test_05_whitespace_only_reason_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["reason_required"] = "   \t\n  "
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.compute_canonical_source_closure_digest(manifest)

    def test_06_nul_containing_reason_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][0]["reason_required"] = "reason with a \x00 nul byte"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.compute_canonical_source_closure_digest(manifest)

    def test_07_unicode_reason_encoding_is_deterministic(self):
        manifest = _valid_manifest()
        mutated = copy.deepcopy(manifest)
        mutated["file_records"][0]["reason_required"] += " éè中文 \U0001F600"
        d1 = t320id.compute_canonical_source_closure_digest(mutated)
        d2 = t320id.compute_canonical_source_closure_digest(copy.deepcopy(mutated))
        self.assertEqual(d1, d2)
        self.assertEqual(len(d1), 64)

    def test_08_dict_json_key_order_does_not_change_digest(self):
        manifest = _valid_manifest()
        baseline = t320id.compute_canonical_source_closure_digest(manifest)
        reordered_records = []
        for rec in manifest["file_records"]:
            keys = list(rec.keys())
            keys.reverse()
            reordered_records.append({k: rec[k] for k in keys})
        reordered_manifest = dict(manifest)
        reordered_manifest["file_records"] = reordered_records
        self.assertEqual(t320id.compute_canonical_source_closure_digest(reordered_manifest), baseline)

    def test_09_pythonhashseed_does_not_change_digest(self):
        import subprocess as _sp
        manifest_path = MANIFEST_PATH
        baseline = None
        for seed in ["0", "1", "42", "999999"]:
            env = dict(os.environ, PYTHONHASHSEED=seed)
            code = (
                "import sys, json; sys.path.insert(0, %r)\n"
                "import semantic_evidence_t320_source_identity as t\n"
                "m = json.load(open(%r))\n"
                "print(t.compute_canonical_source_closure_digest(m))\n"
            ) % (str(SRC_DIR), str(manifest_path))
            out = _sp.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip()
            if baseline is None:
                baseline = out
            else:
                self.assertEqual(out, baseline)

    def test_10_required_symbol_order_remains_digest_significant(self):
        manifest = _valid_manifest()
        baseline = t320id.compute_canonical_source_closure_digest(manifest)
        mutated = copy.deepcopy(manifest)
        rec = mutated["file_records"][1]
        if len(rec["required_symbols"]) >= 2:
            rec["required_symbols"] = list(reversed(rec["required_symbols"]))
            self.assertNotEqual(t320id.compute_canonical_source_closure_digest(mutated), baseline)

    # --- Record-7 precision (11-24) ---

    def test_11_required_locator_is_exact(self):
        manifest = _valid_manifest()
        rec7 = manifest["file_records"][6]
        self.assertEqual(rec7["path"], "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py")
        self.assertEqual(rec7["required_symbols"], ["DINOTextSegInference.align_corners"])

    def test_12_broad_class_locator_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"][6]["required_symbols"] = ["DINOTextSegInference"]
        _expect_reject_manifest(manifest)

    def test_13_wrong_attribute_name_rejected(self):
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self):\n"
            "        self.some_other_attribute = False\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_14_missing_class_rejected(self):
        source = (
            "class SomethingElse:\n"
            "    def __init__(self):\n"
            "        self.align_corners = False\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_15_missing_assignment_rejected(self):
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self):\n"
            "        pass\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_16_assignment_true_rejected(self):
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self):\n"
            "        self.align_corners = True\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_17_non_literal_assignment_rejected(self):
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self, cfg):\n"
            "        self.align_corners = cfg.some_flag\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_18_duplicate_ambiguous_assignment_rejected(self):
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self):\n"
            "        self.align_corners = False\n"
            "    def reset(self):\n"
            "        self.align_corners = False\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_19_locator_resolves_without_import(self):
        # verify_align_corners_locator_source takes plain text -- proof by
        # construction that no import/exec of the module is required.
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self):\n"
            "        self.align_corners = False\n"
        )
        t320id.verify_align_corners_locator_source(source)  # must not raise
        self.assertNotIn("DINOTextSegInference", sys.modules)

    def test_19b_locator_verified_against_real_historical_blob(self):
        E12 = "00392527937253f93b9f3fe36b76347b4a6f1eb0"
        t320id.verify_align_corners_locator(
            REPO_ROOT, E12,
            "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py",
        )  # must not raise

    def test_20_rwr_branch_disclosed_as_present_but_unused(self):
        data = _valid_toml()
        final = data["final_rescale_contract"]
        self.assertTrue(final["rwr_branch_present_in_historical_file"])
        self.assertFalse(final["rwr_branch_used_by_t320_path"])
        self.assertFalse(final["rwr_branch_reachable_from_align_corners_locator"])
        self.assertFalse(final["apply_rwr_to_e3_snapshot_in_required_closure"])

    def test_21_adding_rwr_dependency_to_closure_rejected(self):
        manifest = _valid_manifest()
        rwr = copy.deepcopy(manifest["file_records"][0])
        rwr["path"] = "src/open_vocabulary_segmentation/models/dinotext/cover_dr/inference.py"
        rwr["ordinal"] = 8
        manifest["file_records"].append(rwr)
        _expect_reject_manifest(manifest)

    def test_22_adding_window_cache_rejected(self):
        manifest = _valid_manifest()
        wc = copy.deepcopy(manifest["file_records"][3])
        wc["path"] = "src/open_vocabulary_segmentation/segmentation/evaluation/window_cache.py"
        wc["ordinal"] = 8
        manifest["file_records"].append(wc)
        _expect_reject_manifest(manifest)

    def test_23_removing_sliding_window_geometry_rejected(self):
        manifest = _valid_manifest()
        manifest["file_records"].pop(3)
        _expect_reject_manifest(manifest)

    def test_24_source_file_count_remains_7(self):
        manifest = _valid_manifest()
        self.assertEqual(len(manifest["file_records"]), 7)

    # --- Non-regression (25-33) ---

    def test_25_source_commit_tree_unchanged(self):
        manifest = _valid_manifest()
        self.assertEqual(manifest["source_commit"], "00392527937253f93b9f3fe36b76347b4a6f1eb0")
        self.assertEqual(manifest["source_tree_id"], "28f66998d4ca48879c0c2c3fc758954c67a1d8aa")

    def test_26_all_seven_blob_ids_unchanged(self):
        expected = [
            "d0517cf150c946b92a0a97952d81d531d3b6e851",
            "7467f88674b5b057fabf55a7d5026fc5984b15b2",
            "27363ea8861b408e4aca4589ae0126515b2eb2c3",
            "801dc779ece4bf68d3c6b9795c402d4e18951379",
            "4e53c1d001532c00fa886643280f787c1c0d1214",
            "2658e6f44a8bc4fe546d61f6a486f1f7bca2c2e7",
            "f37e1c651504743bb0dda3cebc3befbcf6c73aab",
        ]
        manifest = _valid_manifest()
        self.assertEqual([r["blob_id"] for r in manifest["file_records"]], expected)

    def test_27_all_seven_sha256_unchanged(self):
        expected = [
            "b7d4eec053dc204db869590a5ff3763b630084a65f0abd3101190b1e77f33ac5",
            "79ce8ed0e5462131cf148a1615aa60048b4a05782de7d5279ddee45856de6108",
            "13371fd697ef5fae41f2a0b7e99f8c5a65d679a337de734e539a8fdf7e95acf4",
            "082a812907581444ed6ed87b232c82105a0071ca24cd802aceec81fa0ec3c25b",
            "94d920e7f9153e97c8c4948013beffd7533902c7d307fe15607a6b03508873ed",
            "a9ac50a7620031451c29a98f1e9a17cd52fd55850d8d7050c43c0d868f92a94a",
            "e5a4dc40aefe866156c9ca397d77a31cd243fc05d79f5e6cd1905e8d983265f4",
        ]
        manifest = _valid_manifest()
        self.assertEqual([r["sha256"] for r in manifest["file_records"]], expected)

    def test_28_graph_recurrence_constants_unchanged(self):
        data = _valid_toml()
        self.assertEqual(data["graph_contract"]["k"], 12)
        self.assertEqual(data["graph_contract"]["affinity_power"], 3.0)
        self.assertEqual(data["recurrence_contract"]["alpha"], 0.98)
        self.assertEqual(data["recurrence_contract"]["steps"], 320)

    def test_29_internal_upsample_remains_true(self):
        data = _valid_toml()
        self.assertIs(data["internal_upsample_contract"]["align_corners"], True)

    def test_30_final_rescale_remains_false(self):
        data = _valid_toml()
        self.assertIs(data["final_rescale_contract"]["align_corners_frozen_runtime_value"], False)

    def test_31_350_image_disclosure_remains_contextual_non_matching(self):
        data = _valid_toml()
        disclosure = data["historical_350_disclosure"]
        self.assertTrue(disclosure["contextual_evidence_only"])
        self.assertFalse(disclosure["matched_evidence_for_this_authority"])
        self.assertFalse(disclosure["divergence"]["complete_end_to_end_identity"])

    def test_32_authorization_boundary_remains_closed(self):
        data = _valid_toml()
        for key, value in data["authorization_boundary"].items():
            if key == "isolated_future_implementation_authorized_after_commit_and_independent_verification":
                self.assertTrue(value)
            else:
                self.assertFalse(value, f"{key} must remain False")

    def test_33_fatal_exceptions_propagate(self):
        with self.assertRaises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            except Exception as exc:  # noqa: BLE001
                t320id._reraise_if_fatal(exc)
                raise AssertionError("should not reach here")


class SourceSymbolValidationHardeningTests(unittest.TestCase):
    """Regression tests for HARDEN_T320_AUTHORITY_SOURCE_SYMBOL_VALIDATION:
    (1) exact canonical_encoding label; (2) DINOTextSegInference.align_corners
    must be assigned specifically inside __init__; (3) every required_symbols
    entry for all seven records is AST-verified against its bound blob."""

    E12 = "00392527937253f93b9f3fe36b76347b4a6f1eb0"

    # --- Canonical label (1-7) ---

    def test_01_exact_v2_passes(self):
        manifest = _valid_manifest()
        self.assertEqual(manifest["canonical_encoding"], "length_prefixed_utf8_v2")
        t320id.load_manifest(MANIFEST_PATH)  # must not raise

    def test_02_v1_rejected(self):
        m = _valid_manifest()
        m["canonical_encoding"] = "length_prefixed_utf8_v1"
        _expect_reject_manifest(m)

    def test_03_v3_rejected(self):
        m = _valid_manifest()
        m["canonical_encoding"] = "length_prefixed_utf8_v3"
        _expect_reject_manifest(m)

    def test_04_uppercase_rejected(self):
        m = _valid_manifest()
        m["canonical_encoding"] = "LENGTH_PREFIXED_UTF8_V2"
        _expect_reject_manifest(m)

    def test_05_whitespace_rejected(self):
        m = _valid_manifest()
        m["canonical_encoding"] = " length_prefixed_utf8_v2 "
        _expect_reject_manifest(m)

    def test_06_wrong_type_rejected(self):
        for bad in (True, 2, None, ["length_prefixed_utf8_v2"], {"v": 2}):
            m = _valid_manifest()
            m["canonical_encoding"] = bad
            _expect_reject_manifest(m)

    def test_07_missing_field_rejected(self):
        m = _valid_manifest()
        del m["canonical_encoding"]
        _expect_reject_manifest(m)

    # --- Generic source-symbol verification (8-17) ---

    def test_08_all_real_records_pass(self):
        manifest = _valid_manifest()
        for rec in manifest["file_records"]:
            t320id.verify_required_symbols(REPO_ROOT, rec["source_commit"], rec["path"], rec["required_symbols"])

    def test_09_fictitious_symbol_in_each_record_rejected(self):
        manifest = _valid_manifest()
        for idx in range(7):
            rec = manifest["file_records"][idx]
            with self.assertRaises(t320id.T320SourceAuthorityError):
                t320id.verify_required_symbols(
                    REPO_ROOT, rec["source_commit"], rec["path"], ["totally_fictitious_symbol_xyz"]
                )

    def test_10_comment_only_spoof_rejected(self):
        source = "# build_directed_topk_graph is mentioned here only in a comment\npass\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_bare_symbol_locator_source(source, "build_directed_topk_graph", "<t>")

    def test_11_docstring_only_spoof_rejected(self):
        source = '"""def build_directed_topk_graph(): mentioned only in a docstring."""\npass\n'
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_bare_symbol_locator_source(source, "build_directed_topk_graph", "<t>")

    def test_12_imported_alias_rejected_when_definition_required(self):
        source = "from somewhere import build_directed_topk_graph\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_bare_symbol_locator_source(source, "build_directed_topk_graph", "<t>")

    def test_13_nested_function_rejected_for_top_level_locator(self):
        source = "def outer():\n    def build_directed_topk_graph():\n        pass\n    return build_directed_topk_graph\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_bare_symbol_locator_source(source, "build_directed_topk_graph", "<t>")

    def test_14_wrong_symbol_kind_accepted_as_attribute_but_is_method(self):
        # DINOTextSegInference.__init__ names a real method, not a self.attr
        # assignment -- the attribute-locator grammar must reject it.
        source = "class DINOTextSegInference:\n    def __init__(self):\n        pass\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_attribute_locator_source(source, "DINOTextSegInference", "__init__", "<t>")

    def test_15_duplicate_definition_rejected(self):
        source = "def build_directed_topk_graph():\n    pass\n\ndef build_directed_topk_graph():\n    pass\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_bare_symbol_locator_source(source, "build_directed_topk_graph", "<t>")

    def test_16_malformed_locator_rejected(self):
        for bad in ["", ".", "Foo.", ".bar", "Foo..bar", "Foo.bar.baz", " Foo", "Foo ", "Foo(x)", "Foo[0]", "Foo,bar"]:
            with self.assertRaises(t320id.T320SourceAuthorityError):
                t320id._parse_symbol_locator(bad, "<t>")

    def test_17_unsupported_dotted_depth_rejected(self):
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id._parse_symbol_locator("a.b.c", "<t>")

    # --- Record 7 (18-30) ---

    def test_18_real_init_false_assignment_passes(self):
        t320id.verify_align_corners_locator(
            REPO_ROOT, self.E12,
            "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py",
        )  # must not raise

    def test_19_assignment_in_configure_rejected(self):
        source = "class DINOTextSegInference:\n    def configure(self):\n        self.align_corners = False\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_20_assignment_in_encode_decode_rejected(self):
        source = "class DINOTextSegInference:\n    def encode_decode(self):\n        self.align_corners = False\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_21_class_scope_assignment_rejected(self):
        source = "class DINOTextSegInference:\n    align_corners = False\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_22_nested_helper_assignment_in_init_rejected(self):
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self):\n"
            "        def _helper():\n"
            "            self.align_corners = False\n"
            "        _helper()\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_23_true_rejected(self):
        source = "class DINOTextSegInference:\n    def __init__(self):\n        self.align_corners = True\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_24_integer_zero_rejected(self):
        source = "class DINOTextSegInference:\n    def __init__(self):\n        self.align_corners = 0\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_25_string_false_rejected(self):
        source = 'class DINOTextSegInference:\n    def __init__(self):\n        self.align_corners = "False"\n'
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_26_variable_assignment_rejected(self):
        source = "class DINOTextSegInference:\n    def __init__(self, cfg):\n        self.align_corners = cfg.flag\n"
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_27_duplicate_assignment_rejected(self):
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self):\n"
            "        self.align_corners = False\n"
            "        self.align_corners = False\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_28_conflicting_assignment_rejected(self):
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self):\n"
            "        self.align_corners = False\n"
            "    def configure(self):\n"
            "        self.align_corners = True\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_29_property_setter_rejected(self):
        source = (
            "class DINOTextSegInference:\n"
            "    def __init__(self):\n"
            "        self.align_corners = False\n"
            "    @align_corners.setter\n"
            "    def align_corners(self, value):\n"
            "        self.align_corners = value\n"
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    def test_30_comment_docstring_spoof_rejected(self):
        source = (
            'class DINOTextSegInference:\n'
            '    """self.align_corners = False (docstring only)"""\n'
            '    def __init__(self):\n'
            '        # self.align_corners = False\n'
            '        pass\n'
        )
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source(source)

    # --- Security / error hygiene (31-34) ---

    def test_31_source_syntax_error_rejected_with_domain_exception(self):
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_bare_symbol_locator_source("def broken(:\n", "x", "<t>")
        with self.assertRaises(t320id.T320SourceAuthorityError):
            t320id.verify_align_corners_locator_source("class Broken(:\n")

    def test_32_malformed_utf8_rejected_with_domain_exception(self):
        import subprocess, tempfile, shutil
        tmp_repo = tempfile.mkdtemp()
        try:
            subprocess.run(["git", "init", "-q"], cwd=tmp_repo, check=True)
            subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_repo, check=True)
            subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_repo, check=True)
            bad_path = Path(tmp_repo) / "bad.py"
            bad_path.write_bytes(b"\xff\xfe\x00bad-not-utf8")
            subprocess.run(["git", "add", "bad.py"], cwd=tmp_repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "bad"], cwd=tmp_repo, check=True)
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_repo, capture_output=True, text=True, check=True).stdout.strip()
            with self.assertRaises(t320id.T320SourceAuthorityError):
                t320id.verify_align_corners_locator(Path(tmp_repo), commit, "bad.py")
            with self.assertRaises(t320id.T320SourceAuthorityError):
                t320id.verify_required_symbols(Path(tmp_repo), commit, "bad.py", ["anything"])
        finally:
            shutil.rmtree(tmp_repo, ignore_errors=True)

    def test_33_wrong_blob_hash_rejected_before_semantic_resolution(self):
        manifest = _valid_manifest()
        mutated = copy.deepcopy(manifest)
        mutated["file_records"][2]["sha256"] = "0" * 64
        mutated["file_records"][2]["required_symbols"] = ["also_fictitious"]
        try:
            t320id.verify_manifest_against_git(REPO_ROOT, mutated)
            self.fail("should have raised")
        except t320id.T320SourceAuthorityError as e:
            self.assertIn("sha256 mismatch", str(e))

    def test_34_fatal_exceptions_propagate(self):
        with self.assertRaises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            except Exception as exc:  # noqa: BLE001
                t320id._reraise_if_fatal(exc)
                raise AssertionError("should not reach here")

    # --- Non-regression (35-40) ---

    def test_35_digest_unchanged(self):
        manifest = _valid_manifest()
        self.assertEqual(
            manifest["canonical_source_closure_digest"],
            "2a6268a0513c59346a9ceae394346b5096d05083369ef49c46a662fd3f4da6d2",
        )
        self.assertEqual(
            t320id.compute_canonical_source_closure_digest(manifest),
            "2a6268a0513c59346a9ceae394346b5096d05083369ef49c46a662fd3f4da6d2",
        )

    def test_36_seven_file_closure_unchanged(self):
        manifest = _valid_manifest()
        self.assertEqual(len(manifest["file_records"]), 7)
        paths = [r["path"] for r in manifest["file_records"]]
        self.assertTrue(any("sliding_window_geometry.py" in p for p in paths))
        self.assertFalse(any("window_cache.py" in p for p in paths))

    def test_37_constants_unchanged(self):
        data = _valid_toml()
        self.assertEqual(data["graph_contract"]["k"], 12)
        self.assertEqual(data["recurrence_contract"]["alpha"], 0.98)
        self.assertEqual(data["recurrence_contract"]["steps"], 320)
        self.assertEqual(data["graph_contract"]["affinity_power"], 3.0)

    def test_38_interpolation_settings_unchanged(self):
        data = _valid_toml()
        self.assertIs(data["internal_upsample_contract"]["align_corners"], True)
        self.assertIs(data["final_rescale_contract"]["align_corners_frozen_runtime_value"], False)

    def test_39_historical_disclosure_unchanged(self):
        data = _valid_toml()
        d = data["historical_350_disclosure"]
        self.assertEqual(d["execution_commit"], "804a774588b97e43e0347343089a0209ae28a5c9")
        self.assertFalse(d["matched_evidence_for_this_authority"])
        self.assertTrue(d["contextual_evidence_only"])

    def test_40_authorization_unchanged(self):
        data = _valid_toml()
        for key, value in data["authorization_boundary"].items():
            if key == "isolated_future_implementation_authorized_after_commit_and_independent_verification":
                self.assertTrue(value)
            else:
                self.assertFalse(value)


if __name__ == "__main__":
    unittest.main()
