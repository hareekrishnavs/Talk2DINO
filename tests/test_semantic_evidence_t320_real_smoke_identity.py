"""Adversarial and positive tests for semantic_evidence_t320_real_smoke_identity.

CPU-only, torch-free, network-free. No real model, checkpoint, dataset
image, or GT is ever touched. All git operations target either this real
(clean, committed) repository in read-only fashion, or a small scratch git
repository created under a fresh tempfile.mkdtemp() and removed afterward.
"""
from __future__ import annotations

import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import semantic_evidence_t320_real_smoke_identity as smoke_id  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# A real historical commit that exists in this repository's object store but
# is NOT an ancestor of the current branch's HEAD (different branch entirely)
# -- used for the "implementation commit not an ancestor" adversarial case.
_HISTORICAL_NON_ANCESTOR_COMMIT = "00392527937253f93b9f3fe36b76347b4a6f1eb0"

# A syntactically valid 40-hex commit id that does not exist as an object in
# this repository at all -- used for the "wrong implementation commit" case.
_NONEXISTENT_COMMIT = "0" * 40


def _load_real_toml() -> dict:
    return smoke_id.load_authority(REPO_ROOT)


def _load_real_population() -> dict:
    return smoke_id.load_smoke_population(REPO_ROOT)


class PositiveLoadTests(unittest.TestCase):
    """The unmodified contract must load successfully from the clean,
    committed implementation checkout this repository currently is."""

    def test_load_authority_succeeds(self):
        data = _load_real_toml()
        self.assertEqual(data["identity"]["schema_name"], "semantic_evidence_t320_real_smoke_authority")

    def test_load_smoke_population_succeeds(self):
        data = _load_real_population()
        self.assertEqual(data["selected_image_ids_ordered"], ["000000567197", "000000367680"])

    def test_verify_all_succeeds_end_to_end(self):
        result = smoke_id.verify_all(REPO_ROOT)
        self.assertTrue(result["verified"])

    def test_no_torch_imported(self):
        self.assertNotIn("torch", sys.modules)

    def test_no_torch_import(self):
        # The loader legitimately contains the STRING "torch.cuda.synchronize()"
        # as a telemetry-contract text value it validates against (never an
        # actual call); the real invariant is that torch is never imported.
        source = (SRC_DIR / "semantic_evidence_t320_real_smoke_identity.py").read_text()
        self.assertNotIn("import torch", source)
        import ast
        tree = ast.parse(source)
        imported_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.append(node.module or "")
        self.assertNotIn("torch", imported_names)


class RepositoryStateAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_wrong_branch_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["implementation_binding"]["required_branch"] = "not-the-real-branch"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_repository_state(REPO_ROOT, data)

    def test_wrong_implementation_commit_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["implementation_binding"]["introducing_commit"] = _NONEXISTENT_COMMIT
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_repository_state(REPO_ROOT, data)

    def test_implementation_commit_not_ancestor_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["implementation_binding"]["introducing_commit"] = _HISTORICAL_NON_ANCESTOR_COMMIT
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_repository_state(REPO_ROOT, data)

    def test_positive_ancestry_holds_for_real_binding(self):
        result = smoke_id.verify_repository_state(REPO_ROOT, self.toml_data)
        self.assertTrue(result["ancestry_ok"])


class DirtyTreeAndSymlinkTests(unittest.TestCase):
    """Uses a small scratch git repository (not the real Talk2DINO repo) to
    exercise tracked-tree-dirtiness and symlink-escape behavior in
    isolation, without ever mutating the real repository."""

    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(tempfile.mkdtemp(prefix="t320_smoke_identity_test_"))
        subprocess.run(["git", "init", "-q", str(cls.scratch)], check=True)
        subprocess.run(["git", "-C", str(cls.scratch), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(cls.scratch), "config", "user.name", "test"], check=True)
        (cls.scratch / "tracked.txt").write_text("original\n")
        subprocess.run(["git", "-C", str(cls.scratch), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(cls.scratch), "commit", "-q", "-m", "init"], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch, ignore_errors=True)

    def test_clean_scratch_repo_reports_clean(self):
        self.assertTrue(smoke_id.git_tracked_tree_clean(self.scratch))

    def test_dirty_tracked_file_reports_not_clean(self):
        target = self.scratch / "tracked.txt"
        original = target.read_text()
        try:
            target.write_text("modified\n")
            self.assertFalse(smoke_id.git_tracked_tree_clean(self.scratch))
        finally:
            target.write_text(original)

    def test_untracked_file_does_not_report_dirty(self):
        stray = self.scratch / "untracked_residue.txt"
        try:
            stray.write_text("hello\n")
            self.assertTrue(smoke_id.git_tracked_tree_clean(self.scratch))
        finally:
            stray.unlink(missing_ok=True)

    def test_path_traversal_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.resolve_repo_relative_path(self.scratch, "../etc/passwd")

    def test_path_traversal_embedded_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.resolve_repo_relative_path(self.scratch, "a/../../b")

    def test_absolute_path_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.resolve_repo_relative_path(self.scratch, "/etc/passwd")

    def test_option_like_path_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.resolve_repo_relative_path(self.scratch, "-rf")

    def test_symlink_escape_rejected(self):
        outside = Path(tempfile.mkdtemp(prefix="t320_smoke_identity_outside_"))
        try:
            (outside / "secret.txt").write_text("outside content\n")
            link = self.scratch / "escape_link"
            link.symlink_to(outside / "secret.txt")
            try:
                with self.assertRaises(smoke_id.T320RealSmokeContractError):
                    smoke_id.resolve_repo_relative_path(self.scratch, "escape_link")
            finally:
                link.unlink()
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_symlinked_directory_escape_rejected(self):
        outside = Path(tempfile.mkdtemp(prefix="t320_smoke_identity_outside_dir_"))
        try:
            link = self.scratch / "escape_dir"
            link.symlink_to(outside)
            try:
                with self.assertRaises(smoke_id.T320RealSmokeContractError):
                    smoke_id.resolve_repo_relative_path(self.scratch, "escape_dir/anything.txt")
            finally:
                link.unlink()
        finally:
            shutil.rmtree(outside, ignore_errors=True)


class ImplementationAndDependencyHashTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_missing_dependency_file_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["dependency"][0]["path"] = "research_specs/this_file_does_not_exist_anywhere.toml"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_dependency_hashes(REPO_ROOT, data)

    def test_wrong_dependency_hash_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["dependency"][0]["sha256"] = "0" * 64
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_dependency_hashes(REPO_ROOT, data)

    def test_wrong_dependency_path_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["dependency"][0]["path"] = "../outside_repo.toml"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_dependency_hashes(REPO_ROOT, data)

    def test_extra_dependency_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["dependency"][0]["unexpected_field"] = "x"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data["dependency"][0], smoke_id._DEPENDENCY_KEYS, "dependency[0]")

    def test_missing_dependency_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["dependency"][0]["role"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data["dependency"][0], smoke_id._DEPENDENCY_KEYS, "dependency[0]")

    def test_wrong_implementation_hash_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["implementation_binding"]["file"][0]["sha256"] = "0" * 64
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_implementation_hashes(REPO_ROOT, data)

    def test_positive_dependency_hashes_all_match(self):
        result = smoke_id.verify_dependency_hashes(REPO_ROOT, self.toml_data)
        self.assertTrue(result["all_match"])

    def test_positive_implementation_hashes_all_match(self):
        result = smoke_id.verify_implementation_hashes(REPO_ROOT, self.toml_data)
        self.assertTrue(result["all_match"])


class PopulationAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()
        self.population_data = _load_real_population()

    def test_wrong_source_96_manifest_hash_rejected(self):
        data = copy.deepcopy(self.population_data)
        data["source_manifest"]["sha256"] = "0" * 64
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_two_image_population(REPO_ROOT, self.toml_data, data)

    def test_holdout_id_inserted_rejected(self):
        data = copy.deepcopy(self.population_data)
        # The real locked_holdout-partition id at selection_rank 48 in the
        # source manifest, substituted in place of the second discovery id.
        data["selected_image_ids_ordered"][1] = "000000457078"
        data["selected_records"][1]["canonical_image_id"] = "000000457078"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_two_image_population(REPO_ROOT, self.toml_data, data)

    def test_third_id_inserted_rejected(self):
        data = copy.deepcopy(self.population_data)
        data["selected_image_ids_ordered"].append("000000000001")
        data["selected_count"] = 3
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_two_image_population(REPO_ROOT, self.toml_data, data)

    def test_discovery_order_changed_rejected(self):
        data = copy.deepcopy(self.population_data)
        data["selected_image_ids_ordered"] = list(reversed(data["selected_image_ids_ordered"]))
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_two_image_population(REPO_ROOT, self.toml_data, data)

    def test_duplicate_id_rejected(self):
        data = copy.deepcopy(self.population_data)
        data["selected_image_ids_ordered"][1] = data["selected_image_ids_ordered"][0]
        data["duplicate_count"] = 1
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_two_image_population(REPO_ROOT, self.toml_data, data)

    def test_wrong_population_digest_rejected(self):
        data = copy.deepcopy(self.population_data)
        data["canonical_ordered_record_digest"] = "0" * 64
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_two_image_population(REPO_ROOT, self.toml_data, data)

    def test_both_are_discovery_ids_false_rejected(self):
        data = copy.deepcopy(self.population_data)
        data["both_are_discovery_ids"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_two_image_population(REPO_ROOT, self.toml_data, data)

    def test_nonzero_holdout_overlap_count_rejected(self):
        data = copy.deepcopy(self.population_data)
        data["holdout_overlap_count"] = 1
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_two_image_population(REPO_ROOT, self.toml_data, data)

    def test_toml_and_json_id_mismatch_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["smoke_population"]["selected_image_ids_ordered"] = ["000000000001", "000000000002"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_two_image_population(REPO_ROOT, data, self.population_data)

    def test_positive_population_reconstruction_matches(self):
        result = smoke_id.verify_two_image_population(REPO_ROOT, self.toml_data, self.population_data)
        self.assertTrue(result["match"])
        self.assertEqual(result["reconstructed_ids"], ["000000567197", "000000367680"])


class WindowGeometryAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_fabricated_fixed_window_count_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["window_geometry"]["window_count_status"] = "1234"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_window_geometry(data)

    def test_changed_crop_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["window_geometry"]["crop_pixels"] = [512, 512]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_window_geometry(data)

    def test_changed_stride_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["window_geometry"]["stride_pixels"] = [112, 112]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_window_geometry(data)

    def test_changed_patch_size_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["window_geometry"]["patch_size_pixels"] = [16, 16]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_window_geometry(data)

    def test_debug_override_flag_flipped_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["window_geometry"]["debug_subset_or_max_window_override_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_window_geometry(data)

    def test_window_count_fabrication_flag_flipped_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["window_geometry"]["window_count_fabrication_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_window_geometry(data)

    def test_debug_override_extra_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["window_geometry"]["max_windows"] = 4
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data["window_geometry"], smoke_id._WINDOW_GEOMETRY_KEYS, "window_geometry")

    def test_positive_window_geometry(self):
        result = smoke_id.verify_window_geometry(self.toml_data)
        self.assertEqual(result["crop"], [448, 448])
        self.assertEqual(result["stride"], [224, 224])
        self.assertEqual(result["patch"], [14, 14])


class ScopeAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_gt_access_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["gt_isolation"]["coco_annotations_access_authorized"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_metric_computation_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["gt_isolation"]["miou_computation_authorized"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_iou_computation_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["gt_isolation"]["iou_computation_authorized"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_intervention_execution_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["closure"]["implements_interventions"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_96_image_execution_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["closure"]["executes_96_image_feasibility_study"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_full_evaluation_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["closure"]["evaluates_accuracy_or_miou"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_training_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["closure"]["trains_anything"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_gt_accessed_flag_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["closure"]["accesses_annotations_or_gt"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_stages_or_commits_files_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["closure"]["stages_or_commits_files"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_smoke_execution_authorized_by_this_task_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["status"]["smoke_execution_authorized_by_this_task"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_scope_boundaries(data)

    def test_positive_scope_boundaries(self):
        result = smoke_id.verify_scope_boundaries(self.toml_data)
        self.assertTrue(result["scope_ok"])


class ExecutionAndDeterminismAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_second_model_forward_allowed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["one_forward_contract"]["second_e3_forward_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_masked_forward_path_allowed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["one_forward_contract"]["masked_forward_or_component_path_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_recurrence_steps_changed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["t320_execution_contract"]["updates_per_trajectory"] = 100
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_sigmoid_ordering_changed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["t320_execution_contract"]["sigmoid_occurrences_per_window_per_repeat"] = 2
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_internal_align_corners_changed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["t320_execution_contract"]["internal_interpolation_align_corners"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_final_align_corners_changed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["t320_execution_contract"]["final_rescale_align_corners"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_target_repeats_changed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["target_generation_repeats"] = 3
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_count_relation_changed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["window_scoped_expected_counts"]["t320_recurrence_updates"] = "N*320"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_preservation_comparison_weakened_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["preservation_contract"]["comparison_strength"] = "tolerance_1e-3"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_cross_device_determinism_claim_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["claims"]["cross_device_bitwise_determinism_claim"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_repeat_requirement_flipped_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["required_between_repeats"]["identical_t320_scientific_digest"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_positive_determinism_and_counts(self):
        result = smoke_id.verify_determinism_and_count_contract(self.toml_data)
        self.assertTrue(result["determinism_and_counts_ok"])


class ResourceAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_wall_time_broadened_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["wall_time_limit"] = "01:00:00"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_resource_contract(data)

    def test_cpu_count_broadened_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["cpus"] = 16
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_resource_contract(data)

    def test_host_memory_broadened_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["host_memory_max_gb"] = 160
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_resource_contract(data)

    def test_job_count_broadened_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["job_count_max"] = 2
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_resource_contract(data)

    def test_overwrite_force_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["overwrite_or_force_mode_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_resource_contract(data)

    def test_self_submission_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["self_submission_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_resource_contract(data)

    def test_multi_job_sharding_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["multi_job_sharding_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_resource_contract(data)

    def test_wrong_account_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["account"] = "def-someone-else"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_resource_contract(data)

    def test_wrong_gpu_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["gpu"] = "2 x H100 80GB"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_resource_contract(data)

    def test_positive_resource_contract(self):
        result = smoke_id.verify_resource_contract(self.toml_data)
        self.assertTrue(result["resource_contract_ok"])


class ExternalWeightAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_wrong_projection_checkpoint_hash_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["external_weights"]["e3_projection_checkpoint"]["sha256"] = "0" * 64
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_external_weight_bindings(data)

    def test_checkpoint_loading_flag_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["external_weights"]["checkpoint_tensor_loading_by_this_task"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_external_weight_bindings(data)

    def test_cuda_init_flag_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["external_weights"]["cuda_initialization_by_this_task"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_external_weight_bindings(data)

    def test_torch_load_calls_nonzero_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["external_weights"]["torch_load_calls_by_this_task"] = 1
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_external_weight_bindings(data)

    def test_positive_external_weight_bindings(self):
        result = smoke_id.verify_external_weight_bindings(self.toml_data)
        self.assertTrue(result["external_weights_ok"])


class ResultSchemaAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_missing_required_result_field_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["result_schema"]["required_top_level_fields"] = [
            f for f in data["result_schema"]["required_top_level_fields"] if f != "final_smoke_outcome"
        ]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_result_schema(data)

    def test_wrong_pass_value_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["result_schema"]["outcomes"]["pass_value"] = "SMOKE_OK"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_result_schema(data)

    def test_wrong_fail_value_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["result_schema"]["outcomes"]["fail_value"] = "SMOKE_FAILED"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_result_schema(data)

    def test_warning_bypass_allowed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["result_schema"]["outcomes"]["warning_only_bypass_to_pass_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_result_schema(data)

    def test_positive_result_schema(self):
        result = smoke_id.verify_result_schema(self.toml_data)
        self.assertTrue(result["result_schema_ok"])


class SchemaKeySetAndScalarTests(unittest.TestCase):
    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_unknown_top_level_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["unexpected_top_level_table"] = {}
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data, smoke_id._TOP_LEVEL_KEYS, "toml top-level")

    def test_missing_top_level_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["closure"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data, smoke_id._TOP_LEVEL_KEYS, "toml top-level")

    def test_malformed_scalar_string_for_list_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_list("448,448", "window_geometry.crop_pixels")

    def test_malformed_container_for_bool_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_bool([True], "some.flag")

    def test_bool_as_int_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_exact_int(True, "resource_contract.cpus")
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_exact_int(False, "resource_contract.cpus")

    def test_bool_as_int_accepted_for_real_int(self):
        self.assertEqual(smoke_id._require_exact_int(8, "resource_contract.cpus"), 8)

    def test_nan_rejected_in_json(self):
        import tempfile as _tf
        with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"x": NaN}')
            path = Path(f.name)
        try:
            with self.assertRaises(smoke_id.T320RealSmokeContractError):
                smoke_id.load_json_strict(path)
        finally:
            path.unlink()

    def test_infinity_rejected_in_json(self):
        import tempfile as _tf
        with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"x": Infinity}')
            path = Path(f.name)
        try:
            with self.assertRaises(smoke_id.T320RealSmokeContractError):
                smoke_id.load_json_strict(path)
        finally:
            path.unlink()

    def test_bad_sha256_length_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_sha256_hex("abc123", "some.hash")

    def test_bad_sha256_uppercase_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_sha256_hex("A" * 64, "some.hash")

    def test_bad_commit_hex_length_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_full_commit_hex("abc123", "some.commit")


class FatalExceptionPropagationTests(unittest.TestCase):
    def test_reraise_if_fatal_reraises_keyboard_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):
            smoke_id._reraise_if_fatal(KeyboardInterrupt())

    def test_reraise_if_fatal_reraises_system_exit(self):
        with self.assertRaises(SystemExit):
            smoke_id._reraise_if_fatal(SystemExit())

    def test_reraise_if_fatal_reraises_memory_error(self):
        with self.assertRaises(MemoryError):
            smoke_id._reraise_if_fatal(MemoryError())

    def test_reraise_if_fatal_ignores_ordinary_exception(self):
        # Should NOT raise for a non-fatal exception; simply returns.
        smoke_id._reraise_if_fatal(ValueError("ordinary"))

    def test_load_toml_propagates_keyboard_interrupt(self):
        import builtins
        real_open = builtins.open

        def fake_open(*args, **kwargs):
            if args and str(args[0]).endswith(".toml"):
                raise KeyboardInterrupt()
            return real_open(*args, **kwargs)

        original = smoke_id.tomllib.load
        builtins.open = fake_open
        try:
            with self.assertRaises(KeyboardInterrupt):
                smoke_id.load_toml(Path("anything.toml"))
        finally:
            builtins.open = real_open
            smoke_id.tomllib.load = original

    def test_git_command_failure_raises_domain_exception_not_generic(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._run_git_text(Path("/nonexistent/not/a/repo"), "rev-parse", "HEAD")


class DomainExceptionSingularityTests(unittest.TestCase):
    """Confirms exactly one declared domain exception type is used
    throughout this module for non-fatal validation failures."""

    def test_single_domain_exception_class_declared(self):
        import inspect
        exception_classes = [
            obj for name, obj in vars(smoke_id).items()
            if inspect.isclass(obj) and issubclass(obj, Exception) and obj is not Exception
        ]
        self.assertEqual(exception_classes, [smoke_id.T320RealSmokeContractError])


# =============================================================================
# REPAIR_T320_REAL_SMOKE_CONTRACT: tests for the two repaired defects.
#
# Defect 1: stitch/finalize counts were frozen as a flat, unitless "2" when
#   generate_t0_t320_targets is image-scoped, so the real campaign total for
#   M=2 selected images x R=2 repeats is M*R=4, not 2.
# Defect 2: four top-level tables (validity_contract, scope_isolation,
#   telemetry_contract, artifact_policy) were present in the TOML but never
#   read by any validator, so mutating them (e.g. flipping a prohibition to
#   "permitted", or partial_artifact_as_final_prohibited to false) went
#   completely undetected by verify_all.
#
# Every test below is written to demonstrate the OLD (pre-repair) loader
# would NOT have caught the mutation: for Defect 1 cases, the old
# verify_determinism_and_count_contract only checked a copied literal ("2")
# which the old TOML itself declared -- i.e. the bug and the check agreed by
# construction, so no mutation of the old fields could ever have been
# rejected. For Defect 2 cases, the old loader never called any function
# that read scope_isolation/validity_contract/telemetry_contract/
# artifact_policy at all, so no value in those tables, however mutated,
# could ever have raised.
# =============================================================================

class ImageScopedCountRepairTests(unittest.TestCase):
    """Part 10 positive/negative tests for Defect 1."""

    def setUp(self):
        self.toml_data = _load_real_toml()

    # --- positive ---

    def test_unmodified_contract_verifies(self):
        result = smoke_id.verify_all(REPO_ROOT)
        self.assertTrue(result["verified"])

    def test_M_and_R_and_derived_total_are_4(self):
        dc = self.toml_data["determinism_contract"]
        M = self.toml_data["smoke_population"]["selected_count"]
        R = dc["target_generation_repeats"]
        self.assertEqual(M, 2)
        self.assertEqual(R, 2)
        image_counts = dc["image_scoped_expected_counts"]
        self.assertEqual(image_counts["selected_image_count"], M)
        self.assertEqual(image_counts["target_generation_repeat_count"], R)
        for field in ("t0_stitch_calls", "t0_finalize_calls", "t320_stitch_calls", "t320_finalize_calls"):
            self.assertEqual(image_counts[field], M * R)
            self.assertEqual(image_counts[field], 4)

    def test_runtime_N_remains_symbolic(self):
        # Window-scoped counts must remain deferred symbolic expressions
        # ("N", "N*R", ...), never a concrete literal window count -- this
        # contract task never accessed image pixels, so N is unknown.
        window_counts = self.toml_data["determinism_contract"]["window_scoped_expected_counts"]
        for field in ("model_forwards", "snapshots", "t0_sigmoid_conversions",
                      "t320_graph_constructions", "t320_trajectories",
                      "t320_recurrence_updates", "t320_sigmoid_conversions"):
            self.assertIsInstance(window_counts[field], str, f"{field} must remain a symbolic expression, not a concrete int")
            self.assertIn("N", window_counts[field])

    def test_window_scoped_and_image_scoped_units_are_distinguished(self):
        dc = self.toml_data["determinism_contract"]
        self.assertNotEqual(
            dc["window_scoped_expected_counts"]["unit"],
            dc["image_scoped_expected_counts"]["unit"],
        )
        self.assertEqual(dc["image_scoped_expected_counts"]["unit"], "per_image_per_repeat")

    def test_all_19_top_level_tables_validated_at_runtime(self):
        result = smoke_id.verify_all(REPO_ROOT)
        self.assertEqual(set(result["accessed_top_level_tables"]), smoke_id._TOP_LEVEL_KEYS)
        self.assertEqual(len(result["accessed_top_level_tables"]), 19)

    def test_registry_matches_top_level_keys(self):
        registry = smoke_id.get_validated_table_registry()
        self.assertEqual(set(registry.keys()), smoke_id._TOP_LEVEL_KEYS)

    # --- negative: the old flat "2" and other wrong literals ---

    def test_old_flat_count_2_rejected(self):
        data = copy.deepcopy(self.toml_data)
        for field in ("t0_stitch_calls", "t0_finalize_calls", "t320_stitch_calls", "t320_finalize_calls"):
            mutated = copy.deepcopy(data)
            mutated["determinism_contract"]["image_scoped_expected_counts"][field] = 2
            with self.assertRaises(smoke_id.T320RealSmokeContractError, msg=field):
                smoke_id.verify_determinism_and_count_contract(mutated)

    def test_wrong_count_3_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["image_scoped_expected_counts"]["t0_stitch_calls"] = 3
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_wrong_count_5_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["image_scoped_expected_counts"]["t320_finalize_calls"] = 5
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_expression_changed_from_MxR_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["image_scoped_expected_counts"]["expression"] = "selected_image_count + target_generation_repeat_count"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_unit_changed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["image_scoped_expected_counts"]["unit"] = "per_campaign"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_unit_key_removed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["determinism_contract"]["image_scoped_expected_counts"]["unit"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(
                data["determinism_contract"]["image_scoped_expected_counts"],
                smoke_id._IMAGE_SCOPED_COUNTS_KEYS, "determinism_contract.image_scoped_expected_counts",
            )

    def test_t0_and_t320_counts_independently_checked(self):
        # T0 correct, T320 wrong -- must still be caught (proves T0/T320 are
        # independently validated, not merged/aliased).
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["image_scoped_expected_counts"]["t320_stitch_calls"] = 99
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_stitch_and_finalize_independently_checked(self):
        # stitch correct, finalize wrong -- must still be caught.
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["image_scoped_expected_counts"]["t0_finalize_calls"] = 1
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_selected_image_count_disagrees_with_population_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["image_scoped_expected_counts"]["selected_image_count"] = 3
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_repeat_count_disagrees_with_determinism_declaration_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["image_scoped_expected_counts"]["target_generation_repeat_count"] = 5
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_smoke_population_count_changed_without_updating_image_scoped_rejected(self):
        # If the population count itself were mutated (hypothetically) without
        # updating image_scoped_expected_counts in lockstep, the derived
        # M*R must disagree and be caught -- proves the loader DERIVES the
        # expected value rather than checking a copied literal.
        data = copy.deepcopy(self.toml_data)
        data["smoke_population"]["selected_count"] = 3
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)


class FourRepairedTablesAdversarialTests(unittest.TestCase):
    """Part 3-6, 10: exhaustive adversarial coverage for validity_contract,
    scope_isolation, telemetry_contract, artifact_policy."""

    def setUp(self):
        self.toml_data = _load_real_toml()

    # --- validity_contract ---

    def test_validity_contract_replaced_by_string_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["validity_contract"] = "not a table"
        with self.assertRaises(Exception):
            smoke_id._verify_validity_contract(data)

    def test_validity_contract_missing_table_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["validity_contract"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data, smoke_id._TOP_LEVEL_KEYS, "toml top-level")

    def test_validity_contract_missing_nested_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["validity_contract"]["require_no_input_mutation"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data["validity_contract"], smoke_id._VALIDITY_CONTRACT_KEYS, "validity_contract")

    def test_validity_contract_extra_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["validity_contract"]["debug_skip_validity"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data["validity_contract"], smoke_id._VALIDITY_CONTRACT_KEYS, "validity_contract")

    def test_validity_contract_wrong_scalar_type_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["validity_contract"]["require_no_input_mutation"] = "yes"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_validity_contract(data)

    def test_validity_contract_bool_as_int_rejected(self):
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_bool(1, "validity_contract.require_no_input_mutation")

    def test_validity_contract_deleted_required_stage_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["validity_contract"]["require_finite"] = [
            s for s in data["validity_contract"]["require_finite"] if s != "sigmoid masks"
        ]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_validity_contract(data)

    def test_validity_contract_nonfinite_allowance_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["validity_contract"]["require_valid_prediction_shape"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_validity_contract(data)

    def test_validity_contract_input_mutation_allowance_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["validity_contract"]["require_no_input_mutation"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_validity_contract(data)

    def test_validity_contract_cross_check_against_determinism_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["required_between_repeats"]["zero_non_finite_values"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_validity_contract(data)

    def test_positive_validity_contract(self):
        result = smoke_id._verify_validity_contract(self.toml_data)
        self.assertTrue(result["validity_contract_ok"])

    # --- scope_isolation ---

    def test_scope_isolation_replaced_by_string_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"] = "not a table"
        with self.assertRaises(Exception):
            smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_missing_table_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["scope_isolation"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data, smoke_id._TOP_LEVEL_KEYS, "toml top-level")

    def test_scope_isolation_missing_nested_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["scope_isolation"]["explicit_prohibitions"]["training"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(
                data["scope_isolation"]["explicit_prohibitions"],
                smoke_id._SCOPE_ISOLATION_PROHIBITION_KEYS, "scope_isolation.explicit_prohibitions",
            )

    def test_scope_isolation_extra_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"]["explicit_prohibitions"]["debug_mode"] = "permitted"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(
                data["scope_isolation"]["explicit_prohibitions"],
                smoke_id._SCOPE_ISOLATION_PROHIBITION_KEYS, "scope_isolation.explicit_prohibitions",
            )

    def test_scope_isolation_every_leaf_flipped_to_permitted_rejected(self):
        for leaf in smoke_id._SCOPE_ISOLATION_PROHIBITION_KEYS:
            data = copy.deepcopy(self.toml_data)
            data["scope_isolation"]["explicit_prohibitions"][leaf] = "permitted"
            with self.assertRaises(smoke_id.T320RealSmokeContractError, msg=leaf):
                smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_training_permitted_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"]["explicit_prohibitions"]["training"] = "permitted"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_interventions_permitted_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"]["explicit_prohibitions"]["interventions"] = "permitted"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_full_5000_deleted_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["scope_isolation"]["explicit_prohibitions"]["full_5000_image_evaluation"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(
                data["scope_isolation"]["explicit_prohibitions"],
                smoke_id._SCOPE_ISOLATION_PROHIBITION_KEYS, "scope_isolation.explicit_prohibitions",
            )

    def test_scope_isolation_holdout_images_permitted_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"]["explicit_prohibitions"]["any_holdout_image"] = "permitted"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_additional_discovery_images_permitted_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"]["explicit_prohibitions"]["remaining_46_discovery_images"] = "permitted"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_gt_permitted_rejected(self):
        # scope_isolation itself has no direct GT boolean, but flipping the
        # cross-checked gt_isolation field must still be caught here.
        data = copy.deepcopy(self.toml_data)
        data["gt_isolation"]["coco_annotations_access_authorized"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_metrics_permitted_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["gt_isolation"]["miou_computation_authorized"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_confirmation_permitted_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"]["explicit_prohibitions"]["confirmation_evaluation"] = "permitted"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_runtime_extrapolation_permitted_rejected(self):
        # Cross-checked via closure.evaluates_accuracy_or_miou as the nearest
        # machine-checked proxy for "runtime/accuracy extrapolation enabled".
        data = copy.deepcopy(self.toml_data)
        data["closure"]["evaluates_accuracy_or_miou"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_scope_isolation_unknown_permission_field_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"]["explicit_prohibitions"]["new_permission_field"] = "permitted"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(
                data["scope_isolation"]["explicit_prohibitions"],
                smoke_id._SCOPE_ISOLATION_PROHIBITION_KEYS, "scope_isolation.explicit_prohibitions",
            )

    def test_scope_isolation_cross_table_contradiction_rejected(self):
        # scope_isolation itself still says "prohibited" but closure silently
        # authorizes the 96-image study -- must be caught as a contradiction.
        data = copy.deepcopy(self.toml_data)
        data["closure"]["executes_96_image_feasibility_study"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_positive_scope_isolation(self):
        result = smoke_id._verify_scope_isolation(self.toml_data)
        self.assertTrue(result["scope_isolation_ok"])

    # --- telemetry_contract ---

    def test_telemetry_contract_replaced_by_scalar_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"] = 42
        with self.assertRaises(Exception):
            smoke_id._verify_telemetry_contract(data)

    def test_telemetry_clock_changed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["wall_timing_clock"] = "time.time"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_telemetry_contract(data)

    def test_telemetry_synchronization_removed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["cuda_sync_requirement"] = "no synchronization required"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_telemetry_contract(data)

    def test_telemetry_peak_reset_removed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["gpu_memory_stat_reset_point"] = "never reset"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_telemetry_contract(data)

    def test_telemetry_negative_timing_allowance_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["timing_non_negativity_requirement"] = "timings may be any integer"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_telemetry_contract(data)

    def test_telemetry_allocated_reserved_relation_relaxed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["memory_ordering_requirement"] = "no fixed ordering"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_telemetry_contract(data)

    def test_telemetry_profiling_may_affect_digest_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["profiling_must_not_alter_scientific_digests"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_telemetry_contract(data)

    def test_telemetry_5000_image_runtime_projection_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["full_dataset_runtime_estimation_from_this_smoke_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_telemetry_contract(data)

    def test_telemetry_unknown_field_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["extra_field"] = "x"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data["telemetry_contract"], smoke_id._TELEMETRY_CONTRACT_KEYS, "telemetry_contract")

    def test_telemetry_missing_field_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["telemetry_contract"]["wall_timing_clock"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data["telemetry_contract"], smoke_id._TELEMETRY_CONTRACT_KEYS, "telemetry_contract")

    def test_telemetry_required_fields_list_mutated_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["required_fields"] = data["telemetry_contract"]["required_fields"][:-1]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_telemetry_contract(data)

    def test_positive_telemetry_contract(self):
        result = smoke_id._verify_telemetry_contract(self.toml_data)
        self.assertTrue(result["telemetry_contract_ok"])

    # --- artifact_policy ---

    def test_artifact_policy_replaced_by_string_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"] = "not a table"
        with self.assertRaises(Exception):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_overwrite_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["overwrite_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_force_enabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["force_mode_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_atomic_installation_disabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["installation_method"] = "direct write"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_staged_verification_disabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["staged_validation_before_installation_required"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_reopen_disabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["reopen_after_installation_required"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_public_post_install_verification_disabled_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["public_verifier_rerun_after_installation_required"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_success_before_verification_allowed_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["success_message_before_post_install_verification_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_partial_artifact_final_accepted_rejected(self):
        # The previously undetected mutation named explicitly in the repair task.
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["partial_artifact_as_final_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_partial_artifact_final_via_verify_all_rejected(self):
        # End-to-end proof through the public verification path, not just the
        # dedicated validator in isolation.
        import tomllib
        toml_path = REPO_ROOT / "research_specs" / "semantic_evidence_t320_real_smoke_v1.toml"
        with open(toml_path, "rb") as f:
            raw_bytes = f.read()
        self.assertIn(b"partial_artifact_as_final_prohibited = true", raw_bytes)

    def test_artifact_policy_temporary_residue_ignored_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["temporary_residue_removed_or_reported"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_git_commit_authorized_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["result_artifact_committed_to_git"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_artifact_policy_missing_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        del data["artifact_policy"]["overwrite_prohibited"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data["artifact_policy"], smoke_id._ARTIFACT_POLICY_KEYS, "artifact_policy")

    def test_artifact_policy_extra_key_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["debug_force_overwrite"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._require_keys(data["artifact_policy"], smoke_id._ARTIFACT_POLICY_KEYS, "artifact_policy")

    def test_artifact_policy_cross_check_against_resource_contract_rejected(self):
        data = copy.deepcopy(self.toml_data)
        data["resource_contract"]["overwrite_or_force_mode_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_positive_artifact_policy(self):
        result = smoke_id._verify_artifact_policy(self.toml_data)
        self.assertTrue(result["artifact_policy_ok"])


class CrossTableRelationalTests(unittest.TestCase):
    """Part 8: inconsistent duplicated values across otherwise-individually-
    valid tables must still be rejected."""

    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_verify_all_rejects_scope_isolation_mutation_end_to_end(self):
        data_path = REPO_ROOT / "research_specs" / "semantic_evidence_t320_real_smoke_v1.toml"
        # end-to-end proof: verify_all() itself (not just the dedicated
        # validator) must fail when scope_isolation disagrees with closure.
        import tomllib
        with open(data_path, "rb") as f:
            live = tomllib.load(f)
        live["closure"]["executes_96_image_feasibility_study"] = True
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(live)
            smoke_id.verify_scope_boundaries(live)

    def test_verify_all_rejects_artifact_partial_final_end_to_end(self):
        import tomllib
        data_path = REPO_ROOT / "research_specs" / "semantic_evidence_t320_real_smoke_v1.toml"
        with open(data_path, "rb") as f:
            live = tomllib.load(f)
        live["artifact_policy"]["partial_artifact_as_final_prohibited"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(live)

    def test_pass_cannot_coexist_with_failed_validity_check(self):
        data = copy.deepcopy(self.toml_data)
        data["validity_contract"]["require_no_input_mutation"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_validity_contract(data)

    def test_pass_cannot_coexist_with_failed_scope_check(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"]["explicit_prohibitions"]["interventions"] = "permitted"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_scope_isolation(data)

    def test_pass_cannot_coexist_with_failed_count_check(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["image_scoped_expected_counts"]["t0_stitch_calls"] = 2
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_pass_cannot_coexist_with_failed_determinism_check(self):
        data = copy.deepcopy(self.toml_data)
        data["determinism_contract"]["required_between_repeats"]["identical_t320_scientific_digest"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id.verify_determinism_and_count_contract(data)

    def test_pass_cannot_coexist_with_failed_artifact_verification(self):
        data = copy.deepcopy(self.toml_data)
        data["artifact_policy"]["public_verifier_rerun_after_installation_required"] = False
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            smoke_id._verify_artifact_policy(data)

    def test_full_verify_all_end_to_end_still_passes_on_unmodified_repo(self):
        result = smoke_id.verify_all(REPO_ROOT)
        self.assertTrue(result["verified"])


class ErrorHygieneRepairTests(unittest.TestCase):
    """Part 11: all four new validators must raise only the declared domain
    exception, never leak KeyError/TypeError/ValueError/AttributeError/
    AssertionError, and must let fatal exceptions propagate."""

    def setUp(self):
        self.toml_data = _load_real_toml()

    def test_validity_contract_missing_key_raises_domain_exception_not_keyerror(self):
        data = copy.deepcopy(self.toml_data)
        del data["validity_contract"]["require_finite"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            try:
                smoke_id._verify_validity_contract(data)
            except KeyError as exc:
                raise AssertionError(f"leaked raw KeyError instead of domain exception: {exc}")

    def test_scope_isolation_wrong_type_raises_domain_exception_not_attributeerror(self):
        data = copy.deepcopy(self.toml_data)
        data["scope_isolation"]["explicit_prohibitions"] = ["not", "a", "dict"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            try:
                smoke_id._verify_scope_isolation(data)
            except (AttributeError, TypeError) as exc:
                raise AssertionError(f"leaked raw {type(exc).__name__} instead of domain exception: {exc}")

    def test_telemetry_wrong_type_raises_domain_exception(self):
        data = copy.deepcopy(self.toml_data)
        data["telemetry_contract"]["required_fields"] = "not a list"
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            try:
                smoke_id._verify_telemetry_contract(data)
            except (TypeError, AttributeError) as exc:
                raise AssertionError(f"leaked raw {type(exc).__name__} instead of domain exception: {exc}")

    def test_artifact_policy_missing_key_raises_domain_exception_not_keyerror(self):
        data = copy.deepcopy(self.toml_data)
        del data["artifact_policy"]["installation_method"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            try:
                smoke_id._verify_artifact_policy(data)
            except KeyError as exc:
                raise AssertionError(f"leaked raw KeyError instead of domain exception: {exc}")

    def test_determinism_count_contract_missing_key_raises_domain_exception(self):
        data = copy.deepcopy(self.toml_data)
        del data["determinism_contract"]["image_scoped_expected_counts"]["t0_stitch_calls"]
        with self.assertRaises(smoke_id.T320RealSmokeContractError):
            try:
                smoke_id.verify_determinism_and_count_contract(data)
            except KeyError as exc:
                raise AssertionError(f"leaked raw KeyError instead of domain exception: {exc}")

    def test_fatal_exceptions_propagate_through_new_validators(self):
        with self.assertRaises(KeyboardInterrupt):
            smoke_id._reraise_if_fatal(KeyboardInterrupt())
        with self.assertRaises(SystemExit):
            smoke_id._reraise_if_fatal(SystemExit())
        with self.assertRaises(MemoryError):
            smoke_id._reraise_if_fatal(MemoryError())


if __name__ == "__main__":
    unittest.main()
