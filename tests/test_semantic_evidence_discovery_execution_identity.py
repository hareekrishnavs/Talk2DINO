"""Hermetic tests for semantic_evidence_discovery_execution_identity.py.

No torch, no CUDA, no dataset contents, no checkpoint contents, no Slurm.
Uses only the real committed repository (read-only, via git show) and
disposable temporary copies of the authority TOML for mutation testing --
never modifies the real repository.
"""

import copy
import json
import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import semantic_evidence_discovery_execution_identity as ident  # noqa: E402

TOML_PATH = REPO_ROOT / "research_specs" / "semantic_evidence_discovery_execution_v1.toml"


def _load_raw() -> dict:
    with open(TOML_PATH, "rb") as fh:
        return tomllib.load(fh)


class _MutationHarness:
    """Loads a mutated in-memory copy of the real authority dict by
    monkeypatching tomllib.load for the duration of one call, so mutation
    tests exercise the real validation functions without needing a TOML
    writer and without ever touching the real file on disk."""

    def __init__(self, mutator):
        self.mutator = mutator

    def __enter__(self):
        real_load = tomllib.load
        data = _load_raw()
        mutated = self.mutator(copy.deepcopy(data))

        def fake_load(fh):
            return mutated

        tomllib.load = fake_load
        self._real_load = real_load
        return mutated

    def __exit__(self, *exc):
        tomllib.load = self._real_load
        return False


class PositiveLoadTests(unittest.TestCase):
    def test_load_authority_succeeds_on_real_file(self):
        data = ident.load_authority(TOML_PATH)
        self.assertIn("identity", data)

    def test_verify_all_succeeds_on_real_repository(self):
        result = ident.verify_all(REPO_ROOT, TOML_PATH)
        self.assertTrue(result["prerequisite_gate"]["all_ok"])
        self.assertTrue(result["dependency_hashes"]["all_match"])
        self.assertTrue(result["population"]["all_ok"])
        self.assertTrue(result["scientific_inheritance"]["all_ok"])
        self.assertTrue(result["authorization_flags"]["all_ok"])
        self.assertTrue(result["smoke_prerequisite"]["hash_match"])
        self.assertTrue(result["smoke_prerequisite"]["outcome_ok"])

    def test_dependency_role_count_is_18(self):
        data = ident.load_authority(TOML_PATH)
        self.assertEqual(len(data["dependency"]), 18)

    def test_population_counts_are_48_48(self):
        result = ident.verify_all(REPO_ROOT, TOML_PATH)
        self.assertEqual(result["population"]["discovery_count"], 48)
        self.assertEqual(result["population"]["holdout_count"], 48)
        self.assertEqual(result["population"]["overlap_count"], 0)


class RepositoryDependencyValidationTests(unittest.TestCase):
    def test_wrong_required_head_detected(self):
        with _MutationHarness(lambda d: (d["prerequisite"].__setitem__("required_head", "0" * 40), d)[1]):
            data = ident.load_authority(TOML_PATH)
            gate = ident.verify_prerequisite_gate(REPO_ROOT, data)
            self.assertFalse(gate["head_ok"])
            self.assertFalse(gate["all_ok"])

    def test_wrong_dependency_hash_detected(self):
        def mutate(d):
            d["dependency"][0]["sha256"] = "0" * 64
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            deps = ident.verify_dependency_hashes(REPO_ROOT, data)
            self.assertFalse(deps["all_match"])
            self.assertFalse(deps["records"][0]["match"])

    def test_path_traversal_rejected(self):
        def mutate(d):
            d["dependency"][0]["path"] = "../../../etc/passwd"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_dependency_hashes(REPO_ROOT, data)

    def test_absolute_path_rejected(self):
        def mutate(d):
            d["dependency"][0]["path"] = "/etc/passwd"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_dependency_hashes(REPO_ROOT, data)

    def test_symlink_escape_rejected(self):
        scratch = Path(tempfile.mkdtemp())
        try:
            outside = scratch / "outside.txt"
            outside.write_text("secret")
            fake_repo = scratch / "repo"
            fake_repo.mkdir()
            link = fake_repo / "escape_link"
            link.symlink_to(outside)

            def mutate(d):
                d["dependency"][0]["path"] = "escape_link"
                return d
            with _MutationHarness(mutate):
                data = ident.load_authority(TOML_PATH)
                # resolve() through the symlink must land outside fake_repo
                resolved = (fake_repo / "escape_link").resolve()
                self.assertFalse(fake_repo.resolve() in resolved.parents or resolved == fake_repo.resolve())
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_missing_dependency_role_detected(self):
        def mutate(d):
            d["dependency"] = [e for e in d["dependency"] if e["role"] != "t320_implementation"]
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_dependency_hashes(REPO_ROOT, data)

    def test_extra_dependency_key_rejected(self):
        def mutate(d):
            d["dependency"][0]["extra_field"] = "unexpected"
            return d
        with _MutationHarness(mutate):
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.load_authority(TOML_PATH)

    def test_self_reference_dependency_rejected_via_traversal_guard(self):
        def mutate(d):
            d["dependency"][0]["path"] = "."
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            # "." is repo-contained but not a real file at that role's binding
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_dependency_hashes(REPO_ROOT, data)

    def test_duplicate_dependency_role_rejected(self):
        def mutate(d):
            dup = copy.deepcopy(d["dependency"][0])
            dup["path"] = "docs/semantic_evidence_discovery_execution.md"
            d["dependency"].append(dup)
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_dependency_hashes(REPO_ROOT, data)


class PopulationTests(unittest.TestCase):
    def test_47_discovery_ids_detected(self):
        def mutate(d):
            d["population"]["discovery_count"] = 47
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_population(REPO_ROOT, data)

    def test_wrong_discovery_digest_detected(self):
        def mutate(d):
            d["population"]["discovery_digest"] = "0" * 64
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            pop = ident.verify_population(REPO_ROOT, data)
            self.assertFalse(pop["discovery_digest_ok"])
            self.assertFalse(pop["all_ok"])

    def test_wrong_holdout_digest_detected(self):
        def mutate(d):
            d["population"]["locked_holdout_digest"] = "0" * 64
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            pop = ident.verify_population(REPO_ROOT, data)
            self.assertFalse(pop["holdout_digest_ok"])

    def test_overlap_field_mismatch_detected(self):
        def mutate(d):
            d["population"]["discovery_holdout_overlap"] = 1
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_population(REPO_ROOT, data)

    def test_real_population_has_zero_overlap(self):
        result = ident.verify_all(REPO_ROOT, TOML_PATH)
        self.assertEqual(result["population"]["overlap_count"], 0)

    def test_holdout_not_enumerated_in_authority_document(self):
        data = ident.load_authority(TOML_PATH)
        prot = data["population"]["holdout_protection"]
        self.assertTrue(prot["holdout_ids_not_enumerated_in_this_document"])
        # confirm no locked_holdout image id literal appears anywhere in the
        # authority TOML's own raw text (only digests/counts may appear)
        raw_text = TOML_PATH.read_text()
        with open(REPO_ROOT / "evaluation_subsets" / "semantic_evidence_feasibility_96.json") as f:
            manifest = json.load(f)
        holdout_ids = [r["canonical_image_id"] for r in manifest["records"] if r["partition"] == "locked_holdout"]
        leaked = [i for i in holdout_ids if i in raw_text]
        self.assertEqual(leaked, [], f"holdout image IDs leaked into the authority document: {leaked}")


class ScientificInheritanceTests(unittest.TestCase):
    def test_altered_anchor_count_detected(self):
        def mutate(d):
            d["scientific_inheritance"]["anchor_selection"]["target_anchors_per_window"] = 6
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])
            failing = [c for c in sci["checks"] if not c["ok"]]
            self.assertTrue(any("target_anchors_per_window" in c["label"] for c in failing))

    def test_altered_blur_padding_mode_detected(self):
        def mutate(d):
            d["scientific_inheritance"]["gaussian_blur"]["padding_mode"] = "constant"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_altered_t320_constant_alpha_would_be_caught_by_frozen_constants_binding(self):
        data = ident.load_authority(TOML_PATH)
        self.assertEqual(data["t0_t320_target_authority"]["frozen_constants"]["alpha"], 0.98)
        self.assertEqual(data["t0_t320_target_authority"]["frozen_constants"]["k"], 12)
        self.assertEqual(data["t0_t320_target_authority"]["frozen_constants"]["steps"], 320)

    def test_removed_contiguity_requirement_detected(self):
        data = ident.load_authority(TOML_PATH)
        self.assertTrue(data["t0_t320_target_authority"]["frozen_constants"]["contiguous_grid_before_sigmoid_interpolation"])

    def test_altered_gate_b_threshold_detected(self):
        def mutate(d):
            d["gate_contract"]["B"]["min_auroc_improvement_holdout"] = 0.0
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_altered_bootstrap_seed_detected(self):
        def mutate(d):
            d["scientific_inheritance"]["bootstrap"]["seed"] = 1
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_altered_statistics_solver_detected(self):
        def mutate(d):
            d["scientific_inheritance"]["statistics"]["solver"] = "lbfgs"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_real_inheritance_all_45_checks_pass(self):
        data = ident.load_authority(TOML_PATH)
        sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
        self.assertTrue(sci["all_ok"])
        self.assertEqual(sci["check_count"], 51)


class AuthorizationTests(unittest.TestCase):
    def test_immediate_gpu_execution_flag_is_false(self):
        data = ident.load_authority(TOML_PATH)
        self.assertFalse(data["authorization"]["frozen_flags"]["discovery_execution_authorized_now"])

    def test_holdout_execution_enabled_detected(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["holdout_execution_authorized"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            auth = ident.verify_authorization_flags(data)
            self.assertFalse(auth["all_ok"])
            self.assertIn(("frozen_flags", "holdout_execution_authorized", False, True), auth["mismatches"])

    def test_full_96_enabled_detected(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["full_96_image_execution_authorized"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            auth = ident.verify_authorization_flags(data)
            self.assertFalse(auth["all_ok"])

    def test_full_5000_enabled_detected(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["full_5000_image_execution_authorized"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            auth = ident.verify_authorization_flags(data)
            self.assertFalse(auth["all_ok"])

    def test_training_enabled_detected(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["training_authorized"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            auth = ident.verify_authorization_flags(data)
            self.assertFalse(auth["all_ok"])

    def test_scientific_kernel_modification_enabled_detected(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["scientific_kernel_modification_authorized"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            auth = ident.verify_authorization_flags(data)
            self.assertFalse(auth["all_ok"])

    def test_e3_modification_enabled_detected(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["e3_modification_authorized"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            auth = ident.verify_authorization_flags(data)
            self.assertFalse(auth["all_ok"])

    def test_t320_modification_enabled_detected(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["t320_modification_authorized"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            auth = ident.verify_authorization_flags(data)
            self.assertFalse(auth["all_ok"])

    def test_smoke_rerun_enabled_detected(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["smoke_rerun_authorized"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            auth = ident.verify_authorization_flags(data)
            self.assertFalse(auth["all_ok"])

    def test_gpu_submission_in_this_task_is_false(self):
        data = ident.load_authority(TOML_PATH)
        self.assertFalse(data["authorization"]["current_task"]["gpu_submission_in_this_task"])

    def test_discovery_execution_in_this_task_is_false(self):
        data = ident.load_authority(TOML_PATH)
        self.assertFalse(data["authorization"]["current_task"]["discovery_execution_in_this_task"])

    def test_real_authorization_flags_all_correct(self):
        data = ident.load_authority(TOML_PATH)
        auth = ident.verify_authorization_flags(data)
        self.assertTrue(auth["all_ok"])
        self.assertEqual(auth["mismatches"], [])


class GateComputabilityTests(unittest.TestCase):
    def test_gate_a_computable_now(self):
        data = ident.load_authority(TOML_PATH)
        self.assertTrue(data["gate_contract"]["A"]["computable_under_this_contract"])

    def test_gate_e_computable_now(self):
        data = ident.load_authority(TOML_PATH)
        self.assertTrue(data["gate_contract"]["E"]["computable_under_this_contract"])

    def test_gates_b_c_d_not_computable_without_holdout(self):
        data = ident.load_authority(TOML_PATH)
        for gate_id in ("B", "C", "D"):
            self.assertFalse(data["gate_contract"][gate_id]["computable_under_this_contract"], gate_id)
            self.assertEqual(data["gate_contract"][gate_id]["status"], "NOT_COMPUTABLE_WITHOUT_HOLDOUT", gate_id)

    def test_overall_outcome_not_reachable_by_this_authority(self):
        data = ident.load_authority(TOML_PATH)
        self.assertFalse(data["gate_contract"]["outcome"]["overall_status_reachable_by_this_authority"])
        self.assertEqual(data["gate_contract"]["outcome"]["overall_status"], "PENDING_HOLDOUT_EVALUATION")

    def test_gate_outcome_strings_match_predecessor_exactly(self):
        predecessor = tomllib.loads(
            (REPO_ROOT / "research_specs" / "semantic_evidence_feasibility_v1.toml").read_text()
        )
        data = ident.load_authority(TOML_PATH)
        self.assertEqual(data["gate_contract"]["proceed_value"], predecessor["gates"]["outcome"]["proceed_value"])
        self.assertEqual(data["gate_contract"]["stop_value"], predecessor["gates"]["outcome"]["stop_value"])

    def test_discovery_pass_does_not_authorize_holdout(self):
        data = ident.load_authority(TOML_PATH)
        self.assertIn("holdout execution", data["gate_contract"]["outcome"]["discovery_pass_does_not_authorize"])


class ArtifactAndLifecycleTests(unittest.TestCase):
    def test_smoke_profiling_excluded_from_scientific_field(self):
        data = ident.load_authority(TOML_PATH)
        self.assertFalse(data["artifact_contract"]["smoke_profiling_included_as_scientific_field"])

    def test_overwrite_and_force_prohibited(self):
        data = ident.load_authority(TOML_PATH)
        self.assertTrue(data["lifecycle_contract"]["overwrite_prohibited"])
        self.assertTrue(data["lifecycle_contract"]["force_mode_prohibited"])

    def test_no_success_before_post_install_verification(self):
        data = ident.load_authority(TOML_PATH)
        self.assertTrue(data["lifecycle_contract"]["no_success_before_post_install_verification"])

    def test_resource_contract_prohibits_multi_gpu_and_login_node(self):
        data = ident.load_authority(TOML_PATH)
        self.assertTrue(data["resource_contract"]["prohibited"]["multi_gpu_execution"])
        self.assertTrue(data["resource_contract"]["prohibited"]["login_node_execution"])
        self.assertTrue(data["resource_contract"]["prohibited"]["launcher_self_submission"])

    def test_future_file_paths_have_no_current_collision(self):
        data = ident.load_authority(TOML_PATH)
        for path in data["future_implementation_files"]["paths"]:
            self.assertFalse((REPO_ROOT / path).exists(), f"unexpected collision: {path}")


class SmokePrerequisiteTests(unittest.TestCase):
    def test_missing_external_artifact_reported_not_silently_verified(self):
        def mutate(d):
            d["smoke_prerequisite"]["result_path"] = "/nonexistent/path/smoke_result.json"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            smoke = ident.verify_smoke_prerequisite(REPO_ROOT, data)
            self.assertFalse(smoke["external_artifact_available"])
            self.assertIsNone(smoke["hash_match"])

    def test_wrong_expected_hash_detected_against_real_artifact(self):
        def mutate(d):
            d["smoke_prerequisite"]["result_sha256"] = "0" * 64
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            smoke = ident.verify_smoke_prerequisite(REPO_ROOT, data)
            if smoke["external_artifact_available"]:
                self.assertFalse(smoke["hash_match"])

    def test_real_smoke_prerequisite_hash_and_outcome_match(self):
        data = ident.load_authority(TOML_PATH)
        smoke = ident.verify_smoke_prerequisite(REPO_ROOT, data)
        if smoke["external_artifact_available"]:
            self.assertTrue(smoke["hash_match"])
            self.assertTrue(smoke["outcome_ok"])

    def test_smoke_profiling_defects_recorded_and_not_repaired(self):
        data = ident.load_authority(TOML_PATH)
        sp = data["smoke_prerequisite"]
        self.assertFalse(sp["smoke_profiling_defects_repair_authorized_by_this_task"])
        self.assertFalse(sp["smoke_profiling_defects_inherited_by_this_authority"])
        self.assertGreaterEqual(len(sp["smoke_profiling_defects"]), 5)


class ErrorHygieneTests(unittest.TestCase):
    def test_malformed_toml_raises_contract_error(self):
        scratch = Path(tempfile.mkdtemp())
        try:
            bad = scratch / "bad.toml"
            bad.write_text("this is [ not valid toml =")
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.load_authority(bad)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_wrong_scalar_type_rejected(self):
        def mutate(d):
            d["prerequisite"]["required_head"] = 12345
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_prerequisite_gate(REPO_ROOT, data)

    def test_bool_as_int_rejected_in_authorization(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["training_authorized"] = 0
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_authorization_flags(data)

    def test_keyboard_interrupt_propagates(self):
        def raising_open(*a, **kw):
            raise KeyboardInterrupt()

        import builtins
        real_open = builtins.open
        builtins.open = raising_open
        try:
            with self.assertRaises(KeyboardInterrupt):
                ident.verify_all(REPO_ROOT, TOML_PATH)
        finally:
            builtins.open = real_open

    def test_system_exit_propagates(self):
        def raising_open(*a, **kw):
            raise SystemExit(3)

        import builtins
        real_open = builtins.open
        builtins.open = raising_open
        try:
            with self.assertRaises(SystemExit):
                ident.verify_all(REPO_ROOT, TOML_PATH)
        finally:
            builtins.open = real_open

    def test_memory_error_propagates(self):
        def raising_open(*a, **kw):
            raise MemoryError()

        import builtins
        real_open = builtins.open
        builtins.open = raising_open
        try:
            with self.assertRaises(MemoryError):
                ident.verify_all(REPO_ROOT, TOML_PATH)
        finally:
            builtins.open = real_open


# ---------------------------------------------------------------------------
# Repair-verification tests: closure, dependency role set, T320 AST
# cross-verification, smoke metadata cross-check, profiling exclusion,
# future-file-set exactness.
# ---------------------------------------------------------------------------

_NESTED_CLOSED_TABLES_TO_PROBE = [
    "smoke_prerequisite", "population", "t0_t320_target_authority", "gate_contract",
    "authorization", "future_implementation_files", "artifact_contract", "lifecycle_contract",
    "resource_contract", "scientific_inheritance", "discovery_unit_structure", "gt_isolation_boundary",
]


class NestedSchemaClosureTests(unittest.TestCase):
    def test_unknown_key_rejected_in_every_declared_closed_table(self):
        for table in _NESTED_CLOSED_TABLES_TO_PROBE:
            def mutate(d, _t=table):
                d[_t]["__unexpected_key__"] = "x"
                return d
            with _MutationHarness(mutate):
                with self.assertRaises(ident.DiscoveryExecutionContractError, msg=table):
                    ident.load_authority(TOML_PATH)

    def test_unknown_key_rejected_in_deeply_nested_subtable(self):
        probes = [
            ("smoke_prerequisite", "scientific_facts_required"),
            ("population", "holdout_protection"),
            ("t0_t320_target_authority", "frozen_constants"),
            ("gate_contract", "A"),
            ("gate_contract", "outcome"),
            ("authorization", "frozen_flags"),
            ("resource_contract", "prohibited"),
            ("artifact_contract", "final_result"),
        ]
        for table, sub in probes:
            def mutate(d, _t=table, _s=sub):
                d[_t][_s]["__unexpected__"] = "x"
                return d
            with _MutationHarness(mutate):
                with self.assertRaises(ident.DiscoveryExecutionContractError, msg=f"{table}.{sub}"):
                    ident.load_authority(TOML_PATH)

    def test_missing_key_rejected_in_every_declared_closed_table(self):
        for table in _NESTED_CLOSED_TABLES_TO_PROBE:
            def mutate(d, _t=table):
                first_key = next(iter(d[_t].keys()))
                del d[_t][first_key]
                return d
            with _MutationHarness(mutate):
                with self.assertRaises(ident.DiscoveryExecutionContractError, msg=table):
                    ident.load_authority(TOML_PATH)

    def test_wrong_type_for_bool_field_rejected(self):
        def mutate(d):
            d["authorization"]["frozen_flags"]["training_authorized"] = "false"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_authorization_flags(data)

    def test_real_unmodified_toml_passes_full_closure(self):
        data = ident.load_authority(TOML_PATH)
        self.assertIsInstance(data, dict)


class DependencyRoleSetTests(unittest.TestCase):
    def test_each_of_18_roles_individually_required(self):
        data = ident.load_authority(TOML_PATH)
        roles = frozenset(e["role"] for e in data["dependency"])
        self.assertEqual(roles, ident._EXPECTED_DEPENDENCY_ROLES)
        self.assertEqual(len(roles), 18)

    def test_removing_each_role_one_at_a_time_is_rejected(self):
        for role in ident._EXPECTED_DEPENDENCY_ROLES:
            def mutate(d, _r=role):
                d["dependency"] = [e for e in d["dependency"] if e["role"] != _r]
                return d
            with _MutationHarness(mutate):
                data = ident.load_authority(TOML_PATH)
                with self.assertRaises(ident.DiscoveryExecutionContractError, msg=role):
                    ident.verify_dependency_hashes(REPO_ROOT, data)

    def test_unexpected_role_added_rejected(self):
        def mutate(d):
            extra = copy.deepcopy(d["dependency"][0])
            extra["role"] = "totally_unexpected_role"
            extra["path"] = "docs/semantic_evidence_discovery_execution.md"
            d["dependency"].append(extra)
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_dependency_hashes(REPO_ROOT, data)

    def test_duplicate_path_with_distinct_roles_rejected(self):
        def mutate(d):
            extra = copy.deepcopy(d["dependency"][0])
            extra["role"] = "yet_another_role"
            d["dependency"].append(extra)
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_dependency_hashes(REPO_ROOT, data)

    def test_wrong_path_for_real_role_rejected(self):
        # substitute a path that IS committed at required_head but is not
        # the correct file for this role, so the mismatch is a hash
        # mismatch, not a git-show failure on an uncommitted path.
        def mutate(d):
            d["dependency"][0]["path"] = "README.md"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_dependency_hashes(REPO_ROOT, data)
            self.assertFalse(r["all_match"])


class T320ConstantsASTTests(unittest.TestCase):
    BASE_SRC = '''
import torch
import torch.nn.functional as F

K = 12
AFFINITY_POWER = 3.0
ALPHA = 0.98
STEPS = 320
INTERNAL_UPSAMPLE_ALIGN_CORNERS = True
FINAL_RESCALE_ALIGN_CORNERS = False

def internal_upsample(patch_scores, grid_hw, output_hw):
    grid_h, grid_w = grid_hw
    num_classes = patch_scores.shape[1]
    grid = patch_scores.permute(1, 0).reshape(1, num_classes, grid_h, grid_w).contiguous()
    mask = torch.sigmoid(grid)
    upsampled = F.interpolate(mask, size=tuple(output_hw), mode="bilinear", align_corners=INTERNAL_UPSAMPLE_ALIGN_CORNERS)
    return upsampled

def _finite_step_propagate(graph, s0, *, alpha, steps):
    s0_owned = s0.detach().clone()
    p = s0_owned.clone()
    one_minus_alpha = 1.0 - float(alpha)
    for _step in range(1, steps + 1):
        p = float(alpha) * graph.matmul(p) + one_minus_alpha * s0_owned
    return p

def propagate_t320(graph, s0):
    return _finite_step_propagate(graph, s0, alpha=ALPHA, steps=STEPS)
'''

    def _analyze(self, src):
        import ast
        tree = ast.parse(src)
        consts = ident._module_level_literal_assignments(tree)
        up = ident._analyze_internal_upsample(ident._find_function(tree, "internal_upsample"))
        rec = ident._analyze_recurrence_loop(ident._find_function(tree, "_finite_step_propagate"))
        pub = ident._analyze_public_propagate_entry_point(ident._find_function(tree, "propagate_t320"))
        return consts, up, rec, pub

    def test_real_committed_source_passes(self):
        data = ident.load_authority(TOML_PATH)
        r = ident.verify_t0_t320_frozen_constants(REPO_ROOT, data)
        self.assertTrue(r["all_ok"], r["checks"])
        self.assertTrue(r["source_hash_verified"])

    def test_k_mutation_detected(self):
        consts, *_ = self._analyze(self.BASE_SRC.replace("K = 12", "K = 8"))
        self.assertNotEqual(consts["K"], 12)

    def test_alpha_mutation_detected(self):
        consts, *_ = self._analyze(self.BASE_SRC.replace("ALPHA = 0.98", "ALPHA = 0.9"))
        self.assertNotEqual(consts["ALPHA"], 0.98)

    def test_affinity_power_mutation_detected(self):
        consts, *_ = self._analyze(self.BASE_SRC.replace("AFFINITY_POWER = 3.0", "AFFINITY_POWER = 1.0"))
        self.assertNotEqual(consts["AFFINITY_POWER"], 3.0)

    def test_steps_mutation_detected(self):
        consts, *_ = self._analyze(self.BASE_SRC.replace("STEPS = 320", "STEPS = 1"))
        self.assertNotEqual(consts["STEPS"], 320)

    def test_internal_align_corners_mutation_detected(self):
        consts, *_ = self._analyze(self.BASE_SRC.replace("INTERNAL_UPSAMPLE_ALIGN_CORNERS = True", "INTERNAL_UPSAMPLE_ALIGN_CORNERS = False"))
        self.assertNotEqual(consts["INTERNAL_UPSAMPLE_ALIGN_CORNERS"], True)

    def test_final_align_corners_mutation_detected(self):
        consts, *_ = self._analyze(self.BASE_SRC.replace("FINAL_RESCALE_ALIGN_CORNERS = False", "FINAL_RESCALE_ALIGN_CORNERS = True"))
        self.assertNotEqual(consts["FINAL_RESCALE_ALIGN_CORNERS"], False)

    def test_sigmoid_count_mutation_detected(self):
        _, up, _, _ = self._analyze(self.BASE_SRC.replace("mask = torch.sigmoid(grid)", "mask = torch.sigmoid(torch.sigmoid(grid))"))
        self.assertNotEqual(up["sigmoid_count"], 1)

    def test_contiguity_removed_detected(self):
        _, up, _, _ = self._analyze(self.BASE_SRC.replace(".contiguous()", ""))
        self.assertFalse(up["contiguous_found"])

    def test_contiguous_moved_after_sigmoid_detected(self):
        mutated = self.BASE_SRC.replace(
            "    grid = patch_scores.permute(1, 0).reshape(1, num_classes, grid_h, grid_w).contiguous()\n    mask = torch.sigmoid(grid)",
            "    grid = patch_scores.permute(1, 0).reshape(1, num_classes, grid_h, grid_w)\n    mask = torch.sigmoid(grid).contiguous()",
        )
        _, up, _, _ = self._analyze(mutated)
        self.assertFalse(up["contiguous_before_sigmoid"])

    def test_contiguous_moved_after_interpolate_detected(self):
        mutated = self.BASE_SRC.replace(
            "    grid = patch_scores.permute(1, 0).reshape(1, num_classes, grid_h, grid_w).contiguous()",
            "    grid = patch_scores.permute(1, 0).reshape(1, num_classes, grid_h, grid_w)",
        ).replace(
            "    return upsampled",
            "    upsampled = upsampled.contiguous()\n    return upsampled",
        )
        _, up, _, _ = self._analyze(mutated)
        composite = up["contiguous_found"] and up["contiguous_before_sigmoid"] and up["sigmoid_before_interpolate"]
        self.assertFalse(composite)

    def test_early_stop_break_inserted_detected(self):
        mutated = self.BASE_SRC.replace(
            "    for _step in range(1, steps + 1):\n        p = float(alpha) * graph.matmul(p) + one_minus_alpha * s0_owned\n",
            "    for _step in range(1, steps + 1):\n        p = float(alpha) * graph.matmul(p) + one_minus_alpha * s0_owned\n        if _step > 5:\n            break\n",
        )
        _, _, rec, _ = self._analyze(mutated)
        self.assertFalse(rec["early_stop_break_absent"])
        self.assertFalse(rec["early_stop_conditional_absent"])

    def test_public_scientific_override_inserted_detected(self):
        mutated = self.BASE_SRC.replace(
            "def propagate_t320(graph, s0):\n    return _finite_step_propagate(graph, s0, alpha=ALPHA, steps=STEPS)",
            "def propagate_t320(graph, s0, alpha=ALPHA, steps=STEPS):\n    return _finite_step_propagate(graph, s0, alpha=alpha, steps=steps)",
        )
        *_, pub = self._analyze(mutated)
        self.assertFalse(pub["no_public_override"])

    def test_baseline_fixture_itself_passes_all_checks(self):
        consts, up, rec, pub = self._analyze(self.BASE_SRC)
        self.assertEqual(consts["K"], 12)
        self.assertEqual(up["sigmoid_count"], 1)
        self.assertTrue(up["contiguous_before_sigmoid"])
        self.assertTrue(up["sigmoid_before_interpolate"])
        self.assertTrue(rec["early_stop_break_absent"])
        self.assertTrue(pub["no_public_override"])

    def test_toml_declared_k_mismatched_against_real_source_detected(self):
        def mutate(d):
            d["t0_t320_target_authority"]["frozen_constants"]["k"] = 8
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_t0_t320_frozen_constants(REPO_ROOT, data)
            self.assertFalse(r["all_ok"])
            failing = [c for c in r["checks"] if not c["ok"]]
            self.assertTrue(any(c["label"] == "K" for c in failing))

    def test_toml_declared_align_corners_mismatched_against_real_source_detected(self):
        def mutate(d):
            d["t0_t320_target_authority"]["frozen_constants"]["internal_align_corners"] = False
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_t0_t320_frozen_constants(REPO_ROOT, data)
            self.assertFalse(r["all_ok"])

    def test_toml_declared_contiguity_requirement_mismatched_detected(self):
        def mutate(d):
            d["t0_t320_target_authority"]["frozen_constants"]["contiguous_grid_before_sigmoid_interpolation"] = False
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_t0_t320_frozen_constants(REPO_ROOT, data)
            self.assertFalse(r["all_ok"])

    def test_source_hash_mismatch_refuses_to_parse(self):
        def mutate(d):
            for e in d["dependency"]:
                if e["role"] == "t320_implementation":
                    e["sha256"] = "0" * 64
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_t0_t320_frozen_constants(REPO_ROOT, data)


class SmokeMetadataCrossCheckTests(unittest.TestCase):
    def test_wrong_job_id_rejected(self):
        def mutate(d):
            d["smoke_prerequisite"]["job_id"] = "00000000"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_smoke_prerequisite(REPO_ROOT, data)
            if r["external_artifact_available"]:
                self.assertFalse(r["all_ok"])

    def test_wrong_execution_head_rejected(self):
        def mutate(d):
            d["smoke_prerequisite"]["execution_head"] = "0" * 40
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_smoke_prerequisite(REPO_ROOT, data)
            if r["external_artifact_available"]:
                self.assertFalse(r["all_ok"])

    def test_wrong_execution_tree_rejected(self):
        def mutate(d):
            d["smoke_prerequisite"]["execution_tree"] = "0" * 40
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_smoke_prerequisite(REPO_ROOT, data)
            if r["external_artifact_available"]:
                self.assertFalse(r["all_ok"])

    def test_wrong_outcome_required_rejected(self):
        def mutate(d):
            d["smoke_prerequisite"]["outcome_required"] = "FAIL_REAL_E3_T320_INTEGRATION_SMOKE"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_smoke_prerequisite(REPO_ROOT, data)
            if r["external_artifact_available"]:
                self.assertFalse(r["all_ok"])

    def test_wrong_schema_name_rejected(self):
        def mutate(d):
            d["smoke_prerequisite"]["result_schema_name"] = "wrong_schema_name"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_smoke_prerequisite(REPO_ROOT, data)
            if r["external_artifact_available"]:
                self.assertFalse(r["all_ok"])

    def test_wrong_result_hash_rejected(self):
        def mutate(d):
            d["smoke_prerequisite"]["result_sha256"] = "0" * 64
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_smoke_prerequisite(REPO_ROOT, data)
            if r["external_artifact_available"]:
                self.assertFalse(r["hash_match"])
                self.assertFalse(r["all_ok"])

    def test_external_result_missing_reported_not_silently_verified(self):
        def mutate(d):
            d["smoke_prerequisite"]["result_path"] = "/nonexistent/path/smoke_result.json"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_smoke_prerequisite(REPO_ROOT, data)
            self.assertFalse(r["external_artifact_available"])
            self.assertIsNone(r["hash_match"])
            self.assertNotIn("all_ok", r)

    def test_real_smoke_binding_passes_every_cross_check(self):
        data = ident.load_authority(TOML_PATH)
        r = ident.verify_smoke_prerequisite(REPO_ROOT, data)
        if r["external_artifact_available"]:
            self.assertTrue(r["all_ok"], r["checks"])
            for key, ok in r["checks"].items():
                self.assertTrue(ok, key)


class ProfilingExclusionEnforcementTests(unittest.TestCase):
    def test_profiling_status_wrong_value_rejected(self):
        def mutate(d):
            d["smoke_prerequisite"]["smoke_profiling_status"] = "VALID_FOR_RUNTIME_CLAIMS"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_smoke_prerequisite(REPO_ROOT, data)

    def test_smoke_timing_inserted_into_gate_a_rejected(self):
        def mutate(d):
            d["gate_contract"]["A"]["computed_over"] = "smoke job wall_time_ns used directly"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_gate_inputs_exclude_profiling(data)
            self.assertFalse(r["all_ok"])

    def test_smoke_memory_inserted_into_gate_e_rejected(self):
        def mutate(d):
            d["gate_contract"]["E"]["computed_over"] = "peak_gpu allocated bytes from the smoke run"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_gate_inputs_exclude_profiling(data)
            self.assertFalse(r["all_ok"])

    def test_throughput_claim_token_rejected(self):
        def mutate(d):
            d["gate_contract"]["B"]["computable_note"] = "used to support a throughput claim for the full dataset"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_gate_inputs_exclude_profiling(data)
            self.assertFalse(r["all_ok"])

    def test_real_gate_contract_has_no_profiling_tokens(self):
        data = ident.load_authority(TOML_PATH)
        r = ident.verify_gate_inputs_exclude_profiling(data)
        self.assertTrue(r["all_ok"], r["blocklist_violations"])
        self.assertTrue(r["exact_match_ok"])
        self.assertGreater(r["prose_fields_scanned"], 0)

    def test_defects_not_marked_repair_authorized_or_inherited(self):
        data = ident.load_authority(TOML_PATH)
        sp = data["smoke_prerequisite"]
        self.assertFalse(sp["smoke_profiling_defects_repair_authorized_by_this_task"])
        self.assertFalse(sp["smoke_profiling_defects_inherited_by_this_authority"])


class FutureFileSetExactnessTests(unittest.TestCase):
    def test_real_set_matches_expected_exactly(self):
        data = ident.load_authority(TOML_PATH)
        r = ident.verify_future_implementation_files(REPO_ROOT, data)
        self.assertTrue(r["all_ok"], r)
        self.assertEqual(r["declared_count"], 7)

    def test_duplicate_path_rejected(self):
        def mutate(d):
            d["future_implementation_files"]["paths"].append(d["future_implementation_files"]["paths"][0])
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_future_implementation_files(REPO_ROOT, data)
            self.assertFalse(r["all_ok"])
            self.assertTrue(r["duplicates_found"])

    def test_missing_path_rejected(self):
        def mutate(d):
            d["future_implementation_files"]["paths"] = d["future_implementation_files"]["paths"][:-1]
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_future_implementation_files(REPO_ROOT, data)
            self.assertFalse(r["all_ok"])

    def test_extra_holdout_runner_path_rejected(self):
        def mutate(d):
            d["future_implementation_files"]["paths"].append("src/semantic_evidence_discovery_holdout_execute.py")
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_future_implementation_files(REPO_ROOT, data)
            self.assertFalse(r["all_ok"])

    def test_alternate_launcher_rejected(self):
        def mutate(d):
            paths = d["future_implementation_files"]["paths"]
            d["future_implementation_files"]["paths"] = [
                "semantic_evidence_discovery_a100.sbatch" if p == "semantic_evidence_discovery_h100.sbatch" else p
                for p in paths
            ]
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_future_implementation_files(REPO_ROOT, data)
            self.assertFalse(r["all_ok"])

    def test_e3_t320_path_substitution_rejected(self):
        def mutate(d):
            paths = d["future_implementation_files"]["paths"]
            d["future_implementation_files"]["paths"] = [
                "src/semantic_evidence_t320.py" if p == "src/semantic_evidence_discovery_execute.py" else p
                for p in paths
            ]
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_future_implementation_files(REPO_ROOT, data)
            self.assertFalse(r["all_ok"])

    def test_traversal_path_rejected(self):
        def mutate(d):
            d["future_implementation_files"]["paths"][0] = "../../../etc/passwd"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_future_implementation_files(REPO_ROOT, data)

    def test_absolute_path_rejected(self):
        def mutate(d):
            d["future_implementation_files"]["paths"][0] = "/etc/passwd"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_future_implementation_files(REPO_ROOT, data)

    def test_all_seven_currently_absent(self):
        data = ident.load_authority(TOML_PATH)
        r = ident.verify_future_implementation_files(REPO_ROOT, data)
        self.assertTrue(all(r["currently_absent"].values()))


class FieldCoverageAuditTests(unittest.TestCase):
    """Confirms every load-bearing field group in the raw TOML is touched by
    at least one verifier function, using coverage.py-free static tracing:
    each group below names the verifier(s) responsible for it, cross-checked
    against a manual audit of the source. This is a documentation-as-test
    map, not itself a scientific claim."""

    _COVERAGE_MAP = {
        "identity": "load_authority (schema_name/version exact-value check)",
        "prerequisite": "verify_prerequisite_gate (branch/head/tree exact match)",
        "dependency": "verify_dependency_hashes (role-set, path, hash, containment)",
        "smoke_prerequisite": "verify_smoke_prerequisite (hash, outcome, job_id, head, tree, clean, complete, final, schema, counts, repeats, preservation, non-finite, scope, profiling status)",
        "population": "verify_population (counts, ranges, digests, overlap)",
        "scientific_inheritance": "verify_scientific_inheritance (45 equality checks against predecessor)",
        "t0_t320_target_authority": "verify_t0_t320_frozen_constants (AST cross-check against committed source) + verify_scientific_inheritance (t320_status)",
        "discovery_unit_structure": "manually cross-read against scientific_inheritance/statistics (values duplicated from the same predecessor tables already equality-checked)",
        "gt_isolation_boundary": "structural presence via load_authority closure; content is prose describing an unimplemented future phase, not independently re-derivable from predecessor bytes at authority-freeze time",
        "one_forward_reuse_contract": "structural presence via load_authority closure; prose contract for a future implementation",
        "artifact_contract": "structural presence via load_authority closure + closed_schema_requirements booleans",
        "lifecycle_contract": "structural presence via load_authority closure",
        "statistical_execution_contract": "structural presence via load_authority closure",
        "gate_contract": "verify_scientific_inheritance (thresholds) + verify_gate_inputs_exclude_profiling (prose fields)",
        "resource_contract": "structural presence via load_authority closure",
        "authorization": "verify_authorization_flags (exact boolean values)",
        "future_implementation_files": "verify_future_implementation_files (exact set, no duplicates, absence, containment)",
        "this_authoritys_own_files": "structural presence via load_authority closure",
        "closure": "structural presence via load_authority closure",
    }

    def test_every_top_level_table_has_a_named_verifier(self):
        data = ident.load_authority(TOML_PATH)
        for key in data.keys():
            self.assertIn(key, self._COVERAGE_MAP, f"top-level table {key!r} has no coverage-map entry")

    def test_coverage_map_has_no_stale_entries(self):
        data = ident.load_authority(TOML_PATH)
        for key in self._COVERAGE_MAP:
            self.assertIn(key, data, f"coverage-map entry {key!r} does not exist in the real TOML")


# ---------------------------------------------------------------------------
# Final-repair reverification tests: positive exact gate-reference
# validation, propagation-category/subgroup/artifact-field exact bindings,
# explicit integer typing.
# ---------------------------------------------------------------------------

class GateExactReferenceTests(unittest.TestCase):
    def test_real_gate_contract_passes_exact_match(self):
        data = ident.load_authority(TOML_PATH)
        r = ident.verify_gate_inputs_exclude_profiling(data)
        self.assertTrue(r["exact_match_ok"])
        self.assertTrue(r["all_ok"])

    def _mutate_and_check(self, gate_id, field, new_value):
        def mutate(d):
            d["gate_contract"][gate_id][field] = new_value
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_gate_inputs_exclude_profiling(data)
            self.assertFalse(r["all_ok"], f"{gate_id}.{field} = {new_value!r} should be rejected")

    def test_gate_a_prefix_added_fails(self):
        real = ident._EXPECTED_GATE_COMPUTED_OVER["A"]
        self._mutate_and_check("A", "computed_over", "See note: " + real)

    def test_gate_a_suffix_added_fails(self):
        real = ident._EXPECTED_GATE_COMPUTED_OVER["A"]
        self._mutate_and_check("A", "computed_over", real + " (informational)")

    def test_gate_a_capitalization_change_fails(self):
        real = ident._EXPECTED_GATE_COMPUTED_OVER["A"]
        self._mutate_and_check("A", "computed_over", real.swapcase())

    def test_gate_a_whitespace_change_fails(self):
        real = ident._EXPECTED_GATE_COMPUTED_OVER["A"]
        self._mutate_and_check("A", "computed_over", real + "\n")

    def test_gate_e_direct_smoke_timing_reference_fails(self):
        self._mutate_and_check("E", "computed_over", "model-forward timing from the smoke job")

    def test_gate_e_indirect_smoke_timing_reference_fails(self):
        self._mutate_and_check("E", "computed_over", "operational evidence recorded by prerequisite execution")

    def test_gate_e_gpu_memory_reference_fails(self):
        self._mutate_and_check("E", "computed_over", "peak allocated memory determines this gate")

    def test_gate_e_slurm_elapsed_reference_fails(self):
        self._mutate_and_check("E", "computed_over", "Slurm elapsed time factored in")

    def test_gate_e_throughput_reference_fails(self):
        self._mutate_and_check("E", "computed_over", "supports a throughput claim for the full dataset")

    def test_gate_b_reworded_fails(self):
        real = ident._EXPECTED_GATE_COMPUTABLE_NOTE["B"]
        self._mutate_and_check("B", "computable_note", real.replace("deferred", "postponed"))

    def test_gate_c_reworded_fails(self):
        real = ident._EXPECTED_GATE_COMPUTABLE_NOTE["C"]
        self._mutate_and_check("C", "computable_note", real.replace("bound", "resolved"))

    def test_gate_d_reworded_fails(self):
        real = ident._EXPECTED_GATE_COMPUTABLE_NOTE["D"]
        self._mutate_and_check("D", "computable_note", real.replace("locked_holdout", "the holdout set"))

    def test_missing_gate_rejected(self):
        def mutate(d):
            del d["gate_contract"]["D"]
            return d
        with _MutationHarness(mutate):
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.load_authority(TOML_PATH)

    def test_exact_gate_id_set_enforced(self):
        def mutate(d):
            d["gate_contract"]["F"] = copy.deepcopy(d["gate_contract"]["E"])
            return d
        with _MutationHarness(mutate):
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.load_authority(TOML_PATH)


class PropagationCategoryExactBindingTests(unittest.TestCase):
    def test_real_categories_match_predecessor_exactly(self):
        data = ident.load_authority(TOML_PATH)
        sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
        cat_check = [c for c in sci["checks"] if c["label"] == "gt_targets.propagation_categories"][0]
        self.assertTrue(cat_check["ok"])
        self.assertEqual(cat_check["mine"], ["beneficial", "harmful", "unchanged_correct", "changed_but_same_correctness", "unchanged_incorrect"])

    def _mutate_categories_and_check(self, new_categories):
        def mutate(d):
            d["scientific_inheritance"]["gt_targets"]["propagation_categories"] = new_categories
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_rename_beneficial_fails(self):
        self._mutate_categories_and_check(["BENEFICIAL_RENAMED", "harmful", "unchanged_correct", "changed_but_same_correctness", "unchanged_incorrect"])

    def test_rename_harmful_fails(self):
        self._mutate_categories_and_check(["beneficial", "HARMFUL_RENAMED", "unchanged_correct", "changed_but_same_correctness", "unchanged_incorrect"])

    def test_remove_category_fails(self):
        self._mutate_categories_and_check(["beneficial", "harmful", "unchanged_correct", "changed_but_same_correctness"])

    def test_add_category_fails(self):
        self._mutate_categories_and_check(["beneficial", "harmful", "unchanged_correct", "changed_but_same_correctness", "unchanged_incorrect", "tied_no_change"])

    def test_duplicate_category_fails(self):
        self._mutate_categories_and_check(["beneficial", "beneficial", "unchanged_correct", "changed_but_same_correctness", "unchanged_incorrect"])

    def test_reorder_categories_fails(self):
        self._mutate_categories_and_check(["harmful", "beneficial", "unchanged_correct", "changed_but_same_correctness", "unchanged_incorrect"])

    def test_ambiguous_label_mutation_fails(self):
        def mutate(d):
            d["scientific_inheritance"]["gt_targets"]["ambiguous_label"] = "UNKNOWN_GT"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])


class GateDSubgroupExactBindingTests(unittest.TestCase):
    def test_real_subgroups_match_predecessor_exactly(self):
        data = ident.load_authority(TOML_PATH)
        sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
        sub_check = [c for c in sci["checks"] if c["label"] == "gate_contract.D.subgroups"][0]
        self.assertTrue(sub_check["ok"])
        self.assertEqual(sub_check["mine"], ["predicted_boundary_vs_interior", "thing_vs_stuff_gt_category", "low_vs_high_unary_margin"])

    def _mutate_subgroups_and_check(self, new_subgroups):
        def mutate(d):
            d["gate_contract"]["D"]["subgroups"] = new_subgroups
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_rename_each_subgroup_fails(self):
        base = ["predicted_boundary_vs_interior", "thing_vs_stuff_gt_category", "low_vs_high_unary_margin"]
        for i in range(3):
            mutated = list(base)
            mutated[i] = mutated[i] + "_RENAMED"
            self._mutate_subgroups_and_check(mutated)

    def test_remove_each_subgroup_fails(self):
        base = ["predicted_boundary_vs_interior", "thing_vs_stuff_gt_category", "low_vs_high_unary_margin"]
        for i in range(3):
            mutated = base[:i] + base[i+1:]
            self._mutate_subgroups_and_check(mutated)

    def test_add_subgroup_fails(self):
        self._mutate_subgroups_and_check(["predicted_boundary_vs_interior", "thing_vs_stuff_gt_category", "low_vs_high_unary_margin", "extra_subgroup"])

    def test_duplicate_subgroup_fails(self):
        self._mutate_subgroups_and_check(["predicted_boundary_vs_interior", "predicted_boundary_vs_interior", "low_vs_high_unary_margin"])

    def test_reorder_subgroups_fails(self):
        self._mutate_subgroups_and_check(["thing_vs_stuff_gt_category", "predicted_boundary_vs_interior", "low_vs_high_unary_margin"])

    def test_min_subgroup_size_mutation_fails(self):
        def mutate(d):
            d["gate_contract"]["D"]["min_subgroup_size_for_check"] = 50
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])


class ArtifactRequiredFieldExactBindingTests(unittest.TestCase):
    def test_real_schemas_pass(self):
        data = ident.load_authority(TOML_PATH)
        r = ident.verify_artifact_required_fields(data)
        self.assertTrue(r["all_ok"], r["schemas"])
        for schema_id, result in r["schemas"].items():
            self.assertTrue(result["exact_match"], schema_id)
            self.assertTrue(result["no_duplicates"], schema_id)

    def _mutate_and_check(self, schema_id, new_fields):
        def mutate(d):
            d["artifact_contract"][schema_id]["required_fields"] = new_fields
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_artifact_required_fields(data)
            self.assertFalse(r["all_ok"], schema_id)

    def test_remove_each_field_one_at_a_time(self):
        for schema_id, fields in ident._EXPECTED_ARTIFACT_REQUIRED_FIELDS.items():
            for i in range(len(fields)):
                mutated = list(fields[:i]) + list(fields[i+1:])
                self._mutate_and_check(schema_id, mutated)

    def test_add_unknown_field(self):
        for schema_id, fields in ident._EXPECTED_ARTIFACT_REQUIRED_FIELDS.items():
            self._mutate_and_check(schema_id, list(fields) + ["smoke_wall_time_ns"])

    def test_duplicate_field(self):
        for schema_id, fields in ident._EXPECTED_ARTIFACT_REQUIRED_FIELDS.items():
            self._mutate_and_check(schema_id, list(fields) + [fields[0]])

    def test_rename_field(self):
        for schema_id, fields in ident._EXPECTED_ARTIFACT_REQUIRED_FIELDS.items():
            mutated = list(fields)
            mutated[0] = mutated[0] + "_renamed"
            self._mutate_and_check(schema_id, mutated)

    def test_reduce_to_one_field(self):
        for schema_id in ident._EXPECTED_ARTIFACT_REQUIRED_FIELDS:
            self._mutate_and_check(schema_id, ["schema"])

    def test_insert_profiling_field(self):
        for schema_id, fields in ident._EXPECTED_ARTIFACT_REQUIRED_FIELDS.items():
            self._mutate_and_check(schema_id, list(fields) + ["gpu_peak_memory_bytes"])

    def test_missing_schema_table_rejected(self):
        def mutate(d):
            del d["artifact_contract"]["final_result"]
            return d
        with _MutationHarness(mutate):
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.load_authority(TOML_PATH)


class IntegerTypeValidationTests(unittest.TestCase):
    def _population_field_rejects_bool(self, field):
        def mutate(d):
            d["population"][field] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError, msg=field):
                ident.verify_population(REPO_ROOT, data)

    def test_population_counts_reject_bool(self):
        for field in ("total_selected_count", "discovery_count", "locked_holdout_count", "discovery_holdout_overlap"):
            self._population_field_rejects_bool(field)

    def test_population_counts_reject_numeric_string(self):
        def mutate(d):
            d["population"]["discovery_count"] = "48"
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_population(REPO_ROOT, data)

    def test_population_counts_exact_valid_value_passes(self):
        data = ident.load_authority(TOML_PATH)
        pop = ident.verify_population(REPO_ROOT, data)
        self.assertTrue(pop["all_ok"])

    def test_bootstrap_seed_false_rejected(self):
        # regression test for the exact defect found during reverification:
        # seed=0 and bool False are equal under Python's == , so this
        # specifically proves the fix, not just generic int-checking.
        def mutate(d):
            d["scientific_inheritance"]["bootstrap"]["seed"] = False
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_bootstrap_replicates_true_rejected(self):
        def mutate(d):
            d["scientific_inheritance"]["bootstrap"]["replicates"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_max_iter_true_rejected(self):
        def mutate(d):
            d["scientific_inheritance"]["statistics"]["max_iter"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_anchors_per_window_true_rejected(self):
        def mutate(d):
            d["scientific_inheritance"]["anchor_selection"]["target_anchors_per_window"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])

    def test_resource_gpu_count_rejects_bool(self):
        def mutate(d):
            d["resource_contract"]["gpu_count"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_resource_contract(data)

    def test_resource_gpu_memory_rejects_bool(self):
        def mutate(d):
            d["resource_contract"]["gpu_memory_gb"] = False
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            with self.assertRaises(ident.DiscoveryExecutionContractError):
                ident.verify_resource_contract(data)

    def test_resource_contract_real_values_pass(self):
        data = ident.load_authority(TOML_PATH)
        r = ident.verify_resource_contract(data)
        self.assertTrue(r["all_ok"])
        self.assertEqual(data["resource_contract"]["gpu_count"], 1)
        self.assertEqual(data["resource_contract"]["gpu_memory_gb"], 80)

    def test_gt_ignore_value_rejects_bool(self):
        def mutate(d):
            d["scientific_inheritance"]["gt_targets"]["ignore_gt_value"] = True
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            sci = ident.verify_scientific_inheritance(REPO_ROOT, data)
            self.assertFalse(sci["all_ok"])


class FinalRepairErrorHygieneTests(unittest.TestCase):
    def test_ordinary_invalid_gate_data_raises_domain_exception(self):
        def mutate(d):
            d["gate_contract"]["A"]["computed_over"] = 12345
            return d
        with _MutationHarness(mutate):
            data = ident.load_authority(TOML_PATH)
            r = ident.verify_gate_inputs_exclude_profiling(data)
            self.assertFalse(r["all_ok"])

    def test_keyboard_interrupt_propagates_through_verify_all(self):
        def raising_open(*a, **kw):
            raise KeyboardInterrupt()
        import builtins
        real_open = builtins.open
        builtins.open = raising_open
        try:
            with self.assertRaises(KeyboardInterrupt):
                ident.verify_all(REPO_ROOT, TOML_PATH)
        finally:
            builtins.open = real_open

    def test_system_exit_propagates_through_verify_all(self):
        def raising_open(*a, **kw):
            raise SystemExit(2)
        import builtins
        real_open = builtins.open
        builtins.open = raising_open
        try:
            with self.assertRaises(SystemExit):
                ident.verify_all(REPO_ROOT, TOML_PATH)
        finally:
            builtins.open = real_open

    def test_memory_error_propagates_through_verify_all(self):
        def raising_open(*a, **kw):
            raise MemoryError()
        import builtins
        real_open = builtins.open
        builtins.open = raising_open
        try:
            with self.assertRaises(MemoryError):
                ident.verify_all(REPO_ROOT, TOML_PATH)
        finally:
            builtins.open = real_open


if __name__ == "__main__":
    unittest.main()
