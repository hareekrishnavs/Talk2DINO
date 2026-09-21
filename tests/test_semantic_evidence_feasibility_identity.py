"""Focused tests for src/semantic_evidence_feasibility_identity.py.

Deliberately uses the standard-library `unittest` framework (not pytest) so
these tests can run in a plain, Torch-free interpreter -- no venv, no
project dependencies. Run with:

    python3 -m unittest tests.test_semantic_evidence_feasibility_identity -v

These tests do not execute model inference, load any checkpoint tensor, or
generate any intervention. Section 15 requires the 96-image subset to be
independently reconstructed in tests WITHOUT using the production
serializer as the oracle: `_reference_reconstruct_selection` below is a
standalone re-implementation, written independently of
`semantic_evidence_feasibility_identity.reconstruct_selection`, used to
cross-check both the production function and the on-disk JSON file.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
PROTOCOL_PATH = REPO_ROOT / "research_specs" / "semantic_evidence_feasibility_v1.toml"
SUBSET_PATH = REPO_ROOT / "evaluation_subsets" / "semantic_evidence_feasibility_96.json"
DOCS_PATH = REPO_ROOT / "docs" / "semantic_evidence_feasibility_protocol.md"
FILENAME_UNIVERSE_PATH = Path(
    "/scratch/haree/semantic_evidence_interventions/step0_clean_e3_baseline_20260920T235625Z/"
    "coco_stuff_val2017_image_filenames_sorted.txt"
)

sys.path.insert(0, str(SRC_DIR))
import semantic_evidence_feasibility_identity as ident  # noqa: E402


def _load_toml_dict() -> dict:
    with open(PROTOCOL_PATH, "rb") as f:
        return tomllib.load(f)


def _load_json_dict() -> dict:
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_toml(data: dict) -> Path:
    """Serialize a (possibly mutated) protocol dict back to a temp TOML file
    using a minimal hand-rolled writer sufficient for this contract's value
    types (no external toml-writer dependency)."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    _write_toml_value(tmp, data, path=[])
    tmp.close()
    return Path(tmp.name)


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


def _write_toml_value(fh, data: dict, path: list[str]) -> None:
    scalars = {}
    tables = {}
    array_of_tables = {}
    for k, v in data.items():
        if isinstance(v, dict):
            tables[k] = v
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            array_of_tables[k] = v
        else:
            scalars[k] = v

    if path:
        fh.write(f"\n[{'.'.join(path)}]\n")
    for k, v in scalars.items():
        fh.write(f"{k} = {_toml_scalar(v)}\n")
    for k, v in tables.items():
        _write_toml_value(fh, v, path + [k])
    for k, v in array_of_tables.items():
        for entry in v:
            fh.write(f"\n[[{'.'.join(path + [k])}]]\n")
            for kk, vv in entry.items():
                if isinstance(vv, dict):
                    raise TypeError("nested tables inside array-of-tables not needed by this contract")
                fh.write(f"{kk} = {_toml_scalar(vv)}\n")


def _write_json(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp, indent=2)
    tmp.close()
    return Path(tmp.name)


def _valid_protocol_dict() -> dict:
    return copy.deepcopy(_load_toml_dict())


def _valid_subset_dict() -> dict:
    return copy.deepcopy(_load_json_dict())


# ---------------------------------------------------------------------------
# Independent (non-production) reference reconstruction of the 96-subset
# ---------------------------------------------------------------------------

def _reference_selection_key(namespace: str, canonical_id: str) -> str:
    # Independently written, not calling ident.selection_key.
    payload = namespace.encode("utf-8") + b"\x00" + canonical_id.encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _reference_digest(ids_in_order: list[str]) -> str:
    hasher = hashlib.sha256()
    for cid in ids_in_order:
        hasher.update(cid.encode("ascii") + b"\n")
    return hasher.hexdigest()


def _reference_reconstruct_selection(canonical_ids: list[str], namespace: str) -> dict:
    keyed = [(_reference_selection_key(namespace, cid), cid) for cid in canonical_ids]
    keyed.sort()
    top96 = keyed[:96]
    records = []
    for rank, (key, cid) in enumerate(top96):
        records.append(
            {
                "canonical_image_id": cid,
                "selection_key": key,
                "selection_rank": rank,
                "partition": "discovery" if rank <= 47 else "locked_holdout",
            }
        )
    discovery = [r["canonical_image_id"] for r in records if r["partition"] == "discovery"]
    holdout = [r["canonical_image_id"] for r in records if r["partition"] == "locked_holdout"]
    return {
        "records": records,
        "canonical_source_digest": _reference_digest(sorted(canonical_ids)),
        "selected_record_digest": _reference_digest([r["canonical_image_id"] for r in records]),
        "discovery_digest": _reference_digest(discovery),
        "locked_holdout_digest": _reference_digest(holdout),
    }


def _synthetic_canonical_ids(n: int = 500) -> list[str]:
    return [f"{i:012d}" for i in range(n)]


class ValidLoadTests(unittest.TestCase):
    def test_valid_protocol_loads(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["schema"]["name"], "semantic_evidence_feasibility_contract")

    def test_valid_subset_loads_without_universe(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        subset = ident.load_and_verify_subset(SUBSET_PATH, protocol)
        self.assertEqual(len(subset["records"]), 96)

    @unittest.skipUnless(FILENAME_UNIVERSE_PATH.is_file(), "frozen dataset filename universe not present on this filesystem")
    def test_valid_subset_reconstructs_against_real_universe(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        with open(FILENAME_UNIVERSE_PATH, "r", encoding="ascii") as f:
            canonical_ids = ident.canonical_ids_from_filename_listing(f.read())
        subset = ident.load_and_verify_subset(SUBSET_PATH, protocol, canonical_ids=canonical_ids)
        self.assertEqual(len(subset["records"]), 96)

        # Independent, non-production reconstruction cross-check.
        ref = _reference_reconstruct_selection(canonical_ids, protocol["population_selection"]["namespace"])
        self.assertEqual(ref["records"], subset["records"])
        self.assertEqual(ref["canonical_source_digest"], subset["canonical_source_digest"])
        self.assertEqual(ref["selected_record_digest"], subset["selected_record_digest"])
        self.assertEqual(ref["discovery_digest"], subset["discovery_digest"])
        self.assertEqual(ref["locked_holdout_digest"], subset["locked_holdout_digest"])

    def test_reference_reconstruction_matches_production_on_synthetic_universe(self):
        ids = _synthetic_canonical_ids(500)
        ns = "talk2dino-semantic-evidence-feasibility-v1"
        ref = _reference_reconstruct_selection(ids, ns)
        prod = ident.reconstruct_selection(ids, ns)
        self.assertEqual(ref["records"], prod["records"])
        self.assertEqual(ref["canonical_source_digest"], prod["canonical_source_digest"])


class RepositoryBindingTests(unittest.TestCase):
    def test_exact_dependency_hashes_pass(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        result = ident.verify_repository_bindings(REPO_ROOT, protocol)
        self.assertEqual(result["dependencies_verified"], len(protocol["dependency"]))

    def test_commit_binding_passes_on_current_head(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        result = ident.verify_commit_binding(REPO_ROOT, protocol)
        self.assertTrue(result["exact_match"])

    def test_path_substitution_detected(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        # Point a dependency at a different, still-existing, real file.
        protocol["dependency"][0]["path"] = "src/local_weights.py"
        with self.assertRaises(ident.ContractError):
            ident.verify_repository_bindings(REPO_ROOT, protocol)

    def test_altered_dependency_bytes_detected(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        real_dep = protocol["dependency"][0]
        real_path = REPO_ROOT / real_dep["path"]
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_repo = Path(tmpdir)
            (fake_repo / os.path.dirname(real_dep["path"])).mkdir(parents=True, exist_ok=True)
            original_bytes = real_path.read_bytes()
            tampered = bytearray(original_bytes)
            tampered[0] = (tampered[0] + 1) % 256
            (fake_repo / real_dep["path"]).write_bytes(bytes(tampered))
            with self.assertRaises(ident.ContractError):
                ident.verify_repository_bindings(fake_repo, {"dependency": [real_dep]})

    def test_lexical_traversal_rejected(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        protocol["dependency"][0]["path"] = "../etc/passwd"
        with self.assertRaises(ident.ContractError):
            ident.verify_repository_bindings(REPO_ROOT, protocol)

    def test_lexical_traversal_rejected_embedded(self):
        with self.assertRaises(ident.ContractError):
            ident.resolve_repo_relative_path(REPO_ROOT, "src/../../etc/passwd")

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as outside, tempfile.TemporaryDirectory() as repo_dir:
            repo_root = Path(repo_dir)
            outside_file = Path(outside) / "secret.txt"
            outside_file.write_text("outside")
            link_path = repo_root / "escape_link.txt"
            os.symlink(outside_file, link_path)
            with self.assertRaises(ident.ContractError):
                ident.resolve_repo_relative_path(repo_root, "escape_link.txt")


class SchemaValidationTests(unittest.TestCase):
    def test_unknown_top_level_key_rejected(self):
        data = _valid_protocol_dict()
        data["unexpected_top_level_key"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_missing_top_level_key_rejected(self):
        data = _valid_protocol_dict()
        del data["gates"]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_unknown_nested_authorization_key_rejected(self):
        data = _valid_protocol_dict()
        data["authorization"]["extra_flag"] = False
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_missing_nested_authorization_key_rejected(self):
        data = _valid_protocol_dict()
        del data["authorization"]["gpu_execution_authorized"]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_wrong_scalar_type_rejected(self):
        data = _valid_protocol_dict()
        data["dataset"]["class_count"] = "171"  # str instead of int
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_bool_as_int_rejected(self):
        data = _valid_protocol_dict()
        data["authorization"]["gpu_execution_authorized"] = 0  # int, not bool False
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_bool_as_int_rejected_true_side(self):
        data = _valid_protocol_dict()
        data["authorization"]["protocol_definition_authorized"] = 1  # int, not bool True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_nan_rejected_in_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"x": NaN}')
            path = Path(f.name)
        with self.assertRaises(ident.ContractError):
            ident.load_strict_json(path)

    def test_infinity_rejected_in_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"x": Infinity}')
            path = Path(f.name)
        with self.assertRaises(ident.ContractError):
            ident.load_strict_json(path)

    def test_check_finite_rejects_nan_in_structure(self):
        with self.assertRaises(ident.ContractError):
            ident._check_finite({"a": [1.0, float("nan")]}, "root")

    def test_check_finite_rejects_inf_in_structure(self):
        with self.assertRaises(ident.ContractError):
            ident._check_finite({"a": float("inf")}, "root")


class SubsetValidationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = ident.load_protocol(PROTOCOL_PATH)

    def test_missing_record_rejected(self):
        data = _valid_subset_dict()
        data["records"].pop()
        path = _write_json(data)
        with self.assertRaises(ident.ContractError):
            ident.load_and_verify_subset(path, self.protocol)

    def test_extra_record_rejected(self):
        data = _valid_subset_dict()
        extra = copy.deepcopy(data["records"][0])
        extra["canonical_image_id"] = "999999999999"
        extra["selection_key"] = ident.selection_key(data["namespace"], "999999999999")
        data["records"].append(extra)
        path = _write_json(data)
        with self.assertRaises(ident.ContractError):
            ident.load_and_verify_subset(path, self.protocol)

    def test_duplicate_record_rejected(self):
        data = _valid_subset_dict()
        data["records"][1] = copy.deepcopy(data["records"][0])
        data["records"][1]["selection_rank"] = 1
        path = _write_json(data)
        with self.assertRaises(ident.ContractError):
            ident.load_and_verify_subset(path, self.protocol)

    def test_changed_rank_rejected(self):
        data = _valid_subset_dict()
        data["records"][5]["selection_rank"] = 999
        path = _write_json(data)
        with self.assertRaises(ident.ContractError):
            ident.load_and_verify_subset(path, self.protocol)

    def test_changed_selection_key_rejected(self):
        data = _valid_subset_dict()
        data["records"][5]["selection_key"] = "0" * 64
        path = _write_json(data)
        with self.assertRaises(ident.ContractError):
            ident.load_and_verify_subset(path, self.protocol)

    def test_changed_partition_rejected(self):
        data = _valid_subset_dict()
        data["records"][0]["partition"] = "locked_holdout"
        path = _write_json(data)
        with self.assertRaises(ident.ContractError):
            ident.load_and_verify_subset(path, self.protocol)

    def test_discovery_holdout_overlap_rejected(self):
        data = _valid_subset_dict()
        # Force an overlap: duplicate a discovery id into a holdout slot's id
        # while keeping array length/ranks intact (still must fail: either
        # duplicate-id check or overlap check will catch it).
        data["records"][48]["canonical_image_id"] = data["records"][0]["canonical_image_id"]
        data["records"][48]["selection_key"] = data["records"][0]["selection_key"]
        path = _write_json(data)
        with self.assertRaises(ident.ContractError):
            ident.load_and_verify_subset(path, self.protocol)

    def test_gt_dependent_selection_mutation_detected(self):
        # Simulate a "GT-informed" re-ordering: swap the boundary pair
        # between discovery and holdout without recomputing hash order.
        # This does not correspond to any valid (selection_key, id) sort and
        # must be caught by the independent-reconstruction cross-check.
        data = _valid_subset_dict()
        data["records"][47], data["records"][48] = data["records"][48], data["records"][47]
        data["records"][47]["selection_rank"], data["records"][48]["selection_rank"] = 47, 48
        data["records"][47]["partition"] = "discovery"
        data["records"][48]["partition"] = "locked_holdout"
        path = _write_json(data)
        ids = _synthetic_canonical_ids(0)  # irrelevant; real-universe check below is what matters
        if FILENAME_UNIVERSE_PATH.is_file():
            with open(FILENAME_UNIVERSE_PATH, "r", encoding="ascii") as f:
                canonical_ids = ident.canonical_ids_from_filename_listing(f.read())
            with self.assertRaises(ident.ContractError):
                ident.load_and_verify_subset(path, self.protocol, canonical_ids=canonical_ids)
        else:
            # Without the real universe, the tampered ordering still fails
            # because it no longer matches the frozen digests carried over
            # from the original file (digests were not recomputed above).
            with self.assertRaises(ident.ContractError):
                ident.load_and_verify_subset(path, self.protocol)

    def test_changed_namespace_rejected(self):
        data = _valid_subset_dict()
        data["namespace"] = "talk2dino-semantic-evidence-feasibility-v2"
        path = _write_json(data)
        with self.assertRaises(ident.ContractError):
            ident.load_and_verify_subset(path, self.protocol)


class FrozenDefinitionMutationTests(unittest.TestCase):
    def test_changed_reliability_formula_rejected(self):
        data = _valid_protocol_dict()
        data["scores"]["formulas"]["intervention_reliability_q"] = "context_stability(q) + region_necessity(q)"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_control_formula_rejected(self):
        data = _valid_protocol_dict()
        for c in data["control"]:
            if c["id"] == 4:
                c["formula"] = "some different formula"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_gaussian_kernel_rejected(self):
        data = _valid_protocol_dict()
        data["gaussian_blur"]["kernel_size"] = [15, 15]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_gaussian_sigma_rejected(self):
        data = _valid_protocol_dict()
        data["gaussian_blur"]["sigma"] = [4.0, 4.0]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_gaussian_randomness_flag_rejected(self):
        data = _valid_protocol_dict()
        data["gaussian_blur"]["randomness"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_anchor_count_rejected(self):
        data = _valid_protocol_dict()
        data["anchor_selection"]["target_anchors_per_window"] = 4
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_strata_rejected(self):
        data = _valid_protocol_dict()
        data["anchor_selection"]["strata"] = ["boundary_low", "boundary_high", "interior_low"]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_boundary_definition_rejected(self):
        data = _valid_protocol_dict()
        data["anchor_selection"]["boundary_definition"] = "a different boundary rule entirely"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)


class StatisticsMutationTests(unittest.TestCase):
    def test_changed_logistic_regression_C_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["model_settings"]["C"] = 10.0
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_logistic_regression_solver_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["model_settings"]["solver"] = "lbfgs"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_bootstrap_replicates_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["bootstrap"]["replicates"] = 1000
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_bootstrap_seed_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["bootstrap"]["seed"] = 42
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_changed_bootstrap_unit_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["bootstrap"]["resample_unit"] = "anchor (not image)"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)


class GateAndAuthorizationMutationTests(unittest.TestCase):
    def test_relaxed_gate_b_threshold_rejected(self):
        data = _valid_protocol_dict()
        data["gates"]["B"]["min_auroc_improvement_holdout"] = 0.01
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_relaxed_gate_d_threshold_rejected(self):
        data = _valid_protocol_dict()
        data["gates"]["D"]["max_negative_auroc_difference"] = -0.05
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_relaxed_gate_e_compute_budget_rejected(self):
        data = _valid_protocol_dict()
        data["gates"]["E"]["max_h100_hours_96_image_feasibility_run"] = 20.0
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_broadened_authorization_flag_rejected(self):
        for flag in ident._REQUIRED_FALSE_AUTHORIZATIONS:
            with self.subTest(flag=flag):
                data = _valid_protocol_dict()
                data["authorization"][flag] = True
                path = _write_toml(data)
                with self.assertRaises(ident.ContractError):
                    ident.load_protocol(path)

    def test_narrowed_authorization_flag_rejected(self):
        for flag in ident._REQUIRED_TRUE_AUTHORIZATIONS:
            with self.subTest(flag=flag):
                data = _valid_protocol_dict()
                data["authorization"][flag] = False
                path = _write_toml(data)
                with self.assertRaises(ident.ContractError):
                    ident.load_protocol(path)


class TorchFreeImportTests(unittest.TestCase):
    def test_import_in_fresh_interpreter_does_not_pull_in_torch(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(SRC_DIR)!r}); "
            "import semantic_evidence_feasibility_identity; "
            "assert 'torch' not in sys.modules, 'torch was imported'; "
            "assert 'mmcv' not in sys.modules, 'mmcv was imported'; "
            "assert 'cv2' not in sys.modules, 'cv2 was imported'; "
            "print('OK')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)


class FatalExceptionPropagationTests(unittest.TestCase):
    def test_reraise_if_fatal_reraises_keyboard_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):
            ident._reraise_if_fatal(KeyboardInterrupt())

    def test_reraise_if_fatal_reraises_system_exit(self):
        with self.assertRaises(SystemExit):
            ident._reraise_if_fatal(SystemExit())

    def test_reraise_if_fatal_reraises_memory_error(self):
        with self.assertRaises(MemoryError):
            ident._reraise_if_fatal(MemoryError())

    def test_reraise_if_fatal_ignores_ordinary_exception(self):
        # Must NOT raise for a plain Exception.
        ident._reraise_if_fatal(ValueError("ordinary"))

    def test_run_git_propagates_memory_error(self):
        with mock.patch("subprocess.run", side_effect=MemoryError("simulated OOM")):
            with self.assertRaises(MemoryError):
                ident._run_git(REPO_ROOT, "rev-parse", "HEAD")

    def test_run_git_propagates_keyboard_interrupt(self):
        with mock.patch("subprocess.run", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                ident._run_git(REPO_ROOT, "rev-parse", "HEAD")

    def test_run_git_wraps_ordinary_failure_as_contract_error(self):
        with mock.patch("subprocess.run", side_effect=OSError("no such file")):
            with self.assertRaises(ident.ContractError):
                ident._run_git(REPO_ROOT, "rev-parse", "HEAD")


class GateReportingTests(unittest.TestCase):
    def test_report_gate_configuration_returns_all_gates(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        gates = ident.report_gate_configuration(protocol)
        self.assertEqual(set(gates.keys()), {"A", "B", "C", "D", "E", "outcome"})
        self.assertEqual(gates["outcome"]["proceed_value"], "PROCEED_TO_DEPLOYABLE_SIGNAL_SPECIFICATION")
        self.assertEqual(gates["outcome"]["stop_value"], "STOP_SEMANTIC_EVIDENCE_INTERVENTION_TRACK")


# =============================================================================
# REPAIR TESTS: Finding 1 (T0/T320 propagation-target authority)
# =============================================================================

class PropagationTargetAuthorityTests(unittest.TestCase):
    def test_valid_contract_has_t0_bound_and_t320_deferred(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        pta = ident.report_propagation_target_authority(protocol)
        self.assertEqual(pta["t0_status"], "BOUND_TO_COMMITTED_E3_UNARY_INFERENCE")
        self.assertEqual(pta["t320_status"], "DEFERRED_REQUIRES_SUCCESSOR_AUTHORITY")
        self.assertFalse(pta["t320_source_bound"])
        self.assertFalse(pta["t320_execution_authorized"])

    def test_feasibility_execution_not_authorized_while_t320_deferred(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        self.assertFalse(ident.is_feasibility_execution_authorized(protocol))

    def test_missing_propagation_target_authority_table_rejected(self):
        data = _valid_protocol_dict()
        del data["propagation_target_authority"]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_falsely_setting_t320_source_bound_true_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["t320_source_bound"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_authorizing_t320_execution_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["t320_execution_authorized"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_authorizing_graph_construction_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["graph_construction_authorized"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_authorizing_recurrence_execution_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["recurrence_execution_authorized"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_authorizing_cross_branch_source_use_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["cross_branch_source_use_authorized"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_treating_bytecode_as_authority_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["ignored_bytecode_is_authority"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_treating_residue_as_source_present_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["residue_disclosure"]["present_as_source_on_this_branch"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_removing_successor_authority_prerequisite_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["feasibility_execution_blocked_until_t320_binding"] = False
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_empty_required_successor_action_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["required_successor_action"] = ""
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_t0_not_bound_to_q1_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["t0_definition"]["identity_with_query"] = "T0 is unrelated to any query"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_t320_specified_by_contract_true_rejected(self):
        data = _valid_protocol_dict()
        data["propagation_target_authority"]["t320_definition"]["specified_by_this_contract"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_execution_authorization_flag_while_t320_deferred_rejected(self):
        # Even if someone flips ONLY the outcome-level flag (leaving
        # t320_status deferred), the loader must still reject it.
        data = _valid_protocol_dict()
        data["gates"]["outcome"]["feasibility_execution_authorized"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_gate_c_computable_true_while_t320_deferred_rejected(self):
        data = _valid_protocol_dict()
        data["gates"]["C"]["computable_under_this_contract"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_propagation_outcome_missing_changed_but_same_correctness_category_rejected(self):
        data = _valid_protocol_dict()
        data["gt_targets"]["propagation_outcome"]["categories"] = [
            "beneficial", "harmful", "unchanged_correct", "unchanged_incorrect",
        ]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)


# =============================================================================
# REPAIR TESTS: Finding 2 (two statistical tasks, four model fits)
# =============================================================================

class StatisticalTaskTests(unittest.TestCase):
    def test_valid_contract_has_two_tasks_and_four_fits(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["statistics"]["task_identifiers"], ["query_correctness", "propagation_outcome"])
        self.assertEqual(len(protocol["statistics"]["model_fits"]), 4)

    def test_query_correctness_unit_is_anchor_query(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["statistics"]["tasks"]["query_correctness"]["unit_of_analysis"], "anchor_query")
        self.assertEqual(protocol["statistics"]["tasks"]["query_correctness"]["rows_per_accepted_anchor"], 2)

    def test_propagation_outcome_unit_is_anchor(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["statistics"]["tasks"]["propagation_outcome"]["unit_of_analysis"], "anchor")
        self.assertEqual(protocol["statistics"]["tasks"]["propagation_outcome"]["rows_per_eligible_anchor"], 1)

    def test_generic_unspecified_unit_of_analysis_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["tasks"]["query_correctness"]["unit_of_analysis"] = "record"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_query_correctness_row_missing_query_rank_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["tasks"]["query_correctness"]["query_ranks"] = ["q1"]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_query_correctness_control1_hardcoded_to_q1_rejected(self):
        data = _valid_protocol_dict()
        for c in data["control"]:
            if c["id"] == 1:
                c["formula"] = "s_original(query_1) -- hardcoded, not parameterized by the row's query"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_query_correctness_control8_averaged_over_both_queries_rejected(self):
        data = _valid_protocol_dict()
        for c in data["control"]:
            if c["id"] == 8:
                c["formula"] = "mean over the anchor's two queries (query_1, query_2) of control_change(q)"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_propagation_outcome_feature_referencing_q2_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["tasks"]["propagation_outcome"]["feature_order"][0]["parameterization"] = "s_original(q2), q fixed at q2"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_propagation_outcome_query_used_q2_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["tasks"]["propagation_outcome"]["query_used"] = "q2"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_wrong_model_fit_count_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["model_fits"].pop()
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_duplicate_task_variant_model_fit_rejected(self):
        data = _valid_protocol_dict()
        # Make fit 2 a duplicate of fit 1's (task, variant) pair.
        data["statistics"]["model_fits"][1]["task"] = data["statistics"]["model_fits"][0]["task"]
        data["statistics"]["model_fits"][1]["variant"] = data["statistics"]["model_fits"][0]["variant"]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_metric_mapped_to_wrong_task_rejected(self):
        data = _valid_protocol_dict()
        for m in data["statistics"]["primary_metrics"]:
            if m["id"] == 1:
                m["task"] = "propagation_outcome"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_metric_producing_fit_ids_wrong_rejected(self):
        data = _valid_protocol_dict()
        for m in data["statistics"]["primary_metrics"]:
            if m["id"] == 3:
                m["producing_model_fit_ids"] = [1, 2]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_propagation_outcome_metric_marked_computable_rejected(self):
        data = _valid_protocol_dict()
        for m in data["statistics"]["primary_metrics"]:
            if m["id"] == 3:
                m["computable_under_this_contract"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_gate_b_task_mismatch_rejected(self):
        data = _valid_protocol_dict()
        data["gates"]["B"]["task"] = "propagation_outcome"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_gate_c_task_mismatch_rejected(self):
        data = _valid_protocol_dict()
        data["gates"]["C"]["task"] = "query_correctness"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_bootstrap_resampling_rows_instead_of_images_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["bootstrap"]["resample_unit"] = "anchor row (not image)"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_bootstrap_refit_inside_replicates_true_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["bootstrap"]["refit_inside_replicates"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_bootstrap_not_paired_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["bootstrap"]["paired"] = False
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_bootstrap_replicate_count_changed_rejected(self):
        data = _valid_protocol_dict()
        data["statistics"]["bootstrap"]["replicates"] = 1000
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)


# =============================================================================
# REPAIR TESTS: Finding 3 (fail-closed formula input/output validity)
# =============================================================================

class FormulaValidityTests(unittest.TestCase):
    def test_valid_contract_requires_input_and_output_finiteness(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        validity = protocol["scores"]["validity"]
        self.assertTrue(validity["input_finiteness_required"])
        self.assertEqual(validity["input_range_required"], [-1.0, 1.0])
        self.assertTrue(validity["output_finiteness_required"])
        self.assertEqual(validity["output_range_required"], [0.0, 1.0])

    def test_removing_input_finiteness_requirement_rejected(self):
        data = _valid_protocol_dict()
        data["scores"]["validity"]["input_finiteness_required"] = False
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_removing_output_finiteness_requirement_rejected(self):
        data = _valid_protocol_dict()
        data["scores"]["validity"]["output_finiteness_required"] = False
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_widening_input_range_rejected(self):
        data = _valid_protocol_dict()
        data["scores"]["validity"]["input_range_required"] = [-2.0, 2.0]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_removing_prohibited_clamp_action_rejected(self):
        data = _valid_protocol_dict()
        data["scores"]["validity"]["input_failure_prohibited_actions"] = ["impute", "replace_with_default", "silently_launder_into_a_finite_output"]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_missing_scores_validity_table_rejected(self):
        data = _valid_protocol_dict()
        del data["scores"]["validity"]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    # --- Regression tests: NaN/Infinity cannot become a finite, valid
    # intervention_reliability THROUGH THE FORMULAS ALONE (Python min/max
    # behavior). These document the exact hazard the input-validity
    # precondition exists to prevent -- run entirely independently of any
    # production code, using only the frozen formula text as a spec.

    @staticmethod
    def _raw_formula(s_original, s_context, s_region, s_control):
        """Direct transliteration of [scores.formulas], WITHOUT the
        mandatory [scores.validity] precondition -- i.e. exactly what a
        naive implementation would compute if it skipped the precondition.
        Used only to demonstrate the hazard, never as sanctioned behavior."""
        context_change = abs(s_context - s_original)
        region_drop = max(0, s_original - s_region)
        control_change = abs(s_control - s_original)
        context_stability = 1 - min(1, context_change / 2)
        specific_region_drop = max(0, region_drop - control_change)
        region_necessity = min(1, specific_region_drop / 2)
        return context_stability * region_necessity

    def test_nan_original_score_is_laundered_by_raw_formula_without_precondition(self):
        import math
        result = self._raw_formula(float("nan"), 0.5, 0.5, 0.5)
        # This IS finite and in-range -- demonstrating exactly why the
        # contract's mandatory precondition (checked before the formula
        # runs, not after) is necessary; Gate A's output-only check alone
        # would NOT catch this.
        self.assertFalse(math.isnan(result))
        self.assertTrue(0.0 <= result <= 1.0)

    def test_infinity_original_score_is_laundered_by_raw_formula_without_precondition(self):
        import math
        result = self._raw_formula(float("inf"), 0.0, 0.0, 0.0)
        self.assertFalse(math.isnan(result))
        self.assertTrue(0.0 <= result <= 1.0)

    def test_precondition_would_reject_nan_before_formula_runs(self):
        # This is the actual contract behavior: NaN fails input_finiteness
        # BEFORE the formula in _raw_formula above is ever allowed to run.
        def guarded_formula(s_original, s_context, s_region, s_control):
            for s in (s_original, s_context, s_region, s_control):
                if not (isinstance(s, (int, float)) and s == s and abs(s) != float("inf") and -1.0 <= s <= 1.0):
                    raise ident.ContractError("input failed finiteness/range precondition")
            return FormulaValidityTests._raw_formula(s_original, s_context, s_region, s_control)

        with self.assertRaises(ident.ContractError):
            guarded_formula(float("nan"), 0.5, 0.5, 0.5)
        with self.assertRaises(ident.ContractError):
            guarded_formula(float("inf"), 0.0, 0.0, 0.0)
        with self.assertRaises(ident.ContractError):
            guarded_formula(1.5, 0.0, 0.0, 0.0)  # out of [-1,1] range
        # A fully valid input set must still pass through untouched.
        self.assertEqual(guarded_formula(1.0, 1.0, -1.0, 1.0), 1.0)


# =============================================================================
# REPAIR TESTS: Finding 4 (Gaussian-blur reflect padding)
# =============================================================================

class BlurPaddingTests(unittest.TestCase):
    def test_valid_contract_requires_reflect_padding(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["gaussian_blur"]["padding_mode"], "reflect")

    def test_wrong_padding_mode_constant_rejected(self):
        data = _valid_protocol_dict()
        data["gaussian_blur"]["padding_mode"] = "constant"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_wrong_padding_mode_replicate_rejected(self):
        data = _valid_protocol_dict()
        data["gaussian_blur"]["padding_mode"] = "replicate"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_wrong_padding_mode_circular_rejected(self):
        data = _valid_protocol_dict()
        data["gaussian_blur"]["padding_mode"] = "circular"
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_missing_padding_mode_field_rejected(self):
        data = _valid_protocol_dict()
        del data["gaussian_blur"]["padding_mode"]
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)


# =============================================================================
# REPAIR TESTS: Finding 5 (renamed external projection-checkpoint field)
# =============================================================================

class ProjectionCheckpointFieldTests(unittest.TestCase):
    def test_valid_contract_uses_new_field_name(self):
        protocol = ident.load_protocol(PROTOCOL_PATH)
        ext = protocol["external_weights"]["e3_projection_checkpoint"]
        self.assertIn("external_untracked_checkout_relative_path", ext)
        self.assertNotIn("repository_relative_location", ext)
        self.assertFalse(ext["git_tracked"])
        self.assertEqual(ext["sha256"], "9632117c18613674eeddcb35f86193d37da203a9cbd837cf500c4121b5fa9942")

    def test_old_field_name_present_rejected(self):
        data = _valid_protocol_dict()
        ext = data["external_weights"]["e3_projection_checkpoint"]
        ext["repository_relative_location"] = ext.pop("external_untracked_checkout_relative_path")
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_git_tracked_true_rejected(self):
        data = _valid_protocol_dict()
        data["external_weights"]["e3_projection_checkpoint"]["git_tracked"] = True
        path = _write_toml(data)
        with self.assertRaises(ident.ContractError):
            ident.load_protocol(path)

    def test_checkpoint_hash_actually_unchanged_on_disk(self):
        # Cross-check against the real repository-relative checkpoint file,
        # independent of the TOML's own declared value.
        checkpoint_path = REPO_ROOT / "weights" / "vitb_mlp_infonce_paired_soft_routing_tau010.pth"
        if not checkpoint_path.is_file():
            self.skipTest("checkpoint not present on this filesystem")
        actual = ident.sha256_file(checkpoint_path)
        self.assertEqual(actual, "9632117c18613674eeddcb35f86193d37da203a9cbd837cf500c4121b5fa9942")


# =============================================================================
# DOCUMENTATION / SCHEMA MECHANICAL CONSISTENCY
# =============================================================================

class DocumentationMechanicalConsistencyTests(unittest.TestCase):
    def test_docs_mention_t320_deferred(self):
        text = DOCS_PATH.read_text()
        self.assertIn("DEFERRED", text)

    def test_docs_mention_reflect_padding(self):
        text = DOCS_PATH.read_text()
        self.assertIn("reflect", text.lower())

    def test_docs_mention_two_tasks(self):
        text = DOCS_PATH.read_text()
        self.assertIn("query_correctness", text)
        self.assertIn("propagation_outcome", text)

    def test_docs_mention_four_model_fits(self):
        text = DOCS_PATH.read_text()
        self.assertTrue("four" in text.lower() or "4" in text)

    def test_docs_do_not_claim_feasibility_authorized(self):
        text = DOCS_PATH.read_text().lower()
        # The docs must not claim the 96-image study is currently authorized.
        self.assertNotIn("the 96-image study is authorized", text)
        self.assertNotIn("feasibility study is authorized", text)


if __name__ == "__main__":
    unittest.main()
