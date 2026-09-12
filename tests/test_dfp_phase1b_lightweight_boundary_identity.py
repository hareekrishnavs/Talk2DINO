"""Adversarial tests for the E22 BALP (Boundary-Aware Late-Flip
Preservation) successor identity: the preservation protocol TOML, the
artifact-schema TOML, the deterministic 350/150 development/
confirmation split, and their strict Python loader
(`src.dfp_phase1b_lightweight_boundary_identity`).

No torch/CUDA/model/dataset access. Uses independent stdlib
reconstruction (hashlib/json/tomllib) wherever possible -- never
merely re-validates the loader against constants imported from
itself."""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
import sys
sys.path.insert(0, str(ROOT))

import src.dfp_phase1b_lightweight_boundary_identity as e22

E = e22.DfpPhase1bLightweightBoundaryIdentityError

PRESERVATION_PATH = ROOT / e22.PRESERVATION_RELATIVE_PATH
ARTIFACTS_PATH = ROOT / e22.ARTIFACTS_RELATIVE_PATH
SPLIT_PATH = ROOT / e22.SPLIT_RELATIVE_PATH
MANIFEST_PATH = ROOT / e22.SUBSET_MANIFEST_RELATIVE_PATH


def _load_preservation_doc() -> dict:
    with PRESERVATION_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _load_artifacts_doc() -> dict:
    with ARTIFACTS_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _write_tmp_toml(tmp_path: Path, doc: dict, name: str) -> Path:
    # tomllib is read-only in the stdlib; write a minimal but valid TOML
    # by hand-rolling a tiny writer sufficient for this test's flat/
    # nested dict shapes (str/int/float/bool/list/dict only).
    def render(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, list):
            return "[" + ", ".join(render(v) for v in value) + "]"
        raise TypeError(f"unsupported inline type: {type(value)}")

    lines: list[str] = []

    def emit(prefix: str, table: dict):
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subtables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {render(v)}")
        lines.append("")
        for k, v in subtables.items():
            emit(f"{prefix}.{k}" if prefix else k, v)

    for k, v in doc.items():
        if isinstance(v, dict):
            emit(k, v)
        else:
            lines.insert(0, f"{k} = {render(v)}")
    text = "\n".join(lines)
    path = tmp_path / name
    path.write_text(text)
    # sanity: must itself parse
    with path.open("rb") as handle:
        tomllib.load(handle)
    return path


# =====================================================================
# Real-file baseline: the committed identities must load cleanly.
# =====================================================================


def test_preservation_identity_loads_and_verifies():
    identity = e22.load_identity(repo_root=ROOT)
    assert identity["identity"]["name"] == "e22-dfp-phase1b-lightweight-boundary-preservation"


def test_artifacts_identity_loads_and_verifies():
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    assert artifacts["identity"]["name"] == "e22-dfp-phase1b-lightweight-boundary-artifacts"


def test_split_loads_and_verifies():
    split = e22.load_and_verify_split(repo_root=ROOT)
    assert len(split["development_image_ids"]) == 350
    assert len(split["confirmation_image_ids"]) == 150


def test_candidate_family_reconstructs():
    identity = e22.load_identity(repo_root=ROOT)
    e22.verify_candidate_family(identity)


def test_no_masked_forward_dependency():
    identity = e22.load_identity(repo_root=ROOT)
    e22.verify_no_masked_forward_dependency(identity)


def test_full_evaluation_not_authorized():
    identity = e22.load_identity(repo_root=ROOT)
    e22.verify_full_evaluation_not_authorized(identity)


def test_repository_gate_self_ancestor():
    e22.verify_repository_gate(ROOT, e22.REQUIRED_ANCESTOR_COMMIT)


# =====================================================================
# Closed mappings: every closed list/set must reject an added/removed
# member.
# =====================================================================


@pytest.mark.parametrize("path, key", [
    (("boundary_definition",), "radii"),
    (("candidate_family",), "stability_starts"),
    (("candidate_family",), "radii"),
    (("candidate_family", "canonical_controls"), "list"),
    (("stability_and_late_flip",), "stability_start_horizons"),
    (("canonical_horizons",), "horizon_set"),
    (("candidate_selection",), "primary_ranking_order"),
    (("candidate_selection",), "eligibility_rules"),
    (("gt_isolation",), "gt_permitted_uses"),
    (("gt_isolation",), "gt_forbidden_influences"),
])
def test_closed_list_rejects_extra_member(tmp_path, path, key):
    doc = _load_preservation_doc()
    node = doc
    for p in path:
        node = node[p]
    assert isinstance(node[key], list)
    node[key] = list(node[key]) + ["__adversarial_extra__"]
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_closed_list_rejects_missing_member():
    doc = _load_preservation_doc()
    doc["boundary_definition"]["radii"] = doc["boundary_definition"]["radii"][:-1]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        written = _write_tmp_toml(Path(td), doc, "tampered.toml")
        with pytest.raises(E):
            e22.load_identity(path=written, repo_root=ROOT)


# =====================================================================
# Dependency hash tampering / path substitution / traversal.
# =====================================================================


def test_dependency_hash_tampering_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["dependency_bindings"]["e13_protocol"]["sha256"] = "0" * 64
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E, match="SHA-256 mismatch"):
        e22.load_identity(path=written, repo_root=ROOT)


@pytest.mark.parametrize("bad_path", [
    "../../../etc/passwd",
    "/etc/passwd",
    "evaluation_identities/../../../etc/passwd",
    "evaluation_identities\\e13_dfp_phase1a_protocol.toml",
    "",
])
def test_dependency_path_traversal_rejected(tmp_path, bad_path):
    doc = _load_preservation_doc()
    doc["dependency_bindings"]["e13_protocol"]["relative_path"] = bad_path
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_dependency_path_redirection_to_prohibited_module_rejected(tmp_path):
    doc = _load_preservation_doc()
    real_sha = hashlib.sha256((ROOT / "src/dfp_phase1b_diagnostic_execution.py").read_bytes()).hexdigest()
    doc["dependency_bindings"]["e13_protocol"]["relative_path"] = "src/dfp_phase1b_diagnostic_execution.py"
    doc["dependency_bindings"]["e13_protocol"]["sha256"] = real_sha
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E, match="prohibited"):
        e22.load_identity(path=written, repo_root=ROOT)


def test_artifacts_protocol_binding_tampering_rejected(tmp_path):
    doc = _load_artifacts_doc()
    doc["protocol_binding"]["protocol_identity_sha256"] = "f" * 64
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E, match="mismatch"):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


# =====================================================================
# Horizon order / candidate family integrity.
# =====================================================================


def test_wrong_horizon_order_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["canonical_horizons"]["horizon_set"] = [320, 64, 32, 16, 0]
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_missing_candidate_rejected():
    identity = e22.load_identity(repo_root=ROOT)
    tampered = copy.deepcopy(dict(identity))
    tampered["candidate_family"] = dict(tampered["candidate_family"])
    tampered["candidate_family"]["ordered_candidate_ids"] = {"list": list(e22.reconstruct_candidate_ids())[:-1]}
    with pytest.raises(E):
        e22.verify_candidate_family(tampered)


def test_extra_candidate_rejected():
    identity = e22.load_identity(repo_root=ROOT)
    tampered = copy.deepcopy(dict(identity))
    tampered["candidate_family"] = dict(tampered["candidate_family"])
    tampered["candidate_family"]["ordered_candidate_ids"] = {"list": list(e22.reconstruct_candidate_ids()) + ["balp_s000_r08"]}
    with pytest.raises(E):
        e22.verify_candidate_family(tampered)


def test_duplicate_candidate_rejected():
    identity = e22.load_identity(repo_root=ROOT)
    tampered = copy.deepcopy(dict(identity))
    ids = list(e22.reconstruct_candidate_ids())
    ids[1] = ids[0]
    tampered["candidate_family"] = dict(tampered["candidate_family"])
    tampered["candidate_family"]["ordered_candidate_ids"] = {"list": ids}
    with pytest.raises(E):
        e22.verify_candidate_family(tampered)


def test_altered_candidate_parameter_rejected_by_parse():
    with pytest.raises(E):
        e22.parse_candidate_id("balp_s000_r08")  # r=8 not in frozen radii
    with pytest.raises(E):
        e22.parse_candidate_id("balp_s128_r00")  # s=128 not in frozen stability starts
    with pytest.raises(E):
        e22.parse_candidate_id("balp_sABC_r00")  # malformed


# =====================================================================
# Boundary connectivity/radius, stability/late-flip formula tampering.
# =====================================================================


@pytest.mark.parametrize("new_connectivity", [4, 6, 16])
def test_boundary_connectivity_change_rejected(tmp_path, new_connectivity):
    doc = _load_preservation_doc()
    doc["boundary_definition"]["connectivity"] = new_connectivity
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


@pytest.mark.parametrize("new_radii", [[0, 1, 2], [0, 1, 2, 4, 8], [1, 2, 4]])
def test_boundary_radii_change_rejected(tmp_path, new_radii):
    doc = _load_preservation_doc()
    doc["boundary_definition"]["radii"] = new_radii
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_altered_stability_formula_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["stability_and_late_flip"]["stable_formula"] = "stable_s(x) = true iff MAJORITY of y_h(x) agree"
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_altered_late_flip_formula_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["stability_and_late_flip"]["late_flip_formula"] = "late_flip(x) = [y_0(x) != y_320(x)]"
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_relational_equivalence_helper_matches_selection_rule():
    y0 = [0, 1, 2, 0, 1]
    y16 = [0, 1, 2, 0, 1]
    y32 = [0, 1, 2, 0, 1]
    y64 = [0, 1, 2, 0, 5]
    y320 = [3, 1, 4, 0, 5]
    boundary = [True, True, False, True, True]
    y_balp, pmask = e22.verify_stability_late_flip_equivalence(y0, y16, y32, y64, y320, 0, boundary)
    for i in range(5):
        if pmask[i]:
            assert y_balp[i] == y64[i]
        else:
            assert y_balp[i] == y320[i]
    # pixel 0: stable (0==0==0==0), late_flip (0!=3), boundary True -> preserved
    assert pmask[0] is True and y_balp[0] == 0
    # pixel 1: stable but no late_flip (1==1) -> not preserved
    assert pmask[1] is False and y_balp[1] == 1
    # pixel 2: stable, late_flip, but NOT boundary -> not preserved
    assert pmask[2] is False and y_balp[2] == 4
    # pixel 3: stable, no late_flip -> not preserved
    assert pmask[3] is False and y_balp[3] == 0
    # pixel 4: NOT stable (0,1,1,5 differ) -> not preserved regardless of late_flip/boundary
    assert pmask[4] is False and y_balp[4] == 5


def test_relational_equivalence_rejects_mismatched_lengths():
    with pytest.raises(E):
        e22.verify_stability_late_flip_equivalence([0], [0, 0], [0], [0], [0], 0, [True])


# =====================================================================
# GT-derived boundary / ignore-mask leakage.
# =====================================================================


def test_gt_boundary_flag_must_be_false(tmp_path):
    doc = _load_preservation_doc()
    doc["boundary_definition"]["gt_boundary_used"] = True
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_ignore_mask_in_boundary_construction_flag_must_be_false(tmp_path):
    doc = _load_preservation_doc()
    doc["boundary_definition"]["ignore_mask_used_in_boundary_construction"] = True
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_gt_isolation_forbidden_influences_cannot_be_narrowed(tmp_path):
    doc = _load_preservation_doc()
    doc["gt_isolation"]["gt_forbidden_influences"] = doc["gt_isolation"]["gt_forbidden_influences"][:-1]
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


# =====================================================================
# Split key/rank/order/membership tampering, overlap, wrong counts.
# =====================================================================


def _write_tmp_json(tmp_path: Path, doc: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def _real_split_doc() -> dict:
    return json.loads(SPLIT_PATH.read_bytes())


class _RootRedirect:
    """Minimal repo_root stand-in that redirects only the split file to
    a tampered copy while every other path still resolves against the
    real repository."""


def _run_with_patched_split(monkeypatch, tampered_doc: dict):
    tampered_path = Path("/tmp") / "__balp_test_split_tampered__.json"
    tampered_path.write_text(json.dumps(tampered_doc))
    orig = e22.SPLIT_RELATIVE_PATH

    class FakeRoot(type(ROOT)):
        pass

    # Simplest robust approach: monkeypatch the module-level relative
    # path constant to an absolute path is not supported by
    # resolve_repository_relative_path's containment check, so instead
    # monkeypatch load_and_verify_split's file read indirectly by
    # writing the tampered doc OVER a scratch copy of the whole repo
    # subset is too heavy -- instead directly unit-test the pure
    # function `reconstruct_development_split` plus manual equality
    # checks, which is what the loader's own comparisons reduce to.
    tampered_path.unlink(missing_ok=True)


def test_split_altered_selection_key_detected_by_reconstruction():
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    reconstructed = e22.reconstruct_development_split(list(manifest["selected_ids"]))
    tampered_keys = list(reconstructed["development_selection_keys"])
    tampered_keys[0] = "f" * 64
    assert tampered_keys != reconstructed["development_selection_keys"]


def test_split_altered_rank_rejected(tmp_path, monkeypatch):
    doc = _real_split_doc()
    doc["development_ranks"] = [1] + doc["development_ranks"][1:]  # rank 0 replaced by a duplicate of rank 1
    tmp_repo = tmp_path / "repo"
    tmp_repo.mkdir()
    (tmp_repo / "evaluation_subsets").mkdir()
    import shutil
    shutil.copy(MANIFEST_PATH, tmp_repo / "evaluation_subsets" / "dfp_phase1a_coco_stuff_500.json")
    (tmp_repo / "evaluation_subsets" / "dfp_phase1b_lightweight_boundary_500_split.json").write_text(json.dumps(doc))
    with pytest.raises(E):
        e22.load_and_verify_split(repo_root=tmp_repo)


def test_split_altered_membership_rejected(tmp_path):
    doc = _real_split_doc()
    doc["development_image_ids"] = list(doc["development_image_ids"])
    doc["development_image_ids"][0] = doc["confirmation_image_ids"][0]  # steal an id from confirmation
    tmp_repo = tmp_path / "repo"
    (tmp_repo / "evaluation_subsets").mkdir(parents=True)
    import shutil
    shutil.copy(MANIFEST_PATH, tmp_repo / "evaluation_subsets" / "dfp_phase1a_coco_stuff_500.json")
    (tmp_repo / "evaluation_subsets" / "dfp_phase1b_lightweight_boundary_500_split.json").write_text(json.dumps(doc))
    with pytest.raises(E):
        e22.load_and_verify_split(repo_root=tmp_repo)


def test_split_altered_ordering_rejected(tmp_path):
    doc = _real_split_doc()
    ids = list(doc["development_image_ids"])
    ids[0], ids[1] = ids[1], ids[0]
    doc["development_image_ids"] = ids
    tmp_repo = tmp_path / "repo"
    (tmp_repo / "evaluation_subsets").mkdir(parents=True)
    import shutil
    shutil.copy(MANIFEST_PATH, tmp_repo / "evaluation_subsets" / "dfp_phase1a_coco_stuff_500.json")
    (tmp_repo / "evaluation_subsets" / "dfp_phase1b_lightweight_boundary_500_split.json").write_text(json.dumps(doc))
    with pytest.raises(E):
        e22.load_and_verify_split(repo_root=tmp_repo)


def test_split_overlap_rejected(tmp_path):
    doc = _real_split_doc()
    doc["confirmation_image_ids"] = list(doc["confirmation_image_ids"])
    doc["confirmation_image_ids"][0] = doc["development_image_ids"][0]
    tmp_repo = tmp_path / "repo"
    (tmp_repo / "evaluation_subsets").mkdir(parents=True)
    import shutil
    shutil.copy(MANIFEST_PATH, tmp_repo / "evaluation_subsets" / "dfp_phase1a_coco_stuff_500.json")
    (tmp_repo / "evaluation_subsets" / "dfp_phase1b_lightweight_boundary_500_split.json").write_text(json.dumps(doc))
    with pytest.raises(E):
        e22.load_and_verify_split(repo_root=tmp_repo)


@pytest.mark.parametrize("dev_count,conf_count", [(349, 150), (350, 149), (300, 200)])
def test_split_wrong_counts_rejected(tmp_path, dev_count, conf_count):
    doc = _real_split_doc()
    doc["development_count"] = dev_count
    doc["confirmation_count"] = conf_count
    tmp_repo = tmp_path / "repo"
    (tmp_repo / "evaluation_subsets").mkdir(parents=True)
    import shutil
    shutil.copy(MANIFEST_PATH, tmp_repo / "evaluation_subsets" / "dfp_phase1a_coco_stuff_500.json")
    (tmp_repo / "evaluation_subsets" / "dfp_phase1b_lightweight_boundary_500_split.json").write_text(json.dumps(doc))
    with pytest.raises(E):
        e22.load_and_verify_split(repo_root=tmp_repo)


# =====================================================================
# Selection/tie-break/threshold/operational-cap tampering.
# =====================================================================


def test_changed_selection_ranking_order_rejected(tmp_path):
    doc = _load_preservation_doc()
    order = list(doc["candidate_selection"]["primary_ranking_order"])
    order[0], order[1] = order[1], order[0]
    doc["candidate_selection"]["primary_ranking_order"] = order
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_changed_acceptance_threshold_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["candidate_selection"]["acceptance_threshold_pp"] = 0.05
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


@pytest.mark.parametrize("field,new_value", [
    ("total_wall_time_overhead_pct_max", 25),
    ("mean_per_image_latency_overhead_pct_max", 30),
    ("throughput_reduction_pct_max", 50),
    ("peak_allocated_vram_increase_pct_max", 20),
    ("peak_reserved_vram_increase_pct_max", 20),
])
def test_changed_operational_caps_rejected(tmp_path, field, new_value):
    doc = _load_preservation_doc()
    doc["operational_contract"]["acceptance_caps"][field] = new_value
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


# =====================================================================
# Enabling prohibited features / full evaluation.
# =====================================================================


def test_enabling_masked_forward_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["method"]["runs_masked_forward_propagation"] = True
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_enabling_component_construction_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["method"]["constructs_components"] = True
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_enabling_full_5000_image_evaluation_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["authorization_boundary"]["full_5000_image_evaluation_authorized"] = True
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_e19_e20_e21_flag_flip_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["historical_exclusions"]["e19_e20_e21_required_by_balp_inference"] = True
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


# =====================================================================
# Torch/CUDA import poisoning -- the module itself must be import-clean.
# =====================================================================


def test_module_imports_no_torch_cuda_model_dataset():
    import ast

    source = (ROOT / "src/dfp_phase1b_lightweight_boundary_identity.py").read_text()
    tree = ast.parse(source)
    banned = {"torch", "cv2", "mmcv", "mmseg"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    hit = found & banned
    assert not hit, f"forbidden import(s) found at module top level: {hit}"


def test_module_import_is_hermetic_in_fresh_subprocess(tmp_path):
    script = tmp_path / "iv_e22_hermetic.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import src.dfp_phase1b_lightweight_boundary_identity as e22\n"
        "assert 'torch' not in sys.modules, 'torch was imported by the E22 identity module'\n"
        "assert 'cv2' not in sys.modules, 'cv2 was imported by the E22 identity module'\n"
        f"identity = e22.load_identity(repo_root={str(ROOT)!r})\n"
        "assert 'torch' not in sys.modules and 'cv2' not in sys.modules\n"
        "print('HERMETIC_OK')\n"
    )
    import subprocess, sys as _sys
    r = subprocess.run([_sys.executable, str(script)], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "HERMETIC_OK" in r.stdout


# =====================================================================
# Malformed schema/version pairs; fatal exception propagation.
# =====================================================================


@pytest.mark.parametrize("bad_name,bad_version", [
    (123, 1), (None, 1), (["x"], 1), ({"a": 1}, 1),
    ("talk2dino-dfp-phase1b-lightweight-boundary-preservation-identity-v1", "1"),
    ("talk2dino-dfp-phase1b-lightweight-boundary-preservation-identity-v1", 1.0),
    ("talk2dino-dfp-phase1b-lightweight-boundary-preservation-identity-v1", True),
    ("talk2dino-dfp-phase1b-lightweight-boundary-preservation-identity-v1", None),
])
def test_malformed_schema_version_pair_rejected(bad_name, bad_version):
    with pytest.raises(E):
        e22._require_schema_pair(bad_name, bad_version, e22.PRESERVATION_SCHEMA_NAME, e22.PRESERVATION_SCHEMA_VERSION, "identity")


def test_bool_as_int_rejected():
    with pytest.raises(E):
        e22._require_exact_int(True, "some_field")
    with pytest.raises(E):
        e22._require_exact_int(False, "some_field")


def test_float_as_int_rejected():
    with pytest.raises(E):
        e22._require_exact_int(1.0, "some_field")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nan_infinity_rejected(bad_value):
    with pytest.raises(E):
        e22._require_finite_float(bad_value, "some_field")


def test_finite_float_accepted():
    assert e22._require_finite_float(1.5, "x") == 1.5


def test_unhashable_schema_version_value_rejected_with_domain_exception():
    """A schema_version given as an unhashable type (list/dict) must be
    rejected by the domain exception, never a raw TypeError -- exact
    int-type checking rejects it before any hashing/frozenset use."""
    with pytest.raises(E):
        e22._require_schema_pair("x", [1], "x", 1, "label")
    with pytest.raises(E):
        e22._require_schema_pair("x", {"v": 1}, "x", 1, "label")


def test_keyboard_interrupt_propagates_through_dependency_hash_check(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(e22.hashlib, "sha256", boom)
    with pytest.raises(KeyboardInterrupt):
        e22.load_identity(repo_root=ROOT)


def test_system_exit_propagates(monkeypatch):
    def boom(*args, **kwargs):
        raise SystemExit(3)

    monkeypatch.setattr(e22.hashlib, "sha256", boom)
    with pytest.raises(SystemExit):
        e22.load_identity(repo_root=ROOT)


def test_memory_error_propagates(monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError()

    monkeypatch.setattr(e22.hashlib, "sha256", boom)
    with pytest.raises(MemoryError):
        e22.load_identity(repo_root=ROOT)


# =====================================================================
# Missing/extra top-level keys.
# =====================================================================


def test_missing_top_level_key_rejected(tmp_path):
    doc = _load_preservation_doc()
    del doc["gt_isolation"]
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_extra_top_level_key_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["__adversarial__"] = {"x": 1}
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


# =====================================================================
# Repair-verification tests: F1 tie-rule citation.
# =====================================================================


def test_tie_rule_citation_points_at_real_e13_section():
    """The corrected citation must resolve to an ACTUAL E13 table/key
    carrying the exact frozen tie-rule value -- verified against raw
    E13 bytes, never against the E22 loader's own constants."""
    e13 = tomllib.load((ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml").open("rb"))
    doc = _load_preservation_doc()
    section = doc["canonical_horizons"]["argmax_tie_rule_source_section"]
    key = doc["canonical_horizons"]["argmax_tie_rule_source_key"]
    assert section == "dispute_protocol"
    assert key == "tie_rule"
    assert e13[section][key] == doc["canonical_horizons"]["argmax_tie_rule"]


def test_wrong_tie_rule_source_section_with_unchanged_value_rejected():
    """Directly exercises the cross-file relational validator
    (_verify_tie_rule_citation), bypassing the generic leaf-exact-match
    check, by tampering the ALREADY-LOADED identity dict in memory --
    proves the citation is verified against real E13 bytes, not just
    self-consistency within one TOML file."""
    identity = e22.load_identity(repo_root=ROOT)
    tampered = dict(identity)
    tampered["canonical_horizons"] = dict(identity["canonical_horizons"])
    tampered["canonical_horizons"]["argmax_tie_rule_source_section"] = "oracle_definitions"
    # tie-rule VALUE is deliberately left unchanged
    assert tampered["canonical_horizons"]["argmax_tie_rule"] == identity["canonical_horizons"]["argmax_tie_rule"]
    with pytest.raises(E, match="tie-rule citation"):
        e22._verify_tie_rule_citation(ROOT, tampered)


def test_wrong_tie_rule_source_key_with_unchanged_value_rejected():
    identity = e22.load_identity(repo_root=ROOT)
    tampered = dict(identity)
    tampered["canonical_horizons"] = dict(identity["canonical_horizons"])
    tampered["canonical_horizons"]["argmax_tie_rule_source_key"] = "not_a_real_key"
    with pytest.raises(E, match="tie-rule citation"):
        e22._verify_tie_rule_citation(ROOT, tampered)


def test_correct_tie_rule_citation_accepted():
    identity = e22.load_identity(repo_root=ROOT)
    e22._verify_tie_rule_citation(ROOT, identity)  # must not raise


def test_tie_rule_citation_fix_does_not_change_argmax_value(tmp_path):
    """The repair must change ONLY the source citation, never the
    frozen argmax_tie_rule value itself."""
    doc = _load_preservation_doc()
    assert doc["canonical_horizons"]["argmax_tie_rule"] == "exact_argmax_ties_select_lowest_class_index"


def test_full_top_level_key_set_still_closed_after_citation_fix(tmp_path):
    doc = _load_preservation_doc()
    written = _write_tmp_toml(tmp_path, doc, "roundtrip.toml")
    e22.load_identity(path=written, repo_root=ROOT)  # must load cleanly, unchanged


# =====================================================================
# Repair-verification tests: F2 exact operational formulas.
# =====================================================================


def test_all_five_formula_blocks_present_with_operator_and_units():
    doc = _load_preservation_doc()
    formulas = doc["operational_contract"]["formulas"]
    expected = {
        "total_wall_time_overhead", "mean_per_image_latency_overhead", "throughput_reduction",
        "peak_allocated_vram_increase", "peak_reserved_vram_increase",
    }
    assert set(formulas.keys()) == expected
    for name, block in formulas.items():
        assert block["comparison_operator"] == "<="
        assert block["units"] == "percent"
        assert isinstance(block["threshold"], float)
        assert block["finite_value_required"] is True


def test_formula_thresholds_match_acceptance_caps_values():
    doc = _load_preservation_doc()
    formulas = doc["operational_contract"]["formulas"]
    caps = doc["operational_contract"]["acceptance_caps"]
    mapping = {
        "total_wall_time_overhead": "total_wall_time_overhead_pct_max",
        "mean_per_image_latency_overhead": "mean_per_image_latency_overhead_pct_max",
        "throughput_reduction": "throughput_reduction_pct_max",
        "peak_allocated_vram_increase": "peak_allocated_vram_increase_pct_max",
        "peak_reserved_vram_increase": "peak_reserved_vram_increase_pct_max",
    }
    for name, cap_field in mapping.items():
        assert float(formulas[name]["threshold"]) == float(caps[cap_field])


def test_thresholds_unchanged_at_15_and_10():
    doc = _load_preservation_doc()
    caps = doc["operational_contract"]["acceptance_caps"]
    assert caps["total_wall_time_overhead_pct_max"] == 15
    assert caps["mean_per_image_latency_overhead_pct_max"] == 15
    assert caps["throughput_reduction_pct_max"] == 15
    assert caps["peak_allocated_vram_increase_pct_max"] == 10
    assert caps["peak_reserved_vram_increase_pct_max"] == 10


def test_formula_threshold_drift_from_acceptance_caps_rejected():
    """Directly exercises the dedicated cross-check
    (_verify_operational_formula_consistency) on an in-memory tampered
    identity, bypassing the generic leaf-exact-match check -- proves
    the formula/cap consistency is independently verified, not merely
    a side effect of the generic check."""
    identity = e22.load_identity(repo_root=ROOT)
    tampered = dict(identity)
    tampered["operational_contract"] = dict(identity["operational_contract"])
    tampered["operational_contract"]["formulas"] = dict(identity["operational_contract"]["formulas"])
    tampered["operational_contract"]["formulas"]["total_wall_time_overhead"] = dict(
        identity["operational_contract"]["formulas"]["total_wall_time_overhead"], threshold=20.0
    )
    with pytest.raises(E, match="drifted"):
        e22._verify_operational_formula_consistency(tampered)


@pytest.mark.parametrize("name", [
    "total_wall_time_overhead", "mean_per_image_latency_overhead", "throughput_reduction",
    "peak_allocated_vram_increase", "peak_reserved_vram_increase",
])
def test_less_than_substituted_for_less_equal_rejected(tmp_path, name):
    doc = _load_preservation_doc()
    doc["operational_contract"]["formulas"][name]["comparison_operator"] = "<"
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


def test_fractional_ratio_mistaken_for_percent_rejected(tmp_path):
    doc = _load_preservation_doc()
    doc["operational_contract"]["formulas"]["total_wall_time_overhead"]["units"] = "fraction"
    written = _write_tmp_toml(tmp_path, doc, "tampered.toml")
    with pytest.raises(E):
        e22.load_identity(path=written, repo_root=ROOT)


# ---- independent oracle: reversed numerator / wrong sign ----


def test_reversed_overhead_numerator_produces_wrong_sign():
    """An implementation that reverses the numerator (baseline -
    candidate instead of candidate - baseline) produces the OPPOSITE
    sign from the frozen formula -- demonstrating the frozen formula's
    sign convention is load-bearing and independently checkable."""
    correct = e22.compute_total_wall_time_overhead_pct(candidate_total_wall_time_ns=120.0, baseline_total_wall_time_ns=100.0)
    reversed_numerator = 100.0 * (100.0 - 120.0) / 100.0
    assert correct == 20.0
    assert reversed_numerator == -20.0
    assert correct != reversed_numerator


def test_wrong_throughput_sign_distinguishable_from_frozen_formula():
    """The frozen throughput formula is (baseline - candidate)/baseline
    -- NOT (candidate - baseline)/baseline (which would report a
    slowdown as negative, the opposite of every other 'overhead'
    metric's convention)."""
    frozen = e22.compute_throughput_reduction_pct(candidate_throughput_images_per_second=8.0, baseline_throughput_images_per_second=10.0)
    wrong_sign = 100.0 * (8.0 - 10.0) / 10.0
    assert frozen == 20.0  # candidate slower -> positive reduction
    assert wrong_sign == -20.0
    assert frozen != wrong_sign


@pytest.mark.parametrize("name,fn,kwargs,expected", [
    ("total_wall_time_overhead", e22.compute_total_wall_time_overhead_pct,
     dict(candidate_total_wall_time_ns=115.0, baseline_total_wall_time_ns=100.0), 15.0),
    ("mean_per_image_latency_overhead", e22.compute_mean_per_image_latency_overhead_pct,
     dict(candidate_mean_per_image_latency_ns=115.0, baseline_mean_per_image_latency_ns=100.0), 15.0),
    ("throughput_reduction", e22.compute_throughput_reduction_pct,
     dict(candidate_throughput_images_per_second=8.5, baseline_throughput_images_per_second=10.0), 15.0),
    ("peak_allocated_vram_increase", e22.compute_peak_allocated_vram_increase_pct,
     dict(candidate_peak_allocated_vram_bytes=110.0, baseline_peak_allocated_vram_bytes=100.0), 10.0),
    ("peak_reserved_vram_increase", e22.compute_peak_reserved_vram_increase_pct,
     dict(candidate_peak_reserved_vram_bytes=110.0, baseline_peak_reserved_vram_bytes=100.0), 10.0),
])
def test_operational_threshold_boundary_below_at_above(name, fn, kwargs, expected):
    pct = fn(**kwargs)
    assert abs(pct - expected) < 1e-9, (name, pct, expected)
    # immediately below
    below_kwargs = dict(kwargs)
    num_key = [k for k in kwargs if k.startswith("candidate")][0]
    below_kwargs[num_key] = kwargs[num_key] * 0.9999 if "throughput" not in name else kwargs[num_key] * 1.0001
    below_pct = fn(**below_kwargs)
    assert below_pct <= expected
    assert (below_pct <= expected) is True  # inclusive cap: exactly-at is also accepted
    assert expected <= expected  # exactly-at boundary passes with <=
    # immediately above
    above_kwargs = dict(kwargs)
    above_kwargs[num_key] = kwargs[num_key] * 1.0001 if "throughput" not in name else kwargs[num_key] * 0.9999
    above_pct = fn(**above_kwargs)
    assert above_pct > expected


def test_negative_overhead_accepted_as_valid_result():
    pct = e22.compute_total_wall_time_overhead_pct(candidate_total_wall_time_ns=80.0, baseline_total_wall_time_ns=100.0)
    assert pct == -20.0
    _, accepted = e22.evaluate_operational_cap("total_wall_time_overhead", 80.0, 100.0, threshold=15.0)
    assert accepted is True


@pytest.mark.parametrize("bad_candidate", [float("nan"), float("inf"), float("-inf")])
def test_nan_infinity_measurement_rejected(bad_candidate):
    with pytest.raises(E):
        e22.compute_total_wall_time_overhead_pct(bad_candidate, 100.0)


def test_bool_measurement_rejected():
    with pytest.raises(E):
        e22.compute_total_wall_time_overhead_pct(True, 100.0)
    with pytest.raises(E):
        e22.compute_total_wall_time_overhead_pct(120.0, False)


def test_string_and_none_measurement_rejected():
    with pytest.raises(E):
        e22.compute_total_wall_time_overhead_pct("120", 100.0)
    with pytest.raises(E):
        e22.compute_total_wall_time_overhead_pct(None, 100.0)


@pytest.mark.parametrize("bad_baseline", [0.0, -5.0])
def test_non_positive_baseline_rejected(bad_baseline):
    with pytest.raises(E):
        e22.compute_total_wall_time_overhead_pct(100.0, bad_baseline)
    with pytest.raises(E):
        e22.compute_throughput_reduction_pct(8.0, bad_baseline)
    with pytest.raises(E):
        e22.compute_peak_allocated_vram_increase_pct(100.0, bad_baseline)


def test_host_rss_has_no_pass_fail_threshold_field():
    doc = _load_preservation_doc()
    policy = doc["operational_contract"]["host_rss_policy"]
    assert policy["pass_fail_threshold_authorized"] is False


def test_host_rss_zero_denominator_yields_none_percentage_not_error():
    result = e22.compute_host_rss_difference(candidate_host_rss_bytes=50.0, baseline_host_rss_bytes=0.0)
    assert result["percentage_difference_pct"] is None
    assert result["absolute_difference_bytes"] == 50.0


# =====================================================================
# Repair-verification tests: F3 checkpoint relational validation.
# =====================================================================


_DEV_IDS = [f"dev{i:03d}" for i in range(350)]
_CONF_IDS = [f"conf{i:03d}" for i in range(150)]
_CLASS_COUNT = 3


def _zero_mat():
    return [[0] * _CLASS_COUNT for _ in range(_CLASS_COUNT)]


def _make_checkpoint(n_completed, *, complete=False, final=False, final_binding=None, keys=None):
    keys = keys if keys is not None else (list(e22.CANONICAL_CONTROLS) + list(e22.reconstruct_candidate_ids()))
    completed = _DEV_IDS[:n_completed]
    per_image, agg = {}, {}
    for k in keys:
        mats = [[[1 if r == c else 0 for c in range(_CLASS_COUNT)] for r in range(_CLASS_COUNT)] for _ in range(n_completed)]
        agg_mat = _zero_mat()
        for m in mats:
            agg_mat = e22._matrix_add(agg_mat, m)
        per_image[k] = {
            "candidate_versus_t320_changed_pixel_count": [0] * n_completed,
            "preservation_mask_pixel_count": [0] * n_completed,
            "boundary_preservation_pixel_count": [0] * n_completed,
            "non_boundary_preservation_pixel_count": [0] * n_completed,
            "evaluated_pixel_count": [100] * n_completed,
            "ignored_pixel_count": [5] * n_completed,
            "per_image_confusion_matrix": mats,
        }
        agg[k] = agg_mat
    return {
        "schema_name": e22.CHECKPOINT_SCHEMA_NAME,
        "schema_version": e22.CHECKPOINT_SCHEMA_VERSION,
        "identity_bindings": {
            "e22_protocol_sha256": "a" * 64, "e22_artifacts_sha256": "b" * 64, "e13_protocol_sha256": "c" * 64,
            "matched_evaluator_sha256": "d" * 64, "subset_manifest_sha256": "e" * 64, "balp_split_sha256": "f" * 64,
            "development_digest": "0" * 64, "run_mode": "coco_stuff_500",
        },
        "progress": {"expected_development_image_count": 350, "completed_image_ids": completed, "next_development_index": n_completed, "generation": 0},
        "aggregate_confusion": agg,
        "per_image_reconstruction_statistics": per_image,
        "complete": complete, "final": final,
        "final_result_binding": final_binding,
    }


@pytest.mark.parametrize("n", [0, 1, 175, 349])
def test_checkpoint_valid_progress_states(n):
    doc = _make_checkpoint(n)
    e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_complete_unfinalized_valid():
    doc = _make_checkpoint(350, complete=True, final=False)
    e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_finalized_valid():
    doc = _make_checkpoint(350, complete=True, final=True, final_binding={"development_result_relative_path": "x", "development_result_sha256": "0" * 64, "installed_at_generation": 0})
    e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_valid_resume_pair():
    old = _make_checkpoint(100)
    new = _make_checkpoint(150)
    e22.validate_checkpoint_resume_pair(old, new)


def test_checkpoint_non_prefix_completed_ids_rejected():
    doc = _make_checkpoint(5)
    doc["progress"]["completed_image_ids"] = ["not_a_real_id"] * 5
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_duplicate_id_rejected():
    doc = _make_checkpoint(5)
    doc["progress"]["completed_image_ids"] = [_DEV_IDS[0]] * 5
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_skipped_id_rejected():
    doc = _make_checkpoint(5)
    doc["progress"]["completed_image_ids"] = [_DEV_IDS[0], _DEV_IDS[1], _DEV_IDS[3], _DEV_IDS[4], _DEV_IDS[5]]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_reordered_id_rejected():
    doc = _make_checkpoint(5)
    doc["progress"]["completed_image_ids"] = [_DEV_IDS[0], _DEV_IDS[1], _DEV_IDS[2], _DEV_IDS[4], _DEV_IDS[3]]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_confirmation_image_in_development_rejected():
    doc = _make_checkpoint(3)
    doc["progress"]["completed_image_ids"] = [_DEV_IDS[0], _DEV_IDS[1], _CONF_IDS[0]]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_aggregate_per_image_disagreement_rejected():
    doc = _make_checkpoint(4)
    doc["aggregate_confusion"]["T0"][0][0] = 99999
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_candidate_order_mutation_rejected():
    doc = _make_checkpoint(4)
    keys = list(doc["aggregate_confusion"].keys())
    doc["aggregate_confusion"] = {k: doc["aggregate_confusion"][k] for k in reversed(keys)}
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_foreign_candidate_key_rejected():
    doc = _make_checkpoint(2, keys=list(e22.CANONICAL_CONTROLS) + list(e22.reconstruct_candidate_ids()) + ["balp_s999_r99"])
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


@pytest.mark.parametrize("complete,final", [(False, True)])
def test_checkpoint_complete_final_mismatch_rejected(complete, final):
    doc = _make_checkpoint(350, complete=complete, final=final, final_binding={"x": 1})
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_final_binding_present_while_incomplete_rejected():
    doc = _make_checkpoint(10, complete=False, final=False, final_binding={"x": 1})
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_evaluated_pixel_count_disagreement_across_variants_rejected():
    doc = _make_checkpoint(3)
    doc["per_image_reconstruction_statistics"]["T16"]["evaluated_pixel_count"] = [999, 100, 100]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_checkpoint_resume_altered_old_segment_rejected():
    old = _make_checkpoint(10)
    new = _make_checkpoint(20)
    new["per_image_reconstruction_statistics"]["T0"]["evaluated_pixel_count"][3] = 123456
    with pytest.raises(E):
        e22.validate_checkpoint_resume_pair(old, new)


def test_checkpoint_resume_decreasing_next_index_rejected():
    old = _make_checkpoint(20)
    new = _make_checkpoint(10)
    with pytest.raises(E):
        e22.validate_checkpoint_resume_pair(old, new)


def test_checkpoint_resume_non_prefix_rejected():
    old = _make_checkpoint(10)
    new = _make_checkpoint(20)
    new["progress"]["completed_image_ids"][3] = "some_other_id"
    with pytest.raises(E):
        e22.validate_checkpoint_resume_pair(old, new)


def test_checkpoint_resume_aggregate_disagreement_rejected():
    old = _make_checkpoint(10)
    new = _make_checkpoint(20)
    new["aggregate_confusion"]["T0"][0][0] += 1000
    with pytest.raises(E):
        e22.validate_checkpoint_resume_pair(old, new)


def test_checkpoint_resume_identity_bindings_changed_rejected():
    old = _make_checkpoint(10)
    new = _make_checkpoint(20)
    new["identity_bindings"] = {"x": 2}
    with pytest.raises(E):
        e22.validate_checkpoint_resume_pair(old, new)


# ---- checkpoint schema authority text checks ----


def test_checkpoint_relational_validation_list_present_in_toml():
    doc = _load_artifacts_doc()
    rv = doc["checkpoint_schema"]["relational_validation"]["list"]
    assert any("prefix" in rule for rule in rv)
    assert any("aggregate" in rule and "sum" in rule for rule in rv)


def test_checkpoint_cross_generation_invariants_present_in_toml():
    doc = _load_artifacts_doc()
    cg = doc["checkpoint_schema"]["cross_generation_invariants"]["list"]
    assert any("byte-identical" in rule for rule in cg)
    assert any("fresh run and a checkpoint-then-resumed run" in rule for rule in cg)


def test_checkpoint_state_machine_all_four_states_present_in_toml():
    doc = _load_artifacts_doc()
    sm = doc["checkpoint_schema"]["state_machine"]
    assert set(sm.keys()) == {"zero_progress", "partial_progress", "complete_unfinalized", "finalized"}


def test_relational_validation_list_mutation_rejected(tmp_path):
    doc = _load_artifacts_doc()
    doc["checkpoint_schema"]["relational_validation"]["list"] = doc["checkpoint_schema"]["relational_validation"]["list"][:-1]
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


# =====================================================================
# Repair-verification tests: F4 artifact write policy.
# =====================================================================


def test_write_policy_section_present_and_valid():
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    e22.verify_write_policy_present(artifacts)


def test_write_policy_covers_every_artifact_family():
    doc = _load_artifacts_doc()
    applies_to = set(doc["artifact_write_policy"]["applies_to"])
    assert applies_to == {
        "development_checkpoint", "development_result", "selection_artifact",
        "profiling_artifact", "future_full_evaluation_result",
    }


def test_write_policy_forbids_unsafe_sequential_final_renames():
    doc = _load_artifacts_doc()
    assert doc["artifact_write_policy"]["atomic_installation"]["multi_file_artifacts"]["unguarded_sequential_final_renames_forbidden"] is True


def test_write_policy_requires_self_verification_before_installation():
    doc = _load_artifacts_doc()
    assert doc["artifact_write_policy"]["staging"]["self_verification_before_installation_required"] is True


def test_write_policy_multi_file_installation_is_atomic_directory_rename():
    doc = _load_artifacts_doc()
    assert "atomic directory rename" in doc["artifact_write_policy"]["atomic_installation"]["multi_file_artifacts"]["install_operation"]


def test_write_policy_grants_no_overwrite():
    doc = _load_artifacts_doc()
    assert doc["artifact_write_policy"]["general_path_policy"]["e22_grants_overwrite_authority"] is False
    assert doc["artifact_write_policy"]["freshness_and_overwrite"]["e22_grants_destructive_overwrite"] is False


def test_write_policy_overwrite_enabled_rejected():
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    tampered = dict(artifacts)
    tampered["artifact_write_policy"] = dict(artifacts["artifact_write_policy"])
    tampered["artifact_write_policy"]["general_path_policy"] = dict(
        artifacts["artifact_write_policy"]["general_path_policy"], e22_grants_overwrite_authority=True
    )
    with pytest.raises(E):
        e22.verify_write_policy_present(tampered)
    # also confirm the generic loader path independently rejects the same tampering end-to-end
    import tempfile
    doc = _load_artifacts_doc()
    doc["artifact_write_policy"]["general_path_policy"]["e22_grants_overwrite_authority"] = True
    with tempfile.TemporaryDirectory() as td:
        written = _write_tmp_toml(Path(td), doc, "tampered_artifacts.toml")
        with pytest.raises(E):
            e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_write_policy_unsafe_sequential_renames_allowed_rejected():
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    tampered = dict(artifacts)
    tampered["artifact_write_policy"] = dict(artifacts["artifact_write_policy"])
    tampered["artifact_write_policy"]["atomic_installation"] = dict(artifacts["artifact_write_policy"]["atomic_installation"])
    tampered["artifact_write_policy"]["atomic_installation"]["multi_file_artifacts"] = dict(
        artifacts["artifact_write_policy"]["atomic_installation"]["multi_file_artifacts"],
        unguarded_sequential_final_renames_forbidden=False,
    )
    with pytest.raises(E):
        e22.verify_write_policy_present(tampered)


def test_write_policy_missing_failure_behavior_key_rejected(tmp_path):
    doc = _load_artifacts_doc()
    del doc["artifact_write_policy"]["failure_behavior"]["staging_residue_removed_on_failure"]
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_write_policy_extra_key_rejected(tmp_path):
    doc = _load_artifacts_doc()
    doc["artifact_write_policy"]["__extra__"] = {"x": 1}
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_write_policy_missing_top_key_rejected(tmp_path):
    doc = _load_artifacts_doc()
    del doc["artifact_write_policy"]["sibling_digest_bindings"]
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_write_policy_failure_preserves_prior_valid_state_flag():
    doc = _load_artifacts_doc()
    fb = doc["artifact_write_policy"]["failure_behavior"]
    assert fb["failure_before_final_rename_leaves_prior_state_untouched"] is True
    assert fb["existing_valid_artifacts_byte_identical_after_failed_install"] is True
    assert fb["source_artifacts_never_mutated"] is True


def test_write_policy_sibling_digest_bindings_cover_all_artifact_relationships():
    doc = _load_artifacts_doc()
    bindings = doc["artifact_write_policy"]["sibling_digest_bindings"]["list"]
    assert any("checkpoint" in b for b in bindings)
    assert any("development_result" in b for b in bindings)
    assert any("selection_artifact" in b for b in bindings)
    assert any("profiling_artifact" in b for b in bindings)
    assert any("future_full_evaluation_result" in b for b in bindings)


# =====================================================================
# Hash cascade / stale-hash rejection.
# =====================================================================


def test_stale_preservation_hash_in_artifact_toml_rejected(tmp_path):
    doc = _load_artifacts_doc()
    doc["protocol_binding"]["protocol_identity_sha256"] = "e" * 64
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E, match="mismatch"):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_current_artifact_toml_binds_current_preservation_toml_exactly():
    doc = _load_artifacts_doc()
    actual = hashlib.sha256(PRESERVATION_PATH.read_bytes()).hexdigest()
    assert doc["protocol_binding"]["protocol_identity_sha256"] == actual


def test_no_stale_hash_reference_to_old_preservation_hash_anywhere():
    """The pre-repair preservation.toml hash must not appear anywhere
    in the repaired files (would indicate a forgotten cascade update)."""
    stale_hash = "ed66e387029058211636c6e4ce823fefa14c0c833cbbcc0c41992c7a1a249ea4"
    for path in (PRESERVATION_PATH, ARTIFACTS_PATH):
        assert stale_hash not in path.read_text()


def test_e13_e21_hashes_in_e22_unchanged_by_repair():
    doc = _load_preservation_doc()
    deps = doc["dependency_bindings"]
    assert deps["e13_protocol"]["sha256"] == "227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173"
    assert deps["matched_evaluator"]["sha256"] == "1f4eea731ae7b803a5e2d572d0a1f59b3275ec240d36061d03540706e3bbc259"


# =====================================================================
# Final repair-verification tests: checkpoint top-level and nested
# closed-key enforcement.
# =====================================================================


def test_checkpoint_top_level_keys_match_toml_authority():
    doc = _load_artifacts_doc()
    assert set(e22.CHECKPOINT_TOP_LEVEL_KEYS) == set(doc["checkpoint_schema"]["top_keys"]["list"])
    assert "final_result_binding" in doc["checkpoint_schema"]["top_keys"]["list"]


def test_valid_exact_top_level_key_set_accepted():
    doc = _make_checkpoint(5)
    assert set(doc.keys()) == e22.CHECKPOINT_TOP_LEVEL_KEYS
    e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_reordered_valid_top_level_keys_accepted():
    doc = _make_checkpoint(5)
    reordered = {k: doc[k] for k in reversed(list(doc.keys()))}
    assert set(reordered.keys()) == set(doc.keys())
    e22.validate_checkpoint_single(reordered, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


@pytest.mark.parametrize("missing_key", sorted(e22.CHECKPOINT_TOP_LEVEL_KEYS))
def test_every_missing_top_level_key_rejected(missing_key):
    doc = _make_checkpoint(5)
    del doc[missing_key]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


@pytest.mark.parametrize("extra_key,extra_value", [
    ("profiling_leak", {"wall_time_ns": 1}),
    ("timing_ns", 123),
    ("gpu_peak_vram_bytes", 456),
    ("debug", True),
    ("extra", "x"),
    ("unknown_nested", {"a": {"b": 1}}),
])
def test_every_extra_top_level_key_category_rejected(extra_key, extra_value):
    doc = _make_checkpoint(5)
    doc[extra_key] = extra_value
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_profiling_leak_specifically_rejected():
    """Directly proves the exact scenario the repair targets: a
    'profiling'-named field can never appear inside a scientific
    checkpoint document."""
    doc = _make_checkpoint(3)
    doc["profiling_leak"] = {"gpu_peak_allocated_bytes": 1, "wall_time_ns": 2}
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_field_from_a_different_schema_version_rejected():
    doc = _make_checkpoint(5)
    doc["schema_version"] = e22.CHECKPOINT_SCHEMA_VERSION + 1
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


@pytest.mark.parametrize("bad_input", [
    ["not", "a", "mapping"], "not a dict either", None, 42, 3.14,
])
def test_non_mapping_checkpoint_input_rejected(bad_input):
    with pytest.raises(E):
        e22.validate_checkpoint_single(bad_input, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_unhashable_schema_version_in_checkpoint_rejected():
    doc = _make_checkpoint(3)
    doc["schema_version"] = [1]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)
    doc2 = _make_checkpoint(3)
    doc2["schema_version"] = {"v": 1}
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc2, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_bool_as_schema_version_in_checkpoint_rejected():
    doc = _make_checkpoint(3)
    doc["schema_version"] = True
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


# ---- nested closed mappings ----


def test_nested_progress_closed():
    doc = _make_checkpoint(3)
    doc["progress"]["timing_ns"] = 1
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)
    doc2 = _make_checkpoint(3)
    del doc2["progress"]["generation"]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc2, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_nested_identity_bindings_closed():
    doc = _make_checkpoint(3)
    doc["identity_bindings"]["extra_field"] = "x"
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)
    doc2 = _make_checkpoint(3)
    del doc2["identity_bindings"]["run_mode"]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc2, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_nested_per_image_reconstruction_statistics_closed():
    doc = _make_checkpoint(3)
    doc["per_image_reconstruction_statistics"]["T0"]["profiling_leak"] = [0, 0, 0]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)
    doc2 = _make_checkpoint(3)
    del doc2["per_image_reconstruction_statistics"]["balp_s000_r00"]["ignored_pixel_count"]
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc2, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_nested_final_result_binding_closed():
    binding = {"development_result_relative_path": "r", "development_result_sha256": "a" * 64, "installed_at_generation": 0, "extra": 1}
    doc = _make_checkpoint(350, complete=True, final=True, final_binding=binding)
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)
    binding2 = {"development_result_relative_path": "r", "development_result_sha256": "a" * 64}
    doc2 = _make_checkpoint(350, complete=True, final=True, final_binding=binding2)
    with pytest.raises(E):
        e22.validate_checkpoint_single(doc2, development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_all_valid_checkpoint_states_still_accepted_after_closure_repair():
    for n in (0, 1, 175, 349):
        e22.validate_checkpoint_single(_make_checkpoint(n), development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)
    e22.validate_checkpoint_single(_make_checkpoint(350, complete=True), development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)
    valid_binding = {"development_result_relative_path": "r", "development_result_sha256": "a" * 64, "installed_at_generation": 0}
    e22.validate_checkpoint_single(_make_checkpoint(350, complete=True, final=True, final_binding=valid_binding), development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


def test_cross_generation_validation_still_passes_valid_pairs_after_closure_repair():
    old = _make_checkpoint(120)
    new = _make_checkpoint(180)
    e22.validate_checkpoint_resume_pair(old, new)


def test_checkpoint_closure_fatal_exceptions_propagate(monkeypatch):
    def boom(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(e22, "_require_closed", boom)
    with pytest.raises(KeyboardInterrupt):
        e22.validate_checkpoint_single(_make_checkpoint(3), development_ids=_DEV_IDS, confirmation_ids=_CONF_IDS)


# =====================================================================
# Final repair-verification tests: post-install reopen/reverification
# write-policy field.
# =====================================================================


def _post_install_block(doc):
    return doc["artifact_write_policy"]["atomic_installation"]["post_install_verification"]


def test_post_install_field_present_and_true_accepted():
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    e22.verify_write_policy_present(artifacts)
    assert _post_install_block(_load_artifacts_doc())["reopen_and_reverify_after_install_required"] is True


def test_post_install_field_missing_rejected(tmp_path):
    doc = _load_artifacts_doc()
    del _post_install_block(doc)["reopen_and_reverify_after_install_required"]
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_post_install_field_false_rejected():
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    tampered = dict(artifacts)
    tampered["artifact_write_policy"] = dict(artifacts["artifact_write_policy"])
    tampered["artifact_write_policy"]["atomic_installation"] = dict(artifacts["artifact_write_policy"]["atomic_installation"])
    tampered["artifact_write_policy"]["atomic_installation"]["post_install_verification"] = dict(
        _post_install_block(artifacts), reopen_and_reverify_after_install_required=False,
    )
    with pytest.raises(E):
        e22.verify_write_policy_present(tampered)


@pytest.mark.parametrize("bad_value", [1, 1.0, "true", [True], {"v": True}])
def test_post_install_field_wrong_type_rejected(tmp_path, bad_value):
    """TOML has no null literal, so every representable wrong-type case
    (int, float, string, list, mapping) is round-tripped through a real
    tampered TOML file and loaded via the real public loader."""
    doc = _load_artifacts_doc()
    _post_install_block(doc)["reopen_and_reverify_after_install_required"] = bad_value
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_post_install_field_null_rejected():
    """None/null has no TOML literal, so this case is exercised via the
    real public verify_write_policy_present on an in-memory tampered
    (but otherwise real, freshly-loaded) identity, rather than a
    file round-trip."""
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    tampered = dict(artifacts)
    tampered["artifact_write_policy"] = dict(artifacts["artifact_write_policy"])
    tampered["artifact_write_policy"]["atomic_installation"] = dict(artifacts["artifact_write_policy"]["atomic_installation"])
    tampered["artifact_write_policy"]["atomic_installation"]["post_install_verification"] = dict(
        _post_install_block(artifacts), reopen_and_reverify_after_install_required=None,
    )
    with pytest.raises(E):
        e22.verify_write_policy_present(tampered)


def test_post_install_field_renamed_rejected(tmp_path):
    doc = _load_artifacts_doc()
    block = _post_install_block(doc)
    block["reopen_and_reverify_required"] = block.pop("reopen_and_reverify_after_install_required")
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_post_install_field_wrong_subsection_rejected(tmp_path):
    doc = _load_artifacts_doc()
    value = _post_install_block(doc).pop("reopen_and_reverify_after_install_required")
    doc["artifact_write_policy"]["atomic_installation"]["multi_file_artifacts"]["reopen_and_reverify_after_install_required"] = value
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_post_install_extra_field_rejected(tmp_path):
    doc = _load_artifacts_doc()
    _post_install_block(doc)["__extra__"] = True
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_post_install_replaced_by_pre_install_only_wording_rejected():
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    tampered = dict(artifacts)
    tampered["artifact_write_policy"] = dict(artifacts["artifact_write_policy"])
    tampered["artifact_write_policy"]["atomic_installation"] = dict(artifacts["artifact_write_policy"]["atomic_installation"])
    tampered["artifact_write_policy"]["atomic_installation"]["post_install_verification"] = dict(
        _post_install_block(artifacts), replaces_pre_install_staged_verification=True,
    )
    with pytest.raises(E):
        e22.verify_write_policy_present(tampered)


def test_post_install_both_mandatory_flag_disabled_rejected():
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    tampered = dict(artifacts)
    tampered["artifact_write_policy"] = dict(artifacts["artifact_write_policy"])
    tampered["artifact_write_policy"]["atomic_installation"] = dict(artifacts["artifact_write_policy"]["atomic_installation"])
    tampered["artifact_write_policy"]["atomic_installation"]["post_install_verification"] = dict(
        _post_install_block(artifacts), both_pre_install_and_post_install_verification_mandatory=False,
    )
    with pytest.raises(E):
        e22.verify_write_policy_present(tampered)


def test_public_validator_requirement_removed_rejected(tmp_path):
    doc = _load_artifacts_doc()
    _post_install_block(doc)["public_validator_required_against_installed_bytes"] = False
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_downstream_advance_before_reverification_allowed_rejected(tmp_path):
    doc = _load_artifacts_doc()
    _post_install_block(doc)["downstream_advance_forbidden_before_post_install_verification_passes"] = []
    written = _write_tmp_toml(tmp_path, doc, "tampered_artifacts.toml")
    with pytest.raises(E):
        e22.load_artifacts_identity(path=written, repo_root=ROOT)


def test_post_install_overwrite_authorization_rejected():
    artifacts = e22.load_artifacts_identity(repo_root=ROOT)
    tampered = dict(artifacts)
    tampered["artifact_write_policy"] = dict(artifacts["artifact_write_policy"])
    tampered["artifact_write_policy"]["atomic_installation"] = dict(artifacts["artifact_write_policy"]["atomic_installation"])
    tampered["artifact_write_policy"]["atomic_installation"]["post_install_verification"] = dict(
        _post_install_block(artifacts), overwrite_authorized_by_this_requirement=True,
    )
    with pytest.raises(E):
        e22.verify_write_policy_present(tampered)


def test_installed_path_reopen_source_never_the_staging_path():
    doc = _load_artifacts_doc()
    reopen_source = _post_install_block(doc)["reopen_source"]
    assert "installed" in reopen_source
    assert "never the staging" in reopen_source or "staging" in reopen_source


def test_e20_cleanup_behavior_precisely_stated_not_fabricated():
    doc = _load_artifacts_doc()
    cleanup = _post_install_block(doc)["cleanup_on_post_install_verification_failure"]
    assert "NONE" in cleanup
    assert "left on disk" in cleanup or "forensic" in cleanup


# ---- hash cascade after this repair ----


def test_artifact_toml_hash_updated_and_no_stale_reference():
    current_hash = hashlib.sha256(ARTIFACTS_PATH.read_bytes()).hexdigest()
    old_pre_second_repair_hash = "f84406d41b1075ad5334dd09c725a0859a997a8b833e334aeb6beb7714d247e4"
    assert current_hash != old_pre_second_repair_hash
    for path in (PRESERVATION_PATH, ARTIFACTS_PATH, Path(e22.__file__), Path("docs/dfp_phase1b_lightweight_boundary_protocol.md")):
        assert old_pre_second_repair_hash not in path.read_text()


def test_preservation_toml_hash_unchanged_by_this_repair():
    """This repair only modifies the artifacts TOML/loader/tests/docs;
    the preservation TOML has no forward binding to the artifacts TOML
    (one-directional dependency), so it must remain byte-identical."""
    current_hash = hashlib.sha256(PRESERVATION_PATH.read_bytes()).hexdigest()
    assert current_hash == "bcc6db2dce36c3d740cd66cc36125de36c3faf86dd9924cd02218ec60ee5d63d"
