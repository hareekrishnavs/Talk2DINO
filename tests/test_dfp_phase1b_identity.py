"""Strict adversarial tests for the DFP Phase 1B scientific protocol
successor identity (evaluation_identities/e15_dfp_phase1b_protocol.toml,
src/dfp_phase1b_identity.py). CPU-only, no CUDA, no network, no dataset
access. Loads the committed identity and mutates only the field under
test -- never maintains a duplicated second authority. Never modifies
the completed Phase 1A protocol/artifacts, e14 Boundary IoU identity,
or the frozen subset manifest.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.dfp_phase1b_identity import (  # noqa: E402
    DfpPhase1bIdentityError,
    load_identity,
    verify_authority_bindings,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e15_dfp_phase1b_protocol.toml"
PHASE1A_PROTOCOL_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml"
PHASE1A_ARTIFACTS_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_artifacts.toml"
BOUNDARY_IOU_PATH = ROOT / "evaluation_identities/e14_dfp_phase1b_boundary_iou.toml"
SUBSET_MANIFEST_PATH = ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json"


def _raw_text() -> str:
    return IDENTITY_PATH.read_text(encoding="utf-8")


def _write_mutated(tmp_path: Path, old: str, new: str, *, name: str = "mutated.toml") -> Path:
    text = _raw_text()
    assert old in text, f"fixture assumption stale: {old!r} not found in the committed identity"
    text = text.replace(old, new, 1)
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Positive loading / protected-file invariance.
# ---------------------------------------------------------------------------


def test_real_identity_loads():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    assert identity["identity"]["name"] == "e15-dfp-phase1b-protocol"


def test_real_authority_bindings_verify():
    verified = verify_authority_bindings(repo_root=ROOT)
    assert verified["phase1a_protocol"] == "227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173"
    assert verified["boundary_iou_successor_identity"]


def test_load_identity_does_not_mutate_source_file():
    before = IDENTITY_PATH.read_bytes()
    load_identity(IDENTITY_PATH, repo_root=ROOT)
    after = IDENTITY_PATH.read_bytes()
    assert before == after


def test_verify_bindings_does_not_mutate_phase1a_protected_files():
    before = {
        p: p.read_bytes()
        for p in (PHASE1A_PROTOCOL_PATH, PHASE1A_ARTIFACTS_PATH, BOUNDARY_IOU_PATH, SUBSET_MANIFEST_PATH)
    }
    verify_authority_bindings(repo_root=ROOT)
    after = {p: p.read_bytes() for p in before}
    assert before == after


def test_no_private_path_in_identity_file():
    text = _raw_text()
    for fragment in ("/scratch/", "/project/", "/home/"):
        assert fragment not in text


def test_dependency_direction_acyclic_documented():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    graph = identity["dependency_graph"]
    assert graph["this_identity_binds_artifact_identity_hash"] is False
    assert graph["artifact_identity_binds_this_identity_hash"] is True


def test_boundary_iou_authority_status_reported_as_unbound_in_phase1a():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    facts = identity["gate1_verified_facts"]
    assert facts["boundary_iou_authority_status_in_phase1a"] == "not_yet_bound"
    assert facts["gate1_verdict"] == "PROCEED_TO_PHASE_1B"
    assert facts["boundary_trigger_status"] == "FIRED"
    assert facts["next_prerequisite_from_gate1"] == "BIND_BOUNDARY_IOU"


def test_rgb_baseline_bound_to_historical_pamr():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    baseline = identity["rgb_local_boundary_baseline"]
    assert baseline["authority_status"] == "bound"
    assert baseline["baseline_family"] == "pamr"
    assert baseline["iterations"] == 10
    assert baseline["dilations"] == [1, 2, 4, 8, 12, 24]
    assert baseline["argmax_order"] == "pamr_before_argmax"
    assert baseline["same_pamr_parameters_for_both"] is True
    assert baseline["pamr_does_not_feed_provenance_features"] is True
    assert baseline["pamr_is_not_used_to_construct_components"] is True
    assert baseline["pamr_is_not_used_to_select_disputed_pixels"] is True
    assert baseline["selection_did_not_use_phase1b_results"] is True


def test_real_pamr_authority_bindings_verify():
    from src.dfp_phase1b_identity import verify_pamr_authority_bindings

    verified = verify_pamr_authority_bindings(repo_root=ROOT)
    assert len(verified["implementation_sha256"]) == 64
    assert len(verified["configuration_sha256"]) == 64
    assert len(verified["inherited"]) == 1
    assert len(verified["resolved_parameter_digest"]) == 64


def test_pamr_resolved_parameter_digest_self_consistent():
    from src.dfp_phase1b_identity import compute_resolved_pamr_parameter_digest

    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    binding = identity["rgb_local_boundary_baseline"]
    assert compute_resolved_pamr_parameter_digest(binding) == binding["resolved_parameter_digest"]


def test_twelve_baseline_variants_in_frozen_order():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    required = identity["baseline_variants"]["required"]
    assert len(required) == 12
    assert required[:10] == [
        "t0", "t16", "t32", "t64", "t320", "oracle_o_p",
        "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",
    ]
    assert required[10:] == ["t0_pamr", "t320_pamr"]


def test_auroc_auprc_units_unambiguous():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    units = identity["gate2_metrics"]["units"]
    assert units["auroc_unit"] == "fraction_0_1"
    assert units["auprc_unit"] == "fraction_0_1"
    assert units["delta_auroc_unit"] == "absolute_fraction_0_1"
    assert units["delta_auprc_unit"] == "absolute_fraction_0_1"
    mandatory = identity["gate2_criteria"]["mandatory_conditions"]
    assert mandatory["c4_auroc_improvement_minimum_absolute_fraction"] == 0.02
    assert mandatory["c6_auprc_improvement_minimum_absolute_fraction"] == 0.02


def test_pamr_gate2_role_frozen():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    role = identity["gate2_criteria"]["pamr_role"]
    assert role["pamr_statistics_required_for_gate2_completion"] is True
    assert role["pamr_not_an_input_to_diagnostic_classifier"] is True
    assert role["pamr_not_a_gate2_feature"] is True
    assert role["pamr_comparison_to_final_dfp_selector_deferred_to_gate3"] is True
    assert role["pamr_numerical_superiority_threshold_at_gate2"] is False
    assert identity["gate2_criteria"]["inconclusive_conditions"]["missing_or_invalid_pamr_statistics"] is True


# ---------------------------------------------------------------------------
# Structural adversarial matrix.
# ---------------------------------------------------------------------------


def test_duplicate_toml_key_rejected(tmp_path):
    text = _raw_text()
    injected = text.replace(
        'name = "e15-dfp-phase1b-protocol"',
        'name = "e15-dfp-phase1b-protocol"\nname = "duplicate"',
        1,
    )
    path = tmp_path / "identity.toml"
    path.write_text(injected, encoding="utf-8")
    with pytest.raises(DfpPhase1bIdentityError):
        load_identity(path, repo_root=ROOT)


def test_unknown_top_level_section_rejected(tmp_path):
    p = _write_mutated(tmp_path, "[future_step_bindings]", "[unexpected_section]\nx = 1\n\n[future_step_bindings]")
    with pytest.raises(DfpPhase1bIdentityError):
        load_identity(p, repo_root=ROOT)


def test_unknown_field_within_section_rejected(tmp_path):
    p = _write_mutated(tmp_path, "fold_count = 5", "fold_count = 5\nextra_field = 1")
    with pytest.raises(DfpPhase1bIdentityError):
        load_identity(p, repo_root=ROOT)


def test_missing_field_within_section_rejected(tmp_path):
    p = _write_mutated(tmp_path, "fold_count = 5\n", "")
    with pytest.raises(DfpPhase1bIdentityError):
        load_identity(p, repo_root=ROOT)


def test_wrong_format_version_rejected(tmp_path):
    p = _write_mutated(
        tmp_path,
        'format_version = "talk2dino-dfp-phase1b-protocol-identity-v1"',
        'format_version = "talk2dino-dfp-phase1b-protocol-identity-v2"',
    )
    with pytest.raises(DfpPhase1bIdentityError):
        load_identity(p, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Authority-binding mutations.
# ---------------------------------------------------------------------------


def test_altered_phase1a_protocol_hash_rejected(tmp_path):
    zeros = "0" * 64
    p = _write_mutated(
        tmp_path,
        '[authority_bindings.phase1a_protocol]\nrelative_path = "evaluation_identities/e13_dfp_phase1a_protocol.toml"\nsha256 = "227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173"',
        f'[authority_bindings.phase1a_protocol]\nrelative_path = "evaluation_identities/e13_dfp_phase1a_protocol.toml"\nsha256 = "{zeros}"',
    )
    with pytest.raises(DfpPhase1bIdentityError):
        load_identity(p, repo_root=ROOT)


def test_altered_boundary_iou_identity_hash_rejected(tmp_path):
    ones = "1" * 64
    p = _write_mutated(
        tmp_path,
        'sha256 = "fe4c33d4376d40fcd53f7751405ffc60e5856ce9919f548df3c039809ebc0ab8"',
        f'sha256 = "{ones}"',
    )
    with pytest.raises(DfpPhase1bIdentityError):
        load_identity(p, repo_root=ROOT)


def test_altered_k12_graph_recurrence_hash_rejected(tmp_path):
    twos = "2" * 64
    p = _write_mutated(
        tmp_path,
        'sha256 = "1f4eea731ae7b803a5e2d572d0a1f59b3275ec240d36061d03540706e3bbc259"',
        f'sha256 = "{twos}"',
    )
    with pytest.raises(DfpPhase1bIdentityError):
        load_identity(p, repo_root=ROOT)


def test_live_phase1a_protocol_hash_mismatch_rejected(tmp_path):
    """Uses a fake root with a modified COPY of the Phase 1A protocol
    (never touching the real one) to exercise the LIVE re-verification
    path in verify_authority_bindings, distinct from the identity-text
    mutation tests above."""
    fake_root = tmp_path
    (fake_root / "evaluation_identities").mkdir(parents=True)
    (fake_root / "evaluation_identities" / "e13_dfp_phase1a_protocol.toml").write_text(
        PHASE1A_PROTOCOL_PATH.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8"
    )
    for rel in (
        "evaluation_identities/e13_dfp_phase1a_artifacts.toml",
        "evaluation_identities/e14_dfp_phase1b_boundary_iou.toml",
        "evaluation_identities/e12_matched_k11_k12_t320.toml",
        "evaluation_identities/e3_paired_soft_routing.toml",
        "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
    ):
        dest = fake_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    for rel in (
        "src/dfp_phase1a_stitching_pullback.py", "src/dfp_phase1a_final_rescale_pullback.py",
        "src/dfp_phase1a_global_unary_components.py", "src/dfp_phase1a_soft_membership.py",
    ):
        dest = fake_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    identity_path = fake_root / "evaluation_identities" / "e15_dfp_phase1b_protocol.toml"
    identity_path.write_bytes(IDENTITY_PATH.read_bytes())
    with pytest.raises(DfpPhase1bIdentityError, match="SHA256 mismatch"):
        verify_authority_bindings(identity_path=identity_path, repo_root=fake_root)


# ---------------------------------------------------------------------------
# PAMR historical-authority live re-verification (minimal fake_root only --
# never a full-repository copytree; see Alliance login-node guidance).
# ---------------------------------------------------------------------------

PAMR_IMPLEMENTATION_REL = "src/open_vocabulary_segmentation/models/dinotext/pamr.py"
PAMR_CONFIGURATION_REL = "src/open_vocabulary_segmentation/models/dinotext/dinotext.py"
PAMR_INHERITED_CONFIG_REL = "src/open_vocabulary_segmentation/configs/stuff/eval_stuff_pamr.yml"


def _build_pamr_fake_root(tmp_path: Path) -> Path:
    """A minimal fake_root containing only the 4 small files
    `verify_pamr_authority_bindings` actually touches: the e15 identity
    itself and the 3 bound PAMR files, byte-identical to the real repo."""
    fake_root = tmp_path / "fake_root"
    for rel in (
        "evaluation_identities/e15_dfp_phase1b_protocol.toml",
        PAMR_IMPLEMENTATION_REL,
        PAMR_CONFIGURATION_REL,
        PAMR_INHERITED_CONFIG_REL,
    ):
        dest = fake_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    return fake_root


def test_pamr_authority_bindings_verify_via_minimal_fake_root(tmp_path):
    from src.dfp_phase1b_identity import verify_pamr_authority_bindings

    fake_root = _build_pamr_fake_root(tmp_path)
    verified = verify_pamr_authority_bindings(repo_root=fake_root)
    assert len(verified["implementation_sha256"]) == 64


def test_pamr_implementation_byte_tamper_rejected(tmp_path):
    from src.dfp_phase1b_identity import verify_pamr_authority_bindings

    fake_root = _build_pamr_fake_root(tmp_path)
    target = fake_root / PAMR_IMPLEMENTATION_REL
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    with pytest.raises(DfpPhase1bIdentityError, match="implementation SHA256 mismatch"):
        verify_pamr_authority_bindings(repo_root=fake_root)


def test_pamr_configuration_byte_tamper_rejected(tmp_path):
    from src.dfp_phase1b_identity import verify_pamr_authority_bindings

    fake_root = _build_pamr_fake_root(tmp_path)
    target = fake_root / PAMR_CONFIGURATION_REL
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    with pytest.raises(DfpPhase1bIdentityError, match="configuration SHA256 mismatch"):
        verify_pamr_authority_bindings(repo_root=fake_root)


def test_pamr_inherited_configuration_byte_tamper_rejected(tmp_path):
    from src.dfp_phase1b_identity import verify_pamr_authority_bindings

    fake_root = _build_pamr_fake_root(tmp_path)
    target = fake_root / PAMR_INHERITED_CONFIG_REL
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    with pytest.raises(DfpPhase1bIdentityError, match="inherited configuration.*SHA256 mismatch"):
        verify_pamr_authority_bindings(repo_root=fake_root)


def test_pamr_implementation_missing_file_rejected(tmp_path):
    from src.dfp_phase1b_identity import verify_pamr_authority_bindings

    fake_root = _build_pamr_fake_root(tmp_path)
    (fake_root / PAMR_IMPLEMENTATION_REL).unlink()
    with pytest.raises(DfpPhase1bIdentityError, match="does not exist"):
        verify_pamr_authority_bindings(repo_root=fake_root)


def test_pamr_inherited_configuration_missing_file_rejected(tmp_path):
    from src.dfp_phase1b_identity import verify_pamr_authority_bindings

    fake_root = _build_pamr_fake_root(tmp_path)
    (fake_root / PAMR_INHERITED_CONFIG_REL).unlink()
    with pytest.raises(DfpPhase1bIdentityError, match="does not exist"):
        verify_pamr_authority_bindings(repo_root=fake_root)


def test_pamr_implementation_symlink_escape_rejected(tmp_path):
    from src.dfp_phase1b_identity import verify_pamr_authority_bindings

    fake_root = _build_pamr_fake_root(tmp_path)
    outside = tmp_path / "outside_repo"
    outside.mkdir()
    decoy = outside / "pamr.py"
    decoy.write_bytes((ROOT / PAMR_IMPLEMENTATION_REL).read_bytes())
    target = fake_root / PAMR_IMPLEMENTATION_REL
    target.unlink()
    try:
        target.symlink_to(decoy)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported in this environment")
    with pytest.raises(DfpPhase1bIdentityError, match="escapes the repository root"):
        verify_pamr_authority_bindings(repo_root=fake_root)


def test_pamr_resolved_parameter_digest_mismatch_rejected(tmp_path):
    """A live-recomputed digest mismatch is caught even if every bound
    file's SHA-256 itself still matches -- i.e. the frozen
    `resolved_parameter_digest` field is not independently mutable
    without the loader's closed-schema check already catching it, so
    this exercises the digest recomputation path directly via a
    monkeypatched binding rather than a TOML mutation (which the
    generic content-mutation matrix already covers for every other
    field in this section)."""
    from src.dfp_phase1b_identity import compute_resolved_pamr_parameter_digest, load_identity

    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    binding = dict(identity["rgb_local_boundary_baseline"])
    binding["iterations"] = 999
    assert compute_resolved_pamr_parameter_digest(binding) != identity["rgb_local_boundary_baseline"]["resolved_parameter_digest"]


# ---------------------------------------------------------------------------
# Protocol-content mutation matrix (Section 27 required list).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old,new,label",
    [
        # Target labels.
        ('harmful_definition = "GT(x) == a(x) and GT(x) != b(x)"', 'harmful_definition = "GT(x) == a(x)"', "target_label_harmful"),
        ('third_label_excluded_from_binary_negative_class = true', 'third_label_excluded_from_binary_negative_class = false', "third_label_binary_leakage_flag"),
        # Feature formulas.
        ('eta_net_formula = "eta_net = m320 / (m320 - m0)"', 'eta_net_formula = "eta_net = m320 / m0"', "eta_net_formula"),
        ('phi_formula = "phi(x) = sum_{q=1..k_star} G_positive_sorted[q] / positive_evidence_sum"', 'phi_formula = "phi(x) = k_star / n_positive_components"', "phi_formula"),
        # Component definition.
        ('connectivity = 8', 'connectivity = 4', "component_connectivity"),
        ('morphological_closing_forbidden = true', 'morphological_closing_forbidden = false', "morphological_closing"),
        # Contribution / conservation.
        ('conservation_identity = "sum_r G[x,r] = m320(x) - m0(x) = delta_m(x)"', 'conservation_identity = "sum_r G[x,r] = m320(x)"', "conservation_identity"),
        ('signed_component_evidence_formula = "G[x,r] = [S320^(r)(x,b) - S320^(r)(x,a)] - [S0^(r)(x,b) - S0^(r)(x,a)]"', 'signed_component_evidence_formula = "G[x,r] = S320^(r)(x,b)"', "contribution_formula"),
        # Q_w.
        ('q_w_definition = "Q_w = { a(x), b(x) : x is disputed and final window w contributes nonzero stitching weight to x }"', 'q_w_definition = "Q_w = all classes in the image"', "q_w_definition"),
        # Fold assignment.
        ('fold_id_formula = "fold_id = sorted_rank mod 5"', 'fold_id_formula = "fold_id = random.randint(0,4)"', "fold_id_formula"),
        ('namespace = "talk2dino-dfp-phase1b-fold-assignment-v1"', 'namespace = "some-other-namespace"', "fold_namespace"),
        # Training subsampling.
        ('maximum_per_image_per_binary_category = 4096', 'maximum_per_image_per_binary_category = 8192', "training_subsample_max"),
        # Model features / logistic regression parameters.
        ("c = 1.0", "c = 10.0", "logreg_c"),
        ('solver = "lbfgs"', 'solver = "liblinear"', "logreg_solver"),
        ("max_iterations = 1000", "max_iterations = 100", "logreg_max_iter"),
        # Bootstrap procedure.
        ("replicates = 10000", "replicates = 1000", "bootstrap_replicates"),
        ('numpy_quantile_method = "linear"', 'numpy_quantile_method = "nearest"', "bootstrap_quantile_method"),
        # Gate-2 thresholds.
        ("c3_provenance_auroc_minimum = 0.65", "c3_provenance_auroc_minimum = 0.55", "gate2_c3_threshold"),
        ("c4_auroc_improvement_minimum_absolute_fraction = 0.02", "c4_auroc_improvement_minimum_absolute_fraction = 0.01", "gate2_c4_threshold"),
        ("provenance_auroc_below = 0.60", "provenance_auroc_below = 0.50", "gate2_stop_threshold"),
        # Boundary IoU / mIoU role.
        ('primary_segmentation_metric = "mean Boundary IoU"', 'primary_segmentation_metric = "mIoU"', "boundary_iou_primary_role"),
        ('secondary_preservation_metric = "mIoU"', 'secondary_preservation_metric = "mean Boundary IoU"', "miou_secondary_role"),
        # Baseline variants.
        (
            'required = [\n    "t0", "t16", "t32", "t64", "t320", "oracle_o_p",\n    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",\n    "t0_pamr", "t320_pamr",\n]',
            'required = ["t0", "t320"]',
            "baseline_variants_required",
        ),
        # Calibration procedure.
        ("window_count = 200", "window_count = 50", "calibration_window_count"),
        ("numeric_tolerance_invented_in_this_protocol = false", "numeric_tolerance_invented_in_this_protocol = true", "calibration_tolerance_invented_flag"),
        # Resource ceiling.
        ("single_gpu_only = true", "single_gpu_only = false", "resource_ceiling_single_gpu"),
        ("peak_allocated_memory_ceiling_fraction_of_physical_gpu_memory = 0.8", "peak_allocated_memory_ceiling_fraction_of_physical_gpu_memory = 0.95", "resource_ceiling_memory_fraction"),
        # GT-isolation contract.
        ("driver_api_must_not_accept_gt_arrays_or_dataset_annotation_objects = true", "driver_api_must_not_accept_gt_arrays_or_dataset_annotation_objects = false", "gt_isolation_driver_api"),
        ("tests_must_poison_gt_access_during_stage_a = true", "tests_must_poison_gt_access_during_stage_a = false", "gt_isolation_poison_requirement"),
        # PAMR: variant-set integrity.
        (
            'required = [\n    "t0", "t16", "t32", "t64", "t320", "oracle_o_p",\n    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",\n    "t0_pamr", "t320_pamr",\n]',
            'required = [\n    "t0", "t16", "t32", "t64", "t320", "oracle_o_p",\n    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",\n    "t0_pamr",\n]',
            "pamr_t320_only_variant_dropped",
        ),
        (
            'required = [\n    "t0", "t16", "t32", "t64", "t320", "oracle_o_p",\n    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",\n    "t0_pamr", "t320_pamr",\n]',
            'required = [\n    "t0", "t16", "t32", "t64", "t320", "oracle_o_p",\n    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",\n    "t320_pamr",\n]',
            "pamr_t0_only_variant_dropped",
        ),
        (
            'required = [\n    "t0", "t16", "t32", "t64", "t320", "oracle_o_p",\n    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",\n    "t0_pamr", "t320_pamr",\n]',
            'required = [\n    "t0", "t16", "t32", "t64", "t320", "oracle_o_p",\n    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",\n    "t0_pamr", "t320_pamr", "t320_pamr_v2",\n]',
            "pamr_extra_unapproved_variant",
        ),
        (
            'required = [\n    "t0", "t16", "t32", "t64", "t320", "oracle_o_p",\n    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",\n    "t0_pamr", "t320_pamr",\n]',
            'required = [\n    "t16", "t0", "t32", "t64", "t320", "oracle_o_p",\n    "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",\n    "t0_pamr", "t320_pamr",\n]',
            "pamr_original_ten_reordered",
        ),
        # PAMR: same-parameters / same-geometry / causal-non-contamination guarantees.
        ("same_pamr_parameters_for_both = true", "same_pamr_parameters_for_both = false", "pamr_different_params_for_both"),
        ("same_rgb_input_for_both = true", "same_rgb_input_for_both = false", "pamr_different_rgb_input"),
        ("same_output_geometry_for_both = true", "same_output_geometry_for_both = false", "pamr_different_output_geometry"),
        ("same_argmax_convention_for_both = true", "same_argmax_convention_for_both = false", "pamr_different_argmax_convention"),
        ('argmax_order = "pamr_before_argmax"', 'argmax_order = "pamr_after_argmax"', "pamr_after_argmax"),
        (
            "pamr_does_not_feed_provenance_features = true",
            "pamr_does_not_feed_provenance_features = false",
            "pamr_feeds_provenance_features",
        ),
        (
            "pamr_is_not_used_to_construct_components = true",
            "pamr_is_not_used_to_construct_components = false",
            "pamr_used_to_construct_components",
        ),
        (
            "pamr_is_not_used_to_select_disputed_pixels = true",
            "pamr_is_not_used_to_select_disputed_pixels = false",
            "pamr_used_to_select_disputed_pixels",
        ),
        (
            "selection_did_not_use_phase1b_results = true",
            "selection_did_not_use_phase1b_results = false",
            "pamr_selection_tuned_on_phase1b_results",
        ),
        (
            "t0_t320_remain_the_source_pair_defining_disputes_and_provenance = true",
            "t0_t320_remain_the_source_pair_defining_disputes_and_provenance = false",
            "pamr_defines_disputes",
        ),
        ("pamr_does_not_redefine_a_or_b = true", "pamr_does_not_redefine_a_or_b = false", "pamr_redefines_a_or_b"),
        (
            "pamr_does_not_affect_g_k_star_phi_eta_net_components_soft_membership_or_q_w = true",
            "pamr_does_not_affect_g_k_star_phi_eta_net_components_soft_membership_or_q_w = false",
            "pamr_affects_provenance_quantities",
        ),
        # PAMR: Gate-2 role guarantees.
        ("pamr_statistics_required_for_gate2_completion = true", "pamr_statistics_required_for_gate2_completion = false", "pamr_gate2_stats_not_required"),
        ("pamr_not_an_input_to_diagnostic_classifier = true", "pamr_not_an_input_to_diagnostic_classifier = false", "pamr_used_as_classifier_input"),
        ("pamr_not_a_gate2_feature = true", "pamr_not_a_gate2_feature = false", "pamr_used_as_gate2_feature"),
        (
            "missing_or_invalid_pamr_statistics = true",
            "missing_or_invalid_pamr_statistics = false",
            "pamr_missing_stats_not_inconclusive",
        ),
        # PAMR: AUROC/AUPRC unit wording (Section-15 required exhaustive matrix).
        ('auroc_unit = "fraction_0_1"', 'auroc_unit = "percent_0_100"', "auroc_unit_as_percent"),
        ('auprc_unit = "fraction_0_1"', 'auprc_unit = "percent_0_100"', "auprc_unit_as_percent"),
        ('delta_auroc_unit = "absolute_fraction_0_1"', 'delta_auroc_unit = "percentage_points"', "delta_auroc_unit_as_pp"),
        ('delta_auprc_unit = "absolute_fraction_0_1"', 'delta_auprc_unit = "percentage_points"', "delta_auprc_unit_as_pp"),
        (
            "c4_auroc_improvement_minimum_absolute_fraction = 0.02",
            "c4_auroc_improvement_minimum_absolute_fraction = 0.0002",
            "gate2_c4_threshold_misrescaled_as_pp_fraction",
        ),
        (
            "c4_auroc_improvement_minimum_absolute_fraction = 0.02",
            "c4_auroc_improvement_minimum_absolute_fraction = 2.0",
            "gate2_c4_threshold_misrescaled_as_percent",
        ),
        (
            "c6_auprc_improvement_minimum_absolute_fraction = 0.02",
            "c6_auprc_improvement_minimum_absolute_fraction = 0.0002",
            "gate2_c6_threshold_misrescaled_as_pp_fraction",
        ),
        (
            "c6_auprc_improvement_minimum_absolute_fraction = 0.02",
            "c6_auprc_improvement_minimum_absolute_fraction = 2.0",
            "gate2_c6_threshold_misrescaled_as_percent",
        ),
        (
            "interval_endpoint_unit = \"absolute_fraction_0_1\"",
            "interval_endpoint_unit = \"percentage_points\"",
            "bootstrap_interval_endpoint_unit_as_pp",
        ),
        (
            "paired_difference_unit = \"absolute_fraction_0_1\"",
            "paired_difference_unit = \"percentage_points\"",
            "bootstrap_paired_difference_unit_as_pp",
        ),
    ],
)
def test_protocol_content_mutation_rejected(tmp_path, old, new, label):
    p = _write_mutated(tmp_path, old, new, name=f"{label}.toml")
    with pytest.raises(DfpPhase1bIdentityError):
        load_identity(p, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Isolation.
# ---------------------------------------------------------------------------


def test_module_import_does_not_import_torch():
    import subprocess

    src = "import sys\nimport src.dfp_phase1b_identity\nprint('torch' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", src], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_keyboard_interrupt_propagates():
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        except DfpPhase1bIdentityError:
            pytest.fail("KeyboardInterrupt must never be caught by the domain error handler")


def test_system_exit_propagates():
    with pytest.raises(SystemExit):
        try:
            raise SystemExit(3)
        except DfpPhase1bIdentityError:
            pytest.fail("SystemExit must never be caught by the domain error handler")


def test_memory_error_propagates():
    with pytest.raises(MemoryError):
        try:
            raise MemoryError()
        except DfpPhase1bIdentityError:
            pytest.fail("MemoryError must never be caught by the domain error handler")
