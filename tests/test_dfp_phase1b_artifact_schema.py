"""Tests for the DFP Phase 1B artifact-schema successor identity
(evaluation_identities/e15_dfp_phase1b_artifacts.toml,
src/dfp_phase1b_artifact_schema.py). CPU-only throughout: no CUDA, no
model, no dataset, no real COCO-Stuff-500 access, no scientific
computation (every fixture is a hand-built, already-consistent array
set; these tests only exercise VALIDATION of already-given arrays).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.dfp_phase1b_artifact_schema import (  # noqa: E402
    ALL_BOUNDARY_MEMBER_NAMES,
    ALL_SEMANTIC_MEMBER_NAMES,
    BOUNDARY_STAT_NAMES,
    DfpPhase1bArtifactSchemaError,
    HARMFUL_BINARY_LABEL_NOT_ELIGIBLE_SENTINEL,
    ORIGINAL_TEN_PREDICTION_VARIANT_NAMES,
    PAMR_PREDICTION_VARIANT_NAMES,
    PREDICTION_VARIANT_NAMES,
    SEMANTIC_STAT_NAMES,
    STAGE_A_FLAT_MEMBER_NAMES,
    STAGE_B_INT64_PER_PIXEL_MEMBER_NAMES,
    TARGET_CATEGORY_BENEFICIAL,
    TARGET_CATEGORY_HARMFUL,
    TARGET_CATEGORY_THIRD_LABEL,
    _check_zip_member_hygiene,
    boundary_member_name,
    load_identity,
    semantic_member_name,
    validate_boundary_confusion_arrays,
    validate_confusion_arrays,
    validate_semantic_confusion_arrays,
    validate_stage_a_flat_arrays,
    validate_stage_a_member_names,
    validate_stage_a_offsets,
    validate_stage_b_digest_binding,
    validate_stage_b_member_names,
    validate_stage_b_per_pixel_arrays,
    validate_stage_b_target_category_counts,
    verify_artifact_schema_bindings,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e15_dfp_phase1b_artifacts.toml"


# ---------------------------------------------------------------------------
# Identity loading.
# ---------------------------------------------------------------------------


def test_real_identity_loads():
    identity = load_identity(IDENTITY_PATH, repo_root=ROOT)
    assert identity["identity"]["name"] == "e15-dfp-phase1b-artifacts"


def test_real_protocol_binding_verifies():
    h = verify_artifact_schema_bindings(repo_root=ROOT)
    assert len(h) == 64


def test_unknown_schema_name_rejected(tmp_path):
    text = IDENTITY_PATH.read_text(encoding="utf-8")
    mutated = text.replace(
        'schema_name = "talk2dino-dfp-phase1b-artifacts-identity-v1"',
        'schema_name = "talk2dino-dfp-phase1b-artifacts-identity-v2"',
        1,
    )
    p = tmp_path / "mutated.toml"
    p.write_text(mutated)
    from src.dfp_phase1b_artifact_schema import DfpPhase1bArtifactSchemaError as Err

    with pytest.raises(Err):
        load_identity(p, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Stage A: member-name closed-set + GT-name denylist.
# ---------------------------------------------------------------------------


def test_stage_a_member_names_accepts_exact_set():
    validate_stage_a_member_names(STAGE_A_FLAT_MEMBER_NAMES)


def test_stage_a_member_names_rejects_missing_member():
    incomplete = set(STAGE_A_FLAT_MEMBER_NAMES) - {"phi"}
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_member_names(incomplete)


def test_stage_a_member_names_rejects_unknown_member():
    extra = set(STAGE_A_FLAT_MEMBER_NAMES) | {"unexpected_member"}
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_member_names(extra)


@pytest.mark.parametrize("gt_name", ["gt_class", "harmful_flag", "beneficial_flag", "third_label_flag", "oracle_decision", "gt_boundary_label"])
def test_stage_a_member_names_rejects_gt_shaped_name(gt_name):
    poisoned = (set(STAGE_A_FLAT_MEMBER_NAMES) - {"phi"}) | {gt_name}
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_member_names(poisoned)


def test_stage_a_v1_style_member_names_rejected():
    """Cross-feeding guard: Phase 1A v2 per-image-stats member names
    (e.g. harmful_pixel_count) must never validate as Stage-A members."""
    poisoned = (set(STAGE_A_FLAT_MEMBER_NAMES) - {"m0"}) | {"harmful_pixel_count"}
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_member_names(poisoned)


# ---------------------------------------------------------------------------
# Stage A: array fixtures.
# ---------------------------------------------------------------------------


def _consistent_stage_a_arrays(n: int, *, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 50, size=n).astype(np.int64)
    b = (a + 1 + rng.integers(0, 5, size=n)).astype(np.int64)  # guarantees a != b
    m0 = -rng.random(n) - 1e-6  # strictly negative-ish, but must be <= 0; allow 0 too
    m0 = np.minimum(m0, 0.0)
    m320 = rng.random(n) + 1e-6  # strictly positive
    margin_change = m320 - m0
    eta_net = m320 / margin_change
    positive_evidence_sum = rng.random(n) + 0.5
    negative_evidence_sum = -rng.random(n)
    negative_evidence_sum = np.minimum(negative_evidence_sum, 0.0)
    sum_component_evidence = margin_change.copy()  # exact conservation
    conservation_residual = np.abs(sum_component_evidence - margin_change)
    n_positive_components = rng.integers(1, 10, size=n).astype(np.int64)
    k_star = np.array([rng.integers(1, int(npos) + 1) for npos in n_positive_components], dtype=np.int64)
    phi = rng.random(n) * 0.9 + 0.1
    top_positive_component_id = rng.integers(0, 100, size=n).astype(np.int64)
    top_positive_evidence = rng.random(n) * positive_evidence_sum
    top1_positive_share = np.clip(top_positive_evidence / positive_evidence_sum, 1e-9, 1.0)
    return {
        "image_row_index": rng.integers(0, 5, size=n).astype(np.int64),
        "final_pixel_linear_index": rng.integers(0, 1000, size=n).astype(np.int64),
        "unary_class_a": a,
        "propagated_class_b": b,
        "t16_label": rng.integers(0, 50, size=n).astype(np.int64),
        "t32_label": rng.integers(0, 50, size=n).astype(np.int64),
        "t64_label": rng.integers(0, 50, size=n).astype(np.int64),
        "m0": m0,
        "m320": m320,
        "margin_change": margin_change,
        "eta_net": eta_net,
        "text_cosine_ab": rng.random(n) * 2 - 1,
        "positive_evidence_sum": positive_evidence_sum,
        "negative_evidence_sum": negative_evidence_sum,
        "sum_component_evidence": sum_component_evidence,
        "conservation_residual": conservation_residual,
        "n_positive_components": n_positive_components,
        "k_star": k_star,
        "phi": phi,
        "top_positive_component_id": top_positive_component_id,
        "top_positive_evidence": top_positive_evidence,
        "top1_positive_share": top1_positive_share,
    }


def test_stage_a_flat_arrays_zero_rows():
    arrays = _consistent_stage_a_arrays(0)
    validate_stage_a_flat_arrays(arrays, row_count=0, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_one_image_single_row():
    arrays = _consistent_stage_a_arrays(1)
    validate_stage_a_flat_arrays(arrays, row_count=1, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_multiple_rows():
    arrays = _consistent_stage_a_arrays(37)
    validate_stage_a_flat_arrays(arrays, row_count=37, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_wrong_rank_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["m0"] = arrays["m0"].reshape(1, 3)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_wrong_dtype_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["m0"] = arrays["m0"].astype(np.float32)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_wrong_shape_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["m0"] = arrays["m0"][:2]
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_nan_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["phi"] = arrays["phi"].copy()
    arrays["phi"][0] = np.nan
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_infinity_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["m320"] = arrays["m320"].copy()
    arrays["m320"][0] = np.inf
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_non_native_endian_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["m0"] = arrays["m0"].astype(">f8")
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_object_dtype_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["top_positive_component_id"] = np.array([1, 2, 3], dtype=object)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_a_equals_b_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["propagated_class_b"] = arrays["unary_class_a"].copy()
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_conservation_mismatch_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["sum_component_evidence"] = arrays["sum_component_evidence"] + 100.0
    arrays["conservation_residual"] = np.abs(arrays["sum_component_evidence"] - arrays["margin_change"])
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_invalid_k_star_zero_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["k_star"] = arrays["k_star"].copy()
    arrays["k_star"][0] = 0
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_invalid_k_star_exceeds_n_positive_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["k_star"] = arrays["k_star"].copy()
    arrays["k_star"][0] = arrays["n_positive_components"][0] + 1
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_invalid_phi_zero_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["phi"] = arrays["phi"].copy()
    arrays["phi"][0] = 0.0
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_invalid_phi_above_one_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["phi"] = arrays["phi"].copy()
    arrays["phi"][0] = 1.5
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_invalid_eta_net_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["eta_net"] = arrays["eta_net"].copy()
    arrays["eta_net"][0] = arrays["eta_net"][0] + 5.0  # disagrees with m320/margin_change
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_invalid_top1_share_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["top1_positive_share"] = arrays["top1_positive_share"].copy()
    arrays["top1_positive_share"][0] = 0.0
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_m0_positive_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["m0"] = arrays["m0"].copy()
    arrays["m0"][0] = 1.0
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_m320_nonpositive_rejected():
    arrays = _consistent_stage_a_arrays(3)
    arrays["m320"] = arrays["m320"].copy()
    arrays["m320"][0] = 0.0
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1e-9)


def test_stage_a_flat_arrays_no_default_conservation_tolerance():
    arrays = _consistent_stage_a_arrays(3)
    with pytest.raises(TypeError):
        validate_stage_a_flat_arrays(arrays, row_count=3)  # type: ignore[call-arg]


def test_stage_a_flat_arrays_conservation_tolerance_wrong_type_rejected():
    arrays = _consistent_stage_a_arrays(3)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_flat_arrays(arrays, row_count=3, conservation_tolerance=1)  # int, not float


def test_stage_a_offsets_valid():
    offsets = np.array([0, 2, 5, 5, 7], dtype=np.int64)
    counts = np.array([2, 3, 0, 2], dtype=np.int64)
    validate_stage_a_offsets(offsets, counts, flat_row_count=7)


def test_stage_a_offsets_empty_dispute_image_valid():
    """A window/image with zero disputes must be representable (offset
    delta of 0) without error."""
    offsets = np.array([0, 0, 3], dtype=np.int64)
    counts = np.array([0, 3], dtype=np.int64)
    validate_stage_a_offsets(offsets, counts, flat_row_count=3)


def test_stage_a_offsets_wrong_first_value_rejected():
    offsets = np.array([1, 2, 5], dtype=np.int64)
    counts = np.array([1, 3], dtype=np.int64)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_offsets(offsets, counts, flat_row_count=5)


def test_stage_a_offsets_non_nondecreasing_rejected():
    offsets = np.array([0, 5, 2], dtype=np.int64)
    counts = np.array([5, -3], dtype=np.int64)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_offsets(offsets, counts, flat_row_count=2)


def test_stage_a_offsets_wrong_final_value_rejected():
    offsets = np.array([0, 2, 5], dtype=np.int64)
    counts = np.array([2, 3], dtype=np.int64)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_offsets(offsets, counts, flat_row_count=99)


def test_stage_a_offsets_diff_disagrees_with_counts_rejected():
    offsets = np.array([0, 2, 5], dtype=np.int64)
    counts = np.array([2, 999], dtype=np.int64)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_a_offsets(offsets, counts, flat_row_count=5)


# ---------------------------------------------------------------------------
# Stage B.
# ---------------------------------------------------------------------------


def _consistent_stage_b_arrays(target_category: np.ndarray) -> dict:
    n = target_category.shape[0]
    binary_eligible = np.isin(target_category, [TARGET_CATEGORY_HARMFUL, TARGET_CATEGORY_BENEFICIAL]).astype(np.int64)
    harmful_binary_label = np.full(n, HARMFUL_BINARY_LABEL_NOT_ELIGIBLE_SENTINEL, dtype=np.int64)
    harmful_binary_label[target_category == TARGET_CATEGORY_HARMFUL] = 1
    harmful_binary_label[target_category == TARGET_CATEGORY_BENEFICIAL] = 0
    return {
        "gt_class": np.zeros(n, dtype=np.int64),
        "target_category": target_category,
        "binary_eligible": binary_eligible,
        "harmful_binary_label": harmful_binary_label,
        "boundary_interior_flag": np.zeros(n, dtype=np.int64),
        "fold_id": np.zeros(n, dtype=np.int64),
        "training_selection_flag": np.ones(n, dtype=np.int64),
    }


def test_stage_b_member_names_without_diagnostics():
    validate_stage_b_member_names(STAGE_B_INT64_PER_PIXEL_MEMBER_NAMES, diagnostic_predictions_present=False)


def test_stage_b_member_names_rejects_premature_diagnostic_members():
    poisoned = set(STAGE_B_INT64_PER_PIXEL_MEMBER_NAMES) | {"out_of_fold_cheap_control_risk"}
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_b_member_names(poisoned, diagnostic_predictions_present=False)


def test_stage_b_per_pixel_arrays_malformed_category_rejected():
    target_category = np.array([0, 1, 99], dtype=np.int64)
    arrays = _consistent_stage_b_arrays(target_category)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_b_per_pixel_arrays(
            arrays, row_count=3, image_fold_assignment=np.zeros(1, dtype=np.int64), image_row_index=np.zeros(3, dtype=np.int64),
        )


def test_stage_b_third_label_binary_leakage_rejected():
    target_category = np.array([TARGET_CATEGORY_THIRD_LABEL, TARGET_CATEGORY_HARMFUL], dtype=np.int64)
    arrays = _consistent_stage_b_arrays(target_category)
    arrays["harmful_binary_label"][0] = 0  # third-label pixel leaked into the negative class
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_b_per_pixel_arrays(
            arrays, row_count=2, image_fold_assignment=np.zeros(1, dtype=np.int64), image_row_index=np.zeros(2, dtype=np.int64),
        )


def test_stage_b_invalid_fold_rejected():
    target_category = np.array([TARGET_CATEGORY_HARMFUL], dtype=np.int64)
    arrays = _consistent_stage_b_arrays(target_category)
    arrays["fold_id"][0] = 7
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_b_per_pixel_arrays(
            arrays, row_count=1, image_fold_assignment=np.zeros(1, dtype=np.int64), image_row_index=np.zeros(1, dtype=np.int64),
        )


def test_stage_b_fold_disagrees_with_image_level_assignment_rejected():
    target_category = np.array([TARGET_CATEGORY_HARMFUL], dtype=np.int64)
    arrays = _consistent_stage_b_arrays(target_category)
    arrays["fold_id"][0] = 2
    image_fold_assignment = np.array([4], dtype=np.int64)  # image 0's frozen fold is 4, not 2
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_b_per_pixel_arrays(
            arrays, row_count=1, image_fold_assignment=image_fold_assignment, image_row_index=np.zeros(1, dtype=np.int64),
        )


def test_stage_b_binary_eligible_disagreement_rejected():
    target_category = np.array([TARGET_CATEGORY_HARMFUL], dtype=np.int64)
    arrays = _consistent_stage_b_arrays(target_category)
    arrays["binary_eligible"][0] = 0  # should be 1 for harmful
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_b_per_pixel_arrays(
            arrays, row_count=1, image_fold_assignment=np.zeros(1, dtype=np.int64), image_row_index=np.zeros(1, dtype=np.int64),
        )


def test_stage_b_target_category_counts_valid():
    target_category = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    validate_stage_b_target_category_counts(target_category, disputes_total=6)


def test_stage_b_target_category_counts_mismatch_rejected():
    target_category = np.array([0, 0, 1], dtype=np.int64)
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_b_target_category_counts(target_category, disputes_total=99)


def test_stage_b_digest_binding_match():
    validate_stage_b_digest_binding("abc123", "abc123")


def test_stage_b_digest_binding_mismatch_rejected():
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_stage_b_digest_binding("abc123", "def456")


# ---------------------------------------------------------------------------
# Prediction-variant confusion arrays (semantic + Boundary IoU).
# ---------------------------------------------------------------------------


def _build_confusion(image_count, class_count, *, stat_names, member_name_fn, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {}
    gt_shared = rng.integers(0, 20, size=(image_count, class_count)).astype(np.int64)
    for v in PREDICTION_VARIANT_NAMES:
        pred = rng.integers(0, 20, size=(image_count, class_count)).astype(np.int64)
        inter = np.minimum(pred, gt_shared)
        union = pred + gt_shared - inter
        arrays[member_name_fn(v, stat_names[0])] = inter
        arrays[member_name_fn(v, stat_names[1])] = union
        arrays[member_name_fn(v, stat_names[2])] = pred
        arrays[member_name_fn(v, stat_names[3])] = gt_shared.copy()
    return arrays


def test_semantic_confusion_arrays_valid():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_boundary_confusion_arrays_valid():
    arrays = _build_confusion(4, 6, stat_names=BOUNDARY_STAT_NAMES, member_name_fn=boundary_member_name)
    validate_boundary_confusion_arrays(arrays, image_count=4, class_count=6)


def test_semantic_confusion_arrays_inconsistent_union_rejected():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    name = semantic_member_name("t0", "union")
    arrays[name] = arrays[name] + 1
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_boundary_confusion_arrays_inconsistent_union_rejected():
    arrays = _build_confusion(4, 6, stat_names=BOUNDARY_STAT_NAMES, member_name_fn=boundary_member_name)
    name = boundary_member_name("t320", "boundary_union")
    arrays[name] = arrays[name] + 1
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_boundary_confusion_arrays(arrays, image_count=4, class_count=6)


def test_semantic_confusion_arrays_cross_variant_gt_mismatch_rejected():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    name = semantic_member_name("t320", "ground_truth_pixels")
    arrays[name] = arrays[name].copy()
    arrays[name][0, 0] += 5
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_boundary_confusion_arrays_cross_variant_gt_mismatch_rejected():
    arrays = _build_confusion(4, 6, stat_names=BOUNDARY_STAT_NAMES, member_name_fn=boundary_member_name)
    name = boundary_member_name("oracle_o_p", "ground_truth_boundary_pixels")
    arrays[name] = arrays[name].copy()
    arrays[name][0, 0] += 5
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_boundary_confusion_arrays(arrays, image_count=4, class_count=6)


def test_confusion_arrays_missing_member_rejected():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    del arrays[semantic_member_name("t64", "intersection")]
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_confusion_arrays_wrong_shape_rejected():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    name = semantic_member_name("t0", "intersection")
    arrays[name] = arrays[name][:2, :]
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_confusion_arrays_negative_values_rejected():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    name = semantic_member_name("t0", "intersection")
    arrays[name] = arrays[name].copy()
    arrays[name][0, 0] = -1
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_all_variant_names_covered():
    assert len(PREDICTION_VARIANT_NAMES) == 12
    assert len(ALL_SEMANTIC_MEMBER_NAMES) == 48
    assert len(ALL_BOUNDARY_MEMBER_NAMES) == 48


# ---------------------------------------------------------------------------
# PAMR variant / 12-variant, 48+48-member set closure (Section 15).
# ---------------------------------------------------------------------------


def test_original_ten_preserved_and_pamr_appended_last():
    assert ORIGINAL_TEN_PREDICTION_VARIANT_NAMES == (
        "t0", "t16", "t32", "t64", "t320", "oracle_o_p",
        "persist_h32_full", "persist_h32_harmful_only", "persist_h32_beneficial_only", "persist_h32_third_label_only",
    )
    assert PAMR_PREDICTION_VARIANT_NAMES == ("t0_pamr", "t320_pamr")
    assert PREDICTION_VARIANT_NAMES == ORIGINAL_TEN_PREDICTION_VARIANT_NAMES + PAMR_PREDICTION_VARIANT_NAMES


def test_semantic_confusion_arrays_missing_pamr_member_rejected():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    del arrays[semantic_member_name("t320_pamr", "intersection")]
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_boundary_confusion_arrays_missing_pamr_member_rejected():
    arrays = _build_confusion(4, 6, stat_names=BOUNDARY_STAT_NAMES, member_name_fn=boundary_member_name)
    del arrays[boundary_member_name("t0_pamr", "boundary_intersection")]
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_boundary_confusion_arrays(arrays, image_count=4, class_count=6)


def test_semantic_confusion_arrays_only_one_pamr_variant_rejected():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    for stat in SEMANTIC_STAT_NAMES:
        del arrays[semantic_member_name("t320_pamr", stat)]
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_semantic_confusion_arrays_unexpected_13th_variant_rejected():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    gt = arrays[semantic_member_name("t0", "ground_truth_pixels")]
    for stat in SEMANTIC_STAT_NAMES:
        arrays[f"semantic_t320_pamr_v2_{stat}"] = gt.copy()
    with pytest.raises(DfpPhase1bArtifactSchemaError, match="unexpected"):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_boundary_confusion_arrays_unexpected_13th_variant_rejected():
    arrays = _build_confusion(4, 6, stat_names=BOUNDARY_STAT_NAMES, member_name_fn=boundary_member_name)
    gt = arrays[boundary_member_name("t0", "ground_truth_boundary_pixels")]
    for stat in BOUNDARY_STAT_NAMES:
        arrays[f"boundary_t320_pamr_v2_{stat}"] = gt.copy()
    with pytest.raises(DfpPhase1bArtifactSchemaError, match="unexpected"):
        validate_boundary_confusion_arrays(arrays, image_count=4, class_count=6)


def test_semantic_confusion_arrays_pamr_gt_count_mismatch_rejected():
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    name = semantic_member_name("t0_pamr", "ground_truth_pixels")
    arrays[name] = arrays[name].copy()
    arrays[name][0, 0] += 3
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


def test_boundary_confusion_arrays_pamr_gt_boundary_count_mismatch_rejected():
    arrays = _build_confusion(4, 6, stat_names=BOUNDARY_STAT_NAMES, member_name_fn=boundary_member_name)
    name = boundary_member_name("t320_pamr", "ground_truth_boundary_pixels")
    arrays[name] = arrays[name].copy()
    arrays[name][0, 0] += 3
    with pytest.raises(DfpPhase1bArtifactSchemaError):
        validate_boundary_confusion_arrays(arrays, image_count=4, class_count=6)


def test_semantic_confusion_arrays_old_ten_variant_set_incomplete_rejected():
    """The pre-repair 10-variant set (no PAMR members at all) must now be
    rejected as incomplete against the frozen 12-variant/48-member schema."""
    arrays = _build_confusion(4, 6, stat_names=SEMANTIC_STAT_NAMES, member_name_fn=semantic_member_name)
    for stat in SEMANTIC_STAT_NAMES:
        del arrays[semantic_member_name("t0_pamr", stat)]
        del arrays[semantic_member_name("t320_pamr", stat)]
    with pytest.raises(DfpPhase1bArtifactSchemaError, match="missing"):
        validate_semantic_confusion_arrays(arrays, image_count=4, class_count=6)


# ---------------------------------------------------------------------------
# ZIP / archive hygiene (malicious ZIP names, object arrays via NPZ).
# ---------------------------------------------------------------------------


def test_zip_member_hygiene_rejects_duplicate_names(tmp_path):
    p = tmp_path / "bad.npz"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.npy", b"x")
        zf.writestr("a.npy", b"y")
    with pytest.raises(DfpPhase1bArtifactSchemaError, match="duplicate"):
        _check_zip_member_hygiene(p)


def test_zip_member_hygiene_rejects_directory_entry(tmp_path):
    p = tmp_path / "bad.npz"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("subdir/", b"")
    with pytest.raises(DfpPhase1bArtifactSchemaError, match="directory entry"):
        _check_zip_member_hygiene(p)


def test_zip_member_hygiene_rejects_path_traversal_name(tmp_path):
    p = tmp_path / "bad.npz"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("../../etc/passwd.npy", b"x")
    with pytest.raises(DfpPhase1bArtifactSchemaError, match="unsafe"):
        _check_zip_member_hygiene(p)


def test_zip_member_hygiene_rejects_bad_zip(tmp_path):
    p = tmp_path / "not_a_zip.npz"
    p.write_bytes(b"this is not a zip file at all")
    with pytest.raises(DfpPhase1bArtifactSchemaError, match="not a valid ZIP"):
        _check_zip_member_hygiene(p)


def test_zip_member_hygiene_accepts_well_formed_npz(tmp_path):
    p = tmp_path / "good.npz"
    np.savez(p, a=np.zeros(3, dtype=np.int64))
    _check_zip_member_hygiene(p)  # must not raise


# ---------------------------------------------------------------------------
# Isolation.
# ---------------------------------------------------------------------------


def test_module_import_does_not_import_torch():
    import subprocess

    src = "import sys\nimport src.dfp_phase1b_artifact_schema\nprint('torch' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", src], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_keyboard_interrupt_propagates():
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        except DfpPhase1bArtifactSchemaError:
            pytest.fail("KeyboardInterrupt must never be caught by the domain error handler")


def test_system_exit_propagates():
    with pytest.raises(SystemExit):
        try:
            raise SystemExit(3)
        except DfpPhase1bArtifactSchemaError:
            pytest.fail("SystemExit must never be caught by the domain error handler")


def test_memory_error_propagates():
    with pytest.raises(MemoryError):
        try:
            raise MemoryError()
        except DfpPhase1bArtifactSchemaError:
            pytest.fail("MemoryError must never be caught by the domain error handler")
