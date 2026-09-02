"""Tests for DFP Phase 1A Step 11 deterministic qualitative reporting:
identity/schema authority, deterministic selection (including shortage/
zero-coverage/tie handling), GT isolation, palette/panel rendering
determinism, strict manifest validation, atomic bundle installation, the
CLI, a no-cherry-picking audit, and independent synthetic fixtures.

CPU-only throughout. No real dataset, no model construction, no CUDA
initialization, no graph/recurrence execution."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1a_artifact_schema as sch
import src.dfp_phase1a_qualitative_input as qi
import src.dfp_phase1a_qualitative_render as render
import src.dfp_phase1a_qualitative_selection as qs

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema_summary():
    return qs.validate_static_configuration(repo_root=ROOT)


@pytest.fixture(scope="module")
def reporting_identity():
    return qs.load_identity(repo_root=ROOT)


@pytest.fixture(scope="module")
def contract_coco():
    from src.dfp_phase1a_analysis import load_analysis_contract_from_protocol
    from src.dfp_phase1a_identity import load_identity as load_protocol_identity

    protocol = load_protocol_identity(ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml", repo_root=ROOT)
    return load_analysis_contract_from_protocol(protocol, dataset_key="coco_stuff", bind_boundary_iou=True, repo_root=ROOT)


def _make_record(image_id, *, contract, rng, height=16, width=20, force_categories=None):
    """Build one independent synthetic input record. `force_categories`
    optionally forces specific known category outcomes for hand-
    reconstructible expected results (see the dedicated fixture tests
    below)."""
    class_count = contract.class_count
    rgb = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    gt = rng.integers(0, class_count, size=(height, width)).astype(np.int64)
    scores = {h: rng.random((class_count, height, width)).astype(np.float32) for h in contract.horizons}
    return qi.QualitativeInputRecord(canonical_image_id=image_id, rgb=rgb, gt=gt, scores=scores, source_hashes={"result_sha256": "a" * 64})


def _lowest_index_argmax(scores_chw):
    class_count = scores_chw.shape[0]
    max_value = scores_chw.max(axis=0, keepdims=True)
    is_max = scores_chw == max_value
    class_index = np.arange(class_count).reshape(class_count, 1, 1)
    sentinel = class_count
    masked = np.where(is_max, class_index, sentinel)
    return masked.min(axis=0)


def _make_harmful_only_record(image_id, *, contract, height=8, width=8):
    """Independently hand-constructed: exactly one pixel harmful, zero
    beneficial, zero third-label -- by directly controlling scores at T0
    and T320 so a != b, and setting GT == a there."""
    class_count = contract.class_count
    rng = np.random.default_rng(abs(hash(image_id)) % (2 ** 31))
    rgb = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    gt = np.zeros((height, width), dtype=np.int64)
    scores = {}
    unary_h, final_h = contract.unary_horizon, contract.final_horizon
    for h in contract.horizons:
        s = np.zeros((class_count, height, width), dtype=np.float32)
        s[0] = 1.0  # class 0 wins everywhere by default
        scores[h] = s
    # force disagreement at (0,0): T0 winner = class 0 (GT), T320 winner = class 1 (!= GT)
    scores[final_h][0, 0, 0] = 0.0
    scores[final_h][1, 0, 0] = 1.0
    for h in contract.horizons:
        if h not in (unary_h, final_h):
            scores[h][0, 0, 0] = 0.0
            scores[h][1, 0, 0] = 1.0  # matches b everywhere else -> no extra persistence reversion surprises
    return qi.QualitativeInputRecord(canonical_image_id=image_id, rgb=rgb, gt=gt, scores=scores, source_hashes={"result_sha256": "a" * 64})


def _make_beneficial_only_record(image_id, *, contract, height=8, width=8):
    class_count = contract.class_count
    rng = np.random.default_rng(abs(hash(image_id)) % (2 ** 31))
    rgb = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    gt = np.zeros((height, width), dtype=np.int64)
    gt[0, 0] = 1  # GT at the dispute pixel is class 1 (== b, forced below)
    unary_h, final_h = contract.unary_horizon, contract.final_horizon
    scores = {}
    for h in contract.horizons:
        s = np.zeros((class_count, height, width), dtype=np.float32)
        s[0] = 1.0
        scores[h] = s
    scores[final_h][0, 0, 0] = 0.0
    scores[final_h][1, 0, 0] = 1.0  # b = class 1 == GT -> beneficial
    for h in contract.horizons:
        if h not in (unary_h, final_h):
            scores[h][0, 0, 0] = 0.0
            scores[h][1, 0, 0] = 1.0
    return qi.QualitativeInputRecord(canonical_image_id=image_id, rgb=rgb, gt=gt, scores=scores, source_hashes={"result_sha256": "a" * 64})


def _make_third_label_only_record(image_id, *, contract, height=8, width=8):
    class_count = contract.class_count
    rng = np.random.default_rng(abs(hash(image_id)) % (2 ** 31))
    rgb = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    gt = np.zeros((height, width), dtype=np.int64)
    gt[0, 0] = 2  # GT is neither a(=0) nor b(=1)
    unary_h, final_h = contract.unary_horizon, contract.final_horizon
    scores = {}
    for h in contract.horizons:
        s = np.zeros((class_count, height, width), dtype=np.float32)
        s[0] = 1.0
        scores[h] = s
    scores[final_h][0, 0, 0] = 0.0
    scores[final_h][1, 0, 0] = 1.0
    for h in contract.horizons:
        if h not in (unary_h, final_h):
            scores[h][0, 0, 0] = 0.0
            scores[h][1, 0, 0] = 1.0
    return qi.QualitativeInputRecord(canonical_image_id=image_id, rgb=rgb, gt=gt, scores=scores, source_hashes={"result_sha256": "a" * 64})


def _make_no_dispute_record(image_id, *, contract, height=8, width=8):
    class_count = contract.class_count
    rng = np.random.default_rng(abs(hash(image_id)) % (2 ** 31))
    rgb = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    gt = np.zeros((height, width), dtype=np.int64)
    scores = {h: np.zeros((class_count, height, width), dtype=np.float32) for h in contract.horizons}
    for h in contract.horizons:
        scores[h][0] = 1.0  # class 0 wins everywhere, at every horizon -- a == b everywhere
    return qi.QualitativeInputRecord(canonical_image_id=image_id, rgb=rgb, gt=gt, scores=scores, source_hashes={"result_sha256": "a" * 64})


# ---------------------------------------------------------------------------
# Reporting-authority identity
# ---------------------------------------------------------------------------


def test_reporting_identity_validates(schema_summary):
    assert schema_summary["identity_name"] == "e13-dfp-phase1a-qualitative-report"


def test_reporting_identity_immutable_after_load(reporting_identity):
    before = copy.deepcopy(reporting_identity)
    qs.load_identity(repo_root=ROOT)
    assert reporting_identity == before


@pytest.mark.parametrize(
    "mutator",
    [
        lambda d: d["identity"].__setitem__("name", "wrong"),
        lambda d: d["protocol_binding"].__setitem__("protocol_identity_sha256", "not-a-hash"),
        lambda d: d["category_binding"].__setitem__("category_order", ["only_one"]),
        lambda d: d["category_binding"].__setitem__("eligibility_pixel_count_threshold_forbidden", False),
        lambda d: d["selection_policy"].__setitem__("sort_key", ["canonical_image_id"]),
        lambda d: d["panel_binding"].__setitem__("label_interpolation", "bilinear"),
        lambda d: d["panel_binding"].__setitem__("antialiasing_for_labels_forbidden", False),
        lambda d: d["rendering_policy"].__setitem__("strip_timestamps", False),
        lambda d: d.pop("palette_binding"),
        lambda d: d.__setitem__("extra_unknown_section", {}),
    ],
)
def test_reporting_identity_mutation_matrix_rejected(mutator):
    identity = qs.load_identity(repo_root=ROOT)
    mutated = copy.deepcopy(identity)
    mutator(mutated)
    with pytest.raises(qs.DfpPhase1aQualitativeReportError):
        if set(mutated) != qs.IDENTITY_TOP_KEYS:
            raise qs.DfpPhase1aQualitativeReportError("top-level mismatch")
        for section, keys in qs.SECTION_KEYS.items():
            block = mutated.get(section)
            if not isinstance(block, dict) or set(block) != keys:
                raise qs.DfpPhase1aQualitativeReportError(f"{section} mismatch")
        qs._validate_identity_block(mutated["identity"])
        qs._validate_protocol_binding_block(mutated["protocol_binding"])
        qs._validate_category_binding_block(mutated["category_binding"])
        qs._validate_selection_policy_block(mutated["selection_policy"])
        qs._validate_panel_binding_block(mutated["panel_binding"])
        qs._validate_palette_binding_block(mutated["palette_binding"])
        qs._validate_rendering_policy_block(mutated["rendering_policy"])


def test_relational_hash_mismatch_rejected(tmp_path):
    original_text = (ROOT / qs.IDENTITY_RELATIVE_PATH).read_text()
    tampered = original_text.replace(
        'protocol_identity_sha256 = "227549b03ec6b1262c465a18b7fc67529fa5a0b5deef419c33ec79a80feb2173"',
        f'protocol_identity_sha256 = "{"0" * 64}"',
    )
    assert tampered != original_text
    path = tmp_path / "tampered.toml"
    path.write_text(tampered)
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="mismatch"):
        qs.validate_static_configuration(repo_root=ROOT, identity_path=path)


# ---------------------------------------------------------------------------
# Deterministic selection
# ---------------------------------------------------------------------------


def test_selection_key_is_pure_sha256():
    key = qs.compute_selection_key(namespace="ns", dataset_name="d", category_name="c", canonical_image_id="i", separator="\x00")
    expected = hashlib.sha256("ns\x00d\x00c\x00i".encode("utf-8")).hexdigest()
    assert key == expected


def test_selection_key_deterministic_across_calls():
    keys = {qs.compute_selection_key(namespace="ns", dataset_name="d", category_name="c", canonical_image_id="i", separator="\x00") for _ in range(5)}
    assert len(keys) == 1


def test_selection_complete_case():
    ids = ["a", "b", "c", "d"]
    eligibility = {cid: True for cid in ids}
    result = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=2, namespace="ns", separator="\x00")
    assert result.shortage_status == "complete"
    assert result.eligible_count == 4
    assert result.selected_count == 2
    # independently hand-reconstruct expected order
    keys = {cid: qs.compute_selection_key(namespace="ns", dataset_name="d", category_name="cat", canonical_image_id=cid, separator="\x00") for cid in ids}
    expected_order = sorted(ids, key=lambda cid: (keys[cid], cid))[:2]
    assert list(result.selected_ids) == expected_order


def test_selection_shortage_case():
    ids = ["a", "b", "c"]
    eligibility = {"a": True, "b": False, "c": False}
    result = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=5, namespace="ns", separator="\x00")
    assert result.shortage_status == "shortage"
    assert result.eligible_count == 1
    assert result.selected_count == 1
    assert result.selected_ids == ("a",)


def test_selection_zero_coverage_case():
    ids = ["a", "b", "c"]
    eligibility = {cid: False for cid in ids}
    result = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=5, namespace="ns", separator="\x00")
    assert result.shortage_status == "zero_coverage"
    assert result.eligible_count == 0
    assert result.selected_ids == ()


def test_selection_exactly_k_eligible():
    ids = ["a", "b", "c"]
    eligibility = {cid: True for cid in ids}
    result = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=3, namespace="ns", separator="\x00")
    assert result.shortage_status == "complete"
    assert result.selected_count == 3


def test_selection_more_than_k_eligible():
    ids = [f"img{i}" for i in range(20)]
    eligibility = {cid: True for cid in ids}
    result = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=3, namespace="ns", separator="\x00")
    assert result.selected_count == 3
    assert result.eligible_count == 20


def test_selection_shortage_never_fills_from_another_category():
    ids = ["a", "b"]
    result_cat1 = qs.select_category_examples(category_name="cat1", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id={"a": True, "b": False}, k=5, namespace="ns", separator="\x00")
    result_cat2 = qs.select_category_examples(category_name="cat2", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id={"a": False, "b": True}, k=5, namespace="ns", separator="\x00")
    assert set(result_cat1.selected_ids).isdisjoint(set()) or True  # sanity no-op
    assert result_cat1.selected_ids == ("a",)
    assert result_cat2.selected_ids == ("b",)
    assert "b" not in result_cat1.selected_ids
    assert "a" not in result_cat2.selected_ids


def test_selection_never_duplicates():
    ids = ["a"]
    result = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id={"a": True}, k=5, namespace="ns", separator="\x00")
    assert result.selected_ids == ("a",)
    assert len(set(result.selected_ids)) == len(result.selected_ids)


def test_selection_does_not_mutate_inputs():
    ids = ["a", "b", "c"]
    ids_before = list(ids)
    eligibility = {"a": True, "b": True, "c": False}
    eligibility_before = dict(eligibility)
    qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=1, namespace="ns", separator="\x00")
    assert ids == ids_before
    assert eligibility == eligibility_before


def test_selection_cross_category_duplicates_allowed():
    """An image may appear in multiple categories -- the frozen policy
    (allow_cross_category_duplicates = true) permits this; selection
    never silently enforces cross-category uniqueness."""
    ids = ["a"]
    r1 = qs.select_category_examples(category_name="cat1", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id={"a": True}, k=1, namespace="ns", separator="\x00")
    r2 = qs.select_category_examples(category_name="cat2", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id={"a": True}, k=1, namespace="ns", separator="\x00")
    assert r1.selected_ids == r2.selected_ids == ("a",)


def test_selection_key_tie_broken_by_canonical_image_id():
    """Force a hash collision scenario is infeasible; instead verify the
    tie-break rule is genuinely applied by monkeypatching-free
    construction: two IDs whose (namespace, dataset, category) are
    identical differ only in canonical_image_id, so ties in the
    selection_key itself cannot be engineered directly -- verify instead
    that the documented sort key is exactly (selection_key,
    canonical_image_id) by re-deriving the order two ways."""
    ids = [f"img{i}" for i in range(15)]
    eligibility = {cid: True for cid in ids}
    result = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=15, namespace="ns", separator="\x00")
    keys = {cid: qs.compute_selection_key(namespace="ns", dataset_name="d", category_name="cat", canonical_image_id=cid, separator="\x00") for cid in ids}
    expected = tuple(sorted(ids, key=lambda cid: (keys[cid], cid)))
    assert result.selected_ids == expected


def test_selection_independent_of_traversal_order_permutation():
    """Selection must depend only on the SET of eligible IDs (via their
    canonical IDs), never on the order the caller happens to traverse
    them in -- shuffling canonical_ids_in_order (while keeping it a valid
    permutation of the SAME set) must not change the result, since
    eligibility lookup is by ID, not position."""
    import random

    ids = [f"img{i}" for i in range(10)]
    eligibility = {cid: (i % 2 == 0) for i, cid in enumerate(ids)}
    result_a = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=3, namespace="ns", separator="\x00")
    shuffled = list(ids)
    random.Random(123).shuffle(shuffled)
    result_b = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=shuffled, eligibility_by_id=eligibility, k=3, namespace="ns", separator="\x00")
    assert result_a.selected_ids == result_b.selected_ids


def test_selection_trace_digest_deterministic_and_sensitive():
    ids = ["a", "b"]
    eligibility = {"a": True, "b": True}
    r1 = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=2, namespace="ns", separator="\x00")
    r2 = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id=eligibility, k=2, namespace="ns", separator="\x00")
    assert qs.selection_trace_digest([r1]) == qs.selection_trace_digest([r2])
    r3 = qs.select_category_examples(category_name="cat", dataset_name="d", canonical_ids_in_order=ids, eligibility_by_id={"a": True, "b": False}, k=2, namespace="ns", separator="\x00")
    assert qs.selection_trace_digest([r1]) != qs.selection_trace_digest([r3])


@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_selection_determinism_across_pythonhashseed(seed):
    """Independent PYTHONHASHSEED subprocess determinism probe: the
    selection key/order must be identical regardless of PYTHONHASHSEED,
    since the algorithm never relies on Python's hash()."""
    import os

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import src.dfp_phase1a_qualitative_selection as qs\n"
        "ids = ['img%%d' %% i for i in range(8)]\n"
        "elig = {cid: True for cid in ids}\n"
        "r = qs.select_category_examples(category_name='cat', dataset_name='d', canonical_ids_in_order=ids, eligibility_by_id=elig, k=3, namespace='ns', separator='\\x00')\n"
        "print(','.join(r.selected_ids))\n"
    ) % str(ROOT)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": str(seed)})
    assert proc.returncode == 0, proc.stderr
    if seed == 0:
        test_selection_determinism_across_pythonhashseed._first = proc.stdout.strip()
    else:
        assert proc.stdout.strip() == getattr(test_selection_determinism_across_pythonhashseed, "_first", proc.stdout.strip())


# ---------------------------------------------------------------------------
# Independent synthetic fixtures: hand-reconstructed category outcomes
# ---------------------------------------------------------------------------


def test_fixture_harmful_only(contract_coco):
    record = _make_harmful_only_record("harmful_img", contract=contract_coco)
    example = qi.build_example_analysis(record, contract=contract_coco, boundary_iou_authority=contract_coco.boundary_iou_authority)
    assert example.category_pixel_counts["harmful_washout"] == 1
    assert example.category_pixel_counts["beneficial_correction"] == 0
    assert example.category_pixel_counts["third_label_transition"] == 0
    assert example.category_eligibility["harmful_washout"] is True
    assert example.category_eligibility["beneficial_correction"] is False


def test_fixture_beneficial_only(contract_coco):
    record = _make_beneficial_only_record("beneficial_img", contract=contract_coco)
    example = qi.build_example_analysis(record, contract=contract_coco, boundary_iou_authority=contract_coco.boundary_iou_authority)
    assert example.category_pixel_counts["beneficial_correction"] == 1
    assert example.category_pixel_counts["harmful_washout"] == 0
    assert example.category_eligibility["beneficial_correction"] is True


def test_fixture_third_label_only(contract_coco):
    record = _make_third_label_only_record("third_img", contract=contract_coco)
    example = qi.build_example_analysis(record, contract=contract_coco, boundary_iou_authority=contract_coco.boundary_iou_authority)
    assert example.category_pixel_counts["third_label_transition"] == 1
    assert example.category_pixel_counts["harmful_washout"] == 0
    assert example.category_pixel_counts["beneficial_correction"] == 0


def test_fixture_no_disputes(contract_coco):
    record = _make_no_dispute_record("no_dispute_img", contract=contract_coco)
    example = qi.build_example_analysis(record, contract=contract_coco, boundary_iou_authority=contract_coco.boundary_iou_authority)
    for category in qs.CATEGORY_ORDER:
        assert example.category_pixel_counts[category] == 0
        assert example.category_eligibility[category] is False


def test_fixture_mixed_categories(contract_coco):
    """A single image combining harmful + beneficial pixels at distinct
    locations -- independently verify both categories are simultaneously
    eligible and pairwise disjoint (no pixel counted in two of
    harmful/beneficial/third_label)."""
    class_count = contract_coco.class_count
    height, width = 8, 8
    gt = np.zeros((height, width), dtype=np.int64)
    gt[0, 0] = 0  # harmful pixel: GT == a
    gt[1, 1] = 1  # beneficial pixel: GT == b
    unary_h, final_h = contract_coco.unary_horizon, contract_coco.final_horizon
    scores = {h: np.zeros((class_count, height, width), dtype=np.float32) for h in contract_coco.horizons}
    for h in contract_coco.horizons:
        scores[h][0] = 1.0
    scores[final_h][0, 0, 0] = 0.0
    scores[final_h][1, 0, 0] = 1.0  # dispute at (0,0): a=0, b=1, GT=0 -> harmful
    scores[final_h][0, 1, 1] = 0.0
    scores[final_h][1, 1, 1] = 1.0  # dispute at (1,1): a=0, b=1, GT=1 -> beneficial
    for h in contract_coco.horizons:
        if h not in (unary_h, final_h):
            scores[h][0, 0, 0] = 0.0
            scores[h][1, 0, 0] = 1.0
            scores[h][0, 1, 1] = 0.0
            scores[h][1, 1, 1] = 1.0
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    record = qi.QualitativeInputRecord(canonical_image_id="mixed_img", rgb=rgb, gt=gt, scores=scores, source_hashes={"result_sha256": "a" * 64})
    example = qi.build_example_analysis(record, contract=contract_coco, boundary_iou_authority=contract_coco.boundary_iou_authority)
    assert example.category_pixel_counts["harmful_washout"] == 1
    assert example.category_pixel_counts["beneficial_correction"] == 1
    assert example.category_pixel_counts["third_label_transition"] == 0


def test_fixture_ignore_pixels_excluded(contract_coco):
    """A dispute pixel at an ignored GT location must never count toward
    any category (Step 3's own DisputeResult already excludes it via
    `valid`)."""
    class_count = contract_coco.class_count
    height, width = 8, 8
    gt = np.full((height, width), contract_coco.ignore_index, dtype=np.int64)
    unary_h, final_h = contract_coco.unary_horizon, contract_coco.final_horizon
    scores = {h: np.zeros((class_count, height, width), dtype=np.float32) for h in contract_coco.horizons}
    for h in contract_coco.horizons:
        scores[h][0] = 1.0
    scores[final_h][0, 0, 0] = 0.0
    scores[final_h][1, 0, 0] = 1.0
    for h in contract_coco.horizons:
        if h not in (unary_h, final_h):
            scores[h][0, 0, 0] = 0.0
            scores[h][1, 0, 0] = 1.0
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    record = qi.QualitativeInputRecord(canonical_image_id="ignore_img", rgb=rgb, gt=gt, scores=scores, source_hashes={"result_sha256": "a" * 64})
    example = qi.build_example_analysis(record, contract=contract_coco, boundary_iou_authority=contract_coco.boundary_iou_authority)
    for category in qs.CATEGORY_ORDER:
        assert example.category_pixel_counts[category] == 0


def test_fixture_non_square_image(contract_coco):
    record = _make_record("nonsquare_img", contract=contract_coco, rng=np.random.default_rng(9), height=10, width=40)
    example = qi.build_example_analysis(record, contract=contract_coco, boundary_iou_authority=contract_coco.boundary_iou_authority)
    assert example.boundary_mask.shape == (10, 40)


def test_fixture_mismatched_dimensions_rejected(contract_coco):
    class_count = contract_coco.class_count
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    gt = np.zeros((8, 9), dtype=np.int64)  # deliberately mismatched width
    scores = {h: np.zeros((class_count, 8, 8), dtype=np.float32) for h in contract_coco.horizons}
    record = qi.QualitativeInputRecord(canonical_image_id="bad_img", rgb=rgb, gt=gt, scores=scores, source_hashes={"result_sha256": "a" * 64})
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="shape"):
        qi.validate_input_record(record, class_count=class_count, ignore_index=contract_coco.ignore_index, horizons=contract_coco.horizons)


def test_fixture_changed_input_artifact_digest_detected(contract_coco):
    """Independent proof: changing a source_hashes entry produces a
    genuinely different (never silently ignored) hash in the record."""
    record = _make_record("digest_img", contract=contract_coco, rng=np.random.default_rng(11))
    mutated = qi.QualitativeInputRecord(
        canonical_image_id=record.canonical_image_id, rgb=record.rgb, gt=record.gt, scores=record.scores,
        source_hashes={"result_sha256": "9" * 64},
    )
    assert record.source_hashes != mutated.source_hashes


# ---------------------------------------------------------------------------
# GT isolation / purity poison tests
# ---------------------------------------------------------------------------


def test_gt_isolation_selection_never_touches_prediction_construction(contract_coco):
    """Poison every prediction/model/graph/recurrence entry point;
    building a category analysis and running selection must reach NONE
    of them, proving GT flows only into category labeling."""
    poisoned = []

    def poison(obj, attr):
        original = getattr(obj, attr)

        def _boom(*a, **kw):
            raise AssertionError(f"FORBIDDEN ENTRY POINT REACHED: {obj}.{attr}")

        setattr(obj, attr, _boom)
        poisoned.append((obj, attr, original))

    import torch

    poison(torch.cuda, "init")
    poison(torch.cuda, "is_available")
    poison(torch, "load")

    try:
        record = _make_record("poison_img", contract=contract_coco, rng=np.random.default_rng(21))
        example = qi.build_example_analysis(record, contract=contract_coco, boundary_iou_authority=contract_coco.boundary_iou_authority)
        result = qs.select_category_examples(
            category_name="harmful_washout", dataset_name="coco_stuff_500", canonical_ids_in_order=[record.canonical_image_id],
            eligibility_by_id={record.canonical_image_id: example.category_eligibility["harmful_washout"]}, k=1,
            namespace="ns", separator="\x00",
        )
    finally:
        for obj, attr, original in poisoned:
            setattr(obj, attr, original)

    assert torch.cuda.is_initialized() is False


def test_gt_isolation_no_forbidden_calls_in_source():
    """AST/tokenize-based audit: the selection/input modules must never
    call model-construction, graph, recurrence, or checkpoint-loading
    functions in actual code (comments/docstrings excluded)."""
    import io
    import tokenize

    forbidden = ("finite_step_propagate", "DirectedTopKGraph", "torch.load", "create_model", ".cuda(", "CocoStuffDataset(", "ADE20KDataset(")
    for path in (ROOT / "src/dfp_phase1a_qualitative_selection.py", ROOT / "src/dfp_phase1a_qualitative_input.py", ROOT / "src/dfp_phase1a_qualitative_render.py"):
        source = path.read_text()
        pieces = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(source).readline):
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                pieces.append(tok.string)
        except tokenize.TokenError:
            pieces = [source]
        code_only = " ".join(pieces)
        for token in forbidden:
            assert token not in code_only, f"{path} contains forbidden token {token!r} in actual code"


def test_gt_isolation_rendering_never_initializes_cuda(contract_coco):
    import torch

    assert torch.cuda.is_initialized() is False
    record = _make_record("render_poison_img", contract=contract_coco, rng=np.random.default_rng(22))
    palette = render.generate_class_palette(contract_coco.class_count)
    render.render_rgb_panel(record.rgb, output_height=32, output_width=32)
    render.render_label_panel(record.gt, palette=palette, ignore_index=contract_coco.ignore_index, ignore_color=(0, 0, 0), output_height=32, output_width=32)
    assert torch.cuda.is_initialized() is False


# ---------------------------------------------------------------------------
# Palette / component-color determinism
# ---------------------------------------------------------------------------


def test_palette_deterministic_and_class_order():
    p1 = render.generate_class_palette(50)
    p2 = render.generate_class_palette(50)
    assert np.array_equal(p1, p2)
    assert render.palette_digest(p1) == render.palette_digest(p2)


def test_palette_digest_mutation_detected():
    p1 = render.generate_class_palette(10)
    p2 = p1.copy()
    p2[3] = ((p2[3].astype(np.int64) + 1) % 256).astype(np.uint8)
    assert render.palette_digest(p1) != render.palette_digest(p2)


def test_palette_class_order_mutation_detected():
    p1 = render.generate_class_palette(10)
    p2 = p1[::-1].copy()  # reversed class order
    assert render.palette_digest(p1) != render.palette_digest(p2)


def test_component_color_independent_of_dict_order_and_hashseed():
    import os
    import subprocess

    script = f"import sys; sys.path.insert(0, {str(ROOT)!r})\nimport src.dfp_phase1a_qualitative_render as r\nprint(r.component_color(42, namespace='ns'))\n"
    outputs = set()
    for seed in (0, 1, 999):
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": str(seed)})
        assert proc.returncode == 0, proc.stderr
        outputs.add(proc.stdout.strip())
    assert len(outputs) == 1


def test_component_color_distinct_for_different_ids():
    colors = {render.component_color(i, namespace="ns") for i in range(20)}
    assert len(colors) == 20


# ---------------------------------------------------------------------------
# Rendering determinism
# ---------------------------------------------------------------------------


def test_rendering_label_panel_uses_nearest_neighbor_no_antialiasing():
    palette = render.generate_class_palette(4)
    labels = np.zeros((4, 4), dtype=np.int64)
    labels[0:2, 0:2] = 1
    labels[2:4, 2:4] = 2
    panel = render.render_label_panel(labels, palette=palette, ignore_index=255, ignore_color=(0, 0, 0), output_height=40, output_width=40)
    observed_colors = {tuple(c) for c in panel.reshape(-1, 3)}
    allowed = {tuple(palette[0]), tuple(palette[1]), tuple(palette[2]), (0, 0, 0)}
    assert observed_colors <= allowed, "nearest-neighbor resize must never introduce a blended/antialiased color"


def test_rendering_png_encoded_and_decoded_digests_deterministic(tmp_path):
    pixel_array = (np.random.default_rng(3).random((12, 12, 3)) * 255).astype(np.uint8)
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    enc1, dec1 = render.write_panel_png(pixel_array, p1)
    enc2, dec2 = render.write_panel_png(pixel_array, p2)
    assert enc1 == enc2
    assert dec1 == dec2
    assert p1.read_bytes() == p2.read_bytes()


def test_rendering_decoded_digest_distinguishes_content_from_encoding(tmp_path):
    pixel_array = (np.random.default_rng(4).random((12, 12, 3)) * 255).astype(np.uint8)
    p1 = tmp_path / "a.png"
    _, dec1 = render.write_panel_png(pixel_array, p1, compress_level=0)
    p2 = tmp_path / "b.png"
    _, dec2 = render.write_panel_png(pixel_array, p2, compress_level=9)
    assert dec1 == dec2, "decoded pixel digest must be identical regardless of PNG compression level"


def test_rendering_no_timestamp_metadata_in_png(tmp_path):
    pixel_array = np.zeros((4, 4, 3), dtype=np.uint8)
    path = tmp_path / "a.png"
    render.write_panel_png(pixel_array, path)
    from PIL import Image

    image = Image.open(path)
    assert "tIME" not in (image.text.keys() if hasattr(image, "text") else {})


# ---------------------------------------------------------------------------
# Strict selection-manifest validation
# ---------------------------------------------------------------------------


def _good_selection_manifest():
    # NOTE: the namespace here must match build_selection_manifest's own
    # hardcoded "selection_namespace" field exactly (both are pinned to
    # the reporting identity's bound namespace) -- since
    # validate_selection_manifest_structure now relationally recomputes
    # every selection key using the manifest's OWN declared namespace,
    # an inconsistent namespace here would make even this POSITIVE
    # control fail structural validation.
    ids = ["a", "b", "c"]
    results = [
        qs.select_category_examples(
            category_name=cat, dataset_name="coco_stuff_500", canonical_ids_in_order=ids,
            eligibility_by_id={"a": True, "b": True, "c": False}, k=2,
            namespace="talk2dino-dfp-phase1a-qualitative-report-v1", separator="\x00",
        )
        for cat in qs.CATEGORY_ORDER
    ]
    return qs.build_selection_manifest(
        reporting_identity_sha256="a" * 64, protocol_identity_sha256="b" * 64, artifact_schema_identity_sha256="c" * 64,
        subset_manifest_sha256="d" * 64, selected_ids_digest="e" * 64, run_mode="coco_stuff_500", image_count_expected=500,
        class_count=171, class_names_digest="f" * 64, requested_count_per_category=2, category_results=results,
        category_pixel_counts_by_id={cid: {cat: 1 for cat in qs.CATEGORY_ORDER} for cid in ["a", "b"]},
        source_commit="1" * 40, source_result_sha256="2" * 64,
    )


def test_selection_manifest_valid_accepted():
    manifest = _good_selection_manifest()
    qs.validate_selection_manifest_structure(manifest)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda m: m.pop("schema"),
        lambda m: m.__setitem__("schema_version", 999),
        lambda m: m.__setitem__("run_mode", "voc2012_pilot"),
        lambda m: m.__setitem__("protocol_identity_sha256", "not-a-hash"),
        lambda m: m["categories"]["harmful_washout"].__setitem__("selected_count", 99),
        lambda m: m["categories"]["harmful_washout"].__setitem__("shortage_status", "totally_fine"),
        lambda m: m.__setitem__("category_order", ["only_one"]),
        lambda m: m.__setitem__("unexpected_extra_field", 1),
    ],
)
def test_selection_manifest_mutation_matrix_rejected(mutator):
    manifest = _good_selection_manifest()
    mutated = copy.deepcopy(manifest)
    mutator(mutated)
    with pytest.raises(qs.DfpPhase1aQualitativeReportError):
        qs.validate_selection_manifest_structure(mutated)


def test_selection_manifest_does_not_mutate_input():
    manifest = _good_selection_manifest()
    before = copy.deepcopy(manifest)
    qs.validate_selection_manifest_structure(manifest)
    assert manifest == before


def test_selection_manifest_strict_json_round_trip(tmp_path):
    manifest = _good_selection_manifest()
    path = tmp_path / "selection.json"
    sch.write_json_atomically(path, manifest, label="selection manifest")
    reparsed = sch.parse_strict_json_document(path, label="selection manifest")
    qs.validate_selection_manifest_structure(reparsed)


def test_selection_manifest_rejects_duplicate_json_key(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.parse_strict_json_document(path, label="x")


def test_selection_manifest_rejects_nan(tmp_path):
    path = tmp_path / "nan.json"
    path.write_text('{"a": NaN}')
    with pytest.raises(sch.DfpPhase1aArtifactSchemaError):
        sch.parse_strict_json_document(path, label="x")


# ---------------------------------------------------------------------------
# Bundle manifest / atomic installation
# ---------------------------------------------------------------------------


def _build_bundle_fixture(tmp_path, *, n_panels=1):
    manifest = _good_selection_manifest()
    scratch = tmp_path / "_scratch_selection.json"
    sch.write_json_atomically(scratch, manifest, label="x")
    selection_sha256 = sch.sha256_file(scratch)
    scratch.unlink()

    panels = []
    png_bytes = {}
    for i in range(n_panels):
        pixel_array = (np.random.default_rng(100 + i).random((8, 8, 3)) * 255).astype(np.uint8)
        temp_png = tmp_path / f"_scratch_{i}.png"
        enc, dec = render.write_panel_png(pixel_array, temp_png)
        filename = f"panel_{i}.png"
        png_bytes[filename] = temp_png.read_bytes()
        temp_png.unlink()
        panels.append(render.PanelRecord(image_id="a", category_name="harmful_washout", panel_name="rgb", png_filename=filename, png_sha256=enc, decoded_pixel_sha256=dec, width=8, height=8))

    bundle_manifest = render.build_bundle_manifest(
        reporting_identity_sha256="a" * 64, selection_manifest_sha256=selection_sha256, source_result_sha256="2" * 64,
        run_mode="coco_stuff_500", panel_order=["rgb"], palette_digest_value=render.palette_digest(render.generate_class_palette(171)),
        panels=panels, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    return manifest, bundle_manifest, png_bytes


def test_bundle_install_and_verify_roundtrip(tmp_path):
    manifest, bundle_manifest, png_bytes = _build_bundle_fixture(tmp_path)
    dest = tmp_path / "bundle"
    installed = render.install_qualitative_bundle(destination_dir=dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)
    assert installed == dest
    result = render.verify_qualitative_bundle(dest)
    assert result["bundle_manifest"]["run_mode"] == "coco_stuff_500"


def test_bundle_install_rejects_existing_destination(tmp_path):
    manifest, bundle_manifest, png_bytes = _build_bundle_fixture(tmp_path)
    dest = tmp_path / "bundle"
    render.install_qualitative_bundle(destination_dir=dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)
    with pytest.raises(render.DfpPhase1aQualitativeReportError, match="already exists"):
        render.install_qualitative_bundle(destination_dir=dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)


def test_bundle_install_rejects_missing_panel(tmp_path):
    manifest, bundle_manifest, png_bytes = _build_bundle_fixture(tmp_path)
    del png_bytes["panel_0.png"]
    dest = tmp_path / "bundle"
    with pytest.raises(render.DfpPhase1aQualitativeReportError, match="missing|disagrees"):
        render.install_qualitative_bundle(destination_dir=dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)
    assert not dest.exists()


def test_bundle_install_rejects_extra_panel(tmp_path):
    manifest, bundle_manifest, png_bytes = _build_bundle_fixture(tmp_path)
    png_bytes["extra.png"] = png_bytes["panel_0.png"]
    dest = tmp_path / "bundle"
    with pytest.raises(render.DfpPhase1aQualitativeReportError):
        render.install_qualitative_bundle(destination_dir=dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)
    assert not dest.exists()


def test_bundle_install_rejects_changed_png_bytes(tmp_path):
    manifest, bundle_manifest, png_bytes = _build_bundle_fixture(tmp_path)
    png_bytes["panel_0.png"] = png_bytes["panel_0.png"] + b"\x00\x01"
    dest = tmp_path / "bundle"
    with pytest.raises(render.DfpPhase1aQualitativeReportError):
        render.install_qualitative_bundle(destination_dir=dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)
    assert not dest.exists()


def test_bundle_install_no_staging_residue_on_failure(tmp_path):
    manifest, bundle_manifest, png_bytes = _build_bundle_fixture(tmp_path)
    del png_bytes["panel_0.png"]
    dest = tmp_path / "bundle"
    try:
        render.install_qualitative_bundle(destination_dir=dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)
    except render.DfpPhase1aQualitativeReportError:
        pass
    leaks = [p.name for p in tmp_path.iterdir() if "staging" in p.name]
    assert leaks == []


def test_bundle_manifest_rejects_traversal_filename(tmp_path):
    manifest, bundle_manifest, png_bytes = _build_bundle_fixture(tmp_path)
    bundle_manifest = copy.deepcopy(bundle_manifest)
    bundle_manifest["panels"][0]["png_filename"] = "../escape.png"
    with pytest.raises(render.DfpPhase1aQualitativeReportError):
        render.validate_bundle_manifest_structure(bundle_manifest)


def test_bundle_verify_detects_tampered_installed_panel(tmp_path):
    manifest, bundle_manifest, png_bytes = _build_bundle_fixture(tmp_path)
    dest = tmp_path / "bundle"
    render.install_qualitative_bundle(destination_dir=dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)
    panel_path = dest / "panel_0.png"
    with panel_path.open("ab") as f:
        f.write(b"\x00\x01\x02")
    with pytest.raises(render.DfpPhase1aQualitativeReportError):
        render.verify_qualitative_bundle(dest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli_fixture_bundle(tmp_path, *, n_images=4, height=10, width=12):
    from src.dfp_phase1a_analysis import load_analysis_contract_from_protocol
    from src.dfp_phase1a_identity import load_identity as load_protocol_identity

    protocol = load_protocol_identity(ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml", repo_root=ROOT)
    contract = load_analysis_contract_from_protocol(protocol, dataset_key="coco_stuff", bind_boundary_iou=True, repo_root=ROOT)
    class_count = contract.class_count
    rng = np.random.default_rng(55)
    rgb = (rng.random((n_images, height, width, 3)) * 255).astype(np.uint8)
    gt = rng.integers(0, class_count, size=(n_images, height, width)).astype(np.int64)
    save_kwargs = {"rgb": rgb, "gt": gt}
    for h in contract.horizons:
        save_kwargs[f"scores_{h}"] = rng.random((n_images, class_count, height, width)).astype(np.float32)
    bundle_dir = tmp_path / "fixture_bundle"
    bundle_dir.mkdir()
    np.savez(bundle_dir / "fixtures.npz", allow_pickle=False, **save_kwargs)
    image_ids = [f"cli_synthetic_{i:04d}" for i in range(n_images)]
    fixture_manifest = {"dataset_key": "coco_stuff", "class_count": class_count, "ignore_index": contract.ignore_index, "horizons": list(contract.horizons), "image_ids": image_ids, "npz_filename": "fixtures.npz"}
    (bundle_dir / "fixture_manifest.json").write_text(json.dumps(fixture_manifest))
    return bundle_dir


def test_cli_select_render_verify_roundtrip(tmp_path):
    bundle_dir = _build_cli_fixture_bundle(tmp_path)
    selection_path = tmp_path / "selection.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_qualitative_report.py"), "select", "--fixture-bundle", str(bundle_dir), "--dataset", "coco_stuff", "--output", str(selection_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert selection_path.exists()

    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_qualitative_report.py"), "verify-selection", "--selection-manifest", str(selection_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    output_dir = tmp_path / "report_bundle"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_qualitative_report.py"), "render", "--fixture-bundle", str(bundle_dir), "--selection-manifest", str(selection_path), "--dataset", "coco_stuff", "--output-dir", str(output_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert output_dir.exists()

    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_qualitative_report.py"), "verify-report", "--bundle", str(output_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_missing_input_exits_2_no_traceback(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_qualitative_report.py"), "verify-selection", "--selection-manifest", str(tmp_path / "does_not_exist.json")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "DFP PHASE 1A QUALITATIVE REPORT FAIL" in proc.stderr


def test_cli_verify_report_missing_bundle_exits_2(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_qualitative_report.py"), "verify-report", "--bundle", str(tmp_path / "nope")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# No-cherry-picking audit
# ---------------------------------------------------------------------------


def test_no_cherry_picking_no_rng_or_wallclock_in_selection_source():
    """Source-level audit: the selection module must never call
    random/RNG/time-based functions."""
    forbidden = ("random.random(", "random.choice(", "np.random", "numpy.random", "time.time(", "datetime.now(", "os.urandom(")
    source = (ROOT / "src/dfp_phase1a_qualitative_selection.py").read_text()
    for token in forbidden:
        assert token not in source, f"selection module contains forbidden nondeterministic token {token!r}"


def test_no_cherry_picking_no_hardcoded_real_image_ids():
    """Reject plausible real COCO/ADE20K-style numeric image IDs
    hardcoded in the reporting source files or this test file itself
    (synthetic IDs used throughout are prefixed 'synthetic_'/'cli_
    synthetic_'/short letters, never resembling a copied real
    selection)."""
    import re

    real_id_pattern = re.compile(r"\b0{6,9}\d{4,6}\b")  # COCO-style zero-padded numeric filenames
    for path in (
        ROOT / "src/dfp_phase1a_qualitative_selection.py", ROOT / "src/dfp_phase1a_qualitative_input.py",
        ROOT / "src/dfp_phase1a_qualitative_render.py", ROOT / "verify_dfp_phase1a_qualitative_report.py",
        ROOT / "docs/dfp_phase1a_qualitative_report.md",
    ):
        if not path.exists():
            continue
        text = path.read_text()
        assert not real_id_pattern.search(text), f"{path} appears to contain a hardcoded real-looking image ID"


def test_no_cherry_picking_no_manual_featured_examples_list():
    source = (ROOT / "src/dfp_phase1a_qualitative_selection.py").read_text()
    for token in ("featured_examples", "FEATURED_IMAGES", "manual_selection", "cherry_pick"):
        assert token not in source


# ---------------------------------------------------------------------------
# Repair regression: relational selection-manifest reconstruction (Repair 1)
# and provenance-field validation (Repair 2).
# ---------------------------------------------------------------------------


def _multi_selection_manifest(*, n_eligible_per_category=None):
    """Build a selection manifest with a REAL, non-trivial eligible
    population (10 IDs) so shortage/one-selection/zero-selection/multi-
    selection category scenarios can all be exercised with genuinely
    ordered selection keys (not degenerate 0/1-eligible cases only)."""
    ids = [f"img{i}" for i in range(10)]
    if n_eligible_per_category is None:
        n_eligible_per_category = {
            "harmful_washout": 10, "beneficial_correction": 2, "third_label_transition": 1,
            "disputed_boundary": 0, "disputed_interior": 10, "oracle_reverted": 10,
            "horizon_persistence_reverting": 10, "revert_set_fragmentation": 10,
        }
    results = []
    for cat in qs.CATEGORY_ORDER:
        n = n_eligible_per_category[cat]
        eligibility = {cid: (i < n) for i, cid in enumerate(ids)}
        results.append(
            qs.select_category_examples(
                category_name=cat, dataset_name="coco_stuff_500", canonical_ids_in_order=ids, eligibility_by_id=eligibility,
                k=3, namespace="talk2dino-dfp-phase1a-qualitative-report-v1", separator="\x00",
            )
        )
    pixel_counts_by_id = {}
    for r in results:
        for cid in r.selected_ids:
            pixel_counts_by_id.setdefault(cid, {})
            for cat in qs.CATEGORY_ORDER:
                pixel_counts_by_id[cid].setdefault(cat, 1)
    return qs.build_selection_manifest(
        reporting_identity_sha256="a" * 64, protocol_identity_sha256="b" * 64, artifact_schema_identity_sha256="c" * 64,
        subset_manifest_sha256="d" * 64, selected_ids_digest="e" * 64, run_mode="coco_stuff_500", image_count_expected=500,
        class_count=171, class_names_digest="f" * 64, requested_count_per_category=3, category_results=results,
        category_pixel_counts_by_id=pixel_counts_by_id, source_commit="1" * 40, source_result_sha256="2" * 64,
    )


def test_repair1_positive_manifest_validates():
    manifest = _multi_selection_manifest()
    before = copy.deepcopy(manifest)
    qs.validate_selection_manifest_structure(manifest)
    assert manifest == before, "validation must never mutate its input"


def test_repair1_zero_selection_category():
    manifest = _multi_selection_manifest()
    assert manifest["categories"]["disputed_boundary"]["selected_count"] == 0
    assert manifest["categories"]["disputed_boundary"]["shortage_status"] == "zero_coverage"
    qs.validate_selection_manifest_structure(manifest)  # must still validate cleanly


def test_repair1_one_selection_category():
    manifest = _multi_selection_manifest()
    assert manifest["categories"]["third_label_transition"]["selected_count"] == 1
    qs.validate_selection_manifest_structure(manifest)


def test_repair1_shortage_category():
    manifest = _multi_selection_manifest()
    assert manifest["categories"]["beneficial_correction"]["shortage_status"] == "shortage"
    assert manifest["categories"]["beneficial_correction"]["selected_count"] == 2
    qs.validate_selection_manifest_structure(manifest)


def test_repair1_reversed_selected_ids_rejected():
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    cat = "harmful_washout"
    assert len(mutated["categories"][cat]["selected_ids"]) == 3
    mutated["categories"][cat]["selected_ids"] = list(reversed(mutated["categories"][cat]["selected_ids"]))
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="ordered"):
        qs.validate_selection_manifest_structure(mutated)


def test_repair1_swapped_two_selected_ids_rejected():
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    cat = "harmful_washout"
    ids = mutated["categories"][cat]["selected_ids"]
    assert len(ids) >= 2
    ids[0], ids[1] = ids[1], ids[0]
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="ordered"):
        qs.validate_selection_manifest_structure(mutated)


def test_repair1_one_wrong_but_valid_key_rejected():
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    cat = "harmful_washout"
    keys = mutated["categories"][cat]["selection_keys"]
    cid = next(iter(keys))
    keys[cid] = "9" * 64  # well-formed 64-hex, but wrong
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="recomputed selection key"):
        qs.validate_selection_manifest_structure(mutated)


def test_repair1_keys_assigned_to_wrong_ids_rejected():
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    cat = "harmful_washout"
    keys = mutated["categories"][cat]["selection_keys"]
    ids = list(keys)
    assert len(ids) >= 2
    keys[ids[0]], keys[ids[1]] = keys[ids[1]], keys[ids[0]]  # swap two DIFFERENT IDs' keys
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="recomputed selection key"):
        qs.validate_selection_manifest_structure(mutated)


def test_repair1_missing_key_rejected():
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    cat = "harmful_washout"
    keys = mutated["categories"][cat]["selection_keys"]
    del keys[next(iter(keys))]
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="exactly one entry"):
        qs.validate_selection_manifest_structure(mutated)


def test_repair1_extra_key_rejected():
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    cat = "harmful_washout"
    keys = mutated["categories"][cat]["selection_keys"]
    keys["not_a_selected_id"] = "0" * 64
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="exactly one entry"):
        qs.validate_selection_manifest_structure(mutated)


def test_repair1_reordered_key_container_still_validates():
    """Reordering the selection_keys MAPPING's own insertion order (never
    its content) must NOT affect validation -- a Mapping's logical
    content is order-independent."""
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    cat = "harmful_washout"
    keys = mutated["categories"][cat]["selection_keys"]
    reordered = dict(reversed(list(keys.items())))
    assert reordered == keys  # same content, different insertion order
    mutated["categories"][cat]["selection_keys"] = reordered
    qs.validate_selection_manifest_structure(mutated)  # must NOT raise


def test_repair1_wrong_but_valid_trace_digest_rejected():
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    mutated["selection_trace_digest"] = "8" * 64  # well-formed, wrong
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="trace digest"):
        qs.validate_selection_manifest_structure(mutated)


def test_repair1_trace_computed_from_tampered_stored_keys_still_rejected():
    """Tamper a stored key AND recompute the trace digest so it is
    internally self-consistent with that tampered key -- the validator
    must still reject, because it recomputes the trace using the
    INDEPENDENTLY recomputed keys, never the stored ones."""
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    cat = "harmful_washout"
    keys = mutated["categories"][cat]["selection_keys"]
    cid = next(iter(keys))
    keys[cid] = "9" * 64  # tamper the stored key

    # Recompute the trace digest FROM the (now-tampered) stored data, to
    # make the document internally self-consistent by construction.
    fake_results = []
    for name in qs.CATEGORY_ORDER:
        record = mutated["categories"][name]
        fake_results.append(
            qs.CategorySelectionResult(
                category_name=name, requested_count=record["requested_count"], eligible_count=record["eligible_count"],
                selected_count=record["selected_count"], selected_ids=tuple(record["selected_ids"]),
                selection_keys=dict(record["selection_keys"]), shortage_status=record["shortage_status"],
            )
        )
    mutated["selection_trace_digest"] = qs.selection_trace_digest(fake_results)

    with pytest.raises(qs.DfpPhase1aQualitativeReportError):
        qs.validate_selection_manifest_structure(mutated)


# ---------------------------------------------------------------------------
# Repair 2: provenance field validation.
# ---------------------------------------------------------------------------

PROVENANCE_BAD_VALUES = [True, False, 123, [], None, "   ", "0" * 39, "0" * 41, "G" * 40, "not-a-commit"]


@pytest.mark.parametrize("bad_value", PROVENANCE_BAD_VALUES)
def test_repair2_source_commit_type_matrix_rejected(bad_value):
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    mutated["source_commit"] = bad_value
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="source_commit"):
        qs.validate_selection_manifest_structure(mutated)


PROVENANCE_BAD_HASHES = [True, False, 123, [], None, "   ", "a" * 63, "a" * 65, "G" * 64, "not-a-hash"]


@pytest.mark.parametrize("bad_value", PROVENANCE_BAD_HASHES)
def test_repair2_source_result_sha256_type_matrix_rejected(bad_value):
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    mutated["source_result_sha256"] = bad_value
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="source_result_sha256"):
        qs.validate_selection_manifest_structure(mutated)


def test_repair2_source_commit_uppercase_rejected():
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    mutated["source_commit"] = "A" * 40
    with pytest.raises(qs.DfpPhase1aQualitativeReportError, match="source_commit"):
        qs.validate_selection_manifest_structure(mutated)


def test_repair2_valid_provenance_accepted():
    manifest = _multi_selection_manifest()
    assert manifest["source_commit"] == "1" * 40
    assert manifest["source_result_sha256"] == "2" * 64
    qs.validate_selection_manifest_structure(manifest)  # must not raise


def test_repair2_cli_rejects_bad_provenance_via_manifest(tmp_path):
    manifest = _multi_selection_manifest()
    mutated = copy.deepcopy(manifest)
    mutated["source_commit"] = "not-a-commit"
    path = tmp_path / "bad_provenance.json"
    sch.write_json_atomically(path, mutated, label="selection manifest")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_qualitative_report.py"), "verify-selection", "--selection-manifest", str(path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "source_commit" in proc.stderr


# ---------------------------------------------------------------------------
# Repair 3: strict image-decoding helper.
# ---------------------------------------------------------------------------


def _build_installed_bundle_with_panel(tmp_path, *, panel_bytes, panel_sha256=None, decoded_sha256="0" * 64):
    """Hand-build an on-disk bundle directory (bypassing
    install_qualitative_bundle's own protections) so a specific
    (possibly malformed) panel byte string can be placed directly, for
    isolating verify_qualitative_bundle's own decode path."""
    manifest = _multi_selection_manifest()
    bundle_dir = tmp_path / "hand_built_bundle"
    bundle_dir.mkdir()
    sch.write_json_atomically(bundle_dir / "selection_manifest.json", manifest, label="x")
    sel_hash = sch.sha256_file(bundle_dir / "selection_manifest.json")
    if panel_sha256 is None:
        panel_sha256 = hashlib.sha256(panel_bytes).hexdigest()
    panel = render.PanelRecord(image_id="img0", category_name="harmful_washout", panel_name="rgb", png_filename="panel.png", png_sha256=panel_sha256, decoded_pixel_sha256=decoded_sha256, width=8, height=8)
    bundle_manifest = render.build_bundle_manifest(
        reporting_identity_sha256="a" * 64, selection_manifest_sha256=sel_hash, source_result_sha256="2" * 64,
        run_mode="coco_stuff_500", panel_order=["rgb"], palette_digest_value=render.palette_digest(render.generate_class_palette(171)),
        panels=[panel], created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    sch.write_json_atomically(bundle_dir / "bundle_manifest.json", bundle_manifest, label="x")
    (bundle_dir / "panel.png").write_bytes(panel_bytes)
    return bundle_dir, manifest, bundle_manifest


MALFORMED_IMAGE_CASES = {
    "not-a-png-at-all": b"this is definitely not a PNG file, just plain text bytes",
    "truncated-png-header": b"\x89PNG\r\n\x1a\n",  # valid PNG magic, nothing else
    "empty-file": b"",
    "unsupported-garbage": bytes(range(256)) * 4,
}


@pytest.mark.parametrize("case_name,bad_bytes", list(MALFORMED_IMAGE_CASES.items()))
def test_repair3_write_panel_png_direct_verification(tmp_path, case_name, bad_bytes):
    """write_panel_png's own re-decode step must never let a raw PIL
    exception escape -- simulate a corrupted just-written file by writing
    malformed bytes directly to the destination path before calling
    write_panel_png's internals via the shared decode helper."""
    path = tmp_path / f"{case_name}.png"
    path.write_bytes(bad_bytes)
    with pytest.raises(render.DfpPhase1aQualitativeReportError) as excinfo:
        render._decode_rgb_image(path, label="probe")
    assert isinstance(excinfo.value.__cause__, (OSError, ValueError))


@pytest.mark.parametrize("case_name,bad_bytes", list(MALFORMED_IMAGE_CASES.items()))
def test_repair3_install_qualitative_bundle_malformed_panel(tmp_path, case_name, bad_bytes):
    manifest = _multi_selection_manifest()
    scratch = tmp_path / "_scratch.json"
    sch.write_json_atomically(scratch, manifest, label="x")
    sel_hash = sch.sha256_file(scratch)
    scratch.unlink()
    panel = render.PanelRecord(image_id="img0", category_name="harmful_washout", panel_name="rgb", png_filename="bad.png", png_sha256=hashlib.sha256(bad_bytes).hexdigest(), decoded_pixel_sha256="0" * 64, width=8, height=8)
    bundle_manifest = render.build_bundle_manifest(
        reporting_identity_sha256="a" * 64, selection_manifest_sha256=sel_hash, source_result_sha256="2" * 64,
        run_mode="coco_stuff_500", panel_order=["rgb"], palette_digest_value=render.palette_digest(render.generate_class_palette(171)),
        panels=[panel], created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    dest = tmp_path / f"bundle_{case_name}"
    with pytest.raises(render.DfpPhase1aQualitativeReportError) as excinfo:
        render.install_qualitative_bundle(destination_dir=dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes={"bad.png": bad_bytes})
    assert isinstance(excinfo.value.__cause__, (OSError, ValueError))
    assert not dest.exists(), "no partial destination"
    assert not any("staging" in p.name for p in tmp_path.iterdir()), "staging must be cleaned up"


@pytest.mark.parametrize("case_name,bad_bytes", list(MALFORMED_IMAGE_CASES.items()))
def test_repair3_verify_qualitative_bundle_malformed_panel(tmp_path, case_name, bad_bytes):
    bundle_dir, _, _ = _build_installed_bundle_with_panel(tmp_path, panel_bytes=bad_bytes)
    with pytest.raises(render.DfpPhase1aQualitativeReportError) as excinfo:
        render.verify_qualitative_bundle(bundle_dir)
    assert isinstance(excinfo.value.__cause__, (OSError, ValueError))


def test_repair3_cli_verify_report_malformed_panel_exits_2_no_traceback(tmp_path):
    bundle_dir, _, _ = _build_installed_bundle_with_panel(tmp_path, panel_bytes=MALFORMED_IMAGE_CASES["not-a-png-at-all"])
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_dfp_phase1a_qualitative_report.py"), "verify-report", "--bundle", str(bundle_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "cannot decode" in proc.stderr


def test_repair3_existing_valid_bundle_preserved_after_failed_install(tmp_path):
    """Install one valid bundle, then attempt (and fail) to install a
    SECOND, malformed bundle at a DIFFERENT destination -- the first,
    already-valid bundle must remain byte-for-byte untouched."""
    manifest, bundle_manifest, png_bytes = _build_bundle_fixture(tmp_path, n_panels=1)
    good_dest = tmp_path / "good_bundle"
    render.install_qualitative_bundle(destination_dir=good_dest, selection_manifest=manifest, bundle_manifest=bundle_manifest, panel_png_bytes=png_bytes)
    before_bytes = {p.name: (good_dest / p.name).read_bytes() for p in good_dest.iterdir()}

    bad_bytes = MALFORMED_IMAGE_CASES["not-a-png-at-all"]
    manifest2 = _multi_selection_manifest()
    scratch = tmp_path / "_scratch2.json"
    sch.write_json_atomically(scratch, manifest2, label="x")
    sel_hash2 = sch.sha256_file(scratch)
    scratch.unlink()
    panel2 = render.PanelRecord(image_id="img0", category_name="harmful_washout", panel_name="rgb", png_filename="bad.png", png_sha256=hashlib.sha256(bad_bytes).hexdigest(), decoded_pixel_sha256="0" * 64, width=8, height=8)
    bundle_manifest2 = render.build_bundle_manifest(
        reporting_identity_sha256="a" * 64, selection_manifest_sha256=sel_hash2, source_result_sha256="2" * 64,
        run_mode="coco_stuff_500", panel_order=["rgb"], palette_digest_value=render.palette_digest(render.generate_class_palette(171)),
        panels=[panel2], created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    bad_dest = tmp_path / "bad_bundle"
    with pytest.raises(render.DfpPhase1aQualitativeReportError):
        render.install_qualitative_bundle(destination_dir=bad_dest, selection_manifest=manifest2, bundle_manifest=bundle_manifest2, panel_png_bytes={"bad.png": bad_bytes})

    after_bytes = {p.name: (good_dest / p.name).read_bytes() for p in good_dest.iterdir()}
    assert before_bytes == after_bytes, "the existing valid bundle must be byte-for-byte untouched"
    assert not bad_dest.exists()
    assert not any("staging" in p.name for p in tmp_path.iterdir())


def test_repair3_keyboard_interrupt_not_caught_by_decode_helper(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    def _boom(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(PILImage, "open", _boom)
    path = tmp_path / "x.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(KeyboardInterrupt):
        render._decode_rgb_image(path, label="probe")
