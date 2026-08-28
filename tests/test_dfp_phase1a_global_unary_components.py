"""Tests for src/dfp_phase1a_global_unary_components.py: the DFP Phase
1A Step 6 global unary component partition. CPU-only, no CUDA, no
dataset/model access, no ground truth anywhere. Independently
reconstructs expected components via a pure-Python BFS flood fill in
this file (never importing the new module's own labeling helper for
that part) to prove it adds no undocumented library-numbering
authority.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

import src.dfp_phase1a_global_unary_components as gc  # noqa: E402
from src.dfp_phase1a_identity import load_identity  # noqa: E402

PROTOCOL_PATH = ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml"
TINY_CONTRACT = gc.DfpComponentContract(connectivity=8, patch_pitch=(2, 2), microcomponent_threshold_patch_equivalents=4.0)


def _real_protocol():
    return load_identity(PROTOCOL_PATH, repo_root=ROOT)


def _s0_from_labels(labels: np.ndarray, class_count: int, *, dtype=np.float32, margin=5.0) -> np.ndarray:
    """Build an S0 field whose lowest-index argmax is exactly `labels`:
    for each pixel, the target class gets score `margin`, everything
    else stays 0 -- no ties unless explicitly constructed by the
    caller."""
    h, w = labels.shape
    s0 = np.zeros((class_count, h, w), dtype=dtype)
    for c in range(class_count):
        s0[c][labels == c] = margin
    return s0


def _reference_components(unary_labels: np.ndarray):
    """Pure-Python 8-connected BFS flood fill, independent of scipy and
    of the module under test. Returns a list of (first_flat_index,
    class_index, pixel_list) sorted by ascending first_flat_index --
    matching the module's own canonical global-id order."""
    h, w = unary_labels.shape
    visited = [[False] * w for _ in range(h)]
    comps = []
    for i in range(h):
        for j in range(w):
            if visited[i][j]:
                continue
            c = int(unary_labels[i, j])
            stack = [(i, j)]
            visited[i][j] = True
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny][nx] and int(unary_labels[ny, nx]) == c:
                            visited[ny][nx] = True
                            stack.append((ny, nx))
            first_flat = min(y * w + x for y, x in pixels)
            comps.append((first_flat, c, pixels))
    comps.sort(key=lambda t: t[0])
    return comps


def _assert_matches_reference(result: gc.GlobalUnaryComponents, unary_labels: np.ndarray):
    ref = _reference_components(unary_labels)
    assert result.component_count == len(ref)
    for gid, (first_flat, c, pixels) in enumerate(ref):
        assert int(result.component_first_flat_indices[gid]) == first_flat
        assert int(result.component_class_indices[gid]) == c
        assert int(result.component_pixel_areas[gid]) == len(pixels)
        for (y, x) in pixels:
            assert int(result.component_ids[y, x]) == gid


# ---------------------------------------------------------------------------
# Contract extraction
# ---------------------------------------------------------------------------


def test_contract_from_real_protocol():
    protocol = _real_protocol()
    contract = gc.dfp_component_contract_from_protocol(protocol, repo_root=ROOT)
    assert contract.connectivity == 8
    assert contract.patch_pitch == (14, 14)
    assert contract.microcomponent_threshold_patch_equivalents == 4.0


def test_contract_does_not_hardcode_pinned_values():
    import ast

    source = Path(gc.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc_node = node.body[0] if node.body else None
            if isinstance(doc_node, ast.Expr) and isinstance(doc_node.value, ast.Constant) and isinstance(doc_node.value.value, str):
                docstring_lines.update(range(doc_node.lineno, (doc_node.end_lineno or doc_node.lineno) + 1))
    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in docstring_lines or line.strip().startswith("#"):
            continue
        assert "14" not in line or "patch_pitch" in line or "contract.patch_pitch" in line, f"suspicious literal 14 at line {lineno}: {line!r}"
        assert "4.0" not in line, f"found a live threshold literal at line {lineno}: {line!r}"


@pytest.mark.parametrize(
    "section,mutate",
    [
        ("structural_components", lambda p: p.__setitem__("primary_connectivity", 4)),
        ("structural_components", lambda p: p.__setitem__("tie_rule", "highest_class_index")),
        ("structural_components", lambda p: p.__setitem__("component_scope", "per_sliding_window")),
        ("structural_components", lambda p: p.__setitem__("component_source", "raw_per_window_scores")),
        ("structural_component_area", lambda p: p.__setitem__("primary_microcomponent_threshold_patch_equivalents", 8.0)),
        ("structural_component_area", lambda p: p.__setitem__("formula", "different formula")),
    ],
)
def test_altered_component_field_rejected(section, mutate):
    protocol = copy.deepcopy(_real_protocol())
    protocol[section] = dict(protocol[section])
    mutate(protocol[section])
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.dfp_component_contract_from_protocol(protocol, repo_root=ROOT)


def test_altered_connectivity_wrong_type_rejected():
    protocol = copy.deepcopy(_real_protocol())
    protocol["structural_components"] = dict(protocol["structural_components"])
    protocol["structural_components"]["primary_connectivity"] = True  # bool-as-int
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.dfp_component_contract_from_protocol(protocol, repo_root=ROOT)


def test_hand_built_contract_with_connectivity_4_rejected_at_compute_time():
    bad_contract = gc.DfpComponentContract(connectivity=4, patch_pitch=(2, 2), microcomponent_threshold_patch_equivalents=4.0)
    s0 = np.zeros((2, 4, 4), dtype=np.float32)
    s0[0] = 1.0
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError, match="connectivity"):
        gc.compute_global_unary_components(s0, contract=bad_contract, class_count=2)


# ---------------------------------------------------------------------------
# Dependent stitching-contract failures are translated to the Step 6
# domain error, not left as a raw DfpPhase1aHorizonStitchingError
# ---------------------------------------------------------------------------


def _mutated_stitching_field(mutate):
    protocol = copy.deepcopy(_real_protocol())
    protocol["score_interpolation_stitching"] = dict(protocol["score_interpolation_stitching"])
    mutate(protocol["score_interpolation_stitching"])
    return protocol


def _assert_translated_to_domain_error(name, protocol, *, match):
    from scipy import ndimage as _ndimage
    from src.dfp_phase1a_horizon_stitching import DfpPhase1aHorizonStitchingError

    before = copy.deepcopy(protocol)
    label_calls = {"n": 0}
    real_label = _ndimage.label

    def _counting(*a, **kw):
        label_calls["n"] += 1
        return real_label(*a, **kw)

    _ndimage.label = _counting
    try:
        with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError, match=match) as excinfo:
            gc.dfp_component_contract_from_protocol(protocol, repo_root=ROOT)
    finally:
        _ndimage.label = real_label

    assert type(excinfo.value) is gc.DfpPhase1aGlobalUnaryComponentsError, (
        f"[{name}] exact outer exception type must be DfpPhase1aGlobalUnaryComponentsError, "
        f"observed {type(excinfo.value).__name__}"
    )
    assert isinstance(excinfo.value.__cause__, DfpPhase1aHorizonStitchingError), (
        f"[{name}] __cause__ must be DfpPhase1aHorizonStitchingError, observed {type(excinfo.value.__cause__)!r}"
    )
    assert label_calls["n"] == 0, f"[{name}] ndimage.label must not be called ({label_calls['n']} calls observed)"
    assert protocol == before, f"[{name}] input protocol mapping must remain unchanged"


def test_crop_mutation_translated_to_domain_error():
    protocol = _mutated_stitching_field(lambda p: p.__setitem__("crop", [224, 224]))
    _assert_translated_to_domain_error("crop", protocol, match="crop")


def test_stride_mutation_translated_to_domain_error():
    protocol = _mutated_stitching_field(lambda p: p.__setitem__("stride", [112, 112]))
    _assert_translated_to_domain_error("stride", protocol, match="stride")


def test_interpolation_mode_mutation_translated_to_domain_error():
    # this repository's stitching contract does not carry a separate
    # "interpolation_mode" string field -- align_corners is the
    # identity-bound interpolation-behavior switch that actually varies
    # the interpolation. window_enumeration is the closest analogue to
    # an "interpolation/geometry mode" string; both are exercised below.
    protocol = _mutated_stitching_field(lambda p: p.__setitem__("window_enumeration", "a different enumeration order"))
    _assert_translated_to_domain_error("window_enumeration", protocol, match="window_enumeration")


def test_alignment_policy_mutation_translated_to_domain_error():
    protocol = _mutated_stitching_field(lambda p: p.__setitem__("align_corners", False))
    _assert_translated_to_domain_error("align_corners", protocol, match="align_corners")


def test_stitching_weight_policy_mutation_translated_to_domain_error():
    protocol = _mutated_stitching_field(lambda p: p.__setitem__("averaging", "hann"))
    _assert_translated_to_domain_error("averaging", protocol, match="averaging")


def test_malformed_exact_type_in_stitching_field_translated_to_domain_error():
    protocol = _mutated_stitching_field(lambda p: p.__setitem__("align_corners", 1))  # int-as-bool
    _assert_translated_to_domain_error("align_corners (int-as-bool)", protocol, match="align_corners")


def test_valid_protocol_still_constructs_identical_component_contract():
    protocol = _real_protocol()
    contract = gc.dfp_component_contract_from_protocol(protocol, repo_root=ROOT)
    assert contract == gc.DfpComponentContract(connectivity=8, patch_pitch=(14, 14), microcomponent_threshold_patch_equivalents=4.0)


def test_unary_and_component_outputs_byte_identical_after_repair():
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[2:5, 2:5] = 1
    labels[5:8, 5:8] = 2
    s0 = _s0_from_labels(labels, 3)
    result = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=3)
    assert np.array_equal(result.unary_labels, labels)
    _assert_matches_reference(result, result.unary_labels)


def test_direct_step6_validation_errors_still_use_own_domain_error():
    # a genuinely Step-6-local violation (not a dependent stitching-contract
    # failure) must still raise gc.DfpPhase1aGlobalUnaryComponentsError with
    # no __cause__ chained from a foreign module.
    s0 = np.zeros((2, 4), dtype=np.float32)  # wrong rank
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError) as excinfo:
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize("exc_cls", [KeyboardInterrupt, SystemExit, MemoryError])
def test_exception_propagates_through_contract_builder(exc_cls, monkeypatch):
    import src.dfp_phase1a_global_unary_components as gc_module

    def _boom(*a, **kw):
        raise exc_cls("poisoned")

    monkeypatch.setattr(gc_module, "dfp_stitching_contract_from_protocol", _boom)
    with pytest.raises(exc_cls):
        gc.dfp_component_contract_from_protocol(_real_protocol(), repo_root=ROOT)


# ---------------------------------------------------------------------------
# Independent fixtures (section 12)
# ---------------------------------------------------------------------------


def _run(labels, class_count, dtype=np.float32):
    s0 = _s0_from_labels(labels, class_count, dtype=dtype)
    result = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=class_count)
    _assert_matches_reference(result, result.unary_labels)
    gc.validate_global_unary_components_partition(result, image_shape=labels.shape, class_count=class_count, contract=TINY_CONTRACT)
    return result


def test_one_pixel_image():
    labels = np.array([[0]], dtype=np.int64)
    result = _run(labels, 1)
    assert result.component_count == 1
    assert result.component_pixel_areas[0] == 1


def test_one_class_full_image_component():
    labels = np.zeros((5, 7), dtype=np.int64)
    result = _run(labels, 1)
    assert result.component_count == 1
    assert result.component_pixel_areas[0] == 35


def test_two_horizontal_components():
    labels = np.zeros((4, 6), dtype=np.int64)
    labels[:, 3:] = 1
    result = _run(labels, 2)
    assert result.component_count == 2


def test_two_vertical_components():
    labels = np.zeros((6, 4), dtype=np.int64)
    labels[3:, :] = 1
    result = _run(labels, 2)
    assert result.component_count == 2


def test_diagonal_pixels_merge_under_8_connectivity():
    labels = np.array([[0, 1], [1, 0]], dtype=np.int64)
    result = _run(labels, 2)
    # under 8-connectivity, the two diagonal 0s merge into one component,
    # and the two diagonal 1s merge into another -> exactly 2 components
    assert result.component_count == 2


def test_fixture_differs_under_4_connectivity():
    # same fixture as above -- explicitly confirm what 4-connectivity
    # WOULD give (4 singleton components), to document the contrast,
    # using the independent scipy call directly (not this module).
    from scipy import ndimage
    labels = np.array([[0, 1], [1, 0]], dtype=np.int64)
    struct4 = ndimage.generate_binary_structure(2, 1)
    _, n4 = ndimage.label(labels == 0, structure=struct4)
    assert n4 == 2  # the two 0-pixels are NOT 4-connected -> 2 separate components for class 0 alone
    result = _run(labels, 2)
    assert result.component_count == 2  # but under 8-connectivity (this module), they merge


def test_checkerboard_labels():
    labels = np.indices((4, 4)).sum(axis=0) % 2
    result = _run(labels.astype(np.int64), 2)
    # every same-parity pixel is diagonally 8-connected to its neighbors -> exactly 2 components
    assert result.component_count == 2


def test_component_with_a_hole():
    labels = np.ones((5, 5), dtype=np.int64)
    labels[2, 2] = 0  # a single-pixel hole of a different class in the middle
    result = _run(labels, 2)
    assert result.component_count == 2
    hole_component = result.component_ids[2, 2]
    assert result.component_pixel_areas[hole_component] == 1


def test_ring_around_another_class():
    labels = np.zeros((7, 7), dtype=np.int64)
    labels[2:5, 2:5] = 1  # inner 3x3 block of class 1
    labels[3, 3] = 0  # single center pixel back to class 0 (a "ring" of class 1 around a class-0 dot)
    result = _run(labels, 2)
    # class 0: outer region + center dot -- are they 8-connected to each other through the ring?
    # the ring is class 1, fully surrounding the center on all 8 sides, so the outer 0-region and
    # the center 0-pixel are NOT 8-connected to each other -> 2 separate class-0 components + 1 class-1 ring
    assert result.component_count == 3


def test_two_disconnected_regions_same_class():
    labels = np.ones((6, 6), dtype=np.int64) * 0
    labels[0:2, 0:2] = 1
    labels[4:6, 4:6] = 1
    result = _run(labels, 2)
    assert result.component_count == 3  # background(0) + two disconnected class-1 blocks


def test_multiple_classes_multiple_components_per_class():
    labels = np.zeros((6, 6), dtype=np.int64)
    labels[0:2, 0:2] = 1
    labels[4:6, 4:6] = 1
    labels[0:2, 4:6] = 2
    labels[4:6, 0:2] = 2
    result = _run(labels, 3)
    assert result.component_count == 5  # background + 2x class1 + 2x class2
    assert int(result.per_class_component_counts[1]) == 2
    assert int(result.per_class_component_counts[2]) == 2


def test_equal_score_ties_lowest_class_wins():
    class_count = 3
    s0 = np.zeros((class_count, 4, 4), dtype=np.float32)
    s0[:, 1, 1] = 7.0  # all three classes tied at pixel (1,1)
    result = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=class_count)
    assert result.unary_labels[1, 1] == 0  # lowest index wins the tie


def test_all_equal_class_scores_everywhere():
    class_count = 4
    s0 = np.full((class_count, 3, 3), 2.0, dtype=np.float32)
    result = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=class_count)
    assert np.all(result.unary_labels == 0)
    assert result.component_count == 1


def test_non_square_image():
    labels = np.zeros((3, 9), dtype=np.int64)
    labels[:, 5:] = 1
    result = _run(labels, 2)
    assert result.unary_labels.shape == (3, 9)


def test_read_only_and_non_contiguous_input():
    labels = np.zeros((5, 5), dtype=np.int64)
    labels[1:4, 1:4] = 1
    s0 = _s0_from_labels(labels, 2)
    s0_transposed_twice = s0.transpose(0, 2, 1).transpose(0, 2, 1)  # may or may not be contiguous
    s0_transposed_twice.setflags(write=False)
    result = gc.compute_global_unary_components(s0_transposed_twice, contract=TINY_CONTRACT, class_count=2)
    _assert_matches_reference(result, result.unary_labels)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_supported(dtype):
    labels = np.zeros((4, 4), dtype=np.int64)
    labels[2:, 2:] = 1
    result = _run(labels, 2, dtype=dtype)
    assert result.component_patch_equivalent_areas.dtype == np.float64


# ---------------------------------------------------------------------------
# Deterministic ID ordering (section 7)
# ---------------------------------------------------------------------------


def test_deterministic_across_repeated_calls():
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[2:5, 2:5] = 1
    labels[6:9, 6:9] = 2
    s0 = _s0_from_labels(labels, 3)
    r1 = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=3)
    r2 = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=3)
    assert np.array_equal(r1.component_ids, r2.component_ids)
    assert np.array_equal(r1.component_first_flat_indices, r2.component_first_flat_indices)


def test_deterministic_across_alternative_per_class_enumeration_order():
    # process classes in reverse order internally by feeding a
    # relabeled-but-equivalent unary map; the GLOBAL id assignment must
    # still be governed by first-flat-index, independent of any
    # per-class processing order.
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[0:3, 0:3] = 2
    labels[5:8, 5:8] = 1
    s0 = _s0_from_labels(labels, 3)
    result = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=3)
    assert np.all(np.diff(result.component_first_flat_indices) > 0)


def test_ids_not_sorted_by_area_or_score():
    labels = np.zeros((10, 3), dtype=np.int64)
    labels[0, 0] = 1  # tiny 1-pixel component first in row-major order
    result = _run(labels, 2)
    # component 0 must be the first-flat-index one (the tiny 1-pixel class-1 blob at (0,0)), not the largest
    assert int(result.component_class_indices[0]) == 1
    assert int(result.component_pixel_areas[0]) == 1


# ---------------------------------------------------------------------------
# Strict validation (section 10)
# ---------------------------------------------------------------------------


def test_wrong_rank_rejected():
    s0 = np.zeros((2, 4), dtype=np.float32)  # 2-D, not 3-D
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)


def test_zero_sized_dimension_rejected():
    s0 = np.zeros((2, 0, 4), dtype=np.float32)
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)


def test_wrong_class_count_rejected():
    s0 = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)


def test_integer_dtype_rejected():
    s0 = np.zeros((2, 4, 4), dtype=np.int32)
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)


def test_bool_dtype_rejected():
    s0 = np.zeros((2, 4, 4), dtype=bool)
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)


def test_complex_dtype_rejected():
    s0 = np.zeros((2, 4, 4), dtype=np.complex64)
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)


def test_nan_rejected():
    s0 = np.zeros((2, 4, 4), dtype=np.float32)
    s0[0, 0, 0] = float("nan")
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)


def test_inf_rejected():
    s0 = np.zeros((2, 4, 4), dtype=np.float32)
    s0[0, 0, 0] = float("inf")
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)


def test_non_dfpcomponentcontract_rejected():
    s0 = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract={"connectivity": 8}, class_count=2)


def test_bool_class_count_rejected():
    s0 = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=True)


def test_wrong_container_for_s0_rejected():
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.compute_global_unary_components([[1, 2], [3, 4]], contract=TINY_CONTRACT, class_count=2)


def test_missing_identity_key_rejected():
    protocol = copy.deepcopy(_real_protocol())
    protocol["structural_components"] = dict(protocol["structural_components"])
    del protocol["structural_components"]["primary_connectivity"]
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.dfp_component_contract_from_protocol(protocol, repo_root=ROOT)


# ---------------------------------------------------------------------------
# Partition invariants / self-check (section 11)
# ---------------------------------------------------------------------------


def test_self_check_catches_corrupted_component_ids():
    labels = np.zeros((5, 5), dtype=np.int64)
    labels[2:, 2:] = 1
    result = _run(labels, 2)
    bad_ids = result.component_ids.copy()
    bad_ids[0, 0] = result.component_count  # out of range
    bad_ids.flags.writeable = False
    corrupted = gc.GlobalUnaryComponents(
        unary_labels=result.unary_labels, component_ids=bad_ids,
        component_class_indices=result.component_class_indices, component_pixel_areas=result.component_pixel_areas,
        component_patch_equivalent_areas=result.component_patch_equivalent_areas,
        component_first_flat_indices=result.component_first_flat_indices, component_count=result.component_count,
        per_class_component_counts=result.per_class_component_counts,
    )
    with pytest.raises(gc.DfpPhase1aGlobalUnaryComponentsError):
        gc.validate_global_unary_components_partition(corrupted, image_shape=(5, 5), class_count=2)


def test_self_check_does_not_mutate_result():
    labels = np.zeros((5, 5), dtype=np.int64)
    labels[2:, 2:] = 1
    result = _run(labels, 2)
    snapshot = {name: getattr(result, name).copy() if isinstance(getattr(result, name), np.ndarray) else getattr(result, name) for name in gc._RESULT_ARRAY_FIELDS}
    gc.validate_global_unary_components_partition(result, image_shape=(5, 5), class_count=2, contract=TINY_CONTRACT)
    for name, snap in snapshot.items():
        assert np.array_equal(getattr(result, name), snap)


# ---------------------------------------------------------------------------
# Execution counts (section 13)
# ---------------------------------------------------------------------------


def test_single_labeling_pass_call_counts(monkeypatch):
    from scipy import ndimage

    labels = np.zeros((10, 10), dtype=np.int64)
    labels[2:5, 2:5] = 1
    labels[6:9, 6:9] = 2
    s0 = _s0_from_labels(labels, 3)

    derive_calls = {"n": 0}
    label_calls = {"n": 0}
    real_derive = gc.derive_horizon_labels
    real_label = gc.ndimage.label

    def _counting_derive(*a, **kw):
        derive_calls["n"] += 1
        return real_derive(*a, **kw)

    def _counting_label(*a, **kw):
        label_calls["n"] += 1
        return real_label(*a, **kw)

    monkeypatch.setattr(gc, "derive_horizon_labels", _counting_derive)
    monkeypatch.setattr(gc.ndimage, "label", _counting_label)
    gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=3)
    assert derive_calls["n"] == 1  # unary-label derivation: once
    assert label_calls["n"] == 3  # once per present class (0, 1, 2), never per window/pixel


def test_zero_model_graph_recurrence_stitching_calls():
    source = Path(gc.__file__).read_text(encoding="utf-8")
    for forbidden in ("finite_step_propagate", "build_directed_topk_graph", "interpolate_raw_scores", "build_stitch_weight", ".model.", "torch."):
        assert forbidden not in source, f"found forbidden reference {forbidden!r}"


# ---------------------------------------------------------------------------
# Mutation / aliasing (section 14)
# ---------------------------------------------------------------------------


def test_s0_not_mutated():
    labels = np.zeros((5, 5), dtype=np.int64)
    labels[2:, 2:] = 1
    s0 = _s0_from_labels(labels, 2)
    before = s0.copy()
    gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)
    assert np.array_equal(s0, before)


def test_output_does_not_alias_s0():
    labels = np.zeros((5, 5), dtype=np.int64)
    s0 = _s0_from_labels(labels, 2)
    result = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)
    assert result.unary_labels.base is not s0 and not np.shares_memory(result.unary_labels, s0)


def test_output_arrays_do_not_alias_each_other():
    labels = np.zeros((6, 6), dtype=np.int64)
    labels[2:4, 2:4] = 1
    s0 = _s0_from_labels(labels, 2)
    result = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)
    arrays = [getattr(result, n) for n in gc._RESULT_ARRAY_FIELDS]
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            assert not np.shares_memory(arrays[i], arrays[j])


def test_mutating_a_writable_copy_does_not_alter_the_result():
    labels = np.zeros((6, 6), dtype=np.int64)
    labels[2:4, 2:4] = 1
    s0 = _s0_from_labels(labels, 2)
    result = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=2)
    writable_copy = result.component_ids.copy()
    writable_copy[0, 0] = 999
    assert not np.array_equal(result.component_ids, writable_copy)
    assert int(result.component_ids[0, 0]) != 999


def test_all_result_arrays_are_read_only():
    labels = np.zeros((4, 4), dtype=np.int64)
    s0 = _s0_from_labels(labels, 1)
    result = gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=1)
    for name in gc._RESULT_ARRAY_FIELDS:
        assert getattr(result, name).flags.writeable is False


def test_contract_not_mutated():
    labels = np.zeros((4, 4), dtype=np.int64)
    s0 = _s0_from_labels(labels, 1)
    before = copy.deepcopy(TINY_CONTRACT)
    gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=1)
    assert TINY_CONTRACT == before


# ---------------------------------------------------------------------------
# No target-label leakage (section 3)
# ---------------------------------------------------------------------------


def test_no_ground_truth_parameter_anywhere():
    import inspect

    for fn in (gc.compute_global_unary_components, gc.dfp_component_contract_from_protocol, gc.validate_global_unary_components_partition, gc.component_summary, gc.microcomponent_summary):
        params = inspect.signature(fn).parameters
        for name in params:
            assert "gt" not in name.lower() and "ground_truth" not in name.lower() and "ignore" not in name.lower() and "annotation" not in name.lower() and "oracle" not in name.lower()


# ---------------------------------------------------------------------------
# Exception behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc_cls", [KeyboardInterrupt, SystemExit, MemoryError])
def test_exception_propagates_uncaught(exc_cls, monkeypatch):
    labels = np.zeros((4, 4), dtype=np.int64)
    s0 = _s0_from_labels(labels, 1)

    def _boom(*a, **kw):
        raise exc_cls("poisoned")

    monkeypatch.setattr(gc, "derive_horizon_labels", _boom)
    with pytest.raises(exc_cls):
        gc.compute_global_unary_components(s0, contract=TINY_CONTRACT, class_count=1)


def test_cuda_never_initialized():
    import torch

    assert torch.cuda.is_initialized() is False


# ---------------------------------------------------------------------------
# Determinism across PYTHONHASHSEED (section 17)
# ---------------------------------------------------------------------------


def test_deterministic_across_hash_seeds(tmp_path):
    script = tmp_path / "run_once.py"
    script.write_text(
        f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {str(ROOT / "src/open_vocabulary_segmentation")!r})
import numpy as np
import src.dfp_phase1a_global_unary_components as gc

labels = np.zeros((10, 10), dtype=np.int64)
labels[2:5, 2:5] = 1
labels[6:9, 6:9] = 2
s0 = np.zeros((3, 10, 10), dtype=np.float32)
for y in range(10):
    for x in range(10):
        s0[int(labels[y, x]), y, x] = 5.0
contract = gc.DfpComponentContract(connectivity=8, patch_pitch=(2, 2), microcomponent_threshold_patch_equivalents=4.0)
result = gc.compute_global_unary_components(s0, contract=contract, class_count=3)
print(result.component_ids.tobytes().hex())
print(list(result.component_first_flat_indices))
print(list(result.component_class_indices))
"""
    )
    import os

    outputs = []
    for seed in ("0", "1", "1337"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    assert len(set(outputs)) == 1, f"non-deterministic across PYTHONHASHSEED: {outputs}"
