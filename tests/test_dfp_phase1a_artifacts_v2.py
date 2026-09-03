"""Tests for the DFP Phase 1A per-image-stats artifact schema v2 -- the
versioned successor to Step 10's v1 schema (see the module-level v2
comment in ``src/dfp_phase1a_artifact_schema.py`` for the full
rationale/derivation).

Covers: v2 identity-TOML section validation (via the shared
``schema_summary``/``artifact_identity`` fixtures already exercised by
``test_dfp_phase1a_artifacts.py``), schema-variant identification
(exact-match on the (schema, schema_version) pair, never guessed),
per-member-category dtype/rank/shape rejection (scalar, confusion-2D,
offset, flattened), every relational content check (dispute-count
partition, boundary/interior partition, persistence-vs-dispute bounds,
per-variant confusion-statistic identities and row sums, offset-diff-
vs-scalar-count agreement, flat-length-vs-offset agreement, segment-sum
reconstruction), zero-progress support without ``np.stack([])``,
v1/v2 coexistence (each independently loadable/verifiable, v1 fully
unmodified), a full NPZ write/round-trip/load cycle, and (see "v2
checkpoint writer" below) the dedicated atomic-generation/self-verify/
orphan-cleanup/reconciliation/resume coverage for
``generate_and_install_checkpoint_v2`` -- the v2 sibling of Step 10's
own ``generate_and_install_checkpoint``, required by the execution
driver's checkpoint-write path (``src/dfp_phase1a_execute.py``'s
``_write_checkpoint``) to persist a v2 per-image-stats row set. It is
an ADDITIVE sibling, architecturally identical to the frozen v1 writer
(see that function's own docstring in ``src/dfp_phase1a_checkpoint.py``
for the full atomicity/reconciliation contract this mirrors): same
unpredictable generation naming, same self-verify-then-atomic-pointer-
install order, same orphan-cleanup-on-pre-replace-failure, same
ambiguous-post-replace-nonce-reconciliation, same never-delete-a-pre-
existing-generation guarantee -- only the schema name/version/member-
set/loader/``class_count`` plumbing differs, and it never touches or
alters the v1 writer's own code path.

CPU-only throughout. No real dataset, no model construction, no CUDA
initialization, no graph/recurrence execution."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1a_artifact_schema as sch
import src.dfp_phase1a_checkpoint as ckpt

ROOT = Path(__file__).resolve().parent.parent
COCO_CLASS_COUNT = 171


# ---------------------------------------------------------------------------
# Fixtures / self-consistent v2 array construction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact_identity():
    return sch.load_identity(repo_root=ROOT)


@pytest.fixture(scope="module")
def schema_summary():
    return sch.validate_static_configuration(repo_root=ROOT)


@pytest.fixture(scope="module")
def coco_selected_ids():
    manifest = json.loads((ROOT / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json").read_text())
    return manifest["selected_ids"], manifest["selected_ids_digest"]


def _random_row_partition(rng: np.random.Generator, total: int, k: int) -> np.ndarray:
    """k non-negative int64 parts summing exactly to `total` (zeros
    allowed) -- used for confusion marginals, where a class may
    legitimately carry zero pixels."""
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    cuts = np.sort(rng.integers(0, total + 1, size=k - 1)) if k > 1 else np.zeros(0, dtype=np.int64)
    edges = np.concatenate(([0], cuts, [total]))
    return np.diff(edges).astype(np.int64)


def _random_positive_partition(rng: np.random.Generator, total: int, k: int) -> np.ndarray:
    """k strictly-positive int64 parts summing exactly to `total` --
    used for component pixel areas / reversion component sizes, where
    every labeled component always has at least one pixel. Requires
    total >= k (guaranteed by every caller below)."""
    if k == 0:
        assert total == 0
        return np.zeros(0, dtype=np.int64)
    assert total >= k
    cuts = np.sort(rng.choice(np.arange(1, total), size=k - 1, replace=False)) if k > 1 else np.zeros(0, dtype=np.int64)
    edges = np.concatenate(([0], cuts, [total]))
    parts = np.diff(edges).astype(np.int64)
    assert np.all(parts >= 1)
    return parts


def _make_v2_arrays(n: int, *, class_count: int = COCO_CLASS_COUNT, seed: int = 0) -> dict[str, np.ndarray]:
    """A fully self-consistent v2 array set satisfying every relational
    invariant `ckpt._validate_per_image_stats_v2_relations` checks --
    the "happy path" fixture every corruption test in this file mutates
    exactly one field of."""
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}

    arrays["dataset_indices"] = np.arange(n, dtype=np.int64)
    for name in ("window_count", "total_node_count", "total_nonzero_count", "total_baseline_recurrence_flops",
                 "total_projected_recurrence_flops", "total_chunk_invocations", "total_projected_width",
                 "maximum_projected_width", "maximum_peak_planned_bytes"):
        arrays[name] = rng.integers(0, 1000, size=n).astype(np.int64)
    arrays["zero_dispute_window_count"] = np.minimum(rng.integers(0, 1000, size=n).astype(np.int64), arrays["window_count"])
    base = np.sort(rng.random((n, 5)), axis=1) if n else np.zeros((0, 5))
    for i, name in enumerate(("q_w_median", "q_w_p90", "q_w_p95", "q_w_p99", "q_w_maximum")):
        arrays[name] = base[:, i].astype(np.float64) if n else np.zeros(0, dtype=np.float64)
    arrays["q_w_mean"] = rng.random(n) if n else np.zeros(0, dtype=np.float64)

    evaluated = rng.integers(50, 200, size=n).astype(np.int64) if n else np.zeros(0, dtype=np.int64)
    ignore = rng.integers(0, 50, size=n).astype(np.int64) if n else np.zeros(0, dtype=np.int64)
    arrays["evaluated_pixel_count"] = evaluated
    arrays["ignore_pixel_count"] = ignore

    dispute = np.array([int(rng.integers(0, int(e) + 1)) for e in evaluated], dtype=np.int64)
    arrays["dispute_pixel_count"] = dispute
    harmful = np.zeros(n, dtype=np.int64)
    beneficial = np.zeros(n, dtype=np.int64)
    third_label = np.zeros(n, dtype=np.int64)
    boundary = np.zeros(n, dtype=np.int64)
    interior = np.zeros(n, dtype=np.int64)
    for i in range(n):
        parts = _random_row_partition(rng, int(dispute[i]), 3)
        harmful[i], beneficial[i], third_label[i] = parts
        b_parts = _random_row_partition(rng, int(dispute[i]), 2)
        boundary[i], interior[i] = b_parts
    arrays["harmful_pixel_count"] = harmful
    arrays["beneficial_pixel_count"] = beneficial
    arrays["third_label_pixel_count"] = third_label
    arrays["disputed_boundary_pixel_count"] = boundary
    arrays["disputed_interior_pixel_count"] = interior

    # harmful x boundary-band JOINT split: harmful_boundary must lie in
    # [max(0, harmful-interior), min(harmful, boundary)] to satisfy both
    # harmful_boundary<=boundary and harmful_interior<=interior -- always
    # feasible since harmful<=dispute==boundary+interior.
    harmful_boundary = np.zeros(n, dtype=np.int64)
    harmful_interior = np.zeros(n, dtype=np.int64)
    for i in range(n):
        lower = max(0, int(harmful[i]) - int(interior[i]))
        upper = min(int(harmful[i]), int(boundary[i]))
        harmful_boundary[i] = int(rng.integers(lower, upper + 1))
        harmful_interior[i] = int(harmful[i]) - int(harmful_boundary[i])
    arrays["harmful_boundary_pixel_count"] = harmful_boundary
    arrays["harmful_interior_pixel_count"] = harmful_interior

    for horizon in sch.PERSISTENCE_HORIZONS_V2:
        h_harmful = np.array([int(rng.integers(0, int(harmful[i]) + 1)) for i in range(n)], dtype=np.int64)
        h_beneficial = np.array([int(rng.integers(0, int(beneficial[i]) + 1)) for i in range(n)], dtype=np.int64)
        h_third_label = np.array([int(rng.integers(0, int(third_label[i]) + 1)) for i in range(n)], dtype=np.int64)
        arrays[f"persistence_h{horizon}_harmful_reverted_count"] = h_harmful
        arrays[f"persistence_h{horizon}_beneficial_reverted_count"] = h_beneficial
        arrays[f"persistence_h{horizon}_third_label_reverted_count"] = h_third_label
    # Deliberately randomized independently of `harmful` above (bounded
    # only by harmful[i], never made equal to it) -- proves reversion
    # coherence's own fixture below is not silently piggy-backing on
    # persistence_h32 by construction.

    # Ground truth does not vary with the prediction variant -- generated
    # ONCE per image and shared (as an independent copy) across every
    # variant, matching the classwise-GT-equality invariant every variant
    # is checked against.
    ground_truth_by_image = np.zeros((n, class_count), dtype=np.int64)
    for i in range(n):
        ground_truth_by_image[i] = _random_row_partition(rng, int(evaluated[i]), class_count)

    for variant in sch.PREDICTION_VARIANT_NAMES_V2:
        intersection = np.zeros((n, class_count), dtype=np.int64)
        predicted = np.zeros((n, class_count), dtype=np.int64)
        for i in range(n):
            predicted[i] = _random_row_partition(rng, int(evaluated[i]), class_count)
            row_intersection = np.array(
                [int(rng.integers(0, min(int(predicted[i, c]), int(ground_truth_by_image[i, c])) + 1)) for c in range(class_count)],
                dtype=np.int64,
            )
            intersection[i] = row_intersection
        ground_truth = ground_truth_by_image.copy()
        union = predicted + ground_truth - intersection
        arrays[sch.confusion_member_name_v2(variant, "intersection")] = intersection
        arrays[sch.confusion_member_name_v2(variant, "union")] = union
        arrays[sch.confusion_member_name_v2(variant, "predicted_pixels")] = predicted
        arrays[sch.confusion_member_name_v2(variant, "ground_truth_pixels")] = ground_truth

    total_pixels = evaluated + ignore
    component_count = np.array(
        [int(rng.integers(1, min(5, int(t)) + 1)) if t > 0 else 0 for t in total_pixels], dtype=np.int64,
    ) if n else np.zeros(0, dtype=np.int64)
    arrays["component_count"] = component_count
    component_offsets = np.concatenate(([0], np.cumsum(component_count))).astype(np.int64)
    arrays["component_offsets"] = component_offsets
    class_indices_parts = []
    pixel_area_parts = []
    patch_area_parts = []
    source_mass_parts = []
    for i in range(n):
        k = int(component_count[i])
        class_indices_parts.append(rng.integers(0, class_count, size=k).astype(np.int64))
        sizes = _random_positive_partition(rng, int(total_pixels[i]), k)
        pixel_area_parts.append(sizes)
        patch_area_parts.append(sizes.astype(np.float64) / 196.0)
        source_mass_parts.append(rng.random(k))
    arrays["component_class_indices_flat"] = np.concatenate(class_indices_parts).astype(np.int64) if n else np.zeros(0, dtype=np.int64)
    arrays["component_pixel_areas_flat"] = np.concatenate(pixel_area_parts).astype(np.int64) if n else np.zeros(0, dtype=np.int64)
    arrays["component_patch_equivalent_areas_flat"] = np.concatenate(patch_area_parts).astype(np.float64) if n else np.zeros(0, dtype=np.float64)
    arrays["component_source_mass_flat"] = np.concatenate(source_mass_parts).astype(np.float64) if n else np.zeros(0, dtype=np.float64)

    reversion_component_count = np.zeros(n, dtype=np.int64)
    reversion_size_parts = []
    perimeter = np.zeros(n, dtype=np.int64)
    for i in range(n):
        total_reverted = int(harmful[i])
        if total_reverted == 0:
            reversion_component_count[i] = 0
            reversion_size_parts.append(np.zeros(0, dtype=np.int64))
        else:
            k = int(rng.integers(1, min(4, total_reverted) + 1))
            reversion_component_count[i] = k
            reversion_size_parts.append(_random_positive_partition(rng, total_reverted, k))
        perimeter[i] = int(rng.integers(0, 500))
    arrays["reversion_component_count"] = reversion_component_count
    arrays["reversion_perimeter"] = perimeter
    reversion_offsets = np.concatenate(([0], np.cumsum(reversion_component_count))).astype(np.int64)
    arrays["reversion_component_offsets"] = reversion_offsets
    arrays["reversion_component_sizes_flat"] = np.concatenate(reversion_size_parts).astype(np.int64) if n else np.zeros(0, dtype=np.int64)

    assert set(arrays) == sch.ALL_NPZ_MEMBER_NAMES_V2, sorted(sch.ALL_NPZ_MEMBER_NAMES_V2 ^ set(arrays))
    return arrays


def _validate_all_v2(arrays: dict[str, np.ndarray], *, image_count: int, class_count: int = COCO_CLASS_COUNT) -> None:
    for name, array in arrays.items():
        ckpt.validate_per_image_stats_array_v2(name, array, image_count=image_count, class_count=class_count)
    ckpt._validate_per_image_stats_v2_relations(arrays, image_count=image_count, class_count=class_count)


# ---------------------------------------------------------------------------
# Identity TOML v2 section
# ---------------------------------------------------------------------------


def test_v2_identity_section_validates(artifact_identity):
    block = artifact_identity["per_image_stats_schema_v2"]
    assert block["schema_name"] == sch.PER_IMAGE_STATS_SCHEMA_NAME_V2
    assert block["schema_version"] == sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2
    assert block["predecessor_schema_name"] == sch.PER_IMAGE_STATS_SCHEMA_NAME
    assert block["predecessor_schema_version"] == sch.PER_IMAGE_STATS_SCHEMA_VERSION
    assert block["predecessor_still_independently_verifiable"] is True


def test_v1_schema_untouched_by_v2_addition():
    """v1's own name/version/member lists are byte-for-byte identical to
    before this change -- the successor schema is purely additive."""
    assert sch.PER_IMAGE_STATS_SCHEMA_NAME == "talk2dino-dfp-phase1a-per-image-stats-v1"
    assert sch.PER_IMAGE_STATS_SCHEMA_VERSION == 1
    assert len(sch.INT64_MEMBER_NAMES) == 13
    assert len(sch.FLOAT64_MEMBER_NAMES) == 6
    assert len(sch.ALL_NPZ_MEMBER_NAMES) == 19


def test_v2_member_count_and_disjointness():
    assert len(sch.ALL_NPZ_MEMBER_NAMES_V2) == 122
    # v2 carries every v1 member name forward unchanged (a strict subset).
    assert set(sch.INT64_MEMBER_NAMES) <= set(sch.SCALAR_INT64_MEMBER_NAMES_V2)
    assert set(sch.FLOAT64_MEMBER_NAMES) == set(sch.SCALAR_FLOAT64_MEMBER_NAMES_V2)
    assert sch.PER_IMAGE_STATS_SCHEMA_NAME_V2 != sch.PER_IMAGE_STATS_SCHEMA_NAME


# ---------------------------------------------------------------------------
# Schema-variant identification -- exact-match, never guessed
# ---------------------------------------------------------------------------


def test_variant_identification_v1_and_v2():
    assert ckpt.identify_per_image_stats_schema_variant(sch.PER_IMAGE_STATS_SCHEMA_NAME, sch.PER_IMAGE_STATS_SCHEMA_VERSION) == ckpt.PER_IMAGE_STATS_SCHEMA_VARIANT_V1
    assert ckpt.identify_per_image_stats_schema_variant(sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2) == ckpt.PER_IMAGE_STATS_SCHEMA_VARIANT_V2


@pytest.mark.parametrize(
    "name,version",
    [
        # Both schemas' current version number happens to be 1, so a
        # (correct name, wrong version NUMBER) case needs an actually-wrong
        # number (2), not the other schema's version (which is also 1 and
        # would otherwise be silently valid by accident of both being 1).
        (sch.PER_IMAGE_STATS_SCHEMA_NAME, 2),
        (sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, 2),
        ("talk2dino-dfp-phase1a-per-image-stats-v3", 1),
        ("", 1),
    ],
)
def test_variant_identification_rejects_unrecognized_pair(name, version):
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="does not match any known schema"):
        ckpt.identify_per_image_stats_schema_variant(name, version)


def test_variant_identification_rejects_unhashable_schema_field():
    """An unhashable schema/schema_version (e.g. a list) must fail
    closed as this module's own error, never leak a raw TypeError from
    the underlying dict lookup."""
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="not a valid \\(hashable\\)"):
        ckpt.identify_per_image_stats_schema_variant(["not", "hashable"], 1)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="not a valid \\(hashable\\)"):
        ckpt.identify_per_image_stats_schema_variant(sch.PER_IMAGE_STATS_SCHEMA_NAME, ["not", "hashable"])


def test_manifest_structure_accepts_both_variants_returns_variant():
    manifest_v1 = {
        "schema": sch.PER_IMAGE_STATS_SCHEMA_NAME, "schema_version": sch.PER_IMAGE_STATS_SCHEMA_VERSION,
        "npz_filename": "a.npz", "npz_sha256": "0" * 64, "run_mode": "coco_stuff_500",
        "class_count": COCO_CLASS_COUNT, "class_names_digest": "0" * 64, "image_count": 0, "image_ids": [],
    }
    assert ckpt.validate_per_image_stats_manifest_structure(manifest_v1) == ckpt.PER_IMAGE_STATS_SCHEMA_VARIANT_V1
    manifest_v2 = dict(manifest_v1, schema=sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, schema_version=sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2)
    assert ckpt.validate_per_image_stats_manifest_structure(manifest_v2) == ckpt.PER_IMAGE_STATS_SCHEMA_VARIANT_V2


# ---------------------------------------------------------------------------
# Happy-path construction round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 3, 17])
def test_v2_arrays_self_consistent(n):
    arrays = _make_v2_arrays(n, seed=n)
    _validate_all_v2(arrays, image_count=n)


def test_v2_zero_row_arrays_valid_and_no_np_stack_empty():
    arrays = ckpt.build_zero_row_arrays_v2(class_count=COCO_CLASS_COUNT)
    assert set(arrays) == sch.ALL_NPZ_MEMBER_NAMES_V2
    _validate_all_v2(arrays, image_count=0)
    for name in sch.OFFSET_MEMBER_NAMES_V2:
        assert arrays[name].shape == (1,)
        assert arrays[name][0] == 0
    for name in sch.CONFUSION_2D_MEMBER_NAMES_V2:
        assert arrays[name].shape == (0, COCO_CLASS_COUNT)


def test_v2_zero_row_arrays_rejects_bad_class_count():
    for bad in (0, -1, True, 1.5, "171"):
        with pytest.raises(ckpt.DfpPhase1aCheckpointError):
            ckpt.build_zero_row_arrays_v2(class_count=bad)


# ---------------------------------------------------------------------------
# Per-member-category structural rejection
# ---------------------------------------------------------------------------


def test_scalar_int64_rejects_wrong_dtype_rank_shape_negative():
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="dtype"):
        ckpt.validate_per_image_stats_array_v2("component_count", np.zeros(3, dtype=np.float64), image_count=3, class_count=COCO_CLASS_COUNT)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="1-dimensional"):
        ckpt.validate_per_image_stats_array_v2("component_count", np.zeros((3, 1), dtype=np.int64), image_count=3, class_count=COCO_CLASS_COUNT)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="rows"):
        ckpt.validate_per_image_stats_array_v2("component_count", np.zeros(2, dtype=np.int64), image_count=3, class_count=COCO_CLASS_COUNT)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="non-negative"):
        ckpt.validate_per_image_stats_array_v2("component_count", np.array([-1, 0, 1], dtype=np.int64), image_count=3, class_count=COCO_CLASS_COUNT)


def test_scalar_float64_rejects_nan_and_wrong_dtype():
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="dtype"):
        ckpt.validate_per_image_stats_array_v2("q_w_mean", np.zeros(3, dtype=np.int64), image_count=3, class_count=COCO_CLASS_COUNT)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="non-finite"):
        ckpt.validate_per_image_stats_array_v2("q_w_mean", np.array([1.0, np.nan, 2.0]), image_count=3, class_count=COCO_CLASS_COUNT)


def test_confusion_2d_rejects_wrong_rank_shape_negative():
    name = "conf2d_t0_intersection"
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="2-dimensional"):
        ckpt.validate_per_image_stats_array_v2(name, np.zeros(3, dtype=np.int64), image_count=3, class_count=5)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="shape"):
        ckpt.validate_per_image_stats_array_v2(name, np.zeros((3, 4), dtype=np.int64), image_count=3, class_count=5)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="non-negative"):
        arr = np.zeros((3, 5), dtype=np.int64)
        arr[0, 0] = -1
        ckpt.validate_per_image_stats_array_v2(name, arr, image_count=3, class_count=5)


def test_offset_array_rejects_wrong_length_nonzero_start_nonmonotone():
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="offset array"):
        ckpt.validate_per_image_stats_array_v2("component_offsets", np.zeros(3, dtype=np.int64), image_count=3, class_count=COCO_CLASS_COUNT)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="start at exactly 0"):
        ckpt.validate_per_image_stats_array_v2("component_offsets", np.array([1, 1, 1, 1], dtype=np.int64), image_count=3, class_count=COCO_CLASS_COUNT)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="non-decreasing"):
        ckpt.validate_per_image_stats_array_v2("component_offsets", np.array([0, 3, 1, 5], dtype=np.int64), image_count=3, class_count=COCO_CLASS_COUNT)


def test_flat_member_rejects_wrong_dtype_or_rank():
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="dtype"):
        ckpt.validate_per_image_stats_array_v2("component_class_indices_flat", np.zeros(3, dtype=np.float64), image_count=3, class_count=COCO_CLASS_COUNT)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="1-dimensional"):
        ckpt.validate_per_image_stats_array_v2("component_source_mass_flat", np.zeros((3, 1), dtype=np.float64), image_count=3, class_count=COCO_CLASS_COUNT)


def test_unknown_member_name_rejected():
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="not part of the authoritative v2 schema"):
        ckpt.validate_per_image_stats_array_v2("totally_unknown_member", np.zeros(3, dtype=np.int64), image_count=3, class_count=COCO_CLASS_COUNT)


def test_big_endian_byteorder_rejected():
    arr = np.zeros(3, dtype=">i8")
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="byte order"):
        ckpt.validate_per_image_stats_array_v2("component_count", arr, image_count=3, class_count=COCO_CLASS_COUNT)


# ---------------------------------------------------------------------------
# Relational content rejection -- one field mutated at a time from a known-
# valid, fully self-consistent baseline.
# ---------------------------------------------------------------------------


def test_dataset_indices_not_canonical_rejected():
    arrays = _make_v2_arrays(5, seed=1)
    arrays["dataset_indices"] = arrays["dataset_indices"][::-1].copy()
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="canonical"):
        _validate_all_v2(arrays, image_count=5)


def test_dispute_partition_violation_rejected():
    arrays = _make_v2_arrays(5, seed=2)
    arrays["harmful_pixel_count"] = arrays["harmful_pixel_count"].copy()
    arrays["harmful_pixel_count"][0] += 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="harmful_pixel_count\\+beneficial_pixel_count"):
        _validate_all_v2(arrays, image_count=5)


def test_boundary_interior_partition_violation_rejected():
    arrays = _make_v2_arrays(5, seed=3)
    arrays["disputed_boundary_pixel_count"] = arrays["disputed_boundary_pixel_count"].copy()
    arrays["disputed_boundary_pixel_count"][0] += 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="disputed_boundary_pixel_count"):
        _validate_all_v2(arrays, image_count=5)


def test_persistence_exceeds_category_count_rejected():
    arrays = _make_v2_arrays(5, seed=4)
    arrays["persistence_h32_harmful_reverted_count"] = arrays["harmful_pixel_count"] + 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="persistence_h32_harmful_reverted_count exceeds"):
        _validate_all_v2(arrays, image_count=5)


def test_confusion_union_identity_violation_rejected():
    arrays = _make_v2_arrays(5, seed=5)
    key = sch.confusion_member_name_v2("t0", "union")
    arrays[key] = arrays[key].copy()
    arrays[key][0, 0] += 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="conf2d_t0_union disagrees"):
        _validate_all_v2(arrays, image_count=5)


def test_confusion_intersection_exceeds_predicted_rejected():
    arrays = _make_v2_arrays(5, seed=6)
    inter_key = sch.confusion_member_name_v2("t320", "intersection")
    pred_key = sch.confusion_member_name_v2("t320", "predicted_pixels")
    gt_key = sch.confusion_member_name_v2("t320", "ground_truth_pixels")
    union_key = sch.confusion_member_name_v2("t320", "union")
    # Pick the class column with the largest ground_truth value for image 0
    # so union stays non-negative (union = predicted + ground_truth -
    # intersection) even after inflating intersection past predicted --
    # the array-level non-negativity check must not fire before the
    # relational "intersection exceeds" check under test.
    c = int(np.argmax(arrays[gt_key][0]))
    assert arrays[gt_key][0, c] >= 1
    arrays[inter_key] = arrays[inter_key].copy()
    arrays[union_key] = arrays[union_key].copy()
    arrays[inter_key][0, c] = arrays[pred_key][0, c] + 1
    arrays[union_key][0, c] = arrays[pred_key][0, c] + arrays[gt_key][0, c] - arrays[inter_key][0, c]
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="intersection exceeds"):
        _validate_all_v2(arrays, image_count=5)


def test_confusion_predicted_row_sum_violation_rejected():
    arrays = _make_v2_arrays(5, seed=7)
    key = sch.confusion_member_name_v2("oracle_o_p", "predicted_pixels")
    union_key = sch.confusion_member_name_v2("oracle_o_p", "union")
    arrays[key] = arrays[key].copy()
    arrays[union_key] = arrays[union_key].copy()
    arrays[key][0, 0] += 1
    arrays[union_key][0, 0] += 1  # keep the union identity intact so only the row-sum check fires
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="does not sum to evaluated_pixel_count"):
        _validate_all_v2(arrays, image_count=5)


def test_component_offsets_diff_mismatch_rejected():
    arrays = _make_v2_arrays(5, seed=8)
    arrays["component_count"] = arrays["component_count"].copy()
    arrays["component_count"][0] += 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="component_offsets diffs disagree"):
        _validate_all_v2(arrays, image_count=5)


def test_reversion_offsets_diff_mismatch_rejected():
    arrays = _make_v2_arrays(5, seed=9)
    arrays["reversion_component_count"] = arrays["reversion_component_count"].copy()
    arrays["reversion_component_count"][0] += 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="reversion_component_offsets diffs disagree"):
        _validate_all_v2(arrays, image_count=5)


def test_flat_length_mismatch_rejected():
    arrays = _make_v2_arrays(5, seed=10)
    arrays["component_class_indices_flat"] = np.concatenate([arrays["component_class_indices_flat"], np.zeros(1, dtype=np.int64)])
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="has length .* expected exactly"):
        _validate_all_v2(arrays, image_count=5)


def test_class_index_out_of_range_rejected():
    arrays = _make_v2_arrays(5, seed=11)
    if arrays["component_class_indices_flat"].size == 0:
        pytest.skip("no components generated for this seed")
    arrays["component_class_indices_flat"] = arrays["component_class_indices_flat"].copy()
    arrays["component_class_indices_flat"][0] = COCO_CLASS_COUNT
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="outside \\[0, class_count\\)"):
        _validate_all_v2(arrays, image_count=5)


def test_component_pixel_area_zero_rejected():
    arrays = _make_v2_arrays(5, seed=12)
    if arrays["component_pixel_areas_flat"].size == 0:
        pytest.skip("no components generated for this seed")
    arrays["component_pixel_areas_flat"] = arrays["component_pixel_areas_flat"].copy()
    arrays["component_pixel_areas_flat"][0] = 0
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match=">= 1"):
        _validate_all_v2(arrays, image_count=5)


def test_component_pixel_area_segment_sum_violation_rejected():
    arrays = _make_v2_arrays(5, seed=13)
    if arrays["component_pixel_areas_flat"].size < 2:
        pytest.skip("need at least 2 flattened components for this seed")
    arrays["component_pixel_areas_flat"] = arrays["component_pixel_areas_flat"].copy()
    # shift one pixel of mass from component 0 to component 1 -- total flat
    # sum unchanged, per-image segment sum for image 0 is thus unaffected
    # UNLESS both components belong to different images; force a definite
    # violation instead by inflating the very first flattened value while
    # compensating nothing, changing image 0's segment sum only if image 0
    # has >=1 component (guaranteed skip above ensures >=2 total, but not
    # necessarily image 0) -- so directly target ignore_pixel_count instead,
    # which is guaranteed present and directly participates in the same
    # relation for every image.
    arrays["ignore_pixel_count"] = arrays["ignore_pixel_count"].copy()
    arrays["ignore_pixel_count"][0] += 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="component_pixel_areas_flat per-image segment sum"):
        _validate_all_v2(arrays, image_count=5)


def test_reversion_component_size_segment_sum_violation_rejected():
    arrays = _make_v2_arrays(5, seed=14)
    if arrays["reversion_component_sizes_flat"].size == 0:
        pytest.skip("no reversion components generated for this seed")
    # Corrupt one flattened SIZE value directly, leaving offsets/counts and
    # every persistence/harmful/beneficial/third_label count untouched --
    # isolates the segment-sum check from every other relation.
    arrays["reversion_component_sizes_flat"] = arrays["reversion_component_sizes_flat"].copy()
    arrays["reversion_component_sizes_flat"][0] += 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="reversion_component_sizes_flat per-image segment sum"):
        _validate_all_v2(arrays, image_count=5)


def test_percentile_monotonicity_still_enforced_in_v2():
    arrays = _make_v2_arrays(5, seed=15)
    arrays["q_w_p95"] = arrays["q_w_p95"].copy()
    arrays["q_w_p95"][0] = arrays["q_w_p90"][0] - 1.0 if arrays["q_w_p90"][0] > 0 else -1.0
    # ensure a genuine violation regardless of the random baseline
    arrays["q_w_p95"][0] = min(arrays["q_w_p95"][0], arrays["q_w_median"][0] - 1.0)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="monotonicity"):
        _validate_all_v2(arrays, image_count=5)


# ---------------------------------------------------------------------------
# Full closed-member-set enforcement, NPZ round trip
# ---------------------------------------------------------------------------


def test_missing_member_rejected(tmp_path):
    arrays = _make_v2_arrays(3, seed=20)
    del arrays["reversion_perimeter"]
    npz_path = tmp_path / "stats.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    digest = sch.sha256_file(npz_path)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="missing="):
        ckpt.load_per_image_stats_npz_v2(npz_path, expected_sha256=digest, image_count=3, class_count=COCO_CLASS_COUNT)


def test_unexpected_member_rejected(tmp_path):
    arrays = _make_v2_arrays(3, seed=21)
    arrays["totally_unexpected"] = np.zeros(3, dtype=np.int64)
    npz_path = tmp_path / "stats.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    digest = sch.sha256_file(npz_path)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="unexpected="):
        ckpt.load_per_image_stats_npz_v2(npz_path, expected_sha256=digest, image_count=3, class_count=COCO_CLASS_COUNT)


def test_digest_mismatch_rejected(tmp_path):
    arrays = _make_v2_arrays(3, seed=22)
    npz_path = tmp_path / "stats.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="SHA256 mismatch"):
        ckpt.load_per_image_stats_npz_v2(npz_path, expected_sha256="0" * 64, image_count=3, class_count=COCO_CLASS_COUNT)


@pytest.mark.parametrize("n", [0, 1, 5])
def test_npz_full_round_trip(tmp_path, n):
    arrays = _make_v2_arrays(n, seed=100 + n)
    npz_path = tmp_path / "stats.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    digest = sch.sha256_file(npz_path)
    loaded = ckpt.load_per_image_stats_npz_v2(npz_path, expected_sha256=digest, image_count=n, class_count=COCO_CLASS_COUNT)
    assert set(loaded) == sch.ALL_NPZ_MEMBER_NAMES_V2
    for name in arrays:
        np.testing.assert_array_equal(loaded[name], arrays[name])


# ---------------------------------------------------------------------------
# Repair task: joint harmful-boundary counts + the third_label_revert_to_t0
# prediction variant + classwise ground-truth equality across variants.
# ---------------------------------------------------------------------------


def test_harmful_boundary_and_interior_members_are_mandatory_not_derivable_from_marginals(tmp_path):
    """The joint harmful x boundary-band counts are NOT reconstructable
    from the marginal harmful_pixel_count and disputed_boundary_pixel_
    count alone (their intersection is genuinely unknown without it) --
    the schema must therefore make them mandatory members, never an
    optional/derivable pair. Deleting them from an otherwise-valid NPZ
    must be rejected as a missing-member error, never silently
    tolerated or silently reconstructed from the marginals."""
    arrays = _make_v2_arrays(3, seed=30)
    del arrays["harmful_boundary_pixel_count"]
    del arrays["harmful_interior_pixel_count"]
    npz_path = tmp_path / "stats.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    digest = sch.sha256_file(npz_path)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="missing=") as excinfo:
        ckpt.load_per_image_stats_npz_v2(npz_path, expected_sha256=digest, image_count=3, class_count=COCO_CLASS_COUNT)
    assert "harmful_boundary_pixel_count" in str(excinfo.value)
    assert "harmful_interior_pixel_count" in str(excinfo.value)


def test_harmful_boundary_plus_interior_must_equal_harmful_total():
    arrays = _make_v2_arrays(5, seed=31)
    arrays["harmful_boundary_pixel_count"] = arrays["harmful_boundary_pixel_count"].copy()
    arrays["harmful_boundary_pixel_count"][0] += 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="harmful_pixel_count disagrees with harmful_boundary_pixel_count\\+harmful_interior_pixel_count"):
        _validate_all_v2(arrays, image_count=5)


def test_harmful_boundary_exceeding_disputed_boundary_rejected():
    arrays = _make_v2_arrays(5, seed=37)  # chosen: harmful_boundary_pixel_count[0] > 0 at this seed
    if arrays["harmful_boundary_pixel_count"][0] == 0:
        pytest.skip("no harmful-boundary mass generated for image 0 at this seed")
    # Force disputed_boundary_pixel_count[0] strictly below the existing
    # harmful_boundary_pixel_count[0], compensating in disputed_interior_
    # pixel_count[0] so the dispute==boundary+interior partition (a
    # SEPARATE, already-covered invariant) stays satisfied and only the
    # harmful-specific bound check under test can fire.
    arrays["disputed_boundary_pixel_count"] = arrays["disputed_boundary_pixel_count"].copy()
    arrays["disputed_interior_pixel_count"] = arrays["disputed_interior_pixel_count"].copy()
    new_boundary = int(arrays["harmful_boundary_pixel_count"][0]) - 1
    delta = int(arrays["disputed_boundary_pixel_count"][0]) - new_boundary
    arrays["disputed_boundary_pixel_count"][0] = new_boundary
    arrays["disputed_interior_pixel_count"][0] += delta
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="harmful_boundary_pixel_count exceeds disputed_boundary_pixel_count"):
        _validate_all_v2(arrays, image_count=5)


def test_harmful_interior_exceeding_disputed_interior_rejected():
    arrays = _make_v2_arrays(5, seed=37)  # chosen: harmful_interior_pixel_count[0] > 0 at this seed
    if arrays["harmful_interior_pixel_count"][0] == 0:
        pytest.skip("no harmful-interior mass generated for image 0 at this seed")
    arrays["disputed_boundary_pixel_count"] = arrays["disputed_boundary_pixel_count"].copy()
    arrays["disputed_interior_pixel_count"] = arrays["disputed_interior_pixel_count"].copy()
    new_interior = int(arrays["harmful_interior_pixel_count"][0]) - 1
    delta = int(arrays["disputed_interior_pixel_count"][0]) - new_interior
    arrays["disputed_interior_pixel_count"][0] = new_interior
    arrays["disputed_boundary_pixel_count"][0] += delta
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="harmful_interior_pixel_count exceeds disputed_interior_pixel_count"):
        _validate_all_v2(arrays, image_count=5)


def test_third_label_revert_variant_confusion_members_are_mandatory(tmp_path):
    """The third_label_revert_to_t0 variant's four per-class confusion
    arrays must be present -- deleting them must be rejected as a
    missing-member error, never silently tolerated as an optional
    variant."""
    arrays = _make_v2_arrays(3, seed=34)
    for stat in sch.CONFUSION_STAT_NAMES_V2:
        del arrays[sch.confusion_member_name_v2("third_label_revert_to_t0", stat)]
    npz_path = tmp_path / "stats.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    digest = sch.sha256_file(npz_path)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="missing=") as excinfo:
        ckpt.load_per_image_stats_npz_v2(npz_path, expected_sha256=digest, image_count=3, class_count=COCO_CLASS_COUNT)
    assert "conf2d_third_label_revert_to_t0_intersection" in str(excinfo.value)


def test_third_label_revert_present_in_prediction_variant_names():
    assert "third_label_revert_to_t0" in sch.PREDICTION_VARIANT_NAMES_V2
    assert len(sch.PREDICTION_VARIANT_NAMES_V2) == 19
    assert len(sch.CONFUSION_2D_MEMBER_NAMES_V2) == 76


@pytest.mark.parametrize("variant", ["t320", "oracle_o_p", "third_label_revert_to_t0", "persist_h32_full"])
def test_classwise_ground_truth_equality_enforced_per_variant(variant):
    """Ground truth does not change across prediction variants -- every
    variant's per-class ground_truth_pixels must be IDENTICAL, class by
    class, to every other variant's, not merely equal in row-sum (a
    corruption that redistributes counts across classes while preserving
    the row total would otherwise slip past the row-sum check alone)."""
    arrays = _make_v2_arrays(5, seed=30)  # chosen: ground_truth[0,0] > 0 and ground_truth[0,1] < row total at this seed
    key = sch.confusion_member_name_v2(variant, "ground_truth_pixels")
    union_key = sch.confusion_member_name_v2(variant, "union")
    arrays[key] = arrays[key].copy()
    arrays[union_key] = arrays[union_key].copy()
    if arrays[key][0, 0] == 0 or arrays[key][0, 1] >= arrays[key][0].sum():
        pytest.skip("insufficient class mass to shift for this seed")
    # Steal one pixel of ground-truth mass from class 0 to class 1 for
    # image 0 -- preserves the row sum (still == evaluated_pixel_count)
    # but breaks classwise equality against every other variant.
    arrays[key][0, 0] -= 1
    arrays[key][0, 1] += 1
    arrays[union_key][0, 0] -= 1
    arrays[union_key][0, 1] += 1
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match=f"conf2d_{variant}_ground_truth_pixels disagrees classwise"):
        _validate_all_v2(arrays, image_count=5)


def test_new_v2_scalar_member_structural_rejection():
    """The two new scalar members go through the same shared category
    validator as every other scalar int64 member -- explicit smoke test
    naming them directly (dtype/rank/negativity)."""
    for name in ("harmful_boundary_pixel_count", "harmful_interior_pixel_count"):
        with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="dtype"):
            ckpt.validate_per_image_stats_array_v2(name, np.zeros(3, dtype=np.float64), image_count=3, class_count=COCO_CLASS_COUNT)
        with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="non-negative"):
            ckpt.validate_per_image_stats_array_v2(name, np.array([-1, 0, 1], dtype=np.int64), image_count=3, class_count=COCO_CLASS_COUNT)


def test_new_confusion_variant_structural_rejection():
    name = sch.confusion_member_name_v2("third_label_revert_to_t0", "intersection")
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="2-dimensional"):
        ckpt.validate_per_image_stats_array_v2(name, np.zeros(3, dtype=np.int64), image_count=3, class_count=5)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="shape"):
        ckpt.validate_per_image_stats_array_v2(name, np.zeros((3, 4), dtype=np.int64), image_count=3, class_count=5)


# ---------------------------------------------------------------------------
# Independent reconstruction proof: a hand-built, fully-specified
# multi-image fixture, using ONLY the already-frozen Step 3 analysis
# primitives (classify_disputes, construct_oracle_maps,
# compute_confusion_statistics, _official_mask_to_boundary) -- never this
# module's own random generator -- to prove the v2 schema's harmful-
# boundary joint counts and third_label_revert_to_t0 confusion arrays
# exactly match what those already-committed, independently-verified
# primitives would produce for a real image.
# ---------------------------------------------------------------------------


def test_independent_reconstruction_from_hand_built_multi_image_fixture():
    from src.dfp_phase1a_analysis import (
        classify_disputes, compute_confusion_statistics, construct_oracle_maps, _official_mask_to_boundary,
    )

    class_count = 3
    ignore_index = 255

    # --- Image 1: an 8x8 solid-rectangle harmful-washout region -------
    # GT = class 0 everywhere. T0 (a) = class 0 everywhere (correct).
    # T320 (b) = class 1 on the left half (columns 0-3), class 0 on the
    # right half (still correct). The left half is therefore a solid
    # harmful-washout rectangle (GT==a, GT!=b); the right half has no
    # dispute at all.
    gt1 = np.zeros((8, 8), dtype=np.int64)
    a1 = np.zeros((8, 8), dtype=np.int64)
    b1 = np.zeros((8, 8), dtype=np.int64)
    b1[:, 0:4] = 1
    dispute1 = classify_disputes(a1, b1, gt1, ignore_index=ignore_index)
    assert dispute1.harmful_count == 32  # 8 rows x 4 columns, hand-counted
    assert dispute1.beneficial_count == 0
    assert dispute1.third_label_count == 0

    boundary_mask1 = _official_mask_to_boundary(dispute1.dispute.astype(np.uint8)).astype(bool)
    expected_harmful_boundary_1 = int((dispute1.harmful & boundary_mask1).sum())
    expected_harmful_interior_1 = int((dispute1.harmful & ~boundary_mask1).sum())
    assert expected_harmful_boundary_1 + expected_harmful_interior_1 == dispute1.harmful_count
    # The boundary band must be a genuine, non-trivial subset of the
    # harmful region for this fixture to actually exercise the joint
    # split (not merely confirm the trivial all-interior or
    # all-boundary case).
    assert 0 < expected_harmful_boundary_1 < dispute1.harmful_count

    # --- Image 2: an 8x8 third-label-transition row --------------------
    # GT = class 0 everywhere. Row 0: T0 (a) = class 1, T320 (b) = class
    # 2 -- GT is neither a nor b, a third-label transition. Rows 1-7:
    # a = b = 0 (no dispute).
    gt2 = np.zeros((8, 8), dtype=np.int64)
    a2 = np.zeros((8, 8), dtype=np.int64)
    b2 = np.zeros((8, 8), dtype=np.int64)
    a2[0, :] = 1
    b2[0, :] = 2
    dispute2 = classify_disputes(a2, b2, gt2, ignore_index=ignore_index)
    assert dispute2.harmful_count == 0
    assert dispute2.beneficial_count == 0
    assert dispute2.third_label_count == 8  # one full row, hand-counted

    oracle2 = construct_oracle_maps(dispute2)
    # third_label_revert_to_t0 per this task's own definition: begin with
    # T320/P; wherever GT is neither T0's a(x) nor T320's b(x), replace
    # b(x) with a(x); leave every other pixel equal to T320 -- exactly
    # OracleMaps.third_label_diagnostic, reused unmodified.
    third_label_revert_map = oracle2.third_label_diagnostic
    assert np.array_equal(third_label_revert_map[0, :], np.full(8, 1, dtype=np.int64))  # row 0 reverted to a=1
    assert np.array_equal(third_label_revert_map[1:, :], np.zeros((7, 8), dtype=np.int64))  # elsewhere unchanged (b=0)

    stats2 = compute_confusion_statistics(third_label_revert_map, gt2, class_count=class_count, ignore_index=ignore_index)
    # Hand-computed expectation: predicted class 0 on 56 pixels (all
    # correct, since GT=0 everywhere), predicted class 1 on 8 pixels (all
    # wrong, GT=0 there); class 2 never predicted (it was only ever a
    # transient T320 value, entirely reverted away).
    np.testing.assert_array_equal(stats2.predicted_pixels, np.array([56, 8, 0], dtype=np.int64))
    np.testing.assert_array_equal(stats2.ground_truth_pixels, np.array([64, 0, 0], dtype=np.int64))
    np.testing.assert_array_equal(stats2.intersection, np.array([56, 0, 0], dtype=np.int64))
    np.testing.assert_array_equal(stats2.union, np.array([64, 8, 0], dtype=np.int64))

    # --- Assemble as a 2-image v2 artifact and confirm the schema's own
    # validators accept it end to end, using ONLY these independently,
    # hand-verified Step-3-derived numbers -- never this file's own
    # random array generator.
    stats1_t0 = compute_confusion_statistics(a1, gt1, class_count=class_count, ignore_index=ignore_index)
    stats1_t320 = compute_confusion_statistics(b1, gt1, class_count=class_count, ignore_index=ignore_index)
    stats2_t0 = compute_confusion_statistics(a2, gt2, class_count=class_count, ignore_index=ignore_index)
    stats2_t320 = compute_confusion_statistics(b2, gt2, class_count=class_count, ignore_index=ignore_index)

    def _stack(*rows: np.ndarray) -> np.ndarray:
        return np.stack(rows, axis=0).astype(np.int64)

    arrays = ckpt.build_zero_row_arrays_v2(class_count=class_count)
    arrays = {name: arr for name, arr in arrays.items()}  # shallow copy of the dict, not the arrays
    n = 2
    arrays["dataset_indices"] = np.arange(n, dtype=np.int64)
    for name in ("window_count", "zero_dispute_window_count", "total_node_count", "total_nonzero_count",
                 "total_baseline_recurrence_flops", "total_projected_recurrence_flops", "total_chunk_invocations",
                 "total_projected_width", "maximum_projected_width", "maximum_peak_planned_bytes"):
        arrays[name] = np.zeros(n, dtype=np.int64)
    for name in sch.SCALAR_FLOAT64_MEMBER_NAMES_V2:
        arrays[name] = np.zeros(n, dtype=np.float64)

    arrays["evaluated_pixel_count"] = np.array([64, 64], dtype=np.int64)
    arrays["ignore_pixel_count"] = np.array([0, 0], dtype=np.int64)
    arrays["dispute_pixel_count"] = np.array([dispute1.dispute_count, dispute2.dispute_count], dtype=np.int64)
    arrays["harmful_pixel_count"] = np.array([dispute1.harmful_count, dispute2.harmful_count], dtype=np.int64)
    arrays["beneficial_pixel_count"] = np.array([dispute1.beneficial_count, dispute2.beneficial_count], dtype=np.int64)
    arrays["third_label_pixel_count"] = np.array([dispute1.third_label_count, dispute2.third_label_count], dtype=np.int64)
    disputed_boundary_1 = int((dispute1.dispute & boundary_mask1).sum())
    disputed_interior_1 = int((dispute1.dispute & ~boundary_mask1).sum())
    arrays["disputed_boundary_pixel_count"] = np.array([disputed_boundary_1, 0], dtype=np.int64)
    arrays["disputed_interior_pixel_count"] = np.array([disputed_interior_1, dispute2.dispute_count], dtype=np.int64)
    arrays["harmful_boundary_pixel_count"] = np.array([expected_harmful_boundary_1, 0], dtype=np.int64)
    arrays["harmful_interior_pixel_count"] = np.array([expected_harmful_interior_1, 0], dtype=np.int64)
    for horizon in sch.PERSISTENCE_HORIZONS_V2:
        arrays[f"persistence_h{horizon}_harmful_reverted_count"] = np.zeros(n, dtype=np.int64)
        arrays[f"persistence_h{horizon}_beneficial_reverted_count"] = np.zeros(n, dtype=np.int64)
        arrays[f"persistence_h{horizon}_third_label_reverted_count"] = np.zeros(n, dtype=np.int64)
    # Global-unary-component data, hand-derived from a1/a2's own class
    # layout (component_pixel_areas_flat must sum, per image, to
    # evaluated_pixel_count+ignore_pixel_count -- the components tile the
    # WHOLE image, not just the dispute region). Image 1's T0 field (a1)
    # is a single class-0 region covering all 64 pixels -> one component.
    # Image 2's T0 field (a2) is row 0 (class 1, 8 pixels, the first
    # component in raster order) plus rows 1-7 (class 0, 56 pixels).
    arrays["component_count"] = np.array([1, 2], dtype=np.int64)
    arrays["component_offsets"] = np.array([0, 1, 3], dtype=np.int64)
    arrays["component_class_indices_flat"] = np.array([0, 1, 0], dtype=np.int64)
    arrays["component_pixel_areas_flat"] = np.array([64, 8, 56], dtype=np.int64)
    arrays["component_patch_equivalent_areas_flat"] = np.array([64.0, 8.0, 56.0], dtype=np.float64)
    arrays["component_source_mass_flat"] = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    # Reversion coherence (committed [reversion_coherence] authority):
    # the reverted-pixel set is harmful washout, GT(x)==a(x) and
    # GT(x)!=b(x) -- exactly dispute1.harmful (the solid 8x4 rectangle,
    # 32 pixels, one connected component under 8-connectivity, perimeter
    # 20) and dispute2.harmful (empty: image 2 is a pure third-label
    # transition, harmful_count=0).
    arrays["reversion_component_count"] = np.array([1, 0], dtype=np.int64)
    arrays["reversion_perimeter"] = np.array([20, 0], dtype=np.int64)
    arrays["reversion_component_offsets"] = np.array([0, 1, 1], dtype=np.int64)

    # Every other prediction variant's confusion arrays are set equal to
    # the T0 stats (a harmless placeholder that still satisfies every
    # relational invariant, including classwise GT equality) EXCEPT the
    # three variants under direct test here (t0, t320,
    # third_label_revert_to_t0), which carry their true, independently
    # computed Step-3 values.
    placeholder_by_image = (stats1_t0, stats2_t0)
    for variant in sch.PREDICTION_VARIANT_NAMES_V2:
        if variant == "t0":
            per_image = (stats1_t0, stats2_t0)
        elif variant == "t320":
            per_image = (stats1_t320, stats2_t320)
        elif variant == "third_label_revert_to_t0":
            # Image 1 has no third-label pixels, so its
            # third_label_diagnostic map equals b1 exactly (Step 3's own
            # construct_oracle_maps behavior: no third-label pixels to
            # revert).
            stats1_revert = compute_confusion_statistics(b1, gt1, class_count=class_count, ignore_index=ignore_index)
            per_image = (stats1_revert, stats2)
        else:
            per_image = placeholder_by_image
        arrays[sch.confusion_member_name_v2(variant, "intersection")] = _stack(per_image[0].intersection, per_image[1].intersection)
        arrays[sch.confusion_member_name_v2(variant, "union")] = _stack(per_image[0].union, per_image[1].union)
        arrays[sch.confusion_member_name_v2(variant, "predicted_pixels")] = _stack(per_image[0].predicted_pixels, per_image[1].predicted_pixels)
        arrays[sch.confusion_member_name_v2(variant, "ground_truth_pixels")] = _stack(per_image[0].ground_truth_pixels, per_image[1].ground_truth_pixels)

    arrays["reversion_component_sizes_flat"] = np.array([32], dtype=np.int64)

    assert set(arrays) == sch.ALL_NPZ_MEMBER_NAMES_V2, sorted(sch.ALL_NPZ_MEMBER_NAMES_V2 ^ set(arrays))
    _validate_all_v2(arrays, image_count=n, class_count=class_count)

    # Final, direct assertions that the SCHEMA's own stored values equal
    # the independently hand-verified Step-3 reconstruction -- the
    # actual "measurement reconstruction" proof this test exists for.
    assert int(arrays["harmful_boundary_pixel_count"][0]) == expected_harmful_boundary_1
    assert int(arrays["harmful_interior_pixel_count"][0]) == expected_harmful_interior_1
    np.testing.assert_array_equal(
        arrays[sch.confusion_member_name_v2("third_label_revert_to_t0", "predicted_pixels")][1], stats2.predicted_pixels,
    )
    np.testing.assert_array_equal(
        arrays[sch.confusion_member_name_v2("third_label_revert_to_t0", "ground_truth_pixels")][1], stats2.ground_truth_pixels,
    )


def test_v1_and_v2_npz_are_independently_loadable_side_by_side(tmp_path):
    """Nothing about adding v2 breaks the ability to build and validate a
    completely ordinary v1 artifact -- the two schemas coexist."""
    rng = np.random.default_rng(999)
    v1_arrays: dict[str, np.ndarray] = {}
    n = 4
    for name in sch.INT64_MEMBER_NAMES:
        v1_arrays[name] = np.arange(n, dtype=np.int64) if name == "dataset_indices" else rng.integers(0, 100, size=n).astype(np.int64)
    base = np.sort(rng.random((n, 5)), axis=1)
    for i, name in enumerate(("q_w_median", "q_w_p90", "q_w_p95", "q_w_p99", "q_w_maximum")):
        v1_arrays[name] = base[:, i]
    v1_arrays["q_w_mean"] = rng.random(n)
    v1_arrays["dispute_pixel_count"] = np.minimum(v1_arrays["dispute_pixel_count"], v1_arrays["evaluated_pixel_count"])
    v1_arrays["zero_dispute_window_count"] = np.minimum(v1_arrays["zero_dispute_window_count"], v1_arrays["window_count"])

    v1_path = tmp_path / "v1.npz"
    np.savez(v1_path, allow_pickle=False, **v1_arrays)
    v1_loaded = ckpt.load_per_image_stats_npz(v1_path, expected_sha256=sch.sha256_file(v1_path), image_count=n)
    assert set(v1_loaded) == sch.ALL_NPZ_MEMBER_NAMES

    v2_arrays = _make_v2_arrays(n, seed=200)
    v2_path = tmp_path / "v2.npz"
    np.savez(v2_path, allow_pickle=False, **v2_arrays)
    v2_loaded = ckpt.load_per_image_stats_npz_v2(v2_path, expected_sha256=sch.sha256_file(v2_path), image_count=n, class_count=COCO_CLASS_COUNT)
    assert set(v2_loaded) == sch.ALL_NPZ_MEMBER_NAMES_V2


# ---------------------------------------------------------------------------
# v2 checkpoint writer (generate_and_install_checkpoint_v2): dedicated
# atomic-generation/self-verify/orphan-cleanup/reconciliation/resume
# coverage, mirroring test_dfp_phase1a_artifacts.py's own v1-writer
# coverage member-for-member (see that file's _make_fields/_install/
# _snapshot/test_A..F for the exact behavior this reproduces against the
# v2 writer instead).
# ---------------------------------------------------------------------------


def _make_fields_v2(*, schema_summary, artifact_identity, selected_ids, selected_ids_digest, n: int, complete: bool = False) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": sch.sha256_file(ROOT / sch.IDENTITY_RELATIVE_PATH),
        "protocol_identity_sha256": schema_summary["protocol_identity_sha256"],
        "subset_manifest_sha256": schema_summary["coco_stuff_manifest_sha256"], "selected_ids_digest": selected_ids_digest,
        "run_mode": "coco_stuff_500", "image_count_expected": n if complete else len(selected_ids),
        "class_count": artifact_identity["dataset_binding"]["coco_stuff_class_count"],
        "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "horizon_order": list(sch.HORIZON_ORDER), "source_commit": "a" * 40,
        "required_ancestor_commit": artifact_identity["parent_identity_binding"]["required_ancestor_commit"],
        "bridge_checkpoint_sha256": schema_summary["bridge_checkpoint_sha256"], "next_dataset_index": n,
        "completed_image_ids": list(selected_ids[:n]), "images_completed_count": n, "total_windows_processed": n * 4,
        "complete": complete, "final_result_produced": False, "created_at_utc": now, "updated_at_utc": now,
    }


def _install_v2(tmp_path, *, schema_summary, artifact_identity, selected_ids, selected_ids_digest, n: int, complete: bool = False, seed: int = 0, class_count: int = COCO_CLASS_COUNT):
    fields = _make_fields_v2(
        schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids,
        selected_ids_digest=selected_ids_digest, n=n, complete=complete,
    )
    arrays = _make_v2_arrays(n, class_count=class_count, seed=seed)
    checkpoints_dir = tmp_path / "checkpoints"
    ckpt.generate_and_install_checkpoint_v2(
        checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
        arrays=arrays, image_ids=selected_ids[:n], class_count=class_count,
    )
    return checkpoints_dir


def _snapshot_v2(cp_dir):
    files = sorted(p.name for p in cp_dir.iterdir())
    hashes = {name: hashlib.sha256((cp_dir / name).read_bytes()).hexdigest() for name in files}
    return files, hashes


def _fully_reverify_v2(checkpoints_dir, *, class_count=COCO_CLASS_COUNT):
    checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    stats_manifest_path = checkpoints_dir / checkpoint["per_image_stats_manifest_relative_path"]
    stats_manifest = json.loads(stats_manifest_path.read_text())
    assert sch.sha256_file(stats_manifest_path) == checkpoint["per_image_stats_manifest_sha256"]
    assert stats_manifest["schema"] == sch.PER_IMAGE_STATS_SCHEMA_NAME_V2
    assert stats_manifest["schema_version"] == sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2
    ckpt.validate_checkpoint_against_per_image_stats(checkpoint, stats_manifest)
    npz_path = checkpoints_dir / stats_manifest["npz_filename"]
    ckpt.load_per_image_stats_npz_v2(
        npz_path, expected_sha256=stats_manifest["npz_sha256"], image_count=stats_manifest["image_count"], class_count=class_count,
    )
    return checkpoint, stats_manifest


def test_v2_writer_end_to_end_install_and_resume(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == 3
    assert result["resumable"] is True
    assert result["run_mode"] == "coco_stuff_500"
    assert result["per_image_stats_schema_variant"] == ckpt.PER_IMAGE_STATS_SCHEMA_VARIANT_V2
    assert set(result["arrays"]) == sch.ALL_NPZ_MEMBER_NAMES_V2


@pytest.mark.parametrize("n,label", [(0, "zero-progress"), (1, "one-row"), (3, "middle-prefix")], ids=["zero", "one", "middle"])
def test_v2_writer_progress_states(tmp_path, schema_summary, artifact_identity, coco_selected_ids, n, label):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=n, complete=False)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == n
    assert result["resumable"] is True
    for name in sch.SCALAR_INT64_MEMBER_NAMES_V2:
        assert result["arrays"][name].shape == (n,)


def test_v2_writer_final_minus_one_state(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """One row shy of the real, frozen 500-image COCO-Stuff subset --
    ``validate_resume_state`` re-derives ``image_count_expected`` from
    the actual committed subset manifest unconditionally (see its own
    step 4), so an authentic "final-minus-one" fixture must use the
    real total rather than an artificially shrunk one."""
    selected_ids, digest = coco_selected_ids
    n = len(selected_ids) - 1
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=n, complete=False, seed=70)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == n and result["resumable"] is True


def test_v2_writer_complete_state(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """The full, real 500-image total, complete=True -- resumable must
    be False and the row count must match the frozen subset exactly."""
    selected_ids, digest = coco_selected_ids
    n = len(selected_ids)
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=n, complete=True, seed=71)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == n and result["resumable"] is False


def test_v2_writer_manifest_and_npz_filenames_use_the_frozen_v2_prefix(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=80)
    checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    manifest_name = checkpoint["per_image_stats_manifest_relative_path"]
    stats_manifest = json.loads((checkpoints_dir / manifest_name).read_text())
    npz_name = stats_manifest["npz_filename"]
    assert manifest_name.startswith("dfp_phase1a_per_image_stats_v2_coco_stuff_500_")
    assert npz_name.startswith("dfp_phase1a_per_image_stats_v2_coco_stuff_500_")
    assert checkpoint["generation_nonce"] in manifest_name
    assert checkpoint["generation_nonce"] in npz_name


def test_v2_writer_rejects_class_count_mismatch(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    arrays = _make_v2_arrays(2, seed=81)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="class_count"):
        ckpt.generate_and_install_checkpoint_v2(
            checkpoints_dir=tmp_path / "checkpoints", checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
            arrays=arrays, image_ids=selected_ids[:2], class_count=COCO_CLASS_COUNT + 1,
        )
    assert not (tmp_path / "checkpoints").exists() or list((tmp_path / "checkpoints").iterdir()) == []


def test_v2_writer_rejects_v1_arrays_cross_feeding(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """v1's strict subset of members must be rejected by the v2 writer --
    the v2 writer accepts only the exact closed v2 member set."""
    selected_ids, digest = coco_selected_ids
    fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    rng = np.random.default_rng(82)
    v1_arrays: dict[str, np.ndarray] = {}
    for name in sch.INT64_MEMBER_NAMES:
        v1_arrays[name] = np.arange(2, dtype=np.int64) if name == "dataset_indices" else rng.integers(0, 100, size=2).astype(np.int64)
    base = np.sort(rng.random((2, 5)), axis=1)
    for i, name in enumerate(("q_w_median", "q_w_p90", "q_w_p95", "q_w_p99", "q_w_maximum")):
        v1_arrays[name] = base[:, i]
    v1_arrays["q_w_mean"] = rng.random(2)
    # Per-array validation runs (and fails) BEFORE the membership-set
    # ("missing"/"unexpected") check ever runs -- the first v1-only
    # member v1 happens to iterate to is rejected outright as unknown to
    # the v2 schema. Either diagnostic proves the same thing: the v2
    # writer never silently widens to accept a v1 array set.
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="not part of the authoritative schema|missing"):
        ckpt.generate_and_install_checkpoint_v2(
            checkpoints_dir=tmp_path / "checkpoints", checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
            arrays=v1_arrays, image_ids=selected_ids[:2], class_count=COCO_CLASS_COUNT,
        )


def test_v1_writer_rejects_v2_arrays_cross_feeding(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """Symmetric control: v2's superset of members must be rejected by
    the unmodified v1 writer -- proves v1 is genuinely untouched, not
    silently widened to also accept v2 rows."""
    selected_ids, digest = coco_selected_ids
    fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    v2_arrays = _make_v2_arrays(2, seed=83)
    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match="not part of the authoritative schema|unexpected"):
        ckpt.generate_and_install_checkpoint(
            checkpoints_dir=tmp_path / "checkpoints", checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
            arrays=v2_arrays, image_ids=selected_ids[:2],
        )


@pytest.mark.parametrize(
    "corrupt_name,corrupt_fn,match",
    [
        ("bad_dtype", lambda a: {**a, "window_count": a["window_count"].astype(np.float32)}, "int64"),
        ("bad_offsets", lambda a: {**a, "component_offsets": a["component_offsets"] + 1}, "component_offsets"),
        ("bad_flat_length", lambda a: {**a, "component_pixel_areas_flat": np.concatenate([a["component_pixel_areas_flat"], [1]])}, "component_pixel_areas_flat"),
        ("bad_reversion_relation", lambda a: {**a, "reversion_component_sizes_flat": a["reversion_component_sizes_flat"] + 1 if a["reversion_component_sizes_flat"].size else a["reversion_component_sizes_flat"]}, "reversion_component_sizes_flat|harmful_pixel_count"),
    ],
)
def test_v2_writer_malformed_arrays_fail_cleanly_and_preserve_previous(tmp_path, schema_summary, artifact_identity, coco_selected_ids, corrupt_name, corrupt_fn, match):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=90)
    before_files, before_hashes = _snapshot_v2(checkpoints_dir)

    good = _make_v2_arrays(3, seed=91)
    # reversion sizes must be nonempty to exercise the reversion-relation
    # corruption meaningfully; regenerate with a different seed until true
    # if the drawn fixture happened to have zero harmful pixels everywhere.
    tries = 0
    while corrupt_name == "bad_reversion_relation" and good["reversion_component_sizes_flat"].size == 0 and tries < 20:
        tries += 1
        good = _make_v2_arrays(3, seed=91 + tries)
    broken = corrupt_fn(good)
    fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)

    with pytest.raises(ckpt.DfpPhase1aCheckpointError, match=match):
        ckpt.generate_and_install_checkpoint_v2(
            checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
            arrays=broken, image_ids=selected_ids[:3], class_count=COCO_CLASS_COUNT,
        )

    after_files, after_hashes = _snapshot_v2(checkpoints_dir)
    assert before_files == after_files
    assert before_hashes == after_hashes
    assert not any(".tmp" in n for n in after_files)


def test_v2_writer_does_not_mutate_input_fields_or_arrays(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    fields_before = copy.deepcopy(fields)
    arrays = _make_v2_arrays(2, seed=92)
    arrays_before = {name: arr.copy() for name, arr in arrays.items()}
    ckpt.generate_and_install_checkpoint_v2(
        checkpoints_dir=tmp_path / "checkpoints", checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
        arrays=arrays, image_ids=selected_ids[:2], class_count=COCO_CLASS_COUNT,
    )
    assert fields == fields_before
    for name, arr in arrays.items():
        np.testing.assert_array_equal(arr, arrays_before[name])


def test_v2_writer_generation_success_installs_exactly_one_new_generation(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=100)
    before_files, _ = _snapshot_v2(checkpoints_dir)

    new_fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_v2_arrays(3, seed=101)
    ckpt.generate_and_install_checkpoint_v2(
        checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
        arrays=new_arrays, image_ids=selected_ids[:3], class_count=COCO_CLASS_COUNT,
    )

    after_files, _ = _snapshot_v2(checkpoints_dir)
    new_files = set(after_files) - set(before_files)
    npz_new = [f for f in new_files if f.endswith(".npz")]
    json_new = [f for f in new_files if f.endswith(".json")]
    assert len(npz_new) == 1 and len(json_new) == 1
    assert not any(".tmp" in name for name in after_files)
    checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    assert checkpoint["per_image_stats_manifest_relative_path"] == json_new[0]
    stats_manifest = json.loads((checkpoints_dir / json_new[0]).read_text())
    assert stats_manifest["npz_filename"] == npz_new[0]
    assert stats_manifest["schema"] == sch.PER_IMAGE_STATS_SCHEMA_NAME_V2


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: OSError("simulated disk-full during v2 checkpoint-pointer install"),
        lambda: ValueError("simulated serialization ValueError during v2 checkpoint-pointer install"),
        lambda: ckpt.DfpPhase1aCheckpointError("simulated domain error during v2 checkpoint-pointer install"),
        lambda: MemoryError("simulated MemoryError during v2 checkpoint-pointer install"),
    ],
    ids=["OSError", "ValueError", "DomainError", "MemoryError"],
)
def test_v2_writer_pointer_write_failure_matrix_preserves_old_and_cleans_new(tmp_path, schema_summary, artifact_identity, coco_selected_ids, exc_factory):
    """Mirrors test_pointer_write_failure_matrix_preserves_old_and_cleans_new
    for the v1 writer: every one of these (all ``Exception`` subclasses,
    so the writer's own ``except Exception`` reconciliation handler
    actually runs) must propagate completely unchanged (never wrapped,
    never swallowed), and must leave the previous checkpoint and its own
    generation byte-identical with no orphan .tmp/generation residue.
    ``MemoryError`` is included here (it subclasses ``Exception``); see
    ``test_v2_writer_keyboard_interrupt_and_system_exit_propagate_uncaught``
    below for ``KeyboardInterrupt``/``SystemExit``, which are
    ``BaseException`` subclasses the writer's handler structurally
    cannot catch at all -- propagation is proven, but orphan cleanup is
    not (and must not be) claimed for them."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=110)
    before_files, before_hashes = _snapshot_v2(checkpoints_dir)

    new_fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_fields_before = copy.deepcopy(new_fields)
    new_arrays = _make_v2_arrays(3, seed=111)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}
    exc_instance = exc_factory()

    def selective_fail(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:  # 1st call = per-image-stats manifest (succeeds); 2nd = checkpoint pointer
            raise exc_instance
        return original(path, doc, label=label)

    ckpt.write_json_atomically = selective_fail
    try:
        with pytest.raises(type(exc_instance)) as excinfo:
            ckpt.generate_and_install_checkpoint_v2(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
                arrays=new_arrays, image_ids=selected_ids[:3], class_count=COCO_CLASS_COUNT,
            )
        assert excinfo.value is exc_instance
    finally:
        ckpt.write_json_atomically = original

    after_files, after_hashes = _snapshot_v2(checkpoints_dir)
    assert before_files == after_files
    assert before_hashes == after_hashes
    assert not any(".tmp" in name for name in after_files)
    assert new_fields == new_fields_before
    _fully_reverify_v2(checkpoints_dir)


@pytest.mark.parametrize(
    "exc_factory", [lambda: KeyboardInterrupt("simulated Ctrl-C during v2 checkpoint-pointer install"), lambda: SystemExit("simulated exit during v2 checkpoint-pointer install")],
    ids=["KeyboardInterrupt", "SystemExit"],
)
def test_v2_writer_keyboard_interrupt_and_system_exit_propagate_uncaught(tmp_path, schema_summary, artifact_identity, coco_selected_ids, exc_factory):
    """KeyboardInterrupt/SystemExit are BaseException subclasses, so the
    writer's own ``except Exception as original_error:`` clause cannot
    (and must not be made to) catch them: they must propagate completely
    unhandled, exactly as raised, with the OLD checkpoint pointer itself
    never overwritten (checkpoint.json content unchanged) -- but, unlike
    a genuine ``Exception``, orphan generation-file cleanup does not run
    for these (the handler that performs it is never entered), which is
    the correct, deliberate consequence of never trapping Ctrl-C/exit
    for cleanup purposes, not a defect."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=112)
    old_checkpoint_bytes = (checkpoints_dir / "checkpoint.json").read_bytes()

    new_fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_v2_arrays(3, seed=113)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}
    exc_instance = exc_factory()

    def selective_fail(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise exc_instance
        return original(path, doc, label=label)

    ckpt.write_json_atomically = selective_fail
    try:
        with pytest.raises(type(exc_instance)) as excinfo:
            ckpt.generate_and_install_checkpoint_v2(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
                arrays=new_arrays, image_ids=selected_ids[:3], class_count=COCO_CLASS_COUNT,
            )
        assert excinfo.value is exc_instance
    finally:
        ckpt.write_json_atomically = original

    # The OLD checkpoint pointer itself was never overwritten (its own
    # os.replace never ran), so the previous, still-valid checkpoint and
    # generation remain exactly as they were and continue to fully verify.
    assert (checkpoints_dir / "checkpoint.json").read_bytes() == old_checkpoint_bytes
    _fully_reverify_v2(checkpoints_dir)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == 2


@pytest.mark.parametrize("target", ["np.savez", "write-manifest", "candidate-validation", "self-verification", "os.replace-npz"])
def test_v2_writer_generation_failure_injection_matrix(tmp_path, schema_summary, artifact_identity, coco_selected_ids, target, monkeypatch):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=120)
    before_files, before_hashes = _snapshot_v2(checkpoints_dir)

    new_fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_v2_arrays(3, seed=121)

    if target == "np.savez":
        monkeypatch.setattr(ckpt.np, "savez", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated np.savez failure")))
    elif target == "write-manifest":
        original = ckpt.write_json_atomically
        state = {"n": 0}

        def fail_first(path, doc, *, label):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("simulated manifest write failure")
            return original(path, doc, label=label)

        monkeypatch.setattr(ckpt, "write_json_atomically", fail_first)
    elif target == "candidate-validation":
        monkeypatch.setattr(ckpt, "validate_checkpoint_structure", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated candidate validation failure")))
    elif target == "self-verification":
        monkeypatch.setattr(ckpt, "load_per_image_stats_npz_v2", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated self-verification failure")))
    elif target == "os.replace-npz":
        import os as os_module

        original_replace = os_module.replace
        state = {"n": 0}

        def fail_first_replace(*a, **kw):
            state["n"] += 1
            if state["n"] == 1:
                raise OSError("simulated os.replace failure during NPZ install")
            return original_replace(*a, **kw)

        monkeypatch.setattr(os_module, "replace", fail_first_replace)

    with pytest.raises(Exception):
        ckpt.generate_and_install_checkpoint_v2(
            checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
            arrays=new_arrays, image_ids=selected_ids[:3], class_count=COCO_CLASS_COUNT,
        )

    after_files, after_hashes = _snapshot_v2(checkpoints_dir)
    assert before_files == after_files
    assert before_hashes == after_hashes
    assert not any(".tmp" in name for name in after_files)


def test_v2_writer_B_post_replacement_memory_error(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """The pointer write's os.replace genuinely succeeds, then
    MemoryError fires from post-replace bookkeeping: MemoryError must
    propagate; the installed checkpoint must carry the CURRENT
    generation and fully verify; nothing referenced is deleted."""
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=130)
    old_checkpoint, _ = _fully_reverify_v2(checkpoints_dir)
    old_generation_files = {p.name for p in checkpoints_dir.iterdir()} - {"checkpoint.json"}

    new_fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_v2_arrays(3, seed=131)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}

    def succeed_then_memory_error(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            original(path, doc, label=label)
            raise MemoryError("simulated MemoryError from v2 post-replace bookkeeping")
        return original(path, doc, label=label)

    ckpt.write_json_atomically = succeed_then_memory_error
    try:
        with pytest.raises(MemoryError):
            ckpt.generate_and_install_checkpoint_v2(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
                arrays=new_arrays, image_ids=selected_ids[:3], class_count=COCO_CLASS_COUNT,
            )
    finally:
        ckpt.write_json_atomically = original

    installed_checkpoint = json.loads((checkpoints_dir / "checkpoint.json").read_text())
    assert installed_checkpoint["next_dataset_index"] == 3
    assert installed_checkpoint != old_checkpoint
    all_files = {p.name for p in checkpoints_dir.iterdir()}
    assert old_generation_files <= all_files, "old generation files must never be deleted"
    manifest_ref = installed_checkpoint["per_image_stats_manifest_relative_path"]
    assert installed_checkpoint["generation_nonce"] in manifest_ref
    _fully_reverify_v2(checkpoints_dir)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert result["next_dataset_index"] == 3
    assert not any(".tmp" in n for n in all_files)


def test_v2_writer_D_installed_pointer_still_references_old_generation(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=140)
    before_files, before_hashes = _snapshot_v2(checkpoints_dir)
    old_checkpoint, _ = _fully_reverify_v2(checkpoints_dir)

    new_fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_v2_arrays(3, seed=141)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}
    injected = OSError("simulated failure, old v2 pointer never replaced")

    def fail_leaving_old_intact(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise injected
        return original(path, doc, label=label)

    ckpt.write_json_atomically = fail_leaving_old_intact
    try:
        with pytest.raises(OSError):
            ckpt.generate_and_install_checkpoint_v2(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
                arrays=new_arrays, image_ids=selected_ids[:3], class_count=COCO_CLASS_COUNT,
            )
    finally:
        ckpt.write_json_atomically = original

    after_files, after_hashes = _snapshot_v2(checkpoints_dir)
    assert before_files == after_files
    assert before_hashes == after_hashes
    reverified_checkpoint, _ = _fully_reverify_v2(checkpoints_dir)
    assert reverified_checkpoint == old_checkpoint


def test_v2_writer_E_reconciliation_itself_fails(tmp_path, schema_summary, artifact_identity, coco_selected_ids, monkeypatch):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = _install_v2(tmp_path, schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2, seed=150)

    new_fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=3)
    new_arrays = _make_v2_arrays(3, seed=151)

    original_write = ckpt.write_json_atomically
    call_count = {"n": 0}
    original_error = ValueError("original v2 pointer-write failure")

    def fail_pointer_write(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise original_error
        return original_write(path, doc, label=label)

    def broken_reconciliation_read(*a, **kw):
        raise RuntimeError("simulated reconciliation read failure -- must never be what propagates")

    monkeypatch.setattr(ckpt, "write_json_atomically", fail_pointer_write)
    monkeypatch.setattr(ckpt, "parse_strict_json_document", broken_reconciliation_read)

    with pytest.raises(ValueError) as excinfo:
        ckpt.generate_and_install_checkpoint_v2(
            checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=new_fields,
            arrays=new_arrays, image_ids=selected_ids[:3], class_count=COCO_CLASS_COUNT,
        )
    assert excinfo.value is original_error

    all_files = {p.name for p in checkpoints_dir.iterdir()}
    npz_files = [f for f in all_files if f.endswith(".npz")]
    json_gen_files = [f for f in all_files if f.endswith(".json") and f != "checkpoint.json"]
    assert len(npz_files) == 2, "both the old and the (conservatively preserved) new generation NPZ must remain"
    assert len(json_gen_files) == 2


def test_v2_writer_F_fresh_checkpoint_pre_replace(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = tmp_path / "fresh_pre_replace_v2"
    checkpoints_dir.mkdir()
    fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    arrays = _make_v2_arrays(2, seed=160)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}
    injected = OSError("simulated failure before v2 checkpoint.json is ever created")

    def fail_before_creation(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise injected
        return original(path, doc, label=label)

    ckpt.write_json_atomically = fail_before_creation
    try:
        with pytest.raises(OSError):
            ckpt.generate_and_install_checkpoint_v2(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
                arrays=arrays, image_ids=selected_ids[:2], class_count=COCO_CLASS_COUNT,
            )
    finally:
        ckpt.write_json_atomically = original

    assert not (checkpoints_dir / "checkpoint.json").exists()
    assert list(checkpoints_dir.iterdir()) == []


def test_v2_writer_F_fresh_checkpoint_post_replace(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    selected_ids, digest = coco_selected_ids
    checkpoints_dir = tmp_path / "fresh_post_replace_v2"
    checkpoints_dir.mkdir()
    fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=2)
    arrays = _make_v2_arrays(2, seed=161)

    original = ckpt.write_json_atomically
    call_count = {"n": 0}

    def succeed_then_error(path, doc, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:
            original(path, doc, label=label)
            raise RuntimeError("simulated post-replace failure on first-ever v2 checkpoint")
        return original(path, doc, label=label)

    ckpt.write_json_atomically = succeed_then_error
    try:
        with pytest.raises(RuntimeError):
            ckpt.generate_and_install_checkpoint_v2(
                checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
                arrays=arrays, image_ids=selected_ids[:2], class_count=COCO_CLASS_COUNT,
            )
    finally:
        ckpt.write_json_atomically = original

    assert (checkpoints_dir / "checkpoint.json").exists()
    _fully_reverify_v2(checkpoints_dir)


def test_v2_writer_full_122_member_fixture_with_harmful_diverging_from_all_t32_counts(tmp_path, schema_summary, artifact_identity, coco_selected_ids):
    """Section-7-equivalent: a complete v2 install where harmful_pixel_
    count differs from every T32 persistence count, reversion sizes sum
    exactly to harmful, and the installed/resumed row round-trips."""
    selected_ids, digest = coco_selected_ids
    fields = _make_fields_v2(schema_summary=schema_summary, artifact_identity=artifact_identity, selected_ids=selected_ids, selected_ids_digest=digest, n=1)
    arrays = _make_v2_arrays(1, seed=0)
    arrays = dict(arrays)
    arrays["harmful_pixel_count"] = np.array([13], dtype=np.int64)
    arrays["beneficial_pixel_count"] = np.array([0], dtype=np.int64)
    arrays["third_label_pixel_count"] = np.array([0], dtype=np.int64)
    arrays["dispute_pixel_count"] = np.array([13], dtype=np.int64)
    arrays["disputed_boundary_pixel_count"] = np.array([13], dtype=np.int64)
    arrays["disputed_interior_pixel_count"] = np.array([0], dtype=np.int64)
    arrays["harmful_boundary_pixel_count"] = np.array([13], dtype=np.int64)
    arrays["harmful_interior_pixel_count"] = np.array([0], dtype=np.int64)
    arrays["evaluated_pixel_count"] = np.maximum(arrays["evaluated_pixel_count"], np.array([20], dtype=np.int64))
    for horizon in (16, 32, 64):
        arrays[f"persistence_h{horizon}_harmful_reverted_count"] = np.array([4], dtype=np.int64)
        arrays[f"persistence_h{horizon}_beneficial_reverted_count"] = np.array([0], dtype=np.int64)
        arrays[f"persistence_h{horizon}_third_label_reverted_count"] = np.array([0], dtype=np.int64)
    arrays["reversion_component_count"] = np.array([2], dtype=np.int64)
    arrays["reversion_component_offsets"] = np.array([0, 2], dtype=np.int64)
    arrays["reversion_component_sizes_flat"] = np.array([5, 8], dtype=np.int64)
    assert int(arrays["reversion_component_sizes_flat"].sum()) == 13
    assert 13 != (4 + 0 + 0)

    checkpoints_dir = tmp_path / "checkpoints"
    ckpt.generate_and_install_checkpoint_v2(
        checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields,
        arrays=arrays, image_ids=selected_ids[:1], class_count=COCO_CLASS_COUNT,
    )
    checkpoint, stats_manifest = _fully_reverify_v2(checkpoints_dir)
    result = ckpt.validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert set(result["arrays"]) == sch.ALL_NPZ_MEMBER_NAMES_V2
    assert int(result["arrays"]["harmful_pixel_count"][0]) == 13
    assert int(result["arrays"]["reversion_component_sizes_flat"].sum()) == 13
