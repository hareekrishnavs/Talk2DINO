"""Numerical tests for the composed final-resolution stitching pullback
(``src/dfp_phase1a_final_rescale_pullback.py``): R forward/adjoint
correctness against explicit dense matrices, the adjoint identity, the
full composed identity against Step 7's existing window forward/pullback,
soft-membership partition-of-unity and source-mass conservation, and a
static poison check that no nearest-neighbor resampling is ever used for
components or memberships in this module."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent

# Canonical import path, matching production code (src/dfp_phase1a_
# horizon_stitching.py, src/dfp_phase1a_execute.py) and every other DFP
# test file that touches this module (test_dfp_phase1a_stitching_
# pullback.py, test_dfp_phase1a_soft_membership.py, test_dfp_phase1a_
# horizon_stitching.py): the bare `segmentation.evaluation.sliding_
# window_geometry` path, never the dotted `src.open_vocabulary_
# segmentation.segmentation...` alias -- importing the SAME physical
# module under two different qualified names creates two distinct class
# objects in sys.modules, which breaks isinstance checks (e.g.
# stitch_dfp_phase1a_horizons's own `isinstance(plan, SlidingWindowPlan)`)
# for any other test file that later receives a plan built under the
# other name in the same pytest process. This file previously had no
# sys.path entry for src/open_vocabulary_segmentation at all, which is
# exactly why it used the dotted alias -- the fix is to set up the same
# path entry those other files already use, not to keep the alias.
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

from src.dfp_phase1a_final_rescale_pullback import (
    DfpFinalResolutionContract,
    DfpPhase1aFinalRescalePullbackError,
    apply_final_rescale_adjoint,
    apply_final_rescale_forward,
    apply_final_window_stitching_pullback,
    build_final_stitching_pullback_plan,
    component_source_mass_float64,
    compute_window_final_soft_memberships,
    iter_image_final_soft_memberships,
)
from src.dfp_phase1a_stitching_pullback import (
    DfpPullbackContract,
    accumulate_membership_column_sums,
    apply_window_stitching_forward,
    build_stitching_pullback_plan,
    validate_conservation,
)
from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize  # noqa: E402

CONTRACT_TRUE_AC = DfpFinalResolutionContract(
    align_corners=True, antialias=False, interpolation_mode="bilinear",
    source_relative_path="src/open_vocabulary_segmentation/models/dinotext/cover_dr/matched_power_evaluator.py",
    source_sha256="0" * 64, source_function="finalize_prediction",
)
CONTRACT_FALSE_AC = DfpFinalResolutionContract(
    align_corners=False, antialias=False, interpolation_mode="bilinear",
    source_relative_path="x", source_sha256="0" * 64, source_function="finalize_prediction",
)


# ---------------------------------------------------------------------------
# 1. R forward matches the existing evaluator exactly.
# ---------------------------------------------------------------------------


def test_r_forward_matches_matched_power_evaluator_finalize_prediction():
    # Bare path (matches every DFP Step 3+ module's own import convention,
    # e.g. dfp_phase1a_horizon_stitching's "from models.dinotext.cover_dr...
    # import ..."): importing the SAME file under its "src.open_vocabulary_
    # segmentation.models...."-prefixed identity in a process that has
    # already imported it bare (as this test file's own
    # dfp_phase1a_stitching_pullback -> dfp_phase1a_horizon_stitching import
    # chain does) double-registers every mmcv MODELS entry point and raises
    # KeyError -- so this file, like the execution driver, never uses the
    # prefixed path.
    from models.dinotext.cover_dr.matched_power_evaluator import finalize_prediction

    torch.manual_seed(0)
    class_count = 5
    stitched = torch.randn(1, class_count, 20, 17, dtype=torch.float64)
    img_meta = {"img_shape": (20, 17, 3), "ori_shape": (33, 11, 3)}

    label_via_evaluator = finalize_prediction(stitched, img_meta, align_corners=True)
    scores_via_ours = apply_final_rescale_forward(stitched, img_shape=(20, 17), ori_shape=(33, 11), contract=CONTRACT_TRUE_AC)
    label_via_ours = scores_via_ours.argmax(dim=1)
    assert torch.equal(label_via_evaluator, label_via_ours)
    # And the underlying rescaled scores are bit-identical to what
    # finalize_prediction computes internally (reconstructed by mirroring
    # its own crop+interpolate lines, since it never returns the
    # pre-argmax field itself).
    import torch.nn.functional as F

    expected = F.interpolate(stitched[:, :, :20, :17], size=(33, 11), mode="bilinear", align_corners=True)
    assert torch.equal(expected, scores_via_ours)


def test_r_forward_padding_crop_removal():
    """stitched larger than img_shape (sliding-window padding) is cropped
    before resize -- never resized directly from the padded extent."""
    torch.manual_seed(1)
    stitched = torch.randn(1, 3, 30, 30, dtype=torch.float64)  # padded canvas
    img_shape = (24, 21)  # the real (unpadded) working resolution
    ori_shape = (24, 21)  # identity resize once cropped
    out = apply_final_rescale_forward(stitched, img_shape=img_shape, ori_shape=ori_shape, contract=CONTRACT_TRUE_AC)
    assert torch.allclose(out, stitched[:, :, :24, :21], atol=1e-10)


# ---------------------------------------------------------------------------
# 2. R^T matches the transpose of an explicit dense matrix.
# ---------------------------------------------------------------------------


def _explicit_r_matrix(*, stitched_shape, img_shape, ori_shape, contract) -> torch.Tensor:
    """Build R's exact dense matrix (single channel) by probing with
    standard basis vectors -- independent of apply_final_rescale_adjoint,
    used only to verify the adjoint against ground truth."""
    sh, sw = stitched_shape
    oh, ow = ori_shape
    n_in = sh * sw
    n_out = oh * ow
    columns = []
    for idx in range(n_in):
        e = torch.zeros(1, 1, sh, sw, dtype=torch.float64)
        e.view(-1)[idx] = 1.0
        out = apply_final_rescale_forward(e, img_shape=img_shape, ori_shape=ori_shape, contract=contract)
        columns.append(out.reshape(-1))
    return torch.stack(columns, dim=1)  # [n_out, n_in]


@pytest.mark.parametrize("align_corners", [True, False])
def test_r_transpose_matches_explicit_matrix(align_corners):
    contract = CONTRACT_TRUE_AC if align_corners else CONTRACT_FALSE_AC
    stitched_shape, img_shape, ori_shape = (5, 4), (5, 4), (7, 3)
    R = _explicit_r_matrix(stitched_shape=stitched_shape, img_shape=img_shape, ori_shape=ori_shape, contract=contract)

    torch.manual_seed(2)
    u = torch.randn(1, *ori_shape, dtype=torch.float64)
    expected = (R.T @ u.reshape(-1)).reshape(1, *stitched_shape)
    observed = apply_final_rescale_adjoint(u, stitched_shape=stitched_shape, img_shape=img_shape, ori_shape=ori_shape, contract=contract)
    torch.testing.assert_close(observed, expected, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 3. Adjoint identity <Rz,u> == <z,R^Tu>, across a fixture matrix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stitched_shape,img_shape,ori_shape,align_corners",
    [
        ((10, 12), (8, 9), (13, 7), True),     # non-integer ratio, mixed up/down, padding present
        ((8, 9), (8, 9), (8, 9), True),        # identity (no crop, no resize)
        ((6, 6), (6, 6), (1, 1), True),        # collapse to a single pixel
        ((6, 6), (6, 6), (1, 1), False),       # single pixel, align_corners=False
        ((9, 5), (9, 5), (17, 13), True),      # upsample both axes
        ((17, 13), (17, 13), (5, 9), True),    # downsample rows, near-identity cols
        ((6, 20), (6, 20), (6, 20), False),    # non-square, align_corners=False
    ],
)
def test_adjoint_identity_holds(stitched_shape, img_shape, ori_shape, align_corners):
    contract = CONTRACT_TRUE_AC if align_corners else CONTRACT_FALSE_AC
    torch.manual_seed(3)
    K = 2
    z = torch.randn(1, K, *stitched_shape, dtype=torch.float64)
    u = torch.randn(K, *ori_shape, dtype=torch.float64)
    Rz = apply_final_rescale_forward(z, img_shape=img_shape, ori_shape=ori_shape, contract=contract)
    RTu = apply_final_rescale_adjoint(u, stitched_shape=stitched_shape, img_shape=img_shape, ori_shape=ori_shape, contract=contract)
    lhs = (Rz.squeeze(0) * u).sum()
    rhs = (z.squeeze(0) * RTu).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-9, rtol=1e-9)


def test_float32_adjoint_identity_holds_to_float32_tolerance():
    contract = CONTRACT_TRUE_AC
    torch.manual_seed(4)
    stitched_shape, img_shape, ori_shape = (10, 12), (8, 9), (13, 7)
    z = torch.randn(1, 1, *stitched_shape, dtype=torch.float32)
    u = torch.randn(1, *ori_shape, dtype=torch.float32)
    Rz = apply_final_rescale_forward(z, img_shape=img_shape, ori_shape=ori_shape, contract=contract)
    RTu = apply_final_rescale_adjoint(u, stitched_shape=stitched_shape, img_shape=img_shape, ori_shape=ori_shape, contract=contract)
    lhs = (Rz.squeeze(0) * u).sum()
    rhs = (z.squeeze(0) * RTu).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Real small window plan fixture, reused across the composed-identity and
# soft-membership tests below.
# ---------------------------------------------------------------------------


def _small_pullback_plan(*, image_size):
    contract = DfpPullbackContract(crop=(4, 4), stride=(2, 2), patch_grid=(2, 2), patch_size=(2, 2), align_corners=True, connectivity=8)
    plan = build_stitching_pullback_plan(image_size, contract=contract, dtype=torch.float64)
    return plan


def test_composed_identity_matches_window_forward_and_pullback():
    """<A_final z, u> == <z, A_final^T u> for one window's patch-source
    vector z, composing Step 7's own (unmodified) apply_window_stitching_
    forward with apply_final_rescale_forward for the forward direction,
    against apply_final_window_stitching_pullback for the adjoint."""
    image_size = (6, 8)  # -> multiple overlapping windows under crop=4,stride=2
    plan = _small_pullback_plan(image_size=image_size)
    assert len(plan.plan.windows) >= 2  # exercise real overlap, not a degenerate single-window case

    img_shape = image_size
    ori_shape = (11, 5)  # deliberately different resolution
    torch.manual_seed(5)
    node_count = plan.patch_grid[0] * plan.patch_grid[1]

    for window_index in range(len(plan.plan.windows)):
        z = torch.randn(1, node_count, dtype=torch.float64)  # [K=1, N]
        window_contribution = apply_window_stitching_forward(z, window_plan=plan, window_index=window_index)  # [1, crop_h, crop_w]
        window = plan.plan.windows[window_index]
        rows, cols = window.accumulation_slice
        full = torch.zeros(1, 1, *image_size, dtype=torch.float64)
        full[:, :, rows, cols] = window_contribution.unsqueeze(0)
        A_final_z = apply_final_rescale_forward(full, img_shape=img_shape, ori_shape=ori_shape, contract=CONTRACT_TRUE_AC)  # [1,1,oh,ow]

        u = torch.randn(1, *ori_shape, dtype=torch.float64)
        A_final_T_u = apply_final_window_stitching_pullback(
            u, window_plan=plan, window_index=window_index, img_shape=img_shape, ori_shape=ori_shape, contract=CONTRACT_TRUE_AC,
        )  # [1, N]

        lhs = (A_final_z.squeeze(0).squeeze(0) * u.squeeze(0)).sum()
        rhs = (z.squeeze(0) * A_final_T_u.squeeze(0)).sum()
        torch.testing.assert_close(lhs, rhs, atol=1e-8, rtol=1e-8, msg=f"composed adjoint identity failed for window {window_index}")


# ---------------------------------------------------------------------------
# 5/6. Soft-membership partition of unity and source-mass conservation.
# ---------------------------------------------------------------------------


def test_partition_of_unity_and_source_mass_conservation():
    image_size = (6, 8)
    plan = _small_pullback_plan(image_size=image_size)
    img_shape = image_size
    ori_shape = (9, 7)
    final_plan = build_final_stitching_pullback_plan(plan, img_shape=img_shape, ori_shape=ori_shape, contract=CONTRACT_TRUE_AC)

    rng = np.random.default_rng(6)
    component_count = 5
    component_ids = rng.integers(0, component_count, size=ori_shape).astype(np.int64)
    # ensure every id 0..component_count-1 actually appears (no silently-empty component)
    flat = component_ids.reshape(-1)
    for r in range(component_count):
        flat[r] = r
    component_ids = flat.reshape(ori_shape)

    node_count = plan.patch_grid[0] * plan.patch_grid[1]
    window_count = len(plan.plan.windows)
    column_sums = {w: None for w in range(window_count)}
    total_mass = torch.zeros(component_count, dtype=torch.float64)

    for start, chunk in iter_image_final_soft_memberships(component_ids, component_count, final_plan=final_plan, component_chunk_size=2):
        for window_index, membership in chunk.items():
            assert torch.isfinite(membership).all()
            assert (membership >= -1e-10).all(), "soft membership must be non-negative"
            column_sums[window_index] = accumulate_membership_column_sums(column_sums[window_index], membership)
        mass = component_source_mass_float64(chunk)
        total_mass[start : start + mass.shape[0]] += mass

    for window_index in range(window_count):
        result = validate_conservation(column_sums[window_index], node_count=node_count)
        assert result["max_conservation_residual"] < 1e-8, f"window {window_index} partition-of-unity residual too large: {result}"

    # sum_r m_r == sum_{w,j} q_final[w,j] == sum_r sum_{w,j} M[r,w,j] == sum_{w,j} 1
    expected_total = sum(float(q.sum()) for q in final_plan.per_window_q_final) / 1.0
    # (q_final IS the per-(w,j) normalizer -- total mass conservation is
    # sum_r m_r == window_count * node_count exactly, since every (w,j)
    # column sums to exactly 1 across components.)
    assert abs(float(total_mass.sum()) - window_count * node_count) < 1e-6


def test_component_ids_shape_must_equal_ori_shape_not_window_plan_image_size():
    image_size = (6, 8)
    plan = _small_pullback_plan(image_size=image_size)
    ori_shape = (9, 7)  # deliberately != image_size
    final_plan = build_final_stitching_pullback_plan(plan, img_shape=image_size, ori_shape=ori_shape, contract=CONTRACT_TRUE_AC)
    wrong_shape_ids = np.zeros(image_size, dtype=np.int64)  # window-plan-shaped, NOT ori-shaped -- must be rejected
    with pytest.raises(DfpPhase1aFinalRescalePullbackError, match="ori_shape"):
        compute_window_final_soft_memberships(wrong_shape_ids, 0, 1, final_plan=final_plan, window_index=0)


# ---------------------------------------------------------------------------
# 9/10. No nearest-neighbor resampling anywhere in this module (static
# poison check over the module's own source).
# ---------------------------------------------------------------------------


def test_no_nearest_neighbor_interpolation_in_module_source():
    """Poison check over actual CODE (every string literal and every
    F.interpolate(..., mode=...) call site), never over the module's own
    prose docstrings/comments (which legitimately discuss, by name, the
    nearest-neighbor resampling this module refuses to perform)."""
    source = (ROOT / "src" / "dfp_phase1a_final_rescale_pullback.py").read_text()
    tree = ast.parse(source)

    docstring_constants: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstring_constants.add(id(first.value))

    offenders = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and "nearest" in node.value.lower() and id(node) not in docstring_constants
    ]
    assert not offenders, f"the string 'nearest' must never appear in this module's CODE, only in docstrings (found: {offenders})"

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "mode":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                assert node.value.value != "nearest", "nearest-neighbor interpolation must never appear in this module"


# ---------------------------------------------------------------------------
# Failure behavior.
# ---------------------------------------------------------------------------


def test_rejects_non_finite_input():
    stitched = torch.full((1, 1, 4, 4), float("nan"), dtype=torch.float64)
    with pytest.raises(DfpPhase1aFinalRescalePullbackError):
        apply_final_rescale_forward(stitched, img_shape=(4, 4), ori_shape=(4, 4), contract=CONTRACT_TRUE_AC)


def test_rejects_wrong_rank():
    stitched = torch.zeros(1, 4, 4, dtype=torch.float64)  # missing batch/channel structure
    with pytest.raises(DfpPhase1aFinalRescalePullbackError):
        apply_final_rescale_forward(stitched, img_shape=(4, 4), ori_shape=(4, 4), contract=CONTRACT_TRUE_AC)


def test_rejects_img_shape_larger_than_stitched():
    stitched = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    with pytest.raises(DfpPhase1aFinalRescalePullbackError):
        apply_final_rescale_forward(stitched, img_shape=(5, 4), ori_shape=(4, 4), contract=CONTRACT_TRUE_AC)


def test_rejects_wrong_window_plan_image_size():
    image_size = (6, 8)
    plan = _small_pullback_plan(image_size=image_size)
    with pytest.raises(DfpPhase1aFinalRescalePullbackError, match="image_size"):
        apply_final_window_stitching_pullback(
            torch.zeros(1, 3, 3, dtype=torch.float64), window_plan=plan, window_index=0,
            img_shape=(99, 99), ori_shape=(3, 3), contract=CONTRACT_TRUE_AC,
        )


# ---------------------------------------------------------------------------
# SlidingWindowPlan import-identity regression: this file previously
# imported SlidingWindowPlan via the non-canonical dotted alias
# `src.open_vocabulary_segmentation.segmentation.evaluation.sliding_
# window_geometry` (fixed above to the canonical bare `segmentation.
# evaluation.sliding_window_geometry`, matching production code and
# every other DFP test file). Importing the SAME physical module under
# two different qualified names makes Python instantiate it TWICE,
# producing two genuinely distinct class objects that share a name --
# `isinstance(plan, SlidingWindowPlan)` then silently disagrees
# depending on which import path built `plan`, exactly the failure mode
# an earlier independent verification reproduced: six
# test_dfp_phase1a_execute.py slow tests failed with
# "DfpPhase1aHorizonStitchingError: plan must be a SlidingWindowPlan"
# ONLY when the full DFP+DFP1B suite ran as one combined pytest process
# (never in isolation), traced to exactly this dual-import split.
# ---------------------------------------------------------------------------


def test_sliding_window_plan_single_canonical_import_source():
    """Static source audit: no DFP-scoped source, test, or diagnostics
    file may import SlidingWindowPlan (or anything else from that
    module) via the dotted `src.open_vocabulary_segmentation.
    segmentation...` alias -- only the canonical bare `segmentation.
    evaluation.sliding_window_geometry` path. A permanent regression
    guard for the exact bug described above, independent of collection
    order or which test happens to run first."""
    forbidden = "src.open_vocabulary_segmentation.segmentation.evaluation.sliding_window_geometry"
    forbidden_slash = "src/open_vocabulary_segmentation/segmentation/evaluation/sliding_window_geometry"

    def _non_docstring_string_hits(source: str, path) -> list[str]:
        """AST-precise scan: forbidden-substring hits inside CODE string
        literals (e.g. a dynamically-generated subprocess script, or a
        forbidden import statement itself) only -- never inside a
        module/function/class docstring, which legitimately DISCUSSES
        this exact historical bug and its fix in prose (this very test's
        own docstring, and this file's module docstring, included).
        Mirrors test_dfp_phase1a_execute.py's own AST-precise deleted-
        path check, which hit and fixed the identical false-positive
        pitfall for a different forbidden-string check."""
        tree = ast.parse(source, filename=str(path))
        docstring_ids: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    docstring_ids.add(id(first.value))
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_ids:
                normalized = node.value.replace("/", ".")
                if forbidden in node.value or forbidden_slash.replace("/", ".") in normalized:
                    hits.append(node.value)
        return hits

    scanned = 0
    for pattern in ("dfp_phase1a_*.py", "dfp_phase1b_*.py"):
        for path in sorted((ROOT / "src").glob(pattern)):
            scanned += 1
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module_name = getattr(node, "module", None) or ""
                    names = [module_name] + [getattr(a, "name", "") for a in node.names]
                    assert not any(forbidden in n for n in names), f"{path}: imports SlidingWindowPlan via the non-canonical dotted alias ({module_name!r})"
            assert not _non_docstring_string_hits(source, path), f"{path}: contains the non-canonical dotted alias in a CODE string literal"
    this_file = Path(__file__).resolve()
    for pattern in ("test_dfp_phase1a_*.py", "test_dfp_phase1b_*.py"):
        for path in sorted((ROOT / "tests").glob(pattern)):
            scanned += 1
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module_name = getattr(node, "module", None) or ""
                    names = [module_name] + [getattr(a, "name", "") for a in node.names]
                    assert not any(forbidden in n for n in names), f"{path}: imports via the non-canonical dotted alias ({module_name!r})"
            if path.resolve() == this_file:
                # This file's own regression-guard code (this function)
                # necessarily assigns the forbidden strings themselves to
                # `forbidden`/`forbidden_slash` above -- a legitimate,
                # singular, self-referential exception; every OTHER file
                # (including this one's own production import statements,
                # already checked above) is still fully scanned.
                continue
            # also catch the alias if it is ever constructed as a plain
            # CODE string (e.g. inside a dynamically-generated subprocess
            # script literal) rather than a real import statement --
            # never a docstring's own explanatory prose.
            hits = _non_docstring_string_hits(source, path)
            assert not hits, f"{path}: contains the non-canonical dotted sliding_window_geometry alias in a CODE string literal: {hits}"
    assert scanned > 10, "the glob patterns above matched suspiciously few files -- audit may be silently skipping the DFP tree"


def test_sliding_window_plan_not_double_imported_in_this_session():
    """Dynamic collection-time check: by the time this test runs, pytest
    has already imported every collected test module in this session. If
    any DFP test file still used the dotted alias, its module would
    already be sitting in sys.modules under that separate key -- assert
    it is not."""
    dotted_key = "src.open_vocabulary_segmentation.segmentation.evaluation.sliding_window_geometry"
    assert dotted_key not in sys.modules, (
        f"{dotted_key} is present in sys.modules during this test session -- some file imported "
        "SlidingWindowPlan under the non-canonical dotted alias, which creates a second, distinct class object"
    )
    # Positive control: the canonical bare-path module IS present (this
    # file itself imports it at module scope, above).
    assert "segmentation.evaluation.sliding_window_geometry" in sys.modules


def test_sliding_window_plan_canonical_identity_survives_full_tree_collection():
    """Direct regression proof for the repository-wide collection-time
    collision this repair closes: tests/test_image_window_cache.py,
    tests/test_matched_power_evaluator.py, tests/test_trust_centrality_
    harness.py, tests/test_t4_audit.py, and tests/test_trust_centrality_
    diagnostics.py each used to load sliding_window_geometry.py via
    ``importlib.util.spec_from_file_location`` under the CANONICAL
    module name ``segmentation.evaluation.sliding_window_geometry`` (to
    avoid segmentation.evaluation/__init__.py's own real mmcv/mmseg-
    dependent imports), unconditionally overwriting sys.modules's
    canonical entry with a freshly-executed, class-identity-distinct
    module at collection time -- silently invalidating whichever
    SlidingWindowPlan class production code (src/dfp_phase1a_horizon_
    stitching.py, imported earlier) had already captured into its own
    namespace, and breaking its own isinstance(plan, SlidingWindowPlan)
    check for the rest of the pytest session. All five now load under a
    private, per-file test-only name instead (never touching the
    canonical sys.modules key), and this test's own assertions hold
    regardless of collection order or how much of the wider test tree
    has already been collected/imported by the time it runs."""
    from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan as CanonicalPlan
    from src.dfp_phase1a_horizon_stitching import SlidingWindowPlan as CapturedPlan

    assert CanonicalPlan is CapturedPlan, (
        "src.dfp_phase1a_horizon_stitching's own captured SlidingWindowPlan is no longer identical to a fresh "
        "canonical import -- something replaced the canonical sys.modules entry after horizon_stitching's own import"
    )
    assert sys.modules["segmentation.evaluation.sliding_window_geometry"].SlidingWindowPlan is CanonicalPlan
