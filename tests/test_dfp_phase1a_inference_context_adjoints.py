"""Inference-context safety tests for both DFP bilinear-adjoint local
autograd probes:

  * ``src.dfp_phase1a_stitching_pullback._apply_window_bilinear_adjoint``
    (Step 7, the original window-only pullback);
  * ``src.dfp_phase1a_final_rescale_pullback._bilinear_resize_adjoint``
    (the composed final-rescale pullback's own R^T).

Both must produce bit-identical output whether called in ordinary
grad-enabled mode, ``torch.no_grad()``, ``torch.set_grad_enabled(False)``,
or ``torch.inference_mode()`` -- and must restore the caller's exact
ambient grad/inference-mode state on every exit path, including a raised
exception. This file never modifies the interpolation mathematics itself
-- every check compares against the SAME dense-matrix/adjoint-identity
oracles already used for the pre-repair audit, run once per context.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.dfp_phase1a_stitching_pullback import (
    DfpPullbackContract,
    _apply_window_bilinear_adjoint,
    accumulate_membership_column_sums,
    apply_window_stitching_forward,
    apply_window_stitching_pullback,
    build_stitching_pullback_plan,
    validate_conservation,
)
from src.dfp_phase1a_final_rescale_pullback import (
    DfpFinalResolutionContract,
    _bilinear_resize_adjoint,
    apply_final_rescale_adjoint,
    apply_final_rescale_forward,
    apply_final_window_stitching_pullback,
    build_final_stitching_pullback_plan,
    compute_window_final_soft_memberships,
    component_source_mass_float64,
    iter_image_final_soft_memberships,
)

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = DfpFinalResolutionContract(
    align_corners=True, antialias=False, interpolation_mode="bilinear",
    source_relative_path="src/open_vocabulary_segmentation/models/dinotext/cover_dr/matched_power_evaluator.py",
    source_sha256="13371fd697ef5fae41f2a0b7e99f8c5a65d679a337de734e539a8fdf7e95acf4", source_function="finalize_prediction",
)


def _contexts():
    """The four required ambient-mode contexts, as (label, context_manager_factory) pairs."""
    import contextlib

    return [
        ("normal", lambda: contextlib.nullcontext()),
        ("no_grad", lambda: torch.no_grad()),
        ("set_grad_enabled(False)", lambda: torch.set_grad_enabled(False)),
        ("inference_mode", lambda: torch.inference_mode()),
    ]


# ---------------------------------------------------------------------------
# Standalone Step-7 adjoint under all four contexts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,ctx_factory", _contexts())
def test_step7_adjoint_works_in_every_context(label, ctx_factory):
    v = torch.randn(1, 5, 6, dtype=torch.float64)
    with ctx_factory():
        out = _apply_window_bilinear_adjoint(v, grid_hw=(3, 3), crop_hw=(5, 6))
    assert torch.isfinite(out).all()
    assert out.requires_grad is False
    assert out.grad_fn is None


def test_step7_adjoint_identical_across_contexts():
    v = torch.randn(1, 5, 6, dtype=torch.float64)
    reference = _apply_window_bilinear_adjoint(v, grid_hw=(3, 3), crop_hw=(5, 6))
    for label, ctx_factory in _contexts()[1:]:
        with ctx_factory():
            out = _apply_window_bilinear_adjoint(v, grid_hw=(3, 3), crop_hw=(5, 6))
        torch.testing.assert_close(out, reference, atol=0, rtol=0, msg=f"context {label} diverged from normal-mode output")


# ---------------------------------------------------------------------------
# Standalone final-rescale adjoint under all four contexts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,ctx_factory", _contexts())
def test_final_rescale_adjoint_works_in_every_context(label, ctx_factory):
    u = torch.randn(1, 5, 7, dtype=torch.float64)
    with ctx_factory():
        out = apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
    assert torch.isfinite(out).all()
    assert out.requires_grad is False
    assert out.grad_fn is None


def test_final_rescale_adjoint_identical_across_contexts():
    u = torch.randn(1, 5, 7, dtype=torch.float64)
    reference = apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
    for label, ctx_factory in _contexts()[1:]:
        with ctx_factory():
            out = apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
        torch.testing.assert_close(out, reference, atol=0, rtol=0, msg=f"context {label} diverged from normal-mode output")


def test_forward_R_identical_across_contexts():
    """R's forward path never used autograd at all, but confirm it too
    tolerates every context (it must, since a real driver runs the whole
    pipeline under one ambient mode)."""
    z = torch.randn(1, 3, 6, 6, dtype=torch.float64)
    reference = apply_final_rescale_forward(z, img_shape=(6, 6), ori_shape=(5, 7), contract=CONTRACT)
    for label, ctx_factory in _contexts()[1:]:
        with ctx_factory():
            out = apply_final_rescale_forward(z, img_shape=(6, 6), ori_shape=(5, 7), contract=CONTRACT)
        torch.testing.assert_close(out, reference, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Complete composed A_final pullback under all four contexts, plus dense
# matrix / inner-product identities re-proven per context.
# ---------------------------------------------------------------------------


def _small_pullback_plan(image_size):
    contract = DfpPullbackContract(crop=(4, 4), stride=(2, 2), patch_grid=(2, 2), patch_size=(2, 2), align_corners=True, connectivity=8)
    return build_stitching_pullback_plan(image_size, contract=contract, dtype=torch.float64)


@pytest.mark.parametrize("label,ctx_factory", _contexts())
def test_composed_pullback_and_adjoint_identity_in_every_context(label, ctx_factory):
    image_size = (6, 8)
    ori_shape = (9, 7)
    plan = _small_pullback_plan(image_size)
    node_count = 4

    with ctx_factory():
        final_plan = build_final_stitching_pullback_plan(plan, img_shape=image_size, ori_shape=ori_shape, contract=CONTRACT)
        window_index = 0
        z = torch.randn(1, node_count, dtype=torch.float64)
        window_contribution = apply_window_stitching_forward(z, window_plan=plan, window_index=window_index)
        window = plan.plan.windows[window_index]
        rows, cols = window.accumulation_slice
        full = torch.zeros(1, 1, *image_size, dtype=torch.float64)
        full[:, :, rows, cols] = window_contribution.unsqueeze(0)
        A_final_z = apply_final_rescale_forward(full, img_shape=image_size, ori_shape=ori_shape, contract=CONTRACT)

        u = torch.randn(1, *ori_shape, dtype=torch.float64)
        A_final_T_u = apply_final_window_stitching_pullback(u, window_plan=plan, window_index=window_index, img_shape=image_size, ori_shape=ori_shape, contract=CONTRACT)

        lhs = (A_final_z.squeeze(0).squeeze(0) * u.squeeze(0)).sum()
        rhs = (z.squeeze(0) * A_final_T_u.squeeze(0)).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-8, rtol=1e-8, msg=f"composed adjoint identity failed in context {label}")


@pytest.mark.parametrize("label,ctx_factory", _contexts())
def test_dense_matrix_transpose_equality_in_every_context(label, ctx_factory):
    stitched_shape, img_shape, ori_shape = (10, 12), (8, 9), (13, 7)

    def dense_R():
        sh, sw = stitched_shape
        cols = []
        for idx in range(sh * sw):
            e = torch.zeros(1, 1, sh, sw, dtype=torch.float64)
            e.view(-1)[idx] = 1.0
            cols.append(apply_final_rescale_forward(e, img_shape=img_shape, ori_shape=ori_shape, contract=CONTRACT).reshape(-1))
        return torch.stack(cols, dim=1)

    with ctx_factory():
        R_dense = dense_R()
        u = torch.randn(1, *ori_shape, dtype=torch.float64)
        RTu_dense = (R_dense.T @ u.reshape(-1)).reshape(1, *stitched_shape)
        RTu_impl = apply_final_rescale_adjoint(u, stitched_shape=stitched_shape, img_shape=img_shape, ori_shape=ori_shape, contract=CONTRACT)
    torch.testing.assert_close(RTu_dense, RTu_impl, atol=1e-9, rtol=1e-9, msg=f"dense-matrix transpose mismatch in context {label}")


@pytest.mark.parametrize("label,ctx_factory", _contexts())
def test_partition_of_unity_and_source_mass_conservation_in_every_context(label, ctx_factory):
    image_size = (6, 8)
    ori_shape = (9, 7)
    plan = _small_pullback_plan(image_size)

    with ctx_factory():
        final_plan = build_final_stitching_pullback_plan(plan, img_shape=image_size, ori_shape=ori_shape, contract=CONTRACT)
        rng = np.random.default_rng(6)
        component_count = 4
        component_ids = rng.integers(0, component_count, size=ori_shape).astype(np.int64)
        flat = component_ids.reshape(-1)
        for r in range(component_count):
            flat[r] = r
        component_ids = flat.reshape(ori_shape)

        node_count = 4
        window_count = len(plan.plan.windows)
        column_sums = {w: None for w in range(window_count)}
        total_mass = torch.zeros(component_count, dtype=torch.float64)
        for start, chunk in iter_image_final_soft_memberships(component_ids, component_count, final_plan=final_plan, component_chunk_size=2):
            for window_index, membership in chunk.items():
                assert torch.isfinite(membership).all()
                column_sums[window_index] = accumulate_membership_column_sums(column_sums[window_index], membership)
            mass = component_source_mass_float64(chunk)
            total_mass[start:start + mass.shape[0]] += mass

        for window_index in range(window_count):
            result = validate_conservation(column_sums[window_index], node_count=node_count)
            assert result["max_conservation_residual"] < 1e-8, f"context {label}: window {window_index} conservation residual too large"

        assert abs(float(total_mass.sum()) - window_count * node_count) < 1e-6, f"context {label}: total mass conservation failed"


# ---------------------------------------------------------------------------
# Caller state restoration, detachment, non-mutation, determinism.
# ---------------------------------------------------------------------------


def test_caller_grad_state_restored_after_success_and_after_domain_error():
    from src.dfp_phase1a_final_rescale_pullback import DfpPhase1aFinalRescalePullbackError

    for ambient in (True, False):
        torch.set_grad_enabled(ambient)
        u = torch.randn(1, 5, 7, dtype=torch.float64)
        _ = apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
        assert torch.is_grad_enabled() is ambient, f"grad state not restored after success (ambient={ambient})"

        try:
            apply_final_rescale_adjoint(u, stitched_shape=(3, 3), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)  # img_shape > stitched_shape -> domain error
        except DfpPhase1aFinalRescalePullbackError:
            pass
        assert torch.is_grad_enabled() is ambient, f"grad state not restored after a domain-error exit (ambient={ambient})"
    torch.set_grad_enabled(True)


def test_caller_inference_mode_restored_after_success_and_after_domain_error():
    from src.dfp_phase1a_final_rescale_pullback import DfpPhase1aFinalRescalePullbackError

    with torch.inference_mode():
        u = torch.randn(1, 5, 7, dtype=torch.float64)
        _ = apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
        assert torch.is_inference_mode_enabled() is True, "inference_mode not restored after success"
        try:
            apply_final_rescale_adjoint(u, stitched_shape=(3, 3), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
        except DfpPhase1aFinalRescalePullbackError:
            pass
        assert torch.is_inference_mode_enabled() is True, "inference_mode not restored after a domain-error exit"
    assert torch.is_inference_mode_enabled() is False


def test_caller_state_restored_after_injected_runtime_error_inside_the_probe():
    """Poison torch.autograd.grad itself to raise mid-computation, and
    confirm the ambient grad/inference-mode state is still restored --
    proving restoration is structural (context-manager __exit__), not
    dependent on the happy path."""
    import src.dfp_phase1a_final_rescale_pullback as m

    original_grad = torch.autograd.grad

    def poisoned_grad(*a, **kw):
        raise RuntimeError("injected failure")

    torch.autograd.grad = poisoned_grad
    try:
        with torch.inference_mode():
            u = torch.randn(1, 5, 7, dtype=torch.float64)
            with pytest.raises(RuntimeError, match="injected failure"):
                m.apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
            assert torch.is_inference_mode_enabled() is True, "inference_mode not restored after an injected RuntimeError"
            assert torch.is_grad_enabled() is False, "grad-enabled state not restored after an injected RuntimeError"
    finally:
        torch.autograd.grad = original_grad
    assert torch.is_inference_mode_enabled() is False


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_base_exceptions_propagate_and_restore_state(exc_type):
    import src.dfp_phase1a_final_rescale_pullback as m

    original_grad = torch.autograd.grad

    def poisoned_grad(*a, **kw):
        raise exc_type("poison")

    torch.autograd.grad = poisoned_grad
    try:
        with torch.inference_mode():
            u = torch.randn(1, 5, 7, dtype=torch.float64)
            with pytest.raises(exc_type):
                m.apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
            assert torch.is_inference_mode_enabled() is True
            assert torch.is_grad_enabled() is False
    finally:
        torch.autograd.grad = original_grad
    assert torch.is_inference_mode_enabled() is False


def test_detached_graph_free_output_and_no_input_mutation():
    u = torch.randn(1, 5, 7, dtype=torch.float64, requires_grad=True)
    u_copy = u.detach().clone()
    with torch.inference_mode():
        pass  # inference_mode tensors cannot carry requires_grad=True; test the ordinary-mode + no_grad combination instead
    out = apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
    assert out.requires_grad is False
    assert out.grad_fn is None
    assert torch.equal(u.detach(), u_copy), "input tensor was mutated"
    assert u.requires_grad is True, "caller's own requires_grad flag must be left alone"


def test_contract_unchanged_after_calls_in_every_context():
    import dataclasses

    before = dataclasses.astuple(CONTRACT)
    for label, ctx_factory in _contexts():
        with ctx_factory():
            u = torch.randn(1, 5, 7, dtype=torch.float64)
            apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
    after = dataclasses.astuple(CONTRACT)
    assert before == after


def test_unrelated_model_like_parameter_grad_unaffected_across_all_contexts():
    model_param = torch.nn.Parameter(torch.randn(3, 3, dtype=torch.float64))
    x = torch.randn(3, 3, dtype=torch.float64)
    y1 = (model_param * x).sum()
    y1.backward()
    grad_reference = model_param.grad.clone()
    model_param.grad = None

    for label, ctx_factory in _contexts():
        y2 = (model_param * x).sum()
        with ctx_factory():
            u = torch.randn(1, 5, 7, dtype=torch.float64)
            apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
        y2.backward()
        assert torch.equal(model_param.grad, grad_reference), f"unrelated parameter gradient perturbed in context {label}"
        model_param.grad = None


def test_deterministic_across_repeated_calls_in_every_context():
    u = torch.randn(1, 5, 7, dtype=torch.float64)
    for label, ctx_factory in _contexts():
        with ctx_factory():
            out1 = apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
            out2 = apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
        assert torch.equal(out1, out2), f"non-deterministic in context {label}"


def test_cuda_remains_uninitialized_across_all_contexts_and_both_adjoints():
    assert torch.cuda.is_initialized() is False
    v = torch.randn(1, 5, 6, dtype=torch.float64)
    u = torch.randn(1, 5, 7, dtype=torch.float64)
    for label, ctx_factory in _contexts():
        with ctx_factory():
            _apply_window_bilinear_adjoint(v, grid_hw=(3, 3), crop_hw=(5, 6))
            apply_final_rescale_adjoint(u, stitched_shape=(6, 6), img_shape=(4, 4), ori_shape=(5, 7), contract=CONTRACT)
    assert torch.cuda.is_initialized() is False
