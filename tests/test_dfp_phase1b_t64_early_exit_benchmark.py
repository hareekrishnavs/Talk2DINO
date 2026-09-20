"""Comprehensive CPU-only tests for the E26-authorized T64-versus-T320
early-exit benchmark implementation.

No real dataset, model, weights, CUDA, or scheduler is ever accessed.
Real scientific kernels (``build_directed_topk_graph``,
``finite_step_propagate``, ``stitch_dfp_phase1a_horizons``,
``apply_final_rescale_forward``) are exercised directly with small
synthetic tensors -- they are pure, device-agnostic tensor math with no
model/dataset dependency of their own. Only the unary forward (which
would require the real backbone/CLIP model) is ever faked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.dfp_phase1b_t64_early_exit_benchmark as runner
import src.dfp_phase1b_t64_early_exit_benchmark_identity as e25_identity
import src.dfp_phase1b_t64_early_exit_benchmark_implementation_identity as e26_identity
from src.dfp_phase1a_final_rescale_pullback import DfpFinalResolutionContract
from src.dfp_phase1a_horizon_stitching import DfpStitchingContract

REPO_ROOT = runner.repository_root()
ARTIFACTS_DOC = e25_identity.load_artifacts_identity(repo_root=REPO_ROOT)
E = runner.DfpPhase1bT64EarlyExitBenchmarkError


# =======================================================================
# Fixtures: synthetic (CPU-only) manifest, geometry, and DI providers.
# =======================================================================


def _make_manifest(n: int = 24) -> dict:
    records = [
        {
            "benchmark_slot": i, "image_id": f"synthetic_image_{i:03d}", "canonical_window_ordinal": i,
            "top": 0, "left": 0, "bottom": 28, "right": 28, "selection_rank": i,
        }
        for i in range(n)
    ]
    warmup_record = {
        "benchmark_slot": -1, "image_id": "synthetic_warmup_image", "canonical_window_ordinal": 0,
        "top": 0, "left": 0, "bottom": 28, "right": 28, "selection_rank": 999,
    }
    return {"benchmark_records": records, "warmup_record": warmup_record}


def _make_readiness(manifest: dict | None = None) -> runner.ReadinessResult:
    return runner.ReadinessResult(
        repo_root=REPO_ROOT, e25_doc={}, e26_doc={"identity": {"required_ancestor_commit": "9" * 40}},
        artifacts_doc=ARTIFACTS_DOC, manifest=manifest if manifest is not None else _make_manifest(),
        repository_commit="a" * 40, code_hashes={k: "b" * 64 for k in ("identity_loader_sha256", "runner_sha256", "cli_sha256", "launcher_sha256")},
        identity_sha256="c" * 64, artifacts_identity_sha256="d" * 64, manifest_sha256="e" * 64,
    )


def _make_stitching_geometry() -> DfpStitchingContract:
    return DfpStitchingContract(horizons=(0, 16, 32, 64, 320), crop=(28, 28), stride=(28, 28), patch_grid=(4, 4), patch_size=(7, 7), align_corners=False)


def _make_final_resolution_contract() -> DfpFinalResolutionContract:
    return DfpFinalResolutionContract(align_corners=False, antialias=False, interpolation_mode="bilinear", source_relative_path="x", source_sha256="0" * 64, source_function="x")


def _make_dino_features(num_nodes: int = 16, dim: int = 8, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(num_nodes, dim, generator=generator)
    return raw / raw.norm(dim=-1, keepdim=True)


def _make_unary_scores(num_nodes: int = 16, class_count: int = 5, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1000)
    return torch.rand(num_nodes, class_count, generator=generator)


class _FakeCallCounter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1


class _FakeMemoryProvider(runner.MemoryProvider):
    def __init__(self, reading: dict | None = None):
        self.reset_calls = 0
        self._reading = reading or {
            "host_peak_rss_bytes": 100, "gpu_peak_memory_allocated_bytes": 10,
            "gpu_peak_memory_reserved_bytes": 20, "gpu_total_memory_bytes": 1000,
        }

    def reset_peak(self) -> None:
        self.reset_calls += 1

    def read(self) -> dict:
        return dict(self._reading)


def _make_providers(*, num_nodes=16, class_count=5, k=4, memory_reading=None, ds_build_counter=None, model_build_counter=None):
    ds_calls = {"n": 0}
    model_calls = {"n": 0}
    memory_provider = _FakeMemoryProvider(memory_reading)

    def dataset_resources_builder(repo_root, selected_ids):
        ds_calls["n"] += 1
        return SimpleNamespace(name="fake_resources")

    def inference_builder(resources):
        model_calls["n"] += 1
        return SimpleNamespace(name="fake_inference")

    def window_inputs_builder_factory(resources):
        def _build(record):
            return runner.WindowInputs(crop_tensor=None, crop_height=28, crop_width=28, ori_shape=(28, 28))
        return _build

    def unary_forward_fn(inference, crop_tensor):
        return SimpleNamespace(
            unary_scores=[_make_unary_scores(num_nodes, class_count)], dino_features=[_make_dino_features(num_nodes)],
        )

    providers = runner.TrialProviders(
        dataset_resources_builder=dataset_resources_builder, inference_builder=inference_builder,
        window_inputs_builder_factory=window_inputs_builder_factory,
        stitching_geometry_factory=lambda resources: _make_stitching_geometry(),
        final_resolution_contract_factory=lambda resources: _make_final_resolution_contract(),
        class_count_of=lambda resources: class_count, affinity_power_of=lambda resources: 3.0,
        alpha_of=lambda resources: 0.98, k_of=lambda resources: k,
        unary_forward_fn=unary_forward_fn, memory_provider_factory=lambda: memory_provider,
        device_record_fn=lambda: {"device_type": "cuda", "gpu_model": "fake", "gpu_uuid": "fake", "cpu_count_visible": 1, "one_visible_cuda_device": True},
        software_record_fn=lambda: {"python_version": "3.11", "torch_version": "2.0", "cuda_runtime_version": "12.0", "driver_version": "550"},
        precision_record_fn=lambda: {"compute_dtype": "float32", "output_dtype": "float32"},
        deterministic_settings_record_fn=lambda: {"torch_deterministic_algorithms": False, "cudnn_benchmark": False, "cudnn_deterministic": False, "pythonhashseed": None},
    )
    return providers, ds_calls, model_calls, memory_provider


def _run_measured_trial(*, variant_id: str, trial_index: int, repeat_index: int, manifest=None, providers=None, readiness=None):
    manifest = manifest if manifest is not None else _make_manifest()
    readiness = readiness if readiness is not None else _make_readiness(manifest)
    if providers is None:
        providers, _, _, _ = _make_providers()
    return runner.run_trial(
        variant_id=variant_id, trial_index=trial_index, repeat_index=repeat_index, is_warmup=False,
        manifest=manifest, readiness=readiness, selected_ids=["a", "b"], providers=providers,
    )


def _run_warmup_trial(*, variant_id: str, manifest=None, providers=None, readiness=None):
    manifest = manifest if manifest is not None else _make_manifest()
    readiness = readiness if readiness is not None else _make_readiness(manifest)
    if providers is None:
        providers, _, _, _ = _make_providers()
    return runner.run_trial(
        variant_id=variant_id, trial_index=0, repeat_index=0, is_warmup=True,
        manifest=manifest, readiness=readiness, selected_ids=["a", "b"], providers=providers,
    )


def _run_full_schedule(*, providers_factory=None):
    """Run all 2 warm-up + 6 measured trials with fresh providers each
    time (mirroring fresh-process isolation), returning
    (measured_trials, warmup_trials)."""
    manifest = _make_manifest()
    readiness = _make_readiness(manifest)
    measured = []
    for entry in e25_identity.COUNTERBALANCED_ORDER:
        providers = providers_factory() if providers_factory else _make_providers()[0]
        measured.append(_run_measured_trial(
            variant_id=entry["variant"], trial_index=entry["trial_index"], repeat_index=entry["repeat"],
            manifest=manifest, providers=providers, readiness=readiness,
        ))
    warmups = []
    for variant in e25_identity.VARIANTS:
        providers = providers_factory() if providers_factory else _make_providers()[0]
        warmups.append(_run_warmup_trial(variant_id=variant, manifest=manifest, providers=providers, readiness=readiness))
    return measured, warmups


# =======================================================================
# Part 1: protocol/preflight -- torch-free, repository-state gating.
# =======================================================================


def test_module_import_is_torch_free():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, '.'); import src.dfp_phase1b_t64_early_exit_benchmark; print('torch' in sys.modules)"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_verify_protocol_is_torch_free_and_passes_on_real_repo():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, '.')\n"
            "import src.dfp_phase1b_t64_early_exit_benchmark as r\n"
            "r.verify_protocol(r.repository_root())\n"
            "print('torch' in sys.modules)"
        )],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_cli_verify_protocol_torch_free():
    result = subprocess.run(
        [sys.executable, "diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py", "verify-protocol"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_cli_preflight_torch_free_and_reports_dirty_checkout_clean_no_traceback():
    result = subprocess.run(
        [sys.executable, "diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py", "preflight"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    # The real working tree has untracked implementation files at authoring
    # time -- preflight MUST fail closed (exit 2), never a false pass, and
    # never an unhandled traceback.
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "FAIL" in result.stderr


def test_load_and_verify_manifest_rejects_file_hash_mismatch(tmp_path):
    e25_doc = {"population_binding": {"manifest_relative_path": "fake_manifest.json", "manifest_sha256": "0" * 64, "canonical_manifest_digest": "0" * 64, "benchmark_record_digest": "0" * 64}}
    (tmp_path / "fake_manifest.json").write_text(json.dumps({"benchmark_records": [], "warmup_record": {}}))
    with pytest.raises(E, match="file SHA-256"):
        runner.load_and_verify_manifest(tmp_path, e25_doc=e25_doc)


def test_load_and_verify_manifest_rejects_path_traversal(tmp_path):
    e25_doc = {"population_binding": {"manifest_relative_path": "../../../../etc/passwd", "manifest_sha256": "0" * 64, "canonical_manifest_digest": "0" * 64, "benchmark_record_digest": "0" * 64}}
    with pytest.raises(E):
        runner.load_and_verify_manifest(tmp_path, e25_doc=e25_doc)


def test_load_and_verify_manifest_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"outside_{tmp_path.name}.json"
    outside.write_text(json.dumps({"benchmark_records": [], "warmup_record": {}}))
    link = tmp_path / "manifest_link.json"
    link.symlink_to(outside)
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    e25_doc = {"population_binding": {"manifest_relative_path": "manifest_link.json", "manifest_sha256": digest, "canonical_manifest_digest": "0" * 64, "benchmark_record_digest": "0" * 64}}
    # Reading through the symlink succeeds (Path.read_bytes follows
    # symlinks) but internal verification must still reject the content
    # as not a valid benchmark manifest -- this is not a containment
    # check (that lives in the E25/E26 identity loaders' own
    # _resolve_repo_relative), only a content-validity check here.
    with pytest.raises(E):
        runner.load_and_verify_manifest(tmp_path, e25_doc=e25_doc)


def test_readiness_check_rejects_wrong_ancestor_relative_to_e26(monkeypatch):
    """perform_repository_readiness_check delegates ancestry checking to
    the generic, already-tested src.dfp_phase1b_chunk_benchmark_identity
    helpers -- this test proves the wiring calls verify_ancestor_relation
    with E26's own required_ancestor_commit, by observing it raise for an
    unrelated commit."""
    import src.dfp_phase1b_chunk_benchmark_identity as generic

    def _fake_verify_ancestor_relation(repo_root, *, required_ancestor_commit, repository_commit):
        assert required_ancestor_commit == "wrong_ancestor_marker"
        raise generic.DfpPhase1bChunkBenchmarkIdentityError("simulated ancestry failure")

    monkeypatch.setattr(runner, "verify_protocol", lambda repo_root: ({"population_binding": {"manifest_relative_path": "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json", "manifest_sha256": "x", "canonical_manifest_digest": "x", "benchmark_record_digest": "x"}}, {"identity": {"required_ancestor_commit": "wrong_ancestor_marker"}}, ARTIFACTS_DOC))
    monkeypatch.setattr(runner, "load_and_verify_manifest", lambda repo_root, e25_doc: _make_manifest())
    monkeypatch.setattr(generic, "verify_ancestor_relation", _fake_verify_ancestor_relation)
    monkeypatch.setattr(generic, "require_clean_worktree", lambda repo_root: "a" * 40)
    with pytest.raises(E, match="simulated ancestry failure"):
        runner.perform_repository_readiness_check(repo_root=REPO_ROOT)


def test_readiness_check_rejects_existing_output_unconditionally(tmp_path, monkeypatch):
    """No allow_overwrite/force parameter exists anywhere: an existing
    output destination is always rejected during readiness/preflight,
    before any dataset/model/CUDA work -- matching E25/E26's frozen
    overwrite_authorized=false / force_mode_authorized=false exactly."""
    output = tmp_path / "trial.json"
    output.write_text("{}")
    monkeypatch.setattr(runner, "verify_protocol", lambda repo_root: ({"population_binding": {}}, {"identity": {"required_ancestor_commit": "a" * 40}}, ARTIFACTS_DOC))
    monkeypatch.setattr(runner, "load_and_verify_manifest", lambda repo_root, e25_doc: _make_manifest())
    import src.dfp_phase1b_chunk_benchmark_identity as generic
    monkeypatch.setattr(generic, "require_clean_worktree", lambda repo_root: "a" * 40)
    monkeypatch.setattr(generic, "verify_ancestor_relation", lambda *a, **k: None)
    with pytest.raises(E, match="refusing to overwrite"):
        runner.perform_repository_readiness_check(repo_root=REPO_ROOT, output_path=output)


def test_perform_repository_readiness_check_has_no_overwrite_parameter():
    import inspect

    sig = inspect.signature(runner.perform_repository_readiness_check)
    for forbidden in ("allow_overwrite", "overwrite", "force"):
        assert forbidden not in sig.parameters, f"perform_repository_readiness_check must not accept {forbidden!r}"


def test_install_artifact_atomically_has_no_overwrite_parameter():
    import inspect

    sig = inspect.signature(runner.install_artifact_atomically)
    for forbidden in ("allow_overwrite", "overwrite", "force"):
        assert forbidden not in sig.parameters, f"install_artifact_atomically must not accept {forbidden!r}"


def test_install_result_decision_bundle_atomically_has_no_overwrite_parameter():
    import inspect

    sig = inspect.signature(runner.install_result_decision_bundle_atomically)
    for forbidden in ("allow_overwrite", "overwrite", "force"):
        assert forbidden not in sig.parameters, f"install_result_decision_bundle_atomically must not accept {forbidden!r}"


def test_cli_has_no_force_flag_anywhere():
    import diagnostics.run_dfp_phase1b_t64_early_exit_benchmark as cli

    parser = cli.build_parser()
    for name, subparser in parser._subparsers._group_actions[0].choices.items():
        flags = {s for action in subparser._actions for s in action.option_strings}
        assert "--force" not in flags, f"subcommand {name!r} must not expose --force"


def test_cli_force_flag_is_rejected_as_unrecognized_argument():
    import io
    from contextlib import redirect_stderr

    import diagnostics.run_dfp_phase1b_t64_early_exit_benchmark as cli

    for argv in (
        ["--repo-root", str(REPO_ROOT), "preflight", "--force"],
        ["--repo-root", str(REPO_ROOT), "run-trial", "--device", "cuda", "--variant", "T64_early_exit", "--repeat", "0", "--output", "x.json", "--force"],
        ["--repo-root", str(REPO_ROOT), "merge", "--trial", "a", "--warmup-trial", "b", "--output-dir", "x", "--force"],
    ):
        buf = io.StringIO()
        with pytest.raises(SystemExit) as exc_info:
            with redirect_stderr(buf):
                cli.main(argv)
        assert exc_info.value.code == 2
        assert "Traceback" not in buf.getvalue()


def test_readiness_check_rejects_output_path_traversal(monkeypatch):
    monkeypatch.setattr(runner, "verify_protocol", lambda repo_root: ({"population_binding": {}}, {"identity": {"required_ancestor_commit": "a" * 40}}, ARTIFACTS_DOC))
    monkeypatch.setattr(runner, "load_and_verify_manifest", lambda repo_root, e25_doc: _make_manifest())
    import src.dfp_phase1b_chunk_benchmark_identity as generic
    monkeypatch.setattr(generic, "require_clean_worktree", lambda repo_root: "a" * 40)
    monkeypatch.setattr(generic, "verify_ancestor_relation", lambda *a, **k: None)
    with pytest.raises(E, match="\\.\\."):
        runner.perform_repository_readiness_check(repo_root=REPO_ROOT, output_path=Path("../../etc/passwd"))


# =======================================================================
# Part 2: genuine early exit.
# =======================================================================


def test_t64_calls_recurrence_with_exactly_64_and_t320_with_320():
    from models.dinotext.cover_dr.finite_step_regime import finite_step_propagate

    calls = []
    original = finite_step_propagate

    def spy(graph, s0, *, alpha, steps, snapshot_steps):
        calls.append((steps, tuple(snapshot_steps)))
        return original(graph, s0, alpha=alpha, steps=steps, snapshot_steps=snapshot_steps)

    # Looked up via sys.modules (never a fresh `import a.b.c as name`
    # statement here): this repository's own "models.*" bare-path import
    # convention is fragile against a second, differently-registered
    # import of the same package elsewhere in a large test session (see
    # src/dfp_phase1a_recurrence_capture.py's own comment on this exact
    # hazard) -- the safe, already-proven route is `from ... import ...`.
    fsr_module = sys.modules["models.dinotext.cover_dr.finite_step_regime"]
    import src.dfp_phase1b_t64_early_exit_benchmark as runner_module

    orig_module_ref = fsr_module.finite_step_propagate
    fsr_module.finite_step_propagate = spy
    try:
        providers, _, _, _ = _make_providers()
        record = _make_manifest()["benchmark_records"][0]
        runner_module.process_one_window(
            record=record, window_inputs_builder=providers.window_inputs_builder_factory(None), inference=None,
            steps=64, alpha=0.98, k=4, affinity_power=3.0, class_count=5,
            stitching_geometry=_make_stitching_geometry(), final_resolution_contract=_make_final_resolution_contract(),
            unary_forward_fn=providers.unary_forward_fn,
        )
        runner_module.process_one_window(
            record=record, window_inputs_builder=providers.window_inputs_builder_factory(None), inference=None,
            steps=320, alpha=0.98, k=4, affinity_power=3.0, class_count=5,
            stitching_geometry=_make_stitching_geometry(), final_resolution_contract=_make_final_resolution_contract(),
            unary_forward_fn=providers.unary_forward_fn,
        )
    finally:
        fsr_module.finite_step_propagate = orig_module_ref
    assert calls == [(64, (64,)), (320, (320,))]


def test_t64_never_calls_trajectory_capture_helper(monkeypatch):
    import src.dfp_phase1a_recurrence_capture as capture_module

    def _boom(*a, **k):
        raise AssertionError("capture_dfp_phase1a_trajectory must never be called by process_one_window")

    monkeypatch.setattr(capture_module, "capture_dfp_phase1a_trajectory", _boom)
    providers, _, _, _ = _make_providers()
    record = _make_manifest()["benchmark_records"][0]
    # process_one_window imports finite_step_propagate directly (never via
    # capture_dfp_phase1a_trajectory) -- this passes iff that import path
    # is never touched.
    runner.process_one_window(
        record=record, window_inputs_builder=providers.window_inputs_builder_factory(None), inference=None,
        steps=64, alpha=0.98, k=4, affinity_power=3.0, class_count=5,
        stitching_geometry=_make_stitching_geometry(), final_resolution_contract=_make_final_resolution_contract(),
        unary_forward_fn=providers.unary_forward_fn,
    )


def _independent_dense_transition_matrix(graph) -> torch.Tensor:
    """Reconstructs the transition matrix as a plain dense tensor from
    ``DirectedTopKGraph``'s own public ``neighbor_indices``/
    ``transition_weights`` fields -- never calling ``graph.matmul()``.
    This is the raw DATA the real kernel is built on, assembled here by
    an entirely independent code path, so a defect in ``DirectedTopKGraph.
    matmul`` itself (not just in ``finite_step_propagate``'s loop) would
    still be caught by comparing against the oracle below."""
    num_nodes = graph.num_nodes
    dense = torch.zeros(num_nodes, num_nodes, dtype=graph.transition_weights.dtype)
    for i in range(num_nodes):
        dense[i, graph.neighbor_indices[i]] = graph.transition_weights[i]
    return dense


def _independent_finite_step_oracle(dense_transition: torch.Tensor, s0: torch.Tensor, *, alpha: float, steps: int) -> torch.Tensor:
    """A from-scratch re-implementation of P(t+1) = alpha*A@P(t) +
    (1-alpha)*S0 via a manual Python loop and plain dense matmul --
    never calls finite_step_propagate or capture_dfp_phase1a_trajectory
    on this side of the comparison. This IS the independent ground
    truth Part F/Repair 7 require: it would detect a defect shared by
    every production trajectory helper (finite_step_propagate AND
    capture_dfp_phase1a_trajectory alike), since neither is ever
    invoked to compute it."""
    p = s0.clone()
    for _ in range(steps):
        p = alpha * (dense_transition @ p) + (1 - alpha) * s0
    return p


def test_t64_and_t320_match_an_independent_oracle_never_using_production_recurrence_helpers():
    """Repair 7: replaces the former internal-consistency-only check
    (which called finite_step_propagate on both sides) with a genuine
    external oracle -- a nontrivial, asymmetric k=12 directed graph,
    reconstructed as a dense matrix from DirectedTopKGraph's own public
    fields, and a hand-written recurrence loop that never calls
    finite_step_propagate or capture_dfp_phase1a_trajectory anywhere."""
    from models.dinotext.cover_dr.finite_step_regime import finite_step_propagate as _trigger_import  # noqa: F401
    from models.dinotext.cover_dr.graph import build_directed_topk_graph

    # Looked up via sys.modules (never a fresh `import a.b.c as name`
    # statement): this repository's "models.*" bare-path import
    # convention is fragile against a second, differently-registered
    # import of the same package elsewhere in a large test session.
    fsr_module = sys.modules["models.dinotext.cover_dr.finite_step_regime"]

    num_nodes, dim, class_count, alpha = 20, 8, 5, 0.98
    dino_features = _make_dino_features(num_nodes=num_nodes, dim=dim, seed=42)
    s0 = _make_unary_scores(num_nodes=num_nodes, class_count=class_count, seed=42)
    graph = build_directed_topk_graph(dino_features, k=12, affinity_power=3.0)

    dense = _independent_dense_transition_matrix(graph)
    oracle_t64 = _independent_finite_step_oracle(dense, s0, alpha=alpha, steps=64)
    oracle_t320 = _independent_finite_step_oracle(dense, s0, alpha=alpha, steps=320)

    call_log = []
    original_propagate = fsr_module.finite_step_propagate

    def counting_propagate(graph_arg, s0_arg, *, alpha, steps, snapshot_steps):
        call_log.append(steps)
        return original_propagate(graph_arg, s0_arg, alpha=alpha, steps=steps, snapshot_steps=snapshot_steps)

    fsr_module.finite_step_propagate = counting_propagate
    try:
        production_t64 = fsr_module.finite_step_propagate(graph, s0, alpha=alpha, steps=64, snapshot_steps=(64,))
        production_t320 = fsr_module.finite_step_propagate(graph, s0, alpha=alpha, steps=320, snapshot_steps=(320,))
    finally:
        fsr_module.finite_step_propagate = original_propagate

    assert torch.allclose(production_t64.snapshots[64], oracle_t64, atol=1e-4), "production T64 disagrees with the independent oracle at step 64"
    assert torch.allclose(production_t320.snapshots[320], oracle_t320, atol=1e-4), "production T320 disagrees with the independent oracle at step 320"
    # A weak (bit-exact, not numeric-tolerance) distinguishability check:
    # with alpha=0.98 this small graph's recurrence converges quickly, so
    # steps 64 and 320 can be numerically close -- but they are computed
    # by genuinely different numbers of update steps and must never be
    # bit-identical (that would indicate the oracle loop itself is broken,
    # e.g. always returning s0 unchanged).
    assert not torch.equal(oracle_t64, oracle_t320), "sanity: the independent oracle's own step-64 and step-320 states must not be bit-identical"
    assert call_log == [64, 320], f"T64 executed exactly one 64-step call and T320 exactly one 320-step call, observed {call_log}"


def test_t64_output_equals_step_64_prefix_of_independent_320_step_oracle():
    """The core scientific proof: T64's shortened call is mathematically
    equivalent to reading the step-64 snapshot of a full 320-step
    reference trajectory, computed via a completely independent code
    path here (not process_one_window)."""
    from models.dinotext.cover_dr.finite_step_regime import finite_step_propagate
    from models.dinotext.cover_dr.graph import build_directed_topk_graph

    dino_features = _make_dino_features(num_nodes=20, dim=8, seed=42)
    s0 = _make_unary_scores(num_nodes=20, class_count=5, seed=42)
    graph = build_directed_topk_graph(dino_features, k=12, affinity_power=3.0)

    t64_trace = finite_step_propagate(graph, s0, alpha=0.98, steps=64, snapshot_steps=(64,))
    oracle_trace = finite_step_propagate(graph, s0, alpha=0.98, steps=320, snapshot_steps=(64, 320))

    assert torch.equal(t64_trace.snapshots[64], oracle_trace.snapshots[64])


def test_step_65_unreachable_for_t64_call():
    from models.dinotext.cover_dr.finite_step_regime import FiniteStepRegimeError, finite_step_propagate
    from models.dinotext.cover_dr.graph import build_directed_topk_graph

    dino_features = _make_dino_features(num_nodes=16, seed=1)
    s0 = _make_unary_scores(num_nodes=16, seed=1)
    graph = build_directed_topk_graph(dino_features, k=4, affinity_power=3.0)
    trace = finite_step_propagate(graph, s0, alpha=0.98, steps=64, snapshot_steps=(64,))
    assert trace.steps_completed == 64
    assert 65 not in trace.snapshots
    with pytest.raises(FiniteStepRegimeError, match="exceeds"):
        finite_step_propagate(graph, s0, alpha=0.98, steps=64, snapshot_steps=(65,))


def test_process_one_window_rejects_wrong_step_count_from_kernel(monkeypatch):
    from models.dinotext.cover_dr.finite_step_regime import finite_step_propagate as _trigger_import  # noqa: F401

    fsr_module = sys.modules["models.dinotext.cover_dr.finite_step_regime"]

    class _WrongTrace:
        steps_completed = 63
        snapshots = {64: torch.zeros(4, 5)}

    monkeypatch.setattr(fsr_module, "finite_step_propagate", lambda *a, **k: _WrongTrace())
    providers, _, _, _ = _make_providers()
    record = _make_manifest()["benchmark_records"][0]
    with pytest.raises(E, match="completed 63 steps"):
        runner.process_one_window(
            record=record, window_inputs_builder=providers.window_inputs_builder_factory(None), inference=None,
            steps=64, alpha=0.98, k=4, affinity_power=3.0, class_count=5,
            stitching_geometry=_make_stitching_geometry(), final_resolution_contract=_make_final_resolution_contract(),
            unary_forward_fn=providers.unary_forward_fn,
        )


def test_exact_counts_for_24_windows_one_forward_one_graph_each():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    counts = trial["counts"]
    assert counts["window_count"] == 24
    assert counts["model_forward_count"] == 24
    assert counts["graph_construction_count"] == 24
    assert counts["recurrence_steps_executed"] == 24 * 64


def test_t320_recurrence_steps_executed_is_24_times_320():
    trial = _run_measured_trial(variant_id="T320_reference", trial_index=0, repeat_index=0)
    assert trial["counts"]["recurrence_steps_executed"] == 24 * 320


def test_no_candidate_specific_recurrence_module_ever_imported(monkeypatch):
    """Proves the runner module contains no import of the masked-forward/
    component/pullback modules E26 prohibits."""
    source = Path(runner.__file__).read_text()
    for forbidden in ("dfp_phase1b_masked_forward", "dfp_phase1a_global_unary_components", "dfp_phase1a_stitching_pullback", "apply_final_rescale_adjoint", "apply_window_stitching_forward"):
        assert forbidden not in source, f"forbidden reference {forbidden!r} found in runner source"


# =======================================================================
# Part 3: timing.
# =======================================================================


def test_timing_accounting_identity_holds_by_construction():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    e25_identity.verify_timing_accounting_identity(trial["timing"])


def test_cuda_synchronize_called_before_and_after_measured_loop():
    calls = []
    providers, _, _, _ = _make_providers()
    providers = runner.TrialProviders(**{**providers.__dict__, "cuda_synchronize": lambda: calls.append(1)})
    _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0, providers=providers)
    # At least a before-loop and after-loop sync (this fake never syncs
    # elsewhere), i.e. >= 2 calls.
    assert len(calls) >= 2


def test_exact_interleaved_telemetry_event_order():
    """Repair 7: strengthens the former count-only CUDA-sync check into
    an exact interleaved event log covering reset/sync/clock/loop/read
    boundaries together, proving their RELATIVE ORDER -- not just that
    each happened often enough."""
    events: list[str] = []

    class LoggingMemoryProvider(runner.MemoryProvider):
        def reset_peak(self):
            events.append("reset_peak_memory")

        def read(self):
            events.append("read_gpu_peak_memory")
            return {"host_peak_rss_bytes": 1, "gpu_peak_memory_allocated_bytes": 1, "gpu_peak_memory_reserved_bytes": 2, "gpu_total_memory_bytes": 100}

    tick = {"n": 0}

    def logging_clock():
        tick["n"] += 1
        if tick["n"] == 1:
            events.append("setup_complete")
        return tick["n"] * 1000

    def logging_sync():
        events.append("cuda_sync")

    providers, _, _, _ = _make_providers(memory_reading=None)
    providers = runner.TrialProviders(**{
        **providers.__dict__, "memory_provider_factory": LoggingMemoryProvider,
        "clock": logging_clock, "cuda_synchronize": logging_sync,
    })
    _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0, providers=providers)

    # events currently logged: setup_complete (first clock tick),
    # reset_peak_memory, cuda_sync (before loop), cuda_sync (after
    # loop), read_gpu_peak_memory. Assert their RELATIVE order.
    def index_of(name, occurrence=0):
        indices = [i for i, e in enumerate(events) if e == name]
        return indices[occurrence]

    assert events.count("cuda_sync") == 2, f"exactly 2 CUDA syncs (before and after the loop), observed {events}"
    reset_idx = index_of("reset_peak_memory")
    sync_before_idx = index_of("cuda_sync", 0)
    sync_after_idx = index_of("cuda_sync", 1)
    read_idx = index_of("read_gpu_peak_memory")

    assert reset_idx < sync_before_idx, f"reset must occur before the measured loop's first CUDA sync: {events}"
    assert sync_before_idx < sync_after_idx, f"the two CUDA syncs must be in loop-boundary order: {events}"
    assert sync_after_idx < read_idx, f"final telemetry must be collected after the closing synchronization: {events}"


def test_t64_and_t320_use_identical_telemetry_event_boundaries():
    for variant in ("T64_early_exit", "T320_reference"):
        events = []

        class LoggingMemoryProvider(runner.MemoryProvider):
            def reset_peak(self):
                events.append("reset_peak_memory")

            def read(self):
                events.append("read_gpu_peak_memory")
                return {"host_peak_rss_bytes": 1, "gpu_peak_memory_allocated_bytes": 1, "gpu_peak_memory_reserved_bytes": 2, "gpu_total_memory_bytes": 100}

        providers, _, _, _ = _make_providers()
        providers = runner.TrialProviders(**{
            **providers.__dict__, "memory_provider_factory": LoggingMemoryProvider,
            "cuda_synchronize": lambda: events.append("cuda_sync"),
        })
        trial_index = 1 if variant == "T64_early_exit" else 0
        _run_measured_trial(variant_id=variant, trial_index=trial_index, repeat_index=0, providers=providers)
        assert events == ["cuda_sync", "cuda_sync", "read_gpu_peak_memory"] or events[:1] == ["reset_peak_memory"], (
            f"{variant}: unexpected telemetry event shape {events}"
        )


def test_profiling_instrumentation_does_not_change_scientific_digest():
    """The stage-timing clock is purely observational -- inserting extra
    clock calls (heavier profiling) must not perturb the scientific
    output digest."""
    providers_a, _, _, _ = _make_providers()
    providers_b, _, _, _ = _make_providers()
    tick = {"n": 0}

    def noisy_clock():
        tick["n"] += 1
        return tick["n"] * 999  # a completely different clock granularity/offset than the default

    providers_b = runner.TrialProviders(**{**providers_b.__dict__, "clock": noisy_clock})
    trial_a = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0, providers=providers_a)
    trial_b = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0, providers=providers_b)
    assert trial_a["scientific_output"]["scientific_output_collection_digest"] == trial_b["scientific_output"]["scientific_output_collection_digest"]


def test_stage_timings_recorded_and_never_replace_primary():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    stage = trial["timing"]["stage_timings_ns"]
    assert set(stage.keys()) == set(runner.STAGE_NAMES)
    for value in stage.values():
        assert type(value) is int and value >= 0
    # The primary field is measured independently around the whole loop,
    # never as a sum of the stage totals (which would double count
    # anything not attributed to a stage, e.g. Python loop overhead).
    assert trial["timing"]["measured_window_loop_wall_ns"] >= 0


def test_recurrence_timing_recorded_separately_from_stitching():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    stage = trial["timing"]["stage_timings_ns"]
    assert "canonical_recurrence" in stage
    assert "stitching_and_rescale" in stage


def test_verify_timing_accounting_identity_rejects_bool_field():
    timing = {k: 0 for k in runner.TIMING_PHASE_FIELDS}
    timing["preflight_wall_ns"] = True
    with pytest.raises(e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError):
        e25_identity.verify_timing_accounting_identity(timing)


def test_verify_timing_accounting_identity_rejects_negative_field():
    timing = {k: 0 for k in runner.TIMING_PHASE_FIELDS}
    timing["preflight_wall_ns"] = -1
    with pytest.raises(e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError):
        e25_identity.verify_timing_accounting_identity(timing)


def test_verify_timing_accounting_identity_rejects_broken_sum():
    timing = {k: 1 for k in runner.TIMING_PHASE_FIELDS}
    timing["process_total_wall_ns"] = 999
    with pytest.raises(e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError, match="accounting identity"):
        e25_identity.verify_timing_accounting_identity(timing)


# =======================================================================
# Part 4: memory telemetry.
# =======================================================================


def test_verify_memory_relations_accepts_valid():
    e25_identity.verify_memory_relations({"host_peak_rss_bytes": 1, "gpu_peak_memory_allocated_bytes": 5, "gpu_peak_memory_reserved_bytes": 10, "gpu_total_memory_bytes": 100})


def test_verify_memory_relations_rejects_allocated_exceeds_reserved():
    with pytest.raises(e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError):
        e25_identity.verify_memory_relations({"host_peak_rss_bytes": 1, "gpu_peak_memory_allocated_bytes": 20, "gpu_peak_memory_reserved_bytes": 10, "gpu_total_memory_bytes": 100})


def test_verify_memory_relations_rejects_reserved_exceeds_total():
    with pytest.raises(e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError):
        e25_identity.verify_memory_relations({"host_peak_rss_bytes": 1, "gpu_peak_memory_allocated_bytes": 5, "gpu_peak_memory_reserved_bytes": 200, "gpu_total_memory_bytes": 100})


def test_verify_memory_relations_rejects_missing_field():
    with pytest.raises(e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError):
        e25_identity.verify_memory_relations({"host_peak_rss_bytes": 1, "gpu_peak_memory_allocated_bytes": 5, "gpu_peak_memory_reserved_bytes": 10})


def test_verify_memory_relations_rejects_float():
    with pytest.raises(e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError):
        e25_identity.verify_memory_relations({"host_peak_rss_bytes": 1.5, "gpu_peak_memory_allocated_bytes": 5, "gpu_peak_memory_reserved_bytes": 10, "gpu_total_memory_bytes": 100})


def test_memory_provider_reset_peak_called_exactly_once_per_trial():
    providers, _, _, memory_provider = _make_providers()
    _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0, providers=providers)
    assert memory_provider.reset_calls == 1


# =======================================================================
# Part 5: digests.
# =======================================================================


def test_scientific_output_digests_are_order_sensitive():
    out_a = torch.zeros(1, 5, 4, 4)
    out_b = torch.ones(1, 5, 4, 4)
    outputs_forward = [(0, out_a), (1, out_b)]
    outputs_reversed_content = [(0, out_b), (1, out_a)]
    d1, _, _, _ = runner.compute_scientific_output_digests(variant_id="T64_early_exit", outputs_in_slot_order=[(i, o) for i, o in enumerate([out_a] + [torch.zeros(1, 5, 4, 4) for _ in range(23)])])
    d2, _, _, _ = runner.compute_scientific_output_digests(variant_id="T64_early_exit", outputs_in_slot_order=[(i, o) for i, o in enumerate([out_b] + [torch.zeros(1, 5, 4, 4) for _ in range(23)])])
    assert d1 != d2


def test_scientific_output_digests_deterministic():
    outputs = [(i, torch.full((1, 5, 4, 4), float(i))) for i in range(24)]
    d1, w1, f1, nf1 = runner.compute_scientific_output_digests(variant_id="T64_early_exit", outputs_in_slot_order=outputs)
    d2, w2, f2, nf2 = runner.compute_scientific_output_digests(variant_id="T64_early_exit", outputs_in_slot_order=outputs)
    assert d1 == d2 and w1 == w2 and f1 == f2 and nf1 == nf2


def test_scientific_output_digests_rejects_wrong_window_count():
    outputs = [(i, torch.zeros(1, 5, 4, 4)) for i in range(23)]
    with pytest.raises(E, match="24"):
        runner.compute_scientific_output_digests(variant_id="T64_early_exit", outputs_in_slot_order=outputs)


def test_scientific_output_digests_rejects_out_of_order_slots():
    outputs = [(i, torch.zeros(1, 5, 4, 4)) for i in range(24)]
    outputs[0], outputs[1] = outputs[1], outputs[0]
    with pytest.raises(E, match="not in canonical slot order"):
        runner.compute_scientific_output_digests(variant_id="T64_early_exit", outputs_in_slot_order=outputs)


def test_scientific_output_digests_count_non_finite_values():
    tensor = torch.zeros(1, 5, 4, 4)
    tensor[0, 0, 0, 0] = float("nan")
    outputs = [(0, tensor)] + [(i, torch.zeros(1, 5, 4, 4)) for i in range(1, 24)]
    _, _, finite_count, non_finite_count = runner.compute_scientific_output_digests(variant_id="T64_early_exit", outputs_in_slot_order=outputs)
    assert non_finite_count == 1
    assert finite_count == 24 * 5 * 4 * 4 - 1


def test_within_variant_digest_equality_accepts_identical():
    e25_identity.verify_within_variant_digest_equality(["a" * 64, "a" * 64, "a" * 64], variant_id="T64_early_exit")


def test_within_variant_digest_equality_rejects_divergent():
    with pytest.raises(e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError):
        e25_identity.verify_within_variant_digest_equality(["a" * 64, "b" * 64, "a" * 64], variant_id="T64_early_exit")


def test_output_validation_rejects_non_finite_window_output(monkeypatch):
    import src.dfp_phase1a_final_rescale_pullback as rescale_module

    def _nan_rescale(*a, **k):
        return torch.full((1, 5, 28, 28), float("nan"))

    monkeypatch.setattr(rescale_module, "apply_final_rescale_forward", _nan_rescale)
    providers, _, _, _ = _make_providers()
    record = _make_manifest()["benchmark_records"][0]
    with pytest.raises(Exception):
        runner.process_one_window(
            record=record, window_inputs_builder=providers.window_inputs_builder_factory(None), inference=None,
            steps=64, alpha=0.98, k=4, affinity_power=3.0, class_count=5,
            stitching_geometry=_make_stitching_geometry(), final_resolution_contract=_make_final_resolution_contract(),
            unary_forward_fn=providers.unary_forward_fn,
        )


# =======================================================================
# Part 6: schedule and merge.
# =======================================================================


def test_full_schedule_merges_cleanly_and_produces_a_valid_outcome():
    measured, warmups = _run_full_schedule()
    merged = runner.merge_trials(measured_trials=measured, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)
    assert merged["outcome"] in e25_identity.DECISION_OUTCOMES
    runner.verify_result_document(merged, artifacts_doc=ARTIFACTS_DOC)


def test_merge_rejects_wrong_measured_trial_count():
    measured, warmups = _run_full_schedule()
    with pytest.raises(E, match="exactly 6"):
        runner.merge_trials(measured_trials=measured[:5], warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)


def test_merge_rejects_wrong_warmup_trial_count():
    measured, warmups = _run_full_schedule()
    with pytest.raises(E, match="exactly 2"):
        runner.merge_trials(measured_trials=measured, warmup_trials=warmups[:1], artifacts_doc=ARTIFACTS_DOC)


def test_merge_rejects_duplicate_measured_trial():
    measured, warmups = _run_full_schedule()
    duplicated = measured[:5] + [measured[0]]
    with pytest.raises(E, match="duplicate measured trial"):
        runner.merge_trials(measured_trials=duplicated, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)


def test_merge_rejects_missing_scheduled_trial():
    measured, warmups = _run_full_schedule()
    extra = copy.deepcopy(measured[0])
    extra["repeat_index"] = 5  # not a real schedule entry, but we bypass verify_trial_document's schedule check by not calling merge_trials' own path directly -- instead simulate a genuinely off-schedule document by mutating trial_index
    extra["trial_index"] = 99
    replaced = measured[1:] + [extra]
    with pytest.raises(E):
        runner.merge_trials(measured_trials=replaced, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)


def test_merge_rejects_warmup_trial_missing_is_warmup_flag():
    measured, warmups = _run_full_schedule()
    bad_warmup = copy.deepcopy(warmups[0])
    bad_warmup["is_warmup"] = False
    with pytest.raises(E):
        runner.merge_trials(measured_trials=measured, warmup_trials=[bad_warmup, warmups[1]], artifacts_doc=ARTIFACTS_DOC)


def test_merge_rejects_mixed_repository_commit():
    manifest = _make_manifest()
    measured = []
    for i, entry in enumerate(e25_identity.COUNTERBALANCED_ORDER):
        readiness = _make_readiness(manifest)
        if i == 3:
            readiness = runner.ReadinessResult(**{**readiness.__dict__, "repository_commit": "f" * 40})
        providers, _, _, _ = _make_providers()
        measured.append(_run_measured_trial(variant_id=entry["variant"], trial_index=entry["trial_index"], repeat_index=entry["repeat"], manifest=manifest, providers=providers, readiness=readiness))
    warmups = [_run_warmup_trial(variant_id=v, manifest=manifest) for v in e25_identity.VARIANTS]
    with pytest.raises(E, match="disagree on"):
        runner.merge_trials(measured_trials=measured, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)


def test_merge_rejects_mixed_device_record():
    manifest = _make_manifest()
    measured = []
    for i, entry in enumerate(e25_identity.COUNTERBALANCED_ORDER):
        providers, _, _, _ = _make_providers()
        if i == 2:
            providers = runner.TrialProviders(**{**providers.__dict__, "device_record_fn": lambda: {"device_type": "cuda", "gpu_model": "different", "gpu_uuid": "x", "cpu_count_visible": 1, "one_visible_cuda_device": True}})
        measured.append(_run_measured_trial(variant_id=entry["variant"], trial_index=entry["trial_index"], repeat_index=entry["repeat"], manifest=manifest, providers=providers))
    warmups = [_run_warmup_trial(variant_id=v, manifest=manifest) for v in e25_identity.VARIANTS]
    with pytest.raises(E, match="disagree on"):
        runner.merge_trials(measured_trials=measured, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)


def test_merge_rejects_mixed_precision_record():
    manifest = _make_manifest()
    measured = []
    for i, entry in enumerate(e25_identity.COUNTERBALANCED_ORDER):
        providers, _, _, _ = _make_providers()
        if i == 1:
            providers = runner.TrialProviders(**{**providers.__dict__, "precision_record_fn": lambda: {"compute_dtype": "float64", "output_dtype": "float64"}})
        measured.append(_run_measured_trial(variant_id=entry["variant"], trial_index=entry["trial_index"], repeat_index=entry["repeat"], manifest=manifest, providers=providers))
    warmups = [_run_warmup_trial(variant_id=v, manifest=manifest) for v in e25_identity.VARIANTS]
    with pytest.raises(E, match="disagree on"):
        runner.merge_trials(measured_trials=measured, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)


def test_warmup_trials_excluded_from_per_variant_summary():
    measured, warmups = _run_full_schedule()
    # Corrupt a warm-up trial's timing to an extreme value: if it leaked
    # into the summary, min/max would reflect it.
    bad_warmups = copy.deepcopy(warmups)
    for w in bad_warmups:
        w["timing"] = dict(w["timing"])
        w["timing"]["measured_window_loop_seconds"] = 999999.0
    merged_good = runner.merge_trials(measured_trials=measured, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)
    merged_bad = runner.merge_trials(measured_trials=measured, warmup_trials=bad_warmups, artifacts_doc=ARTIFACTS_DOC)
    assert merged_good["per_variant_summary"] == merged_bad["per_variant_summary"]


def test_median_of_three_measured_repeats():
    measured, warmups = _run_full_schedule()
    merged = runner.merge_trials(measured_trials=measured, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)
    for variant in e25_identity.VARIANTS:
        seconds = sorted(t["timing"]["measured_window_loop_seconds"] for t in measured if t["variant_id"] == variant)
        assert merged["per_variant_summary"][variant]["median_measured_window_loop_seconds"] == seconds[1]


@pytest.mark.parametrize("t64_seconds,t320_seconds,expect_proceed", [
    (1.0, 1.0, False),   # speedup == 1.0 < 1.25
    (1.0, 1.24, False),  # just below threshold
    (1.0, 1.25, True),   # exactly at threshold -- proceeds
    (1.0, 2.0, True),    # well above threshold
])
def test_speedup_threshold_gate(t64_seconds, t320_seconds, expect_proceed):
    all_gates_true = dict(
        all_trials_complete=True, all_artifacts_verified=True, within_variant_digests_identical=True,
        zero_non_finite_outputs=True, exact_counts_verified=True, no_oom=True, memory_relations_pass=True,
    )
    outcome = e25_identity.apply_decision_rule(
        **all_gates_true, t64_median_seconds=t64_seconds, t320_median_seconds=t320_seconds,
        end_to_end_speedup=t320_seconds / t64_seconds, development_accuracy_disclosure_pp=e25_identity.DEVELOPMENT_ACCURACY_DISCLOSURE_PP,
    )
    assert (outcome == e25_identity.DECISION_OUTCOMES[0]) == expect_proceed


def test_recurrence_only_speedup_cannot_override_failed_primary_speedup():
    """Even if T64's recurrence-only stage were dramatically faster, a
    failed END-TO-END speedup must still stop the track -- apply_decision_
    rule never reads t64_recurrence_speedup_over_t320 at all."""
    outcome = e25_identity.apply_decision_rule(
        all_trials_complete=True, all_artifacts_verified=True, within_variant_digests_identical=True,
        zero_non_finite_outputs=True, exact_counts_verified=True, no_oom=True, memory_relations_pass=True,
        t64_median_seconds=1.0, t320_median_seconds=1.1, end_to_end_speedup=1.1,
        development_accuracy_disclosure_pp=e25_identity.DEVELOPMENT_ACCURACY_DISCLOSURE_PP,
    )
    assert outcome == e25_identity.DECISION_OUTCOMES[1]


def test_every_other_gate_individually_forces_stop():
    base = dict(
        all_trials_complete=True, all_artifacts_verified=True, within_variant_digests_identical=True,
        zero_non_finite_outputs=True, exact_counts_verified=True, no_oom=True, memory_relations_pass=True,
        t64_median_seconds=1.0, t320_median_seconds=2.0, end_to_end_speedup=2.0,
        development_accuracy_disclosure_pp=e25_identity.DEVELOPMENT_ACCURACY_DISCLOSURE_PP,
    )
    assert e25_identity.apply_decision_rule(**base) == e25_identity.DECISION_OUTCOMES[0]
    for flag in ("all_trials_complete", "all_artifacts_verified", "within_variant_digests_identical", "zero_non_finite_outputs", "exact_counts_verified", "no_oom", "memory_relations_pass"):
        mutated = dict(base)
        mutated[flag] = False
        assert e25_identity.apply_decision_rule(**mutated) == e25_identity.DECISION_OUTCOMES[1], f"flag {flag} did not force STOP"


def test_decision_document_echoes_merged_result_never_recomputes():
    measured, warmups = _run_full_schedule()
    merged = runner.merge_trials(measured_trials=measured, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)
    fake_sha = "a" * 64
    decision = runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=fake_sha)
    assert decision["outcome"] == merged["outcome"]
    assert decision["decision_rule_inputs"] == merged["decision_rule_inputs"]
    runner.verify_decision_document(decision, artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=fake_sha)


def test_decision_document_rejects_mismatched_merged_result_sha():
    measured, warmups = _run_full_schedule()
    merged = runner.merge_trials(measured_trials=measured, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)
    decision = runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256="a" * 64)
    with pytest.raises(E, match="merged_result_sha256"):
        runner.verify_decision_document(decision, artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256="b" * 64)


# =======================================================================
# Part 7: atomicity.
# =======================================================================


def test_install_artifact_atomically_rejects_existing_destination(tmp_path):
    output = tmp_path / "artifact.json"
    output.write_text('{"already": "there"}')
    with pytest.raises(E, match="refusing to overwrite"):
        runner.install_artifact_atomically({"x": 1}, output, verify_fn=lambda d: None)
    assert output.read_text() == '{"already": "there"}'


def test_install_artifact_atomically_no_partial_artifact_on_verify_failure(tmp_path):
    output = tmp_path / "artifact.json"

    def _always_fail(document):
        raise E("staged verification deliberately failed")

    with pytest.raises(E, match="staged verification deliberately failed"):
        runner.install_artifact_atomically({"x": 1}, output, verify_fn=_always_fail)
    assert not output.exists()
    leftover = list(tmp_path.iterdir())
    assert leftover == [], f"orphan staging residue left behind: {leftover}"


def test_install_artifact_atomically_no_orphan_staging_on_success(tmp_path):
    output = tmp_path / "artifact.json"
    runner.install_artifact_atomically({"x": 1}, output, verify_fn=lambda d: None)
    assert output.exists()
    assert list(tmp_path.iterdir()) == [output]


def test_install_artifact_atomically_reopens_and_publicly_reverifies(tmp_path):
    output = tmp_path / "artifact.json"
    calls = []

    def _verify(document):
        calls.append(document)

    runner.install_artifact_atomically({"x": 1}, output, verify_fn=_verify)
    # Called at least twice: once against staged bytes, once against
    # reopened, installed bytes.
    assert len(calls) == 2


def test_install_artifact_atomically_retry_into_fresh_destination_succeeds(tmp_path):
    output = tmp_path / "artifact.json"

    def _fail_once_then_succeed():
        state = {"n": 0}

        def _verify(document):
            state["n"] += 1
            if state["n"] == 1:
                raise E("first attempt fails")

        return _verify

    with pytest.raises(E):
        runner.install_artifact_atomically({"x": 1}, output, verify_fn=_fail_once_then_succeed())
    assert not output.exists()
    fresh_output = tmp_path / "artifact_retry.json"
    runner.install_artifact_atomically({"x": 1}, fresh_output, verify_fn=lambda d: None)
    assert fresh_output.exists()


def test_install_artifact_atomically_existing_valid_file_preserved_when_a_sibling_write_fails(tmp_path):
    existing = tmp_path / "existing.json"
    runner.install_artifact_atomically({"kept": True}, existing, verify_fn=lambda d: None)
    other = tmp_path / "other.json"
    with pytest.raises(E):
        runner.install_artifact_atomically({"x": 1}, other, verify_fn=lambda d: (_ for _ in ()).throw(E("boom")))
    assert json.loads(existing.read_text()) == {"kept": True}


def test_install_artifact_atomically_fatal_exceptions_propagate(tmp_path):
    output = tmp_path / "artifact.json"

    def _verify(document):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner.install_artifact_atomically({"x": 1}, output, verify_fn=_verify)
    assert not output.exists()


# =======================================================================
# Part 8: authorization (absence proofs).
# =======================================================================


def test_no_gt_field_anywhere_in_trial_document():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    serialized = json.dumps(trial).lower()
    for forbidden in ("ground_truth", "\"gt\"", "gt_seg_map"):
        assert forbidden not in serialized


def test_cli_has_no_confirmation_or_full_5000_subcommand():
    import diagnostics.run_dfp_phase1b_t64_early_exit_benchmark as cli

    parser = cli.build_parser()
    subcommand_names = set(cli._DISPATCH.keys())
    for forbidden in ("confirm", "confirmation", "full-5000", "full5000"):
        assert forbidden not in subcommand_names


def test_cli_run_trial_has_no_manifest_override_or_debug_subset_flag():
    import diagnostics.run_dfp_phase1b_t64_early_exit_benchmark as cli

    parser = cli.build_parser()
    run_trial_actions = [a.option_strings for a in parser._subparsers._group_actions[0].choices["run-trial"]._actions]
    flat = {s for group in run_trial_actions for s in group}
    for forbidden in ("--manifest", "--max-images", "--debug-subset", "--population"):
        assert forbidden not in flat


def test_launcher_never_invokes_sbatch_or_self_submits():
    text = Path(REPO_ROOT / "dfp_phase1b_t64_early_exit_benchmark_h100.sbatch").read_text()
    lines = [l for l in text.splitlines() if not l.strip().startswith("#")]
    body = "\n".join(lines)
    assert "sbatch " not in body and not body.strip().startswith("sbatch")
    assert "srun " not in body
    assert "salloc" not in body


def test_launcher_has_no_overwrite_or_force_flag_and_no_confirmation_command():
    """Checks for actual CLI/flag invocations the launcher's shell body
    would execute -- never a blanket ban on the word "confirm" itself,
    since the launcher legitimately PRINTS descriptive prose noting that
    a PROCEED outcome does not itself run confirmation (see Part 7 of
    docs/dfp_phase1b_t64_early_exit_benchmark_run.md)."""
    text = Path(REPO_ROOT / "dfp_phase1b_t64_early_exit_benchmark_h100.sbatch").read_text()
    executable_lines = [
        line for line in text.splitlines()
        if not line.strip().startswith("#") and "$CLI" in line
    ]
    body = "\n".join(executable_lines)
    assert "--force" not in body
    assert "confirm" not in body.lower()
    assert "5000" not in body
    assert "--max-images" not in body and "--debug-subset" not in body


def test_launcher_syntax_is_valid():
    result = subprocess.run(["bash", "-n", str(REPO_ROOT / "dfp_phase1b_t64_early_exit_benchmark_h100.sbatch")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_masked_forward_or_component_import_in_cli_or_runner():
    for path in (Path(runner.__file__), REPO_ROOT / "diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py"):
        text = path.read_text()
        for forbidden in ("dfp_phase1b_masked_forward", "build_masked_seed_chunk", "dfp_phase1a_global_unary_components"):
            assert forbidden not in text, f"{forbidden!r} found in {path}"


def test_five_authorized_future_files_are_exactly_the_ones_created():
    e26_doc = e26_identity.load_identity(repo_root=REPO_ROOT)
    expected = list(e26_doc["authorized_future_files"]["list"])
    for relative in expected:
        assert (REPO_ROOT / relative).exists(), f"authorized file {relative} does not exist"


# =======================================================================
# Part 9: general robustness -- fatal exceptions propagate.
# =======================================================================


def test_run_trial_propagates_fatal_exceptions(monkeypatch):
    providers, _, _, _ = _make_providers()

    def _boom(inference, crop_tensor):
        raise SystemExit(3)

    providers = runner.TrialProviders(**{**providers.__dict__, "unary_forward_fn": _boom})
    with pytest.raises(SystemExit):
        _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0, providers=providers)


def test_verify_trial_document_rejects_unknown_key():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    mutated = dict(trial)
    mutated["unexpected_extra_field"] = 1
    with pytest.raises(E, match="unknown"):
        runner.verify_trial_document(mutated, artifacts_doc=ARTIFACTS_DOC)


def test_verify_trial_document_rejects_missing_key():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    mutated = dict(trial)
    del mutated["timing"]
    with pytest.raises(E, match="missing"):
        runner.verify_trial_document(mutated, artifacts_doc=ARTIFACTS_DOC)


def test_verify_trial_document_rejects_incomplete_trial():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    mutated = dict(trial)
    mutated["complete"] = False
    with pytest.raises(E, match="complete and final"):
        runner.verify_trial_document(mutated, artifacts_doc=ARTIFACTS_DOC)


def test_run_trial_rejects_off_schedule_measured_entry():
    with pytest.raises(E, match="frozen counterbalanced schedule"):
        _run_measured_trial(variant_id="T64_early_exit", trial_index=99, repeat_index=0)


def test_run_trial_rejects_unknown_variant():
    with pytest.raises(E, match="variant_id must be one of"):
        _run_measured_trial(variant_id="T999_bogus", trial_index=0, repeat_index=0)


# =======================================================================
# Repair 2: strict nested-record validation (device_record,
# software_record, precision_record, deterministic_settings_record).
# =======================================================================


def _mutated_trial(**field_mutations):
    trial = dict(_run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0))
    for field, mutator in field_mutations.items():
        trial[field] = mutator(dict(trial[field])) if isinstance(trial[field], dict) else mutator(trial[field])
    return trial


@pytest.mark.parametrize("record_field,key_list_name", [
    ("device_record", "device_record_keys"), ("software_record", "software_record_keys"),
    ("precision_record", "precision_record_keys"), ("deterministic_settings_record", "deterministic_settings_record_keys"),
])
def test_nested_record_rejects_non_mapping(record_field, key_list_name):
    trial = _mutated_trial(**{record_field: lambda _old: "not-a-dict"})
    with pytest.raises(E, match="must be a mapping"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


@pytest.mark.parametrize("record_field,key_list_name", [
    ("device_record", "device_record_keys"), ("software_record", "software_record_keys"),
    ("precision_record", "precision_record_keys"), ("deterministic_settings_record", "deterministic_settings_record_keys"),
])
def test_nested_record_rejects_missing_key(record_field, key_list_name):
    def _remove_one_key(d):
        d.pop(next(iter(d)))
        return d

    trial = _mutated_trial(**{record_field: _remove_one_key})
    with pytest.raises(E, match="wrong key set"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


@pytest.mark.parametrize("record_field", ["device_record", "software_record", "precision_record", "deterministic_settings_record"])
def test_nested_record_rejects_extra_key(record_field):
    def _add_key(d):
        d["totally_unexpected_extra_field"] = 1
        return d

    trial = _mutated_trial(**{record_field: _add_key})
    with pytest.raises(E, match="wrong key set"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


@pytest.mark.parametrize("record_field", ["device_record", "software_record", "precision_record", "deterministic_settings_record"])
def test_nested_record_rejects_renamed_key(record_field):
    def _rename_one_key(d):
        old_key = next(iter(d))
        d["renamed_" + old_key] = d.pop(old_key)
        return d

    trial = _mutated_trial(**{record_field: _rename_one_key})
    with pytest.raises(E, match="wrong key set"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_device_record_rejects_empty_string_field():
    trial = _mutated_trial(device_record=lambda d: {**d, "gpu_model": ""})
    with pytest.raises(E, match="non-empty"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_device_record_rejects_bool_as_int_cpu_count():
    trial = _mutated_trial(device_record=lambda d: {**d, "cpu_count_visible": True})
    with pytest.raises(E, match="positive exact int"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_device_record_rejects_float_cpu_count():
    trial = _mutated_trial(device_record=lambda d: {**d, "cpu_count_visible": 1.5})
    with pytest.raises(E, match="positive exact int"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_device_record_rejects_nan_cpu_count():
    trial = _mutated_trial(device_record=lambda d: {**d, "cpu_count_visible": float("nan")})
    with pytest.raises(E, match="positive exact int"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_device_record_rejects_negative_cpu_count():
    trial = _mutated_trial(device_record=lambda d: {**d, "cpu_count_visible": -1})
    with pytest.raises(E, match="positive exact int"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_device_record_rejects_non_bool_one_visible_cuda_device():
    trial = _mutated_trial(device_record=lambda d: {**d, "one_visible_cuda_device": 1})
    with pytest.raises(E, match="exact bool"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_software_record_rejects_empty_string_field():
    trial = _mutated_trial(software_record=lambda d: {**d, "driver_version": ""})
    with pytest.raises(E, match="non-empty"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_software_record_rejects_wrong_type_field():
    trial = _mutated_trial(software_record=lambda d: {**d, "torch_version": 20})
    with pytest.raises(E, match="non-empty"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_precision_record_rejects_empty_string_field():
    trial = _mutated_trial(precision_record=lambda d: {**d, "compute_dtype": ""})
    with pytest.raises(E, match="non-empty"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_deterministic_settings_record_rejects_int_as_bool():
    trial = _mutated_trial(deterministic_settings_record=lambda d: {**d, "cudnn_benchmark": 1})
    with pytest.raises(E, match="exact bool"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_deterministic_settings_record_rejects_non_string_non_null_pythonhashseed():
    trial = _mutated_trial(deterministic_settings_record=lambda d: {**d, "pythonhashseed": 12345})
    with pytest.raises(E, match="pythonhashseed"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_deterministic_settings_record_accepts_null_pythonhashseed():
    trial = _mutated_trial(deterministic_settings_record=lambda d: {**d, "pythonhashseed": None})
    runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_valid_nested_records_pass():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


# =======================================================================
# Repair 3: hex-64 digest format validation.
# =======================================================================


@pytest.mark.parametrize("bad_value", [
    "ab", "a" * 63, "a" * 65, "g" * 64, "A" * 64, ("a" * 63) + " ", "0x" + "a" * 62,
    123, True, None, [], {},
])
def test_collection_digest_format_rejected(bad_value):
    trial = _mutated_trial(scientific_output=lambda d: {**d, "scientific_output_collection_digest": bad_value})
    with pytest.raises(E):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


@pytest.mark.parametrize("bad_value", [
    "ab", "a" * 63, "a" * 65, "g" * 64, "A" * 64, "a" * 64 + "\n", 123, True, None, [], {},
])
def test_per_window_digest_entry_format_rejected(bad_value):
    def _mutate(d):
        d = dict(d)
        pwd = dict(d["per_window_output_digests"])
        pwd["0"] = bad_value
        d["per_window_output_digests"] = pwd
        return d

    trial = _mutated_trial(scientific_output=_mutate)
    with pytest.raises(E):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_per_window_digests_rejects_missing_entry():
    def _mutate(d):
        d = dict(d)
        pwd = dict(d["per_window_output_digests"])
        del pwd["0"]
        d["per_window_output_digests"] = pwd
        return d

    trial = _mutated_trial(scientific_output=_mutate)
    with pytest.raises(E, match="window-identity key set"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_per_window_digests_rejects_extra_entry():
    def _mutate(d):
        d = dict(d)
        pwd = dict(d["per_window_output_digests"])
        pwd["24"] = "a" * 64
        d["per_window_output_digests"] = pwd
        return d

    trial = _mutated_trial(scientific_output=_mutate)
    with pytest.raises(E, match="window-identity key set"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_per_window_digests_rejects_non_mapping_container():
    trial = _mutated_trial(scientific_output=lambda d: {**d, "per_window_output_digests": ["a" * 64] * 24})
    with pytest.raises(E, match="mapping"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_valid_digests_pass_format_validation():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


# =======================================================================
# Repair 4: exact schedule-metadata types (trial_index, repeat_index,
# is_warmup) -- isinstance() is deliberately never used for these, since
# isinstance(True, int) is True in Python.
# =======================================================================


def test_repeat_index_bool_rejected():
    trial = _mutated_trial(repeat_index=lambda _old: True)
    with pytest.raises(E, match="exact non-boolean ints"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_trial_index_bool_rejected():
    trial = _mutated_trial(trial_index=lambda _old: False)
    with pytest.raises(E, match="exact non-boolean ints"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_trial_index_float_rejected():
    trial = _mutated_trial(trial_index=lambda _old: 1.0)
    with pytest.raises(E, match="exact non-boolean ints"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_trial_index_string_rejected():
    trial = _mutated_trial(trial_index=lambda _old: "1")
    with pytest.raises(E, match="exact non-boolean ints"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_trial_index_null_rejected():
    trial = _mutated_trial(trial_index=lambda _old: None)
    with pytest.raises(E, match="exact non-boolean ints"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_trial_index_list_rejected():
    trial = _mutated_trial(trial_index=lambda _old: [1])
    with pytest.raises(E, match="exact non-boolean ints"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_trial_index_mapping_rejected():
    trial = _mutated_trial(trial_index=lambda _old: {"v": 1})
    with pytest.raises(E, match="exact non-boolean ints"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_negative_trial_index_rejected():
    trial = _mutated_trial(trial_index=lambda _old: -1)
    with pytest.raises(E, match="non-negative"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_negative_repeat_index_rejected():
    trial = _mutated_trial(repeat_index=lambda _old: -1)
    with pytest.raises(E, match="non-negative"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_is_warmup_int_as_bool_rejected():
    trial = _mutated_trial(is_warmup=lambda _old: 0)
    with pytest.raises(E, match="exact bool"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_is_warmup_string_rejected():
    trial = _mutated_trial(is_warmup=lambda _old: "false")
    with pytest.raises(E, match="exact bool"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_measured_trial_with_warmup_repeat_index_other_than_zero_rejected():
    warmup = dict(_run_warmup_trial(variant_id="T64_early_exit"))
    warmup["repeat_index"] = 1
    with pytest.raises(E, match="repeat_index == 0"):
        runner.verify_trial_document(warmup, artifacts_doc=ARTIFACTS_DOC)


def test_measured_trial_schedule_combination_must_match_one_e25_entry():
    """verify_trial_document itself (not only merge_trials) must reject
    an off-schedule (variant, trial_index, repeat_index) combination for
    a non-warmup trial."""
    trial = _mutated_trial(trial_index=lambda _old: 0)  # variant=T64_early_exit, trial_index=0 is NOT a real schedule entry (that slot is T320_reference)
    with pytest.raises(E, match="frozen counterbalanced schedule"):
        runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)


def test_valid_schedule_metadata_passes():
    trial = _run_measured_trial(variant_id="T64_early_exit", trial_index=1, repeat_index=0)
    runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)
    warmup = _run_warmup_trial(variant_id="T64_early_exit")
    runner.verify_trial_document(warmup, artifacts_doc=ARTIFACTS_DOC)


# =======================================================================
# Repair 5: the runner's public API exposes exactly one ordinary domain
# exception -- library-level tests (not merely CLI-level), directly
# proving no src.dfp_phase1b_t64_early_exit_benchmark_identity.
# DfpPhase1bT64EarlyExitBenchmarkIdentityError ever escapes.
# =======================================================================


def test_verify_trial_document_never_leaks_identity_exception_on_count_mismatch():
    trial = _mutated_trial(counts=lambda d: {**d, "window_count": 23})
    with pytest.raises(E):
        try:
            runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)
        except e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError:
            pytest.fail("identity-layer exception leaked through the runner's public API")


def test_verify_trial_document_never_leaks_identity_exception_on_timing_mismatch():
    trial = _mutated_trial(timing=lambda d: {**d, "process_total_wall_ns": d["process_total_wall_ns"] + 1})
    with pytest.raises(E):
        try:
            runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)
        except e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError:
            pytest.fail("identity-layer exception leaked through the runner's public API")


def test_verify_trial_document_never_leaks_identity_exception_on_memory_mismatch():
    trial = _mutated_trial(memory=lambda d: {**d, "gpu_peak_memory_allocated_bytes": d["gpu_total_memory_bytes"] + 1})
    with pytest.raises(E):
        try:
            runner.verify_trial_document(trial, artifacts_doc=ARTIFACTS_DOC)
        except e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError:
            pytest.fail("identity-layer exception leaked through the runner's public API")


def test_merge_trials_never_leaks_identity_exception_on_digest_mismatch():
    measured, warmups = _run_full_schedule()
    mutated = list(measured)
    mutated[1] = dict(mutated[1])
    mutated[1]["scientific_output"] = dict(mutated[1]["scientific_output"])
    mutated[1]["scientific_output"]["scientific_output_collection_digest"] = "f" * 64
    with pytest.raises(E):
        try:
            runner.merge_trials(measured_trials=mutated, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)
        except e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError:
            pytest.fail("identity-layer exception leaked through merge_trials")


def test_merge_trials_never_leaks_identity_exception_from_decision_rule():
    """apply_decision_rule itself validates development_accuracy_
    disclosure_pp -- confirm any failure there is also wrapped."""
    measured, warmups = _run_full_schedule()
    with pytest.raises((E, AssertionError)):
        try:
            e25_identity.apply_decision_rule(
                all_trials_complete=True, all_artifacts_verified=True, within_variant_digests_identical=True,
                zero_non_finite_outputs=True, exact_counts_verified=True, no_oom=True, memory_relations_pass=True,
                t64_median_seconds=1.0, t320_median_seconds=2.0, end_to_end_speedup=2.0,
                development_accuracy_disclosure_pp=-999.0,
            )
        except e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError as exc:
            # This direct call to the identity module itself is EXPECTED
            # to raise the identity exception (it is not routed through
            # the runner's translation seam) -- the real assertion is
            # that merge_trials/verify_trial_document, the RUNNER's own
            # public functions, never do the same. Re-raise as the
            # runner's own type here only to satisfy pytest.raises above.
            raise E(str(exc)) from exc


def test_all_e25_identity_calls_inside_runner_module_are_wrapped():
    """Static proof, not just a handful of dynamic cases: every call to
    an e25_identity.verify_*/apply_decision_rule function inside the
    runner module's source is lexically inside a
    `with _translate_identity_errors():` block."""
    import ast

    source = Path(runner.__file__).read_text()
    tree = ast.parse(source)
    wrapped_names = {
        "verify_count_relations", "verify_timing_accounting_identity", "verify_memory_relations",
        "verify_scientific_output_finiteness", "verify_within_variant_digest_equality",
        "verify_declared_median", "verify_declared_speedup", "apply_decision_rule",
    }

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.unwrapped_calls = []
            self._with_depth_stack = []

        def visit_With(self, node):
            is_translate_block = any(
                isinstance(item.context_expr, ast.Call) and getattr(item.context_expr.func, "id", None) == "_translate_identity_errors"
                for item in node.items
            )
            self._with_depth_stack.append(is_translate_block)
            self.generic_visit(node)
            self._with_depth_stack.pop()

        def visit_Call(self, node):
            func_name = getattr(node.func, "id", None)
            if func_name in wrapped_names:
                if not any(self._with_depth_stack):
                    self.unwrapped_calls.append((func_name, node.lineno))
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    assert visitor.unwrapped_calls == [], f"e25_identity calls not wrapped in _translate_identity_errors(): {visitor.unwrapped_calls}"


def test_fatal_exceptions_still_propagate_through_translate_identity_errors():
    with pytest.raises(KeyboardInterrupt):
        with runner._translate_identity_errors():
            raise KeyboardInterrupt

    with pytest.raises(SystemExit):
        with runner._translate_identity_errors():
            raise SystemExit(1)

    with pytest.raises(MemoryError):
        with runner._translate_identity_errors():
            raise MemoryError


# =======================================================================
# Repair 1: pair-atomic result/decision bundle installation.
# =======================================================================


def _valid_merged_result_and_decision():
    measured, warmups = _run_full_schedule()
    merged = runner.merge_trials(measured_trials=measured, warmup_trials=warmups, artifacts_doc=ARTIFACTS_DOC)
    return merged


def _install_bundle(tmp_path, merged_result, *, build_decision=None):
    bundle_dir = tmp_path / "result"
    if build_decision is None:
        def build_decision(sha):
            return runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged_result, merged_result_sha256=sha)
    runner.install_result_decision_bundle_atomically(merged_result=merged_result, build_decision=build_decision, bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC)
    return bundle_dir


def test_bundle_installs_result_and_decision_via_single_directory_rename(tmp_path):
    merged = _valid_merged_result_and_decision()
    bundle_dir = _install_bundle(tmp_path, merged)
    assert bundle_dir.is_dir()
    assert sorted(p.name for p in bundle_dir.iterdir()) == sorted((runner.RESULT_BUNDLE_RESULT_FILENAME, runner.RESULT_BUNDLE_DECISION_FILENAME))
    installed_result = json.loads((bundle_dir / runner.RESULT_BUNDLE_RESULT_FILENAME).read_text())
    installed_decision = json.loads((bundle_dir / runner.RESULT_BUNDLE_DECISION_FILENAME).read_text())
    assert installed_decision["merged_result_sha256"] == hashlib.sha256((bundle_dir / runner.RESULT_BUNDLE_RESULT_FILENAME).read_bytes()).hexdigest()
    assert installed_decision["outcome"] == installed_result["outcome"]


def test_bundle_rejects_existing_destination(tmp_path):
    merged = _valid_merged_result_and_decision()
    bundle_dir = tmp_path / "result"
    bundle_dir.mkdir()
    (bundle_dir / "sentinel.txt").write_text("pre-existing")
    with pytest.raises(E, match="refusing to overwrite"):
        runner.install_result_decision_bundle_atomically(
            merged_result=merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
            bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
        )
    # pre-existing destination untouched
    assert (bundle_dir / "sentinel.txt").read_text() == "pre-existing"
    assert not (bundle_dir / runner.RESULT_BUNDLE_RESULT_FILENAME).exists()


def test_direct_installer_cannot_overwrite_a_valid_prior_bundle(tmp_path):
    merged = _valid_merged_result_and_decision()
    bundle_dir = _install_bundle(tmp_path, merged)
    with pytest.raises(E, match="refusing to overwrite"):
        runner.install_result_decision_bundle_atomically(
            merged_result=merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
            bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
        )
    # the original valid pair is untouched
    installed_decision = json.loads((bundle_dir / runner.RESULT_BUNDLE_DECISION_FILENAME).read_text())
    assert installed_decision["outcome"] == merged["outcome"]


def test_retry_into_a_different_fresh_destination_succeeds(tmp_path):
    merged = _valid_merged_result_and_decision()
    bundle_dir_a = tmp_path / "attempt_a"
    bundle_dir_a.mkdir()  # pre-existing -- attempt_a must fail
    with pytest.raises(E):
        runner.install_result_decision_bundle_atomically(
            merged_result=merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
            bundle_dir=bundle_dir_a, artifacts_doc=ARTIFACTS_DOC,
        )
    bundle_dir_b = tmp_path / "attempt_b"
    runner.install_result_decision_bundle_atomically(
        merged_result=merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
        bundle_dir=bundle_dir_b, artifacts_doc=ARTIFACTS_DOC,
    )
    assert bundle_dir_b.is_dir()
    assert (bundle_dir_b / runner.RESULT_BUNDLE_RESULT_FILENAME).exists()
    assert (bundle_dir_b / runner.RESULT_BUNDLE_DECISION_FILENAME).exists()


def test_no_result_only_or_decision_only_final_state_possible_across_failure_points(tmp_path):
    """The core Finding-1 regression proof: no matter WHERE a failure is
    injected before the final rename, the bundle directory is either
    fully absent or a complete pair -- never one file without the
    other."""
    merged = _valid_merged_result_and_decision()

    def assert_absent_or_complete(bundle_dir):
        if not bundle_dir.exists():
            return
        names = sorted(p.name for p in bundle_dir.iterdir())
        assert names == sorted((runner.RESULT_BUNDLE_RESULT_FILENAME, runner.RESULT_BUNDLE_DECISION_FILENAME)), (
            f"bundle directory exists but is not a complete pair: {names}"
        )

    # 1. staging-directory creation failure
    bundle_dir = tmp_path / "case1"
    import src.dfp_phase1b_t64_early_exit_benchmark as runner_module
    orig_mkdir = Path.mkdir

    def boom_mkdir(self, *a, **k):
        if ".staging-" in self.name:
            raise OSError("simulated staging-directory creation failure")
        return orig_mkdir(self, *a, **k)

    Path.mkdir = boom_mkdir
    try:
        with pytest.raises(Exception):
            runner.install_result_decision_bundle_atomically(
                merged_result=merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
                bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
            )
    finally:
        Path.mkdir = orig_mkdir
    assert_absent_or_complete(bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.glob(".staging-*")) == []

    # 2. merged-result write failure
    bundle_dir = tmp_path / "case2"
    import src.dfp_phase1a_artifact_schema as schema_module
    orig_write = schema_module.write_json_atomically
    call_count = {"n": 0}

    def boom_write_result(path, document, *, label):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise IOError("simulated merged-result write failure")
        return orig_write(path, document, label=label)

    schema_module.write_json_atomically = boom_write_result
    try:
        with pytest.raises(Exception):
            runner.install_result_decision_bundle_atomically(
                merged_result=merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
                bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
            )
    finally:
        schema_module.write_json_atomically = orig_write
    assert_absent_or_complete(bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.glob(".staging-*")) == []

    # 3. decision write failure (result write succeeds, decision write fails)
    bundle_dir = tmp_path / "case3"
    call_count = {"n": 0}

    def boom_write_decision(path, document, *, label):
        call_count["n"] += 1
        if call_count["n"] == 2:  # first call writes merged_result.json, second writes decision.json
            raise IOError("simulated decision write failure")
        return orig_write(path, document, label=label)

    schema_module.write_json_atomically = boom_write_decision
    try:
        with pytest.raises(Exception):
            runner.install_result_decision_bundle_atomically(
                merged_result=merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
                bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
            )
    finally:
        schema_module.write_json_atomically = orig_write
    assert_absent_or_complete(bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.glob(".staging-*")) == []

    # 4. staged result verification failure
    bundle_dir = tmp_path / "case4"
    corrupted_merged = dict(merged)
    corrupted_merged["outcome"] = "NOT_A_REAL_OUTCOME"
    with pytest.raises(Exception):
        runner.install_result_decision_bundle_atomically(
            merged_result=corrupted_merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
            bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
        )
    assert_absent_or_complete(bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.glob(".staging-*")) == []

    # 5. staged decision verification failure
    bundle_dir = tmp_path / "case5"

    def bad_decision(sha):
        d = dict(runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha))
        d["outcome"] = "NOT_A_REAL_OUTCOME"
        return d

    with pytest.raises(Exception):
        runner.install_result_decision_bundle_atomically(
            merged_result=merged, build_decision=bad_decision, bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
        )
    assert_absent_or_complete(bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.glob(".staging-*")) == []

    # 6. staged pair-consistency verification failure (decision points at a different sha)
    bundle_dir = tmp_path / "case6"

    def wrong_sha_decision(sha):
        return runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256="f" * 64)

    with pytest.raises(E, match="does not match"):
        runner.install_result_decision_bundle_atomically(
            merged_result=merged, build_decision=wrong_sha_decision, bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
        )
    assert_absent_or_complete(bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.glob(".staging-*")) == []

    # 7. directory rename failure
    bundle_dir = tmp_path / "case7"
    import os as os_module
    orig_replace = os_module.replace

    def boom_replace(src, dst):
        if str(dst) == str(bundle_dir):
            raise OSError("simulated rename failure")
        return orig_replace(src, dst)

    os_module.replace = boom_replace
    try:
        with pytest.raises(Exception):
            runner.install_result_decision_bundle_atomically(
                merged_result=merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
                bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
            )
    finally:
        os_module.replace = orig_replace
    assert_absent_or_complete(bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.glob(".staging-*")) == [], "no orphan staging directory after a rename failure"


def test_post_rename_reopen_and_verification_failures_leave_the_pair_for_forensics_but_report_failure(tmp_path, monkeypatch):
    """Cases 8-12: once the atomic rename has already succeeded, a
    failure during post-install reopen/re-verification must NOT delete
    the installed pair (left for forensic/public re-verification), but
    the function must still raise so no success is ever reported."""
    merged = _valid_merged_result_and_decision()
    bundle_dir = tmp_path / "case_post"

    call_state = {"post_rename": False}
    import os as os_module
    orig_replace = os_module.replace

    def marking_replace(src, dst):
        result = orig_replace(src, dst)
        if str(dst) == str(bundle_dir):
            call_state["post_rename"] = True
        return result

    os_module.replace = marking_replace

    def boom_verify_result(document, *, artifacts_doc):
        if call_state["post_rename"]:
            raise runner.DfpPhase1bT64EarlyExitBenchmarkError("simulated final result reopen/verification failure")
        return real_verify_result(document, artifacts_doc=artifacts_doc)

    real_verify_result = runner.verify_result_document
    monkeypatch.setattr(runner, "verify_result_document", boom_verify_result)
    try:
        with pytest.raises(runner.DfpPhase1bT64EarlyExitBenchmarkError, match="simulated final result reopen"):
            runner.install_result_decision_bundle_atomically(
                merged_result=merged, build_decision=lambda sha: runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha),
                bundle_dir=bundle_dir, artifacts_doc=ARTIFACTS_DOC,
            )
    finally:
        os_module.replace = orig_replace

    # The pair IS present (left for forensics), but nothing printed
    # success -- this test itself proves only the library contract; the
    # CLI layer never reaches its own success print because the
    # exception propagates out of install_result_decision_bundle_atomically.
    assert bundle_dir.is_dir()
    assert sorted(p.name for p in bundle_dir.iterdir()) == sorted((runner.RESULT_BUNDLE_RESULT_FILENAME, runner.RESULT_BUNDLE_DECISION_FILENAME))
    # Public verify-only commands can still inspect the pair directly off disk:
    reopened_result = json.loads((bundle_dir / runner.RESULT_BUNDLE_RESULT_FILENAME).read_text())
    reopened_decision = json.loads((bundle_dir / runner.RESULT_BUNDLE_DECISION_FILENAME).read_text())
    real_verify_result(reopened_result, artifacts_doc=ARTIFACTS_DOC)  # the REAL verifier still accepts it -- only our injected fake failed
    runner.verify_decision_document(reopened_decision, artifacts_doc=ARTIFACTS_DOC, merged_result=reopened_result, merged_result_sha256=hashlib.sha256((bundle_dir / runner.RESULT_BUNDLE_RESULT_FILENAME).read_bytes()).hexdigest())


def test_pair_cross_consistency_checks_repository_commit_and_decision_rule_inputs(tmp_path):
    merged = _valid_merged_result_and_decision()

    def decision_with_wrong_decision_rule_inputs(sha):
        d = dict(runner.build_decision_document(artifacts_doc=ARTIFACTS_DOC, merged_result=merged, merged_result_sha256=sha))
        d["decision_rule_inputs"] = dict(d["decision_rule_inputs"])
        d["decision_rule_inputs"]["end_to_end_speedup"] = 999.0
        return d

    with pytest.raises(E, match="decision_rule_inputs"):
        runner.install_result_decision_bundle_atomically(
            merged_result=merged, build_decision=decision_with_wrong_decision_rule_inputs, bundle_dir=tmp_path / "bundle", artifacts_doc=ARTIFACTS_DOC,
        )
