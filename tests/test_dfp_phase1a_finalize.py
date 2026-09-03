"""Regression tests for the DFP Phase 1A final-result installation repair
exposed by completed SLURM job 21063877: ``_install_final_result`` called
``install_result_bundle`` without its required keyword-only
``destination_dir`` argument, raising an uncaught ``TypeError`` AFTER all
500 images had already completed successfully, with no way to finalize
that complete checkpoint again without reprocessing every image.

Covers: independent reproduction of the missing-argument ``TypeError``
against the real (unmodified) ``install_result_bundle``; a signature-
contract test proving ``_install_final_result`` now supplies every
required keyword-only argument; the new deterministic
``ExecutionContext.result_bundle_dir`` destination and its containment
check; the finalize-only recovery path in ``execute_run`` (a checkpoint
already at the exact expected image count skips model/dataset/CUDA
construction and finalizes directly, with every relevant entry point
poisoned to prove it is never called); no-reprocessing / result-bundle
correctness / existing-destination / CLI exit-2 behavior.

CPU-only throughout. No real dataset, no model construction, no CUDA
initialization, no GPU job. Every fixture is a hand-built synthetic v2
checkpoint under ``tmp_path`` -- the real, complete production checkpoint
from job 21063877 is never read, copied, or touched by any test here."""

from __future__ import annotations

import datetime
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1a_artifact_schema as sch
import src.dfp_phase1a_checkpoint as ckpt
import src.dfp_phase1a_execute as ex
import src.dfp_phase1a_report as report
from src.dfp_phase1a_preflight import PreflightContext, PreflightTrace

ROOT = Path(__file__).resolve().parent.parent
CLASS_COUNT = 4
_PLACEHOLDER_SHA256 = "a" * 64
_PLACEHOLDER_COMMIT = "b" * 40


# ---------------------------------------------------------------------------
# Synthetic v2 checkpoint + PreflightContext builders. Every identity/
# binding field is a well-formed PLACEHOLDER (not a real file hash) --
# install_result_bundle/verify_result_bundle never re-validate a checkpoint
# against any live identity file (only internal format + self-consistency,
# confirmed by direct reading of src/dfp_phase1a_report.py and
# src/dfp_phase1a_checkpoint.py's validate_checkpoint_structure), so no
# fixture here depends on the repository's own frozen identities at all.
# ---------------------------------------------------------------------------


def _make_v2_arrays(n: int, class_count: int = CLASS_COUNT, *, distinctive_seed: int | None = None) -> dict[str, np.ndarray]:
    """A minimal, fully self-consistent v2 array set (all mass on class
    0, no dispute/persistence/reversion activity -- mirrors ``tests/
    test_dfp_phase1a_report_v2.py``'s own ``_make_v2_arrays`` exactly, so
    every relational invariant ``load_per_image_stats_npz_v2`` enforces
    is already satisfied). When ``distinctive_seed`` is given, only
    ``component_source_mass_flat`` (a float64 member with no relational
    constraint to any other field) is perturbed to distinctive non-zero
    values -- enough to prove a byte-identity comparison is meaningful,
    without touching any field the schema's own relational checks
    constrain."""
    a = ckpt.build_zero_row_arrays_v2(class_count=class_count)
    a["dataset_indices"] = np.arange(n, dtype=np.int64)
    for name in sch.SCALAR_INT64_MEMBER_NAMES_V2:
        if name != "dataset_indices":
            a[name] = np.zeros(n, dtype=np.int64)
    a["evaluated_pixel_count"] = np.full(n, 10, dtype=np.int64)
    for name in sch.SCALAR_FLOAT64_MEMBER_NAMES_V2:
        a[name] = np.zeros(n, dtype=np.float64)
    for variant in sch.PREDICTION_VARIANT_NAMES_V2:
        for stat in ("intersection", "predicted_pixels", "ground_truth_pixels", "union"):
            arr = np.zeros((n, class_count), dtype=np.int64)
            arr[:, 0] = 10
            a[sch.confusion_member_name_v2(variant, stat)] = arr
    a["component_count"] = np.ones(n, dtype=np.int64)
    a["component_offsets"] = np.arange(n + 1, dtype=np.int64)
    a["component_class_indices_flat"] = np.zeros(n, dtype=np.int64)
    a["component_pixel_areas_flat"] = np.full(n, 10, dtype=np.int64)
    a["component_patch_equivalent_areas_flat"] = np.full(n, 10.0, dtype=np.float64)
    a["component_source_mass_flat"] = (
        np.random.default_rng(distinctive_seed).random(n) + 1.0 if distinctive_seed is not None else np.ones(n, dtype=np.float64)
    )
    a["reversion_component_offsets"] = np.zeros(n + 1, dtype=np.int64)
    return a


def _build_synthetic_v2_checkpoint(
    tmp_path: Path, *, n: int, complete: bool, class_count: int = CLASS_COUNT, dirname: str = "checkpoints",
    selected_ids: list[str] | None = None, distinctive_seed: int | None = None,
) -> Path:
    """Hand-build a complete, schema-valid v2 checkpoint + per-image-stats
    manifest + NPZ bundle directly under ``tmp_path`` -- mirrors
    ``tests/test_dfp_phase1a_report_v2.py``'s own ``_install_v2`` pattern."""
    checkpoints_dir = tmp_path / dirname
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ids = selected_ids if selected_ids is not None else [f"{i:012d}" for i in range(1, n + 1)]

    arrays = _make_v2_arrays(n, class_count=class_count, distinctive_seed=distinctive_seed)
    npz_filename = f"dfp_phase1a_per_image_stats_v2_test_{n}_{dirname}.npz"
    npz_path = checkpoints_dir / npz_filename
    np.savez(npz_path, allow_pickle=False, **arrays)
    npz_sha256 = sch.sha256_file(npz_path)

    stats_manifest = {
        "schema": sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, "schema_version": sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2,
        "npz_filename": npz_filename, "npz_sha256": npz_sha256, "run_mode": "coco_stuff_500",
        "class_count": class_count, "class_names_digest": _PLACEHOLDER_SHA256,
        "image_count": n, "image_ids": list(ids[:n]),
    }
    manifest_path = checkpoints_dir / "stats_manifest.json"
    sch.write_json_atomically(manifest_path, stats_manifest, label="per-image-stats manifest")
    manifest_sha256 = sch.sha256_file(manifest_path)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checkpoint_fields = {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": _PLACEHOLDER_SHA256, "protocol_identity_sha256": _PLACEHOLDER_SHA256,
        "subset_manifest_sha256": _PLACEHOLDER_SHA256, "selected_ids_digest": _PLACEHOLDER_SHA256,
        "run_mode": "coco_stuff_500", "image_count_expected": n if complete else n + 1,
        "class_count": class_count, "class_names_digest": _PLACEHOLDER_SHA256,
        "horizon_order": list(sch.HORIZON_ORDER), "source_commit": _PLACEHOLDER_COMMIT,
        "required_ancestor_commit": _PLACEHOLDER_COMMIT, "bridge_checkpoint_sha256": _PLACEHOLDER_SHA256,
        "next_dataset_index": n, "completed_image_ids": list(ids[:n]), "images_completed_count": n,
        "total_windows_processed": max(n, 1), "complete": complete, "final_result_produced": False,
        "created_at_utc": now, "updated_at_utc": now,
        "per_image_stats_manifest_relative_path": "stats_manifest.json", "per_image_stats_manifest_sha256": manifest_sha256,
        "generation_nonce": "deadbeef" * 4,
    }
    sch.write_json_atomically(checkpoints_dir / "checkpoint.json", checkpoint_fields, label="checkpoint")
    return checkpoints_dir


def _build_preflight_context(
    *, output_root: Path, expected_count: int, class_count: int, selected_ids: tuple[str, ...],
    next_dataset_index: int, completed_image_ids: tuple[str, ...], checkpoint_arrays=None, mode: str = "resume",
) -> PreflightContext:
    """Directly construct a :class:`PreflightContext` (a plain frozen
    dataclass with no validation of its own) -- bypassing
    ``run_preflight``'s real Phase A/B/C ordering entirely, exactly the
    way these tests must: every field ``execute_run``/
    ``_install_final_result`` actually READS is set correctly; every
    other field is a well-formed placeholder execute_run never inspects."""
    return PreflightContext(
        run_mode="coco_stuff_500", dataset_key="coco_stuff", expected_count=expected_count, class_count=class_count,
        class_names_digest=_PLACEHOLDER_SHA256, selected_ids=selected_ids, selected_ids_digest=_PLACEHOLDER_SHA256,
        protocol_identity_sha256=_PLACEHOLDER_SHA256, artifact_schema_identity_sha256=_PLACEHOLDER_SHA256,
        qualitative_report_identity_sha256=_PLACEHOLDER_SHA256, subset_manifest_sha256=_PLACEHOLDER_SHA256,
        bridge_checkpoint_sha256=_PLACEHOLDER_SHA256, required_ancestor_commit=_PLACEHOLDER_COMMIT,
        source_commit=_PLACEHOLDER_COMMIT, evaluator_identity_name="e12-matched-k11-k12-t320",
        dataset_root=Path("/nonexistent/unused-by-finalize-only-path"), output_root=output_root, mode=mode,
        next_dataset_index=next_dataset_index, completed_image_ids=completed_image_ids,
        checkpoint_sha256=None, per_image_stats_manifest_sha256=None, checkpoint_arrays=checkpoint_arrays,
    )


def _make_context(checkpoints_dir: Path, *, expected_count: int, class_count: int, selected_ids, next_dataset_index, completed_image_ids) -> ex.ExecutionContext:
    preflight = _build_preflight_context(
        output_root=checkpoints_dir, expected_count=expected_count, class_count=class_count,
        selected_ids=tuple(selected_ids), next_dataset_index=next_dataset_index,
        completed_image_ids=tuple(completed_image_ids),
    )
    return ex.ExecutionContext(preflight=preflight, device="cpu", repo_root=ROOT, trace=PreflightTrace())


# ---------------------------------------------------------------------------
# Section 5: independent reproduction of the missing-argument TypeError.
# ---------------------------------------------------------------------------


def test_install_result_bundle_missing_destination_dir_raises_typeerror(tmp_path):
    """The exact pre-repair call convention, against the REAL, unmodified
    install_result_bundle -- proves the historical defect independently,
    never against the production checkpoint."""
    checkpoints_dir = _build_synthetic_v2_checkpoint(tmp_path, n=2, complete=True, dirname="repro")
    with pytest.raises(TypeError) as excinfo:
        report.install_result_bundle(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)  # destination_dir omitted, exactly like the pre-repair call
    assert "destination_dir" in str(excinfo.value)


def test_install_result_bundle_destination_dir_is_keyword_only_and_required():
    sig = inspect.signature(report.install_result_bundle)
    required_kwonly = {
        name for name, p in sig.parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
    }
    assert required_kwonly == {"destination_dir", "checkpoints_dir", "checkpoint_filename", "repo_root"}


# ---------------------------------------------------------------------------
# Section 7: signature-contract test -- fails against the pre-repair call.
# ---------------------------------------------------------------------------


def test_install_final_result_supplies_every_required_keyword_argument(monkeypatch, tmp_path):
    """Instruments the REAL install_result_bundle/verify_result_bundle
    with a signature-checking spy (never replaces their logic) and drives
    the REAL _install_final_result through them. Fails loudly if any
    required keyword-only argument of either function is missing --
    exactly what the pre-repair 3-argument call would have done."""
    checkpoints_dir = _build_synthetic_v2_checkpoint(tmp_path, n=2, complete=True, dirname="sig_contract")
    context = _make_context(
        checkpoints_dir, expected_count=2, class_count=CLASS_COUNT, selected_ids=["000000000001", "000000000002"],
        next_dataset_index=2, completed_image_ids=["000000000001", "000000000002"],
    )

    real_install = report.install_result_bundle
    real_verify = report.verify_result_bundle
    install_calls = []
    verify_calls = []

    def spying_install(**kwargs):
        sig = inspect.signature(real_install)
        bound = sig.bind(**kwargs)  # raises TypeError on any missing required keyword-only argument, exactly like the real call would
        bound.apply_defaults()
        install_calls.append(kwargs)
        return real_install(**kwargs)

    def spying_verify(bundle_dir):
        verify_calls.append(bundle_dir)
        return real_verify(bundle_dir)

    monkeypatch.setattr(ex, "install_result_bundle", spying_install, raising=False)
    monkeypatch.setattr("src.dfp_phase1a_report.install_result_bundle", spying_install)
    monkeypatch.setattr("src.dfp_phase1a_report.verify_result_bundle", spying_verify)

    ex._install_final_result(context)

    assert len(install_calls) == 1
    assert set(install_calls[0]) == {"destination_dir", "checkpoints_dir", "checkpoint_filename", "repo_root"}
    assert install_calls[0]["destination_dir"] == context.result_bundle_dir
    assert verify_calls == [context.result_bundle_dir]


# ---------------------------------------------------------------------------
# Section 6: authoritative destination derivation + containment.
# ---------------------------------------------------------------------------


def test_result_bundle_dir_is_deterministic_subdirectory_of_output_root(tmp_path):
    checkpoints_dir = tmp_path / "out" / "coco_stuff_500"
    checkpoints_dir.mkdir(parents=True)
    context = _make_context(checkpoints_dir, expected_count=1, class_count=CLASS_COUNT, selected_ids=["000000000001"], next_dataset_index=0, completed_image_ids=[])
    assert context.result_bundle_dir == checkpoints_dir / "result"
    assert context.result_bundle_dir != checkpoints_dir
    # Deterministic: rebuilding a context against the same output_root yields the identical path.
    context2 = _make_context(checkpoints_dir, expected_count=1, class_count=CLASS_COUNT, selected_ids=["000000000001"], next_dataset_index=0, completed_image_ids=[])
    assert context2.result_bundle_dir == context.result_bundle_dir


def test_result_bundle_dir_stays_contained_even_through_a_symlinked_output_root(tmp_path):
    """_resolve_result_bundle_dir's containment check: since the result
    directory is always the fixed literal child ``output_root / "result"``,
    Path.resolve() resolving a symlinked output_root and its child
    together can never break containment by construction -- confirmed
    here with output_root ITSELF a symlink to a real directory elsewhere
    under tmp_path, proving the check accepts this (correct) case rather
    than raising a false positive."""
    from src.dfp_phase1a_execute import _resolve_result_bundle_dir

    real_target = tmp_path / "real_output_root"
    real_target.mkdir()
    symlinked_root = tmp_path / "symlinked_output_root"
    try:
        symlinked_root.symlink_to(real_target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported in this environment")

    resolved = _resolve_result_bundle_dir(symlinked_root)
    assert resolved == symlinked_root / "result"
    assert resolved.resolve(strict=False) == (real_target / "result").resolve(strict=False)


# ---------------------------------------------------------------------------
# Sections 9-11: finalize-only recovery path, fully poisoned.
# ---------------------------------------------------------------------------


def test_finalize_only_recovery_never_touches_poisoned_entrypoints(monkeypatch, tmp_path):
    """A checkpoint already at the exact expected image count must
    finalize via the REAL execute_run without constructing the model,
    loading any weight, building the dataset, decoding any image,
    reading GT, building a graph, running recurrence, stitching,
    appending a row, or writing a new checkpoint generation. Every one
    of those entry points is poisoned (raises if called); result
    installation must be the ONLY thing that happens."""
    import diagnostics.run_k11_k12_stability as rk
    import src.dfp_phase1a_dataset_view as dv

    n = 3
    ids = ["000000000001", "000000000002", "000000000003"]
    checkpoints_dir = _build_synthetic_v2_checkpoint(tmp_path, n=n, complete=True, dirname="poison", selected_ids=ids)
    context = _make_context(checkpoints_dir, expected_count=n, class_count=CLASS_COUNT, selected_ids=ids, next_dataset_index=n, completed_image_ids=ids)

    def poison(name):
        def _raise(*a, **kw):
            raise AssertionError(f"{name} must never be called on the finalize-only recovery path")
        return _raise

    monkeypatch.setattr(rk, "_build_inference", poison("_build_inference"))
    monkeypatch.setattr(rk, "_extract_prepared_image", poison("_extract_prepared_image"))
    monkeypatch.setattr(dv, "build_manifest_execution_view", poison("build_manifest_execution_view"))
    monkeypatch.setattr(ex, "_process_one_image", poison("_process_one_image"))
    monkeypatch.setattr(ex, "_write_checkpoint", poison("_write_checkpoint"))
    monkeypatch.setattr(ex, "_V2Accumulator", poison("_V2Accumulator"))

    torch_was_already_imported = "torch" in sys.modules

    summary = ex.execute_run(context)

    assert summary == {"images_completed": n, "expected_count": n, "complete": True}
    # No new torch import: the finalize-only path itself never imports
    # torch (install_result_bundle/verify_result_bundle/checkpoint/
    # artifact-schema modules import none) -- if torch was already
    # present (e.g. pulled in by an earlier test in this same process),
    # that is unrelated to this call and not evidence of anything.
    if not torch_was_already_imported:
        assert "torch" not in sys.modules
        import torch  # local import only to check CUDA state without asserting torch was ever touched by execute_run itself
        assert not torch.cuda.is_initialized()

    assert (context.result_bundle_dir / "result.json").exists()
    assert (context.result_bundle_dir / "checkpoint.json").exists()


def test_finalize_only_recovery_incomplete_checkpoint_falls_through_to_normal_loop(monkeypatch, tmp_path):
    """The mirror image: next_dataset_index < expected_count must NOT
    take the finalize-only branch -- it must still reach _build_inference
    (poisoned here just to PROVE the normal path is the one taken, not to
    exercise it further)."""
    import diagnostics.run_k11_k12_stability as rk

    n = 3
    ids = ["000000000001", "000000000002", "000000000003"]
    checkpoints_dir = _build_synthetic_v2_checkpoint(tmp_path, n=1, complete=False, dirname="incomplete", selected_ids=ids)

    reached = {"build_inference": False}

    def fake_build_inference(*a, **kw):
        reached["build_inference"] = True
        raise RuntimeError("stop here -- proves the normal per-image path was reached, not the finalize-only one")

    monkeypatch.setattr(rk, "_build_inference", fake_build_inference)
    context = _make_context(checkpoints_dir, expected_count=n, class_count=CLASS_COUNT, selected_ids=ids, next_dataset_index=1, completed_image_ids=["000000000001"])

    with pytest.raises(RuntimeError, match="stop here"):
        ex.execute_run(context)
    assert reached["build_inference"] is True


# ---------------------------------------------------------------------------
# Section 11/12: no reprocessing + result-bundle correctness.
# ---------------------------------------------------------------------------


def test_finalize_only_recovery_produces_byte_identical_arrays_and_no_new_generation(tmp_path):
    n = 4
    ids = [f"{i:012d}" for i in range(1, n + 1)]
    checkpoints_dir = _build_synthetic_v2_checkpoint(tmp_path, n=n, complete=True, dirname="noreproc", selected_ids=ids, distinctive_seed=777)
    context = _make_context(checkpoints_dir, expected_count=n, class_count=CLASS_COUNT, selected_ids=ids, next_dataset_index=n, completed_image_ids=ids)

    source_checkpoint_json = (checkpoints_dir / "checkpoint.json").read_bytes()
    source_manifest_json = (checkpoints_dir / "stats_manifest.json").read_bytes()
    npz_name = json.loads(source_manifest_json)["npz_filename"]
    source_npz_bytes = (checkpoints_dir / npz_name).read_bytes()
    source_arrays = {k: np.array(v) for k, v in np.load(checkpoints_dir / npz_name).items()}

    files_before = sorted(p.name for p in checkpoints_dir.iterdir())
    summary = ex.execute_run(context)
    assert summary["complete"] is True
    files_after = sorted(p.name for p in checkpoints_dir.iterdir())

    # Source checkpoint/manifest/NPZ byte-identical -- no new generation
    # was written into checkpoints_dir; only context.result_bundle_dir
    # (a NEW sibling directory) gained files.
    assert (checkpoints_dir / "checkpoint.json").read_bytes() == source_checkpoint_json
    assert (checkpoints_dir / "stats_manifest.json").read_bytes() == source_manifest_json
    assert (checkpoints_dir / npz_name).read_bytes() == source_npz_bytes
    assert [f for f in files_after if f != "result"] == files_before

    verified = report.verify_result_bundle(context.result_bundle_dir)
    assert verified["checkpoint"]["completed_image_ids"] == ids
    assert set(verified["arrays"].keys()) == set(sch.ALL_NPZ_MEMBER_NAMES_V2)
    for name, arr in source_arrays.items():
        np.testing.assert_array_equal(verified["arrays"][name], arr, err_msg=f"member {name!r} changed during finalize-only installation")


# ---------------------------------------------------------------------------
# Section 15: existing-destination behavior.
# ---------------------------------------------------------------------------


def test_finalize_second_invocation_after_success_fails_cleanly_never_corrupts(tmp_path):
    n = 2
    ids = ["000000000001", "000000000002"]
    checkpoints_dir = _build_synthetic_v2_checkpoint(tmp_path, n=n, complete=True, dirname="repeat", selected_ids=ids)
    context = _make_context(checkpoints_dir, expected_count=n, class_count=CLASS_COUNT, selected_ids=ids, next_dataset_index=n, completed_image_ids=ids)

    ex.execute_run(context)
    installed_result_before = (context.result_bundle_dir / "result.json").read_bytes()

    with pytest.raises(report.DfpPhase1aReportError, match="already exists"):
        ex.execute_run(context)

    assert (context.result_bundle_dir / "result.json").read_bytes() == installed_result_before
    report.verify_result_bundle(context.result_bundle_dir)  # still verifies cleanly -- untouched


def test_finalize_destination_is_a_regular_file_refuses_cleanly(tmp_path):
    n = 1
    ids = ["000000000001"]
    checkpoints_dir = _build_synthetic_v2_checkpoint(tmp_path, n=n, complete=True, dirname="destfile", selected_ids=ids)
    context = _make_context(checkpoints_dir, expected_count=n, class_count=CLASS_COUNT, selected_ids=ids, next_dataset_index=n, completed_image_ids=ids)
    context.result_bundle_dir.write_text("not a directory")

    with pytest.raises(report.DfpPhase1aReportError, match="already exists"):
        ex.execute_run(context)
    assert context.result_bundle_dir.read_text() == "not a directory"  # untouched, never deleted/overwritten


# ---------------------------------------------------------------------------
# Section 14: CLI error boundary.
# ---------------------------------------------------------------------------


def test_cli_execute_reports_operational_report_failure_as_exit_2_no_traceback(monkeypatch, tmp_path):
    import io, contextlib
    import diagnostics.run_dfp_phase1a as cli

    def fake_execute(args):
        # Simulate the ONLY kind of failure install_result_bundle can
        # legitimately raise here: its own domain error (never a bare
        # TypeError -- that class of defect is eliminated by the
        # signature-contract test above, not papered over at the CLI
        # boundary).
        raise report.DfpPhase1aReportError("result destination already exists; refusing to overwrite an existing result")

    orig = cli._DISPATCH["execute"]
    cli._DISPATCH["execute"] = fake_execute
    try:
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            try:
                rc = cli.main(["--repo-root", str(ROOT), "execute", "--run", "coco_stuff_500", "--mode", "resume", "--device", "cpu", "--output-root", str(tmp_path / "unused")])
            except SystemExit as e:
                rc = e.code
    finally:
        cli._DISPATCH["execute"] = orig

    stderr_text = stderr_buf.getvalue()
    assert rc == 2
    assert stderr_text.startswith("DFP PHASE 1A FAIL [execute]:")
    assert "Traceback" not in stderr_text


def test_cli_execute_does_not_silently_relabel_a_bare_typeerror():
    """A genuine programming defect (a bare TypeError, e.g. the historical
    missing-argument bug if ever reintroduced) must NOT be caught by the
    CLI's _KNOWN_ERRORS boundary -- it must propagate as an uncaught
    exception, never be silently mislabeled as an ordinary DFP PHASE 1A
    FAIL operational message."""
    import diagnostics.run_dfp_phase1a as cli
    assert not issubclass(TypeError, cli._KNOWN_ERRORS)
    assert TypeError not in cli._KNOWN_ERRORS


# ---------------------------------------------------------------------------
# Section 8: state-transition table (subset directly testable without a
# real per-image driver run).
# ---------------------------------------------------------------------------


def test_incomplete_checkpoint_absent_destination_refuses_finalization(tmp_path):
    checkpoints_dir = _build_synthetic_v2_checkpoint(tmp_path, n=1, complete=False, dirname="table_incomplete")
    dest = tmp_path / "table_incomplete_dest"
    with pytest.raises(report.DfpPhase1aReportError, match="not complete"):
        report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    assert not dest.exists()


def test_complete_result_false_absent_destination_installs_and_self_verifies(tmp_path):
    n = 2
    checkpoints_dir = _build_synthetic_v2_checkpoint(tmp_path, n=n, complete=True, dirname="table_complete")
    dest = tmp_path / "table_complete_dest"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    verified = report.verify_result_bundle(dest)
    assert verified["result"]["complete"] is True and verified["result"]["final"] is True


# ---------------------------------------------------------------------------
# No scientific/schema drift sentinel.
# ---------------------------------------------------------------------------


def test_no_new_schema_v2_member_introduced_by_this_repair():
    assert len(sch.ALL_NPZ_MEMBER_NAMES_V2) == 122


# ---------------------------------------------------------------------------
# Section 16: fresh-run normal completion still finalizes correctly.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fresh_small_synthetic_run_processes_both_images_then_finalizes_exactly_once(monkeypatch, tmp_path):
    """The real execute_run, real per-image loop (real _process_one_image,
    real recurrence -- small geometry, CPU-only), real checkpoint writer,
    for a 2-image synthetic manifest, ending in exactly one real
    install_result_bundle call at context.result_bundle_dir. No real
    dataset, no real model, no GPU: diagnostics.run_k11_k12_stability.
    _build_inference is monkeypatched (to avoid CUDA/weights) and
    ex._build_contracts is monkeypatched to the same small geometry every
    other synthetic test in this driver's suite uses (execute_run always
    builds its OWN contracts internally -- there is no injection point --
    so this is the only way to keep the per-image loop fast); everything
    else is the genuine driver code."""
    import diagnostics.run_k11_k12_stability as rk
    sys.path.insert(0, str(ROOT / "tests"))
    import test_dfp_phase1a_execute as tmod

    protocol_identity_module = __import__("src.dfp_phase1a_identity", fromlist=["load_identity"])
    protocol_identity = protocol_identity_module.load_identity(repo_root=ROOT)
    class_count = 6
    contracts = tmod._small_contracts(protocol_identity, class_count=class_count, patch_grid=(8, 8), patch_size=(2, 2), stride=(8, 8))
    inf_shape, ori_shape = (24, 16), (20, 13)

    fake_dataset, _log = tmod._make_realistic_native_dataset(class_count=class_count, inf_shape=inf_shape, ori_shape=ori_shape)
    fake_inference = tmod._make_fake_inference(class_count=class_count, patch_grid=contracts["stitching"].patch_grid)
    fake_inference.model._rng = np.random.default_rng(5555)

    def fake_build_inference(repo_root, e3_identity, device, *, log_dir):
        return fake_inference, fake_dataset

    monkeypatch.setattr(rk, "_build_inference", fake_build_inference)
    monkeypatch.setattr(ex, "_build_contracts", lambda protocol_identity, *, repo_root: contracts)

    ids = ["000000000139", "000000000724"]  # both real entries of _make_realistic_native_dataset's own img_infos
    checkpoints_dir = tmp_path / "fresh_run"
    context = _make_context(checkpoints_dir, expected_count=len(ids), class_count=class_count, selected_ids=ids, next_dataset_index=0, completed_image_ids=[])

    install_calls = []
    real_install = report.install_result_bundle
    monkeypatch.setattr(ex, "install_result_bundle", lambda **kw: (install_calls.append(kw), real_install(**kw))[1], raising=False)
    monkeypatch.setattr("src.dfp_phase1a_report.install_result_bundle", lambda **kw: (install_calls.append(kw), real_install(**kw))[1])

    summary = ex.execute_run(context)

    assert summary == {"images_completed": 2, "expected_count": 2, "complete": True}
    assert len(install_calls) == 1
    assert install_calls[0]["destination_dir"] == context.result_bundle_dir
    verified = report.verify_result_bundle(context.result_bundle_dir)
    assert verified["checkpoint"]["completed_image_ids"] == ids
    assert verified["checkpoint"]["images_completed_count"] == 2
    assert verified["result"]["complete"] is True and verified["result"]["final"] is True
