"""DFP Phase 1B: E26-authorized implementation of the T64-versus-T320
early-exit wall-clock benchmark whose scientific/operational contract is
frozen, unmodified, by E25 (``evaluation_identities/
e25_dfp_phase1b_t64_early_exit_benchmark.toml``).

This module implements NO new scientific mathematics: graph
construction (``models.dinotext.cover_dr.graph.build_directed_topk_graph``),
the finite-step recurrence kernel (``models.dinotext.cover_dr.
finite_step_regime.finite_step_propagate``), forward horizon stitching
(``src.dfp_phase1a_horizon_stitching.stitch_dfp_phase1a_horizons``), and
the final-resolution rescale (``src.dfp_phase1a_final_rescale_pullback.
apply_final_rescale_forward``) are all reused completely unmodified, at
their real, committed call signatures. This module's only job is
orchestration: build the fixed 24-window population, call those
kernels in the exact sequence and horizon E25 freezes, time and
validate the result, and assemble/verify the closed-schema trial/
merged-result/decision artifacts E25's own artifacts identity
(``evaluation_identities/e25_dfp_phase1b_t64_early_exit_benchmark_
artifacts.toml``) defines.

The single scientific difference between the two variants is the
recurrence horizon passed to ``finite_step_propagate``: T64 calls it
with ``steps=64, snapshot_steps=(64,)`` directly (never via
``src.dfp_phase1a_recurrence_capture.capture_dfp_phase1a_trajectory``,
which is permanently bound to E13's frozen ``steps=320`` and cannot
produce a genuine 64-step-only run); T320 calls the identical function
with ``steps=320, snapshot_steps=(320,)``. Because
``finite_step_propagate``'s own update loop is ``for step in range(1,
steps + 1)``, calling it with ``steps=64`` makes step 65 structurally
unreachable for that call -- not merely unread output of a longer run.

No exact masked-forward, component masking, membership pullback,
component contribution, solver, BALP candidate, or GT path is imported
or called anywhere in this module. No dataset/model/CUDA import happens
at module scope: every heavy import is local to a function body, and
``verify_protocol``/the manifest loader/the artifact verifiers/the
merge-and-decision path never import torch. Only the real per-window
window-input builder and inference construction (used exclusively
inside a real ``run_trial`` call, itself only ever invoked by the CLI's
``run-trial`` subcommand after preflight has already passed) ever touch
a real dataset, model, checkpoint, or CUDA device.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import secrets
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import src.dfp_phase1b_t64_early_exit_benchmark_identity as e25_identity
import src.dfp_phase1b_t64_early_exit_benchmark_implementation_identity as e26_identity
from src.dfp_phase1b_t64_early_exit_benchmark_identity import (
    COUNTERBALANCED_ORDER,
    DECISION_OUTCOMES,
    DEVELOPMENT_ACCURACY_DISCLOSURE_PP,
    END_TO_END_SPEEDUP_THRESHOLD,
    MEASURED_WINDOW_COUNT,
    T64_STEPS,
    T320_STEPS,
    TOTAL_MEASURED_TRIALS,
    VARIANTS,
    WARMUP_TRIALS,
    apply_decision_rule,
    exact_median_of_three,
    verify_count_relations,
    verify_declared_median,
    verify_declared_speedup,
    verify_memory_relations,
    verify_scientific_output_finiteness,
    verify_timing_accounting_identity,
    verify_within_variant_digest_equality,
)


class DfpPhase1bT64EarlyExitBenchmarkError(RuntimeError):
    """The single declared domain exception for every ordinary
    invariant violation in this module. Always fail closed. Never
    raised for, and never catches, KeyboardInterrupt / SystemExit /
    MemoryError -- those always propagate unmodified."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


STEPS_BY_VARIANT: dict[str, int] = {"T64_early_exit": T64_STEPS, "T320_reference": T320_STEPS}

TIMING_PHASE_FIELDS: tuple[str, ...] = (
    "process_total_wall_ns", "preflight_wall_ns", "dataset_construction_wall_ns",
    "model_construction_and_checkpoint_loading_wall_ns", "measured_window_loop_wall_ns",
    "artifact_validation_and_write_wall_ns", "unattributed_overhead_ns",
)
STAGE_NAMES: tuple[str, ...] = (
    "input_preparation", "unary_model_forward", "graph_construction",
    "canonical_recurrence", "stitching_and_rescale", "output_validation",
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}\Z")


@contextmanager
def _translate_identity_errors():
    """Narrow exception-wrapping seam: every call into
    ``src.dfp_phase1b_t64_early_exit_benchmark_identity``'s pure
    validation helpers (``verify_count_relations``,
    ``verify_timing_accounting_identity``, ``verify_memory_relations``,
    ``verify_scientific_output_finiteness``,
    ``verify_within_variant_digest_equality``, ``verify_declared_median``,
    ``verify_declared_speedup``, ``apply_decision_rule``) is wrapped with
    this, so this module's own public API exposes exactly ONE ordinary
    domain exception (``DfpPhase1bT64EarlyExitBenchmarkError``), never a
    leaked ``DfpPhase1bT64EarlyExitBenchmarkIdentityError``. Catches
    nothing else -- ``KeyboardInterrupt``/``SystemExit``/``MemoryError``
    are never subclasses of the one exception type named here and always
    propagate unmodified."""
    try:
        yield
    except e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkError(str(exc)) from exc


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"{label} must be a mapping, observed {type(value).__name__}")
    return value


def _require_closed_keys(mapping: Mapping[str, Any], expected_keys: Sequence[str], label: str) -> None:
    actual = frozenset(mapping.keys())
    expected = frozenset(expected_keys)
    if actual != expected:
        raise DfpPhase1bT64EarlyExitBenchmarkError(
            f"{label} has the wrong key set -- missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )


def _require_nonempty_str(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"{label} must be a non-empty exact string, observed {value!r}")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"{label} must be an exact bool, observed {type(value).__name__}")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"{label} must be a positive exact int, observed {value!r}")
    return value


def _require_hex64(value: Any, label: str) -> str:
    """Exact lowercase 64-hex-character digest format -- the same
    convention every ``hashlib...hexdigest()`` call in this repository
    already produces, and the same ``^[0-9a-f]{64}$`` pattern
    ``src.dfp_phase1b_t64_early_exit_benchmark_identity._HEX64`` already
    uses. Rejects wrong length, uppercase, non-hex characters,
    whitespace, a ``0x`` prefix, and any non-string type (int, bool,
    None, list, mapping)."""
    if type(value) is not str:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"{label} must be a string, observed {type(value).__name__}")
    if not _HEX64_RE.match(value):
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"{label} must be exactly 64 lowercase hex characters, observed {value!r}")
    return value


def _verify_device_record(record: Any, *, artifacts_doc: Mapping[str, Any]) -> None:
    _require_mapping(record, "device_record")
    expected = artifacts_doc["trial_schema"]["device_record_keys"]["list"]
    _require_closed_keys(record, expected, "device_record")
    _require_nonempty_str(record["device_type"], "device_record.device_type")
    _require_nonempty_str(record["gpu_model"], "device_record.gpu_model")
    _require_nonempty_str(record["gpu_uuid"], "device_record.gpu_uuid")
    _require_positive_int(record["cpu_count_visible"], "device_record.cpu_count_visible")
    _require_exact_bool(record["one_visible_cuda_device"], "device_record.one_visible_cuda_device")


def _verify_software_record(record: Any, *, artifacts_doc: Mapping[str, Any]) -> None:
    _require_mapping(record, "software_record")
    expected = artifacts_doc["trial_schema"]["software_record_keys"]["list"]
    _require_closed_keys(record, expected, "software_record")
    for key in expected:
        _require_nonempty_str(record[key], f"software_record.{key}")


def _verify_precision_record(record: Any, *, artifacts_doc: Mapping[str, Any]) -> None:
    _require_mapping(record, "precision_record")
    expected = artifacts_doc["trial_schema"]["precision_record_keys"]["list"]
    _require_closed_keys(record, expected, "precision_record")
    for key in expected:
        _require_nonempty_str(record[key], f"precision_record.{key}")


def _verify_deterministic_settings_record(record: Any, *, artifacts_doc: Mapping[str, Any]) -> None:
    _require_mapping(record, "deterministic_settings_record")
    expected = artifacts_doc["trial_schema"]["deterministic_settings_record_keys"]["list"]
    _require_closed_keys(record, expected, "deterministic_settings_record")
    for key in ("torch_deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic"):
        _require_exact_bool(record[key], f"deterministic_settings_record.{key}")
    pythonhashseed = record["pythonhashseed"]
    if pythonhashseed is not None and type(pythonhashseed) is not str:
        raise DfpPhase1bT64EarlyExitBenchmarkError(
            f"deterministic_settings_record.pythonhashseed must be a string or null, observed {type(pythonhashseed).__name__}"
        )


def _verify_scientific_output_digest_format(scientific_output: Any) -> None:
    """Format-level validation independent of (and in addition to)
    ``verify_scientific_output_finiteness``'s finite/non-finite COUNT
    checks: exact hex-64 format for the collection digest and every
    per-window digest, exact window-identity key set (no duplicate,
    missing, or unknown window entry), and a mapping container type for
    ``per_window_output_digests``."""
    _require_hex64(scientific_output["scientific_output_collection_digest"], "scientific_output.scientific_output_collection_digest")
    per_window = _require_mapping(scientific_output["per_window_output_digests"], "scientific_output.per_window_output_digests")
    expected_keys = {str(i) for i in range(MEASURED_WINDOW_COUNT)}
    actual_keys = frozenset(per_window.keys())
    if actual_keys != expected_keys:
        raise DfpPhase1bT64EarlyExitBenchmarkError(
            f"scientific_output.per_window_output_digests has the wrong window-identity key set -- "
            f"missing={sorted(expected_keys - actual_keys)} unknown={sorted(actual_keys - expected_keys)}"
        )
    for key, value in per_window.items():
        _require_hex64(value, f"scientific_output.per_window_output_digests[{key!r}]")


# ---------------------------------------------------------------------
# Protocol / manifest verification -- torch-free, safe for preflight.
# ---------------------------------------------------------------------


def verify_protocol(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Loads and fully verifies E25 (closed schema + relational
    hashes), E25's artifacts identity, and E26 (this implementation's
    own authority). Never imports torch. Raises this module's own
    domain exception, translated from either loader's own, on any
    failure."""
    try:
        e25_doc = e25_identity.load_identity(repo_root=repo_root)
        artifacts_doc = e25_identity.load_artifacts_identity(repo_root=repo_root)
        e26_doc = e26_identity.load_identity(repo_root=repo_root)
    except (e25_identity.DfpPhase1bT64EarlyExitBenchmarkIdentityError, e26_identity.DfpPhase1bT64EarlyExitBenchmarkImplementationIdentityError) as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"protocol verification failed: {exc}") from exc
    return e25_doc, e26_doc, artifacts_doc


def load_and_verify_manifest(repo_root: Path, *, e25_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Loads the frozen E17 24-window manifest, verifies its file
    SHA-256 and both internal digests against E25's own
    ``population_binding``, and independently re-verifies the
    manifest's own internal digest self-consistency via the committed
    ``src.dfp_phase1b_chunk_benchmark.verify_benchmark_manifest``.
    Never imports torch."""
    population = e25_doc["population_binding"]
    manifest_path = Path(repo_root) / population["manifest_relative_path"]
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"cannot read benchmark manifest {manifest_path}: {exc}") from exc
    observed_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_hash != population["manifest_sha256"]:
        raise DfpPhase1bT64EarlyExitBenchmarkError("benchmark manifest file SHA-256 disagrees with E25 population_binding.manifest_sha256")

    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"benchmark manifest {manifest_path} is not valid JSON: {exc}") from exc

    from src.dfp_phase1b_chunk_benchmark import DfpPhase1bChunkBenchmarkError, verify_benchmark_manifest

    try:
        verify_benchmark_manifest(manifest)
    except DfpPhase1bChunkBenchmarkError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"benchmark manifest failed internal verification: {exc}") from exc

    if manifest["canonical_manifest_digest"] != population["canonical_manifest_digest"]:
        raise DfpPhase1bT64EarlyExitBenchmarkError("benchmark manifest canonical_manifest_digest disagrees with E25 population_binding")
    if manifest["benchmark_record_digest"] != population["benchmark_record_digest"]:
        raise DfpPhase1bT64EarlyExitBenchmarkError("benchmark manifest benchmark_record_digest disagrees with E25 population_binding")
    if len(manifest["benchmark_records"]) != MEASURED_WINDOW_COUNT:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"benchmark manifest has {len(manifest['benchmark_records'])} records, expected {MEASURED_WINDOW_COUNT}")
    return manifest


def benchmark_records_in_canonical_order(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The 24 measured records, strictly ordered by ``benchmark_slot``
    ascending (E25's own frozen ``deterministic_order``). Rejects a
    manifest whose records are not already exactly slot-ordered 0..23,
    rather than silently re-sorting one that is not."""
    records = list(manifest["benchmark_records"])
    for position, record in enumerate(records):
        if record["benchmark_slot"] != position:
            raise DfpPhase1bT64EarlyExitBenchmarkError(
                f"benchmark_records[{position}].benchmark_slot={record['benchmark_slot']} -- manifest is not in canonical slot order"
            )
    return records


# ---------------------------------------------------------------------
# Repository readiness (identity + manifest + git ancestry/clean-
# worktree), shared verbatim by preflight and run-trial -- never
# duplicated. Reuses the generic, E17-independent git-state helpers
# from src.dfp_phase1b_chunk_benchmark_identity rather than
# reimplementing ancestry/clean-worktree checking a third time in this
# repository.
# ---------------------------------------------------------------------


def _reject_lexical_path_traversal(path: Path) -> None:
    for part in Path(path).parts:
        if part == "..":
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"path {path} contains a lexical '..' component -- refused")


@dataclass(frozen=True)
class ReadinessResult:
    repo_root: Path
    e25_doc: dict[str, Any]
    e26_doc: dict[str, Any]
    artifacts_doc: dict[str, Any]
    manifest: dict[str, Any]
    repository_commit: str
    code_hashes: Mapping[str, str]
    identity_sha256: str
    artifacts_identity_sha256: str
    manifest_sha256: str


def code_binding_hashes(*, repo_root: Path) -> dict[str, str]:
    """The four benchmark-code SHA-256 values, re-read live from disk
    every call -- never trusted from any previously-computed value."""
    paths = {
        "identity_loader_sha256": Path(repo_root) / "src/dfp_phase1b_t64_early_exit_benchmark_identity.py",
        "runner_sha256": Path(repo_root) / "src/dfp_phase1b_t64_early_exit_benchmark.py",
        "cli_sha256": Path(repo_root) / "diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py",
        "launcher_sha256": Path(repo_root) / "dfp_phase1b_t64_early_exit_benchmark_h100.sbatch",
    }
    out: dict[str, str] = {}
    for field, path in paths.items():
        try:
            out[field] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"cannot read {field} source {path}: {exc}") from exc
    return out


def perform_repository_readiness_check(
    *, repo_root: Path, output_path: Path | None = None,
) -> ReadinessResult:
    """The sole repository-readiness check used by both ``preflight``
    and ``run-trial``. Performs, entirely before any torch/CUDA import
    or output creation: repository-root existence, E25/E26/manifest
    verification, current-HEAD validity, E26's own
    ``required_ancestor_commit`` (the committed E25 HEAD) being an
    ancestor of (or equal to) HEAD, full worktree cleanliness, and (if
    an output path was given) that it is lexically safe and absent.

    There is no overwrite/force override anywhere in this function --
    E25/E26 both freeze ``overwrite_authorized = false`` /
    ``force_mode_authorized = false`` unconditionally, so an existing
    destination is always rejected here, before any dataset/model/CUDA
    work, with no parameter or flag able to relax it."""
    try:
        resolved_repo_root = Path(repo_root).resolve(strict=True)
    except OSError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"repository root {repo_root} does not exist or cannot be resolved: {exc}") from exc
    if not resolved_repo_root.is_dir():
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"repository root {resolved_repo_root} is not a directory")

    e25_doc, e26_doc, artifacts_doc = verify_protocol(resolved_repo_root)
    manifest = load_and_verify_manifest(resolved_repo_root, e25_doc=e25_doc)

    from src.dfp_phase1b_chunk_benchmark_identity import (
        DfpPhase1bChunkBenchmarkIdentityError,
        require_clean_worktree,
        verify_ancestor_relation,
    )

    try:
        repository_commit = require_clean_worktree(resolved_repo_root)
        verify_ancestor_relation(
            resolved_repo_root, required_ancestor_commit=e26_doc["identity"]["required_ancestor_commit"],
            repository_commit=repository_commit,
        )
    except DfpPhase1bChunkBenchmarkIdentityError as exc:
        raise DfpPhase1bT64EarlyExitBenchmarkError(str(exc)) from exc

    if output_path is not None:
        _reject_lexical_path_traversal(output_path)
        if output_path.exists():
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"refusing to overwrite existing output {output_path} (no overwrite/force mode authorized)")

    code_hashes = code_binding_hashes(repo_root=resolved_repo_root)
    identity_sha256 = hashlib.sha256((resolved_repo_root / e25_identity.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    artifacts_identity_sha256 = hashlib.sha256((resolved_repo_root / e25_identity.ARTIFACTS_IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256((resolved_repo_root / e25_doc["population_binding"]["manifest_relative_path"]).read_bytes()).hexdigest()
    return ReadinessResult(
        repo_root=resolved_repo_root, e25_doc=e25_doc, e26_doc=e26_doc, artifacts_doc=artifacts_doc, manifest=manifest,
        repository_commit=repository_commit, code_hashes=code_hashes,
        identity_sha256=identity_sha256, artifacts_identity_sha256=artifacts_identity_sha256, manifest_sha256=manifest_sha256,
    )


# ---------------------------------------------------------------------
# Per-window pipeline. Real scientific kernels (build_directed_topk_
# graph, finite_step_propagate, stitch_dfp_phase1a_horizons,
# apply_final_rescale_forward) are always the genuine, unmodified,
# imported functions -- never faked, even in CPU-only tests: they are
# pure, device-agnostic tensor math with no model/dataset/CUDA
# dependency of their own, so a synthetic small tensor exercises the
# exact real call path. Only the unary forward (which requires the real
# backbone/CLIP model) is ever injectable.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class WindowInputs:
    """Everything one window's canonical pipeline needs, already
    resolved by the (real-or-fake) window-input builder: a ready-to-
    forward crop tensor and the geometry needed to stitch/rescale its
    own output back to a target resolution."""

    crop_tensor: Any  # [1, 3, crop_h, crop_w]
    crop_height: int
    crop_width: int
    ori_shape: tuple[int, int]


WindowInputsBuilder = Callable[[Mapping[str, Any]], WindowInputs]


def _default_unary_forward(inference: Any, crop_tensor: Any) -> Any:
    """The real per-window unary forward: exactly one
    ``generate_patch_snapshot`` call, under ``torch.no_grad()``,
    identical to every other production caller in this repository
    (``src.dfp_phase1a_execute._process_one_window_dfp``,
    ``src.dfp_phase1b_calibration_run._process_one_window_for_
    calibration``)."""
    import torch

    with torch.no_grad():
        return inference.model.generate_patch_snapshot(crop_tensor, inference.text_embedding)


@dataclass(frozen=True)
class StageTimings:
    input_preparation_ns: int
    unary_model_forward_ns: int
    graph_construction_ns: int
    canonical_recurrence_ns: int
    stitching_and_rescale_ns: int
    output_validation_ns: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_preparation": self.input_preparation_ns, "unary_model_forward": self.unary_model_forward_ns,
            "graph_construction": self.graph_construction_ns, "canonical_recurrence": self.canonical_recurrence_ns,
            "stitching_and_rescale": self.stitching_and_rescale_ns, "output_validation": self.output_validation_ns,
        }


def process_one_window(
    *, record: Mapping[str, Any], window_inputs_builder: WindowInputsBuilder, inference: Any,
    steps: int, alpha: float, k: int, affinity_power: float, class_count: int,
    stitching_geometry: Any, final_resolution_contract: Any,
    unary_forward_fn: Callable[[Any, Any], Any] = _default_unary_forward,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> tuple[Any, StageTimings]:
    """One window's full canonical pipeline: input_preparation ->
    unary_model_forward -> graph_construction -> canonical_recurrence
    -> stitching_and_rescale -> output_validation. ``steps`` is the
    ONLY thing that differs between T64 (64) and T320 (320) -- every
    other argument, and every kernel called, is identical between the
    two variants.

    Stitching reuses the real, unmodified ``stitch_dfp_phase1a_
    horizons`` against a synthetic single-window ``SlidingWindowPlan``
    sized exactly to this window's own crop (image_size == crop_size ==
    stride == the window's crop dimensions), so the plan has exactly
    one window whose accumulation_slice covers the entire synthetic
    canvas -- a well-defined, fully-covered stitching input that needs
    no sibling windows from the same source image (this benchmark's
    fixed population deliberately need not include every window of any
    one image). ``stitching_geometry`` carries the real, E13-frozen
    crop/stride/patch_grid/patch_size/align_corners; only its
    ``horizons`` field is narrowed here (via ``dataclasses.replace``,
    never a hand-rebuilt contract) to the single captured step, since
    ``stitch_dfp_phase1a_horizons`` itself is fully general over any
    horizon set and this call captures only one."""
    import torch

    from models.dinotext.cover_dr.finite_step_regime import finite_step_propagate
    from models.dinotext.cover_dr.graph import build_directed_topk_graph
    from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize
    from src.dfp_phase1a_final_rescale_pullback import apply_final_rescale_forward
    from src.dfp_phase1a_horizon_stitching import WindowHorizonCaptures, stitch_dfp_phase1a_horizons

    t0 = clock()
    window_inputs = window_inputs_builder(record)
    t1 = clock()

    snapshot = unary_forward_fn(inference, window_inputs.crop_tensor)
    s0 = snapshot.unary_scores[0]
    if s0.shape[1] != class_count:
        raise DfpPhase1bT64EarlyExitBenchmarkError(
            f"window benchmark_slot={record.get('benchmark_slot')}: unary snapshot class dimension {s0.shape[1]} disagrees with class_count {class_count}"
        )
    dino_features = snapshot.dino_features[0]
    t2 = clock()

    graph = build_directed_topk_graph(dino_features, k=k, affinity_power=affinity_power)
    t3 = clock()

    trace = finite_step_propagate(graph, s0, alpha=alpha, steps=steps, snapshot_steps=(steps,))
    if trace.steps_completed != steps:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"finite_step_propagate completed {trace.steps_completed} steps, expected exactly {steps}")
    p_final = trace.snapshots[steps]
    t4 = clock()

    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(window_inputs.crop_height, window_inputs.crop_width),
        crop_size=SpatialSize(window_inputs.crop_height, window_inputs.crop_width),
        stride=SpatialSize(window_inputs.crop_height, window_inputs.crop_width),
    )
    single_horizon_geometry = dataclasses.replace(stitching_geometry, horizons=(steps,))
    stitched = stitch_dfp_phase1a_horizons(
        plan, (WindowHorizonCaptures(window_index=0, captures={steps: p_final}),),
        contract=single_horizon_geometry, class_count=class_count,
    )[steps]
    rescaled = apply_final_rescale_forward(
        stitched, img_shape=(window_inputs.crop_height, window_inputs.crop_width),
        ori_shape=window_inputs.ori_shape, contract=final_resolution_contract,
    )
    t5 = clock()

    if not torch.isfinite(rescaled).all():
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"window benchmark_slot={record.get('benchmark_slot')}: output_validation failed -- non-finite value")
    t6 = clock()

    timings = StageTimings(
        input_preparation_ns=t1 - t0, unary_model_forward_ns=t2 - t1, graph_construction_ns=t3 - t2,
        canonical_recurrence_ns=t4 - t3, stitching_and_rescale_ns=t5 - t4, output_validation_ns=t6 - t5,
    )
    return rescaled, timings


# ---------------------------------------------------------------------
# Scientific-output digests. Reuses the existing repository-wide
# canonical-encoding/length-prefixed-digest convention
# (canonical_encode / digest_over_records, already committed in
# src.dfp_phase1b_chunk_benchmark) rather than a third, independently
# hand-rolled digest formula.
# ---------------------------------------------------------------------


def per_window_output_record(*, benchmark_slot: int, variant_id: str, output: Any) -> dict[str, Any]:
    array = output.detach().cpu().contiguous().numpy()
    return {
        "benchmark_slot": int(benchmark_slot), "variant_id": variant_id,
        "output_dtype": str(array.dtype), "output_shape": list(int(d) for d in array.shape),
        "output_bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }, array


def compute_scientific_output_digests(
    *, variant_id: str, outputs_in_slot_order: Sequence[tuple[int, Any]],
) -> tuple[str, dict[str, str], int, int]:
    """Returns ``(collection_digest, per_window_digests, finite_value_count, non_finite_value_count)``.
    ``outputs_in_slot_order`` must already be exactly the 24 measured
    windows' outputs, in ascending ``benchmark_slot`` order (this
    function never reorders)."""
    import numpy as np

    from src.dfp_phase1b_chunk_benchmark import canonical_encode, digest_over_records

    if len(outputs_in_slot_order) != MEASURED_WINDOW_COUNT:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"expected exactly {MEASURED_WINDOW_COUNT} window outputs, observed {len(outputs_in_slot_order)}")
    records = []
    per_window_digests: dict[str, str] = {}
    finite_count = 0
    non_finite_count = 0
    for position, (slot, output) in enumerate(outputs_in_slot_order):
        if slot != position:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"outputs_in_slot_order[{position}] has benchmark_slot={slot} -- not in canonical slot order")
        record, array = per_window_output_record(benchmark_slot=slot, variant_id=variant_id, output=output)
        records.append(record)
        per_window_digests[str(slot)] = hashlib.sha256(canonical_encode(record)).hexdigest()
        finite_mask = np.isfinite(array)
        finite_count += int(finite_mask.sum())
        non_finite_count += int(array.size - finite_mask.sum())
    collection_digest = digest_over_records([canonical_encode(r) for r in records])
    return collection_digest, per_window_digests, finite_count, non_finite_count


# ---------------------------------------------------------------------
# Memory telemetry. Real GPU+host provider matching E25's exact 4-field
# schema (host_peak_rss_bytes, gpu_peak_memory_allocated_bytes,
# gpu_peak_memory_reserved_bytes, gpu_total_memory_bytes). Uses only
# public torch.cuda accessors -- never the removed private
# torch._C._cuda_getDriverVersion symbol -- and the committed
# /proc/self/status VmRSS parser already used elsewhere in this
# repository for host RSS.
# ---------------------------------------------------------------------


class MemoryProvider:
    def reset_peak(self) -> None:  # pragma: no cover -- interface
        raise NotImplementedError

    def read(self) -> dict[str, int]:  # pragma: no cover -- interface
        raise NotImplementedError


class GpuHostMemoryProvider(MemoryProvider):
    """Real provider. Deferred torch import -- constructed only inside
    a real ``run-trial``, after preflight has already succeeded."""

    def __init__(self) -> None:
        import torch

        self._torch = torch

    def reset_peak(self) -> None:
        self._torch.cuda.reset_peak_memory_stats()

    def read(self) -> dict[str, int]:
        from src.dfp_phase1b_chunk_benchmark import parse_proc_self_status_kib_fields

        torch = self._torch
        torch.cuda.synchronize()
        gpu_peak_allocated = int(torch.cuda.max_memory_allocated())
        gpu_peak_reserved = int(torch.cuda.max_memory_reserved())
        gpu_total = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
        try:
            status_text = Path("/proc/self/status").read_bytes().decode("utf-8")
            parsed = parse_proc_self_status_kib_fields(status_text, required_fields=("VmHWM",))
            host_peak_rss = int(parsed["VmHWM"])
        except OSError as exc:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"cannot read /proc/self/status for host RSS telemetry: {exc}") from exc
        return {
            "host_peak_rss_bytes": host_peak_rss, "gpu_peak_memory_allocated_bytes": gpu_peak_allocated,
            "gpu_peak_memory_reserved_bytes": gpu_peak_reserved, "gpu_total_memory_bytes": gpu_total,
        }


# ---------------------------------------------------------------------
# Device/software/precision/deterministic-settings records, shaped
# exactly to E25's trial_schema field lists.
# ---------------------------------------------------------------------


def real_e25_device_record() -> dict[str, Any]:
    import torch

    from src.dfp_phase1b_calibration_run import _query_cuda_driver_version

    if not torch.cuda.is_available():
        raise DfpPhase1bT64EarlyExitBenchmarkError("real_e25_device_record requires an available CUDA device")
    if torch.cuda.device_count() != 1:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"run-trial requires exactly one visible CUDA device, observed {torch.cuda.device_count()}")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    gpu_uuid = getattr(properties, "uuid", None)
    return {
        "device_type": "cuda", "gpu_model": torch.cuda.get_device_name(index), "gpu_uuid": str(gpu_uuid) if gpu_uuid is not None else "unavailable",
        "cpu_count_visible": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 0),
        "one_visible_cuda_device": True,
    }


def real_e25_software_record() -> dict[str, Any]:
    import torch

    from src.dfp_phase1b_calibration_run import _query_cuda_driver_version

    return {
        "python_version": __import__("platform").python_version(), "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda), "driver_version": str(_query_cuda_driver_version()),
    }


def real_e25_precision_record() -> dict[str, Any]:
    return {"compute_dtype": "float32", "output_dtype": "float32"}


def real_e25_deterministic_settings_record() -> dict[str, Any]:
    import torch

    return {
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


# ---------------------------------------------------------------------
# Trial document construction and verification.
# ---------------------------------------------------------------------


def _trial_top_keys(artifacts_doc: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(artifacts_doc["trial_schema"]["top_keys"]["list"])


def _result_top_keys(artifacts_doc: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(artifacts_doc["result_schema"]["top_keys"]["list"])


def _decision_top_keys(artifacts_doc: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(artifacts_doc["decision_schema"]["top_keys"]["list"])


def build_trial_document(
    *, artifacts_doc: Mapping[str, Any], identity_sha256: str, artifacts_identity_sha256: str, manifest_sha256: str,
    repository_commit: str, required_ancestor_commit: str, code_hashes: Mapping[str, str],
    variant_id: str, trial_index: int, repeat_index: int, is_warmup: bool,
    device_record: Mapping[str, Any], software_record: Mapping[str, Any], precision_record: Mapping[str, Any],
    deterministic_settings_record: Mapping[str, Any], counts: Mapping[str, Any], timing: Mapping[str, Any],
    memory: Mapping[str, Any], scientific_output: Mapping[str, Any], complete: bool, final: bool,
) -> dict[str, Any]:
    if variant_id not in VARIANTS:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"variant_id must be one of {VARIANTS}, observed {variant_id!r}")
    document = {
        "schema_name": artifacts_doc["trial_schema"]["schema_name"], "schema_version": artifacts_doc["trial_schema"]["schema_version"],
        "identity_sha256": identity_sha256, "artifacts_identity_sha256": artifacts_identity_sha256, "manifest_sha256": manifest_sha256,
        "repository_commit": repository_commit, "required_ancestor_commit": required_ancestor_commit,
        "repository_worktree_clean": True,
        "identity_loader_sha256": code_hashes["identity_loader_sha256"], "runner_sha256": code_hashes["runner_sha256"],
        "cli_sha256": code_hashes["cli_sha256"], "launcher_sha256": code_hashes["launcher_sha256"],
        "variant_id": variant_id, "trial_index": int(trial_index), "repeat_index": int(repeat_index), "is_warmup": bool(is_warmup),
        "device_record": dict(device_record), "software_record": dict(software_record), "precision_record": dict(precision_record),
        "deterministic_settings_record": dict(deterministic_settings_record), "counts": dict(counts), "timing": dict(timing),
        "memory": dict(memory), "scientific_output": dict(scientific_output), "complete": bool(complete), "final": bool(final),
    }
    if frozenset(document.keys()) != _trial_top_keys(artifacts_doc):
        raise DfpPhase1bT64EarlyExitBenchmarkError("internal error: trial document key set drifted from the E25 artifacts trial_schema")
    verify_trial_document(document, artifacts_doc=artifacts_doc)
    return document


def verify_trial_document(
    document: Mapping[str, Any], *, artifacts_doc: Mapping[str, Any],
    expected_identity_sha256: str | None = None, expected_manifest_sha256: str | None = None,
) -> Mapping[str, Any]:
    actual_keys = frozenset(document.keys())
    expected_keys = _trial_top_keys(artifacts_doc)
    if actual_keys != expected_keys:
        raise DfpPhase1bT64EarlyExitBenchmarkError(
            f"trial document has the wrong key set -- missing={sorted(expected_keys - actual_keys)} unknown={sorted(actual_keys - expected_keys)}"
        )
    if document["schema_name"] != artifacts_doc["trial_schema"]["schema_name"] or document["schema_version"] != artifacts_doc["trial_schema"]["schema_version"]:
        raise DfpPhase1bT64EarlyExitBenchmarkError("trial document schema_name/schema_version mismatch")
    if document["variant_id"] not in VARIANTS:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"trial document variant_id must be one of {VARIANTS}")
    if expected_identity_sha256 is not None and document["identity_sha256"] != expected_identity_sha256:
        raise DfpPhase1bT64EarlyExitBenchmarkError("trial document identity_sha256 disagrees with the expected E25 identity")
    if expected_manifest_sha256 is not None and document["manifest_sha256"] != expected_manifest_sha256:
        raise DfpPhase1bT64EarlyExitBenchmarkError("trial document manifest_sha256 disagrees with the expected benchmark manifest")
    if document["repository_worktree_clean"] is not True:
        raise DfpPhase1bT64EarlyExitBenchmarkError("a complete/final trial must have repository_worktree_clean == true")
    for name, value in (
        ("repository_commit", document["repository_commit"]), ("required_ancestor_commit", document["required_ancestor_commit"]),
    ):
        if type(value) is not str or len(value) != 40:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"trial document {name} must be a 40-hex commit id")
    for name in ("identity_loader_sha256", "runner_sha256", "cli_sha256", "launcher_sha256", "identity_sha256", "artifacts_identity_sha256", "manifest_sha256"):
        value = document[name]
        if type(value) is not str or len(value) != 64:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"trial document {name} must be a 64-hex SHA-256")

    # Exact schedule-metadata types -- `isinstance(x, int)` is deliberately
    # never used here: `isinstance(True, int)` is True in Python, which
    # would let a bool silently masquerade as trial_index/repeat_index
    # (True == 1, False == 0) and corrupt schedule/merge dict-keying.
    if type(document["trial_index"]) is not int or type(document["repeat_index"]) is not int:
        raise DfpPhase1bT64EarlyExitBenchmarkError("trial document trial_index/repeat_index must be exact non-boolean ints")
    if document["trial_index"] < 0 or document["repeat_index"] < 0:
        raise DfpPhase1bT64EarlyExitBenchmarkError("trial document trial_index/repeat_index must be non-negative")
    if type(document["is_warmup"]) is not bool:
        raise DfpPhase1bT64EarlyExitBenchmarkError("trial document is_warmup must be an exact bool")

    if document["is_warmup"]:
        if document["repeat_index"] != 0:
            raise DfpPhase1bT64EarlyExitBenchmarkError("a warm-up trial document must have repeat_index == 0")
    else:
        schedule_entry = {"trial_index": document["trial_index"], "variant": document["variant_id"], "repeat": document["repeat_index"]}
        if schedule_entry not in COUNTERBALANCED_ORDER:
            raise DfpPhase1bT64EarlyExitBenchmarkError(
                f"trial document (trial_index={document['trial_index']}, variant={document['variant_id']}, repeat={document['repeat_index']}) "
                "is not a member of E25's frozen counterbalanced schedule"
            )

    _verify_device_record(document["device_record"], artifacts_doc=artifacts_doc)
    _verify_software_record(document["software_record"], artifacts_doc=artifacts_doc)
    _verify_precision_record(document["precision_record"], artifacts_doc=artifacts_doc)
    _verify_deterministic_settings_record(document["deterministic_settings_record"], artifacts_doc=artifacts_doc)

    with _translate_identity_errors():
        verify_count_relations(document["counts"], variant_id=document["variant_id"])
        verify_timing_accounting_identity(document["timing"])
        verify_memory_relations(document["memory"])
        if not document["is_warmup"]:
            verify_scientific_output_finiteness(document["scientific_output"])
    _verify_scientific_output_digest_format(document["scientific_output"])
    if not document["complete"] or not document["final"]:
        raise DfpPhase1bT64EarlyExitBenchmarkError("a persisted trial document must be complete and final -- an aborted trial must never be installed")
    return document


# ---------------------------------------------------------------------
# Trial orchestration: fresh-process, single-trial execution. Every
# heavy dependency (dataset resources, inference, per-window input
# preparation) is dependency-injected via TrialProviders -- CPU-only
# tests supply small fakes for dataset/inference/window-input
# construction while still calling the REAL scientific kernels inside
# process_one_window.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TrialProviders:
    dataset_resources_builder: Callable[[Path, Sequence[str]], Any]
    inference_builder: Callable[[Any], Any]
    window_inputs_builder_factory: Callable[[Any], WindowInputsBuilder]
    stitching_geometry_factory: Callable[[Any], Any]
    final_resolution_contract_factory: Callable[[Any], Any]
    class_count_of: Callable[[Any], int]
    affinity_power_of: Callable[[Any], float]
    alpha_of: Callable[[Any], float]
    k_of: Callable[[Any], int] = lambda resources: 12
    unary_forward_fn: Callable[[Any, Any], Any] = _default_unary_forward
    memory_provider_factory: Callable[[], MemoryProvider] = GpuHostMemoryProvider
    device_record_fn: Callable[[], Mapping[str, Any]] = real_e25_device_record
    software_record_fn: Callable[[], Mapping[str, Any]] = real_e25_software_record
    precision_record_fn: Callable[[], Mapping[str, Any]] = real_e25_precision_record
    deterministic_settings_record_fn: Callable[[], Mapping[str, Any]] = real_e25_deterministic_settings_record
    cuda_synchronize: Callable[[], None] = lambda: None
    clock: Callable[[], int] = time.perf_counter_ns


def run_trial(
    *, variant_id: str, trial_index: int, repeat_index: int, is_warmup: bool, manifest: Mapping[str, Any],
    readiness: ReadinessResult, selected_ids: Sequence[str], providers: TrialProviders,
) -> dict[str, Any]:
    """Run exactly one fresh-process trial and return its closed-schema
    trial document. Never writes to disk itself -- callers install the
    returned document atomically. Measures ``measured_window_loop_
    wall_ns`` around exactly the 24-window loop (CUDA-synchronized on
    both boundaries via ``providers.cuda_synchronize``, a no-op by
    default so CPU-only tests need not fake CUDA), and
    ``process_total_wall_ns`` as the exact sum of every other phase
    field -- the accounting identity is true by construction, never
    merely asserted after the fact.

    Warm-up trials (``is_warmup=True``) are a separate, standalone
    fresh-process run of the SAME fixed 24-window population as a
    measured trial (one warm-up trial per variant, per E25's own
    ``execution_schedule.warmup``) -- never a same-process throwaway
    single-window call embedded inside a measured trial. They are
    excluded only from summary-statistic computation (``merge_trials``),
    never from schema/count/timing/memory verification."""
    if variant_id not in VARIANTS:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"variant_id must be one of {VARIANTS}, observed {variant_id!r}")
    if is_warmup:
        if repeat_index != 0:
            raise DfpPhase1bT64EarlyExitBenchmarkError("a warm-up trial must have repeat_index == 0")
    else:
        key = {"trial_index": trial_index, "variant": variant_id, "repeat": repeat_index}
        if key not in COUNTERBALANCED_ORDER:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"(trial_index={trial_index}, variant={variant_id}, repeat={repeat_index}) is not in E25's frozen counterbalanced schedule")

    steps = STEPS_BY_VARIANT[variant_id]
    clock = providers.clock
    process_start = clock()

    preflight_wall_ns = 0  # preflight already ran (via perform_repository_readiness_check) before this call

    t_ds0 = clock()
    resources = providers.dataset_resources_builder(readiness.repo_root, list(selected_ids))
    t_ds1 = clock()
    dataset_construction_wall_ns = t_ds1 - t_ds0

    t_model0 = clock()
    inference = providers.inference_builder(resources)
    t_model1 = clock()
    model_construction_and_checkpoint_loading_wall_ns = t_model1 - t_model0

    window_inputs_builder = providers.window_inputs_builder_factory(resources)
    stitching_geometry = providers.stitching_geometry_factory(resources)
    final_resolution_contract = providers.final_resolution_contract_factory(resources)
    class_count = providers.class_count_of(resources)
    affinity_power = providers.affinity_power_of(resources)
    alpha = providers.alpha_of(resources)
    k = providers.k_of(resources)

    memory_provider = providers.memory_provider_factory()
    memory_provider.reset_peak()

    records = benchmark_records_in_canonical_order(manifest)

    def process(record: Mapping[str, Any]) -> tuple[Any, StageTimings]:
        return process_one_window(
            record=record, window_inputs_builder=window_inputs_builder, inference=inference,
            steps=steps, alpha=alpha, k=k, affinity_power=affinity_power, class_count=class_count,
            stitching_geometry=stitching_geometry, final_resolution_contract=final_resolution_contract,
            unary_forward_fn=providers.unary_forward_fn, clock=clock,
        )

    model_forward_count = 0
    graph_construction_count = 0
    recurrence_steps_executed = 0
    stage_totals_ns: dict[str, int] = {name: 0 for name in STAGE_NAMES}
    outputs: list[tuple[int, Any]] = []

    providers.cuda_synchronize()
    window_loop_start = clock()
    for record in records:
        rescaled, stage_timings = process(record)
        model_forward_count += 1
        graph_construction_count += 1
        recurrence_steps_executed += steps
        for name, value in stage_timings.as_dict().items():
            stage_totals_ns[name] += value
        outputs.append((record["benchmark_slot"], rescaled))
    providers.cuda_synchronize()
    window_loop_stop = clock()
    measured_window_loop_wall_ns = window_loop_stop - window_loop_start

    memory_reading = memory_provider.read()

    t_write0 = clock()
    collection_digest, per_window_digests, finite_count, non_finite_count = compute_scientific_output_digests(
        variant_id=variant_id, outputs_in_slot_order=outputs,
    )
    t_write1 = clock()
    artifact_validation_and_write_wall_ns = t_write1 - t_write0

    process_stop = clock()
    accounted = (
        preflight_wall_ns + dataset_construction_wall_ns + model_construction_and_checkpoint_loading_wall_ns
        + measured_window_loop_wall_ns + artifact_validation_and_write_wall_ns
    )
    process_total_wall_ns = process_stop - process_start
    unattributed_overhead_ns = process_total_wall_ns - accounted
    if unattributed_overhead_ns < 0:
        raise DfpPhase1bT64EarlyExitBenchmarkError("internal error: negative unattributed_overhead_ns -- phase clocks overlapped")

    measured_window_loop_seconds = measured_window_loop_wall_ns / 1e9
    window_throughput_per_second = (MEASURED_WINDOW_COUNT / measured_window_loop_seconds) if measured_window_loop_seconds > 0 else float("inf")
    mean_seconds_per_window = measured_window_loop_seconds / MEASURED_WINDOW_COUNT

    timing = {
        "process_total_wall_ns": process_total_wall_ns, "preflight_wall_ns": preflight_wall_ns,
        "dataset_construction_wall_ns": dataset_construction_wall_ns,
        "model_construction_and_checkpoint_loading_wall_ns": model_construction_and_checkpoint_loading_wall_ns,
        "measured_window_loop_wall_ns": measured_window_loop_wall_ns,
        "artifact_validation_and_write_wall_ns": artifact_validation_and_write_wall_ns,
        "unattributed_overhead_ns": unattributed_overhead_ns,
        "measured_window_loop_seconds": measured_window_loop_seconds,
        "window_throughput_per_second": window_throughput_per_second, "mean_seconds_per_window": mean_seconds_per_window,
        "stage_timings_ns": stage_totals_ns,
    }
    counts = {
        "window_count": MEASURED_WINDOW_COUNT, "recurrence_steps_executed": recurrence_steps_executed,
        "model_forward_count": model_forward_count, "graph_construction_count": graph_construction_count,
    }
    scientific_output = {
        "scientific_output_collection_digest": collection_digest, "per_window_output_digests": per_window_digests,
        "finite_value_count": finite_count, "non_finite_value_count": non_finite_count,
    }

    return build_trial_document(
        artifacts_doc=readiness.artifacts_doc,
        identity_sha256=readiness.identity_sha256, artifacts_identity_sha256=readiness.artifacts_identity_sha256,
        manifest_sha256=readiness.manifest_sha256,
        repository_commit=readiness.repository_commit, required_ancestor_commit=readiness.e26_doc["identity"]["required_ancestor_commit"],
        code_hashes=readiness.code_hashes, variant_id=variant_id, trial_index=trial_index, repeat_index=repeat_index,
        is_warmup=is_warmup, device_record=providers.device_record_fn(), software_record=providers.software_record_fn(),
        precision_record=providers.precision_record_fn(), deterministic_settings_record=providers.deterministic_settings_record_fn(),
        counts=counts, timing=timing, memory=memory_reading, scientific_output=scientific_output, complete=True, final=True,
    )


# ---------------------------------------------------------------------
# Merge: CPU-safe, torch-free aggregation/verification of exactly the 6
# frozen measured trials plus the 2 warm-up trials into one closed-
# schema merged result. Selects no chunk size, computes descriptive
# summaries only over MEASURED trials (warm-up trials are verified for
# schema validity but never contribute to a summary statistic).
# ---------------------------------------------------------------------


def merge_trials(
    *, measured_trials: Sequence[Mapping[str, Any]], warmup_trials: Sequence[Mapping[str, Any]], artifacts_doc: Mapping[str, Any],
) -> dict[str, Any]:
    if len(measured_trials) != TOTAL_MEASURED_TRIALS:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"merge requires exactly {TOTAL_MEASURED_TRIALS} measured trials, observed {len(measured_trials)}")
    if len(warmup_trials) != 2:
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"merge requires exactly 2 warm-up trials, observed {len(warmup_trials)}")

    seen_keys: set[tuple[int, str, int]] = set()
    by_key: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    for trial in measured_trials:
        verify_trial_document(trial, artifacts_doc=artifacts_doc)
        if trial["is_warmup"]:
            raise DfpPhase1bT64EarlyExitBenchmarkError("a measured trial must not have is_warmup == true")
        schedule_entry = {"trial_index": trial["trial_index"], "variant": trial["variant_id"], "repeat": trial["repeat_index"]}
        if schedule_entry not in COUNTERBALANCED_ORDER:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"measured trial {schedule_entry} is not a member of E25's frozen counterbalanced schedule")
        key = (trial["trial_index"], trial["variant_id"], trial["repeat_index"])
        if key in seen_keys:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"duplicate measured trial for {key}")
        seen_keys.add(key)
        by_key[key] = trial
    expected_keys = {(e["trial_index"], e["variant"], e["repeat"]) for e in COUNTERBALANCED_ORDER}
    if seen_keys != expected_keys:
        raise DfpPhase1bT64EarlyExitBenchmarkError(
            f"measured trial set does not match the frozen schedule -- missing={sorted(expected_keys - seen_keys)} unexpected={sorted(seen_keys - expected_keys)}"
        )

    for warmup in warmup_trials:
        verify_trial_document(warmup, artifacts_doc=artifacts_doc)
        if not warmup["is_warmup"]:
            raise DfpPhase1bT64EarlyExitBenchmarkError("a warm-up trial must have is_warmup == true")
        if warmup["repeat_index"] != 0:
            raise DfpPhase1bT64EarlyExitBenchmarkError("a warm-up trial must have repeat_index == 0")
    warmup_variants = sorted(w["variant_id"] for w in warmup_trials)
    if warmup_variants != sorted(VARIANTS):
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"warm-up trials must cover exactly one per variant {sorted(VARIANTS)}, observed {warmup_variants}")

    first = measured_trials[0]
    shared_fields = (
        "identity_sha256", "artifacts_identity_sha256", "manifest_sha256", "repository_commit", "required_ancestor_commit",
        "identity_loader_sha256", "runner_sha256", "cli_sha256", "launcher_sha256",
        "device_record", "software_record", "precision_record", "deterministic_settings_record",
    )
    for trial in measured_trials[1:] + list(warmup_trials):
        for field in shared_fields:
            if trial[field] != first[field]:
                raise DfpPhase1bT64EarlyExitBenchmarkError(f"trials disagree on {field} -- not all trials executed the same benchmark code/repository/device/software state")

    per_variant_trials: dict[str, list[Mapping[str, Any]]] = {v: [] for v in VARIANTS}
    for key, trial in by_key.items():
        per_variant_trials[key[1]].append(trial)

    digest_equal: dict[str, bool] = {}
    for variant in VARIANTS:
        trials_v = sorted(per_variant_trials[variant], key=lambda t: t["repeat_index"])
        digests = [t["scientific_output"]["scientific_output_collection_digest"] for t in trials_v]
        with _translate_identity_errors():
            verify_within_variant_digest_equality(digests, variant_id=variant)
        digest_equal[variant] = True

    non_finite_total = sum(t["scientific_output"]["non_finite_value_count"] for t in list(measured_trials) + list(warmup_trials))

    per_variant_summary: dict[str, dict[str, Any]] = {}
    recurrence_median_seconds: dict[str, float | None] = {}
    for variant in VARIANTS:
        trials_v = per_variant_trials[variant]
        loop_seconds = [t["timing"]["measured_window_loop_seconds"] for t in trials_v]
        throughputs = [t["timing"]["window_throughput_per_second"] for t in trials_v]
        host_peaks = [t["memory"]["host_peak_rss_bytes"] for t in trials_v]
        gpu_alloc_peaks = [t["memory"]["gpu_peak_memory_allocated_bytes"] for t in trials_v]
        gpu_reserved_peaks = [t["memory"]["gpu_peak_memory_reserved_bytes"] for t in trials_v]
        with _translate_identity_errors():
            median_seconds = exact_median_of_three(loop_seconds)
            verify_declared_median(loop_seconds, median_seconds)
        per_variant_summary[variant] = {
            "median_measured_window_loop_seconds": median_seconds,
            "minimum_measured_window_loop_seconds": min(loop_seconds),
            "maximum_measured_window_loop_seconds": max(loop_seconds),
            "median_window_throughput_per_second": exact_median_of_three(throughputs),
            "maximum_host_peak_rss_bytes": max(host_peaks),
            "maximum_gpu_peak_memory_allocated_bytes": max(gpu_alloc_peaks),
            "maximum_gpu_peak_memory_reserved_bytes": max(gpu_reserved_peaks),
        }
        recurrence_ns = [t["timing"]["stage_timings_ns"].get("canonical_recurrence") for t in trials_v]
        recurrence_median_seconds[variant] = exact_median_of_three([n / 1e9 for n in recurrence_ns]) if all(n is not None for n in recurrence_ns) else None

    t320_median = per_variant_summary["T320_reference"]["median_measured_window_loop_seconds"]
    t64_median = per_variant_summary["T64_early_exit"]["median_measured_window_loop_seconds"]
    end_to_end_speedup = t320_median / t64_median
    with _translate_identity_errors():
        verify_declared_speedup(t320_median_seconds=t320_median, t64_median_seconds=t64_median, declared_speedup=end_to_end_speedup)

    t64_recurrence_speedup = None
    if recurrence_median_seconds["T64_early_exit"] is not None and recurrence_median_seconds["T320_reference"] is not None and recurrence_median_seconds["T64_early_exit"] > 0:
        t64_recurrence_speedup = recurrence_median_seconds["T320_reference"] / recurrence_median_seconds["T64_early_exit"]

    speedup = {"t64_speedup_over_t320": end_to_end_speedup, "t64_recurrence_speedup_over_t320": t64_recurrence_speedup}
    memory_summary = {
        variant: {
            "maximum_host_peak_rss_bytes": per_variant_summary[variant]["maximum_host_peak_rss_bytes"],
            "maximum_gpu_peak_memory_allocated_bytes": per_variant_summary[variant]["maximum_gpu_peak_memory_allocated_bytes"],
            "maximum_gpu_peak_memory_reserved_bytes": per_variant_summary[variant]["maximum_gpu_peak_memory_reserved_bytes"],
        }
        for variant in VARIANTS
    }
    digest_equivalence = {
        "t64_within_variant_digest_equal": digest_equal["T64_early_exit"], "t320_within_variant_digest_equal": digest_equal["T320_reference"],
        "non_finite_value_count_total": non_finite_total,
    }
    if digest_equivalence["non_finite_value_count_total"] != 0:
        raise DfpPhase1bT64EarlyExitBenchmarkError("non_finite_value_count_total must be 0")

    all_trials_complete = all(t["complete"] and t["final"] for t in list(measured_trials) + list(warmup_trials))
    decision_rule_inputs = {
        "all_trials_complete": all_trials_complete, "all_artifacts_verified": True,
        "within_variant_digests_identical": digest_equal["T64_early_exit"] and digest_equal["T320_reference"],
        "zero_non_finite_outputs": non_finite_total == 0, "exact_counts_verified": True, "no_oom": all_trials_complete,
        "memory_relations_pass": True, "t64_median_lower_than_t320": t64_median < t320_median,
        "end_to_end_speedup": end_to_end_speedup, "development_accuracy_disclosure_pp": DEVELOPMENT_ACCURACY_DISCLOSURE_PP,
    }
    with _translate_identity_errors():
        outcome = apply_decision_rule(
            all_trials_complete=decision_rule_inputs["all_trials_complete"], all_artifacts_verified=decision_rule_inputs["all_artifacts_verified"],
            within_variant_digests_identical=decision_rule_inputs["within_variant_digests_identical"],
            zero_non_finite_outputs=decision_rule_inputs["zero_non_finite_outputs"], exact_counts_verified=decision_rule_inputs["exact_counts_verified"],
            no_oom=decision_rule_inputs["no_oom"], memory_relations_pass=decision_rule_inputs["memory_relations_pass"],
            t64_median_seconds=t64_median, t320_median_seconds=t320_median, end_to_end_speedup=end_to_end_speedup,
            development_accuracy_disclosure_pp=decision_rule_inputs["development_accuracy_disclosure_pp"],
        )

    document = {
        "schema_name": artifacts_doc["result_schema"]["schema_name"], "schema_version": artifacts_doc["result_schema"]["schema_version"],
        "identity_sha256": first["identity_sha256"], "artifacts_identity_sha256": first["artifacts_identity_sha256"],
        "manifest_sha256": first["manifest_sha256"], "repository_commit": first["repository_commit"],
        "measured_trials": [dict(t) for t in measured_trials], "warmup_trials": [dict(t) for t in warmup_trials],
        "schedule_validation": {"counterbalanced_order_verified": True},
        "per_variant_summary": per_variant_summary, "speedup": speedup, "memory_summary": memory_summary,
        "digest_equivalence": digest_equivalence, "decision_rule_inputs": decision_rule_inputs, "outcome": outcome,
        "complete": True, "final": True,
    }
    if frozenset(document.keys()) != _result_top_keys(artifacts_doc):
        raise DfpPhase1bT64EarlyExitBenchmarkError("internal error: result document key set drifted from the E25 artifacts result_schema")
    if document["outcome"] not in DECISION_OUTCOMES:
        raise DfpPhase1bT64EarlyExitBenchmarkError("internal error: outcome not in the closed decision set")
    return document


def verify_result_document(document: Mapping[str, Any], *, artifacts_doc: Mapping[str, Any]) -> Mapping[str, Any]:
    """Independent re-verification of an installed merged result:
    recomputes the ENTIRE merge from the document's own embedded raw
    trials and requires every derived field to match exactly -- never a
    mere internal self-consistency check."""
    actual_keys = frozenset(document.keys())
    expected_keys = _result_top_keys(artifacts_doc)
    if actual_keys != expected_keys:
        raise DfpPhase1bT64EarlyExitBenchmarkError(
            f"result document has the wrong key set -- missing={sorted(expected_keys - actual_keys)} unknown={sorted(actual_keys - expected_keys)}"
        )
    if document["schema_name"] != artifacts_doc["result_schema"]["schema_name"] or document["schema_version"] != artifacts_doc["result_schema"]["schema_version"]:
        raise DfpPhase1bT64EarlyExitBenchmarkError("result document schema_name/schema_version mismatch")
    recomputed = merge_trials(measured_trials=document["measured_trials"], warmup_trials=document["warmup_trials"], artifacts_doc=artifacts_doc)
    for field in ("per_variant_summary", "speedup", "memory_summary", "digest_equivalence", "decision_rule_inputs", "outcome", "schedule_validation"):
        if document[field] != recomputed[field]:
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"result document field {field!r} does not recompute from its own embedded raw trials")
    return document


# ---------------------------------------------------------------------
# Decision artifact: a thin, independently-verifiable echo of the
# merged result's own decision_rule_inputs/outcome -- never a second,
# competing computation of either.
# ---------------------------------------------------------------------


def build_decision_document(*, artifacts_doc: Mapping[str, Any], merged_result: Mapping[str, Any], merged_result_sha256: str) -> dict[str, Any]:
    document = {
        "schema_name": artifacts_doc["decision_schema"]["schema_name"], "schema_version": artifacts_doc["decision_schema"]["schema_version"],
        "merged_result_sha256": merged_result_sha256, "decision_rule_inputs": dict(merged_result["decision_rule_inputs"]),
        "end_to_end_speedup_threshold": END_TO_END_SPEEDUP_THRESHOLD, "outcome": merged_result["outcome"],
        "proceed_does_not_authorize": list(artifacts_doc["decision_schema"]["proceed_does_not_authorize"]["list"]),
        "complete": True, "final": True,
    }
    if frozenset(document.keys()) != _decision_top_keys(artifacts_doc):
        raise DfpPhase1bT64EarlyExitBenchmarkError("internal error: decision document key set drifted from the E25 artifacts decision_schema")
    verify_decision_document(document, artifacts_doc=artifacts_doc, merged_result=merged_result, merged_result_sha256=merged_result_sha256)
    return document


def verify_decision_document(
    document: Mapping[str, Any], *, artifacts_doc: Mapping[str, Any],
    merged_result: Mapping[str, Any] | None = None, merged_result_sha256: str | None = None,
) -> Mapping[str, Any]:
    actual_keys = frozenset(document.keys())
    expected_keys = _decision_top_keys(artifacts_doc)
    if actual_keys != expected_keys:
        raise DfpPhase1bT64EarlyExitBenchmarkError(
            f"decision document has the wrong key set -- missing={sorted(expected_keys - actual_keys)} unknown={sorted(actual_keys - expected_keys)}"
        )
    if document["schema_name"] != artifacts_doc["decision_schema"]["schema_name"] or document["schema_version"] != artifacts_doc["decision_schema"]["schema_version"]:
        raise DfpPhase1bT64EarlyExitBenchmarkError("decision document schema_name/schema_version mismatch")
    if document["outcome"] not in DECISION_OUTCOMES:
        raise DfpPhase1bT64EarlyExitBenchmarkError("decision document outcome is not in the closed decision set")
    if document["end_to_end_speedup_threshold"] != END_TO_END_SPEEDUP_THRESHOLD:
        raise DfpPhase1bT64EarlyExitBenchmarkError("decision document end_to_end_speedup_threshold disagrees with the frozen E25 threshold")
    expected_proceed_list = list(artifacts_doc["decision_schema"]["proceed_does_not_authorize"]["list"])
    if list(document["proceed_does_not_authorize"]) != expected_proceed_list:
        raise DfpPhase1bT64EarlyExitBenchmarkError("decision document proceed_does_not_authorize disagrees with the frozen required list")
    if merged_result_sha256 is not None and document["merged_result_sha256"] != merged_result_sha256:
        raise DfpPhase1bT64EarlyExitBenchmarkError("decision document merged_result_sha256 disagrees with the expected merged result")
    if merged_result is not None:
        if document["decision_rule_inputs"] != merged_result["decision_rule_inputs"]:
            raise DfpPhase1bT64EarlyExitBenchmarkError("decision document decision_rule_inputs disagrees with the merged result's own field")
        if document["outcome"] != merged_result["outcome"]:
            raise DfpPhase1bT64EarlyExitBenchmarkError("decision document outcome disagrees with the merged result's own field")
    return document


# ---------------------------------------------------------------------
# Atomic artifact installation: stage in a uniquely-named sibling
# directory, self-verify staged bytes, install via exactly one atomic
# rename into an ABSENT destination, reopen, and publicly re-verify --
# identical to E25's own artifact_write_policy.atomic_write_mechanism.
# Never overwrites; never offers a force mode.
# ---------------------------------------------------------------------


def install_artifact_atomically(document: Mapping[str, Any], output_path: Path, *, verify_fn: Callable[[Mapping[str, Any]], Any]) -> None:
    output_path = Path(output_path)
    if output_path.exists():
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"refusing to overwrite existing artifact {output_path} (no overwrite, no force mode authorized)")
    from src.dfp_phase1a_artifact_schema import write_json_atomically

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_path.parent / f".staging-{output_path.name}-{os.getpid()}-{secrets.token_hex(8)}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        staged_path = staging_dir / output_path.name
        write_json_atomically(staged_path, document, label=str(output_path))
        with open(staged_path, "r", encoding="utf-8") as handle:
            staged_document = json.load(handle)
        verify_fn(staged_document)
        if output_path.exists():
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"refusing to overwrite existing artifact {output_path} (no overwrite, no force mode authorized)")
        os.replace(staged_path, output_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    with open(output_path, "r", encoding="utf-8") as handle:
        installed_document = json.load(handle)
    verify_fn(installed_document)


# ---------------------------------------------------------------------
# Pair-atomic result+decision bundle installation. E25's own artifacts
# identity is explicitly "storage-representation-only ... never a
# physical file layout" (its `artifact_write_policy.atomic_write_
# mechanism` speaks generically of "an atomic directory/file rename"),
# so a single bundle DIRECTORY containing both `merged_result.json` and
# `decision.json`, installed via one atomic directory rename, is fully
# compatible with E25's frozen policy -- this never amends E25 or
# invents a new schema field. Cross-document binding reuses E25's own
# frozen `decision_schema.merged_result_sha256` field exclusively --
# never a new, competing binding field.
# ---------------------------------------------------------------------

RESULT_BUNDLE_RESULT_FILENAME = "merged_result.json"
RESULT_BUNDLE_DECISION_FILENAME = "decision.json"


def _verify_bundle_pair_consistency(merged_result: Mapping[str, Any], decision: Mapping[str, Any], merged_result_sha256: str) -> None:
    """Cross-document consistency for one result/decision bundle.
    ``merged_result_sha256`` must be the SHA-256 of the merged result's
    OWN actual on-disk bytes (staged or installed, whichever this is
    being called for) -- never a value merely echoed from memory.
    Because ``merged_result`` has already independently passed
    ``verify_result_document`` (which recomputes summaries, speedup,
    digest_equivalence, decision_rule_inputs, and outcome from its own
    embedded raw trials) before this is ever called, binding the
    decision to these exact bytes transitively binds it to every one of
    those recomputed fields too -- repository_commit, E25/E26 bindings,
    manifest binding, trial-set digests, summary fields, speedup, every
    decision-rule gate value, and the selected outcome all flow through
    the SAME already-verified merged result these checks pin by hash."""
    if decision["merged_result_sha256"] != merged_result_sha256:
        raise DfpPhase1bT64EarlyExitBenchmarkError(
            "decision.merged_result_sha256 does not match the merged result's own actual bytes -- result/decision pair mismatch"
        )
    if decision["decision_rule_inputs"] != merged_result["decision_rule_inputs"]:
        raise DfpPhase1bT64EarlyExitBenchmarkError("decision.decision_rule_inputs disagrees with the bound merged result's own field")
    if decision["outcome"] != merged_result["outcome"]:
        raise DfpPhase1bT64EarlyExitBenchmarkError("decision.outcome disagrees with the bound merged result's own field")
    if not (merged_result["complete"] and merged_result["final"] and decision["complete"] and decision["final"]):
        raise DfpPhase1bT64EarlyExitBenchmarkError("both the merged result and the decision must be complete and final")


def install_result_decision_bundle_atomically(
    *, merged_result: Mapping[str, Any], build_decision: Callable[[str], Mapping[str, Any]],
    bundle_dir: Path, artifacts_doc: Mapping[str, Any],
) -> None:
    """Install ``merged_result.json`` and ``decision.json`` as one
    pair-atomic bundle: after this function returns successfully,
    ``bundle_dir`` contains exactly both files, cross-verified against
    each other; if it raises before the final directory rename,
    ``bundle_dir`` never exists at all and no staging residue is left
    behind. ``build_decision(merged_result_sha256)`` is called with the
    merged result's OWN actual staged-file SHA-256 (never a value
    computed from the in-memory object, and never duplicating
    ``write_json_atomically``'s own serialization format a second time)
    so the decision's frozen ``merged_result_sha256`` field always binds
    to real, already-written bytes.

    No overwrite, no force mode: a pre-existing ``bundle_dir`` is always
    rejected, both up front and immediately before the rename (closing
    the race window between the two checks). If verification fails
    AFTER the atomic rename (post-install reopen), the already-installed
    pair is deliberately left in place for forensic/public
    re-verification -- this function still raises, so the caller must
    never report success, but the pair itself is not deleted."""
    bundle_dir = Path(bundle_dir)
    _reject_lexical_path_traversal(bundle_dir)
    if bundle_dir.exists():
        raise DfpPhase1bT64EarlyExitBenchmarkError(f"refusing to overwrite existing result/decision bundle {bundle_dir} (no overwrite, no force mode authorized)")

    from src.dfp_phase1a_artifact_schema import write_json_atomically

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = bundle_dir.parent / f".staging-{bundle_dir.name}-{os.getpid()}-{secrets.token_hex(8)}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    renamed = False
    try:
        result_path = staging_dir / RESULT_BUNDLE_RESULT_FILENAME
        decision_path = staging_dir / RESULT_BUNDLE_DECISION_FILENAME

        write_json_atomically(result_path, merged_result, label=str(result_path))
        with open(result_path, "r", encoding="utf-8") as handle:
            staged_result = json.load(handle)
        verify_result_document(staged_result, artifacts_doc=artifacts_doc)
        staged_result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()

        decision = build_decision(staged_result_sha256)
        write_json_atomically(decision_path, decision, label=str(decision_path))
        with open(decision_path, "r", encoding="utf-8") as handle:
            staged_decision = json.load(handle)
        verify_decision_document(staged_decision, artifacts_doc=artifacts_doc)
        _verify_bundle_pair_consistency(staged_result, staged_decision, staged_result_sha256)

        staged_names = sorted(p.name for p in staging_dir.iterdir())
        if staged_names != sorted((RESULT_BUNDLE_RESULT_FILENAME, RESULT_BUNDLE_DECISION_FILENAME)):
            raise DfpPhase1bT64EarlyExitBenchmarkError(
                f"staging directory must contain exactly {RESULT_BUNDLE_RESULT_FILENAME} and {RESULT_BUNDLE_DECISION_FILENAME}, observed {staged_names}"
            )

        if bundle_dir.exists():
            raise DfpPhase1bT64EarlyExitBenchmarkError(f"refusing to overwrite existing result/decision bundle {bundle_dir} (no overwrite, no force mode authorized)")
        os.replace(staging_dir, bundle_dir)
        renamed = True
    finally:
        if not renamed:
            shutil.rmtree(staging_dir, ignore_errors=True)

    # Post-install: reopen both files from the FINAL directory and
    # re-verify publicly, independent of the staged-bytes checks above
    # (which only prove the staging directory was internally consistent
    # -- this proves the RENAMED, installed files are too). A failure
    # here deliberately leaves the installed pair in place (never
    # deleted, for forensic/public re-verification) but still raises,
    # so the caller can never print a success message.
    final_result_path = bundle_dir / RESULT_BUNDLE_RESULT_FILENAME
    final_decision_path = bundle_dir / RESULT_BUNDLE_DECISION_FILENAME
    with open(final_result_path, "r", encoding="utf-8") as handle:
        installed_result = json.load(handle)
    with open(final_decision_path, "r", encoding="utf-8") as handle:
        installed_decision = json.load(handle)
    verify_result_document(installed_result, artifacts_doc=artifacts_doc)
    verify_decision_document(installed_decision, artifacts_doc=artifacts_doc)
    installed_result_sha256 = hashlib.sha256(final_result_path.read_bytes()).hexdigest()
    _verify_bundle_pair_consistency(installed_result, installed_decision, installed_result_sha256)


__all__ = [
    "DfpPhase1bT64EarlyExitBenchmarkError",
    "repository_root",
    "STEPS_BY_VARIANT", "TIMING_PHASE_FIELDS", "STAGE_NAMES",
    "verify_protocol", "load_and_verify_manifest", "benchmark_records_in_canonical_order",
    "ReadinessResult", "code_binding_hashes", "perform_repository_readiness_check",
    "WindowInputs", "WindowInputsBuilder", "StageTimings", "process_one_window",
    "per_window_output_record", "compute_scientific_output_digests",
    "MemoryProvider", "GpuHostMemoryProvider",
    "real_e25_device_record", "real_e25_software_record", "real_e25_precision_record", "real_e25_deterministic_settings_record",
    "build_trial_document", "verify_trial_document",
    "TrialProviders", "run_trial",
    "merge_trials", "verify_result_document",
    "build_decision_document", "verify_decision_document",
    "install_artifact_atomically",
    "RESULT_BUNDLE_RESULT_FILENAME", "RESULT_BUNDLE_DECISION_FILENAME",
    "install_result_decision_bundle_atomically",
]
