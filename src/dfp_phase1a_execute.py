"""DFP Phase 1A Step 11: the production COCO-Stuff-500 execution driver.

**Duplication gate.** This module invents no new scientific primitive.
Every number a v2 per-image row carries is produced by an already-
committed, already-tested Step 3-10 module, called through its public
API exactly as its own test suite exercises it:

  * Steps 1/2/9  (identity, subset, preflight)  -- ``src.dfp_phase1a_
    identity``, ``src.dfp_phase1a_subset_manifest``, ``src.dfp_phase1a_
    preflight.run_preflight`` supply the entire Phase-A/B/C ordering
    (protocol/schema/subset validation, source and model binding,
    output-root safety, fresh/resume decision, full checkpoint/NPZ
    resume validation) -- this module adds nothing to that ordering,
    it only adds a thin, additive v2-only enforcement layer on top and
    then continues past it.
  * Step 3   (``src.dfp_phase1a_recurrence_capture``)   -- the finite-
    step trajectory, called once per window.
  * Step 4   (``src.dfp_phase1a_horizon_stitching``)    -- multi-
    horizon stitching, called once per image.
  * Step 5/6 (``src.dfp_phase1a_analysis``)             -- dispute/
    oracle/persistence/confusion/reversion-coherence, called once per
    image via ``analyze_phase1a_image`` plus the small number of
    additional per-variant confusion calls the v2 schema needs beyond
    what that orchestrator already returns (T16/T32/T64, the third-
    label-reversion variant, and the twelve persistence sub-variants).
  * Step 6   (``src.dfp_phase1a_global_unary_components``) -- the
    global unary component partition of the stitched T0 field.
  * Step 7/8 (``src.dfp_phase1a_stitching_pullback`` /
    ``src.dfp_phase1a_soft_membership``) -- the geometry-only pullback
    plan and streamed soft memberships feeding component source mass
    and the Phase 1B active-component projection.
  * Step 1B  (``src.dfp_phase1b_compute_estimator``) -- the per-window
    structural/FLOP/byte compute PROJECTION (never a real Phase 1B
    masked-forward run -- Phase 1B itself is out of this task's scope
    and is never invoked).

**What this module adds**, closed here as narrow orchestration/
plumbing, never duplicating any of the above:

  1. A version-dispatched checkpoint writer (Step 10,
     ``src.dfp_phase1a_checkpoint.generate_and_install_checkpoint_v2``,
     already committed) is driven here, never reimplemented.
  2. The per-window scientific loop: one shared E3/backbone snapshot,
     one canonical directed top-12 graph, one finite-step recurrence
     capturing all five horizons -- reusing Steps 3 exactly once per
     window, mirroring ``matched_power_evaluator.process_one_window``'s
     established shared-snapshot/shared-graph structure but propagating
     only the DFP baseline (k12), never a second (k11) variant.
  3. Per-image assembly of every v2 NPZ member from the above steps'
     own return values -- pure glue, no new confusion/persistence/
     oracle/component/mass formula.
  4. Fresh/resume orchestration: seed the in-memory row accumulator
     from an already-validated v2 resume checkpoint (or start from
     ``build_zero_row_arrays_v2`` for a fresh run), append one row per
     completed image, and install a new checkpoint generation after
     every image.
  5. The CLI ``execute`` entry point (wired from
     ``diagnostics/run_dfp_phase1a.py``) and the mandatory Phase-A
     ordering gate: every identity/schema/subset/binding/output-root/
     resume check in ``run_preflight`` -- plus this module's own
     additive v1-resume-rejection and v2-only enforcement -- completes
     BEFORE this module imports ``torch`` or any Step 3+ module. Every
     function in this file that needs ``torch`` or a Step 3+ import
     performs that import locally, inside its own body, never at
     module scope -- so merely importing this module (as the CLI does
     to register the ``execute`` subcommand) never initializes CUDA,
     never loads a real model, and never touches the dataset.
  6. A manifest-ordered dataset execution view (``src.dfp_phase1a_
     dataset_view.build_manifest_execution_view``): the real mmseg
     dataset (``_build_inference``'s own return value) enumerates the
     COMPLETE 5,000-image COCO-Stuff validation universe, in its own
     native on-disk-listing order -- a DIFFERENT order from the frozen
     500-image manifest's own canonical selection order. Manifest
     position is never assumed to equal native dataset index. The view
     is built once, resolving every one of the 500 frozen ``selected_
     ids`` to its actual native dataset index, and fully validated
     (unique canonical IDs, unique physical paths, correct suffix,
     correct ID format, no missing selected ID) before the first image
     is processed. Both ``dataset[...]`` and ``dataset.get_gt_seg_map_
     by_idx(...)`` are then called with the SAME mapped native index
     for a given image, and the persisted row/checkpoint/result order
     remains exactly the frozen manifest order throughout -- native
     index is purely a lookup detail, never observable in any
     persisted artifact. See that module's own docstring for the full
     rationale, including the real SLURM job (21055181) this closes.
  7. Canonical-identity/relative-path separation at the ``_extract_
     prepared_image`` call boundary: ``ManifestExecutionPosition``
     carries two distinct identities for the same image -- the bare,
     frozen manifest ID (``canonical_id``, e.g. ``"000000000724"``,
     used for row/checkpoint identity throughout this module) and the
     suffix-bearing, image-root-relative path (``relative_path``, e.g.
     ``"000000000724.jpg"``, used ONLY as ``_extract_prepared_image``'s
     own ``canonical_image_id`` argument, which that function's
     docstring defines as the value ``reconcile_canonical_image_id``
     reconciles against the pipeline's resolved physical filename).
     ``execute_run`` passes both explicitly, from the same already-
     validated execution-view position; neither is ever reconstructed
     from the other by string concatenation. See the real SLURM job
     (21060527) this closes: item 6's native-index mapping was already
     correct, but the bare manifest ID was still being forwarded into
     the suffix-bearing parameter, so the pipeline's correctly-resolved
     ``000000000724.jpg`` disagreed with an expected path derived from
     the un-suffixed ``000000000724``.
  8. Final-result installation destination and finalize-only recovery:
     ``ExecutionContext.result_bundle_dir`` (``context.checkpoints_dir /
     "result"``, a fixed, deterministic subdirectory name -- never a
     new CLI-selected path, never the checkpoints directory itself,
     which already exists once any checkpoint has been written and
     would make ``install_result_bundle``'s own already-exists guard
     reject every run) is the ONE destination both
     ``install_result_bundle`` and ``verify_result_bundle`` are called
     with in ``_install_final_result``. ``execute_run`` also checks,
     BEFORE ``_build_inference`` (never after, as a trailing check
     alone would still pay for a full model/dataset/CUDA construction
     just to immediately skip an empty per-image loop), whether
     ``preflight.next_dataset_index`` already equals
     ``preflight.expected_count`` -- i.e. a checkpoint that is already
     complete but whose result bundle was never installed -- and if so
     finalizes directly, constructing no model, loading no weight,
     initializing no CUDA, and building no dataset. See the real SLURM
     job (21063877) this closes: preflight/Phase-A/B/C validation and
     per-image processing had ALREADY produced a complete, self-
     verified 500-image v2 checkpoint; only the final call --
     ``install_result_bundle(checkpoints_dir=..., checkpoint_filename=...,
     repo_root=...)``, omitting the required keyword-only
     ``destination_dir`` -- raised an uncaught ``TypeError`` after all
     500 images had already finished, at which point the driver had no
     way to finalize that checkpoint again without reprocessing every
     image (the per-image loop's own ``range(preflight.next_dataset_
     index, preflight.expected_count)`` would already have been empty
     on any subsequent invocation, so this failure would recur exactly
     the same way every retry without the destination fix and the
     early finalize-only check together).

**Final-resolution analysis authority (now committed, no longer a driver
judgment call).** Every temporary resolution/reconciliation choice this
module previously made on its own authority has been replaced by the
committed ``[final_resolution_analysis]`` / ``[reversion_coherence]``
protocol blocks (``src.dfp_phase1a_identity.validate_final_resolution_
analysis_block`` / ``validate_reversion_coherence_block``) and the
composed final-rescale pullback (``src.dfp_phase1a_final_rescale_
pullback``):

  * Every T0/T16/T32/T64/T320 stitched score field is rescaled into the
    canonical final ``ori_shape`` domain via ``apply_final_rescale_
    forward`` -- byte-identical to ``matched_power_evaluator.
    finalize_prediction``'s own crop+bilinear-resize, bound by file-
    content SHA-256 in the identity -- BEFORE argmax, dispute
    classification, or component construction. ``ori_shape`` comes
    exclusively from ``img_metas['ori_shape']`` (authoritative image/
    pipeline metadata).
  * ``dataset.get_gt_seg_map_by_idx`` is called strictly AFTER every
    structural, score-only stage has already completed and returned:
    all five final score fields, the score-only T0/T320 labels and
    their raw ``a(x)!=b(x)`` dispute set (``derive_horizon_labels``,
    GT-free), final-resolution component construction, the exact
    composed final-resolution memberships, component source mass, and
    the entire Phase 1B compute projection. None of those receive
    ``gt`` as an argument at any point. GT's shape must already equal
    the canonical final domain (checked, never resized).
  * Global unary components (Step 6) are constructed directly from the
    final ``ori_shape`` stitched T0 field -- never the raw pre-rescale
    Step-5 output, never resized or nearest-neighbor-reconciled, and
    (per the point above) before GT is read at all.
  * Component source mass uses the composed operator ``A_final = R
    composed with A_stitch`` (``build_final_stitching_pullback_plan`` /
    ``iter_image_final_soft_memberships`` / ``compute_window_final_
    soft_memberships``) directly on the SAME final-resolution component
    IDs -- an exact adjoint, never an approximation, and never a
    resample of any kind.
  * Reversion coherence is exactly the committed definition: baseline
    T320, fallback T0, reversion set = harmful washout
    (``analysis.dispute.harmful`` -- never all disputes, never T32,
    never beneficial/third-label pixels), evaluated at final
    ``ori_shape``, with connectivity and the small-component pixel
    threshold read from ``AnalysisContract.reversion_connectivity`` /
    ``reversion_small_component_threshold_pixels`` -- no duplicated
    literal ``8``/``20``, no patch-to-pixel conversion, anywhere in this
    module. Genuinely requires GT (via ``analysis.dispute.harmful``)
    and is computed only after it, alongside every other harmful/
    beneficial/third-label/oracle/persistence analysis.

The Phase 1B compute-projection (Step 1B) now uses EXACTLY the persisted
final-resolution ``components`` and the same composed final-resolution
memberships already computed for source mass -- never a separately
recomputed inference-resolution partition. Step 1B's own frozen
``compute_window_active_components`` / ``compute_window_query_classes``
helpers are bound to Step 7's inference-resolution-only
``iter_soft_membership_records`` and cannot accept the composed
final-resolution operator, so this driver reproduces their exact,
documented "active component" definition (nonzero soft membership) and
"query class" derivation (the disputed a(x)/b(x) class values a window
touches) directly, from the composed memberships and from GT-free,
score-only final-resolution labels (``derive_horizon_labels(scores_
final)`` -- the SAME pure helper ``analyze_phase1a_image`` itself uses,
called directly here, before GT is read, never a hand-duplicated
argmax/tie-rule formula) -- never from GT, never from a resampled
component set.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import src.dfp_phase1a_artifact_schema as artifact_schema
import src.dfp_phase1a_checkpoint as checkpoint_mod
from src.dfp_phase1a_identity import DfpPhase1aIdentityError, load_identity as load_protocol_identity
from src.dfp_phase1a_preflight import (
    DfpPhase1aPreflightError,
    PreflightContext,
    PreflightTrace,
    run_preflight,
)

RUN_NAME = "coco_stuff_500"
_RUN_MODE = "coco_stuff_500"  # identical string; kept as a separate name for readability at call sites
_CHECKPOINT_FILENAME = "checkpoint.json"
_COMPONENT_CHUNK_SIZE = 64
"""Component-chunk size used ONLY for the Phase 1B compute PROJECTION
(Step 1B) and for the component-source-mass streaming accumulation
(Step 8). Affects only how the projection's own chunk_count/chunk_bytes
are shaped -- never a scientific identity value, never something the
protocol pins, and never influencing any confusion/oracle/persistence/
component/mass NUMBER itself (Step 7/8's chunking is proven numerically
chunk-size-invariant by their own test suites)."""


class DfpPhase1aExecuteError(RuntimeError):
    """Raised on any production-execution-driver invariant violation.
    Always fail closed: never silently substitute a fresh run for a
    resume, never fall back from v2 to v1, never overwrite an existing
    checkpoint/result without an explicit fresh-overwrite authorization,
    never partially install a checkpoint generation."""


_KNOWN_OPERATIONAL_ERRORS: tuple[type[BaseException], ...] = (
    DfpPhase1aExecuteError,
    DfpPhase1aPreflightError,
    DfpPhase1aIdentityError,
    artifact_schema.DfpPhase1aArtifactSchemaError,
    checkpoint_mod.DfpPhase1aCheckpointError,
    OSError,
    ValueError,
)
"""Every domain error this module (or anything it calls before the
heavy-import boundary) is expected to raise operationally -- the CLI
dispatch catches exactly this tuple and exits 2. KeyboardInterrupt,
SystemExit, and MemoryError are never members of this tuple and are
therefore never caught here."""


# ---------------------------------------------------------------------------
# Execution context: Phase A/B/C (via run_preflight) + additive v1/v2 gate.
# Pure stdlib/numpy code below this line -- no torch, no Step 3+ import.
# ---------------------------------------------------------------------------


_RESULT_BUNDLE_DIRNAME = "result"  # fixed, deterministic -- never CLI-selected, never inferred from any artifact


def _resolve_result_bundle_dir(output_root: Path) -> Path:
    """The one authoritative final-result-bundle destination for a given
    run output root: ``output_root / "result"`` -- a fixed subdirectory
    name, never the output root itself (which already exists once any
    checkpoint has been written, and ``install_result_bundle`` refuses
    to install over an existing destination), never a new CLI-selected
    or hand-typed path. Defensively confirms the lexically-resolved
    result path still lands inside the lexically-resolved output root
    (the same containment shape ``src.dataset_image_identity.
    reconcile_canonical_image_id`` already uses) -- unreachable under
    normal construction (the child is a literal, non-symlink path
    component appended to ``output_root``), but fails closed rather
    than silently trusting it if ``output_root`` itself ever resolves
    somewhere unexpected."""
    result_dir = output_root / _RESULT_BUNDLE_DIRNAME
    resolved_result_dir = result_dir.resolve(strict=False)
    resolved_output_root = output_root.resolve(strict=False)
    if not (resolved_result_dir == resolved_output_root or resolved_output_root in resolved_result_dir.parents):
        raise DfpPhase1aExecuteError(
            f"result bundle directory {result_dir} (resolved {resolved_result_dir}) does not resolve inside "
            f"the run output root {output_root} (resolved {resolved_output_root})"
        )
    return result_dir


class ExecutionContext:
    """Immutable bundle produced once Phase A, B, and C (via
    ``run_preflight``) and this module's own additive v2-only gate have
    ALL passed. Carries nothing torch-flavored -- constructing this
    object never imports torch."""

    __slots__ = ("preflight", "device", "repo_root", "checkpoints_dir", "result_bundle_dir", "trace")

    def __init__(self, *, preflight: PreflightContext, device: str, repo_root: Path, trace: PreflightTrace) -> None:
        self.preflight = preflight
        self.device = device
        self.repo_root = repo_root
        self.checkpoints_dir = preflight.output_root
        self.result_bundle_dir = _resolve_result_bundle_dir(preflight.output_root)
        self.trace = trace


def _require_v2_or_absent(preflight: PreflightContext) -> None:
    """Additive enforcement on top of ``run_preflight``'s own (version-
    agnostic, backward-compatible) resume validation: the PRODUCTION
    EXECUTION driver refuses to resume a v1 checkpoint, and refuses any
    array set that is neither exactly v1's nor exactly v2's closed
    member-name set. Never touches ``dfp_phase1a_preflight.py`` itself
    -- ``PreflightContext`` does not store which schema variant it
    loaded, so this function re-derives it, exactly and only, from the
    member-NAME set already returned (never from a member count, which
    would be a weaker, coincidental check)."""
    if preflight.mode == "fresh":
        return
    arrays = preflight.checkpoint_arrays
    if arrays is None:
        raise DfpPhase1aExecuteError("resume mode produced no checkpoint arrays (internal preflight contract violation)")
    observed = frozenset(arrays)
    if observed == frozenset(artifact_schema.ALL_NPZ_MEMBER_NAMES):
        raise DfpPhase1aExecuteError(
            f"existing checkpoint at {preflight.output_root / _CHECKPOINT_FILENAME} uses per-image-stats schema "
            f"{artifact_schema.PER_IMAGE_STATS_SCHEMA_NAME!r} (v1); the production execution driver writes and "
            f"resumes ONLY {artifact_schema.PER_IMAGE_STATS_SCHEMA_NAME_V2!r} (v2) -- refusing to resume a v1 checkpoint"
        )
    if observed != frozenset(artifact_schema.ALL_NPZ_MEMBER_NAMES_V2):
        raise DfpPhase1aExecuteError(
            "existing checkpoint's per-image-stats member-name set is neither valid v1 nor valid v2 -- refusing to resume"
        )


def build_execution_context(
    *, run: str, mode: str, output_root: Path, device: str, repo_root: Path, allow_fresh_overwrite: bool = False,
    trace: PreflightTrace | None = None,
) -> ExecutionContext:
    """Steps 1-9 of the mandatory Phase-A ordering, entirely: parse and
    validate every CLI argument this driver accepts, then run the full,
    already-committed ``run_preflight`` (protocol, schema-v2 authority,
    frozen subset, source binding, model binding, output-root safety,
    fresh/resume decision, and -- for resume -- the complete checkpoint/
    NPZ validation chain), then apply this module's own additive v1-
    resume-rejection / v2-only gate. Raises before any torch import,
    dataset construction, image decode, model construction, checkpoint-
    weight load, or CUDA check. ADE20K is never reachable from this
    function: it always calls ``run_preflight`` with the frozen
    ``coco_stuff_500`` run mode, never the caller's own string."""
    if type(run) is not str or run != RUN_NAME:
        raise DfpPhase1aExecuteError(f"--run must be exactly {RUN_NAME!r} (no aliases, no ade20k_500); got {run!r}")
    if type(mode) is not str or mode not in ("fresh", "resume"):
        raise DfpPhase1aExecuteError(f"--mode must be exactly 'fresh' or 'resume'; got {mode!r}")
    if type(device) is not str or device not in ("cpu", "cuda"):
        raise DfpPhase1aExecuteError(f"--device must be exactly 'cpu' or 'cuda'; got {device!r}")
    output_root = Path(output_root)
    repo_root = Path(repo_root)
    if not repo_root.is_dir():
        raise DfpPhase1aExecuteError(f"repo root {repo_root} is not a directory")

    own_trace = trace if trace is not None else PreflightTrace()
    preflight = run_preflight(
        repo_root=repo_root, run_mode=_RUN_MODE, mode=mode, output_root=output_root,
        checkpoint_filename=_CHECKPOINT_FILENAME, allow_fresh_overwrite=allow_fresh_overwrite,
        probe_output_writeable=True, trace=own_trace,
    )
    _require_v2_or_absent(preflight)
    return ExecutionContext(preflight=preflight, device=device, repo_root=repo_root, trace=own_trace)


# ---------------------------------------------------------------------------
# In-memory v2 row accumulator. Pure numpy -- no torch.
# ---------------------------------------------------------------------------


class _V2Accumulator:
    """Mutable in-memory accumulator for the v2 per-image-stats arrays.
    Seeded from an already-validated resume array set (copied, never
    aliased) or from ``build_zero_row_arrays_v2`` for a fresh run.
    ``append_image_row`` only ever appends: fixed-row members grow by
    exactly one row, offset members grow by exactly one terminal offset,
    flat members grow by exactly one (possibly empty) segment. No prior
    row is ever mutated."""

    def __init__(self, *, class_count: int, seed_arrays: Mapping[str, np.ndarray] | None) -> None:
        if seed_arrays is not None:
            observed = frozenset(seed_arrays)
            if observed != frozenset(artifact_schema.ALL_NPZ_MEMBER_NAMES_V2):
                raise DfpPhase1aExecuteError("seed_arrays for _V2Accumulator must be exactly the v2 member-name set")
            self._arrays: dict[str, np.ndarray] = {name: np.array(arr, copy=True) for name, arr in seed_arrays.items()}
        else:
            self._arrays = checkpoint_mod.build_zero_row_arrays_v2(class_count=class_count)
        self._class_count = class_count

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return self._arrays

    @property
    def row_count(self) -> int:
        return int(self._arrays["dataset_indices"].shape[0])

    def _append_offset_and_flats(self, offset_name: str, flat_specs: Mapping[str, tuple[Sequence[Any], type]]) -> None:
        prev_offsets = self._arrays[offset_name]
        prev_total = int(prev_offsets[-1])
        seg_len: int | None = None
        for flat_name, (values, dtype) in flat_specs.items():
            values_arr = np.asarray(list(values), dtype=dtype)
            if seg_len is None:
                seg_len = int(values_arr.shape[0])
            elif int(values_arr.shape[0]) != seg_len:
                raise DfpPhase1aExecuteError(f"{flat_name} segment length disagrees with its sibling flat array under {offset_name}")
            self._arrays[flat_name] = np.concatenate([self._arrays[flat_name], values_arr])
        assert seg_len is not None
        self._arrays[offset_name] = np.concatenate([prev_offsets, np.array([prev_total + seg_len], dtype=np.int64)])

    def append_image_row(self, row: Mapping[str, Any]) -> None:
        for name in artifact_schema.SCALAR_INT64_MEMBER_NAMES_V2:
            self._arrays[name] = np.concatenate([self._arrays[name], np.array([row[name]], dtype=np.int64)])
        for name in artifact_schema.SCALAR_FLOAT64_MEMBER_NAMES_V2:
            self._arrays[name] = np.concatenate([self._arrays[name], np.array([row[name]], dtype=np.float64)])
        for variant in artifact_schema.PREDICTION_VARIANT_NAMES_V2:
            for stat in artifact_schema.CONFUSION_STAT_NAMES_V2:
                name = artifact_schema.confusion_member_name_v2(variant, stat)
                values = np.asarray(row[name], dtype=np.int64).reshape(1, self._class_count)
                self._arrays[name] = np.concatenate([self._arrays[name], values], axis=0)
        self._append_offset_and_flats(
            "component_offsets",
            {
                "component_class_indices_flat": (row["component_class_indices"], np.int64),
                "component_pixel_areas_flat": (row["component_pixel_areas"], np.int64),
                "component_patch_equivalent_areas_flat": (row["component_patch_equivalent_areas"], np.float64),
                "component_source_mass_flat": (row["component_source_mass"], np.float64),
            },
        )
        self._append_offset_and_flats(
            "reversion_component_offsets",
            {"reversion_component_sizes_flat": (row["reversion_component_sizes"], np.int64)},
        )


# ---------------------------------------------------------------------------
# Heavy imports and construction -- deferred entirely into function bodies.
# Nothing above this line ever imports torch, mmcv, mmseg, or any DFP
# Step 3+ module.
# ---------------------------------------------------------------------------


def _load_e3_identity(repo_root: Path, protocol_identity: Mapping[str, Any]) -> dict[str, Any]:
    from src.e3_evaluation_identity import load_identity as load_e3_identity

    parents = protocol_identity["parent_identities"]
    return load_e3_identity(repo_root / parents["e3_identity_path"], repo_root=repo_root)


def _build_contracts(protocol_identity: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    from src.dfp_phase1a_analysis import load_analysis_contract_from_protocol
    from src.dfp_phase1a_final_rescale_pullback import dfp_final_resolution_contract_from_protocol
    from src.dfp_phase1a_global_unary_components import dfp_component_contract_from_protocol
    from src.dfp_phase1a_horizon_stitching import dfp_stitching_contract_from_protocol
    from src.dfp_phase1a_recurrence_capture import dfp_recurrence_contract_from_protocol
    from src.dfp_phase1a_stitching_pullback import dfp_pullback_contract_from_protocol
    from src.dfp_phase1b_compute_estimator import dfp_compute_projection_contract_from_protocol

    return {
        "recurrence": dfp_recurrence_contract_from_protocol(protocol_identity),
        "stitching": dfp_stitching_contract_from_protocol(protocol_identity, repo_root=repo_root),
        "analysis": load_analysis_contract_from_protocol(protocol_identity, dataset_key="coco_stuff", repo_root=repo_root),
        "component": dfp_component_contract_from_protocol(protocol_identity, repo_root=repo_root),
        "pullback": dfp_pullback_contract_from_protocol(protocol_identity, repo_root=repo_root),
        "final_resolution": dfp_final_resolution_contract_from_protocol(protocol_identity, repo_root=repo_root),
        "compute_projection": dfp_compute_projection_contract_from_protocol(protocol_identity),
    }


def _process_one_window_dfp(
    inference: Any, image_tensor: Any, window: Any, *, protocol_identity: Mapping[str, Any],
    affinity_power: float, class_count: int, graph_variant: str,
) -> tuple[Any, tuple[int, int], Any]:
    """Exactly the shared-snapshot/shared-graph structure ``matched_
    power_evaluator.process_one_window`` already establishes, minus the
    k11 comparator (DFP propagates only the canonical k12 baseline):
    one backbone/snapshot call, one canonical top-12 directed graph
    build, one finite-step recurrence capturing every frozen horizon
    from Step 3. Returns ``(FiniteStepTrace, grid_hw, WindowGraphMetadata)``.
    Performs zero k11 derivation, zero masked propagation, zero solver
    call, zero second snapshot/graph/recurrence."""
    import torch

    from src.dfp_phase1a_recurrence_capture import capture_dfp_phase1a_trajectory  # noqa: F401 -- also seeds sys.path for the bare "models.*" import below
    from src.dfp_phase1b_compute_estimator import WindowGraphMetadata
    from models.dinotext.cover_dr.graph import build_directed_topk_graph  # bare path: matches dfp_phase1a_recurrence_capture.py's own import convention; a src.open_vocabulary_segmentation.-prefixed import double-registers mmcv modules

    crop_rows, crop_cols = window.crop_slice
    crop = image_tensor[:, :, crop_rows, crop_cols]

    with torch.no_grad():
        snapshot = inference.model.generate_patch_snapshot(crop, inference.text_embedding)
    s0 = snapshot.unary_scores[0]
    if s0.shape[1] != class_count:
        raise DfpPhase1aExecuteError(f"window {window.index}: snapshot class dimension {s0.shape[1]} disagrees with class_count {class_count}")
    dino_features = snapshot.dino_features[0]
    grid_hw = tuple(int(v) for v in snapshot.grid_hw)

    graph12 = build_directed_topk_graph(dino_features, k=12, affinity_power=affinity_power)
    trace = capture_dfp_phase1a_trajectory(graph12, s0, protocol_identity=protocol_identity)

    fallback = graph12.self_loop_fallback
    fallback_included = bool(fallback.any().item()) if torch.is_tensor(fallback) else bool(fallback)
    graph_meta = WindowGraphMetadata(
        node_count=int(graph12.num_nodes),
        nonzero_count=int(graph12.neighbor_indices.numel()),
        index_dtype_bytes=int(graph12.neighbor_indices.element_size()),
        weight_dtype_bytes=int(graph12.transition_weights.element_size()),
        fallback_included=fallback_included,
        graph_variant=graph_variant,
    )
    return trace, grid_hw, graph_meta


def _process_one_image(
    inference: Any, dataset: Any, dataset_index: int, canonical_image_id: str, *, protocol_identity: Mapping[str, Any],
    contracts: Mapping[str, Any], class_count: int, graph_variant: str, native_dataset_index: int | None = None,
    canonical_relative_image_path: str | None = None,
) -> tuple[dict[str, Any], int]:
    """The full per-image scientific pipeline for exactly one image.

    ``dataset_index`` is the MANIFEST POSITION (0..expected_count-1,
    canonical row/progress order -- persisted verbatim as the schema-v2
    ``dataset_indices`` member). ``native_dataset_index`` is the actual
    index to fetch from ``dataset`` itself (``dataset[native_dataset_
    index]`` / ``dataset.get_gt_seg_map_by_idx(native_dataset_index)``)
    -- the native mmseg dataset's own on-disk ordering is a DIFFERENT
    order from the frozen manifest's, so these are never assumed equal
    in production; see ``src/dfp_phase1a_dataset_view.py`` for the
    manifest-order -> native-index mapping and its own full rationale
    (including the real SLURM job -- 21055181 -- this closes: a stale
    positional call here previously fed manifest position directly to
    ``dataset[...]``, which the pre-existing ``reconcile_canonical_
    image_id`` guard correctly rejected rather than silently mis-loading
    the wrong image). Defaults to ``dataset_index`` when omitted, which
    is exactly correct for any dataset whose native order already
    matches manifest order by construction (every synthetic/small test
    fixture in this driver's own test suite, which model no mmseg-style
    metadata surface at all) -- production (``execute_run``) always
    passes the real, distinct, mapped value explicitly. Whenever
    ``native_dataset_index`` IS explicitly given, this function
    independently re-verifies, before ``_extract_prepared_image`` is
    ever called (``src.dfp_phase1a_dataset_view.verify_position_
    identity``), that the dataset's own live metadata at that index
    still yields exactly ``canonical_image_id`` -- never merely
    trusting a value computed once, earlier, when the execution view
    was built.

    ``canonical_image_id`` and ``canonical_relative_image_path`` are two
    DELIBERATELY DISTINCT identities, never interchangeable (this is the
    exact call-site defect SLURM job 21060527 exposed: passing the bare
    manifest ID where the suffix-bearing relative path was required).
    ``canonical_image_id`` is the bare, frozen 12-digit manifest ID (e.g.
    ``"000000000724"``, no extension) -- authoritative for row/checkpoint
    identity, progress accounting, and every diagnostic message in this
    function; it is what ``verify_position_identity`` above checks and
    what ``execute_run`` persists into ``dataset_indices``/
    ``completed_image_ids``. ``canonical_relative_image_path`` is the
    image-root-relative, suffix-bearing path (e.g. ``"000000000724.jpg"``)
    -- authoritative ONLY for ``_extract_prepared_image``'s own
    ``canonical_image_id`` parameter, which that function's docstring
    defines as the value ``reconcile_canonical_image_id`` reconciles
    against the pipeline's actually-resolved physical filename. Defaults
    to ``canonical_image_id`` when omitted (exactly correct for any
    dataset whose native filenames already equal the bare manifest ID --
    every synthetic/small test fixture in this driver's own test suite);
    production (``execute_run``) always passes the adapter's own already-
    validated ``ManifestExecutionPosition.relative_path`` explicitly --
    never a hand-built ``canonical_image_id + suffix`` concatenation.

    Prepare once, build the canonical window plan once, run every
    window's shared snapshot/graph/recurrence exactly once, stitch all
    five horizons once, then assemble every v2 NPZ member from Step
    3-10's own already-tested outputs. Ground truth is read ONLY after
    every structural, score-only stage has already completed: horizon
    stitching, the final-resolution rescale of every score field, the
    score-only T0/T320 labels and their raw a(x)!=b(x) dispute set,
    final-resolution T0 component construction, the exact composed
    final-resolution memberships, component source mass, and the
    Phase-1B compute projection are all computed from ``scores_final``
    and image/pipeline metadata alone -- ``dataset.get_gt_seg_map_by_idx``
    is called strictly after every one of those has already returned,
    and nothing computed before that call ever receives ``gt`` as an
    argument. Harmful/beneficial/third-label classification, the oracle
    and persistence maps, and reversion coherence all require GT and are
    therefore computed only after it -- see the module docstring's
    GT-isolation account."""
    import torch

    from diagnostics.run_k11_k12_stability import _extract_prepared_image
    from src.dfp_phase1a_dataset_view import verify_position_identity
    from src.dfp_phase1a_analysis import (
        analyze_phase1a_image,
        compute_confusion_statistics,
        compute_reversion_coherence,
        derive_horizon_labels,
        _official_mask_to_boundary,
    )
    from src.dfp_phase1a_final_rescale_pullback import (
        apply_final_rescale_forward,
        build_final_stitching_pullback_plan,
        component_source_mass_float64,
        iter_image_final_soft_memberships,
    )
    from src.dfp_phase1a_global_unary_components import compute_global_unary_components
    from src.dfp_phase1a_horizon_stitching import WindowHorizonCaptures, stitch_dfp_phase1a_horizons
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan
    from src.dfp_phase1b_compute_estimator import (
        aggregate_window_estimates,
        descriptive_statistics,
        estimate_window_compute,
    )
    from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize  # bare path: matches dfp_phase1a_horizon_stitching.py's own import convention

    stitching_contract = contracts["stitching"]
    analysis_contract = contracts["analysis"]
    component_contract = contracts["component"]
    pullback_contract = contracts["pullback"]
    final_resolution_contract = contracts["final_resolution"]
    compute_contract = contracts["compute_projection"]
    affinity_power = float(protocol_identity["graph"]["affinity_power"])
    class_count_local = analysis_contract.class_count

    resolved_native_index = dataset_index if native_dataset_index is None else native_dataset_index
    if native_dataset_index is not None:
        # Runtime re-verification (Section 7's "three identities agree"
        # guard), strictly before graph construction/recurrence/GT
        # access: independently re-derives the canonical ID from the
        # dataset's own LIVE metadata at resolved_native_index and
        # requires it to still equal canonical_image_id.
        # _extract_prepared_image's own internal reconcile_canonical_
        # image_id call (immediately below) then checks the THIRD
        # identity -- the pipeline-resolved physical filename --
        # completing all three checks before any window is processed.
        # Only meaningful (and only required) when a real manifest ->
        # native-index MAPPING was actually applied -- i.e. production,
        # via execute_run's own build_manifest_execution_view. When
        # native_dataset_index is omitted, dataset_index is used
        # directly with no remapping (every synthetic/small test
        # fixture in this module's own test suite, which model no
        # mmseg-style img_infos/img_suffix metadata surface at all), so
        # there is no mapping step to re-verify.
        verify_position_identity(
            dataset, manifest_position=dataset_index, canonical_id=canonical_image_id, native_index=resolved_native_index,
        )
    # Bug closed here (SLURM job 21060527): the bare canonical_image_id
    # (manifest identity) must NEVER be forwarded as-is into
    # _extract_prepared_image's own canonical_image_id parameter, which
    # that function's docstring defines as the suffix-bearing, image-
    # root-relative path reconciled against the pipeline's resolved
    # physical filename (see the module docstring above). Falls back to
    # canonical_image_id only when no relative path was supplied (every
    # synthetic/small test fixture whose native filenames already equal
    # the bare ID); production always supplies the adapter's own
    # already-validated ManifestExecutionPosition.relative_path.
    resolved_relative_image_path = (
        canonical_image_id if canonical_relative_image_path is None else canonical_relative_image_path
    )
    prepared = _extract_prepared_image(dataset, resolved_native_index, canonical_image_id=resolved_relative_image_path)
    image_tensor = prepared.image_tensor.unsqueeze(0)
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(prepared.inference_height, prepared.inference_width),
        crop_size=SpatialSize(*stitching_contract.crop), stride=SpatialSize(*stitching_contract.stride),
    )

    window_captures: list[Any] = []
    graph_metadatas: list[Any] = []
    for window in plan.windows:
        trace, grid_hw, graph_meta = _process_one_window_dfp(
            inference, image_tensor, window, protocol_identity=protocol_identity,
            affinity_power=affinity_power, class_count=class_count, graph_variant=graph_variant,
        )
        if trace.steps_completed != contracts["recurrence"].steps:
            raise DfpPhase1aExecuteError(f"window {window.index}: recurrence completed {trace.steps_completed} updates, expected {contracts['recurrence'].steps}")
        if set(trace.snapshots) != set(contracts["recurrence"].horizons):
            raise DfpPhase1aExecuteError(f"window {window.index}: captured horizon set disagrees with the frozen protocol horizons")
        window_captures.append(WindowHorizonCaptures(window_index=window.index, captures=trace.snapshots))
        graph_metadatas.append(graph_meta)

    stitched_inf = stitch_dfp_phase1a_horizons(plan, tuple(window_captures), contract=stitching_contract, class_count=class_count)

    # ---- Canonical final-resolution score fields (Step 11 authority):
    # ori_shape comes EXCLUSIVELY from authoritative image/pipeline
    # metadata, never from GT (GT is not even read yet at this point).
    # Every horizon goes through the SAME bound apply_final_rescale_
    # forward (byte-identical to the existing fixed-320 matched
    # evaluator's own crop+bilinear-resize, minus its trailing argmax --
    # this driver needs linear scores for dispute/persistence/confusion,
    # not a finalized label map). ---------------------------------------
    img_shape = (prepared.inference_height, prepared.inference_width)
    ori_shape = tuple(int(v) for v in prepared.img_metas["ori_shape"][:2])
    if len(ori_shape) != 2 or ori_shape[0] <= 0 or ori_shape[1] <= 0:
        raise DfpPhase1aExecuteError(f"image {canonical_image_id}: malformed ori_shape from image metadata: {ori_shape}")

    scores_final: dict[int, np.ndarray] = {}
    for horizon, tensor in stitched_inf.items():
        if tensor.shape[1] != class_count:
            raise DfpPhase1aExecuteError(f"image {canonical_image_id}: horizon {horizon} stitched field has {tensor.shape[1]} classes, expected {class_count}")
        rescaled = apply_final_rescale_forward(tensor, img_shape=img_shape, ori_shape=ori_shape, contract=final_resolution_contract)
        scores_final[horizon] = rescaled.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # Require all five final fields to share IDENTICAL [class_count,
    # ori_height, ori_width] -- true by construction (one shared
    # img_shape/ori_shape/contract applied uniformly), checked
    # explicitly rather than assumed.
    final_shapes = {h: arr.shape for h, arr in scores_final.items()}
    if len(set(final_shapes.values())) != 1:
        raise DfpPhase1aExecuteError(f"image {canonical_image_id}: final score fields disagree in shape across horizons: {final_shapes}")

    # ---- Score-only T0/T320 labels and their raw dispute set D =
    # {x : a(x) != b(x)}. Uses the SAME pure derive_horizon_labels helper
    # analyze_phase1a_image itself uses internally (never a hand-
    # duplicated argmax/tie-rule) -- called directly, here, on
    # scores_final alone, before GT is read. This is the score-only D,
    # NOT the GT-restricted (non-ignore) dispute set analyze_phase1a_
    # image's own classify_disputes will later compute from the SAME a,
    # b, plus gt -- classify_disputes cannot be called yet (it requires
    # gt for its own ignore-pixel restriction), so its formula is not
    # duplicated here, only the trivial a(x)!=b(x) comparison the driver
    # already needed pre-GT for the compute projection below. ----------
    labels_pre_gt = derive_horizon_labels(scores_final)
    a_pre_gt = labels_pre_gt[analysis_contract.unary_horizon]
    b_pre_gt = labels_pre_gt[analysis_contract.final_horizon]
    dispute_pre_gt_mask = a_pre_gt != b_pre_gt

    # ---- Global unary components (Step 6), at the canonical final
    # ori_shape -- the committed authority's own component domain, and
    # the schema's own component_pixel_areas_flat invariant (its
    # per-image segment must sum to evaluated_pixel_count +
    # ignore_pixel_count). Never the raw pre-rescale stitched field,
    # never resized afterward, no GT value or GT shape used to build it.
    # GT has not been read at all at this point.
    components = compute_global_unary_components(scores_final[0], contract=component_contract, class_count=class_count_local)
    if components.component_ids.shape != ori_shape:
        raise DfpPhase1aExecuteError(
            f"image {canonical_image_id}: internal error -- component_ids.shape {components.component_ids.shape} "
            f"disagrees with the canonical final ori_shape {ori_shape}"
        )

    # ---- Component source mass: the committed composed operator
    # A_final = R composed with A_stitch, applied DIRECTLY to the SAME
    # final-resolution component IDs above -- an exact adjoint, never a
    # dense final-pixel x patch-source matrix, never any resampling of
    # any kind. Uses the exact image/window/crop/ori-shape metadata that
    # produced the final score fields (never reconstructed independently).
    final_pullback_window_plan = build_stitching_pullback_plan(img_shape, contract=pullback_contract)
    final_plan = build_final_stitching_pullback_plan(
        final_pullback_window_plan, img_shape=img_shape, ori_shape=ori_shape, contract=final_resolution_contract,
    )
    component_source_mass = np.zeros(components.component_count, dtype=np.float64)
    # Active components per window, derived from the SAME membership
    # chunks computed for source mass below -- one pass, one exact
    # composed-pullback evaluation per (chunk, window), reused for both
    # the persisted v2 source-mass field and the Phase 1B compute
    # PROJECTION's own "active component" input (Section 5-7: the same
    # component IDs persisted in v2, the same per-window memberships, and
    # the same active-component definition -- "some patch node j has
    # strictly positive membership M[r,w,j] > 0" -- as the frozen
    # compute_window_active_components already defines, reproduced here
    # rather than called, since that function is hard-bound to Step 7's
    # OWN inference-resolution-only iter_soft_membership_records and
    # cannot accept the composed final-resolution operator).
    active_components_by_window: dict[int, set[int]] = {w: set() for w in range(len(final_pullback_window_plan.plan.windows))}
    chunk_groups: dict[tuple[int, int], dict[int, Any]] = defaultdict(dict)
    for start, chunk in iter_image_final_soft_memberships(
        components.component_ids, components.component_count, final_plan=final_plan, component_chunk_size=_COMPONENT_CHUNK_SIZE,
    ):
        for window_index, membership in chunk.items():
            chunk_groups[(start, start + membership.shape[0])][window_index] = membership
            for offset, is_positive in enumerate((membership > 0).any(dim=1).tolist()):
                if is_positive:
                    active_components_by_window[window_index].add(start + offset)
    for (start, stop), by_window in chunk_groups.items():
        component_source_mass[start:stop] = component_source_mass_float64(by_window).detach().cpu().numpy()

    # ---- Phase 1B compute PROJECTION (Step 1B) -- structural/FLOP/byte
    # estimate only; no Phase 1B masked-forward run occurs anywhere.
    # Uses EXACTLY the persisted final-resolution `components` (never a
    # separately recomputed inference-resolution partition) and the
    # active-component sets already derived above from the SAME composed
    # final-resolution memberships computed for source mass. Q_w (the
    # per-window query-class set) is derived from the score-only
    # a_pre_gt/b_pre_gt labels above (GT-free) -- restricted to each
    # window's own active components: a(x) is constant within a T0-
    # defined component (component_class_indices[r] exactly), so only
    # b(x) at that component's disputed pixels needs scanning. No GT is
    # read anywhere in this block, and GT has not been read at all yet
    # at this point in the function. -----------------------------------
    disputed_component_ids = components.component_ids[dispute_pre_gt_mask]
    disputed_b_values = b_pre_gt[dispute_pre_gt_mask]
    component_b_value_sets: dict[int, set[int]] = defaultdict(set)
    for cid, bval in zip(disputed_component_ids.tolist(), disputed_b_values.tolist()):
        component_b_value_sets[cid].add(bval)

    estimates = []
    dtype_bytes = 4
    for window in final_pullback_window_plan.plan.windows:
        active_components = tuple(sorted(active_components_by_window[window.index]))
        query_class_set: set[int] = set()
        for component_id in active_components:
            query_class_set.add(int(components.component_class_indices[component_id]))
            query_class_set.update(component_b_value_sets.get(component_id, ()))
        query_classes = tuple(sorted(query_class_set))
        estimates.append(
            estimate_window_compute(
                window_index=window.index, graph_metadata=graph_metadatas[window.index], class_count=class_count_local,
                active_component_ids=active_components, query_class_indices=query_classes,
                recurrence_steps=compute_contract.recurrence_steps, dtype_bytes=dtype_bytes,
                component_chunk_size=_COMPONENT_CHUNK_SIZE,
            )
        )
    aggregate = aggregate_window_estimates(estimates, total_disputed_pixels=int(dispute_pre_gt_mask.sum()))
    q_w_stats = descriptive_statistics([e.query_class_count for e in estimates])

    # ---- Ground truth is read only now: every structural, score-only
    # stage above (horizon stitching, final rescale, score-only labels
    # and dispute set, final-resolution components, exact composed
    # memberships, component source mass, active components, and the
    # Phase 1B compute projection) has already completed without ever
    # receiving `gt`. GT shape must already equal the canonical final
    # domain -- a mismatch fails closed, it never triggers a resize of
    # any prediction or component. -------------------------------------
    gt_raw = dataset.get_gt_seg_map_by_idx(resolved_native_index)
    ignore_index = int(dataset.ignore_index)
    gt = np.asarray(gt_raw).astype(np.int64)
    if gt.shape != ori_shape:
        raise DfpPhase1aExecuteError(
            f"image {canonical_image_id}: GT shape {gt.shape} disagrees with the canonical final ori_shape {ori_shape} "
            "-- refusing to resize any prediction or component to match"
        )

    # ---- Harmful/beneficial/third-label classification, the oracle
    # diagnostic maps, and every horizon-persistence map -- all
    # genuinely require GT and are computed only now, after it. This
    # re-derives a/b via the same pure derive_horizon_labels helper
    # internally (a cheap, deterministic, GT-free recomputation of the
    # exact same a_pre_gt/b_pre_gt above) -- never a hand-duplicated
    # formula, and analyze_phase1a_image itself is the frozen, unmodified
    # Step-3-10 helper, not reimplemented here.
    analysis = analyze_phase1a_image(scores_final, gt, contract=analysis_contract)

    # ---- The three remaining raw-horizon confusion variants (T0/T320/
    # oracle_O_P already computed inside analyze_phase1a_image). -------
    confusion_by_variant: dict[str, Any] = {
        "t0": analysis.confusion_u, "t320": analysis.confusion_p, "oracle_o_p": analysis.confusion_o_p,
    }
    for horizon in (16, 32, 64):
        confusion_by_variant[f"t{horizon}"] = compute_confusion_statistics(
            analysis.labels[horizon], gt, class_count=class_count_local, ignore_index=ignore_index,
        )
    confusion_by_variant["third_label_revert_to_t0"] = compute_confusion_statistics(
        analysis.oracle.third_label_diagnostic, gt, class_count=class_count_local, ignore_index=ignore_index,
    )
    subvariant_map = {
        "full": "full", "harmful_only": "harmful_recovery_only",
        "beneficial_only": "beneficial_undo_only", "third_label_only": "third_label_fallback_only",
    }
    for horizon, persistence_maps in analysis.persistence.items():
        for schema_subvariant, attribute_name in subvariant_map.items():
            label_map = getattr(persistence_maps, attribute_name)
            confusion_by_variant[f"persist_h{horizon}_{schema_subvariant}"] = compute_confusion_statistics(
                label_map, gt, class_count=class_count_local, ignore_index=ignore_index,
            )
    if set(confusion_by_variant) != set(artifact_schema.PREDICTION_VARIANT_NAMES_V2):
        raise DfpPhase1aExecuteError("internal error: assembled confusion variant set disagrees with the frozen v2 variant set")

    # ---- Boundary-band joint split (the frozen diagnostic primitive
    # only -- never the unbound Boundary-IoU metric). -------------------
    boundary_mask = _official_mask_to_boundary(analysis.dispute.dispute.astype(np.uint8)).astype(bool)
    disputed_boundary = int((analysis.dispute.dispute & boundary_mask).sum())
    disputed_interior = analysis.dispute.dispute_count - disputed_boundary
    harmful_boundary = int((analysis.dispute.harmful & boundary_mask).sum())
    harmful_interior = analysis.dispute.harmful_count - harmful_boundary

    # ---- Reversion coherence (Step 5): the committed definition
    # exactly -- baseline T320, fallback T0, reversion set = harmful
    # washout (never T32, never all disputes, never beneficial/third-
    # label pixels), evaluated at the canonical final ori_shape.
    # Connectivity and the pixel threshold come ONLY from AnalysisContract
    # (sourced from the live [reversion_coherence] identity block) --
    # no duplicated literal, no patch-to-pixel conversion.
    reverted_mask = analysis.dispute.harmful
    reversion = compute_reversion_coherence(
        reverted_mask, connectivity=analysis_contract.reversion_connectivity,
        small_component_threshold=analysis_contract.reversion_small_component_threshold_pixels,
    )

    evaluated_pixel_count = int(analysis.dispute.valid.sum())
    row: dict[str, Any] = {
        "dataset_indices": dataset_index,
        "evaluated_pixel_count": evaluated_pixel_count,
        "ignore_pixel_count": int(gt.size - evaluated_pixel_count),
        # dispute_pixel_count must live in the SAME domain as
        # evaluated_pixel_count (GT-restricted, non-ignore pixels only)
        # -- the schema's own relational validator enforces dispute_
        # pixel_count <= evaluated_pixel_count, and harmful+beneficial+
        # third_label already partition exactly this quantity. The
        # GT-free, score-only dispute set used for the Phase 1B compute
        # PROJECTION (dispute_pre_gt_mask -- final-resolution, but
        # unrestricted by GT/ignore pixels) is a distinct, internal-only
        # quantity that never populates this field.
        "dispute_pixel_count": analysis.dispute.dispute_count,
        "window_count": aggregate.total_windows,
        "zero_dispute_window_count": aggregate.zero_dispute_windows,
        "total_node_count": aggregate.total_node_count,
        "total_nonzero_count": aggregate.total_nonzero_count,
        "total_baseline_recurrence_flops": aggregate.total_baseline_recurrence_flops,
        "total_projected_recurrence_flops": aggregate.total_projected_recurrence_flops,
        "total_chunk_invocations": aggregate.total_chunk_invocations,
        "total_projected_width": int(sum(aggregate.projected_widths)),
        "maximum_projected_width": aggregate.maximum_projected_width,
        "maximum_peak_planned_bytes": aggregate.maximum_peak_planned_bytes,
        "harmful_pixel_count": analysis.dispute.harmful_count,
        "beneficial_pixel_count": analysis.dispute.beneficial_count,
        "third_label_pixel_count": analysis.dispute.third_label_count,
        "disputed_boundary_pixel_count": disputed_boundary,
        "disputed_interior_pixel_count": disputed_interior,
        "harmful_boundary_pixel_count": harmful_boundary,
        "harmful_interior_pixel_count": harmful_interior,
        "component_count": components.component_count,
        "reversion_component_count": reversion.component_count,
        "reversion_perimeter": reversion.perimeter,
        "q_w_mean": q_w_stats["mean"], "q_w_median": q_w_stats["median"], "q_w_p90": q_w_stats["p90"],
        "q_w_p95": q_w_stats["p95"], "q_w_p99": q_w_stats["p99"], "q_w_maximum": q_w_stats["maximum"],
        "component_class_indices": components.component_class_indices,
        "component_pixel_areas": components.component_pixel_areas,
        "component_patch_equivalent_areas": components.component_patch_equivalent_areas,
        "component_source_mass": component_source_mass,
        "reversion_component_sizes": np.array(reversion.component_sizes, dtype=np.int64),
    }
    for horizon in (16, 32, 64):
        pm = analysis.persistence[horizon]
        row[f"persistence_h{horizon}_harmful_reverted_count"] = pm.changed_pixel_counts["harmful"]
        row[f"persistence_h{horizon}_beneficial_reverted_count"] = pm.changed_pixel_counts["beneficial"]
        row[f"persistence_h{horizon}_third_label_reverted_count"] = pm.changed_pixel_counts["third_label"]
    for variant, stats in confusion_by_variant.items():
        row[artifact_schema.confusion_member_name_v2(variant, "intersection")] = stats.intersection
        row[artifact_schema.confusion_member_name_v2(variant, "union")] = stats.union
        row[artifact_schema.confusion_member_name_v2(variant, "predicted_pixels")] = stats.predicted_pixels
        row[artifact_schema.confusion_member_name_v2(variant, "ground_truth_pixels")] = stats.ground_truth_pixels
    return row, len(plan.windows)


# ---------------------------------------------------------------------------
# Fresh/resume orchestration and checkpoint installation.
# ---------------------------------------------------------------------------


def _now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write_checkpoint(
    context: ExecutionContext, accumulator: _V2Accumulator, *, completed_ids: Sequence[str], total_windows_processed: int,
) -> None:
    preflight = context.preflight
    complete = len(completed_ids) == preflight.expected_count
    fields = {
        "schema": artifact_schema.CHECKPOINT_SCHEMA_NAME,
        "schema_version": artifact_schema.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": preflight.artifact_schema_identity_sha256,
        "protocol_identity_sha256": preflight.protocol_identity_sha256,
        "subset_manifest_sha256": preflight.subset_manifest_sha256,
        "selected_ids_digest": preflight.selected_ids_digest,
        "run_mode": preflight.run_mode,
        "image_count_expected": preflight.expected_count,
        "class_count": preflight.class_count,
        "class_names_digest": preflight.class_names_digest,
        "horizon_order": list(artifact_schema.HORIZON_ORDER),
        "source_commit": preflight.source_commit,
        "required_ancestor_commit": preflight.required_ancestor_commit,
        "bridge_checkpoint_sha256": preflight.bridge_checkpoint_sha256,
        "next_dataset_index": len(completed_ids),
        "completed_image_ids": list(completed_ids),
        "images_completed_count": len(completed_ids),
        "total_windows_processed": total_windows_processed,
        "complete": complete,
        "final_result_produced": False,
        "created_at_utc": _now_utc_iso(),
        "updated_at_utc": _now_utc_iso(),
    }
    checkpoint_mod.generate_and_install_checkpoint_v2(
        checkpoints_dir=context.checkpoints_dir, checkpoint_filename=_CHECKPOINT_FILENAME, checkpoint_fields=fields,
        arrays=accumulator.arrays, image_ids=list(completed_ids), class_count=preflight.class_count,
    )


def _install_final_result(context: ExecutionContext) -> None:
    """Install and independently re-verify the final v2 result bundle
    (Step 9's already-committed, v1/v2-aware ``install_result_bundle``/
    ``verify_result_bundle``), then generate and verify the deterministic
    qualitative selection/report bundle. Both are Step 9/Step-qualitative
    responsibilities this driver only invokes -- it computes no metric
    and selects no example of its own."""
    from src.dfp_phase1a_report import install_result_bundle, verify_result_bundle

    install_result_bundle(
        destination_dir=context.result_bundle_dir, checkpoints_dir=context.checkpoints_dir,
        checkpoint_filename=_CHECKPOINT_FILENAME, repo_root=context.repo_root,
    )
    verify_result_bundle(context.result_bundle_dir)


def execute_run(context: ExecutionContext) -> dict[str, Any]:
    """Run every remaining image of ``context.preflight.selected_ids``
    starting at ``next_dataset_index``, checkpointing after each one,
    and install the final result bundle once all ``expected_count``
    images are complete. This is the ONLY function in this module that
    performs real model construction, CUDA initialization, or dataset
    image decoding -- and it does so strictly after ``build_execution_
    context`` has already returned successfully."""
    preflight = context.preflight

    # Finalize-only recovery (closes the real gap SLURM job 21063877
    # exposed): a checkpoint whose next_dataset_index already equals
    # expected_count has nothing left to process -- the per-image loop
    # below would already iterate zero times -- but reaching that empty
    # loop still cost a full model build, weight load, CUDA init, and
    # dataset construction on every retry. Checked here, BEFORE any of
    # that, using only the already-validated preflight fields (no new
    # state, no new flag): if the checkpoint is already complete,
    # finalize directly. install_result_bundle/verify_result_bundle
    # re-derive everything they need from the checkpoint, per-image-
    # stats manifest, and NPZ already on disk -- no image is read,
    # decoded, or reprocessed.
    if preflight.next_dataset_index == preflight.expected_count:
        _install_final_result(context)
        return {
            "images_completed": len(preflight.completed_image_ids),
            "expected_count": preflight.expected_count, "complete": True,
        }

    protocol_identity = load_protocol_identity(repo_root=context.repo_root)
    e3_identity = _load_e3_identity(context.repo_root, protocol_identity)
    contracts = _build_contracts(protocol_identity, repo_root=context.repo_root)
    graph_variant = str(protocol_identity["graph"]["canonical_k_variant"])

    from diagnostics.run_k11_k12_stability import _build_inference
    from src.dfp_phase1a_dataset_view import build_manifest_execution_view

    inference, dataset = _build_inference(context.repo_root, e3_identity, context.device, log_dir=context.checkpoints_dir)

    # Manifest-order -> native-dataset-index mapping, built and FULLY
    # validated for every one of the frozen selected_ids (not merely the
    # remaining resume suffix) before any image is processed -- see
    # src/dfp_phase1a_dataset_view.py. A construction failure here (a
    # missing/duplicate/unsafe/wrongly-suffixed mapping anywhere in the
    # manifest) aborts before the first _process_one_image call, exactly
    # as it would have aborted mid-run under the old positional-index
    # bug, but now at the cheapest possible point instead of after
    # partial GPU work.
    execution_view = build_manifest_execution_view(dataset, preflight.selected_ids)
    if len(execution_view) != preflight.expected_count:
        raise DfpPhase1aExecuteError(
            f"execution view has {len(execution_view)} positions, expected exactly {preflight.expected_count}"
        )

    accumulator = _V2Accumulator(class_count=preflight.class_count, seed_arrays=preflight.checkpoint_arrays)
    completed_ids = list(preflight.completed_image_ids)
    total_windows_processed = 0  # only tracked forward from this invocation; resumed windows are not re-derivable and are not required by the schema

    for index in range(preflight.next_dataset_index, preflight.expected_count):
        position = execution_view[index]
        # canonical_manifest_id: bare, frozen manifest ID -- row/checkpoint
        # identity (never suffix-bearing). position.relative_path: the
        # SAME adapter-validated ManifestExecutionPosition's own suffix-
        # bearing, image-root-relative path -- authoritative ONLY for
        # _extract_prepared_image's reconciliation. These are two
        # deliberately distinct identities (see _process_one_image's
        # docstring); never conflate them, and never reconstruct one from
        # the other by string concatenation -- both already come from this
        # single already-validated execution-view position.
        canonical_manifest_id = preflight.selected_ids[index]
        if position.manifest_position != index or position.canonical_id != canonical_manifest_id:
            raise DfpPhase1aExecuteError(
                f"execution view position {index} (canonical_id={position.canonical_id!r}) disagrees with "
                f"the frozen manifest position {index} (selected_ids={canonical_manifest_id!r})"
            )
        row, window_count = _process_one_image(
            inference, dataset, index, canonical_manifest_id, protocol_identity=protocol_identity,
            contracts=contracts, class_count=preflight.class_count, graph_variant=graph_variant,
            native_dataset_index=position.native_index,
            canonical_relative_image_path=position.relative_path,
        )
        accumulator.append_image_row(row)
        completed_ids.append(canonical_manifest_id)
        total_windows_processed += window_count
        _write_checkpoint(context, accumulator, completed_ids=completed_ids, total_windows_processed=total_windows_processed)

    if len(completed_ids) == preflight.expected_count:
        _install_final_result(context)

    return {"images_completed": len(completed_ids), "expected_count": preflight.expected_count, "complete": len(completed_ids) == preflight.expected_count}


# ---------------------------------------------------------------------------
# The CLI ``execute`` subcommand itself lives in
# ``diagnostics/run_dfp_phase1a.py`` (the unified DFP Phase 1A CLI,
# consistent with every other subcommand there); this module exposes
# only the two functions that CLI wires directly:
# ``build_execution_context`` and ``execute_run``.
# ---------------------------------------------------------------------------

__all__ = [
    "RUN_NAME",
    "DfpPhase1aExecuteError",
    "ExecutionContext",
    "build_execution_context",
    "execute_run",
]
