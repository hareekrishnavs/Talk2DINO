"""DFP Phase 1B: the 200-window conservation-calibration runner,
authorized by the committed E16 successor authority
(`evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml`).

**Duplication gate.** This module invents no new scientific primitive.
Every number this runner produces is computed by an already-committed,
already-tested Phase 1A/1B module, called through its public API
exactly as its own test suite exercises it:

* candidate-window geometry -- ``segmentation.evaluation.sliding_
  window_geometry.SlidingWindowPlan`` (the same canonical window-plan
  implementation E16 hash-binds);
* per-window snapshot/graph/canonical-recurrence -- the SAME calls
  ``src.dfp_phase1a_execute._process_one_window_dfp`` makes
  (``inference.model.generate_patch_snapshot``, ``models.dinotext.
  cover_dr.graph.build_directed_topk_graph``, ``src.dfp_phase1a_
  recurrence_capture.capture_dfp_phase1a_trajectory``) -- reproduced
  here as a parallel, narrow orchestration function (mirroring, never
  importing, that private helper) ONLY because this runner additionally
  needs to retain the built graph instance for selected windows, which
  the existing helper does not expose to its caller;
* multi-window canonical stitching -- ``src.dfp_phase1a_horizon_
  stitching.stitch_dfp_phase1a_horizons``;
* final-resolution rescale -- ``src.dfp_phase1a_final_rescale_pullback.
  apply_final_rescale_forward``;
* T0/T320 label derivation -- ``src.dfp_phase1a_analysis.derive_
  horizon_labels`` (the exact lowest-index tie rule);
* global unary components -- ``src.dfp_phase1a_global_unary_
  components.compute_global_unary_components``;
* the composed final-resolution membership/mass operator -- ``src.
  dfp_phase1a_final_rescale_pullback.iter_image_final_soft_
  memberships`` -- the SAME function ``dfp_phase1b_masked_forward.
  build_masked_seed_chunk``'s own docstring names as the authoritative
  source of ``membership_rows``, and the SAME function ``src.dfp_
  phase1a_execute._process_one_image`` already uses for its own
  component-source-mass/active-component derivation;
* window-restricted forward stitching -- ``src.dfp_phase1a_stitching_
  pullback.apply_window_stitching_forward`` (used both by the masked-
  forward core internally and, here, fed a synthetic all-ones patch
  indicator to determine one window's own final-resolution support --
  see the note below);
* the exact masked-forward provenance core -- every function in ``src.
  dfp_phase1b_masked_forward`` (``build_masked_seed_chunk``,
  ``propagate_masked_seed_chunk``, ``stitch_and_rescale_component_
  scores``, ``compute_signed_contributions``, and all four
  ``compute_*_residual`` functions) is called, never re-implemented;
* the E16 candidate/selection/artifact machinery -- every function in
  ``src.dfp_phase1b_calibration_identity``/``src.dfp_phase1b_
  calibration_artifact`` is called, never re-implemented.

**A genuinely new (but non-scientific) design decision, documented
here because it is load-bearing and because two committed sources
disagree about it:** ``dfp_phase1b_masked_forward.py``'s own module
docstring names ``src.dfp_phase1b_compute_estimator.compute_window_
query_classes`` as "the single frozen Q_w derivation this module's
callers are expected to reuse." But ``tests/test_dfp_phase1a_
execute.py``'s own ``test_driver_source_contains_no_deleted_temporary_
paths`` explicitly documents that function (together with ``compute_
window_active_components``, ``iter_soft_membership_records``, and
``compute_stitched_dispute_set``) as "the old frozen helper bound to
inference-resolution-only dispute maps" -- a judgment-call path Step
11 (``dfp_phase1a_execute.py``) deliberately REPLACED with a final-
(ori-)resolution-consistent derivation computed inline in ``_process_
one_image`` (component-class-index + disputed-b-value union, over
``active_components_by_window`` derived from ``iter_image_final_soft_
memberships``). Since E16's own ``rho_g`` domain is explicitly defined
at final resolution ("the selected window's own final-resolution
support... where a(x) != b(x)"), and since ``compute_window_query_
classes`` structurally REQUIRES its ``a``/``b``/``dispute`` arrays at
``window_plan.image_size`` (the STITCHED, pre-final-rescale, inference-
resolution domain -- confirmed by reading its own source), using it
here would silently mix two different resolutions. This module
therefore reproduces ``_process_one_image``'s CURRENT, already-
committed, final-resolution-consistent Q_w logic exactly (never the
superseded inference-resolution helper), which is also the only choice
that keeps Q_w, the disputed-pixel domain, and the membership/mass
computation all in the SAME (final/ori) resolution throughout.

**A second necessary, explicitly-derived (not invented) design
decision:** E16's ``rho_g``/``delta_m`` formulas both specify "the
[selected window]'s own... contribution" to the final-resolution
field, stitched via the "complete canonical window plan" denominator
(never a partial/reduced one). This module realizes that literally:
for a selected window w, both the masked numerator (``g^(w)``) and the
canonical numerator (``delta_m^(w)``) are stitched by calling the
existing, unmodified ``stitch_and_rescale_component_scores`` with a
``propagated_by_window`` mapping containing ONLY window w's own chunk,
while ``window_plan``/``final_plan`` remain the image's COMPLETE,
all-window plan (so the shared denominator is never reduced to a
single window). The canonical numerator is packaged as a
``PropagatedChunk`` with ``component_count=1`` (t0/t_final = window w's
own canonical T0/T320 states restricted to its Q_w columns) -- the same
stitch/rescale function handles this correctly because it is already
generic in component count; no new formula is introduced, and this is
exactly why E16's text says both terms "use the identical final-
resolution operator."

**A third necessary, explicitly-derived design decision: "the selected
window's own final-resolution support."** No committed function
directly answers "which final(ori)-resolution pixels does window w's
own inference-resolution crop project onto." This module derives it
by feeding an all-ones patch-node indicator (`[1, N_w]`) through the
UNMODIFIED ``apply_window_stitching_forward`` (which already restricts
its output to exactly ``window.accumulation_slice`` and already
divides by the shared denominator, per its own docstring), embedding
that into a zero inference-resolution canvas, and applying the
UNMODIFIED ``apply_final_rescale_forward`` to it -- a standard
"coverage via indicator" probe of an already-existing linear operator,
never a new geometry formula. The boundary is taken as "any nonzero
projected weight" (an exact, non-thresholded boolean partition of the
final domain, analogous to `a(x) != b(x)` already being a hard
boolean), never a magnitude threshold on a *residual* value.

**This module implements and synthetically tests the runner only.** It
does not execute the real calibration, does not access a real dataset
or real model weights, does not initialize CUDA, does not submit
SLURM, and does not select or apply a numeric conservation tolerance.
E16 does not authorize resume/checkpoint semantics anywhere (no
"resume"/"checkpoint" field exists in its TOML) -- this runner is
therefore FRESH-ONLY: it either completes a full run and installs one
complete, final artifact, or it raises without installing anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.dfp_phase1b_calibration_artifact import (
    ARTIFACT_SCHEMA_NAME,
    ARTIFACT_SCHEMA_VERSION,
    RESIDUAL_FAMILIES,
    WINDOW_COUNT,
    aggregate_residual_summaries,
    canonical_candidate_order_key,
    select_calibration_windows,
    sort_candidate_universe,
    validate_calibration_artifact,
)
from src.dfp_phase1b_calibration_identity import (
    DfpPhase1bCalibrationIdentityError,
    load_identity as load_e16_identity,
    repository_root,
)


class DfpPhase1bCalibrationRunError(RuntimeError):
    """Raised on any calibration-runner invariant violation. Always
    fail closed: never silently substitute a default, skip a required
    check, or install a partial/inconsistent artifact. Never raised
    for, and never catches, KeyboardInterrupt, SystemExit, or
    MemoryError -- those always propagate unmodified."""


# ---------------------------------------------------------------------
# Phase A: pure preflight. No torch import, no CUDA, no model, no
# image decoding anywhere in this section.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightReport:
    """Everything Phase A proves before any heavy import. Carries no
    torch object, no model handle, no dataset handle."""

    e16_identity: Mapping[str, Any]
    repo_root: Path
    output_path: Path
    mode: str
    subset_manifest: Mapping[str, Any]
    selected_ids: tuple[str, ...]


def _require_fresh_mode(mode: str) -> str:
    if mode != "fresh":
        raise DfpPhase1bCalibrationRunError(
            f"mode {mode!r} is not authorized -- the committed E16 authority "
            "(evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml) defines no "
            "resume/checkpoint semantics anywhere (no 'resume' or 'checkpoint' field exists in it), "
            "so this runner only supports mode='fresh'. Do not invent resume semantics."
        )
    return mode


def load_subset_manifest(repo_root: Path, e16_identity: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load and relationally verify the frozen COCO-Stuff-500 subset
    manifest against E16's own pinned hash -- never trusted from disk
    alone, and never re-hashed/rewritten here."""
    binding = e16_identity["dependency_bindings"]["subset_manifest"]
    manifest_path = repo_root / binding["relative_path"]
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise DfpPhase1bCalibrationRunError(f"cannot read frozen subset manifest {manifest_path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != binding["sha256"]:
        raise DfpPhase1bCalibrationRunError(
            f"subset manifest {manifest_path} SHA-256 mismatch -- pinned {binding['sha256']}, observed {actual_sha256}"
        )
    try:
        manifest = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DfpPhase1bCalibrationRunError(f"subset manifest {manifest_path} is not valid JSON: {exc}") from exc
    if manifest.get("run_mode") != binding["run_mode"]:
        raise DfpPhase1bCalibrationRunError(
            f"subset manifest run_mode {manifest.get('run_mode')!r} disagrees with the frozen "
            f"dependency_bindings.subset_manifest.run_mode {binding['run_mode']!r}"
        )
    selected_ids = manifest.get("selected_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) != binding["selected_count"]:
        raise DfpPhase1bCalibrationRunError(
            f"subset manifest selected_ids must be a list of exactly {binding['selected_count']} entries"
        )
    return manifest


def _reject_lexical_path_traversal(output_path: Path) -> None:
    """Reject any output path whose LEXICAL (un-resolved) path
    components contain a literal ``'..'`` parent-directory reference --
    checked on ``Path(output_path).parts`` directly, never after
    ``.resolve()``/normalization, so a traversal that would otherwise
    resolve to a valid-looking absolute destination (e.g. an operator
    typo that lands on ``/etc/phase1b_conservation_calibration.json``)
    is caught before that resolution ever happens. Compares whole path
    components (never substrings), so a component like
    ``"name..part"`` -- not equal to the exact string ``".."`` -- is
    never mistaken for a traversal segment. A legitimate absolute path
    with no ``'..'`` component (including the eventual real
    ``/scratch/.../phase1b_conservation_calibration.json`` destination)
    is never rejected by this check, and this function implements no
    authorized-output-root policy of any kind -- only this one lexical
    rule."""
    if ".." in Path(output_path).parts:
        raise DfpPhase1bCalibrationRunError(
            f"output_path {output_path} contains a '..' parent-directory path component -- parent "
            "traversal is forbidden in the calibration artifact output path, whether the path is "
            "relative or the '..' component is embedded inside an otherwise absolute path"
        )


def _validate_output_path(output_path: Path, *, allow_overwrite: bool) -> None:
    _reject_lexical_path_traversal(output_path)
    if not output_path.parent.is_dir():
        raise DfpPhase1bCalibrationRunError(f"output directory does not exist: {output_path.parent}")
    if output_path.exists() and not allow_overwrite:
        raise DfpPhase1bCalibrationRunError(
            f"output path already exists and overwrite was not explicitly authorized: {output_path} "
            "(pass allow_overwrite=True / --force to replace an existing artifact)"
        )
    if output_path.exists() and output_path.is_dir():
        raise DfpPhase1bCalibrationRunError(f"output path is a directory, not a file: {output_path}")


def run_preflight(
    *, output_path: Path, mode: str = "fresh", allow_overwrite: bool = False, repo_root: Path | None = None,
) -> PreflightReport:
    """Phase A. Load and relationally verify E16 (which itself
    relationally re-verifies E15/the masked-forward core/the subset
    manifest/the window-plan implementation/the stitching and final-
    rescale pullback implementations/the graph and recurrence kernels
    against their pinned SHA-256 hashes), verify run-mode binding,
    verify the output path, and reject any unsupported mode -- all
    before importing torch or touching a dataset."""
    root = Path(repo_root) if repo_root is not None else repository_root()
    _require_fresh_mode(mode)
    # Cheap, dependency-free structural validation of the output path
    # runs before E16 loading/hashing (and therefore before any
    # filesystem mutation of any kind) -- a malformed --output should
    # never pay for authority verification it can't possibly use.
    output_path = Path(output_path)
    _reject_lexical_path_traversal(output_path)

    try:
        e16_identity = load_e16_identity(repo_root=root)
    except DfpPhase1bCalibrationIdentityError as exc:
        raise DfpPhase1bCalibrationRunError(f"E16 identity failed to load/verify: {exc}") from exc

    manifest = load_subset_manifest(root, e16_identity)
    if e16_identity["candidate_scope"]["run_mode"] != manifest["run_mode"]:
        raise DfpPhase1bCalibrationRunError(
            "E16 candidate_scope.run_mode disagrees with the loaded subset manifest's run_mode"
        )
    if e16_identity["candidate_scope"]["image_count"] != len(manifest["selected_ids"]):
        raise DfpPhase1bCalibrationRunError(
            "E16 candidate_scope.image_count disagrees with the loaded subset manifest's selected_ids length"
        )

    expected_name = e16_identity["artifact_schema"]["required_artifact_relative_path"]
    if output_path.name != expected_name and output_path.suffix != ".json":
        raise DfpPhase1bCalibrationRunError(
            f"output_path {output_path} does not look like the authority-bound artifact "
            f"({expected_name}) -- refusing to write a differently-named file by mistake"
        )
    _validate_output_path(output_path, allow_overwrite=allow_overwrite)

    return PreflightReport(
        e16_identity=e16_identity, repo_root=root, output_path=output_path, mode=mode,
        subset_manifest=manifest, selected_ids=tuple(manifest["selected_ids"]),
    )


# ---------------------------------------------------------------------
# Phase B: candidate-universe construction and deterministic selection.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ImageGeometry:
    """The minimal, cheap (no full decode, no model forward pass)
    per-image geometry Phase B needs: the resolved inference-canvas
    size and the model's own (data-independent) patch size. In
    production this comes from the dataset's own lightweight metadata
    plus the model's fixed patch-embedding stride -- never from
    running the model."""

    inference_height: int
    inference_width: int
    patch_size: int


class GeometryProvider(Protocol):
    def __call__(self, canonical_id: str) -> ImageGeometry: ...


@dataclass(frozen=True)
class WindowPlanBuilder:
    """Wraps the real, committed ``SlidingWindowPlan.build`` with the
    frozen crop/stride E16's own corrected prose points at --
    ``e13_dfp_phase1a_protocol.toml``'s ``[score_interpolation_
    stitching]`` block, reused verbatim, never re-derived here."""

    crop_height: int
    crop_width: int
    stride_height: int
    stride_width: int

    def build(self, *, inference_height: int, inference_width: int):
        from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize

        return SlidingWindowPlan.build(
            image_size=SpatialSize(inference_height, inference_width),
            crop_size=SpatialSize(self.crop_height, self.crop_width),
            stride=SpatialSize(self.stride_height, self.stride_width),
        )


def default_window_plan_builder(repo_root: Path) -> WindowPlanBuilder:
    """Reads the frozen crop/stride directly from
    ``e13_dfp_phase1a_protocol.toml``'s own ``[score_interpolation_
    stitching]`` block -- the exact location E16's own (corrected)
    ``candidate_windows_per_image_source`` prose names."""
    import tomllib

    with (repo_root / "evaluation_identities/e13_dfp_phase1a_protocol.toml").open("rb") as handle:
        e13 = tomllib.load(handle)
    crop = e13["score_interpolation_stitching"]["crop"]
    stride = e13["score_interpolation_stitching"]["stride"]
    return WindowPlanBuilder(crop_height=crop[0], crop_width=crop[1], stride_height=stride[0], stride_width=stride[1])


def build_candidate_universe(
    selected_ids: Sequence[str], *, geometry_provider: GeometryProvider, window_plan_builder: WindowPlanBuilder,
    image_paths: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate every canonical sliding window for every one of the
    frozen manifest images, in manifest order, using ONLY image/
    pipeline geometry -- never GT, never a prediction, never a model
    forward pass. ``image_paths`` optionally supplies the canonical
    dataset-relative path per ID (defaults to the frozen COCO-Stuff
    val2017 naming convention)."""
    universe: list[dict[str, Any]] = []
    for image_ordinal, canonical_id in enumerate(selected_ids):
        geometry = geometry_provider(canonical_id)
        plan = window_plan_builder.build(inference_height=geometry.inference_height, inference_width=geometry.inference_width)
        patch_grid_height = geometry.inference_height // geometry.patch_size
        patch_grid_width = geometry.inference_width // geometry.patch_size
        relative_path = (image_paths or {}).get(canonical_id, f"val2017/{canonical_id}.jpg")
        for window in plan.windows:
            universe.append(
                {
                    "run_mode": "coco_stuff_500",
                    "image_id": canonical_id,
                    "image_path": relative_path,
                    "image_manifest_ordinal": image_ordinal,
                    "inference_canvas_height": geometry.inference_height,
                    "inference_canvas_width": geometry.inference_width,
                    "canonical_window_ordinal": window.index,
                    "top": int(window.origin.row),
                    "left": int(window.origin.col),
                    "bottom": int(window.end.row),
                    "right": int(window.end.col),
                    "crop_height": window_plan_builder.crop_height,
                    "crop_width": window_plan_builder.crop_width,
                    "patch_grid_height": patch_grid_height,
                    "patch_grid_width": patch_grid_width,
                }
            )
    return sort_candidate_universe(universe)


@dataclass(frozen=True)
class SelectionOutcome:
    selected_rows: tuple[dict[str, Any], ...]  # canonical candidate order, each row carrying selection metadata
    candidate_universe_digest: str
    selected_windows_digest: str
    selection_trace_digest: str
    candidate_window_count: int


def select_and_verify(candidate_universe: Sequence[Mapping[str, Any]]) -> SelectionOutcome:
    """Run the committed E16 selector exactly once."""
    ordered = sort_candidate_universe(candidate_universe)
    result = select_calibration_windows(ordered)
    trace_by_index = {t["canonical_candidate_index"]: t for t in result.selection_trace}
    rows = tuple(
        {
            **dict(record),
            "canonical_candidate_index": index,
            "selection_key": trace_by_index[index]["selection_key"],
            "selection_rank": trace_by_index[index]["selection_rank"],
        }
        for index, record in zip(result.selected_canonical_indices, result.selected_records)
    )
    if len(rows) != WINDOW_COUNT:
        raise DfpPhase1bCalibrationRunError(f"internal error: selected {len(rows)} windows, expected {WINDOW_COUNT}")
    return SelectionOutcome(
        selected_rows=rows,
        candidate_universe_digest=result.candidate_universe_digest,
        selected_windows_digest=result.selected_windows_digest,
        selection_trace_digest=result.selection_trace_digest,
        candidate_window_count=result.candidate_universe_size,
    )


# ---------------------------------------------------------------------
# Pure residual-summary reduction (independently unit-testable with
# plain torch tensors -- no image/model/graph dependency).
# ---------------------------------------------------------------------


def summarize_residual_tensor(residual: Any) -> dict[str, Any]:
    """Reduce one FP64 residual tensor (any shape) to the frozen E16
    raw summary fields. Never clips, rounds, thresholds, or discards a
    value."""
    import torch

    if not torch.is_tensor(residual):
        raise DfpPhase1bCalibrationRunError("summarize_residual_tensor requires a torch.Tensor")
    flat = residual.detach().reshape(-1).to(torch.float64)
    sample_count = int(flat.numel())
    if sample_count == 0:
        return _null_summary(sample_count=0, finite_count=0, non_finite_count=0)
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    non_finite_count = sample_count - finite_count
    flat_for_stats = flat[finite_mask]
    if flat_for_stats.numel() == 0:
        return _null_summary(sample_count=sample_count, finite_count=finite_count, non_finite_count=non_finite_count)
    absolute = flat_for_stats.abs()
    signed_min = float(flat_for_stats.min().item())
    signed_max = float(flat_for_stats.max().item())
    absolute_sum = float(absolute.sum().item())
    squared_sum = float((flat_for_stats * flat_for_stats).sum().item())
    stat_count = int(flat_for_stats.numel())
    return {
        "sample_count": sample_count,
        "finite_count": finite_count,
        "non_finite_count": non_finite_count,
        "signed_minimum": signed_min,
        "signed_maximum": signed_max,
        "maximum_absolute_residual": max(abs(signed_min), abs(signed_max)),
        "mean_absolute_residual": absolute_sum / stat_count,
        "root_mean_square_residual": (squared_sum / stat_count) ** 0.5,
        "absolute_residual_sum": absolute_sum,
        "squared_residual_sum": squared_sum,
    }


def _null_summary(*, sample_count: int, finite_count: int, non_finite_count: int) -> dict[str, Any]:
    return {
        "sample_count": sample_count, "finite_count": finite_count, "non_finite_count": non_finite_count,
        "signed_minimum": None, "signed_maximum": None,
        "maximum_absolute_residual": None, "mean_absolute_residual": None, "root_mean_square_residual": None,
        "absolute_residual_sum": None, "squared_residual_sum": None,
    }


# ---------------------------------------------------------------------
# Phase C: canonical + masked execution for one selected image.
# Dependency-injected: production supplies real inference/dataset
# objects; tests supply small synthetic ones.
# ---------------------------------------------------------------------


class InferenceProvider(Protocol):
    """Bundles exactly what the per-window snapshot call needs, so
    production can supply the real ``ExecutionContext.inference``
    object and tests can supply a tiny synthetic stand-in."""

    model: Any  # .generate_patch_snapshot(crop, text_embedding) -> snapshot
    text_embedding: Any


@dataclass(frozen=True)
class PreparedImage:
    image_tensor: Any  # [1, 3, H, W]
    inference_height: int
    inference_width: int
    ori_shape: tuple[int, int]


class ImagePreparer(Protocol):
    def __call__(self, canonical_id: str) -> PreparedImage: ...


@dataclass(frozen=True)
class WindowExecutionResult:
    """Everything Phase C needs to carry forward from one window's
    canonical processing: the canonical trajectory trace, and -- ONLY
    for selected windows -- the exact graph instance used to build it
    (retained for the paired masked propagation)."""

    window_index: int
    trace: Any
    graph: Any | None  # the real DirectedTopKGraph instance, retained only when selected
    unary_scores: Any | None  # s0, retained only when selected


def _process_one_window_for_calibration(
    inference: InferenceProvider, image_tensor: Any, window: Any, *, protocol_identity: Mapping[str, Any],
    affinity_power: float, class_count: int, retain_graph: bool,
) -> WindowExecutionResult:
    """Mirrors ``src.dfp_phase1a_execute._process_one_window_dfp``
    exactly (same calls, same order, same one-graph-one-trajectory
    structure) but additionally retains the built ``DirectedTopKGraph``
    instance -- and the unary snapshot it was built from -- when
    ``retain_graph`` is true, so a selected window's later masked
    propagation can reuse the SAME graph object."""
    import torch

    from models.dinotext.cover_dr.graph import build_directed_topk_graph
    from src.dfp_phase1a_recurrence_capture import capture_dfp_phase1a_trajectory

    crop_rows, crop_cols = window.crop_slice
    crop = image_tensor[:, :, crop_rows, crop_cols]

    with torch.no_grad():
        snapshot = inference.model.generate_patch_snapshot(crop, inference.text_embedding)
    s0 = snapshot.unary_scores[0]
    if s0.shape[1] != class_count:
        raise DfpPhase1bCalibrationRunError(
            f"window {window.index}: snapshot class dimension {s0.shape[1]} disagrees with class_count {class_count}"
        )
    dino_features = snapshot.dino_features[0]

    graph12 = build_directed_topk_graph(dino_features, k=12, affinity_power=affinity_power)
    trace = capture_dfp_phase1a_trajectory(graph12, s0, protocol_identity=protocol_identity)

    return WindowExecutionResult(
        window_index=window.index, trace=trace,
        graph=graph12 if retain_graph else None, unary_scores=s0 if retain_graph else None,
    )


@dataclass(frozen=True)
class ComponentContractsBundle:
    """The already-committed Phase 1A/1B contracts this runner reuses
    unmodified, extracted once (per protocol identity) and reused for
    every selected image."""

    recurrence: Any
    stitching: Any
    component: Any
    pullback: Any
    final_resolution: Any
    masked_forward: Any


def build_component_contracts(
    phase1a_protocol_identity: Mapping[str, Any], phase1b_identity: Mapping[str, Any], *, repo_root: Path,
) -> ComponentContractsBundle:
    from src.dfp_phase1a_final_rescale_pullback import dfp_final_resolution_contract_from_protocol
    from src.dfp_phase1a_global_unary_components import dfp_component_contract_from_protocol
    from src.dfp_phase1a_horizon_stitching import dfp_stitching_contract_from_protocol
    from src.dfp_phase1a_recurrence_capture import dfp_recurrence_contract_from_protocol
    from src.dfp_phase1a_stitching_pullback import dfp_pullback_contract_from_protocol
    from src.dfp_phase1b_masked_forward import dfp_masked_forward_contract_from_protocol

    return ComponentContractsBundle(
        recurrence=dfp_recurrence_contract_from_protocol(phase1a_protocol_identity),
        stitching=dfp_stitching_contract_from_protocol(phase1a_protocol_identity, repo_root=repo_root),
        component=dfp_component_contract_from_protocol(phase1a_protocol_identity, repo_root=repo_root),
        pullback=dfp_pullback_contract_from_protocol(phase1a_protocol_identity, repo_root=repo_root),
        final_resolution=dfp_final_resolution_contract_from_protocol(phase1a_protocol_identity, repo_root=repo_root),
        masked_forward=dfp_masked_forward_contract_from_protocol(phase1a_protocol_identity, phase1b_identity),
    )


def _window_final_support_mask(window_index: int, *, pullback_plan: Any, final_resolution_contract: Any, img_shape: tuple[int, int], ori_shape: tuple[int, int]):
    """"The selected window's own final-resolution support": feed an
    all-ones patch-node indicator through the UNMODIFIED
    ``apply_window_stitching_forward`` (already restricted to
    ``window.accumulation_slice``, already denominator-normalized),
    embed it into a zero inference-resolution canvas, and apply the
    UNMODIFIED ``apply_final_rescale_forward`` -- a coverage probe of
    an existing linear operator, never a new geometry formula."""
    import torch

    from src.dfp_phase1a_final_rescale_pullback import apply_final_rescale_forward
    from src.dfp_phase1a_stitching_pullback import apply_window_stitching_forward

    patch_count = pullback_plan.patch_grid[0] * pullback_plan.patch_grid[1]
    indicator_patches = torch.ones((1, patch_count), dtype=pullback_plan.dtype)
    contribution = apply_window_stitching_forward(indicator_patches, window_plan=pullback_plan, window_index=window_index)
    window = pullback_plan.plan.windows[window_index]
    rows, cols = window.accumulation_slice
    canvas = torch.zeros((1, 1) + tuple(img_shape), dtype=pullback_plan.dtype)
    canvas[0, 0, rows, cols] = contribution[0]
    final_indicator = apply_final_rescale_forward(canvas, img_shape=img_shape, ori_shape=ori_shape, contract=final_resolution_contract)
    return final_indicator.squeeze(0).squeeze(0) > 0


def process_selected_image(
    inference: InferenceProvider, prepared: PreparedImage, canonical_id: str, *,
    selected_windows_for_image: Sequence[int], protocol_identity: Mapping[str, Any],
    contracts: ComponentContractsBundle, class_count: int, affinity_power: float, component_chunk_size: int,
) -> dict[int, dict[str, dict[str, Any]]]:
    """The full Phase C pipeline for one selected image. Returns
    ``{window_index: {residual_family: raw_summary_dict}}`` for exactly
    ``selected_windows_for_image``. Never reads GT. Never runs masked
    propagation for a window not in ``selected_windows_for_image``.
    Every selected window's canonical and masked propagation share the
    SAME ``DirectedTopKGraph`` instance."""
    import numpy as np
    import torch

    from src.dfp_phase1a_analysis import derive_horizon_labels
    from src.dfp_phase1a_final_rescale_pullback import (
        apply_final_rescale_forward, build_final_stitching_pullback_plan, iter_image_final_soft_memberships,
    )
    from src.dfp_phase1a_global_unary_components import compute_global_unary_components
    from src.dfp_phase1a_horizon_stitching import WindowHorizonCaptures, stitch_dfp_phase1a_horizons
    from src.dfp_phase1a_stitching_pullback import build_stitching_pullback_plan
    from src.dfp_phase1b_masked_forward import (
        PropagatedChunk, build_masked_seed_chunk, compute_membership_partition_residual,
        compute_propagated_reconstruction_residual, compute_seed_reconstruction_residual,
        compute_signed_contributions, compute_signed_margin_conservation_residual, propagate_masked_seed_chunk,
        stitch_and_rescale_component_scores,
    )

    selected_set = set(selected_windows_for_image)
    image_tensor = prepared.image_tensor
    img_shape = (prepared.inference_height, prepared.inference_width)
    ori_shape = prepared.ori_shape
    ori_height, ori_width = ori_shape

    pullback_plan = build_stitching_pullback_plan(img_shape, contract=contracts.pullback)

    window_results: dict[int, WindowExecutionResult] = {}
    window_captures = []
    for window in pullback_plan.plan.windows:
        result = _process_one_window_for_calibration(
            inference, image_tensor, window, protocol_identity=protocol_identity,
            affinity_power=affinity_power, class_count=class_count, retain_graph=window.index in selected_set,
        )
        if result.trace.steps_completed != contracts.recurrence.steps:
            raise DfpPhase1bCalibrationRunError(
                f"image {canonical_id} window {window.index}: recurrence completed {result.trace.steps_completed} "
                f"updates, expected {contracts.recurrence.steps}"
            )
        window_results[window.index] = result
        window_captures.append(WindowHorizonCaptures(window_index=window.index, captures=result.trace.snapshots))

    stitched_inf = stitch_dfp_phase1a_horizons(
        pullback_plan.plan, tuple(window_captures), contract=contracts.stitching, class_count=class_count,
    )

    scores_final: dict[int, Any] = {}
    for horizon, tensor in stitched_inf.items():
        rescaled = apply_final_rescale_forward(tensor, img_shape=img_shape, ori_shape=ori_shape, contract=contracts.final_resolution)
        scores_final[horizon] = rescaled.squeeze(0).detach()

    labels = derive_horizon_labels({h: t.cpu().numpy() for h, t in scores_final.items()})
    a_labels = labels[0]
    b_labels = labels[contracts.recurrence.steps]
    dispute_mask_np = a_labels != b_labels
    dispute_mask_t = torch.from_numpy(dispute_mask_np)

    components = compute_global_unary_components(
        scores_final[0].cpu().numpy(), contract=contracts.component, class_count=class_count,
    )

    final_plan = build_final_stitching_pullback_plan(
        pullback_plan, img_shape=img_shape, ori_shape=ori_shape, contract=contracts.final_resolution,
    )

    active_components_by_window: dict[int, set[int]] = {w.index: set() for w in pullback_plan.plan.windows}
    membership_chunks: dict[int, dict[tuple[int, int], Any]] = {}
    for start, chunk in iter_image_final_soft_memberships(
        components.component_ids, components.component_count, final_plan=final_plan, component_chunk_size=component_chunk_size,
    ):
        for window_index, membership in chunk.items():
            for offset, is_positive in enumerate((membership > 0).any(dim=1).tolist()):
                if is_positive:
                    active_components_by_window[window_index].add(start + offset)
            if window_index in selected_set:
                membership_chunks.setdefault(window_index, {})[(start, start + membership.shape[0])] = membership

    disputed_component_ids = components.component_ids[dispute_mask_np]
    disputed_b_values = b_labels[dispute_mask_np]
    component_b_value_sets: dict[int, set[int]] = {}
    for cid, bval in zip(disputed_component_ids.tolist(), disputed_b_values.tolist()):
        component_b_value_sets.setdefault(cid, set()).add(bval)

    results: dict[int, dict[str, dict[str, Any]]] = {}

    for window_index in sorted(selected_set):
        window_result = window_results[window_index]
        graph = window_result.graph
        s0_full = window_result.unary_scores

        active_components = tuple(sorted(active_components_by_window.get(window_index, ())))
        query_class_set: set[int] = set()
        for component_id in active_components:
            query_class_set.add(int(components.component_class_indices[component_id]))
            query_class_set.update(component_b_value_sets.get(component_id, ()))
        query_classes = tuple(sorted(query_class_set))

        family_summaries: dict[str, dict[str, Any]] = {}

        if not query_classes or not active_components or window_index not in membership_chunks:
            # A valid, legitimately empty selected window: zero active
            # components collapses every residual family to a
            # zero-sample row -- never a fabricated nonzero value.
            for family in RESIDUAL_FAMILIES:
                family_summaries[family] = summarize_residual_tensor(torch.zeros(0, dtype=torch.float64))
            results[window_index] = family_summaries
            continue

        membership_rows_full = torch.cat(
            [membership_chunks[window_index][key] for key in sorted(membership_chunks[window_index])], dim=0,
        )
        rho_m_summary = compute_membership_partition_residual(membership_rows_full)
        family_summaries["rho_M"] = summarize_residual_tensor(rho_m_summary.signed)

        unary_for_query = s0_full[:, list(query_classes)].to(contracts.masked_forward.propagation_dtype)

        seed_by_component_parts = []
        propagated_by_component_parts = []
        chunk_start = 0
        while chunk_start < components.component_count:
            chunk_stop = min(chunk_start + component_chunk_size, components.component_count)
            membership_slice = membership_rows_full[chunk_start:chunk_stop]
            seed = build_masked_seed_chunk(
                unary_for_query, membership_slice, list(range(chunk_start, chunk_stop)), list(query_classes),
                contract=contracts.masked_forward, window_index=window_index,
            )
            propagated = propagate_masked_seed_chunk(graph, seed, contract=contracts.masked_forward)
            # `MaskedSeedChunk.unflatten` returns `[node_count,
            # component_count_in_chunk, query_class_count]` (N first);
            # the residual functions below want `[component_count,
            # node_count, query_class_count]` (R first) -- transpose,
            # never reinterpret the flattening convention itself.
            seed_by_component_parts.append(seed.unflatten(seed.seed_flat).transpose(0, 1))
            propagated_by_component_parts.append(seed.unflatten(propagated.t_final).transpose(0, 1))
            chunk_start = chunk_stop

        seed_by_component = torch.cat(seed_by_component_parts, dim=0)  # [R_total, N, Q]
        propagated_by_component = torch.cat(propagated_by_component_parts, dim=0)  # [R_total, N, Q]

        canonical_t0_full = window_result.trace.snapshots[0].to(contracts.masked_forward.propagation_dtype)
        canonical_tfinal_full = window_result.trace.snapshots[contracts.recurrence.steps].to(contracts.masked_forward.propagation_dtype)
        rho_0_summary = compute_seed_reconstruction_residual(seed_by_component, canonical_t0_full[:, list(query_classes)])
        family_summaries["rho_0"] = summarize_residual_tensor(rho_0_summary.signed)
        rho_320_summary = compute_propagated_reconstruction_residual(propagated_by_component, canonical_tfinal_full[:, list(query_classes)])
        family_summaries["rho_320"] = summarize_residual_tensor(rho_320_summary.signed)

        # --- rho_g: window-restricted masked numerator + full-image denominator. ---
        node_count = seed_by_component.shape[1]
        masked_chunk_for_window = PropagatedChunk(
            # Re-flatten to `[N, R*Q]`, component-major-then-query-class-
            # major (N first) -- exactly `MaskedSeedChunk`'s own
            # flattening convention, so `stitch_and_rescale_component_
            # scores`'s internal `_embed_into_query_union` sees the same
            # shape it always does from a real `propagate_masked_seed_
            # chunk` result.
            t0=seed_by_component.transpose(0, 1).reshape(node_count, -1),
            t_final=propagated_by_component.transpose(0, 1).reshape(node_count, -1),
            window_index=window_index, component_id_start=0, component_id_stop=components.component_count,
            query_class_indices=query_classes, steps_completed=contracts.recurrence.steps,
        )
        s0_masked, s_final_masked = stitch_and_rescale_component_scores(
            {window_index: masked_chunk_for_window}, window_plan=pullback_plan, final_contract=contracts.final_resolution,
            img_shape=img_shape, ori_shape=ori_shape, component_count=components.component_count, query_class_union=query_classes,
        )

        # --- delta_m^(w): window-restricted CANONICAL numerator + full-image denominator. ---
        canonical_chunk_for_window = PropagatedChunk(
            t0=canonical_t0_full[:, list(query_classes)], t_final=canonical_tfinal_full[:, list(query_classes)],
            window_index=window_index, component_id_start=0, component_id_stop=1,
            query_class_indices=query_classes, steps_completed=contracts.recurrence.steps,
        )
        m0_w, m320_w = stitch_and_rescale_component_scores(
            {window_index: canonical_chunk_for_window}, window_plan=pullback_plan, final_contract=contracts.final_resolution,
            img_shape=img_shape, ori_shape=ori_shape, component_count=1, query_class_union=query_classes,
        )

        support_mask = _window_final_support_mask(
            window_index, pullback_plan=pullback_plan, final_resolution_contract=contracts.final_resolution,
            img_shape=img_shape, ori_shape=ori_shape,
        )
        window_dispute_mask = support_mask & dispute_mask_t
        disputed_rows_t, disputed_cols_t = torch.nonzero(window_dispute_mask, as_tuple=True)
        order = torch.argsort(disputed_rows_t * ori_width + disputed_cols_t)
        disputed_rows_t, disputed_cols_t = disputed_rows_t[order], disputed_cols_t[order]

        if disputed_rows_t.numel() == 0:
            family_summaries["rho_g"] = summarize_residual_tensor(torch.zeros(0, dtype=torch.float64))
        else:
            a_vals = a_labels[disputed_rows_t.numpy(), disputed_cols_t.numpy()]
            b_vals = b_labels[disputed_rows_t.numpy(), disputed_cols_t.numpy()]
            query_index_of = {cls: idx for idx, cls in enumerate(query_classes)}
            if not all(int(v) in query_index_of for v in a_vals) or not all(int(v) in query_index_of for v in b_vals):
                raise DfpPhase1bCalibrationRunError(
                    f"image {canonical_id} window {window_index}: a required a(x)/b(x) class is absent from Q_w"
                )
            a_local = torch.tensor([query_index_of[int(v)] for v in a_vals], dtype=torch.int64)
            b_local = torch.tensor([query_index_of[int(v)] for v in b_vals], dtype=torch.int64)
            # `compute_signed_contributions` requires GLOBAL class ids in
            # `disputed_pixel_a`/`disputed_pixel_b` (it performs its own
            # global->local conversion internally against its own
            # `query_class_union` argument) -- `a_local`/`b_local` above
            # are already-localized indices, needed only for indexing
            # `m0_w`/`m320_w` directly below (those tensors are already
            # channel-ordered by `query_classes`' local order). Passing
            # the localized indices to `compute_signed_contributions`
            # would silently reinterpret a local index as if it were a
            # global class id whenever `query_classes` is not the
            # contiguous range `0..len(query_classes)-1`.
            a_global = torch.as_tensor(a_vals, dtype=torch.int64)
            b_global = torch.as_tensor(b_vals, dtype=torch.int64)

            g_chunk = compute_signed_contributions(
                s0_masked, s_final_masked, component_count=components.component_count, query_class_union=query_classes,
                disputed_pixel_rows=disputed_rows_t, disputed_pixel_cols=disputed_cols_t,
                disputed_pixel_a=a_global, disputed_pixel_b=b_global, ori_width=ori_width, component_id_start=0,
            )
            m0_at_a = m0_w[0, a_local, disputed_rows_t, disputed_cols_t]
            m0_at_b = m0_w[0, b_local, disputed_rows_t, disputed_cols_t]
            m320_at_a = m320_w[0, a_local, disputed_rows_t, disputed_cols_t]
            m320_at_b = m320_w[0, b_local, disputed_rows_t, disputed_cols_t]
            m0_x = (m0_at_b - m0_at_a).to(contracts.masked_forward.accumulation_dtype)
            m320_x = (m320_at_b - m320_at_a).to(contracts.masked_forward.accumulation_dtype)
            delta_m_x = m320_x - m0_x

            g_all_components = g_chunk.values.to(contracts.masked_forward.accumulation_dtype)
            rho_g_summary = compute_signed_margin_conservation_residual(g_all_components, delta_m_x)
            family_summaries["rho_g"] = summarize_residual_tensor(rho_g_summary.signed)

        results[window_index] = family_summaries

    return results


# ---------------------------------------------------------------------
# Device / software / precision records, and the atomic artifact
# install flow (Section 13). No scientific computation lives here --
# only metadata collection, pure assembly, and filesystem mechanics.
# ---------------------------------------------------------------------


def _current_repository_commit(repo_root: Path) -> str:
    """The exact HEAD commit id, via ``git rev-parse HEAD`` -- never
    read from a cached/stale value, matching the same ``subprocess``-
    based pattern ``src.dfp_phase1a_identity._check_git_ancestry``
    already uses for git operations in this codebase."""
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        raise DfpPhase1bCalibrationRunError(
            f"cannot determine repository_commit via 'git rev-parse HEAD': {result.stderr.strip()}"
        )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise DfpPhase1bCalibrationRunError(f"'git rev-parse HEAD' did not return a 40-hex commit id: {commit!r}")
    return commit


class DeviceInfoProvider(Protocol):
    def __call__(self) -> Mapping[str, Any]: ...


class SoftwareInfoProvider(Protocol):
    def __call__(self) -> Mapping[str, Any]: ...


def real_device_record() -> dict[str, Any]:
    """Query the single real active CUDA device. Requires ``torch.cuda.
    is_available()`` -- this function never fabricates device metadata
    or provides a CPU-shaped fallback, since the E16 artifact schema
    requires ``device_type == 'cuda'`` unconditionally for a genuine
    completed calibration artifact."""
    import torch

    if not torch.cuda.is_available():
        raise DfpPhase1bCalibrationRunError(
            "real_device_record requires an available CUDA device -- this runner never fabricates "
            "device metadata for a genuine completed calibration artifact"
        )
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)
    driver_version = getattr(torch.cuda, "driver_version", None)
    driver_version = driver_version() if callable(driver_version) else torch._C._cuda_getDriverVersion()
    return {
        "device_type": "cuda",
        "gpu_model_name": torch.cuda.get_device_name(device_index),
        "cuda_compute_capability": f"{major}.{minor}",
        "total_device_memory_bytes": int(properties.total_memory),
        "cuda_driver_version": str(driver_version),
        "cuda_runtime_version": str(torch.version.cuda),
    }


def real_software_record(repo_root: Path) -> dict[str, Any]:
    """Every field read directly from the live interpreter/torch/numpy
    state at call time -- never a hardcoded or cached value."""
    import platform

    import numpy
    import torch

    return {
        "repository_commit": _current_repository_commit(repo_root),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(numpy.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
    }


def build_propagation_precision_record(*, recurrence_contract: Any, component_chunk_size: int) -> dict[str, Any]:
    """Every dtype/``k`` field is an already-frozen constant (matching
    ``src.dfp_phase1b_masked_forward.ACCUMULATION_DTYPE``/
    ``PROPAGATION_DTYPE`` and the frozen k=12 graph); ``alpha``/
    ``steps`` come from the SAME committed recurrence contract every
    canonical/masked propagation call already uses -- never a second,
    independently hardcoded copy. ``component_chunk_size`` is the
    caller's own explicit, required runtime value: E16 authorizes
    recording it but does not freeze it to one number, so this function
    accepts no silent default."""
    if type(component_chunk_size) is not int or component_chunk_size <= 0:
        raise DfpPhase1bCalibrationRunError("component_chunk_size must be a positive exact integer")
    return {
        "graph_storage_dtype": "float32",
        "canonical_recurrence_dtype": "float64",
        "masked_recurrence_dtype": "float64",
        "residual_accumulation_dtype": "float64",
        "alpha": float(recurrence_contract.alpha),
        "steps": int(recurrence_contract.steps),
        "k": 12,
        "chunk_size": component_chunk_size,
    }


def assemble_calibration_artifact(
    *, preflight: PreflightReport, selection: SelectionOutcome, selected_window_rows: Sequence[Mapping[str, Any]],
    device_record: Mapping[str, Any], software_record: Mapping[str, Any],
    propagation_precision_record: Mapping[str, Any], complete: bool, final: bool,
) -> dict[str, Any]:
    """Pure assembly: no I/O, no torch, no residual computation of its
    own. ``selected_window_rows`` must already be exactly
    ``selection.selected_rows``, each merged with its own ``completed``
    flag and four residual-family summary dicts (from
    ``summarize_residual_tensor``) -- the only computation performed
    here is the one additive aggregate reconstruction
    (``aggregate_residual_summaries``, the same committed function
    ``validate_calibration_artifact`` uses to independently re-check
    this artifact later), never a second, parallel aggregation
    implementation."""
    e16_identity = preflight.e16_identity
    # `sort_candidate_universe` cannot be reused here: it enforces the
    # closed, candidate-only field set (Phase B's raw universe rows),
    # while a selected-window row also carries `canonical_candidate_
    # index`/`selection_key`/`selection_rank`/`completed`/the four
    # residual families -- `canonical_candidate_order_key` alone is the
    # schema-agnostic part of that sort, reused directly.
    rows = sorted(selected_window_rows, key=canonical_candidate_order_key)
    completed_rows = [row for row in rows if row["completed"]]
    aggregates = {
        family: aggregate_residual_summaries([row[family] for row in completed_rows], label=f"aggregate.{family}")
        for family in RESIDUAL_FAMILIES
    }
    calibration_identity_path = preflight.repo_root / "evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml"
    calibration_identity_sha256 = hashlib.sha256(calibration_identity_path.read_bytes()).hexdigest()
    bindings = e16_identity["dependency_bindings"]
    return {
        "schema_name": ARTIFACT_SCHEMA_NAME,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "complete": complete,
        "final": final,
        "calibration_identity_sha256": calibration_identity_sha256,
        "bound_e15_protocol_sha256": bindings["e15_protocol"]["sha256"],
        "bound_e15_artifacts_sha256": bindings["e15_artifacts"]["sha256"],
        "bound_masked_forward_sha256": bindings["masked_forward_core"]["sha256"],
        "repository_commit": software_record["repository_commit"],
        "run_mode": "coco_stuff_500",
        "bound_subset_manifest_sha256": bindings["subset_manifest"]["sha256"],
        "selected_ids_digest": preflight.subset_manifest["selected_ids_digest"],
        "candidate_window_count": selection.candidate_window_count,
        "expected_selected_window_count": WINDOW_COUNT,
        "completed_selected_window_count": len(completed_rows),
        "candidate_universe_digest": selection.candidate_universe_digest,
        "selected_windows_digest": selection.selected_windows_digest,
        "selection_trace_digest": selection.selection_trace_digest,
        "device_record": dict(device_record),
        "software_record": dict(software_record),
        "propagation_precision_record": dict(propagation_precision_record),
        "selected_window_records": rows,
        "aggregate_residual_summaries": aggregates,
    }


def install_calibration_artifact(document: Mapping[str, Any], output_path: Path, *, allow_overwrite: bool = False) -> None:
    """Section 13's exact install flow: validate in memory -> write to
    a temp sibling -> fsync -> read back -> independently re-validate
    via the committed validator -> atomic rename -> reconcile an
    ambiguous post-replace failure before deleting anything. Never
    installs a partial artifact: any failure before the final
    ``os.replace`` leaves the destination path completely untouched,
    and the temp file is always removed on any failure path (no orphan
    temp files)."""
    output_path = Path(output_path)
    # Re-checked here via the SAME shared `_validate_output_path` helper
    # Phase A's own `run_preflight` uses (never a second, independently
    # duplicated set of traversal/existence/overwrite rules) -- this
    # function never assumes preflight was previously called, and this
    # closes the time-of-check-to-time-of-use window between preflight
    # and the end of a long GPU run.
    _validate_output_path(output_path, allow_overwrite=allow_overwrite)

    validate_calibration_artifact(document)  # first validation, pre-write, in memory
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"

    fd, temp_name = tempfile.mkstemp(prefix=".phase1b_conservation_calibration.", suffix=".json.tmp", dir=str(output_path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        read_back_raw = temp_path.read_text(encoding="utf-8")
        try:
            read_back_document = json.loads(read_back_raw)
        except json.JSONDecodeError as exc:
            raise DfpPhase1bCalibrationRunError(f"internal error: the just-written artifact failed to parse back as JSON: {exc}") from exc
        validate_calibration_artifact(read_back_document)  # second, independent validation, post-write, from disk

        try:
            os.replace(temp_path, output_path)
        except OSError as exc:
            # Ambiguous: the OS operation may or may not have actually
            # completed. Reconcile by inspecting the destination before
            # concluding anything failed, and before deleting the temp
            # file (the `finally` block below only removes it if it is
            # still present).
            if output_path.exists() and output_path.read_text(encoding="utf-8") == serialized:
                return
            raise DfpPhase1bCalibrationRunError(f"atomic install failed and could not be unambiguously reconciled: {exc}") from exc
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


__all__ = [
    "ComponentContractsBundle",
    "DeviceInfoProvider",
    "DfpPhase1bCalibrationRunError",
    "GeometryProvider",
    "ImageGeometry",
    "ImagePreparer",
    "InferenceProvider",
    "PreflightReport",
    "PreparedImage",
    "SelectionOutcome",
    "SoftwareInfoProvider",
    "WindowExecutionResult",
    "WindowPlanBuilder",
    "assemble_calibration_artifact",
    "build_candidate_universe",
    "build_component_contracts",
    "build_propagation_precision_record",
    "default_window_plan_builder",
    "install_calibration_artifact",
    "load_subset_manifest",
    "process_selected_image",
    "real_device_record",
    "real_software_record",
    "run_preflight",
    "select_and_verify",
    "summarize_residual_tensor",
]
