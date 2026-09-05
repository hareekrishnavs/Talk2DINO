"""DFP Phase 1B: production provider wiring for the E16-authorized
200-window conservation-calibration runner's ``execute`` path.

This module invents no new dataset, model, or preprocessing logic. It
is a narrow adapter that constructs the three Protocol-typed
dependencies ``src.dfp_phase1b_calibration_run`` already declares
(``GeometryProvider``, ``ImagePreparer``, ``InferenceProvider``) by
calling, unmodified, the exact production entry points DFP Phase 1A's
own execution driver (``src/dfp_phase1a_execute.py::execute_run``)
already uses for the identical purpose:

* ``diagnostics.run_k11_k12_stability._build_dataset_only`` -- real
  canonical dataset construction (image/annotation config resolution,
  no model, no checkpoint, no CUDA);
* ``diagnostics.run_k11_k12_stability._build_inference`` -- real model
  construction, weight loading, and ``DINOTextSegInference`` assembly
  (the exact same call ``execute_run`` makes);
* ``diagnostics.run_k11_k12_stability._extract_prepared_image`` -- the
  canonical mmseg test-pipeline run for one image, producing the exact
  ``(image_tensor, img_metas)`` pair ``_process_one_image`` itself
  consumes;
* ``src.dfp_phase1a_dataset_view.build_manifest_execution_view`` /
  ``verify_position_identity`` -- the frozen-manifest-order ->
  native-dataset-index mapping that already closed the exact
  500-vs-5000 positional-index defect this module must not
  reintroduce (SLURM jobs 21055181/21060527, documented in
  ``src/dfp_phase1a_execute.py``'s own module docstring).

Every scientific computation (window plan, graph, recurrence,
stitching, components, memberships, masked-forward, residuals) still
happens exclusively inside ``src.dfp_phase1b_calibration_run.
process_selected_image`` and the modules it calls -- this module never
duplicates that logic, only feeds it.

**Deferred-import boundary.** Importing this module does not import
torch, mmcv, or any model/dataset module -- every heavy import is
local to a function body here, exactly matching
``src.dfp_phase1b_calibration_run``'s own established convention. The
CLI must only import this module AFTER preflight has already
succeeded (see ``diagnostics/run_dfp_phase1b_calibration.py::
_cmd_execute``).

**GT isolation.** Nothing in this module reads
``dataset.get_gt_seg_map_by_idx`` or any GT-shaped field.
``_extract_prepared_image`` returns only ``image_tensor``/``img_metas``
-- the same object ``_process_one_image`` uses for every GT-free stage
of its own pipeline, before its own, much later, explicit GT read.

**Weight resolution.** Real model construction resolves the DINOv2
backbone and CLIP weights via ``src.local_weights.DEFAULT_WEIGHT_DIR``,
which is exactly ``os.environ.get("TALK2DINO_WEIGHT_DIR", "weights")``
(read once, at that module's own import time). Unlike the E13-bound
bridge checkpoint (``weights/vitb_mlp_infonce_paired_soft_routing_
tau010.pth``, hash-pinned in ``evaluation_identities/
e13_dfp_phase1a_artifacts.toml``), the committed e3 identity
(``evaluation_identities/e3_paired_soft_routing.toml``) names the
backbone/CLIP checkpoint FILES ONLY BY FILENAME
(``backbone_checkpoint_name``/``clip_checkpoint_name``) -- no SHA-256
binding exists anywhere in the committed authorities for those two
files. ``verify_backbone_weights_present`` therefore only verifies
existence (never inventing a digest binding no committed authority
defines); see ``docs/dfp_phase1b_calibration_run.md`` for the full
account and the ``TALK2DINO_WEIGHT_DIR`` requirement this adds to the
launcher.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.dfp_phase1b_calibration_artifact import RESIDUAL_FAMILIES, WINDOW_COUNT
from src.dfp_phase1b_calibration_run import (
    ComponentContractsBundle,
    DfpPhase1bCalibrationRunError,
    ImageGeometry,
    PreflightReport,
    PreparedImage,
    SelectionOutcome,
    assemble_calibration_artifact,
    build_candidate_universe,
    build_component_contracts,
    build_propagation_precision_record,
    default_window_plan_builder,
    install_calibration_artifact,
    process_selected_image,
    real_device_record,
    real_software_record,
    run_preflight,
    select_and_verify,
    summarize_residual_tensor,
)


class DfpPhase1bCalibrationProductionError(DfpPhase1bCalibrationRunError):
    """Raised on any production-adapter (dataset/model construction,
    manifest reconciliation, weight resolution) invariant violation --
    always this module's own domain exception (itself a
    ``DfpPhase1bCalibrationRunError``, so the CLI's existing dispatch
    catches it unchanged), always chained via ``from exc`` so
    ``__cause__`` is preserved. Never raised for, and never catches,
    ``KeyboardInterrupt``, ``SystemExit``, or ``MemoryError``."""


# ---------------------------------------------------------------------
# Phase B resources: dataset-only construction (no model, no
# checkpoint, no CUDA), plus the frozen-manifest execution view and the
# identities/contracts every later stage needs. Building this does NOT
# require a GPU and does not load any weight.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionCalibrationResources:
    """Everything built once per ``execute`` invocation before any
    model weight is loaded. ``dataset``/``execution_view`` are built
    here (Phase B); ``inference`` is filled in later, once, only after
    candidate selection has already completed (see
    ``build_production_inference``) -- never influencing selection."""

    repo_root: Path
    protocol_identity: Mapping[str, Any]
    phase1b_identity: Mapping[str, Any]
    e3_identity: Mapping[str, Any]
    contracts: ComponentContractsBundle
    class_count: int
    affinity_power: float
    patch_size: int
    dataset: Any
    execution_view: Any  # src.dfp_phase1a_dataset_view.ManifestExecutionView


def _load_production_identities(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the E13 protocol, E15 protocol, and E3 evaluator identities
    -- pure TOML/hash loading, no torch, exactly the same loaders every
    other committed module in this repository already uses."""
    from src.dfp_phase1a_identity import load_identity as load_e13_identity
    from src.dfp_phase1b_identity import load_identity as load_e15_identity
    from src.e3_evaluation_identity import load_identity as load_e3_identity

    protocol_identity = load_e13_identity(repo_root=repo_root)
    phase1b_identity = load_e15_identity(repo_root=repo_root)
    parents = protocol_identity["parent_identities"]
    e3_identity = load_e3_identity(repo_root / parents["e3_identity_path"], repo_root=repo_root)
    return protocol_identity, phase1b_identity, e3_identity


def _load_patch_size(repo_root: Path, protocol_identity: Mapping[str, Any]) -> int:
    """The ViT patch size is a fixed model-architecture constant (never
    per-image), frozen in the matched-geometry identity
    (``evaluation_identities/e12_matched_k11_k12_t320.toml``'s own
    ``[geometry].patch_size``) that ``dfp_stitching_contract_from_
    protocol`` already relationally cross-checks -- read once here via
    the SAME committed loader, never a second hardcoded copy."""
    from src.matched_k11_k12_identity import load_identity as load_matched_geometry_identity

    parents = protocol_identity["parent_identities"]
    matched_identity = load_matched_geometry_identity(repo_root / parents["matched_k11_k12_identity_path"], repo_root=repo_root)
    patch = matched_identity["geometry"]["patch_size"]
    if patch[0] != patch[1]:
        raise DfpPhase1bCalibrationProductionError(f"matched-geometry patch_size must be square, observed {patch}")
    return int(patch[0])


def _load_class_count(repo_root: Path) -> int:
    """The frozen COCO-Stuff class count, from the SAME E13 artifact-
    schema identity Phase 1A's own preflight (``src.dfp_phase1a_
    preflight``) already reads it from -- never re-derived or
    hardcoded here."""
    from src.dfp_phase1a_artifact_schema import load_identity as load_artifact_identity

    artifact_identity = load_artifact_identity(repo_root=repo_root)
    return int(artifact_identity["dataset_binding"]["coco_stuff_class_count"])


def build_production_dataset_resources(
    *, repo_root: Path, selected_ids: Sequence[str],
) -> ProductionCalibrationResources:
    """Phase B prerequisite: construct the real canonical dataset
    (image/annotation config resolution only -- no model, no
    checkpoint, no CUDA) via the unmodified ``_build_dataset_only``,
    and the frozen-manifest -> native-dataset-index execution view via
    the unmodified ``build_manifest_execution_view`` -- the same
    adapter that already resolved the 500-vs-5000 native-ordering
    problem for DFP Phase 1A's own production driver. Fails closed
    (via this module's own domain exception) on any missing, duplicate,
    or unresolvable manifest ID -- before any image is decoded."""
    try:
        from diagnostics.run_k11_k12_stability import _build_dataset_only
        from src.dfp_phase1a_dataset_view import DfpPhase1aDatasetViewError, build_manifest_execution_view
        from src.k11_k12_stability_gate_identity import K11K12StabilityGateError
    except ImportError as exc:  # pragma: no cover -- environment-dependent heavy-import gap
        raise DfpPhase1bCalibrationProductionError(f"cannot import production dataset infrastructure: {exc}") from exc

    protocol_identity, phase1b_identity, e3_identity = _load_production_identities(repo_root)
    patch_size = _load_patch_size(repo_root, protocol_identity)
    class_count = _load_class_count(repo_root)
    affinity_power = float(protocol_identity["graph"]["affinity_power"])
    contracts = build_component_contracts(protocol_identity, phase1b_identity, repo_root=repo_root)

    try:
        _cfg, dataset, _dataset_config_path = _build_dataset_only(repo_root, e3_identity)
    except K11K12StabilityGateError as exc:
        raise DfpPhase1bCalibrationProductionError(f"real dataset construction failed: {exc}") from exc

    try:
        execution_view = build_manifest_execution_view(dataset, list(selected_ids))
    except DfpPhase1aDatasetViewError as exc:
        raise DfpPhase1bCalibrationProductionError(
            f"frozen manifest -> native dataset index reconciliation failed: {exc}"
        ) from exc
    if len(execution_view) != len(selected_ids):
        raise DfpPhase1bCalibrationProductionError(
            f"execution view has {len(execution_view)} positions, expected exactly {len(selected_ids)}"
        )

    return ProductionCalibrationResources(
        repo_root=repo_root, protocol_identity=protocol_identity, phase1b_identity=phase1b_identity,
        e3_identity=e3_identity, contracts=contracts, class_count=class_count, affinity_power=affinity_power,
        patch_size=patch_size, dataset=dataset, execution_view=execution_view,
    )


# ---------------------------------------------------------------------
# GeometryProvider: real inference-canvas geometry, discarding each
# image tensor immediately after extracting its height/width. No
# model, no score, no GT.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionGeometryProvider:
    """Real ``GeometryProvider``: for each frozen manifest image, runs
    the canonical mmseg test pipeline once (via the unmodified
    ``_extract_prepared_image``) to obtain its real, possibly-per-image-
    varying inference-canvas height/width, then discards the decoded
    tensor immediately -- never retaining more than one image's tensor
    at a time, never caching all 500. ``patch_size`` is the frozen
    model constant (identical for every image), never derived from a
    model forward pass."""

    resources: ProductionCalibrationResources
    _positions_by_id: Mapping[str, Any]

    def __call__(self, canonical_id: str) -> ImageGeometry:
        from diagnostics.run_k11_k12_stability import _extract_prepared_image
        from src.dfp_phase1a_dataset_view import DfpPhase1aDatasetViewError, verify_position_identity
        from src.k11_k12_stability_gate_identity import K11K12StabilityGateError

        position = self._positions_by_id.get(canonical_id)
        if position is None:
            raise DfpPhase1bCalibrationProductionError(
                f"canonical id {canonical_id!r} has no resolved native-dataset position (not in the manifest execution view)"
            )
        try:
            verify_position_identity(
                self.resources.dataset, manifest_position=position.manifest_position,
                canonical_id=position.canonical_id, native_index=position.native_index,
            )
            prepared = _extract_prepared_image(
                self.resources.dataset, position.native_index, canonical_image_id=position.relative_path,
            )
        except DfpPhase1aDatasetViewError as exc:
            raise DfpPhase1bCalibrationProductionError(f"geometry extraction for {canonical_id!r}: {exc}") from exc
        except K11K12StabilityGateError as exc:
            raise DfpPhase1bCalibrationProductionError(f"geometry extraction for {canonical_id!r}: {exc}") from exc

        inference_height, inference_width = prepared.inference_height, prepared.inference_width
        del prepared  # discard the decoded tensor immediately -- geometry only, never cached
        return ImageGeometry(inference_height=inference_height, inference_width=inference_width, patch_size=self.resources.patch_size)


def build_production_geometry_provider(resources: ProductionCalibrationResources) -> ProductionGeometryProvider:
    positions_by_id = {p.canonical_id: p for p in resources.execution_view.positions}
    return ProductionGeometryProvider(resources=resources, _positions_by_id=positions_by_id)


# ---------------------------------------------------------------------
# ImagePreparer: canonical selected-image preparation, reused verbatim
# from the same pipeline entry point GeometryProvider uses -- the
# scientific execution stage's own image tensor, never a second
# resize/crop implementation.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionImagePreparer:
    """Real ``ImagePreparer``: prepares exactly one selected image via
    the canonical Phase 1A path (``_extract_prepared_image``), using
    its already-resolved native dataset index and canonical relative
    path (never a bare manifest ID). Returns only the image tensor and
    the metadata ``process_selected_image`` needs -- never GT, never a
    mutated dataset record."""

    resources: ProductionCalibrationResources
    _positions_by_id: Mapping[str, Any]

    def __call__(self, canonical_id: str) -> PreparedImage:
        from diagnostics.run_k11_k12_stability import _extract_prepared_image
        from src.dfp_phase1a_dataset_view import DfpPhase1aDatasetViewError, verify_position_identity
        from src.k11_k12_stability_gate_identity import K11K12StabilityGateError

        position = self._positions_by_id.get(canonical_id)
        if position is None:
            raise DfpPhase1bCalibrationProductionError(
                f"canonical id {canonical_id!r} has no resolved native-dataset position (not in the manifest execution view)"
            )
        try:
            verify_position_identity(
                self.resources.dataset, manifest_position=position.manifest_position,
                canonical_id=position.canonical_id, native_index=position.native_index,
            )
            prepared = _extract_prepared_image(
                self.resources.dataset, position.native_index, canonical_image_id=position.relative_path,
            )
        except DfpPhase1aDatasetViewError as exc:
            raise DfpPhase1bCalibrationProductionError(f"image preparation for {canonical_id!r}: {exc}") from exc
        except K11K12StabilityGateError as exc:
            raise DfpPhase1bCalibrationProductionError(f"image preparation for {canonical_id!r}: {exc}") from exc

        ori_shape_raw = prepared.img_metas.get("ori_shape")
        if ori_shape_raw is None or len(ori_shape_raw) < 2:
            raise DfpPhase1bCalibrationProductionError(f"image {canonical_id!r}: img_metas is missing a valid 'ori_shape'")
        ori_shape = (int(ori_shape_raw[0]), int(ori_shape_raw[1]))
        if ori_shape[0] <= 0 or ori_shape[1] <= 0:
            raise DfpPhase1bCalibrationProductionError(f"image {canonical_id!r}: malformed ori_shape {ori_shape}")

        return PreparedImage(
            image_tensor=prepared.image_tensor.unsqueeze(0), inference_height=prepared.inference_height,
            inference_width=prepared.inference_width, ori_shape=ori_shape,
        )


def build_production_image_preparer(resources: ProductionCalibrationResources) -> ProductionImagePreparer:
    positions_by_id = {p.canonical_id: p for p in resources.execution_view.positions}
    return ProductionImagePreparer(resources=resources, _positions_by_id=positions_by_id)


# ---------------------------------------------------------------------
# InferenceProvider: the real model, built once, lazily (after
# candidate selection).
# ---------------------------------------------------------------------


def verify_backbone_weights_present(e3_identity: Mapping[str, Any]) -> tuple[Path, Path]:
    """Fail closed, before any heavy model-construction import, if the
    DINO backbone or CLIP weight files named by the committed E3
    identity are absent from ``TALK2DINO_WEIGHT_DIR`` (defaulting to
    ``weights`` -- ``src.local_weights.DEFAULT_WEIGHT_DIR``'s own exact
    fallback, read from the SAME module, never a second, independently
    hardcoded copy). Existence-only: no committed authority pins a
    SHA-256 for either file (unlike the E13-bound bridge checkpoint),
    so this function never invents one -- see this module's own
    docstring."""
    from src.local_weights import DEFAULT_WEIGHT_DIR

    model = e3_identity["model"]
    backbone_name = model["backbone_checkpoint_name"]
    clip_name = model["clip_checkpoint_name"]
    weight_dir = Path(DEFAULT_WEIGHT_DIR)
    backbone_path = weight_dir / backbone_name
    clip_path = weight_dir / clip_name
    missing = [str(p) for p in (backbone_path, clip_path) if not p.is_file()]
    if missing:
        weight_dir_env = os.environ.get("TALK2DINO_WEIGHT_DIR")
        weight_dir_display = repr(weight_dir_env) if weight_dir_env is not None else "<unset, defaulting to 'weights'>"
        raise DfpPhase1bCalibrationProductionError(
            "required backbone/CLIP weight file(s) not found: " + ", ".join(missing) +
            f" (TALK2DINO_WEIGHT_DIR={weight_dir_display}; no SHA-256 binding exists for these files in the "
            "committed E3 identity, so only existence is verified here)"
        )
    return backbone_path, clip_path


def build_production_inference(resources: ProductionCalibrationResources, *, device: str, log_dir: Path) -> Any:
    """Build the real model + inference object via the unmodified
    ``_build_inference`` -- the exact same call DFP Phase 1A's own
    ``execute_run`` makes. Loads real weights, moves the model to
    ``device`` when ``device == "cuda"``, sets eval mode, and disables
    gradients (all internal to ``_build_inference`` itself, never
    reimplemented here). Called at most once per ``execute``
    invocation, always AFTER candidate selection has already completed
    -- its return value never participates in candidate selection.
    The returned object already satisfies ``src.dfp_phase1b_
    calibration_run.InferenceProvider`` structurally (it exposes
    ``.model``/``.text_embedding``, exactly as ``_process_one_window_
    dfp`` already relies on) -- no wrapper class is introduced."""
    verify_backbone_weights_present(resources.e3_identity)
    try:
        from diagnostics.run_k11_k12_stability import _build_inference
        from src.k11_k12_stability_gate_identity import K11K12StabilityGateError
    except ImportError as exc:  # pragma: no cover
        raise DfpPhase1bCalibrationProductionError(f"cannot import production inference infrastructure: {exc}") from exc

    try:
        inference, _dataset = _build_inference(resources.repo_root, resources.e3_identity, device, log_dir=log_dir)
    except K11K12StabilityGateError as exc:
        raise DfpPhase1bCalibrationProductionError(f"real model/inference construction failed: {exc}") from exc
    except OSError as exc:
        raise DfpPhase1bCalibrationProductionError(f"real model/inference construction failed: {exc}") from exc
    return inference


# ---------------------------------------------------------------------
# Top-level orchestration: candidate universe (prediction-free) ->
# exact 200-window selection -> (only then) model construction ->
# selected-image scientific execution -> artifact assembly/install.
# ---------------------------------------------------------------------


def _group_selected_rows_by_image(selected_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected_rows:
        grouped.setdefault(row["image_id"], []).append(row)
    return grouped


def execute_production_calibration(
    *, output_path: Path, device: str, component_chunk_size: int, repo_root: Path | None = None,
    allow_overwrite: bool = False, log_dir: Path | None = None,
) -> Mapping[str, Any]:
    """The full production ``execute`` orchestration:

    1. run the committed ``run_preflight`` (Phase A) -- identical to
       the ``preflight`` subcommand;
    2. build the real dataset + manifest execution view (Phase B
       prerequisite, no model);
    3. build the candidate universe from real geometry and select
       exactly 200 windows (prediction-free; no model output has been
       computed yet);
    4. re-validate output freshness immediately before heavy execution
       (closes the time-of-check-to-time-of-use window between step 1
       and here);
    5. build the real model/inference (only now, after selection);
    6. for every image containing at least one selected window, prepare
       it once and run ``process_selected_image`` (same graph instance,
       k=12, T=320, FP64, exact masked-forward core, exact residuals --
       all inside that already-committed, already-tested function);
    7. assemble and atomically install the final E16 artifact, re-
       validating output freshness immediately before installation;
    8. read the installed artifact back and independently re-validate
       it via the committed validator.

    Raises this module's own domain exception (chained) on any
    operational failure. Never catches ``KeyboardInterrupt``,
    ``SystemExit``, or ``MemoryError``."""
    if device != "cuda":
        raise DfpPhase1bCalibrationProductionError(
            f"execute_production_calibration requires device='cuda' (observed {device!r}) -- the committed "
            "E16 authority requires device_type='cuda' for a genuine completed calibration artifact"
        )
    if type(component_chunk_size) is not int or component_chunk_size <= 0:
        raise DfpPhase1bCalibrationProductionError("component_chunk_size must be a positive exact integer")

    preflight: PreflightReport = run_preflight(output_path=output_path, mode="fresh", allow_overwrite=allow_overwrite, repo_root=repo_root)
    root = preflight.repo_root

    resources = build_production_dataset_resources(repo_root=root, selected_ids=preflight.selected_ids)

    window_plan_builder = default_window_plan_builder(root)
    geometry_provider = build_production_geometry_provider(resources)
    universe = build_candidate_universe(
        preflight.selected_ids, geometry_provider=geometry_provider, window_plan_builder=window_plan_builder,
    )
    selection: SelectionOutcome = select_and_verify(universe)

    # Re-check output freshness immediately before entering heavy
    # (model-construction, GPU) execution -- closes the window between
    # Phase A's own check (above) and here; never assumes it is still
    # valid merely because it was valid a moment ago.
    from src.dfp_phase1b_calibration_run import _validate_output_path

    _validate_output_path(preflight.output_path, allow_overwrite=allow_overwrite)

    effective_log_dir = Path(log_dir) if log_dir is not None else preflight.output_path.parent
    inference = build_production_inference(resources, device=device, log_dir=effective_log_dir)
    image_preparer = build_production_image_preparer(resources)

    grouped = _group_selected_rows_by_image(selection.selected_rows)
    selected_window_rows: list[dict[str, Any]] = []
    for image_id, rows in grouped.items():
        prepared = image_preparer(image_id)
        selected_ordinals = sorted(int(row["canonical_window_ordinal"]) for row in rows)
        per_window_summaries = process_selected_image(
            inference, prepared, image_id, selected_windows_for_image=selected_ordinals,
            protocol_identity=resources.protocol_identity, contracts=resources.contracts,
            class_count=resources.class_count, affinity_power=resources.affinity_power,
            component_chunk_size=component_chunk_size,
        )
        del prepared  # release the prepared image's tensors immediately after this image completes
        rows_by_ordinal = {int(row["canonical_window_ordinal"]): row for row in rows}
        for ordinal, family_summaries in per_window_summaries.items():
            row = dict(rows_by_ordinal[ordinal])
            row["completed"] = True
            for family in RESIDUAL_FAMILIES:
                row[family] = family_summaries[family]
            selected_window_rows.append(row)

    if len(selected_window_rows) != WINDOW_COUNT:
        raise DfpPhase1bCalibrationProductionError(
            f"produced {len(selected_window_rows)} completed selected-window rows, expected exactly {WINDOW_COUNT}"
        )

    device_record = real_device_record()
    software_record = real_software_record(root)
    precision_record = build_propagation_precision_record(
        recurrence_contract=resources.contracts.recurrence, component_chunk_size=component_chunk_size,
    )
    document = assemble_calibration_artifact(
        preflight=preflight, selection=selection, selected_window_rows=selected_window_rows,
        device_record=device_record, software_record=software_record, propagation_precision_record=precision_record,
        complete=True, final=True,
    )

    # Re-check output freshness immediately before the final install --
    # closes the window between candidate/selection/execution (which,
    # for a real 200-window run, takes real wall-clock time) and here.
    _validate_output_path(preflight.output_path, allow_overwrite=allow_overwrite)
    install_calibration_artifact(document, preflight.output_path, allow_overwrite=allow_overwrite)

    from src.dfp_phase1b_calibration_artifact import validate_calibration_artifact
    import json

    with open(preflight.output_path, "r", encoding="utf-8") as handle:
        installed_document = json.load(handle)
    validate_calibration_artifact(installed_document, candidate_universe=universe)

    return {
        "output_path": preflight.output_path, "selected_window_count": len(selected_window_rows),
        "complete": installed_document["complete"], "final": installed_document["final"],
    }


__all__ = [
    "DfpPhase1bCalibrationProductionError",
    "ProductionCalibrationResources",
    "ProductionGeometryProvider",
    "ProductionImagePreparer",
    "build_production_dataset_resources",
    "build_production_geometry_provider",
    "build_production_image_preparer",
    "build_production_inference",
    "verify_backbone_weights_present",
    "execute_production_calibration",
]
