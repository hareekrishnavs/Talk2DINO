"""DFP Phase 1B: E23-authorized real production wiring for the
350-image BALP development execution.

Orchestration only. Implements no BALP scientific formula (reuses
``src.dfp_phase1b_lightweight_boundary`` via the committed
``process_development_image``), no checkpoint format, no artifact
validation, no metric formula, no window-construction geometry, no
graph/recurrence code, and no dataset-source/model-binding logic --
every one of those is reused verbatim, unmodified, from its E23-bound
source (see ``evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml``
``[dependency_bindings]``). This module's only job is to build the
real ``ExecutionProviders`` / ``image_provider_fn`` / identity-binding
objects that ``src.dfp_phase1b_lightweight_boundary_execute.
execute_development`` (the committed, E23-bound checkpoint/resume
engine) already accepts as injectable arguments, and to enforce the
350-image/769-window population and isolation boundary E23 adds.

No CUDA/model/dataset access happens at import time. The real
production builders (``build_production_dataset_resources``,
``build_production_inference``) are only ever invoked from inside
``build_real_execution_providers``, and only when a caller actually
requests real execution -- every torch-free preflight/verification
path in this module never calls them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import src.dfp_phase1b_balp_development_execution_identity as e23
import src.dfp_phase1b_lightweight_boundary_identity as e22

# `src.dfp_phase1b_lightweight_boundary_execute` (`execmod`) is NEVER
# imported at module scope here: it imports the BALP scientific kernel
# (`src.dfp_phase1b_lightweight_boundary`), which imports `torch` at
# its own top level. Importing `execmod` eagerly would make merely
# `import`-ing THIS module pull in torch, breaking the torch-free
# preflight path this module also provides (`load_verified_authority`,
# `load_development_window_index`, `verify_population_isolation`,
# `build_identity_bindings_without_digest`). `execmod` is E23-bound
# (`balp_execution_runner`) and must never be modified, so the fix is
# entirely on this module's side: every function that actually needs
# `execmod` imports it locally, exactly like
# `src.dfp_phase1b_calibration_production`'s own established
# "defer all heavy imports to function bodies" convention.


class DfpPhase1bBalpDevelopmentProductionError(ValueError):
    """Raised on any BALP production-wiring invariant violation.
    Always fail closed. Never raised for, and never catches,
    KeyboardInterrupt / SystemExit / MemoryError -- those always
    propagate unmodified."""


DEVELOPMENT_COUNT = 350
WINDOW_COUNT = 769
WINDOW_UNIVERSE_DIGEST = "b4672854e38199deef26c154a30e45c1835c201a4dbcf046aa29e7fdf12df9e6"
RUN_MODE = "coco_stuff_500"
WINDOW_UNIVERSE_RELATIVE_PATH = Path("evaluation_subsets/dfp_phase1b_development_window_universe.json")


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------
# Identity/population loading (torch-free; safe for preflight).
# ---------------------------------------------------------------------


def load_verified_authority(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Loads and closed-schema/relationally verifies E23, E22 (via
    E23's own re-pinned hash), and the frozen 350/150 split. Fails
    closed if E23 or E22 disagree with committed bytes. Torch-free."""
    e23_doc = e23.load_identity(repo_root=repo_root)
    e22_doc = e22.load_identity(repo_root=repo_root)
    e22.load_artifacts_identity(repo_root=repo_root)
    split = e22.load_and_verify_split(repo_root=repo_root)
    return e23_doc, e22_doc, split, e23_doc["dependency_bindings"]


def build_identity_bindings_without_digest(e23_doc: Mapping[str, Any]) -> dict[str, str]:
    """The 7 real, already-verified identity_bindings keys the
    committed checkpoint runner requires (``development_digest`` is
    computed by the artifact layer itself from staged NPZ bytes --
    never supplied by this function). Sourced from E23's OWN already-
    hash-verified ``dependency_bindings`` -- never a second,
    independently re-hashed copy."""
    db = e23_doc["dependency_bindings"]
    return {
        "e22_protocol_sha256": db["e22_preservation"]["sha256"],
        "e22_artifacts_sha256": db["e22_artifacts"]["sha256"],
        "e13_protocol_sha256": db["e13_protocol"]["sha256"],
        "matched_evaluator_sha256": db["matched_evaluator"]["sha256"],
        "subset_manifest_sha256": db["subset_manifest"]["sha256"],
        "balp_split_sha256": db["balp_split_artifact"]["sha256"],
        "run_mode": RUN_MODE,
    }


@dataclass(frozen=True)
class DevelopmentWindowIndex:
    """The committed 350-image/769-window population, exactly as
    persisted by the frozen window-universe manifest -- the SOLE
    source of per-image window selection this module ever uses."""

    canonical_ids: tuple[str, ...]                     # frozen order, length 350
    windows_per_image: Mapping[str, tuple[int, ...]]    # canonical_id -> ordered window indices
    aggregate_window_count: int
    canonical_record_digest: str


def _canonical_json_digest(payload: Any) -> str:
    """Same convention as ``src.dfp_phase1a_dataset_view.
    compute_mapping_digest`` / ``src.dfp_phase1b_development_window_
    universe._canonical_json_digest`` -- deterministic SHA-256 over a
    fixed-key-order JSON encoding. Reimplemented here (rather than
    imported) SOLELY so this torch-free preflight path never imports
    ``src.dfp_phase1b_development_window_universe``, which is itself
    torch-dependent at module scope (it imports
    ``dfp_stitching_contract_from_protocol`` from
    ``src.dfp_phase1a_horizon_stitching``, which imports ``torch`` at
    the top of that file) -- and that module is E23-bound, so it must
    never be edited merely to defer that import. This function
    re-derives nothing about window geometry itself; it only re-hashes
    already-persisted records, identical to the bound module's own
    digest primitive."""
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_development_window_index(repo_root: Path, *, e23_doc: Mapping[str, Any] | None = None) -> DevelopmentWindowIndex:
    """Torch-free, GT-free, model-free. Loads the committed window-
    universe manifest, independently re-verifies its internal digest/
    count consistency (a self-contained reimplementation -- see
    ``_canonical_json_digest`` -- never importing the E23-bound,
    torch-dependent ``src.dfp_phase1b_development_window_universe``
    module from this preflight-safe path), and cross-checks it against
    E23's pinned image/window counts and digest."""
    e23_doc = e23_doc if e23_doc is not None else e23.load_identity(repo_root=repo_root)
    manifest_path = repo_root / WINDOW_UNIVERSE_RELATIVE_PATH
    try:
        document = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise DfpPhase1bBalpDevelopmentProductionError(f"cannot load development window-universe manifest {manifest_path}: {exc}") from exc

    images = document.get("images")
    if not isinstance(images, list):
        raise DfpPhase1bBalpDevelopmentProductionError("window-universe manifest 'images' must be a list")
    for rank, record in enumerate(images):
        if record.get("rank") != rank:
            raise DfpPhase1bBalpDevelopmentProductionError(f"window-universe manifest images[{rank}].rank disagrees with its own list position")
        if not isinstance(record.get("windows"), list) or len(record["windows"]) != record.get("window_count"):
            raise DfpPhase1bBalpDevelopmentProductionError(
                f"window-universe manifest images[{rank}] ({record.get('canonical_id')!r}): window_count disagrees with len(windows)"
            )
    recomputed_aggregate = sum(record["window_count"] for record in images)
    if recomputed_aggregate != document.get("aggregate_window_count"):
        raise DfpPhase1bBalpDevelopmentProductionError("window-universe manifest aggregate_window_count disagrees with the sum of per-image window_count values")
    recomputed_digest = _canonical_json_digest(images)
    if recomputed_digest != document.get("canonical_record_digest"):
        raise DfpPhase1bBalpDevelopmentProductionError(
            "window-universe manifest canonical_record_digest disagrees with a fresh digest of its own persisted images payload"
        )
    if len(images) != DEVELOPMENT_COUNT:
        raise DfpPhase1bBalpDevelopmentProductionError(f"window-universe manifest has {len(images)} images, expected {DEVELOPMENT_COUNT}")
    if document["aggregate_window_count"] != WINDOW_COUNT:
        raise DfpPhase1bBalpDevelopmentProductionError(f"window-universe manifest aggregate_window_count={document['aggregate_window_count']}, expected {WINDOW_COUNT}")
    if document["canonical_record_digest"] != WINDOW_UNIVERSE_DIGEST:
        raise DfpPhase1bBalpDevelopmentProductionError("window-universe manifest canonical_record_digest disagrees with the frozen required value")

    wu_binding = e23_doc["dependency_bindings"]["development_window_universe"]
    if wu_binding["image_count"] != len(images) or wu_binding["window_count"] != document["aggregate_window_count"] or wu_binding["canonical_record_digest"] != document["canonical_record_digest"]:
        raise DfpPhase1bBalpDevelopmentProductionError("window-universe manifest disagrees with E23's own pinned image_count/window_count/canonical_record_digest")

    canonical_ids = tuple(rec["canonical_id"] for rec in images)
    windows_per_image = {
        rec["canonical_id"]: tuple(w["index"] for w in rec["windows"])
        for rec in images
    }
    for rec in images:
        if len(windows_per_image[rec["canonical_id"]]) != rec["window_count"]:
            raise DfpPhase1bBalpDevelopmentProductionError(f"image {rec['canonical_id']}: window_count disagrees with its own window index list")

    return DevelopmentWindowIndex(
        canonical_ids=canonical_ids, windows_per_image=windows_per_image,
        aggregate_window_count=document["aggregate_window_count"], canonical_record_digest=document["canonical_record_digest"],
    )


def verify_population_isolation(
    window_index: DevelopmentWindowIndex, *, development_ids: Sequence[str], confirmation_ids: Sequence[str],
) -> None:
    """Fail-closed BEFORE any model/dataset access: the window-universe
    manifest's 350 canonical IDs must be byte-identical, in order, to
    E22's own frozen development split, and must not intersect the
    150-image confirmation split. Rejects duplicate, reordered,
    missing, or substituted IDs -- and (via disjointness with
    confirmation_ids) any of the other 4,500 validation images, since
    those can never appear in either frozen 500-image partition."""
    development_ids = tuple(development_ids)
    confirmation_ids = tuple(confirmation_ids)
    observed = window_index.canonical_ids

    if len(observed) != DEVELOPMENT_COUNT:
        raise DfpPhase1bBalpDevelopmentProductionError(f"population has {len(observed)} IDs, expected exactly {DEVELOPMENT_COUNT}")
    if len(set(observed)) != len(observed):
        raise DfpPhase1bBalpDevelopmentProductionError("population contains a duplicate canonical ID")
    if observed != development_ids:
        raise DfpPhase1bBalpDevelopmentProductionError(
            "window-universe population disagrees with E22's frozen, ordered 350-image development split "
            "(reordered, substituted, or drifted population)"
        )
    overlap = set(observed) & set(confirmation_ids)
    if overlap:
        raise DfpPhase1bBalpDevelopmentProductionError(f"population contains confirmation ID(s): {sorted(overlap)[:5]}")
    if window_index.aggregate_window_count != WINDOW_COUNT:
        raise DfpPhase1bBalpDevelopmentProductionError(f"aggregate window count {window_index.aggregate_window_count} != required {WINDOW_COUNT}")


def reject_out_of_population_id(canonical_id: str, *, window_index: DevelopmentWindowIndex) -> None:
    """Explicit runtime gate: raises before any image is ever prepared
    or any model call is made if ``canonical_id`` is not one of the
    350 frozen development IDs (covers confirmation IDs and all other
    4,500 validation images alike -- the population is a strict
    allow-list, never an exclude-list)."""
    if canonical_id not in window_index.windows_per_image:
        raise DfpPhase1bBalpDevelopmentProductionError(
            f"canonical id {canonical_id!r} is not one of the 350 frozen development IDs -- refusing to process"
        )


# ---------------------------------------------------------------------
# Contracts: narrow BalpContracts, discarding the masked-forward/
# component fields ComponentContractsBundle also carries. BALP's own
# committed processor (process_development_image) only ever reads
# .recurrence/.stitching/.pullback/.final_resolution -- constructing a
# narrower object here means even a future accidental reference to
# .component/.masked_forward inside this module fails loudly
# (AttributeError) rather than silently succeeding.
# ---------------------------------------------------------------------


def narrow_balp_contracts(component_contracts_bundle: Any) -> "execmod.BalpContracts":
    import src.dfp_phase1b_lightweight_boundary_execute as execmod

    return execmod.BalpContracts(
        recurrence=component_contracts_bundle.recurrence,
        stitching=component_contracts_bundle.stitching,
        pullback=component_contracts_bundle.pullback,
        final_resolution=component_contracts_bundle.final_resolution,
    )


# ---------------------------------------------------------------------
# Real provider construction. Every heavy (torch/mmcv/model) import is
# deferred into these function bodies, exactly like the E23-bound
# `dfp_phase1b_calibration_production` module they call into.
# ---------------------------------------------------------------------


def build_real_dataset_resources(repo_root: Path, *, development_ids: Sequence[str]) -> Any:
    from src.dfp_phase1b_calibration_production import build_production_dataset_resources

    return build_production_dataset_resources(repo_root=repo_root, selected_ids=list(development_ids))


def build_real_inference(resources: Any, *, device: str, log_dir: Path) -> Any:
    from src.dfp_phase1b_calibration_production import build_production_inference

    return build_production_inference(resources, device=device, log_dir=log_dir)


def build_execution_providers(
    repo_root: Path, *, window_index: DevelopmentWindowIndex, device: str, log_dir: Path,
    dataset_resources_builder: Callable[..., Any], inference_builder: Callable[..., Any],
) -> "execmod.ExecutionProviders":
    """Builds dataset resources exactly once and the model exactly
    once per process (both are single, non-memoized calls made
    directly here -- callers must never call this function twice in
    one process). Narrows the production contracts bundle to
    BalpContracts before it is ever exposed to the scientific
    processor."""
    import src.dfp_phase1b_lightweight_boundary_execute as execmod

    resources = dataset_resources_builder(repo_root, window_index.canonical_ids)
    inference = inference_builder(resources, device, log_dir)
    contracts = narrow_balp_contracts(resources.contracts)
    return execmod.ExecutionProviders(
        dataset_resources=resources, image_preparer=None, inference=inference,
        protocol_identity=resources.protocol_identity, contracts=contracts,
        class_count=resources.class_count, affinity_power=resources.affinity_power,
        windows_per_image=dict(window_index.windows_per_image),
    )


# ---------------------------------------------------------------------
# image_provider_fn: real per-image prepared-tensor + GT resolution.
# GT isolation boundary -- ``prepared`` (fed to process_development_image)
# and ``ground_truth`` (fed only to accumulate_image_metrics, strictly
# AFTER prediction) are resolved here as two independent fields of the
# same DevelopmentImageProvider; the scientific processor never
# receives ``ground_truth`` at all (see execute_development's own call
# sites).
# ---------------------------------------------------------------------


def _expected_window_span(inference_height: int, inference_width: int, *, crop: int = 448, stride: int = 224) -> int:
    """Pure re-derivation of SlidingWindowPlan.build's own window_count
    for a decoded image's real geometry -- used only as a sanity upper
    bound on committed window indices, never to reconstruct the window
    list itself (that always comes from the committed manifest)."""
    grid_rows = max(inference_height - crop + stride - 1, 0) // stride + 1
    grid_cols = max(inference_width - crop + stride - 1, 0) // stride + 1
    return grid_rows * grid_cols


def build_real_image_provider_fn(
    resources: Any, *, window_index: DevelopmentWindowIndex, expected_ignore_index: int,
) -> "Callable[[str], execmod.DevelopmentImageProvider]":
    import src.dfp_phase1b_lightweight_boundary_execute as execmod
    from diagnostics.run_k11_k12_stability import _extract_prepared_image
    from src.dfp_phase1a_dataset_view import verify_position_identity

    positions_by_id = {p.canonical_id: p for p in resources.execution_view.positions}
    dataset = resources.dataset
    observed_ignore_index = int(dataset.ignore_index)
    if observed_ignore_index != expected_ignore_index:
        raise DfpPhase1bBalpDevelopmentProductionError(
            f"real dataset ignore_index={observed_ignore_index} disagrees with the E13-frozen ignore_index={expected_ignore_index}"
        )

    def _provide(canonical_id: str) -> execmod.DevelopmentImageProvider:
        reject_out_of_population_id(canonical_id, window_index=window_index)
        position = positions_by_id.get(canonical_id)
        if position is None:
            raise DfpPhase1bBalpDevelopmentProductionError(
                f"canonical id {canonical_id!r} has no resolved native-dataset position (not in the manifest execution view)"
            )
        verify_position_identity(
            dataset, manifest_position=position.manifest_position,
            canonical_id=position.canonical_id, native_index=position.native_index,
        )
        prepared = _extract_prepared_image(dataset, position.native_index, canonical_image_id=position.relative_path)

        selected_windows = window_index.windows_per_image[canonical_id]
        expected_geometry_windows = _expected_window_span(prepared.inference_height, prepared.inference_width)
        if len(selected_windows) == 0 or max(selected_windows) >= expected_geometry_windows:
            raise DfpPhase1bBalpDevelopmentProductionError(
                f"image {canonical_id}: committed window indices {selected_windows} are inconsistent with the real "
                f"decoded inference-canvas geometry ({prepared.inference_height}x{prepared.inference_width})"
            )

        # GT read only here, strictly for later confusion-statistic
        # accumulation (see accumulate_image_metrics) -- never fed to
        # process_development_image, which receives only `prepared`.
        gt_raw = dataset.get_gt_seg_map_by_idx(position.native_index)
        ground_truth = np.asarray(gt_raw).astype(np.int64)
        ori_shape = tuple(prepared.ori_shape)
        if ground_truth.shape != ori_shape:
            raise DfpPhase1bBalpDevelopmentProductionError(
                f"image {canonical_id}: GT shape {ground_truth.shape} disagrees with the canonical final ori_shape {ori_shape} "
                "-- refusing to resize any prediction or GT to match"
            )

        return execmod.DevelopmentImageProvider(prepared=prepared, ground_truth=ground_truth, selected_windows=selected_windows)

    return _provide


# ---------------------------------------------------------------------
# ProductionBuilders: the injection point for the three real-or-fake
# boundaries this module orchestrates. Production callers use the
# defaults (the real E23-bound builders); tests inject deterministic
# fakes here and nowhere else -- the orchestration logic itself
# (population enforcement, contracts narrowing, ExecutionProviders/
# image_provider_fn assembly, execute_development invocation) is
# IDENTICAL in both cases.
# ---------------------------------------------------------------------


def _default_image_provider_fn_builder(resources: Any, window_index: DevelopmentWindowIndex, expected_ignore_index: int) -> Any:
    """Thin positional-argument adapter so this can be used as a
    dataclass field default: ``build_real_image_provider_fn`` itself
    takes ``window_index``/``expected_ignore_index`` as keyword-only."""
    return build_real_image_provider_fn(resources, window_index=window_index, expected_ignore_index=expected_ignore_index)


@dataclass(frozen=True)
class ProductionBuilders:
    dataset_resources_builder: Callable[[Path, Sequence[str]], Any] = build_real_dataset_resources
    inference_builder: Callable[[Any, str, Path], Any] = build_real_inference
    image_provider_fn_builder: "Callable[[Any, DevelopmentWindowIndex, int], Callable[[str], execmod.DevelopmentImageProvider]]" = _default_image_provider_fn_builder
    # Test-only override of the per-window scientific processor, mirroring
    # the existing committed test suite's own established technique for
    # exercising execute_development's orchestration at full 350-image
    # scale without a real model/graph/recurrence (see
    # tests/test_dfp_phase1b_lightweight_boundary_execute.py's
    # `_fake_processor` + `_clean_fresh_result`). `None` (the production
    # default) means "use the committed, E23-bound
    # execmod.process_development_image unmodified" -- this module never
    # substitutes a different processor in its own real execution path.
    image_processor_fn: Any = None


# ---------------------------------------------------------------------
# Top-level orchestration entry point.
# ---------------------------------------------------------------------


def run_development_execution(
    repo_root: Path, output_root: Path, *, device: str, log_dir: Path,
    builders: ProductionBuilders = ProductionBuilders(),
    recorder: "execmod.ProfilingRecorder | None" = None,
) -> Mapping[str, Any]:
    """The single real (or fake-provider-tested) entry point: verifies
    authority, builds providers exactly once, enforces the 350/769
    population, and delegates ALL checkpoint/resume/finalize-only
    orchestration to the committed, unmodified
    ``execmod.execute_development`` -- this function never touches a
    checkpoint file itself."""
    import src.dfp_phase1b_lightweight_boundary_execute as execmod

    e23_doc, e22_doc, split, _ = load_verified_authority(repo_root)
    window_index = load_development_window_index(repo_root, e23_doc=e23_doc)
    development_ids = tuple(split["development_image_ids"])
    confirmation_ids = tuple(split["confirmation_image_ids"])
    verify_population_isolation(window_index, development_ids=development_ids, confirmation_ids=confirmation_ids)

    ignore_index = e22_doc["canonical_horizons"]["ignore_index"]
    identity_bindings_without_digest = build_identity_bindings_without_digest(e23_doc)
    providers = build_execution_providers(
        repo_root, window_index=window_index, device=device, log_dir=log_dir,
        dataset_resources_builder=builders.dataset_resources_builder, inference_builder=builders.inference_builder,
    )
    image_provider_fn = builders.image_provider_fn_builder(providers.dataset_resources, window_index, ignore_index)

    execute_kwargs: dict[str, Any] = dict(
        providers=providers, development_ids=development_ids, confirmation_ids=confirmation_ids,
        identity_bindings_without_digest=identity_bindings_without_digest, ignore_index=ignore_index,
        image_provider_fn=image_provider_fn, recorder=recorder,
    )
    if builders.image_processor_fn is not None:
        execute_kwargs["image_processor_fn"] = builders.image_processor_fn
    return execmod.execute_development(Path(output_root), **execute_kwargs)
