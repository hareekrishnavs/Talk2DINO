"""DFP Phase 1B: materialization and independent verification of the
frozen 350-image development window universe.

Resolves the still-open "development window-universe authority"
blocker identified during E22/BALP-runner verification: prior to this
module, no committed or frozen repository source recorded the exact
per-image inference-canvas geometry or sliding-window layout for the
350 development images, only the 500-image split membership. This
module closes that gap by RUNNING the real, already-committed
production dataset/geometry adapter (CPU-only, model-free) against the
real images, and persisting the resulting geometry/window records as a
new frozen artifact -- never inventing a synthetic or estimated window
count.

Strict boundaries, enforced by construction (not merely by convention):

* Reads only the 350 frozen ``development_image_ids`` returned by
  ``src.dfp_phase1b_lightweight_boundary_identity.load_and_verify_split``
  (itself an independently-reconstructed, fail-closed 350/150
  partition) -- confirmation IDs and the other 4,500 COCO-Stuff val2017
  images are never even resolvable, because
  ``build_production_dataset_resources`` builds its execution view
  from exactly this 350-ID list (see
  :func:`demonstrate_out_of_scope_rejection` for an explicit runtime
  proof of this).
* Never reads ground truth: ``ProductionGeometryProvider`` (reused
  unmodified from ``src.dfp_phase1b_calibration_production``) never
  calls ``dataset.get_gt_seg_map_by_idx`` or any GT-shaped field.
* Never constructs a model, checkpoint, or CUDA context:
  ``build_production_inference`` is never called anywhere in this
  module.
* Never reimplements resize/window math: the per-image window list is
  built by the single canonical, pure primitive
  ``src.open_vocabulary_segmentation....sliding_window_geometry.
  SlidingWindowPlan.build``, fed the frozen ``crop``/``stride`` from
  ``src.dfp_phase1a_horizon_stitching.dfp_stitching_contract_from_protocol``
  (itself sourced from E13's own ``[score_interpolation_stitching]``
  block, which names ``SlidingWindowPlan.build`` as the frozen
  ``window_enumeration`` authority).

``KeyboardInterrupt``, ``SystemExit``, and ``MemoryError`` always
propagate unmodified; ordinary parse / type / relational / geometry
failures are wrapped in
:class:`DfpPhase1bDevelopmentWindowUniverseError`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.dfp_phase1a_horizon_stitching import dfp_stitching_contract_from_protocol
from src.dfp_phase1b_calibration_production import (
    DfpPhase1bCalibrationProductionError,
    build_production_dataset_resources,
    build_production_geometry_provider,
)
from src.dfp_phase1b_chunk_benchmark_identity import resolve_repository_relative_path
from src.dfp_phase1b_lightweight_boundary_identity import (
    ARTIFACTS_RELATIVE_PATH,
    PRESERVATION_RELATIVE_PATH,
    SPLIT_RELATIVE_PATH,
    load_and_verify_split,
)
from src.open_vocabulary_segmentation.segmentation.evaluation.sliding_window_geometry import (
    SlidingWindowPlan,
    SpatialSize,
    WindowGeometry,
)


class DfpPhase1bDevelopmentWindowUniverseError(ValueError):
    """Raised on any development-window-universe invariant violation.
    Always fail closed. Never raised for, and never catches,
    KeyboardInterrupt / SystemExit / MemoryError -- those always
    propagate unmodified."""


SCHEMA_NAME = "talk2dino-dfp-phase1b-balp-development-window-universe-v1"
SCHEMA_VERSION = 1
MANIFEST_RELATIVE_PATH = Path("evaluation_subsets/dfp_phase1b_development_window_universe.json")
DEVELOPMENT_COUNT = 350

E13_PROTOCOL_RELATIVE_PATH = Path("evaluation_identities/e13_dfp_phase1a_protocol.toml")
HORIZON_STITCHING_MODULE_RELATIVE_PATH = Path("src/dfp_phase1a_horizon_stitching.py")
SLIDING_WINDOW_GEOMETRY_MODULE_RELATIVE_PATH = Path(
    "src/open_vocabulary_segmentation/segmentation/evaluation/sliding_window_geometry.py"
)
PRODUCTION_CALIBRATION_MODULE_RELATIVE_PATH = Path("src/dfp_phase1b_calibration_production.py")
DATASET_VIEW_MODULE_RELATIVE_PATH = Path("src/dfp_phase1a_dataset_view.py")

# Every file this artifact's provenance is bound to -- hashed, never
# re-derived from. Sorted-key iteration below makes the persisted
# dependency_bindings block deterministic regardless of dict construction
# order.
_DEPENDENCY_RELATIVE_PATHS: dict[str, Path] = {
    "e13_protocol": E13_PROTOCOL_RELATIVE_PATH,
    "e22_preservation": PRESERVATION_RELATIVE_PATH,
    "e22_artifacts": ARTIFACTS_RELATIVE_PATH,
    "balp_split_artifact": SPLIT_RELATIVE_PATH,
    "horizon_stitching_module": HORIZON_STITCHING_MODULE_RELATIVE_PATH,
    "sliding_window_geometry_module": SLIDING_WINDOW_GEOMETRY_MODULE_RELATIVE_PATH,
    "production_calibration_module": PRODUCTION_CALIBRATION_MODULE_RELATIVE_PATH,
    "dataset_view_module": DATASET_VIEW_MODULE_RELATIVE_PATH,
}

_TOP_LEVEL_SCALAR_FIELDS = (
    "schema_name",
    "schema_version",
    "namespace",
    "source_development_digest",
    "development_count",
    "crop",
    "stride",
    "align_corners",
    "aggregate_window_count",
    "canonical_record_digest",
    "per_image_window_counts_digest",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_digest(payload: Any) -> str:
    """Deterministic SHA-256 over an explicit, order-preserving JSON
    encoding -- fixed key order, no set/dict-iteration dependence, no
    ``PYTHONHASHSEED`` dependence. Same convention as
    ``src.dfp_phase1a_dataset_view.compute_mapping_digest``."""
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_dependency_bindings(repo_root: Path) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name in sorted(_DEPENDENCY_RELATIVE_PATHS):
        relative_path = _DEPENDENCY_RELATIVE_PATHS[name]
        target = resolve_repository_relative_path(repo_root, str(relative_path), binding_name=name)
        bindings[name] = {"relative_path": str(relative_path), "sha256": sha256_file(target)}
    return bindings


def window_geometry_to_record(window: WindowGeometry) -> dict[str, Any]:
    return {
        "index": window.index,
        "grid_row": window.grid_row,
        "grid_col": window.grid_col,
        "nominal_origin": [window.nominal_origin.row, window.nominal_origin.col],
        "origin": [window.origin.row, window.origin.col],
        "end": [window.end.row, window.end.col],
        "is_top": window.is_top,
        "is_bottom": window.is_bottom,
        "is_left": window.is_left,
        "is_right": window.is_right,
        "clamped_vertical": window.clamped_vertical,
        "clamped_horizontal": window.clamped_horizontal,
    }


def build_image_window_record(
    *, rank: int, canonical_id: str, inference_height: int, inference_width: int, patch_size: int, plan: SlidingWindowPlan,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "canonical_id": canonical_id,
        "inference_height": inference_height,
        "inference_width": inference_width,
        "patch_size": patch_size,
        "window_count": plan.window_count,
        "windows": [window_geometry_to_record(w) for w in plan.windows],
    }


def materialize_development_window_universe(
    *, repo_root: Path, progress_callback: Any = None,
) -> dict[str, Any]:
    """CPU-only, model-free, GT-free. Builds the real production dataset/
    geometry adapter scoped to exactly the 350 frozen development IDs,
    then the real per-image sliding-window plan via the canonical
    ``SlidingWindowPlan.build`` primitive fed the frozen E13 crop/stride.
    Never touches GT, never constructs a model, never initializes CUDA.
    ``progress_callback(rank, canonical_id, window_count)``, if given, is
    invoked once per image in development order (for CLI progress
    reporting only -- has no effect on the returned document)."""
    split = load_and_verify_split(repo_root=repo_root)
    development_ids = tuple(split["development_image_ids"])
    if len(development_ids) != DEVELOPMENT_COUNT:
        raise DfpPhase1bDevelopmentWindowUniverseError(
            f"independently-verified split returned {len(development_ids)} development IDs, expected {DEVELOPMENT_COUNT}"
        )

    resources = build_production_dataset_resources(repo_root=repo_root, selected_ids=list(development_ids))
    provider = build_production_geometry_provider(resources)
    contract = dfp_stitching_contract_from_protocol(resources.protocol_identity, repo_root=repo_root)
    crop = SpatialSize(*contract.crop)
    stride = SpatialSize(*contract.stride)

    images: list[dict[str, Any]] = []
    aggregate_window_count = 0
    for rank, canonical_id in enumerate(development_ids):
        geometry = provider(canonical_id)
        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(geometry.inference_height, geometry.inference_width),
            crop_size=crop,
            stride=stride,
        )
        record = build_image_window_record(
            rank=rank,
            canonical_id=canonical_id,
            inference_height=geometry.inference_height,
            inference_width=geometry.inference_width,
            patch_size=geometry.patch_size,
            plan=plan,
        )
        images.append(record)
        aggregate_window_count += plan.window_count
        if progress_callback is not None:
            progress_callback(rank, canonical_id, plan.window_count)

    canonical_record_digest = _canonical_json_digest(images)
    per_image_window_counts_digest = _canonical_json_digest(
        [{"canonical_id": r["canonical_id"], "window_count": r["window_count"]} for r in images]
    )
    dependency_bindings = compute_dependency_bindings(repo_root)

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "namespace": SCHEMA_NAME,
        "source_development_digest": split["development_digest"],
        "development_count": len(development_ids),
        "crop": list(contract.crop),
        "stride": list(contract.stride),
        "align_corners": contract.align_corners,
        "images": images,
        "aggregate_window_count": aggregate_window_count,
        "canonical_record_digest": canonical_record_digest,
        "per_image_window_counts_digest": per_image_window_counts_digest,
        "dependency_bindings": dependency_bindings,
    }


def verify_window_universe_document(document: Mapping[str, Any], *, repo_root: Path) -> None:
    """Structural/relational re-verification of an already-produced
    document already in memory: schema identity, internal digest
    self-consistency, and current dependency-hash re-check. Does NOT
    recompute geometry from the real dataset -- see
    :func:`independently_reconstruct_and_verify` for the full second
    pass."""
    if document.get("schema_name") != SCHEMA_NAME or document.get("namespace") != SCHEMA_NAME:
        raise DfpPhase1bDevelopmentWindowUniverseError("document schema_name/namespace disagrees with the frozen schema")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise DfpPhase1bDevelopmentWindowUniverseError("document schema_version disagrees with the frozen schema")

    images = document.get("images")
    if not isinstance(images, list) or len(images) != DEVELOPMENT_COUNT:
        raise DfpPhase1bDevelopmentWindowUniverseError(
            f"document.images must be a list of exactly {DEVELOPMENT_COUNT} records, observed "
            f"{len(images) if isinstance(images, list) else type(images).__name__}"
        )
    for rank, record in enumerate(images):
        if record.get("rank") != rank:
            raise DfpPhase1bDevelopmentWindowUniverseError(f"images[{rank}].rank disagrees with its own list position")
        if not isinstance(record.get("windows"), list) or len(record["windows"]) != record.get("window_count"):
            raise DfpPhase1bDevelopmentWindowUniverseError(
                f"images[{rank}] ({record.get('canonical_id')!r}): window_count disagrees with len(windows)"
            )

    observed_aggregate = sum(record["window_count"] for record in images)
    if observed_aggregate != document.get("aggregate_window_count"):
        raise DfpPhase1bDevelopmentWindowUniverseError(
            f"aggregate_window_count {document.get('aggregate_window_count')!r} disagrees with the sum of per-image "
            f"window_count values ({observed_aggregate})"
        )

    recomputed_canonical_digest = _canonical_json_digest(images)
    if recomputed_canonical_digest != document.get("canonical_record_digest"):
        raise DfpPhase1bDevelopmentWindowUniverseError(
            "canonical_record_digest disagrees with a fresh digest of the persisted images payload -- the document "
            "was tampered with, truncated, or reserialized with a different key order after being written"
        )
    recomputed_counts_digest = _canonical_json_digest(
        [{"canonical_id": r["canonical_id"], "window_count": r["window_count"]} for r in images]
    )
    if recomputed_counts_digest != document.get("per_image_window_counts_digest"):
        raise DfpPhase1bDevelopmentWindowUniverseError("per_image_window_counts_digest disagrees with a fresh recomputation")

    observed_bindings = document.get("dependency_bindings")
    expected_bindings = compute_dependency_bindings(repo_root)
    if observed_bindings != expected_bindings:
        raise DfpPhase1bDevelopmentWindowUniverseError(
            "dependency_bindings disagree with the current repository state -- a bound source/E22/runner file "
            "has drifted since this document was generated"
        )

    split = load_and_verify_split(repo_root=repo_root)
    if document.get("source_development_digest") != split["development_digest"]:
        raise DfpPhase1bDevelopmentWindowUniverseError(
            "source_development_digest disagrees with the independently-reconstructed BALP split's own development_digest"
        )
    persisted_ids = [r["canonical_id"] for r in images]
    if persisted_ids != list(split["development_image_ids"]):
        raise DfpPhase1bDevelopmentWindowUniverseError(
            "persisted image order/membership disagrees with the independently-reconstructed 350 development IDs"
        )


def independently_reconstruct_and_verify(*, repo_root: Path, persisted_path: Path) -> dict[str, Any]:
    """The full second pass: re-runs :func:`verify_window_universe_document`
    against the on-disk document, then independently rebuilds the ENTIRE
    window universe from scratch (a fresh dataset construction, fresh
    per-image geometry extraction, fresh window-plan build -- never
    reusing any in-memory object from the run that produced
    ``persisted_path``) and requires byte-for-byte agreement on every
    record and every top-level field. Returns the freshly reconstructed
    document on success."""
    try:
        persisted = json.loads(persisted_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise DfpPhase1bDevelopmentWindowUniverseError(f"cannot load persisted document {persisted_path}: {error}") from error

    verify_window_universe_document(persisted, repo_root=repo_root)

    reconstructed = materialize_development_window_universe(repo_root=repo_root)

    for field in _TOP_LEVEL_SCALAR_FIELDS:
        if reconstructed[field] != persisted.get(field):
            raise DfpPhase1bDevelopmentWindowUniverseError(
                f"independent reconstruction disagrees with the persisted document on field {field!r}: "
                f"reconstructed={reconstructed[field]!r} persisted={persisted.get(field)!r}"
            )
    if reconstructed["images"] != persisted["images"]:
        raise DfpPhase1bDevelopmentWindowUniverseError(
            "independent reconstruction disagrees with the persisted document's images payload"
        )
    if reconstructed["dependency_bindings"] != persisted["dependency_bindings"]:
        raise DfpPhase1bDevelopmentWindowUniverseError(
            "independent reconstruction disagrees with the persisted document's dependency_bindings"
        )

    return reconstructed


def demonstrate_out_of_scope_rejection(*, repo_root: Path) -> dict[str, Any]:
    """Runtime proof (not merely a claim) that the execution view built
    from exactly the 350 development IDs cannot resolve a confirmation
    ID or an arbitrary non-selected COCO-Stuff val2017 ID. Builds its
    own resources scoped only to the development IDs (identical to
    :func:`materialize_development_window_universe`'s own scoping) and
    asserts both probes raise ``DfpPhase1bCalibrationProductionError``.
    Raises :class:`DfpPhase1bDevelopmentWindowUniverseError` if either
    probe unexpectedly succeeds (a scope leak)."""
    split = load_and_verify_split(repo_root=repo_root)
    development_ids = list(split["development_image_ids"])
    confirmation_ids = list(split["confirmation_image_ids"])

    resources = build_production_dataset_resources(repo_root=repo_root, selected_ids=development_ids)
    provider = build_production_geometry_provider(resources)

    development_set = set(development_ids)
    confirmation_set = set(confirmation_ids)
    confirmation_probe_id = confirmation_ids[0]

    img_suffix = ".jpg"
    other_probe_id = next(
        info["filename"][: -len(img_suffix)]
        for info in resources.dataset.img_infos
        if info["filename"].endswith(img_suffix)
        and info["filename"][: -len(img_suffix)] not in development_set
        and info["filename"][: -len(img_suffix)] not in confirmation_set
    )

    probes = {"confirmation_id": confirmation_probe_id, "other_non_development_id": other_probe_id}
    results: dict[str, Any] = {}
    for label, probe_id in probes.items():
        try:
            provider(probe_id)
        except DfpPhase1bCalibrationProductionError as error:
            results[label] = {"probe_id": probe_id, "rejected": True, "error": str(error)}
        else:
            raise DfpPhase1bDevelopmentWindowUniverseError(
                f"scope leak: probe id {probe_id!r} ({label}) resolved through a geometry provider scoped to only "
                f"the 350 development IDs -- it must have been rejected"
            )
    return results
