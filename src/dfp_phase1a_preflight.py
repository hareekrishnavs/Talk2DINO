"""DFP Phase 1A Step 12: unified, ordered, CPU-only preflight orchestration.

This module implements NO scientific logic of its own. Every schema,
identity, subset-manifest, model/evaluator, checkpoint, and qualitative-
report check is delegated to the already-committed, already-verified
Step 1-11 validators (``dfp_phase1a_identity``, ``dfp_phase1a_artifact_
schema``, ``dfp_phase1a_subset_manifest``, ``dfp_phase1a_checkpoint``,
``dfp_phase1a_qualitative_selection``, ``matched_k11_k12_identity``,
``ade20k_matched_evaluator_identity``, ``ade20k_dataset_identity``,
``ade20k_dataset_manifest``). This module's only new logic is: (a) the
explicit ordered composition of those validators into Phase A / B / C,
(b) resolving the frozen dataset root and reconciling the 500 selected
canonical IDs against physical image/annotation paths without ever
guessing a path from an ID alone, (c) output-root safety, and (d) the
fresh/resume decision and the immutable :class:`PreflightContext` a
future execution driver will consume.

No entry point in this module imports ``torch``, constructs a model or
dataset object, calls ``torch.load``, or queries CUDA. See
``tests/test_dfp_phase1a_preflight.py`` for the poison-import proof.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from src.dfp_phase1a_artifact_schema import (
    DfpPhase1aArtifactSchemaError,
    RUN_MODE_NAMES,
    parse_strict_json_document,
    sha256_file,
)
from src.dfp_phase1a_artifact_schema import validate_static_configuration as validate_artifact_schema_configuration
from src.dfp_phase1a_checkpoint import (
    DfpPhase1aCheckpointError,
    validate_resume_state,
)
from src.dfp_phase1a_identity import (
    DfpPhase1aIdentityError,
    IDENTITY_RELATIVE_PATH as PROTOCOL_IDENTITY_RELATIVE_PATH,
)
from src.dfp_phase1a_identity import validate_static_configuration as validate_protocol_configuration
from src.dfp_phase1a_qualitative_selection import (
    DfpPhase1aQualitativeReportError,
)
from src.dfp_phase1a_qualitative_selection import validate_static_configuration as validate_qualitative_configuration
from src.dfp_phase1a_report import DfpPhase1aReportError, verify_result_bundle
from src.dfp_phase1a_subset_manifest import (
    ADE20K_CLASS_COUNT,
    COCO_STUFF_ANNOTATION_SUFFIX,
    COCO_STUFF_IMAGE_SUFFIX,
    DATASET_CONTRACTS,
    DfpPhase1aSubsetManifestError,
    require_env_root,
    resolve_coco_stuff_root,
)
from src.dfp_phase1a_subset_manifest import parse_strict_json_document as parse_strict_json_subset_document
from src.dfp_phase1a_subset_manifest import verify_manifest_document
from src.ade20k_dataset_identity import Ade20kDatasetIdentityError


class DfpPhase1aPreflightError(ValueError):
    """Raised on any DFP Phase 1A Step 12 preflight-ordering, binding, or
    invariant violation. Always fail closed. Never raised for a scientific
    disagreement -- only for a failed proof that execution may safely
    begin."""


# ---------------------------------------------------------------------------
# Run-mode binding table. RUN_MODE_NAMES is the single frozen authority
# (from dfp_phase1a_artifact_schema); this table adds only the additional
# static facts Step 12 itself needs to route to the right validators.
# ---------------------------------------------------------------------------

_DATASET_KEY_BY_RUN_MODE: Mapping[str, str] = MappingProxyType({"coco_stuff_500": "coco_stuff", "ade20k_500": "ade20k"})
ENV_VAR_BY_RUN_MODE: Mapping[str, str] = MappingProxyType(
    {"coco_stuff_500": "COCO_STUFF_REAL_DATA_ROOT", "ade20k_500": "ADE20K_REAL_DATA_ROOT"}
)
MANIFEST_RELATIVE_PATH_BY_RUN_MODE: Mapping[str, str] = MappingProxyType(
    {
        "coco_stuff_500": "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
        "ade20k_500": "evaluation_subsets/dfp_phase1a_ade20k_500.json",
    }
)

# Explicit, immutable ordering of every named stage. Recorded on a
# PreflightTrace (test-only observability) in exactly this order and no
# other; a stage is never skipped and never runs out of order.
PHASE_A_STAGES: tuple[str, ...] = (
    "phase_a.parse_run_mode",
    "phase_a.resolve_repo_root",
    "phase_a.protocol_identity",
    "phase_a.git_ancestry_policy",
    "phase_a.artifact_schema",
    "phase_a.qualitative_report_schema",
    "phase_a.subset_manifest",
    "phase_a.parent_identity_hashes",
    "phase_a.select_run_binding",
)
PHASE_B_STAGES: tuple[str, ...] = (
    "phase_b.resolve_dataset_root",
    "phase_b.dataset_source_identity",
    "phase_b.canonical_selected_ids",
    "phase_b.reconcile_physical_paths",
    "phase_b.model_evaluator_binding",
    "phase_b.bridge_checkpoint_raw_hash",
    "phase_b.output_root_safety",
)
PHASE_C_STAGES: tuple[str, ...] = (
    "phase_c.decide_fresh_or_resume",
    "phase_c.checkpoint_or_absence",
    "phase_c.completed_id_prefix",
    "phase_c.per_image_manifest_and_npz",
    "phase_c.array_reconciliation",
    "phase_c.reject_already_complete_resume",
    "phase_c.result_policy",
    "phase_c.build_context",
)
ALL_STAGES: tuple[str, ...] = PHASE_A_STAGES + PHASE_B_STAGES + PHASE_C_STAGES


class PreflightTrace:
    """Append-only stage-order observability hook. Production callers may
    omit this entirely (every ``trace=`` parameter below defaults to
    ``None`` and every record call is a no-op in that case); tests pass an
    instance and inspect ``.stages`` afterward to prove exact ordering.
    Never influences validation outcomes -- purely an observer."""

    def __init__(self) -> None:
        self._stages: list[str] = []

    def record(self, stage_name: str) -> None:
        if stage_name not in ALL_STAGES:
            raise DfpPhase1aPreflightError(f"internal error: unknown stage name {stage_name!r}")
        self._stages.append(stage_name)

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(self._stages)


def _trace(trace: PreflightTrace | None, stage_name: str) -> None:
    if trace is not None:
        trace.record(stage_name)


def dataset_key_for_run_mode(run_mode: str) -> str:
    if run_mode not in RUN_MODE_NAMES:
        raise DfpPhase1aPreflightError(f"run mode must be one of {list(RUN_MODE_NAMES)}, got {run_mode!r}")
    return _DATASET_KEY_BY_RUN_MODE[run_mode]


# ---------------------------------------------------------------------------
# Phase A: repository and schema. No dataset/model/CUDA entry point is
# reachable from anything in this phase.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseAResult:
    run_mode: str
    dataset_key: str
    repo_root: Path
    protocol_identity_sha256: str
    artifact_schema_identity_sha256: str
    qualitative_report_identity_sha256: str
    subset_manifest_sha256: str
    subset_manifest_relative_path: str
    bridge_checkpoint_sha256: str
    required_ancestor_commit: str
    class_count: int
    class_names_digest: str
    selected_ids: tuple[str, ...]
    selected_ids_digest: str
    expected_count: int
    source_commit: str


def verify_frozen_subset_manifest(
    *, repo_root: Path, dataset_key: str, protocol_identity_sha256: str, manifest_path: Path | None = None,
) -> tuple[int, str, Mapping[str, Any]]:
    """Narrow adapter around the already-committed Step 2 subset-manifest
    verifier (:func:`dfp_phase1a_subset_manifest.verify_manifest_document`):
    resolves the class-count/class-names-digest/source-identity-sha256
    triple exactly as ``prepare_dfp_phase1a_subsets.py``'s own production
    ``verify`` command does, then delegates the actual schema/relational
    check entirely to that production function. Returns
    ``(class_count, class_names_digest, manifest)``. Reused by both Phase
    A and the unified CLI's standalone ``verify-subset`` command so the
    logic exists in exactly one place."""
    root = Path(repo_root)
    if manifest_path is None:
        manifest_path = root / MANIFEST_RELATIVE_PATH_BY_RUN_MODE["coco_stuff_500" if dataset_key == "coco_stuff" else "ade20k_500"]

    try:
        manifest = parse_strict_json_subset_document(manifest_path, label="subset manifest")
    except DfpPhase1aSubsetManifestError as error:
        raise DfpPhase1aPreflightError(str(error)) from error

    class_count_key = "coco_stuff_class_count" if dataset_key == "coco_stuff" else "ade20k_class_count"
    class_names_digest_key = "coco_stuff_class_names_digest" if dataset_key == "coco_stuff" else "ade20k_class_names_digest"

    try:
        from src.dfp_phase1a_artifact_schema import load_identity as load_artifact_identity

        artifact_identity = load_artifact_identity(repo_root=root)
        class_count = artifact_identity["dataset_binding"][class_count_key]
        class_names_digest = artifact_identity["dataset_binding"][class_names_digest_key]

        contract = DATASET_CONTRACTS[dataset_key]
        source_identity_sha256 = sha256_file(root / contract["source_identity_relative_path"])
        verify_manifest_document(
            manifest, dataset_key_expected=dataset_key, protocol_identity_sha256=protocol_identity_sha256,
            source_identity_sha256=source_identity_sha256, class_names_digest=class_names_digest,
        )
    except DfpPhase1aSubsetManifestError as error:
        raise DfpPhase1aPreflightError(f"frozen subset manifest failed verification: {error}") from error

    return class_count, class_names_digest, manifest


def run_phase_a(*, repo_root: Path, run_mode: str, check_git: bool = True, trace: PreflightTrace | None = None) -> PhaseAResult:
    """Ordered Phase A: parse the exact run name, load/validate the
    protocol, artifact schema, qualitative-report schema, and frozen
    subset manifest, verify every parent-identity hash, and select the
    exact run-specific binding. No dataset root, model file, checkpoint,
    or CUDA entry point is touched."""
    _trace(trace, "phase_a.parse_run_mode")
    if run_mode not in RUN_MODE_NAMES:
        raise DfpPhase1aPreflightError(f"run mode must be exactly one of {list(RUN_MODE_NAMES)} (no aliases/case changes), got {run_mode!r}")
    dataset_key = dataset_key_for_run_mode(run_mode)

    _trace(trace, "phase_a.resolve_repo_root")
    root = Path(repo_root)
    if not root.is_dir():
        raise DfpPhase1aPreflightError(f"--repo-root {root} is not a directory")

    _trace(trace, "phase_a.protocol_identity")
    try:
        validate_protocol_configuration(repo_root=root, check_git=check_git)
        protocol_identity_sha256 = sha256_file(root / PROTOCOL_IDENTITY_RELATIVE_PATH)
    except DfpPhase1aIdentityError as error:
        raise DfpPhase1aPreflightError(f"scientific protocol identity validation failed: {error}") from error

    _trace(trace, "phase_a.git_ancestry_policy")
    # Git ancestry is already checked transitively by validate_protocol_
    # configuration above when check_git=True; this stage exists as its
    # own named, independently-orderable step per the frozen phase list.

    _trace(trace, "phase_a.artifact_schema")
    try:
        artifact_summary = validate_artifact_schema_configuration(repo_root=root)
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aPreflightError(f"artifact-schema authority validation failed: {error}") from error

    _trace(trace, "phase_a.qualitative_report_schema")
    try:
        qualitative_summary = validate_qualitative_configuration(repo_root=root)
        qualitative_identity_sha256 = sha256_file(root / "evaluation_identities/e13_dfp_phase1a_qualitative_report.toml")
    except DfpPhase1aQualitativeReportError as error:
        raise DfpPhase1aPreflightError(f"qualitative-report schema validation failed: {error}") from error

    _trace(trace, "phase_a.subset_manifest")
    manifest_relative_path = MANIFEST_RELATIVE_PATH_BY_RUN_MODE[run_mode]
    manifest_path = root / manifest_relative_path
    class_count, class_names_digest, manifest = verify_frozen_subset_manifest(
        repo_root=root, dataset_key=dataset_key, protocol_identity_sha256=protocol_identity_sha256, manifest_path=manifest_path,
    )

    _trace(trace, "phase_a.parent_identity_hashes")
    # Every parent-identity hash (protocol<->artifact-schema<->qualitative-
    # report, matched_k11_k12/rwr/ade20k-source/ade20k-evaluator chains,
    # bridge checkpoint) is already relationally cross-checked inside
    # validate_artifact_schema_configuration above -- never re-implemented
    # here, only asserted as complete.
    if artifact_summary["protocol_identity_sha256"] != protocol_identity_sha256:
        raise DfpPhase1aPreflightError("artifact-schema authority's observed protocol hash disagrees with Phase A's own")
    if qualitative_summary["protocol_identity_sha256"] != protocol_identity_sha256:
        raise DfpPhase1aPreflightError("qualitative-report authority's protocol hash disagrees with Phase A's own")

    _trace(trace, "phase_a.select_run_binding")
    expected_selected_count = manifest["selected_count"]
    artifact_schema_identity_sha256 = sha256_file(root / "evaluation_identities/e13_dfp_phase1a_artifacts.toml")
    required_ancestor_commit = _protocol_required_ancestor(root)

    return PhaseAResult(
        run_mode=run_mode,
        dataset_key=dataset_key,
        repo_root=root,
        protocol_identity_sha256=protocol_identity_sha256,
        artifact_schema_identity_sha256=artifact_schema_identity_sha256,
        qualitative_report_identity_sha256=qualitative_identity_sha256,
        subset_manifest_sha256=sha256_file(manifest_path),
        subset_manifest_relative_path=manifest_relative_path,
        bridge_checkpoint_sha256=artifact_summary["bridge_checkpoint_sha256"],
        required_ancestor_commit=required_ancestor_commit,
        class_count=class_count,
        class_names_digest=class_names_digest,
        selected_ids=tuple(manifest["selected_ids"]),
        selected_ids_digest=manifest["selected_ids_digest"],
        expected_count=expected_selected_count,
        source_commit=_current_head_commit(root),
    )


def _protocol_required_ancestor(root: Path) -> str:
    from src.dfp_phase1a_identity import load_identity as load_protocol_identity

    protocol = load_protocol_identity(root / PROTOCOL_IDENTITY_RELATIVE_PATH, repo_root=root)
    return protocol["identity"]["required_ancestor_commit"]


def _current_head_commit(root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        raise DfpPhase1aPreflightError(f"cannot resolve HEAD commit: {result.stderr.strip()}")
    commit = result.stdout.strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise DfpPhase1aPreflightError(f"git rev-parse HEAD returned a non-canonical commit id: {commit!r}")
    return commit


# ---------------------------------------------------------------------------
# Phase B: filesystem/source validation. Still no model/CUDA entry point.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseBResult:
    dataset_root: Path
    image_relative_root: str
    annotation_relative_root: str
    image_suffix: str
    annotation_suffix: str
    evaluator_identity_name: str
    output_root: Path


def reconcile_selected_ids_to_physical_paths(
    *, dataset_key: str, dataset_root: Path, selected_ids: Sequence[str],
    image_relative_root: str, annotation_relative_root: str, image_suffix: str, annotation_suffix: str,
) -> None:
    """Reconcile every SELECTED canonical ID (never the full source
    population -- that is the separately-available, much heavier
    ``verify-source-binding`` full-rescan path) against its physical
    image/annotation file, using ONLY the caller-supplied suffix/relative-
    root policy (traced back to the dataset source identity, never
    invented here). A canonical ID never carries its own physical suffix
    -- ``dataset_root / image_relative_root / (canonical_id + suffix)`` is
    the only permitted construction; ``dataset_root / canonical_id`` alone
    is never used. Existence-only (no pixel decode) -- decoding is the
    separately-authorized source-manifest presence scan, not this
    lightweight per-run preflight. Fails closed on the first missing
    file, never partially reconciling a mix of present/absent IDs."""
    image_dir = dataset_root / image_relative_root
    annotation_dir = dataset_root / annotation_relative_root
    if not image_dir.is_dir():
        raise DfpPhase1aPreflightError(f"{dataset_key} image directory does not exist: {image_dir}")
    if not annotation_dir.is_dir():
        raise DfpPhase1aPreflightError(f"{dataset_key} annotation directory does not exist: {annotation_dir}")

    for canonical_id in selected_ids:
        image_path = image_dir / f"{canonical_id}{image_suffix}"
        annotation_path = annotation_dir / f"{canonical_id}{annotation_suffix}"
        if not image_path.is_file():
            raise DfpPhase1aPreflightError(
                f"{dataset_key} selected image {canonical_id!r} has no physical file at {image_path} "
                f"(canonical IDs never embed their own physical suffix -- this path is "
                f"image_relative_root/(canonical_id + image_suffix), never canonical_id alone)"
            )
        if not annotation_path.is_file():
            raise DfpPhase1aPreflightError(
                f"{dataset_key} selected annotation {canonical_id!r} has no physical file at {annotation_path}"
            )


def validate_output_root_safety(
    *, output_root: Path, forbidden_roots: Sequence[Path], probe_writeable: bool = False,
) -> Path:
    """Fail-closed output-root safety: not a regular file, does not
    overlap any forbidden (dataset/checkpoint/source) root, its parent
    exists and is a real directory (never silently created through an
    arbitrary ancestor chain), and it is not a symlink escaping its own
    containing directory. When ``probe_writeable`` is set, creates a
    uniquely named empty file directly in the resolved destination,
    fsyncs it, and removes it -- leaving zero residue and never touching
    any pre-existing file there."""
    output_root = Path(output_root)

    # The symlink-escape check MUST run against the ORIGINAL (unresolved)
    # argument -- .resolve() below fully dereferences every symlink
    # component, so checking `.is_symlink()` on its result can never be
    # true and would make this check permanently dead code.
    if output_root.is_symlink() and output_root.resolve().parent != output_root.parent.resolve():
        raise DfpPhase1aPreflightError(f"--output-root {output_root} is a symlink escaping its containing directory")

    resolved = output_root.resolve()

    if resolved.exists() and resolved.is_file():
        raise DfpPhase1aPreflightError(f"--output-root {resolved} is a regular file, not a directory")

    for forbidden in forbidden_roots:
        forbidden_resolved = Path(forbidden).resolve()
        if resolved == forbidden_resolved or forbidden_resolved in resolved.parents or resolved in forbidden_resolved.parents:
            raise DfpPhase1aPreflightError(f"--output-root {resolved} overlaps a protected path {forbidden_resolved}")

    parent = resolved.parent
    if parent.exists() and not parent.is_dir():
        raise DfpPhase1aPreflightError(f"--output-root parent {parent} is not a directory (blocker file)")
    if not parent.exists():
        raise DfpPhase1aPreflightError(f"--output-root parent {parent} does not exist; create it first (never silently created)")

    if probe_writeable:
        resolved.mkdir(parents=False, exist_ok=True)
        probe = resolved / f".dfp_phase1a_preflight_writeable_probe_{secrets.token_hex(8)}"
        try:
            with open(probe, "wb") as handle:
                handle.write(b"x")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise DfpPhase1aPreflightError(f"--output-root {resolved} is not writeable: {error}") from error
        finally:
            probe.unlink(missing_ok=True)

    return resolved


def run_phase_b(
    *, phase_a: PhaseAResult, data_root_override: str | None, output_root: Path,
    probe_output_writeable: bool = False, check_git: bool = True, trace: PreflightTrace | None = None,
) -> PhaseBResult:
    """Ordered Phase B: resolve the dataset root safely, verify the
    dataset source identity, reconcile the 500 selected canonical IDs
    against physical files, verify model/evaluator config-source-hash
    bindings, and validate output-root safety. Still no model
    construction, checkpoint deserialization, or CUDA entry point."""
    root = phase_a.repo_root
    dataset_key = phase_a.dataset_key

    _trace(trace, "phase_b.resolve_dataset_root")
    env_var = ENV_VAR_BY_RUN_MODE[phase_a.run_mode]
    try:
        raw_root = require_env_root(env_var, override=data_root_override)
    except DfpPhase1aSubsetManifestError as error:
        raise DfpPhase1aPreflightError(str(error)) from error

    _trace(trace, "phase_b.dataset_source_identity")
    try:
        if dataset_key == "coco_stuff":
            dataset_root = resolve_coco_stuff_root(raw_root)
            image_relative_root = "images/val2017"
            annotation_relative_root = "annotations/val2017"
            image_suffix = COCO_STUFF_IMAGE_SUFFIX
            annotation_suffix = COCO_STUFF_ANNOTATION_SUFFIX
        else:
            from src.ade20k_dataset_identity import load_identity as load_ade_source
            from src.ade20k_dataset_manifest import resolve_dataset_root as ade_resolve_root

            ade_identity = load_ade_source(root / DATASET_CONTRACTS["ade20k"]["source_identity_relative_path"], repo_root=root)
            dataset_root = ade_resolve_root(raw_root, ade_identity)
            image_relative_root = ade_identity["source"]["image_relative_root"]
            annotation_relative_root = ade_identity["source"]["annotation_relative_root"]
            image_suffix = ade_identity["source"]["image_suffix"]
            annotation_suffix = ade_identity["source"]["annotation_suffix"]
    except DfpPhase1aSubsetManifestError as error:
        raise DfpPhase1aPreflightError(str(error)) from error
    except Ade20kDatasetIdentityError as error:
        raise DfpPhase1aPreflightError(str(error)) from error

    _trace(trace, "phase_b.canonical_selected_ids")
    if len(phase_a.selected_ids) != len(set(phase_a.selected_ids)):
        raise DfpPhase1aPreflightError("phase A selected_ids contains a duplicate -- refusing to reconcile physical paths")

    _trace(trace, "phase_b.reconcile_physical_paths")
    reconcile_selected_ids_to_physical_paths(
        dataset_key=dataset_key, dataset_root=dataset_root, selected_ids=phase_a.selected_ids,
        image_relative_root=image_relative_root, annotation_relative_root=annotation_relative_root,
        image_suffix=image_suffix, annotation_suffix=annotation_suffix,
    )

    _trace(trace, "phase_b.model_evaluator_binding")
    evaluator_identity_name = validate_model_evaluator_binding(root, dataset_key, check_git=check_git)

    _trace(trace, "phase_b.bridge_checkpoint_raw_hash")
    # Already independently proven inside Phase A's artifact-schema
    # validation (raw-byte SHA-256 of the .pth file on disk, cross-checked
    # against both the artifact schema and the protocol identity) -- never
    # duplicated here, only asserted as already complete for this run.

    _trace(trace, "phase_b.output_root_safety")
    forbidden_roots = [dataset_root, root / "weights"]
    resolved_output_root = validate_output_root_safety(
        output_root=output_root, forbidden_roots=forbidden_roots, probe_writeable=probe_output_writeable,
    )

    return PhaseBResult(
        dataset_root=dataset_root,
        image_relative_root=image_relative_root,
        annotation_relative_root=annotation_relative_root,
        image_suffix=image_suffix,
        annotation_suffix=annotation_suffix,
        evaluator_identity_name=evaluator_identity_name,
        output_root=resolved_output_root,
    )


def validate_model_evaluator_binding(root: Path, dataset_key: str, *, check_git: bool) -> str:
    """Model/evaluator-config binding, delegated entirely to the already-
    committed parent-evaluator identity validators -- config source
    hashes, bridge/projection checkpoint raw-byte hash, and the full
    parent-identity chain. No torch import, no model construction, no
    checkpoint deserialization."""
    if dataset_key == "coco_stuff":
        from src.matched_k11_k12_identity import MatchedK11K12Error, validate_static_configuration as validate_matched

        try:
            summary = validate_matched(repo_root=root, check_checkpoint=False, check_git=check_git)
        except MatchedK11K12Error as error:
            raise DfpPhase1aPreflightError(f"COCO-Stuff model/evaluator binding failed: {error}") from error
        return summary["identity_name"]
    else:
        from src.ade20k_matched_evaluator_identity import (
            Ade20kMatchedEvaluatorIdentityError,
            validate_static_configuration as validate_ade_evaluator,
        )

        try:
            summary = validate_ade_evaluator(repo_root=root, check_git=check_git)
        except Ade20kMatchedEvaluatorIdentityError as error:
            raise DfpPhase1aPreflightError(f"ADE20K model/evaluator binding failed: {error}") from error
        return summary["identity_name"]


# ---------------------------------------------------------------------------
# Phase C: resume/result validation. Still no model/CUDA entry point.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseCResult:
    mode: str
    next_dataset_index: int
    completed_image_ids: tuple[str, ...]
    checkpoint_sha256: str | None
    per_image_stats_manifest_sha256: str | None
    arrays: Mapping[str, np.ndarray] | None


def run_phase_c(
    *, phase_a: PhaseAResult, mode: str, checkpoints_dir: Path, checkpoint_filename: str = "checkpoint.json",
    allow_fresh_overwrite: bool = False, trace: PreflightTrace | None = None,
) -> PhaseCResult:
    """Ordered Phase C: decide the exact fresh/resume mode, validate the
    checkpoint/per-image-stats/NPZ chain exactly once via
    :func:`dfp_phase1a_checkpoint.validate_resume_state`, reject an
    already-complete resume, and validate result-artifact policy. Never
    creates a checkpoint or result artifact."""
    _trace(trace, "phase_c.decide_fresh_or_resume")
    if mode not in ("fresh", "resume"):
        raise DfpPhase1aPreflightError(f"mode must be exactly 'fresh' or 'resume', got {mode!r}")

    checkpoints_dir = Path(checkpoints_dir)
    checkpoint_path = checkpoints_dir / checkpoint_filename
    result_path = checkpoints_dir / "result.json"

    _trace(trace, "phase_c.checkpoint_or_absence")
    if mode == "fresh":
        if checkpoint_path.exists() and not allow_fresh_overwrite:
            raise DfpPhase1aPreflightError(
                f"fresh mode requested but a checkpoint already exists at {checkpoint_path}; "
                "pass an explicit fresh-overwrite authorization or use resume mode"
            )
        if result_path.exists() and not allow_fresh_overwrite:
            raise DfpPhase1aPreflightError(
                f"fresh mode requested but a final result already exists at {result_path}; "
                "pass an explicit fresh-overwrite authorization or use resume mode"
            )
        _trace(trace, "phase_c.completed_id_prefix")
        _trace(trace, "phase_c.per_image_manifest_and_npz")
        _trace(trace, "phase_c.array_reconciliation")
        _trace(trace, "phase_c.reject_already_complete_resume")
        _trace(trace, "phase_c.result_policy")
        _trace(trace, "phase_c.build_context")
        return PhaseCResult(
            mode="fresh", next_dataset_index=0, completed_image_ids=(), checkpoint_sha256=None,
            per_image_stats_manifest_sha256=None, arrays=None,
        )

    # mode == "resume"
    if not checkpoint_path.exists():
        raise DfpPhase1aPreflightError(f"resume mode requested but no checkpoint exists at {checkpoint_path}")

    try:
        resume_result = validate_resume_state(checkpoints_dir=checkpoints_dir, checkpoint_filename=checkpoint_filename, repo_root=phase_a.repo_root)
    except DfpPhase1aCheckpointError as error:
        raise DfpPhase1aPreflightError(f"resume validation failed: {error}") from error
    _trace(trace, "phase_c.completed_id_prefix")
    _trace(trace, "phase_c.per_image_manifest_and_npz")
    _trace(trace, "phase_c.array_reconciliation")

    if resume_result["run_mode"] != phase_a.run_mode:
        raise DfpPhase1aPreflightError(
            f"existing checkpoint at {checkpoint_path} is bound to run_mode {resume_result['run_mode']!r}, "
            f"which disagrees with the requested run {phase_a.run_mode!r} -- refusing to resume across runs"
        )

    _trace(trace, "phase_c.reject_already_complete_resume")
    if not resume_result["resumable"]:
        raise DfpPhase1aPreflightError(
            f"checkpoint at {checkpoint_path} is already marked complete; resume mode requires an incomplete, "
            "resumable checkpoint (use fresh mode with an explicit overwrite authorization to redo a run)"
        )

    _trace(trace, "phase_c.result_policy")
    if result_path.exists():
        try:
            result_summary = verify_result_bundle(checkpoints_dir)
        except DfpPhase1aReportError as error:
            raise DfpPhase1aPreflightError(f"existing result at {result_path} failed verification: {error}") from error
        if result_summary["result"]["final"]:
            raise DfpPhase1aPreflightError(f"a final result already exists at {result_path}; resuming an already-finalized run is refused")

    completed_ids = tuple(resume_result["checkpoint"]["completed_image_ids"])
    arrays = resume_result["arrays"]
    frozen_arrays: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        array = np.array(array, copy=True)
        array.setflags(write=False)
        frozen_arrays[name] = array

    _trace(trace, "phase_c.build_context")
    return PhaseCResult(
        mode="resume",
        next_dataset_index=resume_result["next_dataset_index"],
        completed_image_ids=completed_ids,
        checkpoint_sha256=sha256_file(checkpoint_path),
        per_image_stats_manifest_sha256=sha256_file(checkpoints_dir / resume_result["checkpoint"]["per_image_stats_manifest_relative_path"]),
        arrays=MappingProxyType(frozen_arrays),
    )


# ---------------------------------------------------------------------------
# Immutable preflight context and top-level orchestration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightContext:
    """Everything a future execution driver needs, and nothing else: no
    model object, no dataset object, no CUDA device, no torch tensor, no
    mutable caller-owned dict/list. Every collection field is a tuple or a
    read-only ``MappingProxyType``; every array is marked non-writeable.
    Never re-reads the per-image NPZ after Phase C -- ``checkpoint_arrays``
    (when resuming) is exactly the array set Phase C already loaded once.
    Private resolved filesystem paths (dataset root, output root) live
    here in memory only; this object is never itself serialized into a
    portable manifest."""

    run_mode: str
    dataset_key: str
    expected_count: int
    class_count: int
    class_names_digest: str
    selected_ids: tuple[str, ...]
    selected_ids_digest: str
    protocol_identity_sha256: str
    artifact_schema_identity_sha256: str
    qualitative_report_identity_sha256: str
    subset_manifest_sha256: str
    bridge_checkpoint_sha256: str
    required_ancestor_commit: str
    source_commit: str
    evaluator_identity_name: str
    dataset_root: Path
    output_root: Path
    mode: str
    next_dataset_index: int
    completed_image_ids: tuple[str, ...]
    checkpoint_sha256: str | None
    per_image_stats_manifest_sha256: str | None
    checkpoint_arrays: Mapping[str, np.ndarray] | None
    completed_stage_names: tuple[str, ...] = field(default_factory=tuple)

    def to_summary_dict(self) -> Mapping[str, Any]:
        """Schema-safe, non-secret summary suitable for a PASS/READY
        message. Never includes a resolved private filesystem path or the
        raw checkpoint arrays."""
        return MappingProxyType(
            {
                "run_mode": self.run_mode,
                "mode": self.mode,
                "expected_count": self.expected_count,
                "class_count": self.class_count,
                "next_dataset_index": self.next_dataset_index,
                "completed_count": len(self.completed_image_ids),
                "evaluator_identity_name": self.evaluator_identity_name,
                "source_commit": self.source_commit,
            }
        )


def run_preflight(
    *, repo_root: Path, run_mode: str, mode: str, output_root: Path, data_root_override: str | None = None,
    checkpoints_dir: Path | None = None, checkpoint_filename: str = "checkpoint.json",
    allow_fresh_overwrite: bool = False, probe_output_writeable: bool = False, check_git: bool = True,
    trace: PreflightTrace | None = None,
) -> PreflightContext:
    """Run Phase A, then B, then C, in that exact order, and build the
    immutable :class:`PreflightContext`. Raises
    :class:`DfpPhase1aPreflightError` (or a chained domain error from a
    delegated validator) on the first violation -- never partially
    completes a phase and proceeds to the next."""
    phase_a = run_phase_a(repo_root=repo_root, run_mode=run_mode, check_git=check_git, trace=trace)
    phase_b = run_phase_b(
        phase_a=phase_a, data_root_override=data_root_override, output_root=output_root,
        probe_output_writeable=probe_output_writeable, check_git=check_git, trace=trace,
    )
    phase_c = run_phase_c(
        phase_a=phase_a, mode=mode, checkpoints_dir=Path(checkpoints_dir) if checkpoints_dir is not None else phase_b.output_root,
        checkpoint_filename=checkpoint_filename, allow_fresh_overwrite=allow_fresh_overwrite, trace=trace,
    )

    return PreflightContext(
        run_mode=phase_a.run_mode,
        dataset_key=phase_a.dataset_key,
        expected_count=phase_a.expected_count,
        class_count=phase_a.class_count,
        class_names_digest=phase_a.class_names_digest,
        selected_ids=phase_a.selected_ids,
        selected_ids_digest=phase_a.selected_ids_digest,
        protocol_identity_sha256=phase_a.protocol_identity_sha256,
        artifact_schema_identity_sha256=phase_a.artifact_schema_identity_sha256,
        qualitative_report_identity_sha256=phase_a.qualitative_report_identity_sha256,
        subset_manifest_sha256=phase_a.subset_manifest_sha256,
        bridge_checkpoint_sha256=phase_a.bridge_checkpoint_sha256,
        required_ancestor_commit=phase_a.required_ancestor_commit,
        source_commit=phase_a.source_commit,
        evaluator_identity_name=phase_b.evaluator_identity_name,
        dataset_root=phase_b.dataset_root,
        output_root=phase_b.output_root,
        mode=phase_c.mode,
        next_dataset_index=phase_c.next_dataset_index,
        completed_image_ids=phase_c.completed_image_ids,
        checkpoint_sha256=phase_c.checkpoint_sha256,
        per_image_stats_manifest_sha256=phase_c.per_image_stats_manifest_sha256,
        checkpoint_arrays=phase_c.arrays,
        completed_stage_names=trace.stages if trace is not None else (),
    )


__all__ = [
    "DfpPhase1aPreflightError",
    "ENV_VAR_BY_RUN_MODE",
    "MANIFEST_RELATIVE_PATH_BY_RUN_MODE",
    "PHASE_A_STAGES",
    "PHASE_B_STAGES",
    "PHASE_C_STAGES",
    "ALL_STAGES",
    "PreflightTrace",
    "dataset_key_for_run_mode",
    "PhaseAResult",
    "verify_frozen_subset_manifest",
    "run_phase_a",
    "PhaseBResult",
    "reconcile_selected_ids_to_physical_paths",
    "validate_output_root_safety",
    "validate_model_evaluator_binding",
    "run_phase_b",
    "PhaseCResult",
    "run_phase_c",
    "PreflightContext",
    "run_preflight",
]
