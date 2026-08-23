"""Static preflight and stability-gate result binding for the matched
k11/k12 finite-step power evaluator.

This module deliberately avoids importing MMCV, Torch, model code, or
dataset code. All evaluator-execution-contract settings (run-mode image
counts, checkpoint granularity, result/checkpoint schema names, the
stability-gate acceptance policy) live in the authoritative TOML at
:data:`IDENTITY_RELATIVE_PATH`; this module never hardcodes one of its
own. Scientific values (alpha, steps, crop, stride, affinity_power) are
never read from this identity at all -- callers load those directly from
the parent matched identity (:mod:`src.matched_k11_k12_identity`), so a
single authoritative source of truth is preserved end to end.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping

from src.matched_k11_k12_identity import load_identity as _load_matched_identity


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_k11_k12_power_evaluation.toml")

RUN_MODES = ("pilot20", "pilot100", "full")
CHECKPOINT_SCHEMA_NAME = "talk2dino-k11-k12-power-evaluation-checkpoint-v1"
PER_IMAGE_STATS_SCHEMA_NAME = "talk2dino-k11-k12-power-evaluation-per-image-stats-v1"
# Single source of truth for run-mode -> identity-field-name mapping, reused
# by the CLI, the report/verifier, and tests -- never redefined locally.
RUN_MODE_IMAGE_COUNT_KEYS = {"pilot20": "pilot20_image_count", "pilot100": "pilot100_image_count", "full": "full_image_count"}
RUN_MODE_SCHEMA_KEYS = {"pilot20": "pilot20_schema_name", "pilot100": "pilot100_schema_name", "full": "full_schema_name"}

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identity", "stability_gate",
        "run_modes", "checkpoint", "artifacts", "metrics", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "parent_identity": frozenset(
        {"matched_identity_path", "matched_identity_name", "matched_identity_sha256", "required_relationship"}
    ),
    "stability_gate": frozenset(
        {
            "gate_identity_path", "gate_identity_name", "gate_identity_sha256",
            "required_gate_windows_processed", "accepted_classifications", "required_relationship",
        }
    ),
    "run_modes": frozenset(
        {
            "pilot20_image_count", "pilot100_image_count", "full_image_count",
            "pilot20_schema_name", "pilot100_schema_name", "full_schema_name",
            "checkpoint_schema_name", "image_order_source", "window_order_source",
        }
    ),
    "checkpoint": frozenset({"granularity", "serializes_gpu_tensors"}),
    "artifacts": frozenset({"per_image_stats_manifest_schema_name", "per_image_stats_array_format"}),
    "metrics": frozenset(
        {"metric_names", "unit", "precision_source", "class_count", "paired_delta_field"}
    ),
    "prohibited": frozenset({"list"}),
}
# Permitted stability-gate classifications authorize *numerical execution*
# only -- they never claim k11 helps or hurts mIoU. TRUNCATION_SENSITIVE and
# INVALID must never appear here: a caller adding them to the TOML is
# rejected by load_identity below, not silently accepted.
SUPPORTED_ACCEPTED_CLASSIFICATIONS = (
    "CLEAR_MATCHED_SIGNAL",
    "NUMERICALLY_STABLE_BUT_EFFECT_NEAR_NOISE",
    "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE",
)
SUPPORTED_RUN_MODE_IMAGE_COUNTS = {"pilot20_image_count": 20, "pilot100_image_count": 100, "full_image_count": 5000}
SUPPORTED_IMAGE_ORDER_SOURCE = "coco_stuff_164k_validation_canonical_dataset_order"
SUPPORTED_WINDOW_ORDER_SOURCE = "sliding_window_geometry_slidingwindowplan_build_row_major_flat_index"
SUPPORTED_ARRAY_FORMAT = "npz_allow_pickle_false"
SUPPORTED_METRIC_NAMES = ("aAcc", "mIoU", "mAcc")
SUPPORTED_METRIC_UNIT = "percent_0_100"
SUPPORTED_PRECISION_SOURCE = "full_precision_area_statistics_from_mmseg_pre_eval"
SUPPORTED_CLASS_COUNT = 171

KERNEL_MODULE_RELATIVE_PATH = Path(
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py"
)
GRAPH_MODULE_RELATIVE_PATH = Path(
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py"
)


class K11K12PowerEvaluationError(ValueError):
    """Raised when the evaluator identity, a stability-result binding, or a
    result/checkpoint fails closed. Always fail closed: never silently
    substitute a default or proceed with an unverified assumption."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise K11K12PowerEvaluationError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise K11K12PowerEvaluationError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise K11K12PowerEvaluationError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise K11K12PowerEvaluationError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise K11K12PowerEvaluationError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise K11K12PowerEvaluationError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise K11K12PowerEvaluationError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise K11K12PowerEvaluationError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise K11K12PowerEvaluationError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise K11K12PowerEvaluationError(f"{label} elements must be exact strings")
    return tuple(value)


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise K11K12PowerEvaluationError(
            f"cannot load k11/k12 power evaluation identity {source}: {error}"
        ) from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise K11K12PowerEvaluationError("power evaluation identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-k11-k12-power-evaluation-identity-v1":
        raise K11K12PowerEvaluationError("unsupported power evaluation identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise K11K12PowerEvaluationError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    parent = identity["parent_identity"]
    _require_relative_path(parent["matched_identity_path"], "parent_identity.matched_identity_path")
    _require_exact_string(parent["matched_identity_name"], "parent_identity.matched_identity_name")
    _require_sha256(parent["matched_identity_sha256"], "parent_identity.matched_identity_sha256")
    _require_exact_string(parent["required_relationship"], "parent_identity.required_relationship")

    gate = identity["stability_gate"]
    _require_relative_path(gate["gate_identity_path"], "stability_gate.gate_identity_path")
    _require_exact_string(gate["gate_identity_name"], "stability_gate.gate_identity_name")
    _require_sha256(gate["gate_identity_sha256"], "stability_gate.gate_identity_sha256")
    _require_exact_int(gate["required_gate_windows_processed"], "stability_gate.required_gate_windows_processed", minimum=1)
    accepted = _require_exact_string_list(gate["accepted_classifications"], "stability_gate.accepted_classifications")
    if set(accepted) != set(SUPPORTED_ACCEPTED_CLASSIFICATIONS):
        raise K11K12PowerEvaluationError(
            "stability_gate.accepted_classifications must be exactly "
            f"{sorted(SUPPORTED_ACCEPTED_CLASSIFICATIONS)}; INVALID and TRUNCATION_SENSITIVE "
            "must never be accepted"
        )
    _require_exact_string(gate["required_relationship"], "stability_gate.required_relationship")

    modes = identity["run_modes"]
    for name, expected in SUPPORTED_RUN_MODE_IMAGE_COUNTS.items():
        count = _require_exact_int(modes[name], f"run_modes.{name}", minimum=1)
        if count != expected:
            raise K11K12PowerEvaluationError(f"run_modes.{name} must be exactly {expected}, observed {count!r}")
    for name in ("pilot20_schema_name", "pilot100_schema_name", "full_schema_name", "checkpoint_schema_name"):
        _require_exact_string(modes[name], f"run_modes.{name}")
    if modes["checkpoint_schema_name"] != CHECKPOINT_SCHEMA_NAME:
        raise K11K12PowerEvaluationError("run_modes.checkpoint_schema_name must match the implemented checkpoint schema")
    if modes["pilot20_schema_name"] == modes["pilot100_schema_name"] or modes["pilot20_schema_name"] == modes["full_schema_name"] or modes["pilot100_schema_name"] == modes["full_schema_name"]:
        raise K11K12PowerEvaluationError("pilot20/pilot100/full result schema names must all be distinct")
    if modes["image_order_source"] != SUPPORTED_IMAGE_ORDER_SOURCE:
        raise K11K12PowerEvaluationError("run_modes.image_order_source must match the implemented dataset order")
    if modes["window_order_source"] != SUPPORTED_WINDOW_ORDER_SOURCE:
        raise K11K12PowerEvaluationError("run_modes.window_order_source must match the implemented window order")

    checkpoint = identity["checkpoint"]
    _require_exact_string(checkpoint["granularity"], "checkpoint.granularity")
    if _require_exact_bool(checkpoint["serializes_gpu_tensors"], "checkpoint.serializes_gpu_tensors") is not False:
        raise K11K12PowerEvaluationError("checkpoint.serializes_gpu_tensors must be false")

    artifacts = identity["artifacts"]
    _require_exact_string(artifacts["per_image_stats_manifest_schema_name"], "artifacts.per_image_stats_manifest_schema_name")
    if artifacts["per_image_stats_manifest_schema_name"] != PER_IMAGE_STATS_SCHEMA_NAME:
        raise K11K12PowerEvaluationError("artifacts.per_image_stats_manifest_schema_name must match the implemented schema")
    if artifacts["per_image_stats_array_format"] != SUPPORTED_ARRAY_FORMAT:
        raise K11K12PowerEvaluationError(f"artifacts.per_image_stats_array_format must be {SUPPORTED_ARRAY_FORMAT!r}")

    metrics = identity["metrics"]
    names = _require_exact_string_list(metrics["metric_names"], "metrics.metric_names")
    if names != SUPPORTED_METRIC_NAMES:
        raise K11K12PowerEvaluationError(f"metrics.metric_names must be exactly {list(SUPPORTED_METRIC_NAMES)}")
    if metrics["unit"] != SUPPORTED_METRIC_UNIT:
        raise K11K12PowerEvaluationError(f"metrics.unit must be {SUPPORTED_METRIC_UNIT!r}")
    if metrics["precision_source"] != SUPPORTED_PRECISION_SOURCE:
        raise K11K12PowerEvaluationError(f"metrics.precision_source must be {SUPPORTED_PRECISION_SOURCE!r}")
    if _require_exact_int(metrics["class_count"], "metrics.class_count", minimum=1) != SUPPORTED_CLASS_COUNT:
        raise K11K12PowerEvaluationError(f"metrics.class_count must be exactly {SUPPORTED_CLASS_COUNT}")
    _require_exact_string(metrics["paired_delta_field"], "metrics.paired_delta_field")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise K11K12PowerEvaluationError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_parent_identity(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    parent = identity["parent_identity"]
    matched_path = root / parent["matched_identity_path"]
    try:
        matched_bytes = matched_path.read_bytes()
    except OSError as error:
        raise K11K12PowerEvaluationError(f"cannot read matched parent identity {matched_path}: {error}") from error
    if hashlib.sha256(matched_bytes).hexdigest() != parent["matched_identity_sha256"]:
        raise K11K12PowerEvaluationError("matched parent identity file SHA256 mismatch")
    matched_identity = _load_matched_identity(matched_path, repo_root=root)
    if matched_identity["identity"]["name"] != parent["matched_identity_name"]:
        raise K11K12PowerEvaluationError("matched parent identity name mismatch")
    return matched_identity


def _validate_stability_gate_identity_reference(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load and byte-verify the referenced stability-gate identity TOML
    (never trust the recorded SHA256 alone -- always re-hash the file on
    disk), and load it via the gate's own strict loader so a malformed or
    tampered gate identity is rejected the same way the gate itself would
    reject it."""
    from src.k11_k12_stability_gate_identity import load_identity as _load_gate_identity

    gate = identity["stability_gate"]
    gate_path = root / gate["gate_identity_path"]
    try:
        gate_bytes = gate_path.read_bytes()
    except OSError as error:
        raise K11K12PowerEvaluationError(f"cannot read stability gate identity {gate_path}: {error}") from error
    observed_sha256 = hashlib.sha256(gate_bytes).hexdigest()
    if observed_sha256 != gate["gate_identity_sha256"]:
        raise K11K12PowerEvaluationError("stability gate identity file SHA256 mismatch")
    gate_identity = _load_gate_identity(gate_path, repo_root=root)
    if gate_identity["identity"]["name"] != gate["gate_identity_name"]:
        raise K11K12PowerEvaluationError("stability gate identity name mismatch")
    if gate_identity["sample_selection"]["canonical_window_count"] != gate["required_gate_windows_processed"]:
        raise K11K12PowerEvaluationError(
            "stability_gate.required_gate_windows_processed disagrees with the gate identity's own "
            "sample_selection.canonical_window_count"
        )
    return gate_identity, observed_sha256


def validate_static_configuration(
    *, repo_root: Path | None = None, identity_path: Path | None = None, check_git: bool = True
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")
    matched_identity = _validate_parent_identity(root, identity)
    gate_identity, _ = _validate_stability_gate_identity_reference(root, identity)

    from src.matched_k11_k12_identity import validate_static_configuration as _matched_preflight

    matched_result = _matched_preflight(repo_root=root, check_git=check_git)

    return {
        "identity_name": identity["identity"]["name"],
        "matched_identity": matched_result["identity_name"],
        "stability_gate_identity": gate_identity["identity"]["name"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
        "pilot20_image_count": identity["run_modes"]["pilot20_image_count"],
        "pilot100_image_count": identity["run_modes"]["pilot100_image_count"],
        "full_image_count": identity["run_modes"]["full_image_count"],
    }


def _hash_blob_at_commit(root: Path, commit: str, relative_path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path.as_posix()}"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise K11K12PowerEvaluationError(
            f"cannot read {relative_path} at commit {commit}: {detail}"
        )
    return hashlib.sha256(result.stdout).hexdigest()


def validate_stability_result_binding(
    stability_result_path: Path, *, identity: Mapping[str, Any], repo_root: Path, check_git: bool = True
) -> dict[str, Any]:
    """Read, structurally verify, and relationally bind ``--stability-result``
    before any dataset/model construction is permitted.

    Reuses the stability gate's own strict result verifier
    (:func:`src.k11_k12_stability_report.verify_record`) -- never
    reimplements that schema/threshold logic -- then applies the
    additional power-evaluator-specific acceptance policy: an explicit
    ``failure_reason is None`` requirement (the gate's own verifier
    permits, but does not require, ``None``), a permitted-classification
    allowlist that excludes ``INVALID``/``TRUNCATION_SENSITIVE``, a
    matched-identity cross-binding, and byte-identical finite-step-kernel /
    graph-construction source provenance between the commit the gate ran
    at and the commit this evaluator is running at.
    """
    from src.k11_k12_stability_gate_identity import parse_structured_document
    from src.k11_k12_stability_report import verify_record

    root = Path(repo_root)
    gate_identity, gate_identity_sha256 = _validate_stability_gate_identity_reference(root, identity)

    try:
        record = parse_structured_document(Path(stability_result_path), label="stability-gate result")
        verify_record(record, gate_identity, identity_sha256=gate_identity_sha256)
    except Exception as error:  # noqa: BLE001 -- re-raise as our own error type, fail closed
        raise K11K12PowerEvaluationError(f"--stability-result failed gate verification: {error}") from error

    if record.get("failure_reason") is not None:
        raise K11K12PowerEvaluationError("--stability-result.failure_reason must be null")

    classification = record["gate_classification"]
    accepted = identity["stability_gate"]["accepted_classifications"]
    if classification not in accepted:
        raise K11K12PowerEvaluationError(
            f"--stability-result gate_classification {classification!r} is not permitted for evaluation "
            f"(permitted: {list(accepted)}); this classification only authorizes numerical execution when "
            "permitted, and never claims efficacy either way"
        )

    if record["windows_processed"] != identity["stability_gate"]["required_gate_windows_processed"]:
        raise K11K12PowerEvaluationError("--stability-result.windows_processed does not match the required gate window count")

    if record["matched_identity_sha256"] != identity["parent_identity"]["matched_identity_sha256"]:
        raise K11K12PowerEvaluationError(
            "--stability-result.matched_identity_sha256 does not match this evaluator's own parent matched identity"
        )

    gate_commit = record["git_commit"]
    if check_git:
        _check_git_ancestry(root, gate_commit, label="stability-result.git_commit")

    current_kernel_sha256 = hashlib.sha256((root / KERNEL_MODULE_RELATIVE_PATH).read_bytes()).hexdigest()
    gate_kernel_sha256 = _hash_blob_at_commit(root, gate_commit, KERNEL_MODULE_RELATIVE_PATH)
    if current_kernel_sha256 != gate_kernel_sha256:
        raise K11K12PowerEvaluationError(
            "finite-step kernel source has changed since the stability-gate run "
            f"(gate commit {gate_commit}); refusing to evaluate against a kernel the gate never validated"
        )

    current_graph_sha256 = hashlib.sha256((root / GRAPH_MODULE_RELATIVE_PATH).read_bytes()).hexdigest()
    gate_graph_sha256 = _hash_blob_at_commit(root, gate_commit, GRAPH_MODULE_RELATIVE_PATH)
    if current_graph_sha256 != gate_graph_sha256:
        raise K11K12PowerEvaluationError(
            "graph construction source has changed since the stability-gate run "
            f"(gate commit {gate_commit}); refusing to evaluate against graph construction the gate never validated"
        )

    return {
        "stability_result_sha256": hashlib.sha256(Path(stability_result_path).read_bytes()).hexdigest(),
        "stability_schema": record["schema"],
        "gate_classification": classification,
        "gate_git_commit": gate_commit,
        "gate_identity_sha256": gate_identity_sha256,
        "finite_step_kernel_sha256": current_kernel_sha256,
        "graph_construction_sha256": current_graph_sha256,
    }


__all__ = [
    "CHECKPOINT_SCHEMA_NAME",
    "IDENTITY_RELATIVE_PATH",
    "K11K12PowerEvaluationError",
    "PER_IMAGE_STATS_SCHEMA_NAME",
    "RUN_MODES",
    "RUN_MODE_IMAGE_COUNT_KEYS",
    "RUN_MODE_SCHEMA_KEYS",
    "SUPPORTED_ACCEPTED_CLASSIFICATIONS",
    "SUPPORTED_ARRAY_FORMAT",
    "SUPPORTED_CLASS_COUNT",
    "SUPPORTED_METRIC_NAMES",
    "SUPPORTED_METRIC_UNIT",
    "SUPPORTED_PRECISION_SOURCE",
    "SUPPORTED_RUN_MODE_IMAGE_COUNTS",
    "GRAPH_MODULE_RELATIVE_PATH",
    "KERNEL_MODULE_RELATIVE_PATH",
    "load_identity",
    "repository_root",
    "validate_stability_result_binding",
    "validate_static_configuration",
]
