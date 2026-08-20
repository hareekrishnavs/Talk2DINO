"""Static preflight and strict result verification for canonical E3 RWR."""

from __future__ import annotations

import json
import hashlib
import math
import re
import subprocess
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping

from src.e3_evaluation_identity import (
    load_identity as _load_e3_identity,
    resolve_complete_e3_configuration as _resolve_complete_e3_configuration,
    validate_complete_e3_configuration as _validate_complete_e3_configuration,
    validate_static_configuration as _validate_e3,
)
from src.typed_configuration import (
    TYPED_CONFIGURATION_ENCODING,
    TypedConfigurationError,
    clone_configuration,
    first_typed_difference,
    typed_configuration_sha256,
)


IDENTITY_RELATIVE_PATH = Path(
    "evaluation_identities/e3_canonical_directed_rwr.toml"
)
RESULT_PREFIX = "TALK2DINO_RWR_RESULT "
RESULT_FORMAT_VERSION = "talk2dino-canonical-rwr-result-v2"
SUPPORTED_CANONICAL_GRAPH_MODE = "directed_topk"
SUPPORTED_CANONICAL_SOLVER_METHOD = "cgls"
RESULT_KEYS = frozenset(
    {
        "format_version",
        "identity_name",
        "image_count",
        "class_count",
        "aAcc",
        "mIoU",
        "mAcc",
        "metrics_precision",
        "gain_over_e3_miou",
        "rwr_enabled",
        "alpha",
        "top_k",
        "affinity_power",
        "graph_mode",
        "solver",
        "solver_rtol",
        "solver_atol",
        "solver_max_iterations",
        "crop",
        "stride",
        "pamr",
        "background_class",
        "config_path",
        "config_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "source_e10_commit",
        "cache_source_commit",
        "cache_manifest_sha256",
        "cache_manifest_evidence",
        "cache_manifest_attestation_commit",
        "cache_manifest_attestation_path",
        "cache_manifest_attestation_blob_sha256",
        "cache_manifest_archived",
        "historical_cache_used_by_current_run",
        "source_git_commit",
        "source_git_branch",
        "source_git_dirty",
        "gpu_model",
        "torch_version",
        "cuda_version",
        "elapsed_seconds",
        "solver_summary",
    }
)
# v3: current structured-result version. Adds explicit metric-source
# provenance (rejecting rounded/approximate metrics claiming full
# precision) and an optional, separately-scoped parity-check solver
# summary. v2 remains fully, unconditionally verifiable for historical
# logs/artifacts -- see _verify_record_v2, byte-identical to the original
# verify_record body.
RESULT_FORMAT_VERSION_V3 = "talk2dino-canonical-rwr-result-v3"
FULL_PRECISION_METRIC_SOURCE = "full_precision_area_statistics_from_mmseg_pre_eval"
SUPPORTED_RESULT_FORMAT_VERSIONS = (RESULT_FORMAT_VERSION, RESULT_FORMAT_VERSION_V3)
RESULT_KEYS_V3 = RESULT_KEYS | {"metric_source", "parity_solver_summary"}
SUMMARY_KEYS = frozenset(
    {
        "window_count",
        "converged_window_count",
        "total_iterations",
        "minimum_iterations",
        "maximum_iterations",
        "total_restarts",
        "nonzero_restart_windows",
        "total_residual_replacements",
        "total_fallback_rows",
        "maximum_scaled_residual",
    }
)
IDENTITY_SECTION_KEYS = {
    "dataset": frozenset({"name", "images", "classes", "background_class"}),
    "evaluation": frozenset(
        {
            "mode",
            "crop",
            "stride",
            "resize_dimension",
            "template",
            "pamr",
            "uniform_stitching",
        }
    ),
    "rwr": frozenset(
        {
            "enabled",
            "graph_mode",
            "alpha",
            "top_k",
            "affinity_power",
            "self_excluded",
            "zero_row_self_loop",
            "score_stage",
        }
    ),
    "solver": frozenset(
        {"method", "rtol", "atol", "max_iterations", "silent_fallback"}
    ),
    "checkpoint": frozenset(
        {
            "projection_class",
            "projection_name",
            "projection_model",
            "path",
            "sha256",
            "loading_required",
        }
    ),
    "expected_metrics": frozenset(
        {"e3_mIoU", "rwr_aAcc", "rwr_mIoU", "rwr_mAcc", "gain_mIoU"}
    ),
    "acceptance": frozenset(
        {
            "structured_absolute_mIoU",
            "minimum_metric_decimal_places",
            "rounded_mIoU",
        }
    ),
    "historical": frozenset(
        {
            "metrics_commit", "metrics_path", "metrics_blob", "metrics_sha256",
            "metrics_record", "baseline_metrics_record", "iteration_stats_record",
            "cache_evidence_commit", "cache_evidence_path", "cache_evidence_blob",
            "cache_evidence_sha256", "graph_commit", "graph_path", "graph_blob",
            "graph_sha256", "solver_commit", "solver_path", "solver_blob",
            "solver_sha256", "config_commit", "config_path", "config_blob",
            "config_sha256", "eval_base_commit", "eval_base_path",
            "eval_base_blob", "eval_base_sha256",
        }
    ),
    "resolved_configuration": frozenset(
        {
            "encoding_version", "full_sha256", "without_rwr_sha256",
            "evaluation_sha256",
        }
    ),
}
CONFIGURATION_SOURCE_KEYS = frozenset(
    {"order", "role", "path", "sha256", "git_blob"}
)
HISTORICAL_METRIC_RECORD_KEYS = frozenset(
    {
        "aAcc", "mIoU", "mAcc", "per_class_iou", "per_class_accuracy",
        "intersection", "union", "predicted_pixels", "ground_truth_pixels",
        "evaluated_images", "runtime_seconds", "peak_cpu_ram_bytes",
        "peak_gpu_bytes",
    }
)
HISTORICAL_EVIDENCE_KEYS = frozenset(
    {
        "cache_manifest_sha256", "class_order_sha256", "command",
        "dataset_config_sha256", "experiment", "format_version", "git_commit",
        "git_dirty", "invocation", "payload", "seed",
    }
)
HISTORICAL_EVIDENCE_PAYLOAD_KEYS = frozenset(
    {
        "alpha_grid", "kappa_k_baked_at_construction_time",
        "kappa_k_sweep_limitation", "label", "optimum", "rows",
        "steps_grid", "warning",
    }
)
HISTORICAL_EVIDENCE_ROW_KEYS = frozenset(
    {
        "aAcc", "alpha", "evaluated_images", "ground_truth_pixels",
        "intersection", "mAcc", "mIoU", "peak_cpu_ram_bytes",
        "peak_gpu_bytes", "per_class_accuracy", "per_class_iou",
        "predicted_pixels", "runtime_seconds", "steps", "union",
    }
)


class RWRReproductionError(ValueError):
    """Raised when canonical configuration or result identity fails closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise RWRReproductionError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RWRReproductionError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise RWRReproductionError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise RWRReproductionError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise RWRReproductionError(f"{label} must be an exact float")
    if not math.isfinite(value):
        raise RWRReproductionError(f"{label} must be finite")
    return value


def _require_exact_integer_pair(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise RWRReproductionError(f"{label} must be an exact two-element list")
    if any(type(item) is not int for item in value):
        raise RWRReproductionError(f"{label} elements must be exact integers")
    return value[0], value[1]


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise RWRReproductionError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise RWRReproductionError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise RWRReproductionError(f"{label} must be a safe repository-relative path")
    return token


def _require_json_float(value: Any, label: str) -> float:
    if not isinstance(value, Decimal):
        raise RWRReproductionError(f"{label} must be a JSON floating-point number")
    if not value.is_finite():
        raise RWRReproductionError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise RWRReproductionError(f"{label} is outside finite float range")
    return result


def _require_json_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    return _require_exact_int(value, label, minimum=minimum)


def _require_json_bool(value: Any, label: str) -> bool:
    return _require_exact_bool(value, label)


def validate_supported_capabilities(identity: Mapping[str, Any]) -> None:
    """Fail closed when an identity requests an unsupported implementation."""
    graph_mode = identity["rwr"]["graph_mode"]
    if type(graph_mode) is not str or graph_mode != SUPPORTED_CANONICAL_GRAPH_MODE:
        raise RWRReproductionError(
            "rwr.graph_mode capability mismatch: expected "
            f"{SUPPORTED_CANONICAL_GRAPH_MODE!r}, observed {graph_mode!r}"
        )
    solver_method = identity["solver"]["method"]
    if (
        type(solver_method) is not str
        or solver_method != SUPPORTED_CANONICAL_SOLVER_METHOD
    ):
        raise RWRReproductionError(
            "solver.method capability mismatch: expected "
            f"{SUPPORTED_CANONICAL_SOLVER_METHOD!r}, observed {solver_method!r}"
        )


def load_identity(
    path: Path | None = None, *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RWRReproductionError(
            f"cannot load canonical RWR identity {source}: {error}"
        ) from error
    expected = {
        "format_version",
        "identity_name",
        "e3_identity_path",
        "source_e10_commit",
        "cache_source_commit",
        "cache_manifest_sha256",
        "cache_manifest_evidence",
        "cache_manifest_attestation_commit",
        "cache_manifest_attestation_path",
        "cache_manifest_attestation_blob_sha256",
        "cache_manifest_archived",
        "historical_cache_used_by_current_run",
        "canonical_config_path",
        "canonical_config_sha256",
        "dataset",
        "evaluation",
        "rwr",
        "solver",
        "checkpoint",
        "expected_metrics",
        "acceptance",
        "historical",
        "resolved_configuration",
        "configuration_sources",
    }
    archived = identity.get("cache_manifest_archived")
    if archived is True:
        expected.add("archived_cache_manifest")
    if set(identity) != expected:
        raise RWRReproductionError("RWR identity has an unexpected schema")
    _require_exact_string(identity["format_version"], "RWR format_version")
    if identity["format_version"] != "talk2dino-rwr-reproduction-identity-v3":
        raise RWRReproductionError("unsupported RWR identity version")
    _require_exact_string(identity["identity_name"], "RWR identity name")
    _require_relative_path(identity["e3_identity_path"], "E3 identity path")
    _require_relative_path(identity["canonical_config_path"], "canonical config path")
    _require_sha256(identity["canonical_config_sha256"], "canonical config SHA256")
    for section, expected_keys in IDENTITY_SECTION_KEYS.items():
        value = identity.get(section)
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise RWRReproductionError(
                f"RWR identity {section} has an unexpected schema"
            )
    for name in ("source_e10_commit", "cache_source_commit"):
        _require_git_identity(identity[name], name)
    _require_sha256(identity["cache_manifest_sha256"], "cache manifest SHA256")
    _require_exact_string(identity["cache_manifest_evidence"], "cache manifest evidence")
    if identity["cache_manifest_evidence"] != "committed_attestation":
        raise RWRReproductionError("cache manifest evidence type mismatch")
    _require_git_identity(
        identity["cache_manifest_attestation_commit"],
        "cache manifest attestation commit",
    )
    _require_relative_path(
        identity["cache_manifest_attestation_path"],
        "cache manifest attestation path",
    )
    _require_sha256(
        identity["cache_manifest_attestation_blob_sha256"],
        "cache manifest attestation blob SHA256",
    )
    if identity["cache_manifest_attestation_commit"] != identity["source_e10_commit"]:
        raise RWRReproductionError("cache manifest attestation commit mismatch")
    _require_exact_bool(identity["cache_manifest_archived"], "cache_manifest_archived")
    _require_exact_bool(
        identity["historical_cache_used_by_current_run"],
        "historical_cache_used_by_current_run",
    )
    if identity["historical_cache_used_by_current_run"]:
        raise RWRReproductionError(
            "current direct reproduction must not use the historical cache"
        )
    if identity["cache_manifest_archived"]:
        archive = identity.get("archived_cache_manifest")
        expected_archive = {"commit", "path", "blob", "sha256"}
        if not isinstance(archive, Mapping) or set(archive) != expected_archive:
            raise RWRReproductionError(
                "archived cache manifest identity is incomplete"
            )
    elif "archived_cache_manifest" in identity:
        raise RWRReproductionError(
            "unarchived cache manifest must not declare archive fields"
        )
    dataset = identity["dataset"]
    _require_exact_string(dataset["name"], "dataset name")
    _require_exact_int(dataset["images"], "dataset images", minimum=1)
    _require_exact_int(dataset["classes"], "dataset classes", minimum=1)
    _require_exact_bool(dataset["background_class"], "dataset background_class")
    evaluation = identity["evaluation"]
    for name in ("mode", "template"):
        _require_exact_string(evaluation[name], f"evaluation {name}")
    _require_exact_integer_pair(evaluation["crop"], "evaluation crop")
    _require_exact_integer_pair(evaluation["stride"], "evaluation stride")
    _require_exact_int(
        evaluation["resize_dimension"], "evaluation resize_dimension", minimum=1
    )
    _require_exact_bool(evaluation["pamr"], "evaluation pamr")
    _require_exact_bool(
        evaluation["uniform_stitching"], "evaluation uniform_stitching"
    )
    rwr = identity["rwr"]
    _require_exact_bool(rwr["enabled"], "RWR enabled")
    _require_exact_string(rwr["score_stage"], "RWR score_stage")
    _require_exact_float(rwr["alpha"], "RWR alpha")
    if not 0 <= rwr["alpha"] < 1:
        raise RWRReproductionError("RWR alpha must satisfy 0 <= alpha < 1")
    _require_exact_int(rwr["top_k"], "RWR top_k", minimum=1)
    if _require_exact_float(rwr["affinity_power"], "RWR affinity_power") <= 0:
        raise RWRReproductionError("RWR affinity_power must be positive")
    _require_exact_bool(rwr["self_excluded"], "RWR self_excluded")
    _require_exact_bool(rwr["zero_row_self_loop"], "RWR zero_row_self_loop")
    solver = identity["solver"]
    validate_supported_capabilities(identity)
    for name in ("rtol", "atol"):
        if _require_exact_float(solver[name], f"solver {name}") < 0:
            raise RWRReproductionError(f"solver {name} must be non-negative")
    _require_exact_int(
        solver["max_iterations"], "solver max_iterations", minimum=1
    )
    _require_exact_bool(solver["silent_fallback"], "solver silent_fallback")
    checkpoint = identity["checkpoint"]
    for name in ("projection_class", "projection_name", "projection_model"):
        _require_exact_string(checkpoint[name], f"checkpoint {name}")
    _require_relative_path(checkpoint["path"], "checkpoint path")
    _require_sha256(checkpoint["sha256"], "checkpoint SHA256")
    _require_exact_bool(checkpoint["loading_required"], "checkpoint loading_required")
    tolerance = _require_exact_float(
        identity["acceptance"]["structured_absolute_mIoU"],
        "structured mIoU tolerance",
    )
    if not 0 < tolerance <= 0.005:
        raise RWRReproductionError("structured mIoU tolerance exceeds 0.005")
    for name in ("e3_mIoU", "rwr_aAcc", "rwr_mIoU", "rwr_mAcc", "gain_mIoU"):
        _require_exact_float(identity["expected_metrics"][name], f"expected {name}")
    expected_gain = (
        identity["expected_metrics"]["rwr_mIoU"]
        - identity["expected_metrics"]["e3_mIoU"]
    )
    if abs(expected_gain - identity["expected_metrics"]["gain_mIoU"]) > 1e-12:
        raise RWRReproductionError("RWR identity expected gain is inconsistent")
    _require_exact_int(
        identity["acceptance"]["minimum_metric_decimal_places"],
        "minimum metric decimal places",
        minimum=1,
    )
    _require_exact_float(identity["acceptance"]["rounded_mIoU"], "rounded mIoU")
    historical = identity["historical"]
    for name in ("metrics_record", "baseline_metrics_record", "iteration_stats_record"):
        _require_exact_string(historical[name], f"historical {name}")
    if len(
        {
            historical["metrics_record"], historical["baseline_metrics_record"],
            historical["iteration_stats_record"],
        }
    ) != 3:
        raise RWRReproductionError("historical metrics record names must be distinct")
    for prefix in ("metrics", "cache_evidence", "graph", "solver", "config", "eval_base"):
        commit = historical[f"{prefix}_commit"]
        path_value = historical[f"{prefix}_path"]
        blob = historical[f"{prefix}_blob"]
        digest = historical[f"{prefix}_sha256"]
        _require_git_identity(commit, f"historical {prefix} commit")
        _require_relative_path(path_value, f"historical {prefix} Git path")
        _require_git_identity(blob, f"historical {prefix} blob identity")
        _require_sha256(digest, f"historical {prefix} SHA256")
    if historical["metrics_commit"] != identity["source_e10_commit"]:
        raise RWRReproductionError("historical metrics commit mismatch")
    if historical["cache_evidence_commit"] != identity["source_e10_commit"]:
        raise RWRReproductionError("historical cache evidence commit mismatch")
    if identity["cache_manifest_attestation_path"] != historical["cache_evidence_path"]:
        raise RWRReproductionError("cache manifest attestation path mismatch")
    if identity["cache_manifest_attestation_blob_sha256"] != historical["cache_evidence_sha256"]:
        raise RWRReproductionError("cache manifest attestation blob SHA256 mismatch")
    for prefix in ("graph", "config", "eval_base"):
        if historical[f"{prefix}_commit"] != identity["cache_source_commit"]:
            raise RWRReproductionError(f"historical {prefix} commit mismatch")
    if historical["solver_commit"] != identity["source_e10_commit"]:
        raise RWRReproductionError("historical solver commit mismatch")
    if identity["cache_manifest_archived"]:
        archive = identity["archived_cache_manifest"]
        _require_git_identity(archive["commit"], "archived manifest commit")
        _require_relative_path(archive["path"], "archived manifest Git path")
        _require_git_identity(archive["blob"], "archived manifest blob")
        _require_sha256(archive["sha256"], "archived manifest SHA256")
        if archive["sha256"] != identity["cache_manifest_sha256"]:
            raise RWRReproductionError("archived manifest SHA256 identity mismatch")
    resolved = identity["resolved_configuration"]
    _require_exact_string(resolved["encoding_version"], "typed encoding version")
    if resolved["encoding_version"] != TYPED_CONFIGURATION_ENCODING:
        raise RWRReproductionError("unsupported typed configuration encoding")
    for name in (
        "full_sha256", "without_rwr_sha256", "evaluation_sha256"
    ):
        _require_sha256(resolved[name], f"resolved_configuration.{name}")
    sources = identity["configuration_sources"]
    if type(sources) is not list or not sources:
        raise RWRReproductionError(
            "RWR configuration_sources must be a non-empty array"
        )
    for index, source_record in enumerate(sources):
        if not isinstance(source_record, Mapping) or set(source_record) != CONFIGURATION_SOURCE_KEYS:
            raise RWRReproductionError(
                f"RWR configuration_sources[{index}] has an unexpected schema"
            )
        if _require_exact_int(
            source_record["order"], f"RWR source {index} order", minimum=0
        ) != index:
            raise RWRReproductionError("RWR configuration source order is not contiguous")
        _require_exact_string(source_record["role"], f"RWR source {index} role")
        _require_relative_path(source_record["path"], f"RWR source {index} path")
        _require_sha256(source_record["sha256"], f"RWR source {index} SHA256")
        _require_git_identity(source_record["git_blob"], f"RWR source {index} Git blob")
    return identity


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RWRReproductionError(
            f"Git provenance check failed: {error.stderr.strip()}"
        ) from error


ArtifactReader = Callable[[str, str], bytes]


def _git_artifact_reader(root: Path) -> ArtifactReader:
    def read(commit: str, path: str) -> bytes:
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise RWRReproductionError("historical commit is not a full Git SHA")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in path:
            raise RWRReproductionError("historical Git path is unsafe")
        try:
            return subprocess.run(
                ["git", "-C", str(root), "show", f"{commit}:{path}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise RWRReproductionError(
                f"cannot read committed artifact {commit}:{path}: "
                f"{error.stderr.decode(errors='replace').strip()}"
            ) from error
    return read


def _validate_artifact(
    reader: ArtifactReader, historical: Mapping[str, Any], prefix: str
) -> bytes:
    commit = historical[f"{prefix}_commit"]
    path = historical[f"{prefix}_path"]
    data = reader(commit, path)
    digest = hashlib.sha256(data).hexdigest()
    if digest != historical[f"{prefix}_sha256"]:
        raise RWRReproductionError(f"historical {prefix} SHA256 mismatch")
    # Git blob IDs are SHA1 over the canonical blob header and bytes.
    header = f"blob {len(data)}\0".encode("ascii")
    blob = hashlib.sha1(header + data).hexdigest()
    if blob != historical[f"{prefix}_blob"]:
        raise RWRReproductionError(f"historical {prefix} blob identity mismatch")
    return data


def _require_closed_mapping(
    value: Any, expected_keys: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RWRReproductionError(f"{label} has an unexpected schema")
    return value


def _validate_json_float_array(
    value: Any, label: str, *, length: int | None = None
) -> None:
    if type(value) is not list or (length is not None and len(value) != length):
        raise RWRReproductionError(f"{label} must be an exact JSON array")
    for index, item in enumerate(value):
        _require_json_float(item, f"{label}[{index}]")


def _validate_json_int_array(
    value: Any, label: str, *, length: int | None = None
) -> None:
    if type(value) is not list or (length is not None and len(value) != length):
        raise RWRReproductionError(f"{label} must be an exact JSON array")
    for index, item in enumerate(value):
        _require_json_int(item, f"{label}[{index}]", minimum=0)


def _validate_historical_metric_record(
    value: Any, identity: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    record = _require_closed_mapping(value, HISTORICAL_METRIC_RECORD_KEYS, label)
    for name in ("aAcc", "mIoU", "mAcc", "runtime_seconds"):
        _require_json_float(record[name], f"{label}.{name}")
    _require_json_int(record["evaluated_images"], f"{label}.evaluated_images", minimum=1)
    for name in ("peak_cpu_ram_bytes", "peak_gpu_bytes"):
        _require_json_int(record[name], f"{label}.{name}", minimum=0)
    classes = identity["dataset"]["classes"]
    for name in ("per_class_iou", "per_class_accuracy"):
        _validate_json_float_array(record[name], f"{label}.{name}", length=classes)
    for name in ("intersection", "union", "predicted_pixels", "ground_truth_pixels"):
        _validate_json_int_array(record[name], f"{label}.{name}", length=classes)
    return record


def _validate_historical_metrics_document(
    metrics: Any, identity: Mapping[str, Any]
) -> Mapping[str, Any]:
    historical = identity["historical"]
    metric_keys = frozenset(
        {
            historical["metrics_record"], historical["baseline_metrics_record"],
            historical["iteration_stats_record"],
        }
    )
    document = _require_closed_mapping(metrics, metric_keys, "historical metrics")
    for key in sorted(
        {historical["metrics_record"], historical["baseline_metrics_record"]}
    ):
        _validate_historical_metric_record(document[key], identity, f"historical metrics.{key}")
    stats = _require_closed_mapping(
        document[historical["iteration_stats_record"]],
        frozenset({"n_windows", "min", "mean", "max"}),
        "historical metrics.cgls_iteration_stats",
    )
    _require_json_int(stats["n_windows"], "historical cgls n_windows", minimum=1)
    _require_json_int(stats["min"], "historical cgls min", minimum=0)
    _require_json_float(stats["mean"], "historical cgls mean")
    _require_json_int(stats["max"], "historical cgls max", minimum=0)
    return document


def _validate_historical_evidence_document(
    evidence: Any, identity: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    document = _require_closed_mapping(
        evidence, HISTORICAL_EVIDENCE_KEYS, "historical cache evidence"
    )
    for name in (
        "cache_manifest_sha256", "class_order_sha256", "dataset_config_sha256"
    ):
        _require_sha256(document[name], f"historical evidence {name}")
    for name in ("command", "experiment", "format_version", "invocation"):
        _require_exact_string(document[name], f"historical evidence {name}")
    _require_git_identity(document["git_commit"], "historical evidence git_commit")
    _require_json_bool(document["git_dirty"], "historical evidence git_dirty")
    if document["seed"] is not None:
        raise RWRReproductionError("historical evidence seed must be null")
    payload = _require_closed_mapping(
        document["payload"],
        HISTORICAL_EVIDENCE_PAYLOAD_KEYS,
        "historical cache evidence payload",
    )
    _validate_json_float_array(payload["alpha_grid"], "historical alpha_grid")
    _validate_json_int_array(payload["steps_grid"], "historical steps_grid")
    for name in ("kappa_k_sweep_limitation", "label", "warning"):
        _require_exact_string(payload[name], f"historical payload {name}")
    baked = _require_closed_mapping(
        payload["kappa_k_baked_at_construction_time"],
        frozenset({"affinity_power", "knn_k"}),
        "historical baked graph settings",
    )
    _require_json_bool(baked["affinity_power"], "historical baked affinity_power")
    _require_json_bool(baked["knn_k"], "historical baked knn_k")
    rows = payload["rows"]
    if type(rows) is not list or len(rows) != 1:
        raise RWRReproductionError("historical propagation rows must be a one-element array")
    row = _require_closed_mapping(
        rows[0], HISTORICAL_EVIDENCE_ROW_KEYS, "historical propagation row"
    )
    for name in ("aAcc", "alpha", "mIoU", "mAcc", "runtime_seconds"):
        _require_json_float(row[name], f"historical row {name}")
    for name in ("evaluated_images", "peak_cpu_ram_bytes", "peak_gpu_bytes", "steps"):
        _require_json_int(row[name], f"historical row {name}", minimum=0)
    classes = identity["dataset"]["classes"]
    for name in ("per_class_iou", "per_class_accuracy"):
        _validate_json_float_array(row[name], f"historical row {name}", length=classes)
    for name in ("intersection", "union", "predicted_pixels", "ground_truth_pixels"):
        _validate_json_int_array(row[name], f"historical row {name}", length=classes)
    optimum = _require_closed_mapping(
        payload["optimum"],
        frozenset(
            {
                "aAcc", "alpha", "at_grid_edge",
                "delta_mIoU_from_canonical_alpha_0.95_T10", "mAcc", "mIoU",
                "steps",
            }
        ),
        "historical optimum",
    )
    for name in (
        "aAcc", "alpha", "delta_mIoU_from_canonical_alpha_0.95_T10",
        "mAcc", "mIoU",
    ):
        _require_json_float(optimum[name], f"historical optimum {name}")
    _require_json_bool(optimum["at_grid_edge"], "historical optimum at_grid_edge")
    _require_json_int(optimum["steps"], "historical optimum steps", minimum=1)
    return document, row


def validate_historical_provenance(
    identity: Mapping[str, Any], *, artifact_reader: ArtifactReader
) -> dict[str, Any]:
    if type(identity.get("cache_manifest_archived")) is not bool:
        raise RWRReproductionError("cache_manifest_archived must be boolean")
    if identity.get("historical_cache_used_by_current_run") is not False:
        raise RWRReproductionError(
            "current inference must not use the historical cache"
        )
    historical = identity["historical"]
    artifacts = {
        prefix: _validate_artifact(artifact_reader, historical, prefix)
        for prefix in ("metrics", "cache_evidence", "graph", "solver", "config", "eval_base")
    }
    metrics = _validate_historical_metrics_document(_parse_json(
        artifacts["metrics"].decode("utf-8", errors="strict"),
        "historical canonical metrics",
    ), identity)
    canonical = metrics.get(identity["historical"]["metrics_record"])
    if not isinstance(canonical, Mapping):
        raise RWRReproductionError("historical metrics omit converged CGLS result")
    expected_metrics = identity["expected_metrics"]
    for observed_name, identity_name in (
        ("aAcc", "rwr_aAcc"), ("mIoU", "rwr_mIoU"), ("mAcc", "rwr_mAcc")
    ):
        observed = _require_json_float(canonical.get(observed_name), f"historical {observed_name}")
        if observed != expected_metrics[identity_name]:
            raise RWRReproductionError(f"historical {observed_name} mismatch")
    if _require_json_int(canonical.get("evaluated_images"), "historical evaluated_images") != identity["dataset"]["images"]:
        raise RWRReproductionError("historical evaluated image count mismatch")
    per_class = canonical.get("per_class_iou")
    if not isinstance(per_class, list) or len(per_class) != identity["dataset"]["classes"]:
        raise RWRReproductionError("historical class count mismatch")

    evidence, row = _validate_historical_evidence_document(_parse_json(
        artifacts["cache_evidence"].decode("utf-8", errors="strict"),
        "historical cache evidence",
    ), identity)
    if evidence.get("cache_manifest_sha256") != identity["cache_manifest_sha256"]:
        raise RWRReproductionError("historical cache-manifest hash mismatch")
    if identity["cache_manifest_evidence"] != "committed_attestation":
        raise RWRReproductionError("historical manifest evidence is not an attestation")
    if identity["cache_manifest_attestation_commit"] != historical["cache_evidence_commit"]:
        raise RWRReproductionError("historical attestation commit mismatch")
    if identity["cache_manifest_attestation_path"] != historical["cache_evidence_path"]:
        raise RWRReproductionError("historical attestation path mismatch")
    if identity["cache_manifest_attestation_blob_sha256"] != historical["cache_evidence_sha256"]:
        raise RWRReproductionError("historical attestation blob SHA256 mismatch")
    if evidence.get("git_commit") != identity["cache_source_commit"] or evidence.get("git_dirty") is not False:
        raise RWRReproductionError("historical cache source provenance mismatch")
    payload = evidence["payload"]
    baked = payload.get("kappa_k_baked_at_construction_time")
    if baked.get("affinity_power") is not True:
        raise RWRReproductionError("historical affinity-power evidence mismatch")
    if baked.get("knn_k") is not True:
        raise RWRReproductionError("historical graph construction evidence mismatch")
    if _require_json_float(row.get("alpha"), "historical alpha") != identity["rwr"]["alpha"]:
        raise RWRReproductionError("historical alpha mismatch")
    if _require_json_int(row.get("evaluated_images"), "historical cache images") != identity["dataset"]["images"]:
        raise RWRReproductionError("historical cache image count mismatch")
    result = {
        f"{prefix}_sha256": historical[f"{prefix}_sha256"]
        for prefix in artifacts
    }
    result.update(
        {
            "cache_manifest_evidence": identity["cache_manifest_evidence"],
            "cache_manifest_archived": identity["cache_manifest_archived"],
        }
    )
    if identity["cache_manifest_archived"]:
        archive = identity.get("archived_cache_manifest")
        if not isinstance(archive, Mapping):
            raise RWRReproductionError("archived cache manifest fields are missing")
        data = artifact_reader(archive["commit"], archive["path"])
        digest = hashlib.sha256(data).hexdigest()
        if digest != identity["cache_manifest_sha256"] or digest != archive["sha256"]:
            raise RWRReproductionError("archived cache manifest direct SHA256 mismatch")
        header = f"blob {len(data)}\0".encode("ascii")
        if hashlib.sha1(header + data).hexdigest() != archive["blob"]:
            raise RWRReproductionError("archived cache manifest blob mismatch")
        _parse_json(data.decode("utf-8", errors="strict"), "archived cache manifest")
        result["archived_cache_manifest_sha256"] = digest
    return result


def validate_resolved_config_pair(
    e3_resolved: Mapping[str, Any],
    rwr_resolved: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Enforce the sole evaluation-affecting allowlist: ``evaluate.rwr``."""
    try:
        e3_clean = clone_configuration(e3_resolved)
        rwr_clean = clone_configuration(rwr_resolved)
    except TypedConfigurationError as error:
        raise RWRReproductionError(f"invalid resolved configuration: {error}") from error
    evaluate = rwr_clean.get("evaluate")
    if not isinstance(evaluate, Mapping) or "rwr" not in evaluate:
        raise RWRReproductionError("resolved canonical configuration omits evaluate.rwr")
    try:
        rwr_mapping = clone_configuration(evaluate["rwr"])
    except TypedConfigurationError as error:
        raise RWRReproductionError(f"invalid evaluate.rwr mapping: {error}") from error
    del rwr_clean["evaluate"]["rwr"]
    difference = first_typed_difference(e3_clean, rwr_clean)
    if difference is not None:
        raise RWRReproductionError(
            "resolved RWR configuration differs from E3 outside evaluate.rwr at "
            + difference
        )
    expected_rwr = {
        "enabled": identity["rwr"]["enabled"],
        "identity_path": IDENTITY_RELATIVE_PATH.as_posix(),
    }
    difference = first_typed_difference(expected_rwr, rwr_mapping)
    if difference is not None:
        raise RWRReproductionError(
            "canonical evaluate.rwr settings mismatch at " + difference
        )


def _resolve_complete_rwr_configuration(
    root: Path,
    identity: Mapping[str, Any],
    e3_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_path = root / e3_identity["evaluation"]["base_config_path"]
    e3_resolved = _resolve_complete_e3_configuration(
        repo_root=root,
        identity=e3_identity,
        eval_config=root / e3_identity["evaluation"]["config_path"],
        eval_base_config=base_path,
    )
    rwr_resolved = _resolve_complete_e3_configuration(
        repo_root=root,
        identity=e3_identity,
        eval_config=root / identity["canonical_config_path"],
        eval_base_config=base_path,
    )
    e3_path = e3_identity["evaluation"]["config_path"]
    rwr_path = identity["canonical_config_path"]
    for source in rwr_resolved["sources"]:
        if source["path"] == e3_path:
            source["role"] = "e3_leaf"
        elif source["path"] == rwr_path:
            source["role"] = "rwr_leaf"
    return e3_resolved, rwr_resolved


def _validate_complete_rwr_configuration(
    identity: Mapping[str, Any],
    e3_identity: Mapping[str, Any],
    e3_resolved: Mapping[str, Any],
    rwr_resolved: Mapping[str, Any],
) -> dict[str, str]:
    _validate_complete_e3_configuration(e3_identity, e3_resolved)
    validate_resolved_config_pair(
        e3_resolved["effective"], rwr_resolved["effective"], identity
    )
    try:
        without_rwr = clone_configuration(rwr_resolved["complete"])
    except TypedConfigurationError as error:
        raise RWRReproductionError(f"invalid complete RWR configuration: {error}") from error
    del without_rwr["runtime"]["evaluate"]["rwr"]
    expected_complete = clone_configuration(e3_resolved["complete"])
    expected_complete["runtime"]["evaluate"]["rwr"] = {
        "enabled": identity["rwr"]["enabled"],
        "identity_path": IDENTITY_RELATIVE_PATH.as_posix(),
    }
    difference = first_typed_difference(expected_complete, rwr_resolved["complete"])
    if difference is not None:
        raise RWRReproductionError(
            "complete resolved RWR configuration differs at " + difference
        )
    difference = first_typed_difference(e3_resolved["complete"], without_rwr)
    if difference is not None:
        raise RWRReproductionError(
            "resolved RWR-without-evaluate.rwr differs from E3 at " + difference
        )
    observed = {
        "full_sha256": typed_configuration_sha256(rwr_resolved["complete"]),
        "without_rwr_sha256": typed_configuration_sha256(without_rwr),
        "evaluation_sha256": typed_configuration_sha256(rwr_resolved["evaluation"]),
    }
    expected = identity["resolved_configuration"]
    locations = {
        "full_sha256": "$",
        "without_rwr_sha256": "$.runtime.evaluate (without rwr)",
        "evaluation_sha256": "$.runtime.evaluate",
    }
    expected_sources = identity["configuration_sources"]
    observed_sources = rwr_resolved["sources"]
    if len(expected_sources) != len(observed_sources):
        raise RWRReproductionError(
            "RWR configuration source chain length mismatch: "
            f"expected {len(expected_sources)}, observed {len(observed_sources)}"
        )
    for index, (expected_source, observed_source) in enumerate(
        zip(expected_sources, observed_sources)
    ):
        difference = first_typed_difference(
            expected_source, observed_source, path=f"$.configuration_sources[{index}]"
        )
        if difference is not None:
            raise RWRReproductionError("RWR configuration source mismatch at " + difference)
    for name, digest in observed.items():
        if digest != expected[name]:
            raise RWRReproductionError(
                f"complete RWR configuration mismatch at {locations[name]}: "
                f"expected {expected[name]}, observed {digest}"
            )
    return observed


def _validate_identity_against_e3(identity: Mapping[str, Any], e3: Mapping[str, Any]) -> None:
    comparisons = {
        "dataset.name": (identity["dataset"]["name"], e3["dataset"]["name"]),
        "dataset.images": (identity["dataset"]["images"], e3["dataset"]["images"]),
        "dataset.classes": (identity["dataset"]["classes"], e3["dataset"]["classes"]),
        "dataset.background_class": (identity["dataset"]["background_class"], e3["dataset"]["background_class"]),
        "evaluation.crop": (identity["evaluation"]["crop"], e3["evaluation"]["crop"]),
        "evaluation.stride": (identity["evaluation"]["stride"], e3["evaluation"]["stride"]),
        "evaluation.template": (identity["evaluation"]["template"], e3["evaluation"]["template"]),
        "evaluation.pamr": (identity["evaluation"]["pamr"], e3["evaluation"]["pamr"]),
        "checkpoint.projection_class": (identity["checkpoint"]["projection_class"], e3["projection"]["class"]),
        "checkpoint.projection_name": (identity["checkpoint"]["projection_name"], e3["projection"]["name"]),
        "checkpoint.projection_model": (identity["checkpoint"]["projection_model"], e3["projection"]["model"]),
        "checkpoint.path": (identity["checkpoint"]["path"], e3["projection"]["checkpoint_path"]),
        "checkpoint.loading_required": (identity["checkpoint"]["loading_required"], e3["projection"]["checkpoint_loading_required"]),
    }
    for label, (observed, expected) in comparisons.items():
        if observed != expected:
            raise RWRReproductionError(f"RWR identity {label} disagrees with E3 identity")


def validate_static_configuration(
    *,
    repo_root: Path | None = None,
    identity_path: Path | None = None,
    check_checkpoint: bool = False,
    dataset_root: Path | None = None,
    weight_dir: Path | None = None,
    artifact_reader: ArtifactReader | None = None,
    check_git: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    validate_supported_capabilities(identity)
    e3_identity_path = root / identity["e3_identity_path"]
    e3_result = _validate_e3(
        repo_root=root,
        identity_path=e3_identity_path,
        check_checkpoint=check_checkpoint,
        dataset_root=dataset_root,
        weight_dir=weight_dir,
        check_git=check_git,
    )
    e3_identity = _load_e3_identity(e3_identity_path, repo_root=root)
    _validate_identity_against_e3(identity, e3_identity)
    config_path = root / identity["canonical_config_path"]
    try:
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as error:
        raise RWRReproductionError(f"cannot hash canonical RWR config: {error}") from error
    if config_digest != identity["canonical_config_sha256"]:
        raise RWRReproductionError("canonical RWR config SHA256 mismatch")
    e3_resolved, rwr_resolved = _resolve_complete_rwr_configuration(
        root, identity, e3_identity
    )
    complete_hashes = _validate_complete_rwr_configuration(
        identity, e3_identity, e3_resolved, rwr_resolved
    )
    provenance = validate_historical_provenance(
        identity,
        artifact_reader=artifact_reader or _git_artifact_reader(root),
    )
    if check_checkpoint:
        checkpoint = Path(e3_result["projection_checkpoint"])
        digest = hashlib.sha256()
        try:
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise RWRReproductionError(
                f"cannot hash canonical projection checkpoint {checkpoint}: {error}"
            ) from error
        if digest.hexdigest() != identity["checkpoint"]["sha256"]:
            raise RWRReproductionError("canonical projection checkpoint SHA256 mismatch")
    return {
        "identity_name": identity["identity_name"],
        "canonical_config": str(config_path),
        "e3_identity": e3_result["identity_name"],
        "source_e10_commit": identity["source_e10_commit"],
        "cache_source_commit": identity["cache_source_commit"],
        "cache_manifest_sha256": identity["cache_manifest_sha256"],
        "cache_manifest_evidence": identity["cache_manifest_evidence"],
        "cache_manifest_attestation_commit": identity[
            "cache_manifest_attestation_commit"
        ],
        "cache_manifest_attestation_path": identity[
            "cache_manifest_attestation_path"
        ],
        "cache_manifest_attestation_blob_sha256": identity[
            "cache_manifest_attestation_blob_sha256"
        ],
        "cache_manifest_archived": identity["cache_manifest_archived"],
        "historical_cache_used_by_current_run": identity[
            "historical_cache_used_by_current_run"
        ],
        "historical_artifacts": provenance,
        "checkpoint_checked": check_checkpoint,
        "dataset_checked": dataset_root is not None,
        "external_weights_checked": weight_dir is not None,
        "typed_configuration_encoding": TYPED_CONFIGURATION_ENCODING,
        "resolved_configuration_hashes": complete_hashes,
        "configuration_sources": rwr_resolved["sources"],
    }


def _parse_json(text: str, label: str) -> Mapping[str, Any]:
    def closed_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RWRReproductionError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_float=Decimal,
            object_pairs_hook=closed_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RWRReproductionError(f"{label} contains non-finite {token}")
            ),
        )
    except RWRReproductionError:
        raise
    except (json.JSONDecodeError, TypeError) as error:
        raise RWRReproductionError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise RWRReproductionError(f"{label} must contain one JSON object")
    return value


def parse_structured_result(path: Path, *, source_kind: str) -> Mapping[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="strict")
    except OSError as error:
        raise RWRReproductionError(f"cannot read result {path}: {error}") from error
    if source_kind == "json":
        return _parse_json(text, "structured result")
    if source_kind != "log":
        raise RWRReproductionError(f"unsupported result source kind {source_kind!r}")
    blocks = [
        line.split(RESULT_PREFIX, 1)[1].strip()
        for line in text.splitlines()
        if RESULT_PREFIX in line
    ]
    if len(blocks) != 1:
        raise RWRReproductionError(
            f"evaluation log must contain exactly one structured RWR result; found {len(blocks)}"
        )
    return _parse_json(blocks[0], "structured log result")


def _require_integer(value: Any, label: str) -> int:
    return _require_exact_int(value, label, minimum=0)


def verify_record(
    record: Mapping[str, Any], identity: Mapping[str, Any]
) -> str:
    """Version-aware dispatcher. Historical (v2) structured results remain
    fully, unconditionally verifiable via :func:`_verify_record_v2`
    (byte-identical logic to the original, single-version verifier this
    function used to be). Current (v3) results -- which require explicit
    full-precision metric-source provenance -- are verified via
    :func:`_verify_record_v3`. Any other format_version is rejected."""
    if not hasattr(record, "get"):
        raise RWRReproductionError("structured result must be a mapping")
    format_version = record.get("format_version")
    if format_version == RESULT_FORMAT_VERSION:
        return _verify_record_v2(record, identity)
    if format_version == RESULT_FORMAT_VERSION_V3:
        return _verify_record_v3(record, identity)
    raise RWRReproductionError(
        f"unsupported structured result format_version {format_version!r}; "
        f"supported versions: {SUPPORTED_RESULT_FORMAT_VERSIONS}"
    )


def _verify_record_v2(
    record: Mapping[str, Any], identity: Mapping[str, Any]
) -> str:
    """Historical verifier, UNCHANGED from the original single-version
    ``verify_record`` -- preserves the ability to re-verify any structured
    result ever logged under format_version v2 (including the E10
    provenance artifacts), forever."""
    if set(record) != RESULT_KEYS:
        raise RWRReproductionError("structured result has an unexpected schema")
    if record["format_version"] != RESULT_FORMAT_VERSION:
        raise RWRReproductionError("structured result format version mismatch")
    if record["identity_name"] != identity["identity_name"]:
        raise RWRReproductionError("structured result identity mismatch")
    _require_exact_string(record["format_version"], "result format_version")
    _require_exact_string(record["identity_name"], "result identity_name")
    if _require_exact_string(record["metrics_precision"], "metrics_precision") != "full":
        raise RWRReproductionError("structured result is not full precision")
    minimum_places = identity["acceptance"]["minimum_metric_decimal_places"]
    values: dict[str, float] = {}
    for name in ("aAcc", "mIoU", "mAcc"):
        token = record[name]
        if not isinstance(token, Decimal):
            raise RWRReproductionError(f"{name} must be a decimal JSON number")
        if max(0, -token.as_tuple().exponent) < minimum_places:
            raise RWRReproductionError(
                f"{name} is rounded-only; at least {minimum_places} decimals required"
            )
        values[name] = _require_json_float(token, name)
    if all(0 <= value <= 1 for value in values.values()):
        raise RWRReproductionError(
            "fractional [0,1] metrics cannot masquerade as percentages"
        )
    if any(value < 0 or value > 100 for value in values.values()):
        raise RWRReproductionError("metrics must use percentage units")
    if _require_json_int(record["image_count"], "image_count", minimum=1) != identity["dataset"]["images"]:
        raise RWRReproductionError("canonical image count mismatch")
    if _require_json_int(record["class_count"], "class_count", minimum=1) != identity["dataset"]["classes"]:
        raise RWRReproductionError("canonical class count mismatch")
    expected_fields = {
        "rwr_enabled": identity["rwr"]["enabled"],
        "alpha": identity["rwr"]["alpha"],
        "top_k": identity["rwr"]["top_k"],
        "affinity_power": identity["rwr"]["affinity_power"],
        "graph_mode": identity["rwr"]["graph_mode"],
        "solver": identity["solver"]["method"],
        "solver_rtol": identity["solver"]["rtol"],
        "solver_atol": identity["solver"]["atol"],
        "solver_max_iterations": identity["solver"]["max_iterations"],
        "crop": identity["evaluation"]["crop"],
        "stride": identity["evaluation"]["stride"],
        "pamr": identity["evaluation"]["pamr"],
        "background_class": identity["dataset"]["background_class"],
        "config_path": identity["canonical_config_path"],
        "config_sha256": identity["canonical_config_sha256"],
        "checkpoint_path": identity["checkpoint"]["path"],
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "source_e10_commit": identity["source_e10_commit"],
        "cache_source_commit": identity["cache_source_commit"],
        "cache_manifest_sha256": identity["cache_manifest_sha256"],
        "cache_manifest_evidence": identity["cache_manifest_evidence"],
        "cache_manifest_attestation_commit": identity[
            "cache_manifest_attestation_commit"
        ],
        "cache_manifest_attestation_path": identity[
            "cache_manifest_attestation_path"
        ],
        "cache_manifest_attestation_blob_sha256": identity[
            "cache_manifest_attestation_blob_sha256"
        ],
        "cache_manifest_archived": identity["cache_manifest_archived"],
        "historical_cache_used_by_current_run": identity[
            "historical_cache_used_by_current_run"
        ],
    }
    for name, expected in expected_fields.items():
        observed = record[name]
        if type(expected) is float:
            if _require_json_float(observed, name) != expected:
                raise RWRReproductionError(f"canonical {name} mismatch")
        elif type(expected) is bool:
            if _require_json_bool(observed, name) is not expected:
                raise RWRReproductionError(f"canonical {name} mismatch")
        elif type(expected) is int:
            if _require_json_int(observed, name) != expected:
                raise RWRReproductionError(f"canonical {name} mismatch")
        elif type(expected) is list:
            if _require_exact_integer_pair(observed, name) != tuple(expected):
                raise RWRReproductionError(f"canonical {name} mismatch")
        elif _require_exact_string(observed, name) != expected:
            raise RWRReproductionError(f"canonical {name} mismatch")
    _require_git_identity(record["source_git_commit"], "runtime Git commit")
    _require_exact_string(record["source_git_branch"], "runtime Git branch")
    _require_json_bool(record["source_git_dirty"], "runtime Git dirty flag")
    for name in ("gpu_model", "torch_version"):
        _require_exact_string(record[name], f"runtime {name}")
    _require_exact_string(record["cuda_version"], "runtime CUDA version")
    elapsed = _require_json_float(record["elapsed_seconds"], "elapsed_seconds")
    if elapsed < 0:
        raise RWRReproductionError("elapsed_seconds must be non-negative")

    summary = record["solver_summary"]
    if not isinstance(summary, Mapping) or set(summary) != SUMMARY_KEYS:
        raise RWRReproductionError("solver summary schema mismatch")
    for name in SUMMARY_KEYS - {"maximum_scaled_residual"}:
        _require_integer(summary[name], f"solver_summary.{name}")
    maximum_scaled = _require_json_float(
        summary["maximum_scaled_residual"], "maximum scaled residual"
    )
    if maximum_scaled < 0 or maximum_scaled > 1:
        raise RWRReproductionError("solver residual contract was not satisfied")
    if summary["window_count"] <= 0:
        raise RWRReproductionError("no RWR crop windows were evaluated")
    if summary["converged_window_count"] != summary["window_count"]:
        raise RWRReproductionError("not every RWR crop solve converged")
    if summary["nonzero_restart_windows"] > summary["window_count"]:
        raise RWRReproductionError("restart window count is inconsistent")

    expected_miou = float(identity["expected_metrics"]["rwr_mIoU"])
    delta = abs(values["mIoU"] - expected_miou)
    tolerance = float(identity["acceptance"]["structured_absolute_mIoU"])
    if delta > tolerance:
        raise RWRReproductionError(
            f"canonical mIoU mismatch: expected {expected_miou}, observed "
            f"{values['mIoU']}, delta {delta}, tolerance {tolerance}"
        )
    if round(values["mIoU"], 2) != float(identity["acceptance"]["rounded_mIoU"]):
        raise RWRReproductionError(
            "canonical mIoU does not round to the identity acceptance value"
        )
    for metric_name, identity_name in (("aAcc", "rwr_aAcc"), ("mAcc", "rwr_mAcc")):
        expected_metric = float(identity["expected_metrics"][identity_name])
        if abs(values[metric_name] - expected_metric) > tolerance:
            raise RWRReproductionError(f"canonical {metric_name} mismatch")
    expected_base = float(identity["expected_metrics"]["e3_mIoU"])
    observed_gain = _require_json_float(record["gain_over_e3_miou"], "mIoU gain")
    if abs(observed_gain - (values["mIoU"] - expected_base)) > 1e-9:
        raise RWRReproductionError("reported mIoU gain is inconsistent")
    return (
        "RWR REPRODUCTION PASS "
        f"images={record['image_count']} classes={record['class_count']} "
        f"aAcc={values['aAcc']:.12f} mIoU={values['mIoU']:.12f} "
        f"mAcc={values['mAcc']:.12f} delta={delta:.12g} "
        f"gain={observed_gain:.12f}"
    )


def _verify_record_v3(
    record: Mapping[str, Any], identity: Mapping[str, Any]
) -> str:
    """Current verifier. Identical acceptance logic to v2 (same tolerance,
    same rounding contract, same canonical-config/identity fields, same
    solver-summary shape), PLUS: an explicit, closed-vocabulary
    ``metric_source`` that must name a genuine full-precision computation
    (never a rounded/approximate one masquerading as full precision), and
    an optional, separately-scoped ``parity_solver_summary`` that is
    NEVER substituted for the primary ``solver_summary`` in any check."""
    if set(record) != RESULT_KEYS_V3:
        raise RWRReproductionError("structured result has an unexpected schema")
    if record["format_version"] != RESULT_FORMAT_VERSION_V3:
        raise RWRReproductionError("structured result format version mismatch")
    if record["identity_name"] != identity["identity_name"]:
        raise RWRReproductionError("structured result identity mismatch")
    _require_exact_string(record["format_version"], "result format_version")
    _require_exact_string(record["identity_name"], "result identity_name")
    if _require_exact_string(record["metrics_precision"], "metrics_precision") != "full":
        raise RWRReproductionError("structured result is not full precision")
    known_metric_sources = {FULL_PRECISION_METRIC_SOURCE}
    observed_metric_source = _require_exact_string(record["metric_source"], "metric_source")
    if observed_metric_source not in known_metric_sources:
        raise RWRReproductionError(
            f"metric_source {observed_metric_source!r} does not name a recognized full-precision "
            f"computation (known: {sorted(known_metric_sources)}) -- refusing to accept a "
            "rounded/approximate metric object as full precision"
        )
    minimum_places = identity["acceptance"]["minimum_metric_decimal_places"]
    values: dict[str, float] = {}
    for name in ("aAcc", "mIoU", "mAcc"):
        token = record[name]
        if not isinstance(token, Decimal):
            raise RWRReproductionError(f"{name} must be a decimal JSON number")
        if max(0, -token.as_tuple().exponent) < minimum_places:
            raise RWRReproductionError(
                f"{name} is rounded-only; at least {minimum_places} decimals required"
            )
        values[name] = _require_json_float(token, name)
    if all(0 <= value <= 1 for value in values.values()):
        raise RWRReproductionError(
            "fractional [0,1] metrics cannot masquerade as percentages"
        )
    if any(value < 0 or value > 100 for value in values.values()):
        raise RWRReproductionError("metrics must use percentage units")
    if _require_json_int(record["image_count"], "image_count", minimum=1) != identity["dataset"]["images"]:
        raise RWRReproductionError("canonical image count mismatch")
    if _require_json_int(record["class_count"], "class_count", minimum=1) != identity["dataset"]["classes"]:
        raise RWRReproductionError("canonical class count mismatch")
    expected_fields = {
        "rwr_enabled": identity["rwr"]["enabled"],
        "alpha": identity["rwr"]["alpha"],
        "top_k": identity["rwr"]["top_k"],
        "affinity_power": identity["rwr"]["affinity_power"],
        "graph_mode": identity["rwr"]["graph_mode"],
        "solver": identity["solver"]["method"],
        "solver_rtol": identity["solver"]["rtol"],
        "solver_atol": identity["solver"]["atol"],
        "solver_max_iterations": identity["solver"]["max_iterations"],
        "crop": identity["evaluation"]["crop"],
        "stride": identity["evaluation"]["stride"],
        "pamr": identity["evaluation"]["pamr"],
        "background_class": identity["dataset"]["background_class"],
        "config_path": identity["canonical_config_path"],
        "config_sha256": identity["canonical_config_sha256"],
        "checkpoint_path": identity["checkpoint"]["path"],
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "source_e10_commit": identity["source_e10_commit"],
        "cache_source_commit": identity["cache_source_commit"],
        "cache_manifest_sha256": identity["cache_manifest_sha256"],
        "cache_manifest_evidence": identity["cache_manifest_evidence"],
        "cache_manifest_attestation_commit": identity[
            "cache_manifest_attestation_commit"
        ],
        "cache_manifest_attestation_path": identity[
            "cache_manifest_attestation_path"
        ],
        "cache_manifest_attestation_blob_sha256": identity[
            "cache_manifest_attestation_blob_sha256"
        ],
        "cache_manifest_archived": identity["cache_manifest_archived"],
        "historical_cache_used_by_current_run": identity[
            "historical_cache_used_by_current_run"
        ],
    }
    for name, expected in expected_fields.items():
        observed = record[name]
        if type(expected) is float:
            if _require_json_float(observed, name) != expected:
                raise RWRReproductionError(f"canonical {name} mismatch")
        elif type(expected) is bool:
            if _require_json_bool(observed, name) is not expected:
                raise RWRReproductionError(f"canonical {name} mismatch")
        elif type(expected) is int:
            if _require_json_int(observed, name) != expected:
                raise RWRReproductionError(f"canonical {name} mismatch")
        elif type(expected) is list:
            if _require_exact_integer_pair(observed, name) != tuple(expected):
                raise RWRReproductionError(f"canonical {name} mismatch")
        elif _require_exact_string(observed, name) != expected:
            raise RWRReproductionError(f"canonical {name} mismatch")
    _require_git_identity(record["source_git_commit"], "runtime Git commit")
    _require_exact_string(record["source_git_branch"], "runtime Git branch")
    _require_json_bool(record["source_git_dirty"], "runtime Git dirty flag")
    for name in ("gpu_model", "torch_version"):
        _require_exact_string(record[name], f"runtime {name}")
    _require_exact_string(record["cuda_version"], "runtime CUDA version")
    elapsed = _require_json_float(record["elapsed_seconds"], "elapsed_seconds")
    if elapsed < 0:
        raise RWRReproductionError("elapsed_seconds must be non-negative")

    def _verify_summary(summary: Any, label: str) -> None:
        if not isinstance(summary, Mapping) or set(summary) != SUMMARY_KEYS:
            raise RWRReproductionError(f"{label} schema mismatch")
        for name in SUMMARY_KEYS - {"maximum_scaled_residual"}:
            _require_integer(summary[name], f"{label}.{name}")
        maximum_scaled = _require_json_float(summary["maximum_scaled_residual"], f"{label} maximum scaled residual")
        if maximum_scaled < 0 or maximum_scaled > 1:
            raise RWRReproductionError(f"{label} residual contract was not satisfied")

    summary = record["solver_summary"]
    _verify_summary(summary, "solver_summary")
    if summary["window_count"] <= 0:
        raise RWRReproductionError("no RWR crop windows were evaluated")
    if summary["converged_window_count"] != summary["window_count"]:
        raise RWRReproductionError("not every RWR crop solve converged")
    if summary["nonzero_restart_windows"] > summary["window_count"]:
        raise RWRReproductionError("restart window count is inconsistent")

    parity_summary = record["parity_solver_summary"]
    if parity_summary is not None:
        _verify_summary(parity_summary, "parity_solver_summary")
        if parity_summary["nonzero_restart_windows"] > parity_summary["window_count"]:
            raise RWRReproductionError("parity restart window count is inconsistent")
        # The parity summary must never silently satisfy the primary
        # window-count reconciliation on its own; it is a distinct phase.
        if parity_summary is summary:
            raise RWRReproductionError("parity_solver_summary must not alias solver_summary")

    expected_miou = float(identity["expected_metrics"]["rwr_mIoU"])
    delta = abs(values["mIoU"] - expected_miou)
    tolerance = float(identity["acceptance"]["structured_absolute_mIoU"])
    if delta > tolerance:
        raise RWRReproductionError(
            f"canonical mIoU mismatch: expected {expected_miou}, observed "
            f"{values['mIoU']}, delta {delta}, tolerance {tolerance}"
        )
    if round(values["mIoU"], 2) != float(identity["acceptance"]["rounded_mIoU"]):
        raise RWRReproductionError(
            "canonical mIoU does not round to the identity acceptance value"
        )
    for metric_name, identity_name in (("aAcc", "rwr_aAcc"), ("mAcc", "rwr_mAcc")):
        expected_metric = float(identity["expected_metrics"][identity_name])
        if abs(values[metric_name] - expected_metric) > tolerance:
            raise RWRReproductionError(f"canonical {metric_name} mismatch")
    expected_base = float(identity["expected_metrics"]["e3_mIoU"])
    observed_gain = _require_json_float(record["gain_over_e3_miou"], "mIoU gain")
    if abs(observed_gain - (values["mIoU"] - expected_base)) > 1e-9:
        raise RWRReproductionError("reported mIoU gain is inconsistent")
    return (
        "RWR REPRODUCTION PASS (v3, full-precision) "
        f"images={record['image_count']} classes={record['class_count']} "
        f"aAcc={values['aAcc']:.12f} mIoU={values['mIoU']:.12f} "
        f"mAcc={values['mAcc']:.12f} delta={delta:.12g} "
        f"gain={observed_gain:.12f} metric_source={observed_metric_source}"
    )


def verify_result(
    path: Path,
    *,
    source_kind: str,
    identity_path: Path | None = None,
    repo_root: Path | None = None,
) -> str:
    identity = load_identity(identity_path, repo_root=repo_root)
    return verify_record(
        parse_structured_result(path, source_kind=source_kind), identity
    )


__all__ = [
    "FULL_PRECISION_METRIC_SOURCE",
    "IDENTITY_RELATIVE_PATH",
    "RESULT_FORMAT_VERSION",
    "RESULT_FORMAT_VERSION_V3",
    "RESULT_KEYS_V3",
    "RESULT_PREFIX",
    "SUPPORTED_CANONICAL_GRAPH_MODE",
    "SUPPORTED_CANONICAL_SOLVER_METHOD",
    "SUPPORTED_RESULT_FORMAT_VERSIONS",
    "RWRReproductionError",
    "load_identity",
    "parse_structured_result",
    "repository_root",
    "validate_historical_provenance",
    "validate_resolved_config_pair",
    "validate_static_configuration",
    "validate_supported_capabilities",
    "verify_record",
    "verify_result",
]
