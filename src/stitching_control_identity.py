"""Static identity loading/validation for the reusable stitching control
suite.

This module deliberately avoids importing MMCV, Torch, model code, or
dataset code. All scientific values already registered in the parent
matched identity (alpha, steps, crop, stride, affinity_power, k=12) are
never redeclared here as bare literals -- callers load them from the
parent chain (:mod:`src.matched_k11_k12_identity` via
:mod:`src.k11_k12_power_evaluation_identity`) directly. This identity only
registers execution-contract facts specific to the stitching-control
suite: variant names/staging/weighting, the Hann formula, accumulator
dtypes, and run-mode/schema names.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e12_stitching_control_suite.toml")

RUN_MODES = ("pilot20", "pilot100", "full5000")
CHECKPOINT_SCHEMA_NAME = "talk2dino-stitching-control-checkpoint-v1"
PER_IMAGE_STATS_SCHEMA_NAME = "talk2dino-stitching-control-per-image-stats-v1"
RUN_MODE_IMAGE_COUNT_KEYS = {
    "pilot20": "pilot20_image_count", "pilot100": "pilot100_image_count", "full5000": "full5000_image_count",
}
RUN_MODE_SCHEMA_KEYS = {
    "pilot20": "pilot20_schema_name", "pilot100": "pilot100_schema_name", "full5000": "full5000_schema_name",
}

CANONICAL_VARIANT_NAMES = ("uniform_probability", "hann_probability", "uniform_score", "hann_score")
SUPPORTED_STAGES = ("probability", "score")
SUPPORTED_WEIGHTINGS = ("uniform", "hann")
SUPPORTED_SIGMOID_STAGES = ("before_interpolation", "after_stitch")
SUPPORTED_VARIANT_CONTRACT = {
    "uniform_probability": {"stage": "probability", "weighting": "uniform", "sigmoid_stage": "before_interpolation", "weighting_formula": "uniform_count", "reproduces_existing_evaluator": True},
    "hann_probability": {"stage": "probability", "weighting": "hann", "sigmoid_stage": "before_interpolation", "weighting_formula": "pixel_centred_separable_hann", "reproduces_existing_evaluator": False},
    "uniform_score": {"stage": "score", "weighting": "uniform", "sigmoid_stage": "after_stitch", "weighting_formula": "uniform_count", "reproduces_existing_evaluator": False},
    "hann_score": {"stage": "score", "weighting": "hann", "sigmoid_stage": "after_stitch", "weighting_formula": "pixel_centred_separable_hann", "reproduces_existing_evaluator": False},
}

SUPPORTED_K = 12
SUPPORTED_ALPHA = 0.98
SUPPORTED_STEPS = 320
SUPPORTED_GRAPH_MODE = "directed_topk"
SUPPORTED_AFFINITY_FUNCTION = "relu_cosine_power"
SUPPORTED_AFFINITY_POWER = 3.0
SUPPORTED_SCORE_STAGE = "pre_sigmoid_pre_upsample_pre_stitch"
SUPPORTED_CROP = (448, 448)
SUPPORTED_STRIDE = (224, 224)
SUPPORTED_INTERPOLATION_MODE = "bilinear"
SUPPORTED_CROP_ORDER = "row_major_flat_index"
SUPPORTED_CLASS_COUNT = 171
SUPPORTED_IMAGE_COUNT = 5000
SUPPORTED_METRIC_NAMES = ("aAcc", "mIoU", "mAcc")
SUPPORTED_METRIC_UNIT = "percent_0_100"
SUPPORTED_PRECISION_SOURCE = "full_precision_area_statistics_from_mmseg_pre_eval"
SUPPORTED_ACCUMULATOR_DTYPE = "float32"
SUPPORTED_HANN_FORMULA = "h_N(x) = 0.5 - 0.5*cos(2*pi*(x+0.5)/N) for x=0,...,N-1; W(y,x) = h_H(y) * h_W(x)"
SUPPORTED_RUN_MODE_IMAGE_COUNTS = {
    "pilot20_image_count": 20, "pilot100_image_count": 100, "full5000_image_count": 5000,
}
SUPPORTED_MAX_INTERPOLATIONS_PER_WINDOW = 2
SUPPORTED_CHECKPOINT_GRANULARITY = "one_complete_image_all_variants"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "parent_identity", "propagation", "dataset", "geometry",
        "checkpoint_binding", "variants", "hann", "accumulation", "metrics", "run_modes",
        "execution_contract", "prohibited",
    }
)
IDENTITY_SECTION_KEYS = {
    "identity": frozenset({"name", "schema_version", "description", "required_ancestor_commit"}),
    "parent_identity": frozenset(
        {
            "power_evaluation_identity_path", "power_evaluation_identity_name", "power_evaluation_identity_sha256",
            "matched_identity_path", "matched_identity_name", "matched_identity_sha256",
            "e3_identity_path", "e3_identity_name", "e3_identity_sha256", "required_relationship",
        }
    ),
    "propagation": frozenset(
        {
            "shared_variant", "k", "alpha", "steps", "graph_mode", "affinity_function", "affinity_power",
            "self_edge_policy", "fallback_row_policy", "score_stage",
        }
    ),
    "dataset": frozenset({"name", "protocol", "images", "classes", "background_class"}),
    "geometry": frozenset(
        {"crop", "stride", "interpolation_mode", "align_corners", "crop_order", "window_order_source"}
    ),
    "checkpoint_binding": frozenset({"projection_name", "projection_sha256"}),
    "variants": frozenset({"names", "order", "uniform_identity_anchor", *CANONICAL_VARIANT_NAMES}),
    "hann": frozenset({"formula", "pixel_centred", "epsilon", "strictly_positive_at_boundary"}),
    "accumulation": frozenset(
        {
            "numerator_dtype", "denominator_dtype", "in_place_source_mutation",
            "accumulator_storage_shared_between_variants", "restitching_method",
        }
    ),
    "metrics": frozenset(
        {"metric_names", "unit", "precision_source", "class_count", "paired_delta_reference_variant", "paired_delta_field_prefix"}
    ),
    "run_modes": frozenset(
        {
            "pilot20_image_count", "pilot100_image_count", "full5000_image_count",
            "pilot20_schema_name", "pilot100_schema_name", "full5000_schema_name",
            "checkpoint_schema_name", "per_image_stats_manifest_schema_name",
        }
    ),
    "execution_contract": frozenset(
        {
            "model_calls_equal_windows", "graph_builds_equal_windows", "propagation_calls_equal_windows",
            "max_interpolations_per_window", "checkpoint_granularity",
        }
    ),
    "prohibited": frozenset({"list"}),
}
_VARIANT_SECTION_KEYS = frozenset({"stage", "weighting", "sigmoid_stage", "weighting_formula", "reproduces_existing_evaluator"})


class StitchingControlIdentityError(ValueError):
    """Raised when the stitching-control identity, a checkpoint, or a
    result fails closed. Always fail closed: never silently substitute a
    default variant, formula, or unverified assumption."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise StitchingControlIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise StitchingControlIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise StitchingControlIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise StitchingControlIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise StitchingControlIdentityError(f"{label} must be an exact float")
    if not math.isfinite(value):
        raise StitchingControlIdentityError(f"{label} must be finite")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise StitchingControlIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", token) is None:
        raise StitchingControlIdentityError(f"{label} must be a full Git identity")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise StitchingControlIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise StitchingControlIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string_list(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise StitchingControlIdentityError(f"{label} must be a non-empty exact list")
    if any(type(item) is not str for item in value):
        raise StitchingControlIdentityError(f"{label} elements must be exact strings")
    return tuple(value)


def _require_int_pair(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2 or any(type(v) is not int for v in value):
        raise StitchingControlIdentityError(f"{label} must be an exact [int, int] pair")
    return (value[0], value[1])


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise StitchingControlIdentityError(
            f"cannot load stitching control suite identity {source}: {error}"
        ) from error

    if set(identity) != IDENTITY_TOP_KEYS:
        raise StitchingControlIdentityError("stitching control identity has an unexpected top-level schema")
    _require_exact_string(identity["format_version"], "format_version")
    if identity["format_version"] != "talk2dino-stitching-control-suite-identity-v1":
        raise StitchingControlIdentityError("unsupported stitching control identity format_version")
    for section, keys in IDENTITY_SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    block = identity["identity"]
    _require_exact_string(block["name"], "identity.name")
    _require_exact_string(block["schema_version"], "identity.schema_version")
    if block["schema_version"] != identity["format_version"]:
        raise StitchingControlIdentityError("identity.schema_version disagrees with format_version")
    _require_exact_string(block["description"], "identity.description")
    _require_git_identity(block["required_ancestor_commit"], "identity.required_ancestor_commit")

    parent = identity["parent_identity"]
    _require_relative_path(parent["power_evaluation_identity_path"], "parent_identity.power_evaluation_identity_path")
    _require_exact_string(parent["power_evaluation_identity_name"], "parent_identity.power_evaluation_identity_name")
    _require_sha256(parent["power_evaluation_identity_sha256"], "parent_identity.power_evaluation_identity_sha256")
    _require_relative_path(parent["matched_identity_path"], "parent_identity.matched_identity_path")
    _require_exact_string(parent["matched_identity_name"], "parent_identity.matched_identity_name")
    _require_sha256(parent["matched_identity_sha256"], "parent_identity.matched_identity_sha256")
    _require_relative_path(parent["e3_identity_path"], "parent_identity.e3_identity_path")
    _require_exact_string(parent["e3_identity_name"], "parent_identity.e3_identity_name")
    _require_sha256(parent["e3_identity_sha256"], "parent_identity.e3_identity_sha256")
    _require_exact_string(parent["required_relationship"], "parent_identity.required_relationship")

    prop = identity["propagation"]
    if prop["shared_variant"] != "k12":
        raise StitchingControlIdentityError("propagation.shared_variant must be 'k12' -- this suite never varies the graph")
    if _require_exact_int(prop["k"], "propagation.k") != SUPPORTED_K:
        raise StitchingControlIdentityError(f"propagation.k must be exactly {SUPPORTED_K}")
    if _require_exact_float(prop["alpha"], "propagation.alpha") != SUPPORTED_ALPHA:
        raise StitchingControlIdentityError(f"propagation.alpha must be exactly {SUPPORTED_ALPHA}")
    if _require_exact_int(prop["steps"], "propagation.steps") != SUPPORTED_STEPS:
        raise StitchingControlIdentityError(f"propagation.steps must be exactly {SUPPORTED_STEPS}")
    if prop["graph_mode"] != SUPPORTED_GRAPH_MODE:
        raise StitchingControlIdentityError(f"propagation.graph_mode must be {SUPPORTED_GRAPH_MODE!r}")
    if prop["affinity_function"] != SUPPORTED_AFFINITY_FUNCTION:
        raise StitchingControlIdentityError(f"propagation.affinity_function must be {SUPPORTED_AFFINITY_FUNCTION!r}")
    if _require_exact_float(prop["affinity_power"], "propagation.affinity_power") != SUPPORTED_AFFINITY_POWER:
        raise StitchingControlIdentityError(f"propagation.affinity_power must be exactly {SUPPORTED_AFFINITY_POWER}")
    _require_exact_string(prop["self_edge_policy"], "propagation.self_edge_policy")
    _require_exact_string(prop["fallback_row_policy"], "propagation.fallback_row_policy")
    if prop["score_stage"] != SUPPORTED_SCORE_STAGE:
        raise StitchingControlIdentityError(f"propagation.score_stage must be {SUPPORTED_SCORE_STAGE!r}")

    dataset = identity["dataset"]
    _require_exact_string(dataset["name"], "dataset.name")
    _require_exact_string(dataset["protocol"], "dataset.protocol")
    if _require_exact_int(dataset["images"], "dataset.images") != SUPPORTED_IMAGE_COUNT:
        raise StitchingControlIdentityError(f"dataset.images must be exactly {SUPPORTED_IMAGE_COUNT}")
    if _require_exact_int(dataset["classes"], "dataset.classes") != SUPPORTED_CLASS_COUNT:
        raise StitchingControlIdentityError(f"dataset.classes must be exactly {SUPPORTED_CLASS_COUNT}")
    if _require_exact_bool(dataset["background_class"], "dataset.background_class") is not False:
        raise StitchingControlIdentityError("dataset.background_class must be false")

    geometry = identity["geometry"]
    if _require_int_pair(geometry["crop"], "geometry.crop") != SUPPORTED_CROP:
        raise StitchingControlIdentityError(f"geometry.crop must be exactly {list(SUPPORTED_CROP)}")
    if _require_int_pair(geometry["stride"], "geometry.stride") != SUPPORTED_STRIDE:
        raise StitchingControlIdentityError(f"geometry.stride must be exactly {list(SUPPORTED_STRIDE)}")
    if geometry["interpolation_mode"] != SUPPORTED_INTERPOLATION_MODE:
        raise StitchingControlIdentityError(f"geometry.interpolation_mode must be {SUPPORTED_INTERPOLATION_MODE!r}")
    if _require_exact_bool(geometry["align_corners"], "geometry.align_corners") is not True:
        raise StitchingControlIdentityError("geometry.align_corners must be true")
    if geometry["crop_order"] != SUPPORTED_CROP_ORDER:
        raise StitchingControlIdentityError(f"geometry.crop_order must be {SUPPORTED_CROP_ORDER!r}")
    _require_exact_string(geometry["window_order_source"], "geometry.window_order_source")

    binding = identity["checkpoint_binding"]
    _require_exact_string(binding["projection_name"], "checkpoint_binding.projection_name")
    _require_sha256(binding["projection_sha256"], "checkpoint_binding.projection_sha256")

    variants = identity["variants"]
    names = _require_exact_string_list(variants["names"], "variants.names")
    order = _require_exact_string_list(variants["order"], "variants.order")
    if names != CANONICAL_VARIANT_NAMES:
        raise StitchingControlIdentityError(f"variants.names must be exactly {list(CANONICAL_VARIANT_NAMES)}")
    if order != CANONICAL_VARIANT_NAMES:
        raise StitchingControlIdentityError(f"variants.order must be exactly {list(CANONICAL_VARIANT_NAMES)}")
    if variants["uniform_identity_anchor"] != "uniform_probability":
        raise StitchingControlIdentityError("variants.uniform_identity_anchor must be 'uniform_probability'")
    for variant_name in CANONICAL_VARIANT_NAMES:
        section = _require_closed_mapping(variants[variant_name], _VARIANT_SECTION_KEYS, f"variants.{variant_name}")
        expected = SUPPORTED_VARIANT_CONTRACT[variant_name]
        if section["stage"] not in SUPPORTED_STAGES:
            raise StitchingControlIdentityError(f"variants.{variant_name}.stage must be one of {SUPPORTED_STAGES}")
        if section["weighting"] not in SUPPORTED_WEIGHTINGS:
            raise StitchingControlIdentityError(f"variants.{variant_name}.weighting must be one of {SUPPORTED_WEIGHTINGS}")
        if section["sigmoid_stage"] not in SUPPORTED_SIGMOID_STAGES:
            raise StitchingControlIdentityError(f"variants.{variant_name}.sigmoid_stage must be one of {SUPPORTED_SIGMOID_STAGES}")
        for field, expected_value in expected.items():
            observed = section[field] if field != "reproduces_existing_evaluator" else _require_exact_bool(section[field], f"variants.{variant_name}.reproduces_existing_evaluator")
            if observed != expected_value:
                raise StitchingControlIdentityError(
                    f"variants.{variant_name}.{field} must be exactly {expected_value!r}, observed {observed!r}"
                )

    hann = identity["hann"]
    if hann["formula"] != SUPPORTED_HANN_FORMULA:
        raise StitchingControlIdentityError("hann.formula does not match the implemented pixel-centred formula")
    if _require_exact_bool(hann["pixel_centred"], "hann.pixel_centred") is not True:
        raise StitchingControlIdentityError("hann.pixel_centred must be true")
    if _require_exact_float(hann["epsilon"], "hann.epsilon") != 0.0:
        raise StitchingControlIdentityError("hann.epsilon must be exactly 0.0 -- no hidden epsilon/floor is permitted")
    if _require_exact_bool(hann["strictly_positive_at_boundary"], "hann.strictly_positive_at_boundary") is not True:
        raise StitchingControlIdentityError("hann.strictly_positive_at_boundary must be true")

    accumulation = identity["accumulation"]
    if accumulation["numerator_dtype"] != SUPPORTED_ACCUMULATOR_DTYPE:
        raise StitchingControlIdentityError(f"accumulation.numerator_dtype must be {SUPPORTED_ACCUMULATOR_DTYPE!r}")
    if accumulation["denominator_dtype"] != SUPPORTED_ACCUMULATOR_DTYPE:
        raise StitchingControlIdentityError(f"accumulation.denominator_dtype must be {SUPPORTED_ACCUMULATOR_DTYPE!r}")
    if _require_exact_bool(accumulation["in_place_source_mutation"], "accumulation.in_place_source_mutation") is not False:
        raise StitchingControlIdentityError("accumulation.in_place_source_mutation must be false")
    if _require_exact_bool(accumulation["accumulator_storage_shared_between_variants"], "accumulation.accumulator_storage_shared_between_variants") is not False:
        raise StitchingControlIdentityError("accumulation.accumulator_storage_shared_between_variants must be false")
    _require_exact_string(accumulation["restitching_method"], "accumulation.restitching_method")

    metrics = identity["metrics"]
    names_m = _require_exact_string_list(metrics["metric_names"], "metrics.metric_names")
    if names_m != SUPPORTED_METRIC_NAMES:
        raise StitchingControlIdentityError(f"metrics.metric_names must be exactly {list(SUPPORTED_METRIC_NAMES)}")
    if metrics["unit"] != SUPPORTED_METRIC_UNIT:
        raise StitchingControlIdentityError(f"metrics.unit must be {SUPPORTED_METRIC_UNIT!r}")
    if metrics["precision_source"] != SUPPORTED_PRECISION_SOURCE:
        raise StitchingControlIdentityError(f"metrics.precision_source must be {SUPPORTED_PRECISION_SOURCE!r}")
    if _require_exact_int(metrics["class_count"], "metrics.class_count", minimum=1) != SUPPORTED_CLASS_COUNT:
        raise StitchingControlIdentityError(f"metrics.class_count must be exactly {SUPPORTED_CLASS_COUNT}")
    if metrics["paired_delta_reference_variant"] != "uniform_probability":
        raise StitchingControlIdentityError("metrics.paired_delta_reference_variant must be 'uniform_probability'")
    _require_exact_string(metrics["paired_delta_field_prefix"], "metrics.paired_delta_field_prefix")

    modes = identity["run_modes"]
    for name, expected_count in SUPPORTED_RUN_MODE_IMAGE_COUNTS.items():
        count = _require_exact_int(modes[name], f"run_modes.{name}", minimum=1)
        if count != expected_count:
            raise StitchingControlIdentityError(f"run_modes.{name} must be exactly {expected_count}, observed {count!r}")
    for name in ("pilot20_schema_name", "pilot100_schema_name", "full5000_schema_name", "checkpoint_schema_name", "per_image_stats_manifest_schema_name"):
        _require_exact_string(modes[name], f"run_modes.{name}")
    if modes["checkpoint_schema_name"] != CHECKPOINT_SCHEMA_NAME:
        raise StitchingControlIdentityError("run_modes.checkpoint_schema_name must match the implemented checkpoint schema")
    if modes["per_image_stats_manifest_schema_name"] != PER_IMAGE_STATS_SCHEMA_NAME:
        raise StitchingControlIdentityError("run_modes.per_image_stats_manifest_schema_name must match the implemented schema")
    schema_names = {modes["pilot20_schema_name"], modes["pilot100_schema_name"], modes["full5000_schema_name"]}
    if len(schema_names) != 3:
        raise StitchingControlIdentityError("pilot20/pilot100/full5000 result schema names must all be distinct")

    execution = identity["execution_contract"]
    if _require_exact_bool(execution["model_calls_equal_windows"], "execution_contract.model_calls_equal_windows") is not True:
        raise StitchingControlIdentityError("execution_contract.model_calls_equal_windows must be true")
    if _require_exact_bool(execution["graph_builds_equal_windows"], "execution_contract.graph_builds_equal_windows") is not True:
        raise StitchingControlIdentityError("execution_contract.graph_builds_equal_windows must be true")
    if _require_exact_bool(execution["propagation_calls_equal_windows"], "execution_contract.propagation_calls_equal_windows") is not True:
        raise StitchingControlIdentityError("execution_contract.propagation_calls_equal_windows must be true")
    if _require_exact_int(execution["max_interpolations_per_window"], "execution_contract.max_interpolations_per_window") != SUPPORTED_MAX_INTERPOLATIONS_PER_WINDOW:
        raise StitchingControlIdentityError(f"execution_contract.max_interpolations_per_window must be exactly {SUPPORTED_MAX_INTERPOLATIONS_PER_WINDOW}")
    if execution["checkpoint_granularity"] != SUPPORTED_CHECKPOINT_GRANULARITY:
        raise StitchingControlIdentityError(f"execution_contract.checkpoint_granularity must be {SUPPORTED_CHECKPOINT_GRANULARITY!r}")

    _require_exact_string_list(identity["prohibited"]["list"], "prohibited.list")

    return identity


def _check_git_ancestry(root: Path, commit: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"{label} is not an ancestor of HEAD"
        raise StitchingControlIdentityError(f"Git ancestry check failed for {label} ({commit}): {detail}")


def _validate_parent_identity(root: Path, identity: Mapping[str, Any]) -> Mapping[str, Any]:
    from src.k11_k12_power_evaluation_identity import load_identity as _load_power_identity

    parent = identity["parent_identity"]
    power_path = root / parent["power_evaluation_identity_path"]
    try:
        power_bytes = power_path.read_bytes()
    except OSError as error:
        raise StitchingControlIdentityError(f"cannot read power-evaluation parent identity {power_path}: {error}") from error
    if hashlib.sha256(power_bytes).hexdigest() != parent["power_evaluation_identity_sha256"]:
        raise StitchingControlIdentityError("power-evaluation parent identity file SHA256 mismatch")
    power_identity = _load_power_identity(power_path, repo_root=root)
    if power_identity["identity"]["name"] != parent["power_evaluation_identity_name"]:
        raise StitchingControlIdentityError("power-evaluation parent identity name mismatch")

    matched_path = root / parent["matched_identity_path"]
    if matched_path != root / power_identity["parent_identity"]["matched_identity_path"]:
        raise StitchingControlIdentityError("matched_identity_path disagrees with the power-evaluation identity's own parent")
    if parent["matched_identity_sha256"] != power_identity["parent_identity"]["matched_identity_sha256"]:
        raise StitchingControlIdentityError("matched_identity_sha256 disagrees with the power-evaluation identity's own parent")

    return power_identity


def validate_static_configuration(
    *, repo_root: Path | None = None, identity_path: Path | None = None, check_git: bool = True
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    if check_git:
        _check_git_ancestry(root, identity["identity"]["required_ancestor_commit"], label="identity.required_ancestor_commit")
    power_identity = _validate_parent_identity(root, identity)

    from src.k11_k12_power_evaluation_identity import validate_static_configuration as _power_preflight

    power_result = _power_preflight(repo_root=root, check_git=check_git)

    return {
        "identity_name": identity["identity"]["name"],
        "power_evaluation_identity": power_result["identity_name"],
        "matched_identity": power_result["matched_identity"],
        "required_ancestor_commit": identity["identity"]["required_ancestor_commit"],
        "pilot20_image_count": identity["run_modes"]["pilot20_image_count"],
        "pilot100_image_count": identity["run_modes"]["pilot100_image_count"],
        "full5000_image_count": identity["run_modes"]["full5000_image_count"],
        "variant_names": identity["variants"]["names"],
    }


__all__ = [
    "CANONICAL_VARIANT_NAMES",
    "CHECKPOINT_SCHEMA_NAME",
    "IDENTITY_RELATIVE_PATH",
    "PER_IMAGE_STATS_SCHEMA_NAME",
    "RUN_MODES",
    "RUN_MODE_IMAGE_COUNT_KEYS",
    "RUN_MODE_SCHEMA_KEYS",
    "StitchingControlIdentityError",
    "SUPPORTED_ACCUMULATOR_DTYPE",
    "SUPPORTED_ALPHA",
    "SUPPORTED_CLASS_COUNT",
    "SUPPORTED_CROP",
    "SUPPORTED_HANN_FORMULA",
    "SUPPORTED_IMAGE_COUNT",
    "SUPPORTED_K",
    "SUPPORTED_MAX_INTERPOLATIONS_PER_WINDOW",
    "SUPPORTED_METRIC_NAMES",
    "SUPPORTED_METRIC_UNIT",
    "SUPPORTED_PRECISION_SOURCE",
    "SUPPORTED_RUN_MODE_IMAGE_COUNTS",
    "SUPPORTED_STEPS",
    "SUPPORTED_STRIDE",
    "SUPPORTED_VARIANT_CONTRACT",
    "load_identity",
    "repository_root",
    "validate_static_configuration",
]
