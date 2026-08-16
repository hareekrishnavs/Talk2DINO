"""Static E3 evaluation identity and result verification utilities.

This module deliberately avoids importing MMCV, Torch, model code, or dataset
code.  It is safe to use on a login node for configuration/result preflight.
"""

from __future__ import annotations

import ast
import json
import math
import re
import runpy
import subprocess
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


IDENTITY_RELATIVE_PATH = Path(
    "evaluation_identities/e3_paired_soft_routing.toml"
)
METRIC_NAMES = ("aAcc", "mIoU", "mAcc")
IMAGE_COUNT_KEYS = (
    "evaluated_images",
    "image_count",
    "images",
    "num_images",
    "test_images",
)
FORBIDDEN_MODE_TOKENS = (
    "rwr",
    "diffusion",
    "cover_dr",
    "cover-dr",
    "coverdr",
    "graph_repair",
    "graph-repair",
    "graph repair",
)
MODEL_FLAG_NAMES = (
    "avg_self_attn_token",
    "disentangled_self_attn_token",
    "pre_trained",
    "is_eval",
    "use_avg_text_token",
    "keep_cls",
    "keep_end_seq",
    "with_bg_clean",
)
DATASET_CLASS_OVERRIDE_KEYS = frozenset(
    {
        "classes",
        "palette",
        "metainfo",
        "meta_info",
        "metadata",
        "class_names",
        "custom_classes",
    }
)
_NUMBER_TOKEN = (
    r"[-+]?(?:nan|inf(?:inity)?|"
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


class E3IdentityError(ValueError):
    """Raised when the canonical configuration or a result is not E3-identical."""


@dataclass(frozen=True)
class ParsedMetrics:
    values: Mapping[str, float]
    image_count: int | None
    printed_tokens: Mapping[str, str] | None = None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_identity(
    path: Path | None = None, *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity_path = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with identity_path.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise E3IdentityError(
            f"cannot load E3 identity specification {identity_path}: {error}"
        ) from error

    expected_sections = {
        "format_version",
        "identity_name",
        "base_commit",
        "expected_branch_ancestry",
        "seed",
        "dataset",
        "model",
        "projection",
        "evaluation",
        "expected_metrics",
        "tolerances",
    }
    if set(identity) != expected_sections:
        raise E3IdentityError(
            "E3 identity specification has an unexpected top-level schema"
        )
    if identity["format_version"] != "talk2dino-e3-evaluation-identity-v1":
        raise E3IdentityError("unsupported E3 identity specification version")
    if set(identity["expected_metrics"]) != set(METRIC_NAMES):
        raise E3IdentityError("E3 identity metric schema is incomplete")
    model_flags = identity.get("model", {}).get("flags")
    if not isinstance(model_flags, Mapping) or set(model_flags) != set(MODEL_FLAG_NAMES):
        raise E3IdentityError("E3 identity model flag schema is incomplete")
    if any(type(model_flags[name]) is not bool for name in MODEL_FLAG_NAMES):
        raise E3IdentityError("E3 identity model flags must be booleans")
    dataset_identity = identity.get("dataset", {})
    classes = dataset_identity.get("classes")
    if isinstance(classes, bool) or not isinstance(classes, int) or classes <= 0:
        raise E3IdentityError("E3 identity dataset class count must be a positive integer")
    if type(dataset_identity.get("background_class")) is not bool:
        raise E3IdentityError("E3 identity background_class must be boolean")
    projection_identity = identity.get("projection", {})
    if projection_identity.get("checkpoint_loading_required") is not True:
        raise E3IdentityError(
            "E3 identity must require projection checkpoint loading"
        )
    for name in METRIC_NAMES:
        _finite_number(identity["expected_metrics"][name], f"expected {name}")
    tolerance = _finite_number(
        identity["tolerances"]["structured_absolute"],
        "structured metric tolerance",
    )
    if tolerance <= 0:
        raise E3IdentityError("structured metric tolerance must be positive")
    decimals = identity["tolerances"]["rounded_log_minimum_decimal_places"]
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 1:
        raise E3IdentityError("rounded-log decimal precision must be a positive integer")
    allowance = _finite_number(
        identity["tolerances"]["two_decimal_log_reproducibility_allowance"],
        "two-decimal log reproducibility allowance",
    )
    if allowance < 0:
        raise E3IdentityError(
            "two-decimal log reproducibility allowance must be non-negative"
        )
    return identity


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise E3IdentityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise E3IdentityError(f"{label} must be finite")
    return result


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise E3IdentityError(
            "PyYAML is required because Talk2DINO's canonical configs are YAML"
        ) from error
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise E3IdentityError(f"cannot load YAML configuration {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise E3IdentityError(f"YAML configuration must be a mapping: {path}")
    return dict(value)


def _load_yaml_with_bases(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise E3IdentityError(f"cyclic YAML base configuration: {path}")
    value = _load_yaml(path)
    bases = value.pop("_base_", None)
    if bases is None:
        return value
    base_names = [bases] if isinstance(bases, str) else bases
    if not isinstance(base_names, Sequence) or isinstance(base_names, (bytes, str)):
        raise E3IdentityError(f"invalid _base_ declaration in {path}")
    merged: dict[str, Any] = {}
    for base_name in base_names:
        if not isinstance(base_name, str):
            raise E3IdentityError(f"invalid _base_ entry in {path}")
        merged = _deep_merge(
            merged,
            _load_yaml_with_bases(path.parent / base_name, (*stack, path)),
        )
    return _deep_merge(merged, value)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise E3IdentityError(
            f"{label} mismatch: expected {expected!r}, observed {actual!r}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise E3IdentityError(f"{label} must be a mapping")
    return value


def _integer_pair(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise E3IdentityError(f"{label} must be a pair of integers")
    return int(value[0]), int(value[1])


def _load_constructor_defaults(
    root: Path,
    model_identity: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Any]:
    source_path = root / str(model_identity["constructor_path"])
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as error:
        raise E3IdentityError(
            f"cannot establish DINOText constructor defaults from {source_path}: {error}"
        ) from error

    class_name = str(model_identity["constructor_class"])
    constructor = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            constructor = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "__init__"
                ),
                None,
            )
            break
    if constructor is None:
        raise E3IdentityError(
            f"cannot establish DINOText constructor defaults: class {class_name!r} "
            f"or its __init__ is missing from {source_path}"
        )

    arguments = [*constructor.args.posonlyargs, *constructor.args.args]
    default_nodes: list[ast.expr | None] = [None] * (
        len(arguments) - len(constructor.args.defaults)
    ) + list(constructor.args.defaults)
    defaults_by_name: dict[str, ast.expr | None] = {
        argument.arg: default
        for argument, default in zip(arguments, default_nodes)
    }
    defaults_by_name.update(
        {
            argument.arg: default
            for argument, default in zip(
                constructor.args.kwonlyargs,
                constructor.args.kw_defaults,
            )
        }
    )

    resolved: dict[str, Any] = {}
    for name in names:
        default_node = defaults_by_name.get(name)
        if default_node is None:
            raise E3IdentityError(
                f"cannot establish production default for model.{name} from {source_path}"
            )
        try:
            resolved[name] = ast.literal_eval(default_node)
        except (ValueError, TypeError) as error:
            raise E3IdentityError(
                f"production default for model.{name} is not statically resolvable "
                f"in {source_path}"
            ) from error
    return resolved


def _reject_dataset_class_overrides(
    dataset: Mapping[str, Any],
    location: str = "dataset test split",
) -> None:
    for key in sorted(dataset, key=str):
        normalized = str(key).strip().lower().replace("-", "_")
        if normalized in DATASET_CLASS_OVERRIDE_KEYS:
            raise E3IdentityError(
                f"custom dataset class metadata override is forbidden at {location}.{key}"
            )

    for nested_key in ("dataset", "datasets"):
        nested = dataset.get(nested_key)
        if isinstance(nested, Mapping):
            _reject_dataset_class_overrides(nested, f"{location}.{nested_key}")
        elif isinstance(nested, (list, tuple)):
            for index, child in enumerate(nested):
                if isinstance(child, Mapping):
                    _reject_dataset_class_overrides(
                        child,
                        f"{location}.{nested_key}[{index}]",
                    )


def _resolve_dataset_class_names(
    dataset_identity: Mapping[str, Any],
) -> tuple[str, ...]:
    distribution_name = str(dataset_identity["class_metadata_distribution"])
    relative_source = str(dataset_identity["class_metadata_path"])
    class_name = str(dataset_identity["dataset_type"])
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as error:
        raise E3IdentityError(
            f"cannot establish canonical dataset class metadata: Python distribution "
            f"{distribution_name!r} is not installed"
        ) from error

    matching_files = [
        item
        for item in (distribution.files or ())
        if item.as_posix() == relative_source
    ]
    if len(matching_files) != 1:
        raise E3IdentityError(
            f"cannot establish canonical dataset class metadata: expected exactly one "
            f"{relative_source!r} in {distribution_name!r}, found {len(matching_files)}"
        )
    source_path = Path(distribution.locate_file(matching_files[0]))
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as error:
        raise E3IdentityError(
            f"cannot establish canonical dataset class metadata from {source_path}: {error}"
        ) from error

    classes_node = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.Assign):
                targets = child.targets
                value = child.value
            elif isinstance(child, ast.AnnAssign):
                targets = [child.target]
                value = child.value
            else:
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "CLASSES"
                for target in targets
            ):
                classes_node = value
                break
        break
    if classes_node is None:
        raise E3IdentityError(
            f"cannot establish canonical dataset class metadata: {class_name}.CLASSES "
            f"is missing from {source_path}"
        )
    try:
        classes = ast.literal_eval(classes_node)
    except (ValueError, TypeError) as error:
        raise E3IdentityError(
            f"cannot establish canonical dataset class metadata: {class_name}.CLASSES "
            f"is not statically resolvable in {source_path}"
        ) from error
    if (
        not isinstance(classes, (list, tuple))
        or not classes
        or any(not isinstance(name, str) for name in classes)
    ):
        raise E3IdentityError(
            f"cannot establish canonical dataset class metadata: {class_name}.CLASSES "
            "must be a non-empty sequence of strings"
        )
    return tuple(classes)


def _reject_forbidden_modes(value: Any, location: str = "configuration") -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            normalized_key = str(key).lower()
            if any(token in normalized_key for token in FORBIDDEN_MODE_TOKENS):
                raise E3IdentityError(
                    f"forbidden RWR/diffusion/COVER key is present at {location}.{key}"
                )
            _reject_forbidden_modes(value[key], f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_modes(item, f"{location}[{index}]")
    elif isinstance(value, str):
        normalized_value = value.lower()
        if any(token in normalized_value for token in FORBIDDEN_MODE_TOKENS):
            raise E3IdentityError(
                f"forbidden RWR/diffusion/COVER mode is active at {location}"
            )


def _check_git_ancestry(root: Path, identity: Mapping[str, Any]) -> None:
    command = [
        "git",
        "merge-base",
        "--is-ancestor",
        str(identity["base_commit"]),
        "HEAD",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise E3IdentityError(f"cannot execute Git ancestry check: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or "required base is not an ancestor of HEAD"
        raise E3IdentityError(
            f"Git ancestry check failed for {identity['base_commit']}: {detail}"
        )


def validate_static_configuration(
    *,
    repo_root: Path | None = None,
    identity_path: Path | None = None,
    eval_config: Path | None = None,
    eval_base_config: Path | None = None,
    check_checkpoint: bool = False,
    dataset_root: Path | None = None,
    weight_dir: Path | None = None,
    check_git: bool = True,
) -> dict[str, Any]:
    """Validate the effective E3 configuration without importing model/data code."""

    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    evaluation_identity = _mapping(identity["evaluation"], "identity.evaluation")
    eval_path = Path(eval_config) if eval_config is not None else root / evaluation_identity["config_path"]
    base_path = (
        Path(eval_base_config)
        if eval_base_config is not None
        else root / evaluation_identity["base_config_path"]
    )
    if check_git:
        _check_git_ancestry(root, identity)

    effective = _deep_merge(
        _load_yaml_with_bases(eval_path),
        _load_yaml_with_bases(base_path),
    )
    _reject_forbidden_modes(effective)
    model = _mapping(effective.get("model"), "effective model configuration")
    evaluate = _mapping(
        effective.get("evaluate"), "effective evaluation configuration"
    )
    expected_model = _mapping(identity["model"], "identity.model")
    expected_projection = _mapping(identity["projection"], "identity.projection")
    expected_dataset = _mapping(identity["dataset"], "identity.dataset")
    expected_model_flags = _mapping(expected_model["flags"], "identity.model.flags")

    _require_equal(model.get("type"), expected_model["type"], "model type")
    _require_equal(model.get("model_name"), expected_model["name"], "DINO backbone")
    _require_equal(
        model.get("resize_dim"),
        expected_model["resize_dimension"],
        "model resize dimension",
    )
    _require_equal(
        model.get("clip_model_name"),
        expected_model["clip_model_name"],
        "CLIP model",
    )
    _require_equal(
        model.get("proj_class"), expected_projection["class"], "projection class"
    )
    _require_equal(
        model.get("proj_name"), expected_projection["name"], "projection checkpoint name"
    )
    _require_equal(
        model.get("proj_model"), expected_projection["model"], "projection model"
    )
    _require_equal(evaluate.get("task"), [expected_dataset["task"]], "evaluation task")
    _require_equal(evaluate.get("pamr"), False, "PAMR setting")
    _require_equal(
        evaluate.get("template"),
        evaluation_identity["template"],
        "evaluation prompt template",
    )

    constructor_defaults = _load_constructor_defaults(
        root,
        expected_model,
        MODEL_FLAG_NAMES,
    )
    effective_model_flags: dict[str, bool] = {}
    for name in MODEL_FLAG_NAMES:
        observed = model[name] if name in model else constructor_defaults[name]
        expected = expected_model_flags[name]
        if type(observed) is not bool:
            raise E3IdentityError(
                f"model.{name} mismatch: expected boolean {expected!r}, "
                f"observed {observed!r}"
            )
        if observed is not expected:
            if name == "pre_trained" and expected_projection["checkpoint_loading_required"]:
                raise E3IdentityError(
                    "E3 projection checkpoint would not be loaded: "
                    f"model.pre_trained expected {expected!r}, observed {observed!r}"
                )
            raise E3IdentityError(
                f"model.{name} mismatch: expected {expected!r}, observed {observed!r}"
            )
        effective_model_flags[name] = observed

    configured_dataset_path = evaluate.get(expected_dataset["task"])
    _require_equal(
        configured_dataset_path,
        expected_dataset["config_path"],
        "COCO-Stuff dataset configuration path",
    )
    dataset_config_path = root / str(configured_dataset_path)
    try:
        dataset_config = runpy.run_path(str(dataset_config_path))
    except (OSError, RuntimeError, SyntaxError) as error:
        raise E3IdentityError(
            f"cannot read dataset configuration {dataset_config_path}: {error}"
        ) from error
    _reject_forbidden_modes(dataset_config, "dataset configuration")
    _require_equal(
        dataset_config.get("dataset_type"),
        expected_dataset["dataset_type"],
        "dataset type",
    )
    _require_equal(
        dataset_config.get("data_root"),
        expected_dataset["configured_root"],
        "configured dataset root",
    )
    test_data = _mapping(
        _mapping(dataset_config.get("data"), "dataset data").get("test"),
        "dataset test split",
    )
    _reject_dataset_class_overrides(test_data)
    _require_equal(test_data.get("img_dir"), expected_dataset["image_dir"], "validation image directory")
    _require_equal(test_data.get("ann_dir"), expected_dataset["annotation_dir"], "validation annotation directory")
    runtime_classes = _resolve_dataset_class_names(expected_dataset)
    _require_equal(
        len(runtime_classes),
        expected_dataset["classes"],
        "runtime dataset class count",
    )
    runtime_background = runtime_classes[0].strip().lower() == "background"
    _require_equal(
        runtime_background,
        expected_dataset["background_class"],
        "runtime dataset background class",
    )
    test_cfg = _mapping(dataset_config.get("test_cfg"), "dataset test_cfg")
    _require_equal(test_cfg.get("mode"), evaluation_identity["mode"], "inference mode")
    _require_equal(
        _integer_pair(test_cfg.get("crop_size"), "sliding crop"),
        _integer_pair(evaluation_identity["crop"], "identity sliding crop"),
        "sliding crop",
    )
    _require_equal(
        _integer_pair(test_cfg.get("stride"), "sliding stride"),
        _integer_pair(evaluation_identity["stride"], "identity sliding stride"),
        "sliding stride",
    )

    projection_path = root / expected_projection["config_path"]
    projection_config = _mapping(
        _load_yaml(projection_path).get("model"), "projection configuration"
    )
    _reject_forbidden_modes(projection_config, "projection configuration")
    _require_equal(
        projection_config.get("alignment_strategy"),
        expected_projection["alignment_strategy"],
        "projection alignment strategy",
    )
    _require_equal(
        _finite_number(
            projection_config.get("routing_temperature"),
            "projection routing temperature",
        ),
        _finite_number(
            expected_projection["routing_temperature"],
            "identity routing temperature",
        ),
        "routing temperature",
    )

    checkpoint_path = root / expected_projection["checkpoint_path"]
    derived_checkpoint = root / "weights" / f"{model['proj_name']}.pth"
    _require_equal(
        derived_checkpoint.resolve(),
        checkpoint_path.resolve(),
        "projection checkpoint derivation",
    )
    if check_checkpoint and not checkpoint_path.is_file():
        raise E3IdentityError(
            f"missing E3 projection checkpoint: {checkpoint_path}; place the canonical "
            "checkpoint at that path before evaluation"
        )

    if dataset_root is not None:
        external_dataset = Path(dataset_root)
        for relative, label in (
            (expected_dataset["image_dir"], "validation images"),
            (expected_dataset["annotation_dir"], "validation annotations"),
        ):
            required = external_dataset / relative
            if not required.is_dir():
                raise E3IdentityError(
                    f"missing COCO-Stuff {label} directory: {required}"
                )

    if weight_dir is not None:
        external_weights = Path(weight_dir)
        for filename, label in (
            (expected_model["backbone_checkpoint_name"], "DINOv2 backbone"),
            (expected_model["clip_checkpoint_name"], "CLIP model"),
        ):
            required = external_weights / filename
            if not required.is_file():
                raise E3IdentityError(
                    f"missing {label} checkpoint: {required}"
                )

    return {
        "identity_name": identity["identity_name"],
        "evaluation_config": str(eval_path),
        "evaluation_base_config": str(base_path),
        "dataset_config": str(dataset_config_path),
        "projection_config": str(projection_path),
        "projection_checkpoint": str(checkpoint_path),
        "projection_checkpoint_loading": effective_model_flags["pre_trained"],
        "evaluation_template": evaluate.get("template"),
        "dataset_classes": len(runtime_classes),
        "dataset_first_class": runtime_classes[0],
        "dataset_background_class": runtime_background,
        "effective_model_flags": effective_model_flags,
        "checkpoint_checked": check_checkpoint,
        "dataset_checked": dataset_root is not None,
        "external_weights_checked": weight_dir is not None,
    }


def _collect_structured_metric_blocks(value: Any) -> list[Mapping[str, Any]]:
    blocks: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        present = set(METRIC_NAMES).intersection(value)
        if present:
            missing = set(METRIC_NAMES) - present
            if missing:
                raise E3IdentityError(
                    f"structured result metric block is missing: {', '.join(sorted(missing))}"
                )
            blocks.append(value)
        for key in sorted(value, key=str):
            blocks.extend(_collect_structured_metric_blocks(value[key]))
    elif isinstance(value, list):
        for item in value:
            blocks.extend(_collect_structured_metric_blocks(item))
    return blocks


def _collect_image_counts(value: Any) -> list[int]:
    counts: list[int] = []
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            item = value[key]
            if key in IMAGE_COUNT_KEYS:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise E3IdentityError(f"result {key} must be an integer")
                counts.append(item)
            else:
                counts.extend(_collect_image_counts(item))
    elif isinstance(value, list):
        for item in value:
            counts.extend(_collect_image_counts(item))
    return counts


def parse_structured_metrics(path: Path) -> ParsedMetrics:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                E3IdentityError(f"structured result contains non-finite value {token}")
            ),
        )
    except E3IdentityError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise E3IdentityError(f"cannot load structured metrics {path}: {error}") from error
    blocks = _collect_structured_metric_blocks(value)
    if not blocks:
        raise E3IdentityError("structured result has no complete aAcc/mIoU/mAcc block")
    if len(blocks) != 1:
        raise E3IdentityError(
            f"structured result has {len(blocks)} duplicate or ambiguous metric blocks"
        )
    values = {
        name: _finite_number(blocks[0][name], f"observed {name}")
        for name in METRIC_NAMES
    }
    counts = sorted(set(_collect_image_counts(value)))
    if len(counts) > 1:
        raise E3IdentityError(f"structured result has ambiguous image counts: {counts}")
    return ParsedMetrics(values=values, image_count=counts[0] if counts else None)


def _pipe_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_log_table_blocks(lines: Sequence[str]) -> list[ParsedMetrics]:
    blocks: list[ParsedMetrics] = []
    for index, line in enumerate(lines):
        header = _pipe_cells(line) if "|" in line else []
        if not set(METRIC_NAMES).issubset(header):
            continue
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if stripped and set(stripped) <= {"-", ":", "+", "|", " "}:
                continue
            if "|" not in candidate:
                if stripped:
                    break
                continue
            cells = _pipe_cells(candidate)
            if cells and all(set(cell) <= {"-", ":", "+"} for cell in cells):
                continue
            if len(cells) != len(header):
                break
            tokens = {name: cells[header.index(name)] for name in METRIC_NAMES}
            try:
                values = {name: float(tokens[name]) for name in METRIC_NAMES}
            except ValueError:
                break
            blocks.append(ParsedMetrics(values=values, image_count=None, printed_tokens=tokens))
            break
    return blocks


def _parse_log_inline_blocks(lines: Sequence[str]) -> list[ParsedMetrics]:
    pattern = re.compile(
        rf"\b(aAcc|mIoU|mAcc)\b\s*[:=]\s*({_NUMBER_TOKEN})",
        flags=re.IGNORECASE,
    )
    canonical = {name.lower(): name for name in METRIC_NAMES}
    blocks: list[ParsedMetrics] = []
    for line in lines:
        found: dict[str, str] = {}
        for name, token in pattern.findall(line):
            key = canonical[name.lower()]
            if key in found:
                raise E3IdentityError(f"duplicate {key} value in one log metric block")
            found[key] = token
        if found and set(found) != set(METRIC_NAMES):
            continue
        if set(found) == set(METRIC_NAMES):
            blocks.append(
                ParsedMetrics(
                    values={name: float(found[name]) for name in METRIC_NAMES},
                    image_count=None,
                    printed_tokens=found,
                )
            )
    return blocks


def parse_log_metrics(path: Path) -> ParsedMetrics:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise E3IdentityError(f"cannot read evaluation log {path}: {error}") from error
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = text.splitlines()
    blocks = _parse_log_table_blocks(lines) + _parse_log_inline_blocks(lines)
    if not blocks:
        raise E3IdentityError("evaluation log has no complete aAcc/mIoU/mAcc metric block")
    if len(blocks) != 1:
        raise E3IdentityError(
            f"evaluation log has {len(blocks)} duplicate or ambiguous metric blocks"
        )
    image_counts = sorted(
        {
            int(value)
            for value in re.findall(r"mIoU\s+of\s+(\d+)\s+test\s+images", text)
        }
    )
    if len(image_counts) > 1:
        raise E3IdentityError(f"evaluation log has ambiguous image counts: {image_counts}")
    block = blocks[0]
    return ParsedMetrics(
        values=block.values,
        image_count=image_counts[0] if image_counts else None,
        printed_tokens=block.printed_tokens,
    )


def _validate_percentage_scale(values: Mapping[str, float]) -> None:
    observed = [values[name] for name in METRIC_NAMES]
    if all(0 <= value <= 1 for value in observed):
        raise E3IdentityError(
            "metric scale mismatch: observed [0,1] fractions, expected [0,100] percentages"
        )
    if any(0 <= value <= 1 for value in observed) and any(value > 1 for value in observed):
        raise E3IdentityError("metric scale is inconsistent within the result block")
    if any(value < 0 or value > 100 for value in observed):
        raise E3IdentityError("percentage metrics must be in the [0,100] range")


def _rounded_tolerance(
    token: str,
    minimum_decimal_places: int,
    two_decimal_reproducibility_allowance: float,
) -> float:
    try:
        decimal_value = Decimal(token)
    except InvalidOperation as error:
        raise E3IdentityError(f"invalid rounded metric token: {token!r}") from error
    if not decimal_value.is_finite():
        raise E3IdentityError(f"observed rounded metric must be finite: {token}")
    decimal_places = max(0, -decimal_value.as_tuple().exponent)
    if decimal_places < minimum_decimal_places:
        raise E3IdentityError(
            f"rounded log metric {token!r} has {decimal_places} decimal places; "
            f"at least {minimum_decimal_places} are required"
        )
    rounding_uncertainty = Decimal("0.5") * (
        Decimal(10) ** decimal_value.as_tuple().exponent
    )
    if decimal_places == 2:
        rounding_uncertainty += Decimal(
            str(two_decimal_reproducibility_allowance)
        )
    return float(rounding_uncertainty)


def verify_metrics(
    parsed: ParsedMetrics,
    identity: Mapping[str, Any],
    *,
    source_kind: str,
) -> str:
    values = {
        name: _finite_number(parsed.values[name], f"observed {name}")
        for name in METRIC_NAMES
    }
    _validate_percentage_scale(values)
    expected_images = identity["dataset"]["images"]
    if parsed.image_count is not None and parsed.image_count != expected_images:
        raise E3IdentityError(
            f"image count mismatch: expected {expected_images}, observed {parsed.image_count}"
        )

    mismatches: list[str] = []
    for name in METRIC_NAMES:
        expected = float(identity["expected_metrics"][name])
        observed = values[name]
        if source_kind == "structured":
            tolerance = float(identity["tolerances"]["structured_absolute"])
        elif source_kind == "log":
            if parsed.printed_tokens is None:
                raise E3IdentityError("rounded log tokens are unavailable")
            tolerance = _rounded_tolerance(
                parsed.printed_tokens[name],
                int(identity["tolerances"]["rounded_log_minimum_decimal_places"]),
                float(
                    identity["tolerances"][
                        "two_decimal_log_reproducibility_allowance"
                    ]
                ),
            )
        else:
            raise E3IdentityError(f"unknown metric source kind: {source_kind}")
        delta = abs(observed - expected)
        if delta > tolerance + 1e-12:
            mismatches.append(
                f"{name}: expected={expected:.6f}, observed={observed:.12g}, "
                f"tolerance={tolerance:.12g}, absolute_delta={delta:.12g}"
            )
    if mismatches:
        raise E3IdentityError("E3 identity mismatch:\n  " + "\n  ".join(mismatches))
    count_text = str(parsed.image_count) if parsed.image_count is not None else "not reported"
    return (
        "E3 IDENTITY PASS "
        f"images={count_text} aAcc={values['aAcc']:.6f} "
        f"mIoU={values['mIoU']:.6f} mAcc={values['mAcc']:.6f}"
    )


def verify_result(
    path: Path,
    *,
    source_kind: str,
    identity_path: Path | None = None,
    repo_root: Path | None = None,
) -> str:
    identity = load_identity(identity_path, repo_root=repo_root)
    parsed = (
        parse_structured_metrics(path)
        if source_kind == "structured"
        else parse_log_metrics(path)
    )
    return verify_metrics(parsed, identity, source_kind=source_kind)


__all__ = [
    "E3IdentityError",
    "IDENTITY_RELATIVE_PATH",
    "ParsedMetrics",
    "load_identity",
    "parse_log_metrics",
    "parse_structured_metrics",
    "repository_root",
    "validate_static_configuration",
    "verify_metrics",
    "verify_result",
]
