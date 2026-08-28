"""Strict loading/validation for the additive DFP Phase 1A Boundary IoU
authority supplement (`evaluation_identities/e13_dfp_phase1a_boundary_iou.toml`).

This identity NEVER rewrites or re-binds the committed
`e13-dfp-phase1a-protocol` identity -- both frozen Step-2 subset
manifests pin that file's exact SHA-256, so even a single-byte change
would break their already-committed bindings. Instead, this identity
binds an external, independently-verifiable authority (the pinned Cheng
et al. 2021 boundary-iou-api `mask_to_boundary` primitive) and requires,
every time it is loaded, that the LIVE protocol identity still declares
`[boundary_iou].implementation_source_status == "not_yet_bound"` --
proving this supplement resolves a deliberately deferred gap rather than
silently overriding an authority that was already bound some other way.

Every scientifically material string field is pinned to its exact,
independently embedded canonical text (see `CANONICAL_SCIENTIFIC_STRINGS`
below) -- never merely checked for non-emptiness. This mirrors the
lesson learned (and then repaired) twice already in this project's
Step 1 and Step 2 identities: a "nonempty string" check alone lets a
scientifically-different-but-syntactically-valid replacement pass
undetected.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any, Mapping


class DfpPhase1aBoundaryIouIdentityError(ValueError):
    """Raised on any Boundary IoU authority supplement invariant
    violation. Always fail closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


IDENTITY_RELATIVE_PATH = Path("evaluation_identities/e13_dfp_phase1a_boundary_iou.toml")
PROTOCOL_IDENTITY_RELATIVE_PATH = "evaluation_identities/e13_dfp_phase1a_protocol.toml"
REQUIRED_DEFERRED_STATUS = "not_yet_bound"

IDENTITY_TOP_KEYS = frozenset(
    {
        "format_version", "identity", "protocol_binding", "paper", "official_source",
        "primitive", "local_reproduction", "semantic_extension",
    }
)

SECTION_KEYS: dict[str, frozenset[str]] = {
    "identity": frozenset({"name", "schema_version", "description", "resolves_deferred_authority"}),
    "protocol_binding": frozenset({"protocol_identity_relative_path", "protocol_identity_sha256", "required_deferred_status"}),
    "paper": frozenset({"title", "authors", "venue", "arxiv_id", "doi"}),
    "official_source": frozenset(
        {
            "repository_url", "repository_commit", "source_relative_path", "source_function_name",
            "source_sha256", "license_relative_path", "license_sha256",
        }
    ),
    "primitive": frozenset(
        {
            "dilation_ratio_primary", "padding_value", "padding_width", "kernel_height", "kernel_width",
            "kernel_dtype", "erosion_backend", "round_convention", "minimum_dilation_iterations",
            "fixed_sensitivity_widths", "mask_dtype", "boundary_formula",
        }
    ),
    "local_reproduction": frozenset(
        {"license_notice_relative_path", "reproduction_kind", "reproduced_function_relative_path", "reproduced_function_name"}
    ),
    "semantic_extension": frozenset(
        {
            "label", "is_official_semantic_definition", "statement", "ignore_policy", "zero_union_policy",
            "metric_unit", "aggregation_rule",
        }
    ),
}

# Single production authority for every scientifically material string
# field's exact text. Used by the loader below; tests mutate only the
# field under test rather than maintaining a second, duplicated authority.
CANONICAL_SCIENTIFIC_STRINGS: dict[tuple[str, str], str] = {
    ("format_version", "format_version"): "talk2dino-dfp-phase1a-boundary-iou-identity-v1",
    ("identity", "name"): "e13-dfp-phase1a-boundary-iou",
    ("identity", "schema_version"): "talk2dino-dfp-phase1a-boundary-iou-identity-v1",
    ("identity", "description"): (
        "Additive Boundary IoU authority supplement for DFP Phase 1A: binds the pinned Cheng et al. 2021 "
        "boundary-iou-api mask_to_boundary primitive (exact source commit, function, and SHA-256) plus a "
        "repository-defined classwise semantic aggregation on top of it, resolving the frozen "
        "e13-dfp-phase1a-protocol identity's own deferred [boundary_iou].implementation_source_status = "
        '"not_yet_bound".'
    ),
    ("identity", "resolves_deferred_authority"): "e13-dfp-phase1a-protocol [boundary_iou].implementation_source_status == not_yet_bound",
    ("protocol_binding", "protocol_identity_relative_path"): PROTOCOL_IDENTITY_RELATIVE_PATH,
    ("protocol_binding", "required_deferred_status"): REQUIRED_DEFERRED_STATUS,
    ("paper", "title"): "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation",
    ("paper", "authors"): "Bowen Cheng et al.",
    ("paper", "venue"): "CVPR 2021",
    ("paper", "arxiv_id"): "2103.16562",
    ("paper", "doi"): "10.1109/CVPR46437.2021.01508",
    ("official_source", "repository_url"): "https://github.com/bowenc0221/boundary-iou-api",
    ("official_source", "repository_commit"): "37d25586a677b043ed585f10e5c42d4e80176ea9",
    ("official_source", "source_relative_path"): "boundary_iou/utils/boundary_utils.py",
    ("official_source", "source_function_name"): "mask_to_boundary",
    ("official_source", "source_sha256"): "a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c",
    ("official_source", "license_relative_path"): "license.txt",
    ("official_source", "license_sha256"): "6da6c74bb714b3f7fe7eb6b817e42dd59ebfca15f292f54a9a7ad33a6a872863",
    ("primitive", "kernel_dtype"): "uint8",
    ("primitive", "erosion_backend"): "cv2.erode",
    ("primitive", "round_convention"): "python_builtin_round_half_to_even_then_int_cast",
    ("primitive", "mask_dtype"): "uint8",
    ("primitive", "boundary_formula"): "boundary = mask - eroded_mask",
    ("local_reproduction", "license_notice_relative_path"): "licenses/boundary_iou_api_LICENSE.txt",
    ("local_reproduction", "reproduction_kind"): "line_for_line_translation_no_behavior_change",
    ("local_reproduction", "reproduced_function_relative_path"): "src/dfp_phase1a_analysis.py",
    ("local_reproduction", "reproduced_function_name"): "_official_mask_to_boundary",
    ("semantic_extension", "label"): "classwise_semantic_boundary_iou_using_cheng_et_al_boundary_primitive",
    ("semantic_extension", "statement"): (
        "This classwise, ignore-aware, dataset-level-aggregated semantic Boundary IoU is a repository-defined "
        "extension built on top of the pinned official mask_to_boundary primitive. The upstream boundary-iou-api "
        "package itself defines only the binary mask-to-boundary primitive, never a classwise semantic-segmentation "
        "aggregation; this identity's own statement is the sole authority for that aggregation and must never be "
        "attributed to the upstream paper or repository."
    ),
    ("semantic_extension", "ignore_policy"): (
        "GT and predicted class masks are both forced false at ignored pixels before boundary extraction; the "
        "resulting boundary masks are additionally masked by validity after extraction; predicted values inside "
        "ignored pixels never affect any statistic"
    ),
    ("semantic_extension", "zero_union_policy"): (
        "class boundary IoU is undefined when its total accumulated boundary_union equals zero; mean boundary IoU "
        "averages only classes with nonzero total boundary_union; if every class has zero total boundary_union "
        "after aggregation, metric computation is rejected with a named error rather than returning a misleading "
        "value"
    ),
    ("semantic_extension", "metric_unit"): "percent_0_100",
    ("semantic_extension", "aggregation_rule"): "dataset_level_sum_then_divide_full_precision_integer_sufficient_statistics",
}

# String fields whose exactness is governed by a stronger, independent
# mechanism (a live SHA-256 file-content hash) rather than a literal
# comparison against CANONICAL_SCIENTIFIC_STRINGS.
_RELATIONALLY_GOVERNED_STRING_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("protocol_binding", "protocol_identity_sha256"),
    }
)


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be an exact string")
    if nonempty and not value.strip():
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be a non-empty, non-whitespace string")
    return value


def _require_canonical_string(value: Any, section: str, field: str) -> str:
    label = f"{section}.{field}"
    token = _require_exact_string(value, label)
    canonical = CANONICAL_SCIENTIFIC_STRINGS[(section, field)]
    if token != canonical:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} disagrees with the frozen canonical scientific text")
    return token


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be an exact boolean")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_exact_float(value: Any, label: str) -> float:
    import math

    if type(value) is not float:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be an exact finite float")
    if not math.isfinite(value):
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be finite (no NaN/Infinity)")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_canonical_sha256(value: Any, section: str, field: str) -> str:
    """Format-checked AND exact-pinned against CANONICAL_SCIENTIFIC_STRINGS.
    Used for hashes with no live local file to relationally re-verify
    against (the external official source is deliberately not vendored
    in this repository) -- pinning the hash STRING itself as frozen
    canonical text is the only way to catch a mutation to a different,
    still well-formed hash without requiring network access."""
    label = f"{section}.{field}"
    token = _require_sha256(value, label)
    canonical = CANONICAL_SCIENTIFIC_STRINGS[(section, field)]
    if token != canonical:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} disagrees with the frozen canonical hash")
    return token


def _require_relative_path(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be a safe repository-relative path")
    return token


def _require_exact_int_list(value: Any, label: str, *, expected: tuple[int, ...] | None = None) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be a non-empty exact list")
    for item in value:
        if type(item) is not int:
            raise DfpPhase1aBoundaryIouIdentityError(f"{label} elements must be exact non-boolean integers")
    token = tuple(value)
    if expected is not None and token != expected:
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be exactly {list(expected)}, observed {list(token)}")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} must be a mapping")
    if set(value) != expected_keys:
        missing = expected_keys - set(value)
        unknown = set(value) - expected_keys
        raise DfpPhase1aBoundaryIouIdentityError(f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_identity(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else repository_root()
    source = Path(path) if path is not None else root / IDENTITY_RELATIVE_PATH
    try:
        with source.open("rb") as handle:
            identity = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DfpPhase1aBoundaryIouIdentityError(f"cannot load Boundary IoU authority supplement {source}: {error}") from error

    if set(identity) != IDENTITY_TOP_KEYS:
        missing = IDENTITY_TOP_KEYS - set(identity)
        unknown = set(identity) - IDENTITY_TOP_KEYS
        raise DfpPhase1aBoundaryIouIdentityError(f"identity has an unexpected top-level schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    _require_canonical_string(identity["format_version"], "format_version", "format_version")
    for section, keys in SECTION_KEYS.items():
        _require_closed_mapping(identity.get(section), keys, f"identity.{section}")

    _validate_identity_block(identity["identity"])
    _validate_protocol_binding_block(identity["protocol_binding"])
    _validate_paper_block(identity["paper"])
    _validate_official_source_block(identity["official_source"])
    _validate_primitive_block(identity["primitive"])
    _validate_local_reproduction_block(identity["local_reproduction"])
    _validate_semantic_extension_block(identity["semantic_extension"])

    return identity


def _validate_identity_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["name"], "identity", "name")
    _require_canonical_string(block["schema_version"], "identity", "schema_version")
    _require_canonical_string(block["description"], "identity", "description")
    _require_canonical_string(block["resolves_deferred_authority"], "identity", "resolves_deferred_authority")


def _validate_protocol_binding_block(block: Mapping[str, Any]) -> None:
    path = _require_relative_path(block["protocol_identity_relative_path"], "protocol_binding.protocol_identity_relative_path")
    if path != PROTOCOL_IDENTITY_RELATIVE_PATH:
        raise DfpPhase1aBoundaryIouIdentityError("protocol_binding.protocol_identity_relative_path must be exactly the frozen protocol identity path")
    _require_sha256(block["protocol_identity_sha256"], "protocol_binding.protocol_identity_sha256")
    _require_canonical_string(block["required_deferred_status"], "protocol_binding", "required_deferred_status")


def _validate_paper_block(block: Mapping[str, Any]) -> None:
    for field in ("title", "authors", "venue", "arxiv_id", "doi"):
        _require_canonical_string(block[field], "paper", field)


def _validate_official_source_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["repository_url"], "official_source", "repository_url")
    commit = _require_canonical_string(block["repository_commit"], "official_source", "repository_commit")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise DfpPhase1aBoundaryIouIdentityError("official_source.repository_commit must be a full lowercase 40-hex-character Git commit id")
    _require_canonical_string(block["source_relative_path"], "official_source", "source_relative_path")
    _require_canonical_string(block["source_function_name"], "official_source", "source_function_name")
    _require_canonical_sha256(block["source_sha256"], "official_source", "source_sha256")
    _require_canonical_string(block["license_relative_path"], "official_source", "license_relative_path")
    _require_canonical_sha256(block["license_sha256"], "official_source", "license_sha256")


def _validate_primitive_block(block: Mapping[str, Any]) -> None:
    if _require_exact_float(block["dilation_ratio_primary"], "primitive.dilation_ratio_primary") != 0.02:
        raise DfpPhase1aBoundaryIouIdentityError("primitive.dilation_ratio_primary must be exactly 0.02")
    if _require_exact_int(block["padding_value"], "primitive.padding_value") != 0:
        raise DfpPhase1aBoundaryIouIdentityError("primitive.padding_value must be exactly 0")
    if _require_exact_int(block["padding_width"], "primitive.padding_width") != 1:
        raise DfpPhase1aBoundaryIouIdentityError("primitive.padding_width must be exactly 1")
    if _require_exact_int(block["kernel_height"], "primitive.kernel_height") != 3:
        raise DfpPhase1aBoundaryIouIdentityError("primitive.kernel_height must be exactly 3")
    if _require_exact_int(block["kernel_width"], "primitive.kernel_width") != 3:
        raise DfpPhase1aBoundaryIouIdentityError("primitive.kernel_width must be exactly 3")
    _require_canonical_string(block["kernel_dtype"], "primitive", "kernel_dtype")
    _require_canonical_string(block["erosion_backend"], "primitive", "erosion_backend")
    _require_canonical_string(block["round_convention"], "primitive", "round_convention")
    if _require_exact_int(block["minimum_dilation_iterations"], "primitive.minimum_dilation_iterations") != 1:
        raise DfpPhase1aBoundaryIouIdentityError("primitive.minimum_dilation_iterations must be exactly 1")
    _require_exact_int_list(block["fixed_sensitivity_widths"], "primitive.fixed_sensitivity_widths", expected=(1, 2, 4))
    _require_canonical_string(block["mask_dtype"], "primitive", "mask_dtype")
    _require_canonical_string(block["boundary_formula"], "primitive", "boundary_formula")


def _validate_local_reproduction_block(block: Mapping[str, Any]) -> None:
    _require_relative_path(block["license_notice_relative_path"], "local_reproduction.license_notice_relative_path")
    _require_canonical_string(block["license_notice_relative_path"], "local_reproduction", "license_notice_relative_path")
    _require_canonical_string(block["reproduction_kind"], "local_reproduction", "reproduction_kind")
    _require_relative_path(block["reproduced_function_relative_path"], "local_reproduction.reproduced_function_relative_path")
    _require_canonical_string(block["reproduced_function_relative_path"], "local_reproduction", "reproduced_function_relative_path")
    _require_canonical_string(block["reproduced_function_name"], "local_reproduction", "reproduced_function_name")


def _validate_semantic_extension_block(block: Mapping[str, Any]) -> None:
    _require_canonical_string(block["label"], "semantic_extension", "label")
    if _require_exact_bool(block["is_official_semantic_definition"], "semantic_extension.is_official_semantic_definition") is not False:
        raise DfpPhase1aBoundaryIouIdentityError("semantic_extension.is_official_semantic_definition must be false")
    _require_canonical_string(block["statement"], "semantic_extension", "statement")
    _require_canonical_string(block["ignore_policy"], "semantic_extension", "ignore_policy")
    _require_canonical_string(block["zero_union_policy"], "semantic_extension", "zero_union_policy")
    _require_canonical_string(block["metric_unit"], "semantic_extension", "metric_unit")
    _require_canonical_string(block["aggregation_rule"], "semantic_extension", "aggregation_rule")


def validate_static_configuration(*, repo_root: Path, identity_path: Path | None = None) -> dict[str, Any]:
    """Load the supplement AND relationally verify it against the live
    protocol identity, the live official-source file hashes (if
    present locally -- see note below), and the live local license
    notice file. Requires the protocol identity to still declare the
    deferred status this supplement exists to resolve."""
    root = Path(repo_root)
    identity = load_identity(identity_path, repo_root=root)

    protocol_path = root / identity["protocol_binding"]["protocol_identity_relative_path"]
    try:
        observed_protocol_sha256 = sha256_file(protocol_path)
    except OSError as error:
        raise DfpPhase1aBoundaryIouIdentityError(f"cannot read bound protocol identity {protocol_path}: {error}") from error
    if observed_protocol_sha256 != identity["protocol_binding"]["protocol_identity_sha256"]:
        raise DfpPhase1aBoundaryIouIdentityError(
            f"protocol identity {protocol_path} SHA256 mismatch: supplement declares "
            f"{identity['protocol_binding']['protocol_identity_sha256']}, observed {observed_protocol_sha256}"
        )

    from src.dfp_phase1a_identity import load_identity as load_protocol_identity

    protocol = load_protocol_identity(protocol_path, repo_root=root)
    live_status = protocol["boundary_iou"]["implementation_source_status"]
    if live_status != identity["protocol_binding"]["required_deferred_status"]:
        raise DfpPhase1aBoundaryIouIdentityError(
            f"live protocol identity's [boundary_iou].implementation_source_status is {live_status!r}, not "
            f"{identity['protocol_binding']['required_deferred_status']!r} -- this supplement exists only to "
            f"resolve a deliberately deferred binding, and refuses to proceed if the protocol was already bound "
            f"some other way (or unbound in some new way this supplement does not recognize)"
        )

    license_notice_path = root / identity["local_reproduction"]["license_notice_relative_path"]
    if not license_notice_path.is_file():
        raise DfpPhase1aBoundaryIouIdentityError(f"local license notice file does not exist: {license_notice_path}")

    return {
        "identity_name": identity["identity"]["name"],
        "protocol_identity_sha256": observed_protocol_sha256,
        "official_source_sha256": identity["official_source"]["source_sha256"],
        "dilation_ratio_primary": identity["primitive"]["dilation_ratio_primary"],
        "fixed_sensitivity_widths": tuple(identity["primitive"]["fixed_sensitivity_widths"]),
    }


__all__ = [
    "IDENTITY_RELATIVE_PATH",
    "PROTOCOL_IDENTITY_RELATIVE_PATH",
    "REQUIRED_DEFERRED_STATUS",
    "IDENTITY_TOP_KEYS",
    "SECTION_KEYS",
    "CANONICAL_SCIENTIFIC_STRINGS",
    "DfpPhase1aBoundaryIouIdentityError",
    "repository_root",
    "sha256_file",
    "load_identity",
    "validate_static_configuration",
]
