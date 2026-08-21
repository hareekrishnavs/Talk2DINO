"""Final T4 signal-alignment report: a read-only, fail-closed analysis of an
already-finalized trust/centrality report (``talk2dino-trust-centrality-
report-v4``), quantifying whether strict-T4 cross-view consensus is a
trustworthy enough signal to justify semantic-consensus-guided COVER-DR.

This module never runs a model, never touches CUDA, and never re-derives
T4/trust/centrality sufficient statistics -- it only reads already-computed
fields from the finalized report (and, optionally, a canonical per-class
pixel-union artifact) and reports derived, clearly-labeled quantities.

Three concepts this module is careful never to conflate (see the module's
own docs/t4_signal_alignment_report.md for the full discussion):

1. operator attribution: T4 proves the RWR diffusion operator, on the
   source crop's own neighborhood graph, reversed the frozen pre-RWR unary
   preference relative to unanimous cross-view context.
2. semantic correctness: cross-view consensus agreeing with the reversed
   unary does NOT by itself establish that the reversal was semantically
   wrong -- a shared systematic unary bias could make every context view
   agree on an incorrect label just as easily as a correct one.
3. direct anchor alignment: consensus_correct - dissent_correct at
   evaluated T4 anchors is a patch-anchor-count quantity, not a pixel or
   mIoU quantity -- interpolation, unions, argmax margins, and nonlocal
   graph effects all separate anchor counts from segmentation-metric
   impact (see build_report()'s mIoU-impact caveat fields).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.e3_evaluation_identity import E3IdentityError  # noqa: E402
from src.e3_evaluation_identity import _resolve_dataset_class_names  # noqa: E402
from src.e3_evaluation_identity import load_identity as _load_e3_identity  # noqa: E402

_EVAL_DIR = REPOSITORY_ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation"
_TRUST_CENTRALITY_HARNESS_PATH = _EVAL_DIR / "trust_centrality_harness.py"


def _load_trust_centrality_harness():
    """Loads trust_centrality_harness.py directly by file path.

    That module has zero relative imports (only stdlib + torch), so this
    avoids importing it via the ``segmentation.evaluation`` package, whose
    ``__init__.py`` pulls in mmcv/cv2 for dataloader-building code this
    CPU-only, read-only analysis module never needs. Reused so this module
    can call the existing v4 report/section validators directly rather than
    re-implementing a weaker parallel parser.
    """
    module_name = "t4_signal_alignment_report._trust_centrality_harness"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _TRUST_CENTRALITY_HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise T4SignalAlignmentError(
            f"cannot locate trust_centrality_harness.py at {_TRUST_CENTRALITY_HARNESS_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class T4SignalAlignmentError(ValueError):
    """Raised on any input-validation, reconciliation, or contract failure.

    Fail-closed by design: every branch that cannot be positively validated
    raises this rather than silently substituting a default or a legacy
    interpretation.
    """


REPORT_SCHEMA_VERSION = "talk2dino-t4-signal-alignment-report-v1"
SUPPORTED_TRUST_REPORT_SCHEMA_VERSION = "talk2dino-trust-centrality-report-v4"
SUPPORTED_TRUST_CENTRALITY_SECTION_SCHEMA_VERSION = "talk2dino-trust-centrality-full-report-v3"

UNIT_FRACTION_0_TO_1 = "fraction_0_to_1"
UNIT_PERCENTAGE_POINTS = "percentage_points"
UNIT_FRACTION_DIFFERENCE = "fraction_difference"
UNIT_ANCHOR_COUNT = "anchor_count"
UNIT_ANCHOR_EQUIVALENT_DESCRIPTIVE = "anchor_equivalent_descriptive_not_integer_ci"
UNIT_MIXED_SIGN_PROXY = "mixed_unit_sign_alignment_proxy_not_delta_miou"

CANONICAL_ALPHA = 0.98
CANONICAL_STEPS = 320
CANONICAL_CLASS_COUNT = 171

# ---------------------------------------------------------------------------
# Strict JSON loading: reject NaN/Infinity and duplicate object keys eagerly,
# rather than relying on the caller to notice a silently-accepted non-finite
# value later.
# ---------------------------------------------------------------------------


def _reject_constant(token: str) -> float:
    raise T4SignalAlignmentError(f"input JSON contains a non-finite numeric constant: {token}")


def _closed_object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise T4SignalAlignmentError(f"input JSON contains a duplicate object key: {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise T4SignalAlignmentError(f"cannot read {label} {path}: {error}") from error
    try:
        value = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_closed_object_pairs)
    except T4SignalAlignmentError:
        raise
    except json.JSONDecodeError as error:
        raise T4SignalAlignmentError(f"cannot parse {label} {path} as JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise T4SignalAlignmentError(f"{label} {path} must contain one JSON object at its root")
    return value


def _sha256_of_file(path: Path, *, label: str = "file") -> str:
    """Streaming SHA-256 over ``path``. Every filesystem-level failure
    (missing file, permission denied, path is a directory, or any other
    read/open failure) is converted to ``T4SignalAlignmentError`` with
    exception chaining -- never a raw ``OSError`` traceback. Does not catch
    ``KeyboardInterrupt``/``SystemExit`` (neither is an ``OSError``) or
    arbitrary programming errors."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise T4SignalAlignmentError(f"unable to read {label}: {path}: {error}") from error
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Exact-type scalar validators (bool is never accepted as int; NaN/Infinity
# already excluded by strict JSON loading, re-checked here defensively for
# any value that reaches these helpers via a non-JSON-loading path, e.g. a
# value computed in this module itself).
# ---------------------------------------------------------------------------


def _require_exact_int(value: Any, label: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise T4SignalAlignmentError(f"{label} must be an exact integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise T4SignalAlignmentError(f"{label} must be >= {minimum}, got {value}")
    return value


def _require_optional_exact_int(value: Any, label: str, *, minimum: Optional[int] = None) -> Optional[int]:
    if value is None:
        return None
    return _require_exact_int(value, label, minimum=minimum)


def _require_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T4SignalAlignmentError(f"{label} must be numeric, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise T4SignalAlignmentError(f"{label} must be finite, got {result}")
    return result


def _require_optional_finite_float(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    return _require_finite_float(value, label)


def _require_fraction(value: Any, label: str) -> float:
    result = _require_finite_float(value, label)
    if not (0.0 <= result <= 1.0):
        raise T4SignalAlignmentError(f"{label} must be in [0,1], got {result}")
    return result


def _require_optional_fraction(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    return _require_fraction(value, label)


def _require_str(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise T4SignalAlignmentError(f"{label} must be a non-empty string")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise T4SignalAlignmentError(f"{label} must be a JSON object")
    return value


def _require_class_id(raw_key: Any, label: str) -> int:
    """Accepts an exact integer, or a string that is an exact decimal
    integer representation (no sign, no leading zeros beyond '0' itself, no
    whitespace, no fractional part) -- exactly the shape ``json.dumps``
    produces for a Python ``int`` dict key."""
    if isinstance(raw_key, bool):
        raise T4SignalAlignmentError(f"{label} must not be a boolean")
    if isinstance(raw_key, int):
        return raw_key
    if isinstance(raw_key, str):
        if raw_key == "0":
            return 0
        if raw_key and raw_key[0] in "123456789" and raw_key.isdigit():
            return int(raw_key)
        raise T4SignalAlignmentError(f"{label}={raw_key!r} is not an exact decimal integer string")
    raise T4SignalAlignmentError(f"{label} must be an integer or an exact decimal integer string")


# ---------------------------------------------------------------------------
# Section A: input contract -- eager, fail-closed validation of the
# finalized trust/centrality report.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedTrustReport:
    """The parsed, validated, still-read-only trust/centrality report plus
    the raw bytes/hash needed for provenance -- never a mutated copy."""

    path: Path
    raw_bytes_sha256: str
    report: Mapping[str, Any]


def load_and_validate_trust_report(path: Path) -> LoadedTrustReport:
    """Loads and eagerly, fail-closed validates a finalized trust/centrality
    report. Reuses the existing v4 unit-contract and section-completeness
    validators from trust_centrality_harness.py rather than re-implementing
    them; never mutates the parsed mapping or the source file."""
    path = Path(path)
    raw_bytes_sha256 = _sha256_of_file(path, label="trust report")
    report = _load_strict_json(path, label="trust report")

    h = _load_trust_centrality_harness()

    schema_version = report.get("schema_version")
    if schema_version != SUPPORTED_TRUST_REPORT_SCHEMA_VERSION:
        raise T4SignalAlignmentError(
            "unsupported or unrecognized trust report schema_version: "
            f"{schema_version!r}; only {SUPPORTED_TRUST_REPORT_SCHEMA_VERSION!r} is supported "
            "(no silent fallback to a legacy or incomplete report)"
        )

    if report.get("final") is not True:
        raise T4SignalAlignmentError("trust report final must be exactly true")
    if report.get("complete") is not True:
        raise T4SignalAlignmentError("trust report complete must be exactly true")

    # Reuse the existing outer-report validators rather than a weaker
    # parallel parser.
    try:
        h.validate_report_v4_unit_contract(report)
        h.require_full_run_complete(report)
    except h.TrustCentralityHarnessError as error:
        raise T4SignalAlignmentError(f"trust report failed existing harness validation: {error}") from error

    trust_centrality = report.get("trust_centrality")
    if not isinstance(trust_centrality, Mapping):
        raise T4SignalAlignmentError("trust report is missing a trust_centrality section")
    if trust_centrality.get("status") != "available":
        raise T4SignalAlignmentError(
            f"trust_centrality.status must be 'available', got {trust_centrality.get('status')!r}"
        )
    if trust_centrality.get("schema_version") != SUPPORTED_TRUST_CENTRALITY_SECTION_SCHEMA_VERSION:
        raise T4SignalAlignmentError(
            "unsupported trust_centrality.schema_version: "
            f"{trust_centrality.get('schema_version')!r}; only "
            f"{SUPPORTED_TRUST_CENTRALITY_SECTION_SCHEMA_VERSION!r} is supported"
        )
    try:
        h.validate_trust_centrality_section_complete(trust_centrality)
    except h.TrustCentralityHarnessError as error:
        raise T4SignalAlignmentError(f"trust_centrality section failed existing completeness validation: {error}") from error

    dataset_length = _require_exact_int(report.get("dataset_length"), "dataset_length", minimum=1)
    images_processed = _require_exact_int(report.get("images_processed"), "images_processed", minimum=0)
    unique_image_count = _require_exact_int(report.get("unique_image_count"), "unique_image_count", minimum=0)
    if images_processed != dataset_length:
        raise T4SignalAlignmentError(
            f"images_processed ({images_processed}) does not equal dataset_length ({dataset_length}) "
            "for a report marked complete=true"
        )
    if unique_image_count != images_processed:
        raise T4SignalAlignmentError(
            f"unique_image_count ({unique_image_count}) does not equal images_processed ({images_processed})"
        )

    for required_key in ("funnel", "trust_by_stage", "bootstrap", "per_class_t4", "class_concentration"):
        if required_key not in trust_centrality:
            raise T4SignalAlignmentError(f"trust_centrality section is missing required key: {required_key!r}")
    for required_stage in ("t4", "t4_prime"):
        if required_stage not in trust_centrality["funnel"]:
            raise T4SignalAlignmentError(f"trust_centrality.funnel is missing stage {required_stage!r}")
        if required_stage not in trust_centrality["trust_by_stage"]:
            raise T4SignalAlignmentError(f"trust_centrality.trust_by_stage is missing stage {required_stage!r}")
    if "t4" not in trust_centrality["bootstrap"]:
        raise T4SignalAlignmentError("trust_centrality.bootstrap is missing the aggregate 't4' population")

    return LoadedTrustReport(path=path, raw_bytes_sha256=raw_bytes_sha256, report=report)


# ---------------------------------------------------------------------------
# Section B: source provenance.
# ---------------------------------------------------------------------------


def _git_head(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_provenance(loaded: LoadedTrustReport, *, repo_root: Path = REPOSITORY_ROOT) -> dict:
    report = loaded.report
    trust_centrality = report["trust_centrality"]
    bootstrap_t4 = trust_centrality["bootstrap"]["t4"]
    provenance = report.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}

    return {
        "report_schema_name": REPORT_SCHEMA_VERSION,
        "input_trust_report": {
            "filename": Path(loaded.path).name,
            "sha256": loaded.raw_bytes_sha256,
            "schema_version": report.get("schema_version"),
            "trust_centrality_schema_version": trust_centrality.get("schema_version"),
        },
        "input_provenance_fields": {
            key: provenance[key]
            for key in (
                "git_head", "git_branch", "canonical_config_sha256", "e3_identity_sha256",
                "rwr_identity_sha256", "checkpoint_context_git_head", "checkpoint_context_git_branch",
                "diagnostic_schema_version",
            )
            if key in provenance
        },
        "image_count": _require_exact_int(report.get("images_processed"), "images_processed"),
        "window_count": _require_optional_exact_int(report.get("windows_processed"), "windows_processed"),
        "class_count": None,  # resolved below once per_class_t4/class_concentration are available
        "bootstrap_seed": _require_exact_int(bootstrap_t4.get("seed"), "bootstrap.t4.seed"),
        "bootstrap_resamples_requested": _require_exact_int(
            bootstrap_t4.get("resamples_requested"), "bootstrap.t4.resamples_requested"
        ),
        "bootstrap_confidence_level": _require_fraction(
            bootstrap_t4.get("confidence_level"), "bootstrap.t4.confidence_level"
        ),
        "bootstrap_unit": _require_str(bootstrap_t4.get("bootstrap_unit"), "bootstrap.t4.bootstrap_unit"),
        "generator_git_commit": _git_head(repo_root),
        "generated_at_utc": None,  # isolated volatile field, set once by build_report()
    }


# ---------------------------------------------------------------------------
# Section C: strict-T4 summary.
# ---------------------------------------------------------------------------


def build_strict_t4_summary(report: Mapping[str, Any]) -> dict:
    trust_centrality = report["trust_centrality"]
    funnel_t4 = _require_mapping(trust_centrality["funnel"]["t4"], "funnel.t4")
    stage_t4 = _require_mapping(trust_centrality["trust_by_stage"]["t4"], "trust_by_stage.t4")
    bootstrap_t4 = _require_mapping(trust_centrality["bootstrap"]["t4"], "bootstrap.t4")

    raw_anchor_count = _require_exact_int(funnel_t4.get("count"), "funnel.t4.count", minimum=0)
    valid_gt_count = _require_exact_int(funnel_t4.get("valid_gt"), "funnel.t4.valid_gt", minimum=0)
    ignored_gt_count = _require_exact_int(funnel_t4.get("ignored_gt"), "funnel.t4.ignored_gt", minimum=0)
    if valid_gt_count + ignored_gt_count != raw_anchor_count:
        raise T4SignalAlignmentError(
            "funnel.t4 valid_gt + ignored_gt does not reconcile with raw count: "
            f"{valid_gt_count} + {ignored_gt_count} != {raw_anchor_count}"
        )
    contributing_images = _require_exact_int(funnel_t4.get("contributing_images"), "funnel.t4.contributing_images", minimum=0)
    zero_target_images = _require_exact_int(funnel_t4.get("images_with_zero_records"), "funnel.t4.images_with_zero_records", minimum=0)
    total_anchor_fraction = _require_fraction(funnel_t4.get("total_anchor_fraction"), "funnel.t4.total_anchor_fraction")

    consensus_correct = _require_exact_int(stage_t4.get("consensus_correct"), "trust_by_stage.t4.consensus_correct", minimum=0)
    dissent_correct = _require_exact_int(stage_t4.get("dissent_correct"), "trust_by_stage.t4.dissent_correct", minimum=0)
    if stage_t4.get("valid_gt") != valid_gt_count:
        raise T4SignalAlignmentError(
            f"trust_by_stage.t4.valid_gt ({stage_t4.get('valid_gt')}) does not reconcile with "
            f"funnel.t4.valid_gt ({valid_gt_count})"
        )
    direct_anchor_net = consensus_correct - dissent_correct

    consensus_accuracy = _require_fraction(stage_t4.get("consensus_accuracy"), "trust_by_stage.t4.consensus_accuracy")
    dissent_accuracy = _require_fraction(stage_t4.get("dissent_accuracy"), "trust_by_stage.t4.dissent_accuracy")
    delta_trust = _require_finite_float(stage_t4.get("delta_trust"), "trust_by_stage.t4.delta_trust")
    if not (-1.0 <= delta_trust <= 1.0):
        raise T4SignalAlignmentError(f"trust_by_stage.t4.delta_trust={delta_trust} outside [-1,1]")
    expected_delta_trust = direct_anchor_net / valid_gt_count if valid_gt_count else 0.0
    if valid_gt_count and abs(delta_trust - expected_delta_trust) > 1e-9:
        raise T4SignalAlignmentError(
            f"trust_by_stage.t4.delta_trust ({delta_trust}) does not reconcile with "
            f"direct_anchor_net/valid_gt_count ({expected_delta_trust})"
        )

    third_label_count = _require_exact_int(stage_t4.get("gt_third_label"), "trust_by_stage.t4.gt_third_label", minimum=0)
    third_label_fraction = _require_fraction(stage_t4.get("third_label_fraction"), "trust_by_stage.t4.third_label_fraction")

    image_macro_estimate = _require_optional_finite_float(
        bootstrap_t4.get("image_macro_estimate"), "bootstrap.t4.image_macro_estimate"
    )
    ci = bootstrap_t4.get("ci95_fraction_difference")
    if not isinstance(ci, list) or len(ci) != 2:
        raise T4SignalAlignmentError("bootstrap.t4.ci95_fraction_difference must be a two-element array")
    ci_low = _require_optional_finite_float(ci[0], "bootstrap.t4.ci95_fraction_difference[0]")
    ci_high = _require_optional_finite_float(ci[1], "bootstrap.t4.ci95_fraction_difference[1]")
    if ci_low is not None and ci_high is not None and ci_low > ci_high:
        raise T4SignalAlignmentError(f"bootstrap.t4 CI ordering invalid: low={ci_low} > high={ci_high}")
    ci_contains_zero = ci_low is not None and ci_high is not None and ci_low <= 0.0 <= ci_high

    anchor_equivalent_ci = None
    if ci_low is not None and ci_high is not None:
        anchor_equivalent_ci = [ci_low * valid_gt_count, ci_high * valid_gt_count]

    return {
        "raw_anchor_count": raw_anchor_count,
        "valid_gt_count": valid_gt_count,
        "ignored_gt_count": ignored_gt_count,
        "contributing_images": contributing_images,
        "zero_target_images": zero_target_images,
        "total_anchor_fraction": total_anchor_fraction,
        "total_anchor_fraction_unit": UNIT_FRACTION_0_TO_1,
        "consensus_correct": consensus_correct,
        "dissent_correct": dissent_correct,
        "direct_anchor_net": direct_anchor_net,
        "direct_anchor_net_unit": UNIT_ANCHOR_COUNT,
        "direct_anchor_net_formula": "consensus_correct - dissent_correct",
        "consensus_accuracy": consensus_accuracy,
        "dissent_accuracy": dissent_accuracy,
        "accuracy_unit": UNIT_FRACTION_0_TO_1,
        "delta_trust_fraction": delta_trust,
        "delta_trust_percentage_points": delta_trust * 100.0,
        "delta_trust_unit": UNIT_FRACTION_DIFFERENCE,
        "image_macro_estimate_fraction": image_macro_estimate,
        "image_macro_estimate_percentage_points": None if image_macro_estimate is None else image_macro_estimate * 100.0,
        "third_label_count": third_label_count,
        "third_label_fraction": third_label_fraction,
        "third_label_fraction_unit": UNIT_FRACTION_0_TO_1,
        "bootstrap": {
            "status": _require_str(bootstrap_t4.get("status"), "bootstrap.t4.status"),
            "ci95_fraction_difference": [ci_low, ci_high],
            "ci95_percentage_points": [None if ci_low is None else ci_low * 100.0, None if ci_high is None else ci_high * 100.0],
            "ci_contains_zero": ci_contains_zero,
            "confidence_level": _require_fraction(bootstrap_t4.get("confidence_level"), "bootstrap.t4.confidence_level"),
            "resamples_requested": _require_exact_int(bootstrap_t4.get("resamples_requested"), "bootstrap.t4.resamples_requested"),
            "valid_replicate_count": _require_optional_exact_int(bootstrap_t4.get("valid_replicate_count"), "bootstrap.t4.valid_replicate_count"),
        },
        "anchor_equivalent_bootstrap_interval_descriptive": anchor_equivalent_ci,
        "anchor_equivalent_bootstrap_interval_descriptive_unit": UNIT_ANCHOR_EQUIVALENT_DESCRIPTIVE,
        "anchor_equivalent_bootstrap_interval_descriptive_caveats": [
            "this is a descriptive anchor-equivalent interval only",
            "it is not an integer confidence interval",
            "it is not a bound on the number of anchors that could be corrected",
            "it is not an mIoU interval",
        ],
    }


# ---------------------------------------------------------------------------
# Section D: T4-prime summary (diagnostic ablation only -- never combined
# with strict-T4 in any downstream decision).
# ---------------------------------------------------------------------------


def build_t4_prime_summary(report: Mapping[str, Any], strict_t4: Mapping[str, Any]) -> dict:
    trust_centrality = report["trust_centrality"]
    funnel_prime = _require_mapping(trust_centrality["funnel"]["t4_prime"], "funnel.t4_prime")
    stage_prime = _require_mapping(trust_centrality["trust_by_stage"]["t4_prime"], "trust_by_stage.t4_prime")

    raw_anchor_count = _require_exact_int(funnel_prime.get("count"), "funnel.t4_prime.count", minimum=0)
    valid_gt_count = _require_exact_int(funnel_prime.get("valid_gt"), "funnel.t4_prime.valid_gt", minimum=0)
    ignored_gt_count = _require_exact_int(funnel_prime.get("ignored_gt"), "funnel.t4_prime.ignored_gt", minimum=0)
    if valid_gt_count + ignored_gt_count != raw_anchor_count:
        raise T4SignalAlignmentError(
            "funnel.t4_prime valid_gt + ignored_gt does not reconcile with raw count: "
            f"{valid_gt_count} + {ignored_gt_count} != {raw_anchor_count}"
        )

    consensus_correct = _require_exact_int(stage_prime.get("consensus_correct"), "trust_by_stage.t4_prime.consensus_correct", minimum=0)
    dissent_correct = _require_exact_int(stage_prime.get("dissent_correct"), "trust_by_stage.t4_prime.dissent_correct", minimum=0)
    direct_anchor_net = consensus_correct - dissent_correct
    delta_trust = _require_finite_float(stage_prime.get("delta_trust"), "trust_by_stage.t4_prime.delta_trust")
    third_label_count = _require_exact_int(stage_prime.get("gt_third_label"), "trust_by_stage.t4_prime.gt_third_label", minimum=0)
    third_label_fraction = _require_fraction(stage_prime.get("third_label_fraction"), "trust_by_stage.t4_prime.third_label_fraction")

    strict_raw = strict_t4["raw_anchor_count"]
    strict_valid = strict_t4["valid_gt_count"]
    raw_growth_ratio = (raw_anchor_count / strict_raw) if strict_raw else None
    valid_growth_ratio = (valid_gt_count / strict_valid) if strict_valid else None

    # No aggregate 't4_prime' bootstrap population is invented if the
    # source report does not carry one -- only stratified t4_prime_*
    # sub-populations (by centrality bin/edge band) may exist.
    bootstrap_section = trust_centrality["bootstrap"]
    if "t4_prime" in bootstrap_section:
        prime_bootstrap = _require_mapping(bootstrap_section["t4_prime"], "bootstrap.t4_prime")
        bootstrap_out = {
            "status": _require_str(prime_bootstrap.get("status"), "bootstrap.t4_prime.status"),
            "reason": None,
        }
    else:
        bootstrap_out = {
            "status": "unavailable",
            "reason": "the source report contains no aggregate 't4_prime' bootstrap population "
                      "(only stratified t4_prime_centrality_*/t4_prime_edge_* sub-populations); "
                      "no confidence interval is fabricated",
        }

    delta_trust_weaker_than_strict_t4 = delta_trust < strict_t4["delta_trust_fraction"]
    third_label_more_frequent_than_strict_t4 = third_label_fraction > strict_t4["third_label_fraction"]

    return {
        "raw_anchor_count": raw_anchor_count,
        "valid_gt_count": valid_gt_count,
        "ignored_gt_count": ignored_gt_count,
        "consensus_correct": consensus_correct,
        "dissent_correct": dissent_correct,
        "direct_anchor_net": direct_anchor_net,
        "direct_anchor_net_unit": UNIT_ANCHOR_COUNT,
        "delta_trust_fraction": delta_trust,
        "delta_trust_percentage_points": delta_trust * 100.0,
        "delta_trust_unit": UNIT_FRACTION_DIFFERENCE,
        "third_label_count": third_label_count,
        "third_label_fraction": third_label_fraction,
        "third_label_fraction_unit": UNIT_FRACTION_0_TO_1,
        "relative_growth_from_strict_t4": {
            "raw_anchor_growth_ratio": raw_growth_ratio,
            "valid_gt_growth_ratio": valid_growth_ratio,
        },
        "bootstrap": bootstrap_out,
        "weaker_than_strict_t4": {
            "delta_trust_weaker": delta_trust_weaker_than_strict_t4,
            "third_label_more_frequent": third_label_more_frequent_than_strict_t4,
            "both": delta_trust_weaker_than_strict_t4 and third_label_more_frequent_than_strict_t4,
        },
        "diagnostic_only_caveat": (
            "T4-prime is a diagnostic ablation only; its counts and Delta_trust are never "
            "combined with strict-T4's, and it is never promoted to the default population "
            "based on any validation-metric comparison."
        ),
    }


# ---------------------------------------------------------------------------
# Section E: per-class alignment.
# ---------------------------------------------------------------------------


def _resolve_class_names(*, repo_root: Path = REPOSITORY_ROOT) -> Optional[tuple[str, ...]]:
    """Best-effort resolution of authoritative COCO-Stuff class names via
    the same mechanism e3_evaluation_identity.py's own identity validator
    uses (the installed mmsegmentation distribution's own COCOStuffDataset.
    CLASSES, located and AST-parsed, never hand-typed here). Returns None
    if the identity/distribution cannot be resolved -- class names are
    optional decoration, never load-bearing for this report's numbers."""
    try:
        identity = _load_e3_identity(repo_root=repo_root)
        return _resolve_dataset_class_names(identity["dataset"])
    except (E3IdentityError, OSError, KeyError):
        return None


def build_per_class_alignment(
    report: Mapping[str, Any], strict_t4: Mapping[str, Any], *, repo_root: Path = REPOSITORY_ROOT
) -> dict:
    trust_centrality = report["trust_centrality"]
    per_class_raw = _require_mapping(trust_centrality["per_class_t4"], "per_class_t4")
    concentration = _require_mapping(trust_centrality["class_concentration"], "class_concentration")

    class_names = _resolve_class_names(repo_root=repo_root)

    seen_class_ids: set[int] = set()
    records: list[dict] = []
    sum_observations = sum_consensus = sum_dissent = sum_third_label = sum_net = 0
    for raw_key, entry in per_class_raw.items():
        class_id = _require_class_id(raw_key, "per_class_t4 key")
        entry = _require_mapping(entry, f"per_class_t4[{raw_key!r}]")
        entry_class_id = _require_exact_int(entry.get("class_id"), f"per_class_t4[{raw_key!r}].class_id", minimum=0)
        if entry_class_id != class_id:
            raise T4SignalAlignmentError(
                f"per_class_t4 key {raw_key!r} does not match its own class_id field {entry_class_id}"
            )
        if class_id in seen_class_ids:
            raise T4SignalAlignmentError(f"per_class_t4 contains a duplicate class_id after normalization: {class_id}")
        seen_class_ids.add(class_id)

        observations = _require_exact_int(entry.get("observations"), f"per_class_t4[{class_id}].observations", minimum=0)
        consensus_correct = _require_exact_int(entry.get("consensus_correct"), f"per_class_t4[{class_id}].consensus_correct", minimum=0)
        dissent_correct = _require_exact_int(entry.get("dissent_correct"), f"per_class_t4[{class_id}].dissent_correct", minimum=0)
        third_label = _require_exact_int(entry.get("third_label"), f"per_class_t4[{class_id}].third_label", minimum=0)
        net = consensus_correct - dissent_correct
        sign = "positive" if net > 0 else ("negative" if net < 0 else "zero")

        sum_observations += observations
        sum_consensus += consensus_correct
        sum_dissent += dissent_correct
        sum_third_label += third_label
        sum_net += net

        name = class_names[class_id] if class_names is not None and 0 <= class_id < len(class_names) else None
        records.append({
            "class_id": class_id, "class_name": name, "observations": observations,
            "consensus_correct": consensus_correct, "dissent_correct": dissent_correct,
            "third_label": third_label, "net": net, "sign": sign,
        })

    if sum_observations != strict_t4["valid_gt_count"]:
        raise T4SignalAlignmentError(
            f"sum(per_class observations)={sum_observations} does not reconcile with "
            f"strict-T4 valid_gt_count={strict_t4['valid_gt_count']}"
        )
    if sum_consensus != strict_t4["consensus_correct"]:
        raise T4SignalAlignmentError(
            f"sum(per_class consensus_correct)={sum_consensus} does not reconcile with "
            f"strict-T4 consensus_correct={strict_t4['consensus_correct']}"
        )
    if sum_dissent != strict_t4["dissent_correct"]:
        raise T4SignalAlignmentError(
            f"sum(per_class dissent_correct)={sum_dissent} does not reconcile with "
            f"strict-T4 dissent_correct={strict_t4['dissent_correct']}"
        )
    if sum_third_label != strict_t4["third_label_count"]:
        raise T4SignalAlignmentError(
            f"sum(per_class third_label)={sum_third_label} does not reconcile with "
            f"strict-T4 third_label_count={strict_t4['third_label_count']}"
        )
    if sum_net != strict_t4["direct_anchor_net"]:
        raise T4SignalAlignmentError(
            f"sum(per_class net)={sum_net} does not reconcile with "
            f"strict-T4 direct_anchor_net={strict_t4['direct_anchor_net']}"
        )

    represented_class_count = len(records)
    if concentration.get("classes_represented") != represented_class_count:
        raise T4SignalAlignmentError(
            f"class_concentration.classes_represented ({concentration.get('classes_represented')}) does not "
            f"reconcile with len(per_class_t4) ({represented_class_count})"
        )

    positive = [r for r in records if r["sign"] == "positive"]
    negative = [r for r in records if r["sign"] == "negative"]
    zero = [r for r in records if r["sign"] == "zero"]
    gross_positive_net = sum(r["net"] for r in positive)
    gross_negative_net = sum(r["net"] for r in negative)

    # Deterministic ordering: sort by the ranking key descending, tie-break
    # by ascending class_id, so ties never depend on dict/insertion order.
    records_by_class_id = sorted(records, key=lambda r: r["class_id"])
    by_observations = sorted(records_by_class_id, key=lambda r: (-r["observations"], r["class_id"]))
    by_net_desc = sorted(records_by_class_id, key=lambda r: (-r["net"], r["class_id"]))
    by_net_asc = sorted(records_by_class_id, key=lambda r: (r["net"], r["class_id"]))

    largest_positive_contributor = by_net_desc[0] if by_net_desc and by_net_desc[0]["net"] > 0 else None
    total_net_excluding_largest_positive = (
        sum_net - largest_positive_contributor["net"] if largest_positive_contributor is not None else sum_net
    )

    class_zero_record = next((r for r in records if r["class_id"] == 0), None)

    top_n = 10

    return {
        "represented_class_count": represented_class_count,
        "positive_class_count": len(positive),
        "negative_class_count": len(negative),
        "zero_class_count": len(zero),
        "gross_positive_net": gross_positive_net,
        "gross_negative_net": gross_negative_net,
        "total_net": sum_net,
        "total_net_unit": UNIT_ANCHOR_COUNT,
        "largest_positive_contributor": largest_positive_contributor,
        "total_net_excluding_largest_positive_contributor": total_net_excluding_largest_positive,
        "class_zero": class_zero_record,
        "class_zero_caveat": (
            "class 0's identity is reported only when resolvable from the authoritative "
            "mmsegmentation COCOStuffDataset.CLASSES definition; its numeric fields are never "
            "assumed to correspond to any particular semantic label without that resolution"
        ),
        "top_classes_by_observations": by_observations[:top_n],
        "top_positive_net_classes": [r for r in by_net_desc if r["net"] > 0][:top_n],
        "top_negative_net_classes": [r for r in by_net_asc if r["net"] < 0][:top_n],
        "concentration_fractions_from_source": {
            "top_1_fraction": _require_fraction(concentration.get("top_1_fraction"), "class_concentration.top_1_fraction"),
            "top_5_fraction": _require_fraction(concentration.get("top_5_fraction"), "class_concentration.top_5_fraction"),
            "top_10_fraction": _require_fraction(concentration.get("top_10_fraction"), "class_concentration.top_10_fraction"),
            "top_20_fraction": _require_fraction(concentration.get("top_20_fraction"), "class_concentration.top_20_fraction"),
        },
        "class_names_resolved": class_names is not None,
        "records": records_by_class_id,
    }


# ---------------------------------------------------------------------------
# Section F: optional union-weighted anchor-alignment proxy.
# ---------------------------------------------------------------------------

# Sensitivity constants are named for what they represent, not treated as
# magic numbers: 196 = 14x14, the average bilinear interpolation weight
# footprint per patch node (DINOv2 patch size 14, one node's "typical"
# pixel footprint under align_corners=True upsampling); ~830 approximates
# the maximal joint bilinear-support footprint across the four neighboring
# nodes a boundary pixel can draw from. Neither is a bound or a prediction.
_INTERPOLATION_WEIGHT_FOOTPRINT_AVERAGE = 196
_INTERPOLATION_WEIGHT_FOOTPRINT_MAXIMAL = 830


def _validate_union_array(union: Sequence[Any], *, source_label: str) -> tuple[float, ...]:
    if not isinstance(union, list) or len(union) != CANONICAL_CLASS_COUNT:
        raise T4SignalAlignmentError(
            f"{source_label}: union array must have exactly {CANONICAL_CLASS_COUNT} elements, "
            f"got {len(union) if isinstance(union, list) else type(union).__name__}"
        )
    values: list[float] = []
    for class_id, raw in enumerate(union):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise T4SignalAlignmentError(f"{source_label}: union[{class_id}] must be numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise T4SignalAlignmentError(f"{source_label}: union[{class_id}] must be finite, got {value}")
        if value < 0:
            raise T4SignalAlignmentError(f"{source_label}: union[{class_id}]={value} must be non-negative")
        values.append(value)
    return tuple(values)


def parse_canonical_stats(path: Path) -> tuple[float, ...]:
    """Parses --canonical-stats into a 171-element per-class pixel-union
    tuple indexed by class_id.

    Two accepted shapes:

    1. The historical e10 affinity-oracle sweep artifact shape (first
       candidate: ``global_sweep_ext_converged.json``), identified by the
       presence of ``payload.rows`` -- an array of typed per-configuration
       result rows. The canonical alpha=0.98, steps=320 row is selected by
       explicit typed equality on those two fields, never by array
       position or fuzzy string matching.
    2. A strict generic form: a top-level ``union`` key holding exactly a
       171-element array. Documented in docs/t4_signal_alignment_report.md
       as the form to use if a different canonical-union artifact schema
       needs its own adapter later -- add a new branch here keyed off an
       unambiguous, explicit shape marker, never by fuzzy detection.
    """
    path = Path(path)
    data = _load_strict_json(path, label="canonical-stats artifact")

    if "payload" in data:
        payload = _require_mapping(data["payload"], "canonical-stats artifact payload")
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise T4SignalAlignmentError("canonical-stats artifact payload.rows must be a non-empty array")
        matches = [
            row for row in rows
            if isinstance(row, Mapping)
            and type(row.get("alpha")) is float and row.get("alpha") == CANONICAL_ALPHA
            and type(row.get("steps")) is int and row.get("steps") == CANONICAL_STEPS
        ]
        if len(matches) != 1:
            raise T4SignalAlignmentError(
                f"canonical-stats artifact must contain exactly one row with alpha={CANONICAL_ALPHA} "
                f"and steps={CANONICAL_STEPS} (explicit typed fields, not array position); found {len(matches)}"
            )
        row = matches[0]
        return _validate_union_array(row.get("union"), source_label=f"{path}:payload.rows[alpha={CANONICAL_ALPHA},steps={CANONICAL_STEPS}].union")

    if "union" in data:
        return _validate_union_array(data["union"], source_label=f"{path}:union")

    raise T4SignalAlignmentError(
        f"canonical-stats artifact {path} matches neither the historical payload.rows shape "
        "nor the strict generic {'union': [...171 values...]} shape; refusing to fabricate a union"
    )


def build_union_weighted_proxy(
    per_class_alignment: Mapping[str, Any], union: Optional[Sequence[float]]
) -> dict:
    if union is None:
        return {
            "status": "unavailable",
            "reason": "no --canonical-stats input was supplied",
            "S": None, "sensitivity_196": None, "sensitivity_830": None,
        }

    represented = per_class_alignment["records"]
    for record in represented:
        class_id = record["class_id"]
        if not (0 <= class_id < CANONICAL_CLASS_COUNT):
            raise T4SignalAlignmentError(f"represented class_id {class_id} outside [0,{CANONICAL_CLASS_COUNT - 1}]")
        if union[class_id] <= 0.0:
            raise T4SignalAlignmentError(
                f"represented T4 class_id={class_id} has non-positive canonical union ({union[class_id]}); "
                "cannot compute net_c / U_c"
            )

    s_value = sum(record["net"] / union[record["class_id"]] for record in represented)
    sensitivity_196 = (100.0 / CANONICAL_CLASS_COUNT) * _INTERPOLATION_WEIGHT_FOOTPRINT_AVERAGE * s_value
    sensitivity_830 = (100.0 / CANONICAL_CLASS_COUNT) * _INTERPOLATION_WEIGHT_FOOTPRINT_MAXIMAL * s_value

    return {
        "status": "available",
        "S": s_value,
        "S_name": "union_weighted_anchor_alignment_proxy",
        "S_formula": "sum_c(net_c / U_c) over represented T4 classes",
        "S_unit": UNIT_MIXED_SIGN_PROXY,
        "S_caveats": [
            "net_c is measured in patch-anchor counts",
            "U_c is measured in pixels",
            "S is therefore a mixed-unit sign/alignment proxy, not a physically dimensioned quantity",
            "S is not Delta-mIoU",
            "S is not a lower or upper bound on any mIoU change",
            "S is not an oracle result",
        ],
        "homogeneous_neighbourhood_footprint_sensitivity_descriptive": {
            "sensitivity_196": sensitivity_196,
            "sensitivity_196_formula": "(100/171) * 196 * S",
            "sensitivity_830": sensitivity_830,
            "sensitivity_830_formula": "(100/171) * 830 * S",
            "caveats": [
                "196 corresponds only to the average bilinear interpolation weight footprint per patch node",
                "830 approximates a maximal bilinear-support footprint, not a typical one",
                "neither 196 nor 830 is a bound",
                "neither sensitivity value predicts actual mIoU change",
                "pixels within a node's footprint may already be correct, may carry a different GT label "
                "entirely, or may never cross an argmax decision boundary even if the anchor's label changes",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Section G: decision.
# ---------------------------------------------------------------------------

DECISION_STOP_SEMANTIC_CONSENSUS_COVER_DR = "STOP_SEMANTIC_CONSENSUS_COVER_DR"
DECISION_CONTINUE_SEMANTIC_CONSENSUS_EVALUATION = "CONTINUE_SEMANTIC_CONSENSUS_EVALUATION"


def build_decision(strict_t4: Mapping[str, Any], t4_prime: Mapping[str, Any], per_class: Mapping[str, Any]) -> dict:
    ci_contains_zero = strict_t4["bootstrap"]["ci_contains_zero"]
    image_macro_estimate = strict_t4["image_macro_estimate_fraction"]
    image_macro_nonpositive = image_macro_estimate is not None and image_macro_estimate <= 0.0
    t4_prime_weaker_and_more_third_label = t4_prime["weaker_than_strict_t4"]["both"]

    evidence = {
        "strict_t4_ci_contains_zero": ci_contains_zero,
        "strict_t4_image_macro_estimate_nonpositive": image_macro_nonpositive,
        "strict_t4_third_label_fraction": strict_t4["third_label_fraction"],
        "strict_t4_population_fraction": strict_t4["total_anchor_fraction"],
        "net_excluding_dominant_positive_class": per_class["total_net_excluding_largest_positive_contributor"],
        "t4_prime_weakens_trust_and_increases_third_label": t4_prime_weaker_and_more_third_label,
    }

    stop = ci_contains_zero and image_macro_nonpositive and t4_prime_weaker_and_more_third_label
    decision = DECISION_STOP_SEMANTIC_CONSENSUS_COVER_DR if stop else DECISION_CONTINUE_SEMANTIC_CONSENSUS_EVALUATION

    rationale = (
        "Strict-T4 direct anchor alignment is measurably positive in raw count, but the image-level "
        "bootstrap 95% CI for Delta_trust contains zero, the image-macro estimate (which does not let a "
        "few heavily-sampled images dominate) is non-positive, and broadening to T4-prime weakens "
        "Delta_trust while increasing the third-label failure rate rather than validating the strict "
        "population as merely 'too strict'. Together this evidence is not strong enough to justify "
        "semantic-consensus-guided COVER-DR."
        if stop else
        "The measured evidence conjunction (CI-contains-zero AND image-macro-nonpositive AND "
        "T4-prime-weaker-with-more-third-label) does not all hold for this report, so no stop "
        "recommendation is issued from this evidence alone."
    )

    scope_caveats = [
        "this decision is scoped to STOP semantic-consensus-guided COVER-DR specifically",
        "it does not claim all graph-topology methods are invalid",
        "it does not claim cross-view structural edge support is invalid",
        "it does not claim T4 reversals are necessarily semantically correct",
        "T4 is preserved as a useful operator-attributed-diffusion-reversal diagnostic",
    ]
    next_experiment = (
        "matched k11/k12 connectivity experiment: compare directed top-k=11 against the canonical "
        "top-k=12 graph under identical unary/affinity/solver settings to isolate whether indiscriminate "
        "single-neighbor pruning helps or hurts, independent of any semantic-consensus repair mechanism"
    )

    return {
        "decision": decision,
        "evidence": evidence,
        "rationale": rationale,
        "scope_caveats": scope_caveats,
        "next_experiment": next_experiment,
    }


# ---------------------------------------------------------------------------
# Orchestration and deterministic serialization.
# ---------------------------------------------------------------------------


def build_report(
    trust_report_path: Path,
    *,
    canonical_stats_path: Optional[Path] = None,
    repo_root: Path = REPOSITORY_ROOT,
    now: Optional[datetime] = None,
) -> dict:
    """Builds the complete derived report. Deterministic given identical
    inputs, except for the isolated ``provenance.generated_at_utc`` field."""
    loaded = load_and_validate_trust_report(trust_report_path)
    provenance = build_provenance(loaded, repo_root=repo_root)

    strict_t4 = build_strict_t4_summary(loaded.report)
    t4_prime = build_t4_prime_summary(loaded.report, strict_t4)
    per_class = build_per_class_alignment(loaded.report, strict_t4, repo_root=repo_root)

    provenance["class_count"] = per_class["represented_class_count"]
    generated_at = now if now is not None else datetime.now(timezone.utc)
    provenance["generated_at_utc"] = generated_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    union: Optional[tuple[float, ...]] = None
    if canonical_stats_path is not None:
        union = parse_canonical_stats(canonical_stats_path)
        provenance["input_canonical_stats"] = {
            "filename": Path(canonical_stats_path).name,
            "sha256": _sha256_of_file(canonical_stats_path, label="canonical-stats artifact"),
        }
    union_proxy = build_union_weighted_proxy(per_class, union)

    decision = build_decision(strict_t4, t4_prime, per_class)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "provenance": provenance,
        "strict_t4": strict_t4,
        "t4_prime": t4_prime,
        "per_class_alignment": per_class,
        "union_weighted_proxy": union_proxy,
        "decision": decision,
        "mIoU_impact_caveat": (
            "Direct anchor net and per-class net are patch-anchor counts. They are not directly "
            "convertible to mIoU impact: patches, bilinear-interpolated pixels, per-class unions, "
            "argmax decision margins, and nonlocal graph effects (a repaired anchor can shift "
            "neighboring anchors' RWR equilibrium) all separate anchor counts from segmentation-metric "
            "impact. Only the optional union-weighted proxy attempts any pixel-scale connection, and "
            "even that remains a sign/alignment proxy, not an mIoU estimate."
        ),
    }


_VOLATILE_FIELD_PATH = ("provenance", "generated_at_utc")


def canonical_projection(report: Mapping[str, Any]) -> dict:
    """Deep copy of ``report`` with only the isolated timestamp field
    removed -- used to prove determinism across repeated generation."""
    import copy

    projected = copy.deepcopy(dict(report))
    node = projected
    for key in _VOLATILE_FIELD_PATH[:-1]:
        node = node[key]
    node.pop(_VOLATILE_FIELD_PATH[-1], None)
    return projected


def serialize_report_json(report: Mapping[str, Any]) -> str:
    """Stable JSON serialization: sorted keys, fixed indentation, finite
    numbers only, no numpy scalar serialization, no ``default=str`` escape
    hatch (every value must already be a plain JSON-native type), newline
    at EOF."""
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    return payload + "\n"


def write_report_json(report: Mapping[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_report_json(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# Section H: deterministic Markdown renderer.
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], *, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    provenance = report["provenance"]
    strict_t4 = report["strict_t4"]
    t4_prime = report["t4_prime"]
    per_class = report["per_class_alignment"]
    union_proxy = report["union_weighted_proxy"]
    decision = report["decision"]

    lines: list[str] = []
    lines.append("# Final T4 Signal-Alignment Report")
    lines.append("")

    lines.append("## 1. Scope and provenance")
    lines.append("")
    lines.append(
        "This report analyzes an already-finalized trust/centrality report; it runs no model, no CUDA, "
        "and no dataset inference, and it re-derives no T4/trust/centrality sufficient statistics."
    )
    lines.append(f"- Input trust report: `{provenance['input_trust_report']['filename']}` "
                 f"(sha256 `{provenance['input_trust_report']['sha256']}`)")
    lines.append(f"- Trust report schema: `{provenance['input_trust_report']['schema_version']}`")
    lines.append(f"- Images: {provenance['image_count']}  Windows: {provenance['window_count']}  "
                 f"Represented T4 classes: {provenance['class_count']}")
    lines.append(f"- Bootstrap: seed={provenance['bootstrap_seed']}, "
                 f"resamples={provenance['bootstrap_resamples_requested']}, "
                 f"confidence={provenance['bootstrap_confidence_level']}, unit={provenance['bootstrap_unit']}")
    lines.append(f"- Generated: {provenance['generated_at_utc']} "
                 f"(generator git commit: {provenance.get('generator_git_commit') or 'unknown'})")
    lines.append("")
    lines.append(
        "Leave-one-window-out cross-view context is **not** a set of statistically independent "
        "observations: overlapping crops share most of their pixels, most of their DINO features, and "
        "the same frozen backbone and projection head. Unanimity among them is evidence that the "
        "crop-independent signal favors one label, not evidence of independent replication."
    )
    lines.append("")

    lines.append("## 2. Strict-T4 funnel")
    lines.append("")
    lines.append(f"- Raw anchors: {strict_t4['raw_anchor_count']}")
    lines.append(f"- Valid-GT anchors: {strict_t4['valid_gt_count']}  (ignored: {strict_t4['ignored_gt_count']})")
    lines.append(f"- Contributing images: {strict_t4['contributing_images']}  "
                 f"Zero-target images: {strict_t4['zero_target_images']}")
    lines.append(f"- Population fraction of all T0 anchors: {_fmt(strict_t4['total_anchor_fraction'], decimals=6)} "
                 "(strict T4 is a very sparse population)")
    lines.append("")
    lines.append(
        "T4 proves an **operator-attributed diffusion reversal**: the RWR diffusion operator, on the "
        "source crop's own neighborhood graph, reversed the frozen pre-RWR unary preference relative to "
        "unanimous cross-view context. It does not by itself establish that the reversal was semantically "
        "wrong -- a shared systematic unary bias could produce the same unanimous cross-view agreement "
        "on an incorrect label just as easily as a correct one."
    )
    lines.append("")

    lines.append("## 3. Trust result (direct anchor alignment)")
    lines.append("")
    lines.append(f"- consensus_correct: {strict_t4['consensus_correct']}  dissent_correct: {strict_t4['dissent_correct']}")
    lines.append(f"- **Direct anchor net** (consensus_correct - dissent_correct): "
                 f"**{strict_t4['direct_anchor_net']:+d}** anchors")
    lines.append(f"- Delta_trust: {_fmt(strict_t4['delta_trust_fraction'], decimals=6)} "
                 f"({_fmt(strict_t4['delta_trust_percentage_points'], decimals=4)} pp)")
    ci = strict_t4["bootstrap"]["ci95_fraction_difference"]
    lines.append(f"- Image-level bootstrap 95% CI (Delta_trust, fraction): [{_fmt(ci[0], decimals=5)}, {_fmt(ci[1], decimals=5)}]"
                 f"  -- contains zero: {strict_t4['bootstrap']['ci_contains_zero']}")
    aeci = strict_t4["anchor_equivalent_bootstrap_interval_descriptive"]
    if aeci is not None:
        lines.append(f"- Anchor-equivalent bootstrap interval (**descriptive only**): "
                     f"[{_fmt(aeci[0], decimals=2)}, {_fmt(aeci[1], decimals=2)}]. "
                     "This is not an integer confidence interval, not a bound on correctable anchors, "
                     "and not an mIoU interval.")
    lines.append("")
    lines.append(
        "Direct anchor net is a **patch-anchor-count** quantity. It is not directly convertible to mIoU "
        "impact: patches, bilinear-interpolated pixels, per-class unions, argmax decision margins, and "
        "nonlocal graph effects all separate anchor counts from segmentation-metric impact."
    )
    lines.append("")

    lines.append("## 4. Micro-versus-image-macro disagreement")
    lines.append("")
    lines.append(
        f"- Anchor-weighted (micro) Delta_trust: {_fmt(strict_t4['delta_trust_fraction'], decimals=6)}"
    )
    lines.append(
        f"- Image-macro estimate (equal weight per image, not per anchor): "
        f"{_fmt(strict_t4['image_macro_estimate_fraction'], decimals=6)}"
    )
    lines.append(
        "The micro estimate is positive; the image-macro estimate is "
        f"{'non-positive' if (strict_t4['image_macro_estimate_fraction'] or 0) <= 0 else 'positive'}. "
        "This disagreement means the raw positive anchor net is not evenly spread across images -- a "
        "small number of images with many anchors can dominate the micro estimate."
    )
    lines.append("")

    lines.append("## 5. Third-label failure mode")
    lines.append("")
    lines.append(
        f"- third_label_fraction at strict T4: {_fmt(strict_t4['third_label_fraction'], decimals=4)} "
        f"({strict_t4['third_label_count']} of {strict_t4['valid_gt_count']} anchors)"
    )
    lines.append(
        "A third-label outcome means ground truth agrees with **neither** the consensus label nor the "
        "dissenting label -- a large third-label fraction means direct anchor net, even when positive, "
        "resolves only a minority of the actionable disagreements towards a GT-correct outcome."
    )
    lines.append("")

    lines.append("## 6. Per-class concentration")
    lines.append("")
    lines.append(f"- Represented classes: {per_class['represented_class_count']}  "
                 f"(positive: {per_class['positive_class_count']}, negative: {per_class['negative_class_count']}, "
                 f"zero: {per_class['zero_class_count']})")
    lines.append(f"- Gross positive net: {per_class['gross_positive_net']:+d}  "
                 f"Gross negative net: {per_class['gross_negative_net']:+d}  Total net: {per_class['total_net']:+d}")
    if per_class["class_zero"] is not None:
        cz = per_class["class_zero"]
        name = f" ({cz['class_name']})" if cz["class_name"] else ""
        lines.append(f"- class 0{name} net: {cz['net']:+d}")
    lines.append(f"- Total net excluding the largest positive contributor: "
                 f"{per_class['total_net_excluding_largest_positive_contributor']:+d}")
    lines.append(
        "The positive total net is concentrated in a small number of classes; excluding the single "
        "largest positive contributor changes the sign of the total net, which weakens any claim that "
        "the positive direct anchor net reflects a broad, class-general improvement."
    )
    lines.append("")

    lines.append("## 7. T4 versus T4-prime")
    lines.append("")
    lines.append(f"- T4-prime raw/valid anchors: {t4_prime['raw_anchor_count']} / {t4_prime['valid_gt_count']} "
                 f"(growth over strict T4: "
                 f"{_fmt(t4_prime['relative_growth_from_strict_t4']['valid_gt_growth_ratio'], decimals=2)}x by valid count)")
    lines.append(f"- T4-prime direct anchor net: {t4_prime['direct_anchor_net']:+d}  "
                 f"Delta_trust: {_fmt(t4_prime['delta_trust_fraction'], decimals=6)}  "
                 f"third_label_fraction: {_fmt(t4_prime['third_label_fraction'], decimals=4)}")
    lines.append(
        f"- T4-prime bootstrap: {t4_prime['bootstrap']['status']}"
        + (f" ({t4_prime['bootstrap']['reason']})" if t4_prime["bootstrap"].get("reason") else "")
    )
    lines.append(
        "T4-prime expands coverage but "
        f"{'weakens' if t4_prime['weaker_than_strict_t4']['both'] else 'does not uniformly weaken'} "
        "alignment: broadening the population does not validate strict T4 as merely 'too strict'."
    )
    lines.append("")

    lines.append("## 8. Optional union-weighted proxy")
    lines.append("")
    if union_proxy["status"] == "unavailable":
        lines.append(f"Status: unavailable ({union_proxy['reason']}).")
    else:
        lines.append(f"- S (union_weighted_anchor_alignment_proxy): {union_proxy['S']:.6g}")
        sens = union_proxy["homogeneous_neighbourhood_footprint_sensitivity_descriptive"]
        lines.append(f"- Descriptive sensitivity at footprint 196 (average bilinear-support footprint per node, "
                     f"**not a bound**): {sens['sensitivity_196']:.6g}")
        lines.append(f"- Descriptive sensitivity at footprint 830 (maximal bilinear-support footprint, "
                     f"**not a bound**): {sens['sensitivity_830']:.6g}")
        lines.append(
            "S is a **mixed-unit sign/alignment proxy** (patch-anchor counts divided by pixel counts). "
            "It is **not Delta-mIoU**, not a lower or upper bound on any mIoU change, and not an oracle "
            "result. Neither descriptive sensitivity is a bound, and neither predicts actual mIoU change: "
            "pixels within a node's footprint may already be correct, may carry a different GT label "
            "entirely, or may never cross an argmax decision boundary."
        )
    lines.append("")

    lines.append("## 9. Final decision")
    lines.append("")
    lines.append(f"**{decision['decision']}**")
    lines.append("")
    lines.append(decision["rationale"])
    lines.append("")
    lines.append("Scope of this decision:")
    for caveat in decision["scope_caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")

    lines.append("## 10. Next experiment")
    lines.append("")
    lines.append(decision["next_experiment"])
    lines.append("")

    return "\n".join(lines) + "\n"


def write_markdown_report(report: Mapping[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")


__all__ = [
    "T4SignalAlignmentError",
    "REPORT_SCHEMA_VERSION",
    "SUPPORTED_TRUST_REPORT_SCHEMA_VERSION",
    "SUPPORTED_TRUST_CENTRALITY_SECTION_SCHEMA_VERSION",
    "CANONICAL_ALPHA",
    "CANONICAL_STEPS",
    "CANONICAL_CLASS_COUNT",
    "DECISION_STOP_SEMANTIC_CONSENSUS_COVER_DR",
    "DECISION_CONTINUE_SEMANTIC_CONSENSUS_EVALUATION",
    "LoadedTrustReport",
    "load_and_validate_trust_report",
    "build_provenance",
    "build_strict_t4_summary",
    "build_t4_prime_summary",
    "build_per_class_alignment",
    "parse_canonical_stats",
    "build_union_weighted_proxy",
    "build_decision",
    "build_report",
    "canonical_projection",
    "serialize_report_json",
    "write_report_json",
    "render_markdown",
    "write_markdown_report",
]
