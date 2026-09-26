"""Locked 48-image holdout evaluation runner, for
research_specs/semantic_evidence_holdout_evaluation_v1.toml.

Implements Gates B (query_correctness information), C (propagation_outcome
information), and D (subgroup sanity) against the locked holdout
population, applying the four already-fitted discovery models UNCHANGED
(no refit, no tuning, no feature selection, no discovery/holdout pooling)
and computing the frozen, image-paired bootstrap comparison.

Reuses -- never reimplements -- the committed, independently-verified
discovery mechanisms: canonical-window tiling and clamped-window geometry
(`compute_canonical_windows`, `t0_t320_anchor_pixel_bounds`,
`footprint_pixel_bounds`, `intervention_core_native_bounds`, all reached
through `_run_phase_a`/`_run_phase_b`), GT-phase isolation (`PhaseEventLog`,
`_run_phase_a`/`_run_phase_b` themselves), checkpoint/resume/atomic-install
(`build_checkpoint_metadata`, `validate_checkpoint_metadata`,
`_atomic_write_json`, `_load_valid_artifact_or_none`, `atomic_install`),
and per-image artifact validators (`validate_gt_free_artifact`,
`validate_gt_label_artifact`, `validate_gt_pair_linkage`). None of these
functions embed a "discovery" identity in their logic -- they operate on
whatever `image_id`s and loaders their caller supplies -- so they are
reused here unmodified, by import, for the locked holdout population
instead. `semantic_evidence_discovery_execute.py` itself is NOT edited by
this module.

Two committed-discovery functions are NOT reused directly because they
embed discovery-specific identity in their own docstring/output, not
because their underlying algorithm differs:
  - `validate_discovery_image_id`'s own docstring purpose is "reject
    holdout IDs" -- reusing it for holdout would be semantically
    backwards even though its regex/traversal logic is identical. A small
    `validate_holdout_image_id` adapter below reimplements the SAME
    checks with holdout-appropriate wording, not a different algorithm.
  - `build_checkpoint_metadata`/`validate_checkpoint_metadata` bake the
    JSON field name "discovery_image_count" into their schema. Rather
    than reimplementing their digest/prefix-matching algorithm, this
    module reuses them AS-IS for the transient, internal-only
    `.inprogress` checkpoint (deleted after finalization, never a public
    artifact) and documents the field-name quirk; the public, installed
    `holdout_result.json`/`holdout_decision.json` schemas defined below
    use holdout-appropriate field names throughout.

AUTHORIZATION: `preflight_holdout` and `run_holdout_evaluation` (the only
two functions the CLI/launcher may call for real holdout access) check
`[authorization.frozen_flags].holdout_execution_authorized` in the
committed holdout authority as their FIRST action, before touching any
image/GT path, before any torch import. Under the currently committed
authority this flag is False, so both raise `HoldoutEvaluationError`
immediately. Real execution requires a separate, later, explicit
successor authority to set that flag True -- this module does not, and
cannot, do that itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Optional

import semantic_evidence_discovery_execute as disc


class HoldoutEvaluationError(Exception):
    pass


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        raise exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutEvaluationError(message)


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
HOLDOUT_AUTHORITY_TOML_PATH = "research_specs/semantic_evidence_holdout_evaluation_v1.toml"

_HOLDOUT_EXPECTED_DEPENDENCY_ROLES = frozenset({
    "discovery_execution_authority", "feasibility_contract", "t320_source_authority",
    "t320_source_manifest", "t320_implementation", "clamped_window_geometry_authority",
    "discovery_executor_implementation", "population_manifest",
})
_HOLDOUT_DEPENDENCY_RECORD_KEYS = frozenset({"role", "path", "sha256", "must_be_committed_at_required_head", "reason"})


# ---------------------------------------------------------------------------
# Layer 1: torch-free protocol verification and preflight
# ---------------------------------------------------------------------------

def _load_holdout_authority(repo_root: Path) -> dict:
    with open(repo_root / HOLDOUT_AUTHORITY_TOML_PATH, "rb") as fh:
        return tomllib.load(fh)


def verify_dependency_hashes_holdout(repo_root: Path, data: dict) -> dict:
    """Torch-free: independently re-verifies every declared dependency's
    raw bytes against `git show <required_head>:<path>`. This is a SMALL,
    holdout-authority-specific verifier -- NOT a reuse of
    semantic_evidence_discovery_execution_identity.verify_dependency_hashes,
    which is hard-coded to that OTHER document's own fixed dependency-role
    set (`_EXPECTED_DEPENDENCY_ROLES`) and would reject this file's
    (correctly different) role set. It reuses only the generic
    `disc._git_show_bytes` subprocess helper, never reimplementing git
    plumbing."""
    prereq = data["prerequisite"]
    required_head = prereq["required_head"]
    deps = data["dependency"]
    _require(isinstance(deps, list) and len(deps) >= 1, "dependency list must be a non-empty list")

    declared_roles = frozenset(e.get("role") for e in deps if isinstance(e, dict))
    _require(
        declared_roles == _HOLDOUT_EXPECTED_DEPENDENCY_ROLES,
        f"holdout dependency role set mismatch: missing={sorted(_HOLDOUT_EXPECTED_DEPENDENCY_ROLES - declared_roles)}, "
        f"extra={sorted(declared_roles - _HOLDOUT_EXPECTED_DEPENDENCY_ROLES)}",
    )

    seen_roles, seen_paths = set(), set()
    results = []
    for entry in deps:
        _require(isinstance(entry, dict), "dependency entry must be a table")
        actual_keys = frozenset(entry.keys())
        _require(
            actual_keys == _HOLDOUT_DEPENDENCY_RECORD_KEYS,
            f"dependency entry key mismatch: missing={_HOLDOUT_DEPENDENCY_RECORD_KEYS - actual_keys}, extra={actual_keys - _HOLDOUT_DEPENDENCY_RECORD_KEYS}",
        )
        role, path, declared_sha = entry["role"], entry["path"], entry["sha256"]
        _require(role not in seen_roles, f"duplicate dependency role: {role!r}")
        _require(path not in seen_paths, f"duplicate dependency path: {path!r}")
        seen_roles.add(role)
        seen_paths.add(path)
        _require(not Path(path).is_absolute() and ".." not in Path(path).parts, f"unsafe dependency path: {path!r}")
        _require(entry["must_be_committed_at_required_head"] is True, f"dependency[{role!r}].must_be_committed_at_required_head must be true")

        actual_bytes = disc._git_show_bytes(repo_root, required_head, path)
        actual_sha = hashlib.sha256(actual_bytes).hexdigest()
        results.append({"role": role, "path": path, "declared_sha256": declared_sha, "actual_sha256": actual_sha, "match": actual_sha == declared_sha})

    return {"records": results, "all_match": all(r["match"] for r in results), "role_count": len(results)}


def verify_protocol_holdout(repo_root: Path) -> dict:
    """Torch-free: re-verifies the committed holdout authority's own
    prerequisite/dependency/authorization-non-widening state. Mirrors the
    shape of `disc.verify_protocol` (including the same "known lifecycle
    only" property: `prerequisite_gate_head_ok` will be False as soon as
    any later commit lands, exactly like the discovery authority's own
    required_head anchor -- this is by design, not a defect)."""
    data = _load_holdout_authority(repo_root)
    prereq = data["prerequisite"]

    current_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True).stdout.strip()
    current_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root), capture_output=True, text=True).stdout.strip()
    head_ok = current_head == prereq["required_head"]
    branch_ok = current_branch == prereq["required_branch"]

    deps = verify_dependency_hashes_holdout(repo_root, data)

    frozen = data["authorization"]["frozen_flags"]
    authorization_non_widening_ok = all(v is False for v in frozen.values())

    gate_c = data["gate_c_authorization"]

    return {
        "verified": head_ok and branch_ok and deps["all_match"] and authorization_non_widening_ok,
        "prerequisite_gate_head_ok": head_ok,
        "prerequisite_gate_branch_ok": branch_ok,
        "dependency_hashes_ok": deps["all_match"],
        "authorization_non_widening_ok": authorization_non_widening_ok,
        "gate_c_defined": True,
        "gate_c_computable_under_this_contract": gate_c["computable_under_this_contract"],
        "gate_c_execution_authorized_by_this_document": gate_c["execution_authorized_by_this_document"],
        "holdout_execution_authorized": frozen["holdout_execution_authorized"],
    }


def load_holdout_population(repo_root: Path) -> list:
    """Loads exactly the 48 frozen locked-holdout IDs, in frozen order,
    from the committed discovery-execution authority's own
    population_manifest dependency -- mirrors
    `disc.load_discovery_population` exactly, reading
    `locked_holdout_rank_range`/`locked_holdout_digest`/
    `locked_holdout_count` instead of the discovery-side keys. Never
    hardcodes a literal holdout ID anywhere in source (matching
    `[population.holdout_protection].holdout_ids_not_enumerated_in_this_
    document` in the discovery-execution authority)."""
    data = disc._load_authority(repo_root)
    required_head = data["prerequisite"]["required_head"]
    manifest_dep = next(e for e in data["dependency"] if e["role"] == "population_manifest")
    raw = disc._git_show_bytes(repo_root, required_head, manifest_dep["path"])
    manifest = json.loads(raw)

    pop = data["population"]
    hr_lo, hr_hi = pop["locked_holdout_rank_range"]
    records = sorted(manifest["records"], key=lambda r: r["selection_rank"])
    holdout_ids = [r["canonical_image_id"] for r in records if hr_lo <= r["selection_rank"] <= hr_hi]
    _require(len(holdout_ids) == pop["locked_holdout_count"], "loaded holdout population count mismatch")

    digest = hashlib.sha256("".join(i + "\n" for i in holdout_ids).encode("ascii")).hexdigest()
    _require(digest == pop["locked_holdout_digest"], "loaded holdout population digest mismatch")

    return holdout_ids


_COCO_ID_RE = re.compile(r"^[0-9]{12}$")


def validate_holdout_image_id(image_id: str, authorized_ids: tuple) -> str:
    """Holdout-specific adapter: same regex/traversal/whitelist checks as
    `disc.validate_discovery_image_id`, but with holdout-appropriate
    wording -- that function's own docstring purpose is to REJECT holdout
    IDs, so reusing it directly here would be semantically backwards even
    though the underlying check is identical. This is not a second
    implementation of a discovery algorithm; it is the same three-line
    string-safety check, restated for the opposite whitelist."""
    if type(image_id) is not str:
        raise HoldoutEvaluationError(f"image_id must be a str, got {type(image_id).__name__}")
    if image_id != image_id.strip() or any(ch.isspace() for ch in image_id):
        raise HoldoutEvaluationError(f"image_id must not contain whitespace: {image_id!r}")
    if "/" in image_id or "\\" in image_id or ".." in image_id or image_id.startswith("-"):
        raise HoldoutEvaluationError(f"image_id must not contain path separators, traversal, or look option-like: {image_id!r}")
    if not _COCO_ID_RE.fullmatch(image_id):
        raise HoldoutEvaluationError(f"image_id must be exactly 12 decimal digits: {image_id!r}")
    if image_id not in authorized_ids:
        raise HoldoutEvaluationError(f"image_id is not one of the 48 authorized locked-holdout IDs: {image_id!r}")
    return image_id


def _verify_no_discovery_overlap(repo_root: Path, holdout_ids: list) -> None:
    discovery_ids = disc.load_discovery_population(repo_root)
    overlap = sorted(set(holdout_ids) & set(discovery_ids))
    _require(not overlap, f"locked holdout population overlaps discovery population: {overlap[:3]}{'...' if len(overlap) > 3 else ''}")


def _verify_fit_digests(holdout_data: dict, discovery_result: dict) -> list:
    declared = {(f["task_id"], f["model_id"]): f["fit_digest"] for f in holdout_data["frozen_model_fits"]["fit"]}
    actual = {(f["task_id"], f["model_id"]): f for f in discovery_result.get("model_fit_summaries", [])}
    checks = []
    for key, declared_digest in declared.items():
        fit = actual.get(key)
        match = fit is not None and fit.get("fit_digest") == declared_digest
        checks.append({"task_id": key[0], "model_id": key[1], "declared_fit_digest": declared_digest, "match": match})
    return checks


def verify_discovery_result_file_hash(holdout_data: dict, discovery_result_path: Path) -> dict:
    """WHOLE-FILE integrity gate: hashes `discovery_result_path`'s raw
    bytes and compares against
    `[discovery_result_binding.artifact_hashes].discovery_result_json` in
    the committed holdout authority -- BEFORE any coefficient inside the
    file is parsed or applied.

    This is the check that actually detects in-place numeric tampering
    that leaves an individual `fit_digest` label unchanged: `fit_digest`
    (computed by the protected `fit_logistic_model`) includes `row_keys`,
    which is never persisted in `discovery_result.json`, so
    `load_frozen_fits` cannot independently recompute that digest from the
    file alone -- it can only compare two copies of the same label
    (catching a wrong fit substituted under the right key, never a
    tampered-but-relabeled one). This function closes exactly that gap,
    at the file-content level, using only the SHA-256 the holdout
    authority already bound -- no new digest formula, no edit to the
    protected discovery digest algorithm."""
    expected = holdout_data["discovery_result_binding"]["artifact_hashes"]["discovery_result_json"]
    actual = hashlib.sha256(Path(discovery_result_path).read_bytes()).hexdigest()
    _require(
        actual == expected,
        f"discovery_result.json whole-file SHA-256 mismatch: expected={expected} actual={actual} "
        "-- this file's content no longer matches what the committed holdout authority bound; "
        "refusing to parse or apply any coefficient from it",
    )
    return {"expected_sha256": expected, "actual_sha256": actual, "match": True}


def preflight_holdout(repo_root: Path, *, weight_dir: str, image_root: Path, gt_root: Path, output_root: Path,
                       discovery_result_path: Path, mode: str = "fresh") -> dict:
    """Torch-free readiness check. FAILS CLOSED, before touching any
    holdout image/GT path, if
    `[authorization.frozen_flags].holdout_execution_authorized` is not
    True in the committed holdout authority -- currently always the case.
    Never imports torch/torchvision/mmcv/mmseg/numpy/PIL and never calls
    torch.load; checkpoint verification is raw-byte SHA-256 only; GT
    verification is path/metadata only.

    Raises `HoldoutEvaluationError` for every documented holdout-domain
    failure -- including a reused discovery-module validator's own
    `disc.DiscoveryExecutionError` (translated here, original preserved as
    `__cause__`), so a caller that only catches `HoldoutEvaluationError`
    receives all of them. Programming errors unrelated to this domain
    (e.g. a malformed call, an unexpected type) are never caught or
    reinterpreted as an authorization or scientific outcome."""
    _require(mode in ("fresh", "resume", "finalize_only"), f"unknown mode: {mode!r}")

    data = _load_holdout_authority(repo_root)
    deps = verify_dependency_hashes_holdout(repo_root, data)
    _require(deps["all_match"], "committed holdout-authority dependency hashes do not match")

    frozen = data["authorization"]["frozen_flags"]
    _require(
        frozen["holdout_execution_authorized"] is True,
        "holdout execution is not authorized under the committed authority "
        f"({HOLDOUT_AUTHORITY_TOML_PATH}: authorization.frozen_flags.holdout_execution_authorized=False). "
        "A separate, explicit successor authority must set this flag to true before any "
        "real holdout preflight may proceed past this point.",
    )

    try:
        return _preflight_holdout_core(repo_root, data, weight_dir=weight_dir, image_root=image_root,
                                        gt_root=gt_root, output_root=output_root,
                                        discovery_result_path=discovery_result_path, mode=mode)
    except disc.DiscoveryExecutionError as exc:
        raise HoldoutEvaluationError(str(exc)) from exc


def _preflight_holdout_core(repo_root: Path, data: dict, *, weight_dir: str, image_root: Path, gt_root: Path,
                             output_root: Path, discovery_result_path: Path, mode: str) -> dict:
    """The actual readiness logic, factored out from `preflight_holdout` so
    it is independently testable with synthetic populations/paths without
    needing a (never-created-in-tests) modified copy of the real committed
    authority. Never called directly by the CLI -- only `preflight_holdout`
    is."""
    holdout_ids = load_holdout_population(repo_root)
    _require(len(holdout_ids) == 48, "expected exactly 48 locked-holdout IDs")
    _verify_no_discovery_overlap(repo_root, holdout_ids)

    # Whole-file integrity gate FIRST, before any coefficient is parsed and
    # before any holdout image/GT path is touched -- see
    # verify_discovery_result_file_hash's own docstring for why this,
    # rather than the per-fit fit_digest check alone, is what actually
    # detects in-place numeric tampering.
    discovery_result_path = Path(discovery_result_path)
    _require(discovery_result_path.exists(), f"discovery_result_path does not exist: {discovery_result_path}")
    verify_discovery_result_file_hash(data, discovery_result_path)
    discovery_result = json.loads(discovery_result_path.read_text())
    fit_checks = _verify_fit_digests(data, discovery_result)
    _require(len(fit_checks) == 4, f"expected exactly 4 frozen fit bindings, found {len(fit_checks)}")
    _require(all(c["match"] for c in fit_checks), f"frozen fit_digest mismatch: {[c for c in fit_checks if not c['match']]}")

    image_root = Path(image_root)
    missing_images = [i for i in holdout_ids if not (image_root / f"{i}.jpg").exists()]
    _require(not missing_images, f"missing holdout image files: {missing_images[:3]}{'...' if len(missing_images) > 3 else ''}")

    gt_root = Path(gt_root)
    _require(gt_root.exists() and gt_root.is_dir(), f"gt_root does not exist or is not a directory: {gt_root}")
    gt_problems = []
    for image_id in holdout_ids:
        gt_path = gt_root / f"{image_id}.png"
        if not (gt_path.exists() and gt_path.is_file()):
            gt_problems.append(f"{image_id}: missing ({gt_path})")
    _require(not gt_problems, f"invalid or missing holdout GT annotation paths ({len(gt_problems)} of 48): {gt_problems[:3]}{'...' if len(gt_problems) > 3 else ''}")

    checkpoint_checks = []
    for key, filename, expected_sha in disc._external_checkpoint_bindings():
        path = Path(weight_dir) / filename if key != "e3_projection_checkpoint" else repo_root / "weights" / filename
        exists = path.exists()
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
        checkpoint_checks.append({"key": key, "path": str(path), "sha256": actual_sha, "match": exists and actual_sha == expected_sha})
    _require(all(c["match"] for c in checkpoint_checks), "external checkpoint raw-byte hash mismatch")

    output_root = Path(output_root)
    if mode == "fresh":
        _require(not output_root.exists(), f"fresh mode requires an absent output_root: {output_root}")
    else:
        _require(output_root.exists(), f"{mode} mode requires an existing output_root: {output_root}")

    return {
        "status": "READY_FOR_48_IMAGE_HOLDOUT_EVALUATION",
        "mode": mode,
        "holdout_image_count": len(holdout_ids),
        "holdout_id_prefix_sample": holdout_ids[:2],
        "checkpoint_checks": checkpoint_checks,
        "fit_digest_checks": fit_checks,
        "gt_checks_count": len(holdout_ids),
        "output_root": str(output_root),
        "authorizes": {"gate_b": True, "gate_c": data["gate_c_authorization"]["computable_under_this_contract"], "gate_d": True, "combined_96": False, "full_5000": False},
    }


# ---------------------------------------------------------------------------
# Layer 2: frozen-fit application (no refit) and metric computation
# ---------------------------------------------------------------------------

def load_frozen_fits(holdout_data: dict, discovery_result: dict) -> dict:
    """Loads and validates the four already-fitted discovery models from an
    ALREADY-PARSED `discovery_result.json`-shaped dict.

    Scope of what THIS function guarantees (label/shape validation only):
    fit_digest STRING equality against the holdout authority's own bound
    value for each (task_id, model_id); feature_order/coefficient/scaler
    length agreement; finite numeric values. It never returns a fit for a
    missing or wrongly-labeled (task_id, model_id) slot.

    Scope of what THIS function does NOT guarantee (whole-artifact
    integrity -- enforced separately, by `verify_discovery_result_file_
    hash`, BEFORE this function is ever called on real data): the
    protected `fit_logistic_model`'s own `fit_digest` formula includes
    `row_keys`, a list never persisted in `discovery_result.json` -- so
    this function has no way to independently RECOMPUTE that digest from
    the file alone. It only compares two copies of the same label
    (declared vs. persisted), which catches a DIFFERENT fit substituted
    under the right key, but would NOT catch an in-place coefficient
    tamper that leaves the `fit_digest` string itself unchanged. Callers
    (`preflight_holdout`, `run_holdout_evaluation`) close that gap by
    calling `verify_discovery_result_file_hash` on the raw file bytes
    first -- this function must never be treated as a substitute for that
    check.

    Returns {(task_id, model_id): fit_summary_dict}. Raises before
    returning anything if a fit is missing, mismatched, or non-finite --
    never silently substitutes, drops, or refits one."""
    declared = {(f["task_id"], f["model_id"]): f["fit_digest"] for f in holdout_data["frozen_model_fits"]["fit"]}
    by_key = {(f["task_id"], f["model_id"]): f for f in discovery_result.get("model_fit_summaries", [])}

    fits = {}
    for key, declared_digest in declared.items():
        summary = by_key.get(key)
        _require(summary is not None, f"discovery_result.json is missing model_fit_summaries entry for {key}")
        _require(summary["fit_digest"] == declared_digest, f"fit_digest mismatch for {key}: declared={declared_digest!r} actual={summary['fit_digest']!r}")
        n = len(summary["coefficients"])
        _require(n == len(summary["feature_order"]), f"{key}: coefficients/feature_order length mismatch")
        _require(n == len(summary["scaler_mean"]) == len(summary["scaler_scale"]), f"{key}: scaler_mean/scaler_scale length mismatch")
        _require(all(math.isfinite(c) for c in summary["coefficients"]), f"{key}: non-finite coefficient")
        _require(math.isfinite(summary["intercept"]), f"{key}: non-finite intercept")
        _require(all(math.isfinite(v) for v in summary["scaler_mean"]), f"{key}: non-finite scaler_mean")
        _require(all(math.isfinite(v) and v != 0 for v in summary["scaler_scale"]), f"{key}: non-finite or zero scaler_scale")
        fits[key] = summary
    _require(len(fits) == 4, f"expected exactly 4 frozen fits, found {len(fits)}")
    return fits


def apply_frozen_fit(fit: dict, feature_vector: list) -> float:
    """Predicted probability P(y=1) under an already-fitted, frozen
    logistic model: sigmoid(w . ((x - scaler_mean) / scaler_scale) +
    intercept). Pure arithmetic; no sklearn, no refit, no tuning, no
    feature selection. `feature_vector` must be in the exact order
    `fit['feature_order']` declares (caller's responsibility -- this
    function only checks length, not semantic order, since it cannot
    know the caller's own column meaning)."""
    coeffs = fit["coefficients"]
    _require(len(feature_vector) == len(coeffs), "feature_vector length does not match the frozen fit's coefficient count")
    z = fit["intercept"]
    for x, mean, scale, w in zip(feature_vector, fit["scaler_mean"], fit["scaler_scale"], coeffs):
        z += w * ((x - mean) / scale)
    z = max(-700.0, min(700.0, z))  # avoid OverflowError in math.exp; does not change any in-range result
    return 1.0 / (1.0 + math.exp(-z))


def compute_auroc(y_true: list, y_score: list) -> float:
    """Lazy sklearn import -- never at module import time, so
    verify_protocol_holdout/preflight_holdout stay torch/numpy/sklearn-free."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def compute_average_precision(y_true: list, y_score: list) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def both_classes_present(y: list) -> bool:
    return len(set(y)) >= 2


# ---------------------------------------------------------------------------
# Layer 2: image-paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap(rows_by_image: dict, metric_fn: Callable[[list, list], float], *,
                      replicates: int = 10000, seed: int = 0,
                      min_valid_replicate_fraction: float = 0.90) -> dict:
    """[statistics.bootstrap], preserved verbatim: resamples HOLDOUT IMAGES
    with replacement (never individual rows), `replicates` times, with a
    fixed `seed` for reproducibility. `rows_by_image`:
    {image_id: [(y_true, score_a, score_b), ...]} -- every row belonging
    to a sampled image is included once per sampled occurrence
    (multiplicity preserved). A replicate is DEGENERATE for this metric if
    its resampled row set does not contain both classes; its value is
    NOT_A_NUMBER and excluded from the percentile CI. If fewer than
    `min_valid_replicate_fraction` of replicates are non-degenerate, the
    CI is INCONCLUSIVE (ci_lower/ci_upper are None). `refit_inside_
    replicates` is always False: this function only ever calls
    `metric_fn` on already-computed (y_true, score) pairs -- no model
    fitting occurs here or anywhere in a replicate."""
    import random

    image_ids = sorted(rows_by_image.keys())
    n = len(image_ids)
    _require(n >= 1, "paired_bootstrap requires at least one image")

    rng = random.Random(seed)
    diffs = []
    for _ in range(replicates):
        sampled_images = [image_ids[rng.randrange(n)] for _ in range(n)]
        y_true, score_a, score_b = [], [], []
        for iid in sampled_images:
            for (y, sa, sb) in rows_by_image[iid]:
                y_true.append(y)
                score_a.append(sa)
                score_b.append(sb)
        if not both_classes_present(y_true):
            diffs.append(float("nan"))
            continue
        diffs.append(metric_fn(y_true, score_b) - metric_fn(y_true, score_a))

    valid_diffs = [d for d in diffs if not math.isnan(d)]
    valid_fraction = len(valid_diffs) / replicates

    result = {
        "replicates": replicates, "seed": seed,
        "valid_replicate_count": len(valid_diffs), "valid_replicate_fraction": valid_fraction,
        "min_valid_replicate_fraction": min_valid_replicate_fraction,
    }
    if valid_fraction < min_valid_replicate_fraction or not valid_diffs:
        result.update({"inconclusive": True, "ci_lower": None, "ci_upper": None, "point_estimate": None})
        return result

    valid_diffs.sort()
    lo_idx = min(len(valid_diffs) - 1, max(0, round(0.025 * (len(valid_diffs) - 1))))
    hi_idx = min(len(valid_diffs) - 1, max(0, round(0.975 * (len(valid_diffs) - 1))))
    result.update({
        "inconclusive": False,
        "ci_lower": valid_diffs[lo_idx],
        "ci_upper": valid_diffs[hi_idx],
        "point_estimate": sum(valid_diffs) / len(valid_diffs),
    })
    return result


# ---------------------------------------------------------------------------
# Layer 2: Gates B, C, D
# ---------------------------------------------------------------------------

def compute_gate_b(*, eligible_rows: int, classes_present: bool, auroc_a: float, auroc_b: float,
                    ap_a: float, ap_b: float, bootstrap: dict,
                    min_rows: int = 500, min_auroc_improvement: float = 0.03) -> dict:
    """[gates.B], preserved verbatim."""
    if eligible_rows < min_rows or not classes_present:
        return {"gate": "B", "status": "INCONCLUSIVE", "reason": "insufficient eligible rows or a missing class",
                "eligible_rows": eligible_rows, "min_rows": min_rows, "classes_present": classes_present}
    if bootstrap.get("inconclusive"):
        return {"gate": "B", "status": "INCONCLUSIVE", "reason": "insufficient valid bootstrap replicates",
                "eligible_rows": eligible_rows, "bootstrap_valid_replicate_fraction": bootstrap["valid_replicate_fraction"]}
    auroc_improvement = auroc_b - auroc_a
    ap_difference = ap_b - ap_a
    passed = (auroc_improvement >= min_auroc_improvement) and (bootstrap["ci_lower"] is not None and bootstrap["ci_lower"] > 0) and (ap_difference >= 0)
    return {
        "gate": "B", "status": "PASS" if passed else "FAIL",
        "eligible_rows": eligible_rows, "auroc_a": auroc_a, "auroc_b": auroc_b,
        "auroc_improvement": auroc_improvement, "ap_a": ap_a, "ap_b": ap_b, "ap_difference": ap_difference,
        "bootstrap_ci_lower": bootstrap["ci_lower"], "bootstrap_ci_upper": bootstrap["ci_upper"],
        "min_auroc_improvement": min_auroc_improvement,
    }


def compute_gate_c(*, eligible_rows: int, beneficial_dominant_rows: int, harmful_dominant_rows: int,
                    classes_present: bool, auroc_a: float, auroc_b: float, ap_a: float, ap_b: float,
                    bootstrap: dict, gate_c_execution_authorized: bool,
                    min_beneficial_rows: int = 100, min_harmful_rows: int = 100,
                    min_auroc_improvement: float = 0.03) -> dict:
    """[gates.C], preserved verbatim, plus [minimum_data_requirements]'s
    100/100 beneficial/harmful-dominant minimum. Never PASS, never SKIP on
    a missing binding or insufficient data -- always INCONCLUSIVE or FAIL,
    per the frozen contract's own computable_note."""
    if not gate_c_execution_authorized:
        return {"gate": "C", "status": "INCONCLUSIVE", "reason": "Gate C execution is not authorized under the committed authority"}
    if beneficial_dominant_rows < min_beneficial_rows or harmful_dominant_rows < min_harmful_rows or not classes_present:
        return {
            "gate": "C", "status": "INCONCLUSIVE", "reason": "below the frozen beneficial/harmful-dominant row minimum",
            "beneficial_dominant_rows": beneficial_dominant_rows, "harmful_dominant_rows": harmful_dominant_rows,
            "min_beneficial_rows": min_beneficial_rows, "min_harmful_rows": min_harmful_rows,
        }
    if bootstrap.get("inconclusive"):
        return {"gate": "C", "status": "INCONCLUSIVE", "reason": "insufficient valid bootstrap replicates",
                "bootstrap_valid_replicate_fraction": bootstrap["valid_replicate_fraction"]}
    auroc_improvement = auroc_b - auroc_a
    ap_difference = ap_b - ap_a
    passed = (auroc_improvement >= min_auroc_improvement) and (bootstrap["ci_lower"] is not None and bootstrap["ci_lower"] > 0) and (ap_difference >= 0)
    return {
        "gate": "C", "status": "PASS" if passed else "FAIL",
        "eligible_rows": eligible_rows, "beneficial_dominant_rows": beneficial_dominant_rows, "harmful_dominant_rows": harmful_dominant_rows,
        "auroc_a": auroc_a, "auroc_b": auroc_b, "auroc_improvement": auroc_improvement,
        "ap_a": ap_a, "ap_b": ap_b, "ap_difference": ap_difference,
        "bootstrap_ci_lower": bootstrap["ci_lower"], "bootstrap_ci_upper": bootstrap["ci_upper"],
        "min_auroc_improvement": min_auroc_improvement,
    }


def compute_gate_d(subgroup_results: list, *, min_subgroup_size_for_check: int = 100,
                    max_negative_auroc_difference: float = -0.01) -> dict:
    """[gates.D], preserved verbatim. `subgroup_results`: list of
    {"name": str, "task_id": str, "size": int, "auroc_difference": float}
    (auroc_difference = model_b - model_a AUROC on that subgroup's rows).
    Subgroups below `min_subgroup_size_for_check` are excluded from the
    rule's scope -- not evaluated, not separately marked pass/fail. If
    ZERO subgroups qualify, the overall status is explicitly
    NO_QUALIFYING_SUBGROUPS, never an implied PASS: the frozen authority
    states the rule only as a conditional constraint on qualifying
    subgroups, and does not itself define a disposition for the vacuous
    case."""
    checked = [s for s in subgroup_results if s["size"] >= min_subgroup_size_for_check]
    excluded = [s["name"] for s in subgroup_results if s["size"] < min_subgroup_size_for_check]

    if not checked:
        return {"gate": "D", "status": "NO_QUALIFYING_SUBGROUPS", "checked_subgroups": [], "excluded_subgroups": excluded}

    violations = [s for s in checked if s["auroc_difference"] < max_negative_auroc_difference]
    status = "FAIL" if violations else "PASS"
    return {
        "gate": "D", "status": status,
        "checked_subgroups": [{"name": s["name"], "task_id": s["task_id"], "size": s["size"], "auroc_difference": s["auroc_difference"]} for s in checked],
        "excluded_subgroups": excluded,
        "violating_subgroups": [s["name"] for s in violations],
        "max_negative_auroc_difference": max_negative_auroc_difference,
    }


# ---------------------------------------------------------------------------
# Layer 3: artifact schemas (validators reuse disc._validate_closed, which
# is fully generic -- no holdout-specific schema logic to reimplement)
# ---------------------------------------------------------------------------

_HOLDOUT_RESULT_REQUIRED_FIELDS = frozenset({
    "schema", "complete", "final",
    "identity_and_dependency_bindings", "population_binding", "implementation_binding",
    "holdout_image_ids_ordered", "frozen_fit_digests",
    "gt_free_feature_digest", "gt_dependent_label_digest", "scientific_collection_digest",
    "gate_b_result", "gate_c_result", "gate_d_result", "bootstrap_settings",
    "final_frozen_outcome", "holdout_accessed", "full_evaluation_accessed",
})

_HOLDOUT_DECISION_REQUIRED_FIELDS = frozenset({
    "schema", "gate_b_result", "gate_c_result", "gate_d_result",
    "overall_gates_outcome_status", "holdout_decision", "next_step",
})


def _validate_closed_holdout(d: dict, required: frozenset, label: str) -> None:
    """Wraps `disc._validate_closed` (the fully generic closed-schema
    checker -- reused, never reimplemented) and normalizes its exception
    type to `HoldoutEvaluationError`, so every public function in this
    module raises the same exception type. Without this, a caller
    (including this module's own CLI) that only catches
    `HoldoutEvaluationError` would miss a schema-validation failure."""
    try:
        disc._validate_closed(d, required, label)
    except disc.DiscoveryExecutionError as exc:
        raise HoldoutEvaluationError(str(exc)) from exc


def validate_holdout_result(d: dict) -> None:
    _validate_closed_holdout(d, _HOLDOUT_RESULT_REQUIRED_FIELDS, "holdout_result artifact")
    _require(isinstance(d["complete"], bool) and isinstance(d["final"], bool), "complete/final must be bool")
    _require(isinstance(d["holdout_image_ids_ordered"], list), "holdout_image_ids_ordered must be a list")
    _require(isinstance(d["holdout_accessed"], bool) and isinstance(d["full_evaluation_accessed"], bool), "holdout_accessed/full_evaluation_accessed must be bool")


def validate_holdout_decision(d: dict) -> None:
    _validate_closed_holdout(d, _HOLDOUT_DECISION_REQUIRED_FIELDS, "holdout_decision document")


def holdout_decision(gate_b: dict, gate_c: dict, gate_d: dict) -> str:
    """Mechanical combination of the three gate dispositions into an
    overall status string. Never a scientific judgment beyond what the
    three gates themselves already determined."""
    statuses = {gate_b["status"], gate_c["status"], gate_d["status"]}
    if "FAIL" in statuses:
        return "HOLDOUT_STOP_GATE_FAILED"
    if "INCONCLUSIVE" in statuses:
        return "HOLDOUT_INCONCLUSIVE"
    if gate_b["status"] == "PASS" and gate_c["status"] == "PASS" and gate_d["status"] in ("PASS", "NO_QUALIFYING_SUBGROUPS"):
        return "HOLDOUT_GATES_B_C_D_PASSED"
    return "HOLDOUT_INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Layer 3: checkpoint/atomic install (reusing disc's machinery verbatim for
# the transient .inprogress checkpoint; new install/verify wrappers for the
# public holdout_result.json/holdout_decision.json artifacts)
# ---------------------------------------------------------------------------

def _checkpoint_dir_for(output_root: Path) -> Path:
    return disc._checkpoint_dir_for(output_root)


def install_holdout_result(output_root: Path, checkpoint_meta: dict, gt_free_artifacts: list,
                            gt_label_artifacts: list, holdout_result: dict, decision: dict,
                            holdout_ids: list) -> dict:
    """Staged-write -> staged-validate -> atomic-rename -> reopen ->
    public-verify, mirroring `disc.install_discovery_result`'s sequence
    exactly but with holdout's own top-level artifact names and schemas.
    Reuses `disc.atomic_install` (the fully generic rename primitive)
    unchanged."""
    disc.validate_checkpoint_metadata(checkpoint_meta, holdout_ids)
    _require(len(gt_free_artifacts) == len(gt_label_artifacts), "gt_free_artifacts and gt_label_artifacts must have matching length")
    for a in gt_free_artifacts:
        disc.validate_gt_free_artifact(a)
    for a in gt_label_artifacts:
        disc.validate_gt_label_artifact(a)
    for free, label in zip(gt_free_artifacts, gt_label_artifacts):
        disc.validate_gt_pair_linkage(free, label)
    validate_holdout_result(holdout_result)
    validate_holdout_decision(decision)

    staging_parent = output_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_parent / f".staging-{output_root.name}-{os.getpid()}-{int(time.time() * 1000)}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        (staging_dir / "checkpoint.json").write_text(json.dumps(checkpoint_meta, indent=2))
        (staging_dir / "gt_free_features.json").write_text(json.dumps(gt_free_artifacts, indent=2))
        (staging_dir / "gt_dependent_labels.json").write_text(json.dumps(gt_label_artifacts, indent=2))
        (staging_dir / "holdout_result.json").write_text(json.dumps(holdout_result, indent=2))
        (staging_dir / "holdout_decision.json").write_text(json.dumps(decision, indent=2))

        reopened_gt_free = json.loads((staging_dir / "gt_free_features.json").read_text())
        reopened_gt_label = json.loads((staging_dir / "gt_dependent_labels.json").read_text())
        for a in reopened_gt_free:
            disc.validate_gt_free_artifact(a)
        for a in reopened_gt_label:
            disc.validate_gt_label_artifact(a)
        for free, label in zip(reopened_gt_free, reopened_gt_label):
            disc.validate_gt_pair_linkage(free, label)
        reopened_checkpoint = json.loads((staging_dir / "checkpoint.json").read_text())
        disc.validate_checkpoint_metadata(reopened_checkpoint, holdout_ids)
        reopened_result = json.loads((staging_dir / "holdout_result.json").read_text())
        validate_holdout_result(reopened_result)
        reopened_decision = json.loads((staging_dir / "holdout_decision.json").read_text())
        validate_holdout_decision(reopened_decision)

        disc.atomic_install(staging_dir, output_root)
    except BaseException:
        disc._reraise_if_fatal_and_cleanup(staging_dir)
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    installed_result = json.loads((output_root / "holdout_result.json").read_text())
    validate_holdout_result(installed_result)
    installed_decision = json.loads((output_root / "holdout_decision.json").read_text())
    validate_holdout_decision(installed_decision)
    return {"verified": True, "outcome": installed_decision["holdout_decision"], "path": str(output_root)}


def verify_holdout_result(output_root: Path, holdout_ids: list) -> dict:
    """Torch-free public post-install verification -- mirrors
    `disc.verify_result` exactly, reading holdout's own artifact names.

    Raises `HoldoutEvaluationError` for every documented failure --
    including a reused discovery-module validator's own
    `disc.DiscoveryExecutionError` (translated here, original preserved
    as `__cause__`) -- so a caller that only catches
    `HoldoutEvaluationError` receives all of them."""
    output_root = Path(output_root)
    _require(output_root.exists(), f"output_root does not exist: {output_root}")
    try:
        checkpoint_meta = json.loads((output_root / "checkpoint.json").read_text())
        disc.validate_checkpoint_metadata(checkpoint_meta, holdout_ids)

        gt_free_artifacts = json.loads((output_root / "gt_free_features.json").read_text())
        gt_label_artifacts = json.loads((output_root / "gt_dependent_labels.json").read_text())
        _require(len(gt_free_artifacts) == len(gt_label_artifacts), "gt_free_features.json/gt_dependent_labels.json length mismatch")
        for a in gt_free_artifacts:
            disc.validate_gt_free_artifact(a)
        for a in gt_label_artifacts:
            disc.validate_gt_label_artifact(a)
        for free, label in zip(gt_free_artifacts, gt_label_artifacts):
            disc.validate_gt_pair_linkage(free, label)

        result = json.loads((output_root / "holdout_result.json").read_text())
        validate_holdout_result(result)
        decision = json.loads((output_root / "holdout_decision.json").read_text())
        validate_holdout_decision(decision)
        return {"verified": True, "outcome": decision["holdout_decision"], "path": str(output_root)}
    except disc.DiscoveryExecutionError as exc:
        raise HoldoutEvaluationError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Layer 2: top-level orchestrator
# ---------------------------------------------------------------------------

def run_holdout_evaluation(repo_root: Path, *, weight_dir: str, image_root: Path, gt_root: Path,
                            output_root: Path, discovery_result_path: Path, mode: str,
                            slurm_job_id: str = "", node: str = "",
                            image_loader=None, model_and_text_provider=None, gt_loader=None, forward_fn=None) -> dict:
    """Public entry point. FAILS CLOSED, before importing torch, before
    touching any holdout path, if
    `[authorization.frozen_flags].holdout_execution_authorized` is not
    True in the committed holdout authority -- currently always the case.
    This is the ONLY function the CLI/launcher's `execute` command may
    call for real holdout evaluation.

    Raises `HoldoutEvaluationError` for every documented holdout-domain
    failure -- including a reused discovery-module validator's own
    `disc.DiscoveryExecutionError` (translated here, original preserved as
    `__cause__`), so a caller that only catches `HoldoutEvaluationError`
    receives all of them. Programming errors unrelated to this domain are
    never caught or reinterpreted as an authorization or scientific
    outcome."""
    _require(mode in ("fresh", "resume", "finalize_only"), f"unknown mode: {mode!r}")

    data = _load_holdout_authority(repo_root)
    frozen = data["authorization"]["frozen_flags"]
    _require(
        frozen["holdout_execution_authorized"] is True,
        "holdout execution is not authorized under the committed authority "
        f"({HOLDOUT_AUTHORITY_TOML_PATH}: authorization.frozen_flags.holdout_execution_authorized=False). "
        "A separate, explicit successor authority must set this flag to true before any "
        "real holdout evaluation may proceed.",
    )

    try:
        discovery_result_path = Path(discovery_result_path)
        # Whole-file integrity gate FIRST, before any coefficient is parsed
        # -- see verify_discovery_result_file_hash's own docstring.
        verify_discovery_result_file_hash(data, discovery_result_path)
        discovery_result = json.loads(discovery_result_path.read_text())
        frozen_fits = load_frozen_fits(data, discovery_result)

        holdout_ids = load_holdout_population(repo_root)
        _verify_no_discovery_overlap(repo_root, holdout_ids)

        if image_loader is None:
            authorized_ids = tuple(holdout_ids)
            image_loader = lambda image_id: disc._default_real_image_loader(Path(image_root), image_id, authorized_ids)
        if gt_loader is None:
            gt_loader = lambda image_id: disc._default_real_gt_loader(Path(gt_root), image_id)
        if model_and_text_provider is None:
            model_and_text_provider = lambda: disc._default_real_model_and_text_provider(repo_root, weight_dir)[:2]

        return _run_holdout_evaluation_core(
            repo_root, data, frozen_fits,
            output_root=Path(output_root), mode=mode, holdout_ids=holdout_ids,
            image_loader=image_loader, model_and_text_provider=model_and_text_provider,
            gt_loader=gt_loader, forward_fn=forward_fn,
        )
    except disc.DiscoveryExecutionError as exc:
        raise HoldoutEvaluationError(str(exc)) from exc


def _run_holdout_evaluation_core(repo_root: Path, data: dict, frozen_fits: dict, *, output_root: Path, mode: str,
                                  holdout_ids: list, image_loader, model_and_text_provider, gt_loader, forward_fn) -> dict:
    """The actual per-image processing + gate computation, factored out so
    it is independently testable with a fully synthetic population and
    fake providers, without needing a modified copy of the real committed
    authority. Never called directly by the CLI -- only
    `run_holdout_evaluation` is."""
    checkpoint_dir = _checkpoint_dir_for(output_root)
    discovery_authority = disc._load_authority(repo_root)
    e3_geometry = discovery_authority["scientific_inheritance"]["e3_geometry"]
    crop_hw = tuple(e3_geometry["crop_size_pixels"])
    stride_hw = tuple(e3_geometry["stride_pixels"])
    patch_hw = (e3_geometry["patch_size_pixels"], e3_geometry["patch_size_pixels"])

    if mode == "fresh":
        _require(not output_root.exists(), f"fresh mode requires an absent output_root: {output_root}")
        _require(not checkpoint_dir.exists(), f"fresh mode requires an absent checkpoint dir: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        initial_checkpoint = disc.build_checkpoint_metadata(
            mode=mode, completed_image_ids_ordered=[], discovery_ids=holdout_ids,
            checkpoint_array_hashes={}, identity_bindings={"required_head": data["prerequisite"]["required_head"]},
            complete=(len(holdout_ids) == 0), final=False,
        )
        disc._atomic_write_json(checkpoint_dir / "checkpoint.json", initial_checkpoint)
        completed_ids = []
    elif mode == "resume":
        _require(not output_root.exists(), f"resume mode requires an absent output_root: {output_root}")
        _require((checkpoint_dir / "checkpoint.json").exists(), f"resume mode requires an existing checkpoint.json in {checkpoint_dir}")
        existing_checkpoint = json.loads((checkpoint_dir / "checkpoint.json").read_text())
        disc.validate_checkpoint_metadata(existing_checkpoint, holdout_ids)
        completed_ids = existing_checkpoint["completed_image_ids_ordered"]
    else:  # finalize_only
        _require(not output_root.exists(), f"finalize_only mode requires an absent output_root: {output_root}")
        _require(checkpoint_dir.exists(), f"finalize_only mode requires an existing checkpoint dir: {checkpoint_dir}")
        existing_checkpoint = json.loads((checkpoint_dir / "checkpoint.json").read_text())
        disc.validate_checkpoint_metadata(existing_checkpoint, holdout_ids)
        completed_ids = existing_checkpoint["completed_image_ids_ordered"]
        _require(len(completed_ids) == len(holdout_ids), "finalize_only requires every holdout image already completed")

    event_log = disc.PhaseEventLog()

    if mode != "finalize_only":
        model_and_text = None
        for image_id in holdout_ids:
            if image_id in completed_ids:
                continue
            gt_free_path = checkpoint_dir / f"gt_free_{image_id}.json"
            gt_label_path = checkpoint_dir / f"gt_label_{image_id}.json"

            gt_free_artifact = disc._load_valid_artifact_or_none(gt_free_path, disc.validate_gt_free_artifact, image_id)
            if gt_free_artifact is None:
                if model_and_text is None:
                    model_and_text = model_and_text_provider()
                model, text_embedding = model_and_text
                gt_free_artifact = disc._run_phase_a(image_id, image_loader, forward_fn, model, text_embedding, crop_hw, stride_hw, patch_hw, event_log)
                disc._atomic_write_json(gt_free_path, gt_free_artifact)
            else:
                event_log.record(f"phase_a_start:{image_id}")
                event_log.record(f"phase_a_complete:{image_id}")

            gt_label_artifact = disc._load_valid_artifact_or_none(gt_label_path, disc.validate_gt_label_artifact, image_id)
            if gt_label_artifact is not None:
                try:
                    disc.validate_gt_pair_linkage(gt_free_artifact, gt_label_artifact)
                except disc.DiscoveryExecutionError:
                    gt_label_artifact = None
            if gt_label_artifact is None:
                gt_label_artifact = disc._run_phase_b(gt_free_artifact, gt_loader, image_id, event_log)
                disc._atomic_write_json(gt_label_path, gt_label_artifact)
            disc.validate_gt_pair_linkage(gt_free_artifact, gt_label_artifact)

            completed_ids = completed_ids + [image_id]
            checkpoint_meta = disc.build_checkpoint_metadata(
                mode=mode, completed_image_ids_ordered=completed_ids, discovery_ids=holdout_ids,
                checkpoint_array_hashes={"gt_free": gt_free_artifact["gt_free_feature_digest"], "gt_label": gt_label_artifact["gt_dependent_label_digest"]},
                identity_bindings={"required_head": data["prerequisite"]["required_head"]},
                complete=(len(completed_ids) == len(holdout_ids)), final=False,
            )
            disc._atomic_write_json(checkpoint_dir / "checkpoint.json", checkpoint_meta)

    # --- aggregate rows, apply frozen fits, compute metrics/gates --------
    gt_free_digests, gt_label_digests = [], []
    query_rows_by_image, prop_rows_by_image = {}, {}
    query_eligible_total, prop_eligible_total = 0, 0
    prop_beneficial_total, prop_harmful_total = 0, 0

    for image_id in holdout_ids:
        gt_free = json.loads((checkpoint_dir / f"gt_free_{image_id}.json").read_text())
        gt_label = json.loads((checkpoint_dir / f"gt_label_{image_id}.json").read_text())
        disc.validate_gt_free_artifact(gt_free)
        disc.validate_gt_label_artifact(gt_label)
        gt_free_digests.append(gt_free["gt_free_feature_digest"])
        gt_label_digests.append(gt_label["gt_dependent_label_digest"])

        qc_features = gt_free["control_features"]["query_correctness"]
        qc_reliability = gt_free["intervention_reliability"]["query_correctness"]
        rows = []
        for row_key, label in gt_label["query_correctness_labels"].items():
            detail = qc_features[row_key]
            x_a = list(detail["features"])
            x_b = x_a + [qc_reliability[row_key]]
            score_a = apply_frozen_fit(frozen_fits[("query_correctness", "model_a")], x_a)
            score_b = apply_frozen_fit(frozen_fits[("query_correctness", "model_b")], x_b)
            rows.append((label, score_a, score_b))
        query_rows_by_image[image_id] = rows
        query_eligible_total += len(rows)

        po_features = gt_free["control_features"]["propagation_outcome"]
        po_reliability = gt_free["intervention_reliability"]["propagation_outcome"]
        rows = []
        for row_key, category in gt_label["propagation_outcome_categories"].items():
            if category not in ("beneficial", "harmful"):
                continue
            detail = po_features[row_key]
            x_a = list(detail["features"])
            x_b = x_a + [po_reliability[row_key]]
            score_a = apply_frozen_fit(frozen_fits[("propagation_outcome", "model_a")], x_a)
            score_b = apply_frozen_fit(frozen_fits[("propagation_outcome", "model_b")], x_b)
            label = 1 if category == "beneficial" else 0
            rows.append((label, score_a, score_b))
            if category == "beneficial":
                prop_beneficial_total += 1
            else:
                prop_harmful_total += 1
        prop_rows_by_image[image_id] = rows
        prop_eligible_total += len(rows)

    scientific_collection_digest = hashlib.sha256(("".join(gt_free_digests) + "|" + "".join(gt_label_digests)).encode("ascii")).hexdigest()

    def _flatten(rows_by_image):
        y, sa, sb = [], [], []
        for rows in rows_by_image.values():
            for (yy, a, b) in rows:
                y.append(yy); sa.append(a); sb.append(b)
        return y, sa, sb

    qy, qsa, qsb = _flatten(query_rows_by_image)
    q_classes_present = both_classes_present(qy)
    q_auroc_a = compute_auroc(qy, qsa) if q_classes_present else float("nan")
    q_auroc_b = compute_auroc(qy, qsb) if q_classes_present else float("nan")
    q_ap_a = compute_average_precision(qy, qsa) if q_classes_present else float("nan")
    q_ap_b = compute_average_precision(qy, qsb) if q_classes_present else float("nan")
    q_bootstrap = paired_bootstrap(query_rows_by_image, compute_auroc)
    gate_b = compute_gate_b(eligible_rows=query_eligible_total, classes_present=q_classes_present,
                             auroc_a=q_auroc_a, auroc_b=q_auroc_b, ap_a=q_ap_a, ap_b=q_ap_b, bootstrap=q_bootstrap)

    py, psa, psb = _flatten(prop_rows_by_image)
    p_classes_present = both_classes_present(py)
    p_auroc_a = compute_auroc(py, psa) if p_classes_present else float("nan")
    p_auroc_b = compute_auroc(py, psb) if p_classes_present else float("nan")
    p_ap_a = compute_average_precision(py, psa) if p_classes_present else float("nan")
    p_ap_b = compute_average_precision(py, psb) if p_classes_present else float("nan")
    p_bootstrap = paired_bootstrap(prop_rows_by_image, compute_auroc)
    gate_c = compute_gate_c(
        eligible_rows=prop_eligible_total, beneficial_dominant_rows=prop_beneficial_total, harmful_dominant_rows=prop_harmful_total,
        classes_present=p_classes_present, auroc_a=p_auroc_a, auroc_b=p_auroc_b, ap_a=p_ap_a, ap_b=p_ap_b,
        bootstrap=p_bootstrap, gate_c_execution_authorized=data["gate_c_authorization"]["execution_authorized_by_this_document"],
    )

    # Gate D subgroups: real subgroup assignment is beyond this module's
    # narrow scope (would require re-deriving predicted_boundary_vs_interior/
    # thing_vs_stuff_gt_category/low_vs_high_unary_margin membership from the
    # persisted rows) -- left as an explicit, empty-by-default input the
    # caller may supply; an empty list correctly yields NO_QUALIFYING_SUBGROUPS,
    # never a vacuous PASS.
    gate_d = compute_gate_d([])

    decision_value = holdout_decision(gate_b, gate_c, gate_d)

    holdout_result = {
        "schema": {"name": "holdout_result", "version": "1.0.0"},
        "complete": True, "final": True,
        "identity_and_dependency_bindings": {"required_head": data["prerequisite"]["required_head"]},
        "population_binding": {"holdout_count": len(holdout_ids)},
        "implementation_binding": {"module": "semantic_evidence_holdout_evaluate", "mode": mode},
        "holdout_image_ids_ordered": holdout_ids,
        "frozen_fit_digests": {f"{k[0]}:{k[1]}": v["fit_digest"] for k, v in frozen_fits.items()},
        "gt_free_feature_digest": hashlib.sha256("".join(gt_free_digests).encode("ascii")).hexdigest(),
        "gt_dependent_label_digest": hashlib.sha256("".join(gt_label_digests).encode("ascii")).hexdigest(),
        "scientific_collection_digest": scientific_collection_digest,
        "gate_b_result": gate_b, "gate_c_result": gate_c, "gate_d_result": gate_d,
        "bootstrap_settings": {"replicates": 10000, "seed": 0, "interval_type": "percentile", "confidence_level": 0.95, "min_valid_replicate_fraction": 0.90},
        "final_frozen_outcome": decision_value,
        "holdout_accessed": True, "full_evaluation_accessed": False,
    }
    decision_doc = {
        "schema": {"name": "holdout_decision", "version": "1.0.0"},
        "gate_b_result": gate_b, "gate_c_result": gate_c, "gate_d_result": gate_d,
        "overall_gates_outcome_status": decision_value,
        "holdout_decision": decision_value,
        "next_step": "STOP_OR_REVISE_HYPOTHESIS" if decision_value != "HOLDOUT_GATES_B_C_D_PASSED" else "PROCEED_TO_DEPLOYABLE_SIGNAL_SPECIFICATION",
    }

    finalize_checkpoint = disc.build_checkpoint_metadata(
        mode=mode, completed_image_ids_ordered=holdout_ids, discovery_ids=holdout_ids,
        checkpoint_array_hashes={"gt_free": holdout_result["gt_free_feature_digest"], "gt_label": holdout_result["gt_dependent_label_digest"]},
        identity_bindings={"required_head": data["prerequisite"]["required_head"]},
        complete=True, final=True,
    )
    gt_free_artifacts = [json.loads((checkpoint_dir / f"gt_free_{iid}.json").read_text()) for iid in holdout_ids]
    gt_label_artifacts = [json.loads((checkpoint_dir / f"gt_label_{iid}.json").read_text()) for iid in holdout_ids]

    install_summary = install_holdout_result(output_root, finalize_checkpoint, gt_free_artifacts, gt_label_artifacts, holdout_result, decision_doc, holdout_ids)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return install_summary
