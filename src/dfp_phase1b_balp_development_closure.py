"""DFP Phase 1B: E24-authorized closure-analysis implementation for the
completed E23 real 350-image BALP development execution (SLURM job
22071646).

Read-only interpretation only. Implements no BALP scientific formula
(reuses nothing from ``src.dfp_phase1b_lightweight_boundary`` -- this
module never imports torch or the scientific kernel at all), no
checkpoint format, and no artifact-write policy beyond a small,
independent JSON+Markdown closure document this module defines itself.
Every metric this module reports is INDEPENDENTLY recomputed here, in
pure NumPy, from the already-installed, already-immutable checkpoint
confusion-matrix arrays -- never merely re-read from the source
``result.json`` (which is instead used only as an equality cross-check,
see ``verify_source_result``).

No CUDA/model access anywhere in this module. No image or GT pixel is
ever touched (this module reads only already-aggregated confusion
matrices).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import src.dfp_phase1b_balp_development_closure_identity as e24
import src.dfp_phase1b_lightweight_boundary_artifact as art
import src.dfp_phase1b_lightweight_boundary_identity as e22


class DfpPhase1bBalpDevelopmentClosureError(ValueError):
    """Raised on any closure invariant violation. Always fail closed.
    Never raised for, and never catches, KeyboardInterrupt / SystemExit
    / MemoryError -- those always propagate unmodified."""


CLOSURE_SCHEMA_NAME = "dfp-phase1b-balp-development-closure-v1"
CLOSURE_SCHEMA_VERSION = 1

# The two files whose committed-HEAD bytes back closure_generation's
# hashes. Relative to repo_root; never resolved against anything else.
IMPLEMENTATION_RELATIVE_PATH = Path("src/dfp_phase1b_balp_development_closure.py")
CLI_RELATIVE_PATH = Path("diagnostics/run_dfp_phase1b_balp_development_closure.py")

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

CLASS_COUNT = 171
IGNORE_INDEX = 255

_REQUIRED_PROHIBITED = e24._REQUIRED_PROHIBITED  # re-exported for tests; not re-derived


# ---------------------------------------------------------------------
# Independent metric reconstruction -- fresh NumPy, never calling
# `src.dfp_phase1a_analysis.compute_semantic_metrics`.
# ---------------------------------------------------------------------


def _oracle_metrics_from_confusion(cm: np.ndarray) -> tuple[float, float, float, int]:
    """cm: [C,C] int64, row=GT, col=predicted, evaluated (non-ignore)
    pixels only (see E22 artifacts' own
    `shared.confusion_sufficient_statistics_member.description`).
    Returns (mIoU, mAcc, aAcc) in percentage points, and the total
    evaluated-pixel count."""
    intersection = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    gt = cm.sum(axis=1).astype(np.float64)
    union = predicted + gt - intersection
    valid = union > 0
    iou = np.full(cm.shape[0], np.nan)
    iou[valid] = intersection[valid] / union[valid] * 100.0
    miou = float(np.nanmean(iou))
    valid_acc = gt > 0
    acc = np.full(cm.shape[0], np.nan)
    acc[valid_acc] = intersection[valid_acc] / gt[valid_acc] * 100.0
    macc = float(np.nanmean(acc))
    total = int(cm.sum())
    aacc = float(intersection.sum() / total * 100.0) if total > 0 else float("nan")
    return miou, macc, aacc, total


def independently_reconstruct_variant_metrics(checkpoint: Mapping[str, Any], *, variant_ids: list[str]) -> dict[str, dict[str, Any]]:
    """For every variant, reconstructs (mIoU, mAcc, aAcc, evaluated_pixel_count)
    directly from `checkpoint["aggregate_confusion"][variant]` -- a
    fresh oracle, never the committed `compute_semantic_metrics`."""
    agg = checkpoint["aggregate_confusion"]
    out: dict[str, dict[str, Any]] = {}
    for variant in variant_ids:
        cm = np.asarray(agg[variant], dtype=np.int64)
        if cm.shape != (CLASS_COUNT, CLASS_COUNT):
            raise DfpPhase1bBalpDevelopmentClosureError(f"{variant}: aggregate confusion has unexpected shape {cm.shape}, expected ({CLASS_COUNT},{CLASS_COUNT})")
        if (cm < 0).any():
            raise DfpPhase1bBalpDevelopmentClosureError(f"{variant}: aggregate confusion contains a negative entry")
        miou, macc, aacc, evaluated = _oracle_metrics_from_confusion(cm)
        out[variant] = {"miou": miou, "macc": macc, "aacc": aacc, "evaluated_pixel_count": evaluated}
    return out


def independently_apply_eligibility_rule(control_metrics: Mapping[str, Any], candidate_metrics: Mapping[str, Any]) -> dict[str, bool]:
    """E22's own frozen rule, applied fresh here: a candidate is
    eligible iff its independently-reconstructed development mIoU is
    STRICTLY GREATER than T320's independently-reconstructed
    development mIoU. Equality is NOT eligible."""
    t320 = control_metrics["T320"]["miou"]
    return {cid: (metrics["miou"] > t320) for cid, metrics in candidate_metrics.items()}


# ---------------------------------------------------------------------
# Source verification: public checkpoint/result reverification, plus
# the E22/E23/E24 identity chain, over the SOURCE (read-only) output
# root -- never the destination this closure installs into.
# ---------------------------------------------------------------------


def verify_source_result(source_output_root: Path, *, repo_root: Path) -> dict[str, Any]:
    """Loads and publicly reverifies the E24-pinned source
    checkpoint/result (schema validity + the E24-pinned exact SHA-256
    values), and returns the reopened checkpoint document. Raises if
    ANY pinned hash disagrees with the bytes actually found on disk --
    this is the gate that keeps this module from ever closing over a
    drifted or substituted source artifact."""
    e24_doc = e24.load_identity(repo_root=repo_root)
    variant_ids = list(e22.CANONICAL_CONTROLS) + list(e22.reconstruct_candidate_ids())

    checkpoint = art.load_full_checkpoint(source_output_root, variant_ids=variant_ids)
    if checkpoint is None:
        raise DfpPhase1bBalpDevelopmentClosureError(f"no checkpoint found at source output root {source_output_root}")
    split = e22.load_and_verify_split(repo_root=repo_root)
    art.validate_full_checkpoint(checkpoint, development_ids=split["development_image_ids"], confirmation_ids=split["confirmation_image_ids"])
    if not (checkpoint["complete"] and checkpoint["final"]):
        raise DfpPhase1bBalpDevelopmentClosureError("source checkpoint is not complete+final -- refusing to close over an in-progress run")

    checkpoint_json_path = _installed_checkpoint_json_path(source_output_root, checkpoint)
    checkpoint_arrays_path = checkpoint_json_path.with_name("phase1b_lightweight_boundary_checkpoint_arrays.npz")
    observed_json_sha = hashlib.sha256(checkpoint_json_path.read_bytes()).hexdigest()
    observed_arrays_sha = hashlib.sha256(checkpoint_arrays_path.read_bytes()).hexdigest()
    if observed_json_sha != e24.CHECKPOINT_JSON_SHA256:
        raise DfpPhase1bBalpDevelopmentClosureError(f"checkpoint JSON sha256 drift: expected {e24.CHECKPOINT_JSON_SHA256}, observed {observed_json_sha}")
    if observed_arrays_sha != e24.CHECKPOINT_ARRAYS_SHA256:
        raise DfpPhase1bBalpDevelopmentClosureError(f"checkpoint arrays sha256 drift: expected {e24.CHECKPOINT_ARRAYS_SHA256}, observed {observed_arrays_sha}")

    result_path = source_output_root / "development_result" / "result.json"
    result_doc = json.loads(result_path.read_bytes())
    art.validate_development_result(result_doc, variant_ids=variant_ids)
    observed_result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if observed_result_sha != e24.DEVELOPMENT_RESULT_SHA256:
        raise DfpPhase1bBalpDevelopmentClosureError(f"development result sha256 drift: expected {e24.DEVELOPMENT_RESULT_SHA256}, observed {observed_result_sha}")
    if result_doc["scientific_output_digest"] != e24.SCIENTIFIC_OUTPUT_DIGEST:
        raise DfpPhase1bBalpDevelopmentClosureError("development result scientific_output_digest disagrees with E24's pinned value")

    return checkpoint


def _installed_checkpoint_json_path(output_root: Path, checkpoint: Mapping[str, Any]) -> Path:
    pointer = art.load_current_pointer(output_root)
    if pointer is None:
        raise DfpPhase1bBalpDevelopmentClosureError(f"no current_checkpoint.json pointer at {output_root}")
    gen_dir = output_root / "checkpoint_generations" / pointer["generation_directory"]
    return gen_dir / "phase1b_lightweight_boundary_checkpoint.json"


# ---------------------------------------------------------------------
# Mandatory clean-HEAD generation gate. Real closure generation (the CLI
# `generate` command) must call this FIRST, before verify_source_result
# or build_closure_document, and before any output path is created.
# Fails closed with DfpPhase1bBalpDevelopmentClosureError -- never
# creates or modifies a file. The two hashes it returns are computed
# from bytes read via `git show <head>:<path>` (the committed blob),
# not from re-reading the on-disk working-tree copy -- the on-disk
# bytes are only used for an explicit equality check against that
# committed blob, never as the hash source themselves.
# ---------------------------------------------------------------------


def verify_clean_head_generation_gate(repo_root: Path) -> dict[str, str]:
    E = DfpPhase1bBalpDevelopmentClosureError
    repo_root = Path(repo_root)

    def _run_git(*args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=False)
        except OSError as exc:
            raise E(f"cannot invoke git {' '.join(args)}: {exc}") from exc

    # Checked in this order deliberately: `git status --porcelain` alone
    # already reports staged changes too (its first status column), so
    # checking `--cached` first gives the more specific/actionable
    # "changes are staged" message when that is the actual cause,
    # before falling back to the general "not clean" message for
    # unstaged-only dirtiness.
    staged = _run_git("diff", "--cached", "--name-only")
    if staged.returncode != 0:
        raise E(f"git diff --cached failed: {staged.stderr.decode(errors='replace').strip()}")
    if staged.stdout.strip():
        raise E("closure generation refused: changes are staged")

    status = _run_git("status", "--porcelain")
    if status.returncode != 0:
        raise E(f"git status failed: {status.stderr.decode(errors='replace').strip()}")
    if status.stdout.strip():
        raise E(
            "closure generation refused: repository working tree is not clean -- "
            "commit or discard changes before generating an authoritative closure"
        )

    head_result = _run_git("rev-parse", "HEAD")
    if head_result.returncode != 0:
        raise E(f"cannot resolve HEAD commit: {head_result.stderr.decode(errors='replace').strip()}")
    head = head_result.stdout.decode().strip()
    if not _HEX40.match(head):
        raise E(f"git rev-parse HEAD returned a non-canonical commit id: {head!r}")

    hashes: dict[str, str] = {}
    for label, rel_path in (("implementation", IMPLEMENTATION_RELATIVE_PATH), ("cli", CLI_RELATIVE_PATH)):
        show = _run_git("show", f"{head}:{rel_path.as_posix()}")
        if show.returncode != 0:
            raise E(f"cannot read committed {label} bytes at {head}:{rel_path.as_posix()}: {show.stderr.decode(errors='replace').strip()}")
        committed_bytes = show.stdout
        on_disk_path = repo_root / rel_path
        try:
            on_disk_bytes = on_disk_path.read_bytes()
        except OSError as exc:
            raise E(f"cannot read on-disk {label} file {on_disk_path}: {exc}") from exc
        if on_disk_bytes != committed_bytes:
            raise E(f"closure generation refused: on-disk {label} bytes ({rel_path.as_posix()}) differ from committed HEAD {head}")
        hashes[label] = hashlib.sha256(committed_bytes).hexdigest()

    # E24 itself, and its dependency-hash relational chain, must verify.
    e24.load_identity(repo_root=repo_root)
    # HEAD must descend from (or equal) the required source-run ancestor.
    e24.verify_repository_gate(repo_root, head)

    return {
        "repository_commit": head,
        "implementation_sha256": hashes["implementation"],
        "cli_sha256": hashes["cli"],
    }


# ---------------------------------------------------------------------
# Closure document assembly.
# ---------------------------------------------------------------------

CLOSURE_TOP_KEYS = frozenset({
    "schema_name", "schema_version", "provenance", "population", "controls", "candidates",
    "decision", "profiling_disclosure", "interpretation_boundary", "linear_extrapolation",
})

_LABEL = "LINEAR EXTRAPOLATION -- NOT A MEASURED 5000-IMAGE RUNTIME"


def _elapsed_seconds(hh_mm_ss: str) -> int:
    h, m, s = (int(x) for x in hh_mm_ss.split(":"))
    return h * 3600 + m * 60 + s


def build_closure_document(
    checkpoint: Mapping[str, Any], *, repo_root: Path, source_output_root: Path, closure_generation: Mapping[str, str],
) -> dict[str, Any]:
    """``closure_generation`` is never derived internally by this
    function -- it is always injected by the caller, either the real
    ``verify_clean_head_generation_gate(repo_root)`` result (the only
    authorized path for an authoritative closure) or explicit test
    metadata for a controlled fixture/unit test. This function never
    runs git itself."""
    e24_doc = e24.load_identity(repo_root=repo_root)
    source_output_root = Path(source_output_root)
    variant_ids = list(e22.CANONICAL_CONTROLS) + list(e22.reconstruct_candidate_ids())
    metrics = independently_reconstruct_variant_metrics(checkpoint, variant_ids=variant_ids)

    control_metrics = {v: metrics[v] for v in e22.CANONICAL_CONTROLS}
    candidate_ids = list(e22.reconstruct_candidate_ids())
    candidate_metrics_raw = {v: metrics[v] for v in candidate_ids}
    eligibility = independently_apply_eligibility_rule(control_metrics, candidate_metrics_raw)

    per_image = checkpoint["per_image_reconstruction_statistics"]
    t320_miou = control_metrics["T320"]["miou"]

    controls_out = {
        cid: {
            "control_id": cid, "miou": m["miou"], "macc": m["macc"], "aacc": m["aacc"],
            "evaluated_pixel_count": m["evaluated_pixel_count"],
        }
        for cid, m in control_metrics.items()
    }

    candidates_out = {}
    for cid in candidate_ids:
        s, r = e22.parse_candidate_id(cid)
        m = candidate_metrics_raw[cid]
        changed = int(sum(per_image[cid]["candidate_versus_t320_changed_pixel_count"]))
        preserved = int(sum(per_image[cid]["preservation_mask_pixel_count"]))
        evaluated = m["evaluated_pixel_count"]
        candidates_out[cid] = {
            "candidate_id": cid, "stability_start": s, "radius": r, "miou": m["miou"],
            "gain_vs_t320": m["miou"] - t320_miou,
            "preservation_fraction": (preserved / evaluated) if evaluated else 0.0,
            "changed_pixel_count": changed, "eligible": bool(eligibility[cid]),
        }

    eligible_ids = sorted(cid for cid, ok in eligibility.items() if ok)
    # E22/E23/E24 freeze only the zero-eligible outcome string (the real,
    # observed state of job 22071646): `NO_BALP_CANDIDATE_PROCEEDS`. No
    # committed identity defines an outcome string for a nonzero-eligible
    # closure (that state has never actually occurred), so this module
    # never claims the frozen zero-eligible string for it -- it uses a
    # distinct, clearly-scoped label instead of silently mislabeling a
    # hypothetical/mutated-fixture result as the real frozen outcome.
    outcome = e24.DEVELOPMENT_OUTCOME if not eligible_ids else "BALP_CANDIDATE_ELIGIBLE_OUTCOME_UNDEFINED_BY_E22"
    e23_doc_sha = e24_doc["dependency_bindings"]["e23"]["sha256"]

    checkpoint_json_path = _installed_checkpoint_json_path(source_output_root, checkpoint)
    checkpoint_arrays_path = checkpoint_json_path.with_name("phase1b_lightweight_boundary_checkpoint_arrays.npz")

    # source_run: frozen facts about the completed, immutable SLURM job
    # 22071646 execution -- its repository_commit is E24's pinned
    # REQUIRED_ANCESTOR_COMMIT and never varies across generations.
    #
    # closure_generation: facts about THIS closure-analysis invocation
    # -- its repository_commit is whatever commit closure generation
    # actually ran under (always injected via
    # verify_clean_head_generation_gate for a real, authoritative
    # closure; never internally derived here). These two
    # repository_commit values name two DIFFERENT commits and must
    # never be conflated into one ambiguous field.
    provenance = {
        "source_run": {
            "job_id": e24.SLURM_JOB_ID,
            "repository_commit": e24.REQUIRED_ANCESTOR_COMMIT,
            "result_path": "development_result/result.json",
            "result_sha256": e24.DEVELOPMENT_RESULT_SHA256,
            "checkpoint_path": checkpoint_json_path.relative_to(source_output_root).as_posix(),
            "checkpoint_sha256": e24.CHECKPOINT_JSON_SHA256,
            "checkpoint_arrays_path": checkpoint_arrays_path.relative_to(source_output_root).as_posix(),
            "checkpoint_arrays_sha256": e24.CHECKPOINT_ARRAYS_SHA256,
            "scientific_output_digest": e24.SCIENTIFIC_OUTPUT_DIGEST,
        },
        "closure_generation": {
            "repository_commit": closure_generation["repository_commit"],
            "implementation_sha256": closure_generation["implementation_sha256"],
            "cli_sha256": closure_generation["cli_sha256"],
        },
        "e22_preservation_sha256": e24_doc["dependency_bindings"]["e22_preservation"]["sha256"],
        "e22_artifacts_sha256": e24_doc["dependency_bindings"]["e22_artifacts"]["sha256"],
        "e23_sha256": e23_doc_sha,
        "e24_sha256": hashlib.sha256((repo_root / e24.IDENTITY_RELATIVE_PATH).read_bytes()).hexdigest(),
        "balp_split_sha256": e24_doc["dependency_bindings"]["balp_split_artifact"]["sha256"],
        "development_window_universe_sha256": e24_doc["dependency_bindings"]["development_window_universe"]["sha256"],
    }

    population = {
        "development_images": e24.DEVELOPMENT_IMAGES, "development_windows": e24.DEVELOPMENT_WINDOWS,
        "evaluated_pixels": e24.EVALUATED_PIXELS,
    }

    decision = {
        "eligibility_rule": e24.ELIGIBILITY_RULE, "eligible_candidates": eligible_ids,
        "eligible_candidate_count": len(eligible_ids), "candidate_selected": False,
        "outcome": outcome, "confirmation_must_not_run": True, "full_5000_balp_must_not_run": True,
    }

    profiling_disclosure = {
        "profiling_artifact_present": False, "profiling_status": "NOT_RECORDED",
        "stage_times_available": False, "host_rss_available": False, "gpu_vram_available": False,
        "slurm_elapsed_available": True, "slurm_elapsed": e24_doc["source_execution"]["slurm_elapsed"],
        "bottleneck_identified": False,
    }

    interpretation_boundary = {
        "scientific_conclusion_valid": True, "operational_profiling_available": False,
        "bottleneck_claim_made": False, "confirmation_claim_made": False, "full_validation_claim_made": False,
        "development_only_negative_result": True,
        "notes": (
            "This is a development-only negative result: zero of 16 BALP candidates strictly exceed "
            "the T320 baseline under E22's frozen strict eligibility rule. No confirmation evaluation "
            "and no full 5,000-image evaluation have run or are authorized by this closure. No "
            "operational-profiling or stage-level bottleneck claim is made for job 22071646 -- no "
            "profiling artifact was produced for that run (see profiling_disclosure)."
        ),
    }

    elapsed_seconds = _elapsed_seconds(e24_doc["source_execution"]["slurm_elapsed"])
    projected_seconds = elapsed_seconds / e24.DEVELOPMENT_IMAGES * 5000.0
    linear_extrapolation = {
        "label": _LABEL, "observed_elapsed_seconds": elapsed_seconds, "observed_image_count": e24.DEVELOPMENT_IMAGES,
        "projected_5000_image_seconds": projected_seconds, "projected_5000_image_hours": projected_seconds / 3600.0,
    }

    doc = {
        "schema_name": CLOSURE_SCHEMA_NAME, "schema_version": CLOSURE_SCHEMA_VERSION,
        "provenance": provenance, "population": population, "controls": controls_out, "candidates": candidates_out,
        "decision": decision, "profiling_disclosure": profiling_disclosure,
        "interpretation_boundary": interpretation_boundary, "linear_extrapolation": linear_extrapolation,
    }
    validate_closure_document(doc)
    return doc


_PROVENANCE_KEYS = frozenset({
    "source_run", "closure_generation", "e22_preservation_sha256", "e22_artifacts_sha256",
    "e23_sha256", "e24_sha256", "balp_split_sha256", "development_window_universe_sha256",
})
_SOURCE_RUN_KEYS = frozenset({
    "job_id", "repository_commit", "result_path", "result_sha256",
    "checkpoint_path", "checkpoint_sha256", "checkpoint_arrays_path", "checkpoint_arrays_sha256",
    "scientific_output_digest",
})
_CLOSURE_GENERATION_KEYS = frozenset({"repository_commit", "implementation_sha256", "cli_sha256"})


def _validate_provenance(prov: Any) -> None:
    """Enforces that source_run and closure_generation are distinct,
    closed-schema, unambiguous fields -- never one shared
    `repository_commit`. source_run's repository_commit is a FROZEN
    historical fact (must exactly equal E24's pinned value);
    closure_generation's is whatever commit generation actually ran
    under, and is only format/non-conflation-checked here -- its
    ancestry was already verified by verify_clean_head_generation_gate
    before this document was ever built."""
    E = DfpPhase1bBalpDevelopmentClosureError
    if not isinstance(prov, dict):
        raise E(f"provenance must be a mapping, observed {type(prov).__name__}")
    actual = frozenset(prov.keys())
    if actual != _PROVENANCE_KEYS:
        raise E(f"provenance has the wrong key set -- missing={sorted(_PROVENANCE_KEYS - actual)} unknown={sorted(actual - _PROVENANCE_KEYS)}")

    sr = prov["source_run"]
    if not isinstance(sr, dict) or frozenset(sr.keys()) != _SOURCE_RUN_KEYS:
        raise E(f"provenance.source_run has the wrong key set, observed {sorted(sr.keys()) if isinstance(sr, dict) else type(sr).__name__}")
    if sr["job_id"] != e24.SLURM_JOB_ID:
        raise E(f"provenance.source_run.job_id must be {e24.SLURM_JOB_ID!r}, observed {sr['job_id']!r}")
    if not (isinstance(sr["repository_commit"], str) and _HEX40.match(sr["repository_commit"])):
        raise E(f"provenance.source_run.repository_commit must be 40-hex, observed {sr['repository_commit']!r}")
    if sr["repository_commit"] != e24.REQUIRED_ANCESTOR_COMMIT:
        raise E(
            f"provenance.source_run.repository_commit must be E24's frozen source-run commit "
            f"{e24.REQUIRED_ANCESTOR_COMMIT!r}, observed {sr['repository_commit']!r}"
        )
    if sr["result_sha256"] != e24.DEVELOPMENT_RESULT_SHA256:
        raise E("provenance.source_run.result_sha256 disagrees with E24's frozen value")
    if sr["checkpoint_sha256"] != e24.CHECKPOINT_JSON_SHA256:
        raise E("provenance.source_run.checkpoint_sha256 disagrees with E24's frozen value")
    if sr["checkpoint_arrays_sha256"] != e24.CHECKPOINT_ARRAYS_SHA256:
        raise E("provenance.source_run.checkpoint_arrays_sha256 disagrees with E24's frozen value")
    if sr["scientific_output_digest"] != e24.SCIENTIFIC_OUTPUT_DIGEST:
        raise E("provenance.source_run.scientific_output_digest disagrees with E24's frozen value")
    if not isinstance(sr["result_path"], str) or not sr["result_path"]:
        raise E("provenance.source_run.result_path must be a non-empty string")
    if not isinstance(sr["checkpoint_path"], str) or not sr["checkpoint_path"]:
        raise E("provenance.source_run.checkpoint_path must be a non-empty string")
    if not isinstance(sr["checkpoint_arrays_path"], str) or not sr["checkpoint_arrays_path"]:
        raise E("provenance.source_run.checkpoint_arrays_path must be a non-empty string")

    cg = prov["closure_generation"]
    if not isinstance(cg, dict) or frozenset(cg.keys()) != _CLOSURE_GENERATION_KEYS:
        raise E(f"provenance.closure_generation has the wrong key set, observed {sorted(cg.keys()) if isinstance(cg, dict) else type(cg).__name__}")
    if not (isinstance(cg["repository_commit"], str) and _HEX40.match(cg["repository_commit"])):
        raise E(f"provenance.closure_generation.repository_commit must be 40-hex, observed {cg['repository_commit']!r}")
    if not (isinstance(cg["implementation_sha256"], str) and _HEX64.match(cg["implementation_sha256"])):
        raise E("provenance.closure_generation.implementation_sha256 must be 64-hex")
    if not (isinstance(cg["cli_sha256"], str) and _HEX64.match(cg["cli_sha256"])):
        raise E("provenance.closure_generation.cli_sha256 must be 64-hex")
    if cg["repository_commit"] == sr["repository_commit"]:
        raise E(
            "provenance.closure_generation.repository_commit must not conflate with "
            "provenance.source_run.repository_commit -- the closure implementation cannot "
            "exist at the source-run commit, so an equal value always indicates a swapped "
            "or conflated field, never a legitimate coincidence"
        )

    for key in ("e22_preservation_sha256", "e22_artifacts_sha256", "e23_sha256", "e24_sha256", "balp_split_sha256", "development_window_universe_sha256"):
        if not (isinstance(prov[key], str) and _HEX64.match(prov[key])):
            raise E(f"provenance.{key} must be 64-hex, observed {prov[key]!r}")


def validate_closure_document(doc: Mapping[str, Any]) -> None:
    E = DfpPhase1bBalpDevelopmentClosureError
    if not isinstance(doc, dict):
        raise E(f"closure document must be a mapping, observed {type(doc).__name__}")
    actual = frozenset(doc.keys())
    if actual != CLOSURE_TOP_KEYS:
        raise E(f"closure document has the wrong key set -- missing={sorted(CLOSURE_TOP_KEYS - actual)} unknown={sorted(actual - CLOSURE_TOP_KEYS)}")
    if doc["schema_name"] != CLOSURE_SCHEMA_NAME or doc["schema_version"] != CLOSURE_SCHEMA_VERSION:
        raise E("closure schema_name/schema_version mismatch")

    _validate_provenance(doc["provenance"])

    pop = doc["population"]
    if pop["development_images"] != e24.DEVELOPMENT_IMAGES or pop["development_windows"] != e24.DEVELOPMENT_WINDOWS or pop["evaluated_pixels"] != e24.EVALUATED_PIXELS:
        raise E("closure population disagrees with E24's frozen population")

    controls = doc["controls"]
    if set(controls.keys()) != set(e22.CANONICAL_CONTROLS):
        raise E("closure controls key set disagrees with the frozen 5-control family")
    for cid, block in controls.items():
        if block["control_id"] != cid:
            raise E(f"controls[{cid}].control_id disagrees with its own key")

    candidates = doc["candidates"]
    expected_candidate_ids = set(e22.reconstruct_candidate_ids())
    if set(candidates.keys()) != expected_candidate_ids:
        raise E(f"closure candidates key set disagrees with the frozen 16-candidate family -- missing={sorted(expected_candidate_ids - set(candidates))} unknown={sorted(set(candidates) - expected_candidate_ids)}")
    for cid, block in candidates.items():
        if block["candidate_id"] != cid:
            raise E(f"candidates[{cid}].candidate_id disagrees with its own key")
        s, r = e22.parse_candidate_id(cid)
        if block["stability_start"] != s or block["radius"] != r:
            raise E(f"candidates[{cid}] stability_start/radius disagree with its own id")
        if not isinstance(block["eligible"], bool):
            raise E(f"candidates[{cid}].eligible must be a boolean")

    decision = doc["decision"]
    if decision["eligibility_rule"] != e24.ELIGIBILITY_RULE:
        raise E("decision.eligibility_rule disagrees with E24's frozen rule")
    recomputed_eligible = sorted(cid for cid, block in candidates.items() if block["eligible"])
    if decision["eligible_candidates"] != recomputed_eligible:
        raise E("decision.eligible_candidates disagrees with the per-candidate eligible flags")
    if decision["eligible_candidate_count"] != len(recomputed_eligible):
        raise E("decision.eligible_candidate_count disagrees with len(eligible_candidates)")
    if decision["candidate_selected"] is not False:
        raise E("decision.candidate_selected must be false -- this closure selects no candidate")
    if decision["confirmation_must_not_run"] is not True or decision["full_5000_balp_must_not_run"] is not True:
        raise E("decision.confirmation_must_not_run / full_5000_balp_must_not_run must both be true")
    if decision["eligible_candidate_count"] == 0 and decision["outcome"] != e24.DEVELOPMENT_OUTCOME:
        raise E(f"decision.outcome must be {e24.DEVELOPMENT_OUTCOME!r} when eligible_candidate_count is 0")
    if decision["eligible_candidate_count"] > 0 and decision["outcome"] == e24.DEVELOPMENT_OUTCOME:
        raise E("decision.outcome disagrees with a nonzero eligible_candidate_count -- E22 authorizes no such combination in this closure")

    prof = doc["profiling_disclosure"]
    if prof["profiling_artifact_present"] is not False or prof["profiling_status"] != "NOT_RECORDED":
        raise E("profiling_disclosure must record the historical NOT_RECORDED state for job 22071646 -- fabrication is forbidden")
    if prof["stage_times_available"] is not False or prof["host_rss_available"] is not False or prof["gpu_vram_available"] is not False:
        raise E("profiling_disclosure must not claim any unavailable telemetry as available")
    if prof["bottleneck_identified"] is not False:
        raise E("profiling_disclosure.bottleneck_identified must be false -- no bottleneck claim is authorized")

    ib = doc["interpretation_boundary"]
    if ib["bottleneck_claim_made"] is not False or ib["confirmation_claim_made"] is not False or ib["full_validation_claim_made"] is not False:
        raise E("interpretation_boundary must not claim confirmation, full-validation, or a bottleneck")

    lx = doc["linear_extrapolation"]
    if lx["label"] != _LABEL:
        raise E("linear_extrapolation.label disagrees with the frozen required disclosure text")
    recomputed_projection = lx["observed_elapsed_seconds"] / lx["observed_image_count"] * 5000.0
    if abs(lx["projected_5000_image_seconds"] - recomputed_projection) > 1e-6:
        raise E("linear_extrapolation.projected_5000_image_seconds disagrees with a fresh recomputation")
    if abs(lx["projected_5000_image_hours"] - recomputed_projection / 3600.0) > 1e-6:
        raise E("linear_extrapolation.projected_5000_image_hours disagrees with a fresh recomputation")


# ---------------------------------------------------------------------
# Markdown report (Section 14).
# ---------------------------------------------------------------------


def build_closure_markdown(doc: Mapping[str, Any]) -> str:
    ctrl_rows = "\n".join(
        f"| {cid} | {doc['controls'][cid]['miou']:.6f} | {doc['controls'][cid]['macc']:.6f} | {doc['controls'][cid]['aacc']:.6f} | {doc['controls'][cid]['evaluated_pixel_count']} |"
        for cid in e22.CANONICAL_CONTROLS
    )
    cand_rows = "\n".join(
        f"| {cid} | {doc['candidates'][cid]['stability_start']} | {doc['candidates'][cid]['radius']} | "
        f"{doc['candidates'][cid]['miou']:.6f} | {doc['candidates'][cid]['gain_vs_t320']:+.6f} | "
        f"{doc['candidates'][cid]['preservation_fraction']:.6f} | {doc['candidates'][cid]['changed_pixel_count']} | "
        f"{'YES' if doc['candidates'][cid]['eligible'] else 'no'} |"
        for cid in sorted(doc["candidates"].keys())
    )
    best_id = max(doc["candidates"], key=lambda c: doc["candidates"][c]["gain_vs_t320"])
    best = doc["candidates"][best_id]
    lx = doc["linear_extrapolation"]

    sr = doc["provenance"]["source_run"]
    cg = doc["provenance"]["closure_generation"]

    return f"""# DFP Phase 1B BALP Development Closure -- SLURM job {sr['job_id']}

BALP DEVELOPMENT OUTCOME: {doc['decision']['outcome']}
ELIGIBLE CANDIDATES: {doc['decision']['eligible_candidate_count']}/16
CONFIRMATION RUN AUTHORIZED: NO
FULL 5000-IMAGE BALP RUN AUTHORIZED: NO
PROFILING ARTIFACT: NOT RECORDED
STAGE-LEVEL BOTTLENECK CLAIM: NOT AVAILABLE

## Scope

Development-only evaluation over the frozen 350-image/769-window population
({doc['population']['evaluated_pixels']} evaluated pixels). No confirmation-split
image and no other validation image was ever accessed. This is a negative
development result, not a final judgment on BALP as a method.

## Eligibility rule

`{doc['decision']['eligibility_rule']}` (E22, frozen): a candidate is eligible
only if its development mIoU is STRICTLY greater than the T320 baseline's
development mIoU. Equality is not eligible. Applied here by independent
recomputation from the persisted confusion-matrix arrays, not by re-reading
the source result's own reported metrics.

## Controls

| control | mIoU (pp) | mAcc (pp) | aAcc (pp) | evaluated px |
|---|---:|---:|---:|---:|
{ctrl_rows}

## Candidates (all 16)

| candidate | start | radius | mIoU (pp) | gain vs T320 (pp) | preservation frac | changed px | eligible |
|---|---:|---:|---:|---:|---:|---:|:---:|
{cand_rows}

Best candidate by gain: **{best_id}** ({best['gain_vs_t320']:+.4f} pp vs T320, eligible={best['eligible']}).
No candidate is recommended or selected: this closure recommends no winning BALP candidate.

## Scientific digest

`{sr['scientific_output_digest']}`

## SLURM accounting

job_id={sr['job_id']}  elapsed={doc['profiling_disclosure']['slurm_elapsed']}

## Provenance -- two separate commits

This closure distinguishes the commit SLURM job {sr['job_id']} actually ran
under from the commit closure generation itself ran under. These are never
the same commit (the closure implementation did not exist at the source-run
commit) and must never be conflated into one field.

| | repository_commit | notes |
|---|---|---|
| source_run | `{sr['repository_commit']}` | commit job {sr['job_id']} executed under (frozen, E24-pinned) |
| closure_generation | `{cg['repository_commit']}` | commit this closure document was generated under |

source_run: result_path=`{sr['result_path']}` result_sha256=`{sr['result_sha256']}`
checkpoint_path=`{sr['checkpoint_path']}` checkpoint_sha256=`{sr['checkpoint_sha256']}`
checkpoint_arrays_path=`{sr['checkpoint_arrays_path']}` checkpoint_arrays_sha256=`{sr['checkpoint_arrays_sha256']}`

closure_generation: implementation_sha256=`{cg['implementation_sha256']}` cli_sha256=`{cg['cli_sha256']}`

## Profiling limitation

{doc['interpretation_boundary']['notes']}

## {lx['label']}

observed: {doc['profiling_disclosure']['slurm_elapsed']} for {lx['observed_image_count']} images
projected 5000-image time under identical average throughput: approximately {lx['projected_5000_image_hours']:.1f} hours
This is NOT a measured 5000-image runtime, NOT profiling, and NOT a bottleneck measurement --
a plain linear extrapolation of total wall time only.

## Scope limitations

- No confirmation evaluation has run or is authorized.
- No full 5,000-image evaluation has run or is authorized.
- No operational/stage-level profiling is available for this run.
- A future T64/early-exit idea may be listed only as a separately evaluated
  future direction, never as a conclusion of this experiment (none is claimed here).

NEXT OPTIONAL EXPERIMENT:
freeze and run a small representative BALP profiling benchmark
to measure model/graph/recurrence/stitching time shares.
"""


# ---------------------------------------------------------------------
# Atomic install (JSON + Markdown together, one staging rename) and
# read-only reload. No overwrite path -- an existing destination is
# always preserved untouched.
# ---------------------------------------------------------------------


def install_closure(destination_root: Path, *, json_doc: Mapping[str, Any], markdown_text: str) -> Path:
    from src.dfp_phase1a_artifact_schema import DfpPhase1aArtifactSchemaError, write_json_atomically

    destination_root = Path(destination_root)
    if destination_root.exists():
        raise DfpPhase1bBalpDevelopmentClosureError(f"closure destination {destination_root} already exists -- no overwrite path is authorized; use a fresh destination")
    validate_closure_document(json_doc)

    staging = destination_root.parent / f".staging-closure-{secrets.token_hex(16)}"
    staging.mkdir(parents=True)
    try:
        json_path = staging / "closure.json"
        try:
            write_json_atomically(json_path, json_doc, label="BALP development closure")
        except DfpPhase1aArtifactSchemaError as exc:
            raise DfpPhase1bBalpDevelopmentClosureError(f"cannot stage closure JSON: {exc}") from exc
        md_path = staging / "closure.md"
        md_path.write_text(markdown_text, encoding="utf-8")

        staged_doc = json.loads(json_path.read_bytes())
        validate_closure_document(staged_doc)
        os.replace(staging, destination_root)
    except (OSError, DfpPhase1bBalpDevelopmentClosureError):
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    installed_json = json.loads((destination_root / "closure.json").read_bytes())
    validate_closure_document(installed_json)
    if not (destination_root / "closure.md").is_file():
        raise DfpPhase1bBalpDevelopmentClosureError("installed closure.md is missing")
    return destination_root


def load_closure(destination_root: Path) -> dict[str, Any] | None:
    json_path = Path(destination_root) / "closure.json"
    if not json_path.is_file():
        return None
    doc = json.loads(json_path.read_bytes())
    validate_closure_document(doc)
    return doc
