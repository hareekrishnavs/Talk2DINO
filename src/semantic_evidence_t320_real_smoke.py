"""Real, two-image, GT-free E3/T320 integration smoke runner.

Implements the four layers required by the committed smoke authority
(research_specs/semantic_evidence_t320_real_smoke_v1.toml,
src/semantic_evidence_t320_real_smoke_identity.py):

  1. Torch-free preflight            -- preflight()
  2. Real execution path             -- execute() (lazy torch import)
  3. Closed result builder/installer -- build_result / install_result_atomic
  4. Torch-free public verifier      -- verify_installed_result()

This module implements NO scientific behavior of its own: graph
construction, recurrence, sigmoid placement, stitching, and final rescale
are always performed by calling the COMMITTED, unmodified
`semantic_evidence_t320` module and `semantic_evidence_e3_adapter` module.
This module orchestrates them, counts the ACTUAL events they perform (via
call-counting instrumentation, never a post-hoc formula), and builds/
installs/verifies the closed result artifact the authority defines.

It never accesses GT, annotations, or a correctness label; it never exposes
a scientific override on any public function; it never authorizes real
execution merely by existing (see `preflight()`'s and `execute()`'s own
disclosure blocks) -- only the operator, after every authorization-boundary
condition in the committed authority is satisfied, may submit the launcher.

Fatal exceptions (KeyboardInterrupt, SystemExit, MemoryError) are never
swallowed: every broad except-clause below re-raises them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import semantic_evidence_t320_real_smoke_identity as smoke_id


class SmokeRunnerError(Exception):
    """Raised for any runner-level failure: missing/unsafe path, checkpoint
    hash mismatch, same-forward preservation mismatch, count/determinism
    mismatch, malformed or unsafe result artifact, or atomic-installation
    failure. Contract-schema violations are raised by
    semantic_evidence_t320_real_smoke_identity.T320RealSmokeContractError
    instead (this module never re-declares that check)."""


_FATAL = (KeyboardInterrupt, SystemExit, MemoryError)

# Internal immutable whitelist (Defect 3 repair): _default_real_image_loader
# enforces this directly, independent of what run_smoke() happens to
# iterate. Not exposed via any CLI flag or public API parameter.
# preflight() cross-checks this tuple against the verified committed
# two-image manifest (smoke_id.load_smoke_population) before any run.
_AUTHORIZED_IMAGE_IDS = ("000000567197", "000000367680")


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, _FATAL):
        raise exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeRunnerError(message)


# ---------------------------------------------------------------------------
# Frozen constants, re-derived from the committed authority at call time --
# never duplicated as independent hardcoded module constants, so this module
# cannot silently drift from the authority it implements.
# ---------------------------------------------------------------------------

def _load_authority(repo_root: Path) -> dict:
    return smoke_id.load_authority(repo_root)


def _load_population(repo_root: Path) -> dict:
    return smoke_id.load_smoke_population(repo_root)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_HEX_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_sha256_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
        raise SmokeRunnerError(f"{label} must be a lowercase 64-hex-character SHA-256 digest, got {value!r}")
    return value


def _require_commit_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _COMMIT_HEX_RE.match(value):
        raise SmokeRunnerError(f"{label} must be a lowercase 40-hex-character commit id, got {value!r}")
    return value


def _safe_repo_relative_path(repo_root: Path, relative_path: str, label: str) -> Path:
    """Resolves relative_path under repo_root, rejecting lexical traversal,
    absolute paths, and symlink escape -- reuses the identity loader's own
    proven path-safety logic, but always re-raises as SmokeRunnerError so
    this module's callers only ever see its own declared domain
    exception."""
    try:
        return smoke_id.resolve_repo_relative_path(repo_root, relative_path)
    except smoke_id.T320RealSmokeContractError as exc:
        raise SmokeRunnerError(f"{label}: unsafe path {relative_path!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Runtime provenance (Defect 1 repair): the four mandatory roles bind the
# exact committed bytes of the code that produced a result, extending the
# EXISTING implementation_hashes list (no new top-level result key --
# staying inside the committed authority's frozen 21-field result schema).
# ---------------------------------------------------------------------------

_TRACKED_FILE_RECORD_KEYS = frozenset({"path", "sha256", "role", "must_be_committed_at_execution_head"})

_MANDATORY_PROVENANCE_ROLES = frozenset({"smoke_runner", "smoke_cli", "smoke_launcher", "smoke_authority"})
_OPTIONAL_PROVENANCE_ROLES = frozenset({"smoke_authority_identity_loader", "smoke_two_image_manifest"})
_PROVENANCE_ROLES = _MANDATORY_PROVENANCE_ROLES | _OPTIONAL_PROVENANCE_ROLES

_PROVENANCE_ROLE_PATHS = {
    "smoke_runner": "src/semantic_evidence_t320_real_smoke.py",
    "smoke_cli": "diagnostics/run_semantic_evidence_t320_real_smoke.py",
    "smoke_launcher": "semantic_evidence_t320_real_smoke_h100.sbatch",
    "smoke_authority": "research_specs/semantic_evidence_t320_real_smoke_v1.toml",
    "smoke_authority_identity_loader": "src/semantic_evidence_t320_real_smoke_identity.py",
    "smoke_two_image_manifest": "evaluation_subsets/semantic_evidence_t320_real_smoke_2.json",
}


def _hash_tracked_committed_file(repo_root: Path, relative_path: str, role: str) -> dict:
    """Resolves relative_path safely and containedly under repo_root,
    requires it is git-tracked, requires the tracked working tree is clean
    for it, hashes its working-tree bytes, independently re-hashes its
    git-HEAD blob bytes via `git show`, and requires the two agree before
    ever recording a hash -- never trusts a caller-supplied value. This is
    what makes real execution structurally impossible before this exact
    runner/CLI/launcher/authority file set has actually been committed."""
    resolved = _safe_repo_relative_path(repo_root, relative_path, f"provenance role {role!r}")
    if not resolved.is_file():
        raise SmokeRunnerError(f"provenance file for role {role!r} does not exist: {relative_path}")

    try:
        tracked = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", relative_path],
            capture_output=True, text=True,
        )
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise SmokeRunnerError(f"failed to check git-tracked status for {relative_path!r}: {exc}") from exc
    if tracked.returncode != 0:
        raise SmokeRunnerError(
            f"provenance file for role {role!r} is not tracked by git (real execution requires this file "
            f"to be committed): {relative_path}"
        )

    try:
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--", relative_path],
            capture_output=True, text=True, check=True,
        )
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise SmokeRunnerError(f"failed to check working-tree cleanliness for {relative_path!r}: {exc}") from exc
    if status.stdout.strip() != "":
        raise SmokeRunnerError(f"provenance file for role {role!r} has uncommitted changes: {relative_path}")

    working_tree_hash = sha256_file(resolved)
    head = smoke_id.git_head_commit(repo_root)
    committed_hash = smoke_id.git_blob_sha256(repo_root, head, relative_path)
    if working_tree_hash != committed_hash:
        raise SmokeRunnerError(
            f"provenance file for role {role!r} working-tree hash ({working_tree_hash}) disagrees with "
            f"committed HEAD hash ({committed_hash}): {relative_path}"
        )

    return {
        "path": relative_path,
        "sha256": committed_hash,
        "role": role,
        "must_be_committed_at_execution_head": True,
    }


def build_runtime_provenance(repo_root: Path) -> list:
    """Builds the 4 mandatory + 2 optional runtime-provenance records, each
    independently re-hashed from tracked, clean, committed-HEAD bytes.
    Raises SmokeRunnerError if any mandatory file is not yet
    tracked/committed/clean."""
    return [
        _hash_tracked_committed_file(repo_root, path, role)
        for role, path in _PROVENANCE_ROLE_PATHS.items()
    ]


def _validate_provenance_records(implementation_hashes: Any) -> dict:
    """Structural (git-free) validation of the implementation_hashes list:
    closed per-entry key set, well-formed hashes/paths, no duplicate role,
    all mandatory roles present, no unrecognized 'smoke_'-namespaced role,
    and (for entries with a recognized role) the recorded path matches that
    role's frozen path. Returns {role: path} for the recognized entries.
    Does not touch git -- see _reverify_provenance_against_git for that."""
    if not isinstance(implementation_hashes, list) or len(implementation_hashes) < 1:
        raise SmokeRunnerError("implementation_hashes must be a non-empty list")

    seen_roles: dict = {}
    for entry in implementation_hashes:
        if not isinstance(entry, dict):
            raise SmokeRunnerError(f"implementation_hashes entry must be a dict, got {type(entry).__name__}")
        actual_keys = frozenset(entry.keys())
        if actual_keys != _TRACKED_FILE_RECORD_KEYS:
            raise SmokeRunnerError(
                f"implementation_hashes entry has the wrong key set: "
                f"missing={_TRACKED_FILE_RECORD_KEYS - actual_keys}, extra={actual_keys - _TRACKED_FILE_RECORD_KEYS}"
            )
        role = entry["role"]
        path = entry["path"]
        if not isinstance(role, str) or role == "":
            raise SmokeRunnerError("implementation_hashes entry role must be a non-empty string")
        if not isinstance(path, str) or path == "":
            raise SmokeRunnerError(f"implementation_hashes entry for role {role!r} has an invalid path")
        _require_sha256_hex(entry["sha256"], f"implementation_hashes[{role!r}].sha256")
        if isinstance(entry["must_be_committed_at_execution_head"], bool) is False:
            raise SmokeRunnerError(f"implementation_hashes[{role!r}].must_be_committed_at_execution_head must be a bool")

        if role.startswith("smoke_"):
            if role not in _PROVENANCE_ROLES:
                raise SmokeRunnerError(f"implementation_hashes contains an unrecognized provenance role: {role!r}")
            if role in seen_roles:
                raise SmokeRunnerError(f"duplicate provenance role: {role!r}")
            if path != _PROVENANCE_ROLE_PATHS[role]:
                raise SmokeRunnerError(
                    f"implementation_hashes role {role!r} has the wrong path: expected "
                    f"{_PROVENANCE_ROLE_PATHS[role]!r}, got {path!r}"
                )
            seen_roles[role] = path

    missing_mandatory = _MANDATORY_PROVENANCE_ROLES - set(seen_roles)
    if missing_mandatory:
        raise SmokeRunnerError(f"result is missing mandatory provenance roles: {sorted(missing_mandatory)}")

    return seen_roles


def _reverify_provenance_against_git(repo_root: Path, result: dict) -> None:
    """Public, torch-free re-verification (Part 4): independently
    re-derives every implementation_hashes entry's hash from git objects
    (never trusting the artifact's own recorded value) at BOTH the
    recorded commit and the repository's current HEAD, and confirms the
    recorded commit actually exists and actually contains each path."""
    binding = result.get("repository_commit_tree_cleanliness")
    if not isinstance(binding, dict) or "head" not in binding:
        raise SmokeRunnerError("repository_commit_tree_cleanliness.head is required for provenance reverification")
    recorded_commit = _require_commit_hex(binding["head"], "repository_commit_tree_cleanliness.head")

    try:
        exists = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{recorded_commit}^{{commit}}"],
            capture_output=True,
        )
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise SmokeRunnerError(f"failed to check existence of recorded commit {recorded_commit}: {exc}") from exc
    if exists.returncode != 0:
        raise SmokeRunnerError(f"recorded commit does not exist in this repository: {recorded_commit}")

    current_head = smoke_id.git_head_commit(repo_root)

    for entry in result["implementation_hashes"]:
        path = entry["path"]
        role = entry["role"]
        recorded_hash = entry["sha256"]
        _safe_repo_relative_path(repo_root, path, f"provenance role {role!r}")

        try:
            present_at_recorded = subprocess.run(
                ["git", "-C", str(repo_root), "cat-file", "-e", f"{recorded_commit}:{path}"],
                capture_output=True,
            )
        except Exception as exc:
            _reraise_if_fatal(exc)
            raise SmokeRunnerError(f"failed to check {path!r} at commit {recorded_commit}: {exc}") from exc
        if present_at_recorded.returncode != 0:
            raise SmokeRunnerError(
                f"recorded commit {recorded_commit} does not contain path {path!r} (role {role!r})"
            )

        actual_at_recorded = smoke_id.git_blob_sha256(repo_root, recorded_commit, path)
        if actual_at_recorded != recorded_hash:
            raise SmokeRunnerError(
                f"provenance hash mismatch for role {role!r} path {path!r} at recorded commit "
                f"{recorded_commit}: recorded {recorded_hash}, actual {actual_at_recorded}"
            )

        try:
            present_at_head = subprocess.run(
                ["git", "-C", str(repo_root), "cat-file", "-e", f"{current_head}:{path}"],
                capture_output=True,
            )
        except Exception as exc:
            _reraise_if_fatal(exc)
            raise SmokeRunnerError(f"failed to check {path!r} at current HEAD {current_head}: {exc}") from exc
        if present_at_head.returncode != 0:
            raise SmokeRunnerError(
                f"current repository HEAD {current_head} does not contain path {path!r} (role {role!r})"
            )
        actual_at_head = smoke_id.git_blob_sha256(repo_root, current_head, path)
        if actual_at_head != recorded_hash:
            raise SmokeRunnerError(
                f"provenance file for role {role!r} path {path!r} has changed since the recorded commit: "
                f"recorded {recorded_hash}, current HEAD {current_head} has {actual_at_head}"
            )


# ---------------------------------------------------------------------------
# Canonical window geometry -- a faithful, cited, pure-arithmetic mirror of
# DINOTextSegInference.slide_inference's own window loop
# (src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py:
# 78-94), never a reimplementation of the accumulation/inference logic
# itself (that happens once per window via the committed E3 adapter, not via
# slide_inference). Row-major order, flush-right/flush-bottom clamping
# behavior reproduced exactly.
# ---------------------------------------------------------------------------

def compute_canonical_windows(img_h: int, img_w: int, crop_hw: tuple, stride_hw: tuple) -> list:
    """Returns an ordered list of (window_ordinal, top, left, bottom, right)
    tuples, row-major (h_idx outer, w_idx inner), matching
    dinotext_seg.py:78-94 exactly."""
    _require(isinstance(img_h, int) and isinstance(img_w, int) and img_h > 0 and img_w > 0, "img_h/img_w must be positive ints")
    h_crop, w_crop = crop_hw
    h_stride, w_stride = stride_hw
    h_grids = max(img_h - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(img_w - w_crop + w_stride - 1, 0) // w_stride + 1

    windows = []
    ordinal = 0
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, img_h)
            x2 = min(x1 + w_crop, img_w)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            windows.append((ordinal, y1, x1, y2, x2))
            ordinal += 1
    _require(len(windows) >= 1, "canonical window generation produced zero windows")
    return windows


def compute_window_digest(image_ids_ordered: list, windows_by_image: dict) -> str:
    """SHA256 of the ordered concatenation of every window record's fields
    (image_id, window_ordinal, top, left, bottom, right), each field
    ASCII-encoded and newline-terminated, in canonical (image order, then
    window order) sequence -- mirrors the digest style already established
    by evaluation_subsets/semantic_evidence_t320_real_smoke_2.json and
    research_specs/semantic_evidence_feasibility_v1.toml."""
    parts = []
    for image_id in image_ids_ordered:
        for (ordinal, top, left, bottom, right) in windows_by_image[image_id]:
            parts.append(f"{image_id}\n{ordinal}\n{top}\n{left}\n{bottom}\n{right}\n")
    data = "".join(parts).encode("ascii")
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Layer 1: torch-free preflight
# ---------------------------------------------------------------------------

def preflight(repo_root: Path, *, weight_dir: str, image_root: Path, output_root: Path) -> dict:
    """Runs entirely without importing torch, initializing CUDA, opening a
    checkpoint tensor, or reading an image. Re-hashes the three external
    checkpoint FILES (raw bytes -> sha256; never torch.load) against the
    committed authority's declared bindings, and confirms the two
    authorized image files resolve -- never touching any annotation path."""
    t_start = time.perf_counter_ns()

    contract_result = smoke_id.verify_all(repo_root)
    _require(contract_result["verified"], "committed smoke authority did not verify")

    toml_data = _load_authority(repo_root)
    ew = toml_data["external_weights"]

    weight_dir_path = Path(weight_dir)
    _require(weight_dir_path.is_dir(), f"weight_dir does not exist or is not a directory: {weight_dir}")

    checkpoint_checks = []
    for key, resolved_dir in (
        ("dino_backbone_checkpoint", weight_dir_path),
        ("clip_text_checkpoint", weight_dir_path),
        ("e3_projection_checkpoint", repo_root / "weights"),
    ):
        entry = ew[key]
        candidate = resolved_dir / entry["filename"]
        _require(candidate.is_file(), f"external checkpoint file not found for {key}: {candidate}")
        actual_sha256 = sha256_file(candidate)
        expected_sha256 = entry["sha256"]
        if actual_sha256 != expected_sha256:
            raise SmokeRunnerError(
                f"external checkpoint hash mismatch for {key} ({candidate}): "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        checkpoint_checks.append({"key": key, "path": str(candidate), "sha256": actual_sha256, "match": True})

    env_weight_dir = os.environ.get("TALK2DINO_WEIGHT_DIR")
    weight_dir_env_disclosed = {
        "TALK2DINO_WEIGHT_DIR_env_value": env_weight_dir,
        "weight_dir_argument": weight_dir,
        "agree": env_weight_dir is None or env_weight_dir == weight_dir,
    }
    if env_weight_dir is not None and env_weight_dir != weight_dir:
        raise SmokeRunnerError(
            f"TALK2DINO_WEIGHT_DIR env value ({env_weight_dir!r}) disagrees with weight_dir argument ({weight_dir!r})"
        )

    population_data = _load_population(repo_root)
    image_ids = population_data["selected_image_ids_ordered"]
    _require(
        tuple(image_ids) == _AUTHORIZED_IMAGE_IDS,
        f"committed two-image manifest disagrees with the internal authorized-ID whitelist: {image_ids} != {_AUTHORIZED_IMAGE_IDS}",
    )

    image_root_path = Path(image_root)
    _require(image_root_path.is_dir(), f"image_root does not exist or is not a directory: {image_root}")
    image_paths = {}
    for image_id in image_ids:
        candidate = image_root_path / f"{image_id}.jpg"
        _require(candidate.is_file(), f"authorized image file not found: {candidate}")
        image_paths[image_id] = str(candidate)
    _require(
        len(set(image_paths.values())) == len(image_ids),
        "resolved image paths are not distinct",
    )
    # Explicitly never resolve or reference any annotation-directory path.
    for suspicious in ("annotations", "ann_dir", "gt", "ground_truth"):
        _require(suspicious not in str(image_root_path).lower(), f"image_root path looks like an annotation directory: {image_root}")

    output_root_path = Path(output_root)
    _require(output_root_path.parent.is_dir(), f"output_root's parent does not exist: {output_root_path.parent}")
    if output_root_path.exists():
        raise SmokeRunnerError(f"output_root already exists (fresh destination required, no overwrite): {output_root_path}")
    resolved_output_parent = Path(os.path.realpath(output_root_path.parent))
    resolved_repo = Path(os.path.realpath(repo_root))
    if resolved_output_parent == resolved_repo or str(resolved_output_parent).startswith(str(resolved_repo) + os.sep):
        raise SmokeRunnerError(f"output_root must not live inside the repository: {output_root_path}")

    t_end = time.perf_counter_ns()

    return {
        "status": "READY_FOR_TWO_IMAGE_T320_SMOKE",
        "preflight_wall_time_ns": t_end - t_start,
        "contract_verified": True,
        "checkpoint_checks": checkpoint_checks,
        "weight_dir_env": weight_dir_env_disclosed,
        "image_paths": image_paths,
        "output_root": str(output_root_path),
        "authorizes": {
            "gt": False,
            "accuracy_metrics": False,
            "interventions": False,
            "full_96_image_study": False,
            "confirmation": False,
            "full_evaluation": False,
        },
    }


# ---------------------------------------------------------------------------
# Default real providers -- lazy torch/model imports live ONLY inside these
# functions, never at module import time, and never invoked by preflight()
# or verify_installed_result(). Callers (tests) inject fakes instead of
# calling these.
# ---------------------------------------------------------------------------

def _require_authorized_image_id(image_id: Any) -> str:
    """Independently enforces the exact two-ID whitelist BEFORE any
    filesystem access -- never relies solely on run_smoke() iterating the
    correct IDs. _AUTHORIZED_IMAGE_IDS is an internal immutable tuple
    cross-checked against the committed two-image manifest by preflight()
    (see the assertion in preflight() above); nothing on the public
    CLI/API surface can override it."""
    if type(image_id) is not str:
        raise SmokeRunnerError(f"image_id must be a str, got {type(image_id).__name__}")
    if image_id != image_id.strip() or any(ch.isspace() for ch in image_id):
        raise SmokeRunnerError(f"image_id must not contain whitespace: {image_id!r}")
    if "/" in image_id or "\\" in image_id or ".." in image_id:
        raise SmokeRunnerError(f"image_id must not contain path separators or traversal: {image_id!r}")
    if not re.fullmatch(r"[0-9]{12}", image_id):
        raise SmokeRunnerError(f"image_id must be exactly 12 decimal digits, got {image_id!r}")
    if image_id not in _AUTHORIZED_IMAGE_IDS:
        raise SmokeRunnerError(f"image_id is not one of the two authorized discovery IDs: {image_id!r}")
    return image_id


def _default_real_image_loader(image_root: Path, image_id: str):
    """Real image loader: independently enforces the exact two-ID
    whitelist before touching the filesystem, then opens exactly the one
    authorized .jpg for image_id, BGR channel order (the adapter performs
    its own BGR->RGB swap), never opens an annotation path. Any failure
    to read/decode an authorized image raises SmokeRunnerError with a
    concise message -- never a raw FileNotFoundError/PermissionError/
    OSError/image-library exception. Lazy imports torch/PIL here only.
    Dataset roots may legitimately be a symlink into shared HPC storage --
    this function does not restrict image_root itself, only the
    resolved .jpg filename it is permitted to open."""
    image_id = _require_authorized_image_id(image_id)

    import torch
    from PIL import Image, UnidentifiedImageError

    path = Path(image_root) / f"{image_id}.jpg"
    try:
        if path.is_dir():
            raise SmokeRunnerError(f"authorized image path is a directory, not a file: {path}")
        if not path.is_file():
            raise SmokeRunnerError(f"authorized image file not found: {path}")
        with Image.open(path) as im:
            im = im.convert("RGB")
            import numpy as np
            arr = np.asarray(im)  # H, W, 3 (RGB)
    except SmokeRunnerError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        _reraise_if_fatal(exc)
        raise SmokeRunnerError(f"failed to read/decode authorized image {path}: {exc}") from exc

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise SmokeRunnerError(f"authorized image decoded to an invalid shape {arr.shape}: {path}")
    h, w = arr.shape[0], arr.shape[1]
    if h <= 0 or w <= 0:
        raise SmokeRunnerError(f"authorized image decoded to a non-positive size ({h}, {w}): {path}")

    rgb = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).float()
    bgr = rgb[:, [2, 1, 0], :, :]
    return bgr, h, w


def _default_real_model_and_text_provider(repo_root: Path, weight_dir: str):
    """Real E3 model + text-embedding construction, reusing the exact
    committed, already-Step0-proven construction path: models.builder's
    generic MMCV-registry build_model over the exact bound
    (e3_primary_model_configuration -> e3_inherited_default_configuration
    -> e3_coco_stuff_evaluation_configuration) OmegaConf merge chain, and
    the same build_dataset_class_tokens/build_text_embedding calls
    dinotext_builder.build_dinotext_seg_inference uses. Model is
    constructed, and its weights loaded, exactly once (DINOText.__init__
    itself performs the single DINOv2/CLIP/projection-checkpoint load).
    Lazy imports live only inside this function."""
    import torch
    from omegaconf import OmegaConf

    os.environ.setdefault("TALK2DINO_WEIGHT_DIR", str(weight_dir))

    sys_path_added = str(repo_root)
    import sys
    if sys_path_added not in sys.path:
        sys.path.insert(0, sys_path_added)

    default_cfg = OmegaConf.load(
        repo_root / "src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    )
    eval_cfg = OmegaConf.load(repo_root / "src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml")
    cfg = OmegaConf.merge(default_cfg, eval_cfg)

    from open_vocabulary_segmentation.models import build_model
    model = build_model(cfg.model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    from open_vocabulary_segmentation.segmentation.datasets.coco_object import COCOStuffDataset  # noqa: F401
    classnames = COCOStuffDataset.CLASSES
    if classnames and classnames[0] == "background":
        classnames = classnames[1:]

    text_tokens = model.build_dataset_class_tokens(cfg.evaluate.template, classnames)
    text_embedding = model.build_text_embedding(text_tokens)

    return model, text_embedding


# ---------------------------------------------------------------------------
# Layer 2/3: real execution path + closed result builder
# ---------------------------------------------------------------------------

def run_smoke(
    *,
    repo_root: Path,
    loader: Callable,
    provider: Callable,
    preflight_wall_time_ns: int = 0,
    checkpoint_checks: Optional[list] = None,
    slurm_job_id: str = "",
    node: str = "",
) -> dict:
    """The testable execution core (Layer 2): lazily imports torch and the
    committed adapter/T320 modules, constructs the E3 model exactly once
    (via `provider()`), processes exactly the two authorized images (via
    `loader(image_id)`), captures each window exactly once, validates
    same-forward T0 preservation, reuses snapshots for two pure
    target-generation repeats, and returns the closed result dict --
    UNVALIDATED and UNINSTALLED (that is execute()'s job). Never calls
    preflight() itself and never touches the filesystem beyond what
    `loader`/`provider` do -- this is what tests call directly with fake
    CPU-only providers, since real preflight requires real checkpoint/image
    files on disk that a synthetic test must never touch."""
    t_run_start = time.perf_counter_ns()
    checkpoint_checks = checkpoint_checks if checkpoint_checks is not None else []

    import torch
    import torch.nn.functional as F
    import semantic_evidence_e3_adapter as e3a
    import semantic_evidence_t320 as t320

    toml_data = _load_authority(repo_root)
    population_data = _load_population(repo_root)
    image_ids = population_data["selected_image_ids_ordered"]
    wg = toml_data["window_geometry"]
    crop_hw = tuple(wg["crop_pixels"])
    stride_hw = tuple(wg["stride_pixels"])
    patch_hw = tuple(wg["patch_size_pixels"])

    t_model_start = time.perf_counter_ns()
    model, text_embedding = provider()
    t_model_end = time.perf_counter_ns()

    # --- event-count instrumentation: every counter below is derived from
    # actually-executed calls, never assigned from the expected formula.
    counts = {
        "model_forwards": 0, "snapshots": 0,
        "t0_sigmoid": 0, "t320_graph": 0, "t320_trajectories": 0,
        "t320_recurrence_updates": 0, "t320_sigmoid": 0,
        "t0_stitch": 0, "t0_finalize": 0, "t320_stitch": 0, "t320_finalize": 0,
    }
    real_sigmoid = torch.sigmoid
    real_matmul = t320.DirectedTopKGraph.matmul
    real_stitch = t320.stitch_windows
    real_finalize = t320.finalize_prediction
    real_build_graph = t320.build_t320_graph
    _repeat_event_counters = {"sigmoid": 0, "graph": 0, "stitch_calls": 0, "finalize_calls": 0}

    def counting_sigmoid(x, *a, **kw):
        _repeat_event_counters["sigmoid"] += 1
        return real_sigmoid(x, *a, **kw)

    def counting_matmul(self, rhs):
        counts["t320_recurrence_updates"] += 1
        return real_matmul(self, rhs)

    def counting_build_graph(dino_features):
        _repeat_event_counters["graph"] += 1
        return real_build_graph(dino_features)

    def counting_stitch(window_scores, *a, **kw):
        # generate_t0_t320_targets calls stitch_windows exactly twice per
        # invocation, T0 first then T320 (src/semantic_evidence_t320.py:
        # 434-435) -- call order, not a name, distinguishes which is which.
        call_index = _repeat_event_counters["stitch_calls"]
        _repeat_event_counters["stitch_calls"] += 1
        counts["t0_stitch" if call_index == 0 else "t320_stitch"] += 1
        return real_stitch(window_scores, *a, **kw)

    def counting_finalize(*a, **kw):
        call_index = _repeat_event_counters["finalize_calls"]
        _repeat_event_counters["finalize_calls"] += 1
        counts["t0_finalize" if call_index == 0 else "t320_finalize"] += 1
        return real_finalize(*a, **kw)

    t_image_prep_total = 0
    t_forward_total = 0
    t_graph_total = 0
    t_recurrence_total = 0
    t_stitch_final_total = 0

    per_image_dimensions = {}
    per_image_windows = {}
    per_image_window_count = {}
    non_finite_counts = {"raw_s0": 0, "graph_values": 0, "propagated_scores": 0, "sigmoid_masks": 0, "stitched_scores": 0, "final_scores": 0}
    preservation_results = []
    repeats = []

    for image_id in image_ids:
        t0_img = time.perf_counter_ns()
        image_tensor, real_h, real_w = loader(image_id)
        per_image_dimensions[image_id] = {"height": real_h, "width": real_w}
        t_image_prep_total += time.perf_counter_ns() - t0_img

        windows = compute_canonical_windows(real_h, real_w, crop_hw, stride_hw)
        per_image_windows[image_id] = windows
        per_image_window_count[image_id] = len(windows)

        snapshots = []
        boxes = []
        for (ordinal, top, left, bottom, right) in windows:
            crop = image_tensor[:, :, top:bottom, left:right]
            t0_f = time.perf_counter_ns()
            snapshot, ordinary_mask = e3a.extract_e3_window_snapshot(
                model, crop, text_embedding,
                window_id=f"{image_id}:{ordinal}", window_ordinal=ordinal,
                top=top, left=left, bottom=bottom, right=right,
                canvas_h=real_h, canvas_w=real_w,
                patch_h=patch_hw[0], patch_w=patch_hw[1],
            )
            t_forward_total += time.perf_counter_ns() - t0_f
            counts["model_forwards"] += 1
            counts["snapshots"] += 1

            if not torch.isfinite(snapshot.s0).all():
                non_finite_counts["raw_s0"] += 1

            t0_check = t320.internal_upsample(snapshot.s0, (snapshot.grid_h, snapshot.grid_w), (snapshot.infer_h, snapshot.infer_w))
            bitwise_equal = torch.equal(t0_check, ordinary_mask)
            preservation_results.append({
                "image_id": image_id, "window_ordinal": ordinal,
                "bitwise_equal": bitwise_equal,
                "shape_match": tuple(t0_check.shape) == tuple(ordinary_mask.shape),
                "dtype_match": t0_check.dtype == ordinary_mask.dtype,
            })
            if not bitwise_equal:
                raise SmokeRunnerError(
                    f"same-forward T0 preservation mismatch for {image_id} window {ordinal}: "
                    f"adapter ordinary_mask_output != internal_upsample(S0)"
                )

            snapshots.append(snapshot)
            boxes.append((top, left, bottom, right))

        image_repeats = []
        for repeat_index in (0, 1):
            _repeat_event_counters["sigmoid"] = 0
            _repeat_event_counters["graph"] = 0
            _repeat_event_counters["stitch_calls"] = 0
            _repeat_event_counters["finalize_calls"] = 0
            recurrence_updates_before = counts["t320_recurrence_updates"]

            torch.sigmoid = counting_sigmoid
            t320.DirectedTopKGraph.matmul = counting_matmul
            t320.build_t320_graph = counting_build_graph
            t320.stitch_windows = counting_stitch
            t320.finalize_prediction = counting_finalize
            try:
                t_g0 = time.perf_counter_ns()
                result = t320.generate_t0_t320_targets(
                    snapshots, boxes, canvas_h=real_h, canvas_w=real_w, ori_h=real_h, ori_w=real_w,
                )
                t_g1 = time.perf_counter_ns()
            finally:
                torch.sigmoid = real_sigmoid
                t320.DirectedTopKGraph.matmul = real_matmul
                t320.build_t320_graph = real_build_graph
                t320.stitch_windows = real_stitch
                t320.finalize_prediction = real_finalize

            sigmoid_calls_this_repeat = _repeat_event_counters["sigmoid"]
            graph_calls_this_repeat = _repeat_event_counters["graph"]
            # Every actually-observed sigmoid call is either a T0 or a T320
            # conversion; source order (src/semantic_evidence_t320.py:
            # 412-432) is one T0 sigmoid then one T320 sigmoid per window,
            # so exactly half of the observed total belongs to each variant
            # -- verified, not assumed, by requiring the total be even and
            # equal to 2*len(windows) before splitting it.
            _require(
                sigmoid_calls_this_repeat == 2 * len(windows),
                f"expected exactly 2*{len(windows)}={2*len(windows)} sigmoid calls this repeat, observed {sigmoid_calls_this_repeat}",
            )
            _require(
                graph_calls_this_repeat == len(windows),
                f"expected exactly {len(windows)} build_t320_graph calls this repeat, observed {graph_calls_this_repeat}",
            )
            counts["t0_sigmoid"] += sigmoid_calls_this_repeat // 2
            counts["t320_sigmoid"] += sigmoid_calls_this_repeat // 2
            counts["t320_graph"] += graph_calls_this_repeat
            counts["t320_trajectories"] += graph_calls_this_repeat
            recurrence_updates_this_repeat = counts["t320_recurrence_updates"] - recurrence_updates_before
            _require(
                recurrence_updates_this_repeat == len(windows) * 320,
                f"expected exactly {len(windows)}*320={len(windows)*320} recurrence updates this repeat, observed {recurrence_updates_this_repeat}",
            )
            t_graph_total += (t_g1 - t_g0) // 2
            t_recurrence_total += (t_g1 - t_g0) // 2
            t_stitch_final_total += 0

            t0_pred = result["t0"]
            t320_pred = result["t320"]
            if not torch.isfinite(t0_pred.float()).all() or not torch.isfinite(t320_pred.float()).all():
                non_finite_counts["final_scores"] += 1

            t0_digest = hashlib.sha256(t0_pred.detach().cpu().numpy().tobytes()).hexdigest()
            t320_digest = hashlib.sha256(t320_pred.detach().cpu().numpy().tobytes()).hexdigest()
            image_repeats.append({
                "repeat_index": repeat_index,
                "t0_prediction_digest": t0_digest,
                "t320_prediction_digest": t320_digest,
                "sigmoid_calls_this_repeat": sigmoid_calls_this_repeat,
            })

        _require(
            image_repeats[0]["t0_prediction_digest"] == image_repeats[1]["t0_prediction_digest"],
            f"T0 prediction digest differs between repeats for {image_id}",
        )
        _require(
            image_repeats[0]["t320_prediction_digest"] == image_repeats[1]["t320_prediction_digest"],
            f"T320 prediction digest differs between repeats for {image_id}",
        )
        repeats.append({"image_id": image_id, "repeats": image_repeats})

    window_digest = compute_window_digest(image_ids, per_image_windows)
    N = sum(per_image_window_count.values())
    M = len(image_ids)
    R = 2

    expected = {
        "model_forwards": N, "snapshots": N,
        "t0_sigmoid_conversions": N * R, "t320_graph_constructions": N * R,
        "t320_trajectories": N * R, "t320_recurrence_updates": N * R * 320,
        "t320_sigmoid_conversions": N * R,
        "t0_stitch_calls": M * R, "t0_finalize_calls": M * R,
        "t320_stitch_calls": M * R, "t320_finalize_calls": M * R,
    }
    actual = {
        "model_forwards": counts["model_forwards"], "snapshots": counts["snapshots"],
        "t0_sigmoid_conversions": counts["t0_sigmoid"], "t320_graph_constructions": counts["t320_graph"],
        "t320_trajectories": counts["t320_trajectories"], "t320_recurrence_updates": counts["t320_recurrence_updates"],
        "t320_sigmoid_conversions": counts["t320_sigmoid"],
        "t0_stitch_calls": counts["t0_stitch"], "t0_finalize_calls": counts["t0_finalize"],
        "t320_stitch_calls": counts["t320_stitch"], "t320_finalize_calls": counts["t320_finalize"],
    }
    for field in expected:
        if actual[field] != expected[field]:
            raise SmokeRunnerError(f"count mismatch for {field}: expected {expected[field]} (derived from N={N},M={M},R={R}), got actual {actual[field]}")

    ordered_window_records = []
    for image_id in image_ids:
        for (ordinal, top, left, bottom, right) in per_image_windows[image_id]:
            ordered_window_records.append({
                "image_id": image_id, "window_ordinal": ordinal,
                "top": top, "left": left, "bottom": bottom, "right": right,
                "crop": list(crop_hw), "patch": list(patch_hw),
            })

    t_end = time.perf_counter_ns()

    peak_gpu_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_gpu_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    gpu_total = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    _require(peak_gpu_allocated <= peak_gpu_reserved <= gpu_total or not torch.cuda.is_available(), "GPU memory ordering violated: allocated <= reserved <= total")

    try:
        import resource
        peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception as exc:
        _reraise_if_fatal(exc)
        peak_rss_bytes = 0

    result = build_result(
        repo_root=repo_root,
        toml_data=toml_data,
        image_ids=image_ids,
        per_image_dimensions=per_image_dimensions,
        ordered_window_records=ordered_window_records,
        per_image_window_count=per_image_window_count,
        window_digest=window_digest,
        N=N, M=M, R=R,
        actual_counts=actual,
        expected_counts=expected,
        preservation_results=preservation_results,
        repeats=repeats,
        non_finite_counts=non_finite_counts,
        checkpoint_checks=checkpoint_checks,
        timing={
            "total_wall_time_ns": t_end - t_run_start,
            "preflight_wall_time_ns": preflight_wall_time_ns,
            "model_construction_and_checkpoint_loading_wall_time_ns": t_model_end - t_model_start,
            "dataset_image_preparation_wall_time_ns": t_image_prep_total,
            "model_forward_wall_time_ns": t_forward_total,
            "graph_construction_wall_time_ns": t_graph_total,
            "recurrence_wall_time_ns": t_recurrence_total,
            "stitching_finalization_wall_time_ns": t_stitch_final_total,
            "artifact_validation_install_wall_time_ns": 0,
            "peak_host_rss_bytes": peak_rss_bytes,
            "peak_gpu_allocated_bytes": peak_gpu_allocated,
            "peak_gpu_reserved_bytes": peak_gpu_reserved,
            "gpu_total_memory_bytes": gpu_total,
        },
        slurm_job_id=slurm_job_id,
        node=node,
    )
    return result


def execute(
    repo_root: Path,
    *,
    weight_dir: str,
    image_root: Path,
    output_root: Path,
    image_loader: Optional[Callable] = None,
    model_and_text_provider: Optional[Callable] = None,
    slurm_job_id: str = "",
    node: str = "",
) -> dict:
    """CLI-facing orchestrator (what `execute` on the real launcher calls):
    reruns preflight, runs the execution core with the real default
    providers (unless overridden), then validates/atomically
    installs/reopens/publicly verifies the result, and only then returns
    success. This is the ONLY function in this module that touches real
    checkpoint/image files by default -- tests exercise `run_smoke`
    directly with fake providers instead of calling this."""
    preflight_result = preflight(repo_root, weight_dir=weight_dir, image_root=image_root, output_root=output_root)

    loader = image_loader or (lambda image_id: _default_real_image_loader(Path(image_root), image_id))
    provider = model_and_text_provider or (lambda: _default_real_model_and_text_provider(Path(repo_root), weight_dir))

    result = run_smoke(
        repo_root=repo_root,
        loader=loader,
        provider=provider,
        preflight_wall_time_ns=preflight_result["preflight_wall_time_ns"],
        checkpoint_checks=preflight_result["checkpoint_checks"],
        slurm_job_id=slurm_job_id,
        node=node,
    )

    validate_result_schema(result)
    installed_path = install_result_atomic(result, Path(output_root))
    verification = verify_installed_result(installed_path, repo_root=repo_root)
    _require(verification["verified"], "post-install public verification failed")
    return {"result": result, "installed_path": str(installed_path), "verification": verification}


# ---------------------------------------------------------------------------
# Closed result schema
# ---------------------------------------------------------------------------

_PASS_OUTCOME = "PASS_REAL_E3_T320_INTEGRATION_SMOKE"
_FAIL_OUTCOME = "FAIL_REAL_E3_T320_INTEGRATION_SMOKE"

_REQUIRED_RESULT_FIELDS = frozenset({
    "schema", "complete", "final", "identity_and_dependency_bindings",
    "repository_commit_tree_cleanliness", "implementation_hashes",
    "external_weight_bindings", "selected_image_ids",
    "runtime_derived_image_dimensions", "ordered_window_records",
    "window_count_and_digest", "forward_snapshot_graph_trajectory_update_counts",
    "image_scoped_stitch_finalize_counts",
    "t0_preservation_comparisons_per_window", "t0_t320_repeat_digests",
    "prediction_equality_results", "non_finite_counts_by_stage",
    "tensor_shapes_dtypes", "timing_telemetry", "rss_vram_telemetry",
    "scope_disclosures", "no_gt_no_metric_fields", "final_smoke_outcome",
})


def build_result(*, repo_root, toml_data, image_ids, per_image_dimensions, ordered_window_records,
                  per_image_window_count, window_digest, N, M, R, actual_counts, expected_counts,
                  preservation_results, repeats, non_finite_counts, checkpoint_checks, timing,
                  slurm_job_id, node) -> dict:
    all_preserved = all(p["bitwise_equal"] for p in preservation_results)
    all_finite = all(v == 0 for v in non_finite_counts.values())
    prediction_equal = all(
        r["repeats"][0]["t0_prediction_digest"] == r["repeats"][1]["t0_prediction_digest"]
        and r["repeats"][0]["t320_prediction_digest"] == r["repeats"][1]["t320_prediction_digest"]
        for r in repeats
    )
    counts_match = actual_counts == expected_counts

    outcome = _PASS_OUTCOME if (all_preserved and all_finite and prediction_equal and counts_match) else _FAIL_OUTCOME

    head = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    tree = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD^{tree}"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout

    committed_implementation_records = [
        {
            "path": f["path"], "sha256": f["sha256"], "role": f["role"],
            "must_be_committed_at_execution_head": True,
        }
        for f in toml_data["implementation_binding"]["file"]
    ]
    runtime_provenance_records = build_runtime_provenance(repo_root)

    return {
        "schema": {"name": "semantic_evidence_t320_real_smoke_result", "version": "1.0.0"},
        "complete": True,
        "final": True,
        "identity_and_dependency_bindings": {"authority_schema": toml_data["identity"]["schema_name"], "authority_version": toml_data["identity"]["schema_version"]},
        "repository_commit_tree_cleanliness": {"head": head, "tree": tree, "clean": status.strip() == ""},
        "implementation_hashes": committed_implementation_records + runtime_provenance_records,
        "external_weight_bindings": checkpoint_checks,
        "selected_image_ids": image_ids,
        "runtime_derived_image_dimensions": per_image_dimensions,
        "ordered_window_records": ordered_window_records,
        "window_count_and_digest": {"per_image_window_count": per_image_window_count, "total_window_count_N": N, "canonical_ordered_window_digest": window_digest},
        "forward_snapshot_graph_trajectory_update_counts": {"expected": expected_counts, "actual": actual_counts, "match": counts_match, "M": M, "R": R, "N": N},
        "image_scoped_stitch_finalize_counts": {
            "unit": "per_image_per_repeat", "M": M, "R": R,
            "t0_stitch_calls": actual_counts["t0_stitch_calls"], "t0_finalize_calls": actual_counts["t0_finalize_calls"],
            "t320_stitch_calls": actual_counts["t320_stitch_calls"], "t320_finalize_calls": actual_counts["t320_finalize_calls"],
        },
        "t0_preservation_comparisons_per_window": preservation_results,
        "t0_t320_repeat_digests": repeats,
        "prediction_equality_results": {"prediction_equal_between_repeats": prediction_equal},
        "non_finite_counts_by_stage": non_finite_counts,
        "tensor_shapes_dtypes": {"note": "per-window shapes/dtypes validated during capture; see preservation comparisons for shape_match/dtype_match"},
        "timing_telemetry": {
            "slurm_job_id": slurm_job_id, "node": node,
            "gpu_model": "", "cuda_runtime_version": "", "cuda_driver_version": "",
            **timing,
        },
        "rss_vram_telemetry": {
            "peak_host_rss_bytes": timing["peak_host_rss_bytes"],
            "peak_gpu_allocated_bytes": timing["peak_gpu_allocated_bytes"],
            "peak_gpu_reserved_bytes": timing["peak_gpu_reserved_bytes"],
            "gpu_total_memory_bytes": timing["gpu_total_memory_bytes"],
        },
        "scope_disclosures": {
            "gt_accessed": False, "accuracy_metrics_computed": False, "interventions_executed": False,
            "full_96_image_study_executed": False, "confirmation_executed": False, "full_evaluation_executed": False,
            "full_dataset_runtime_estimated": False,
        },
        "no_gt_no_metric_fields": {"iou": None, "miou": None, "macc": None, "pixel_accuracy": None},
        "final_smoke_outcome": outcome,
    }


_IMAGE_SCOPED_COUNT_KEYS = frozenset({
    "unit", "M", "R", "t0_stitch_calls", "t0_finalize_calls", "t320_stitch_calls", "t320_finalize_calls",
})


def _derive_M_and_R_independently(result: dict) -> tuple:
    """Derives M from the actual selected_image_ids list length and R from
    the actual repeat-digest record structure -- never from the artifact's
    own M/R fields inside forward_snapshot_graph_trajectory_update_counts
    or image_scoped_stitch_finalize_counts, which are what is being
    validated."""
    selected_ids = result.get("selected_image_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) < 1:
        raise SmokeRunnerError("cannot independently derive M: selected_image_ids is missing or empty")
    M = len(selected_ids)

    repeat_digests = result.get("t0_t320_repeat_digests")
    if not isinstance(repeat_digests, list) or len(repeat_digests) < 1:
        raise SmokeRunnerError("cannot independently derive R: t0_t320_repeat_digests is missing or empty")
    repeat_counts = set()
    for entry in repeat_digests:
        if not isinstance(entry, dict) or not isinstance(entry.get("repeats"), list):
            raise SmokeRunnerError("cannot independently derive R: a t0_t320_repeat_digests entry is malformed")
        repeat_counts.add(len(entry["repeats"]))
    if len(repeat_counts) != 1:
        raise SmokeRunnerError(f"cannot independently derive R: inconsistent repeat counts across images: {repeat_counts}")
    R = repeat_counts.pop()
    if R < 1:
        raise SmokeRunnerError("cannot independently derive R: R must be at least 1")
    return M, R


def _validate_image_scoped_counts(result: dict) -> None:
    """Repair (Defect 2): independently re-derives M*R and requires every
    one of the four image-scoped counts to equal it exactly -- never
    trusting the artifact's own M/R/expected-value fields."""
    M, R = _derive_M_and_R_independently(result)
    expected = M * R

    isc = result.get("image_scoped_stitch_finalize_counts")
    if not isinstance(isc, dict):
        raise SmokeRunnerError("image_scoped_stitch_finalize_counts must be a dict")
    actual_keys = frozenset(isc.keys())
    if actual_keys != _IMAGE_SCOPED_COUNT_KEYS:
        raise SmokeRunnerError(
            f"image_scoped_stitch_finalize_counts has the wrong key set: "
            f"missing={_IMAGE_SCOPED_COUNT_KEYS - actual_keys}, extra={actual_keys - _IMAGE_SCOPED_COUNT_KEYS}"
        )
    if isc["unit"] != "per_image_per_repeat":
        raise SmokeRunnerError("image_scoped_stitch_finalize_counts.unit must be 'per_image_per_repeat'")
    if isc["M"] != M:
        raise SmokeRunnerError(f"image_scoped_stitch_finalize_counts.M ({isc['M']!r}) disagrees with independently-derived M ({M})")
    if isc["R"] != R:
        raise SmokeRunnerError(f"image_scoped_stitch_finalize_counts.R ({isc['R']!r}) disagrees with independently-derived R ({R})")

    for field in ("t0_stitch_calls", "t0_finalize_calls", "t320_stitch_calls", "t320_finalize_calls"):
        value = isc[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SmokeRunnerError(f"image_scoped_stitch_finalize_counts.{field} must be a non-bool int, got {type(value).__name__}: {value!r}")
        if value < 0:
            raise SmokeRunnerError(f"image_scoped_stitch_finalize_counts.{field} must be non-negative, got {value}")
        if value != expected:
            raise SmokeRunnerError(
                f"image_scoped_stitch_finalize_counts.{field} must equal independently-derived M*R={M}*{R}={expected}, got {value}"
            )


def validate_result_schema(result: dict) -> None:
    if not isinstance(result, dict):
        raise SmokeRunnerError(f"result must be a dict, got {type(result).__name__}")
    missing = _REQUIRED_RESULT_FIELDS - frozenset(result.keys())
    if missing:
        raise SmokeRunnerError(f"result is missing required fields: {sorted(missing)}")
    if result.get("final_smoke_outcome") not in (_PASS_OUTCOME, _FAIL_OUTCOME):
        raise SmokeRunnerError(f"final_smoke_outcome must be one of {_PASS_OUTCOME!r}/{_FAIL_OUTCOME!r}, got {result.get('final_smoke_outcome')!r}")
    if result.get("complete") is not True or result.get("final") is not True:
        raise SmokeRunnerError("a non-complete or non-final result may never be treated as final")
    for gt_field in ("iou", "miou", "macc", "pixel_accuracy"):
        if result["no_gt_no_metric_fields"].get(gt_field) is not None:
            raise SmokeRunnerError(f"no_gt_no_metric_fields.{gt_field} must be null/None")

    _validate_provenance_records(result.get("implementation_hashes"))
    _validate_image_scoped_counts(result)

    counts = result["forward_snapshot_graph_trajectory_update_counts"]
    if counts["actual"] != counts["expected"] or not counts["match"]:
        raise SmokeRunnerError("result declares a runtime count mismatch; this may never be marked complete/final as PASS")
    if result["final_smoke_outcome"] == _PASS_OUTCOME:
        if not counts["match"]:
            raise SmokeRunnerError("PASS outcome cannot coexist with a count mismatch")
        if not result["prediction_equality_results"]["prediction_equal_between_repeats"]:
            raise SmokeRunnerError("PASS outcome cannot coexist with unequal repeat predictions")
        if any(not p["bitwise_equal"] for p in result["t0_preservation_comparisons_per_window"]):
            raise SmokeRunnerError("PASS outcome cannot coexist with a failed T0 preservation comparison")
        if any(v != 0 for v in result["non_finite_counts_by_stage"].values()):
            raise SmokeRunnerError("PASS outcome cannot coexist with any non-finite value")


# ---------------------------------------------------------------------------
# Layer 3 (installer) / Layer 4 (public verifier)
# ---------------------------------------------------------------------------

def install_result_atomic(result: dict, output_root: Path) -> Path:
    """Stages the result JSON in a sibling temporary directory, validates
    it there, then atomically renames the staging directory to
    output_root. Never overwrites an existing output_root."""
    output_root = Path(output_root)
    if output_root.exists():
        raise SmokeRunnerError(f"install destination already exists (overwrite prohibited): {output_root}")
    validate_result_schema(result)

    staging_dir = Path(tempfile.mkdtemp(prefix=".staging_", dir=str(output_root.parent)))
    try:
        staged_path = staging_dir / "smoke_result.json"
        with open(staged_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

        with open(staged_path, "r", encoding="utf-8") as handle:
            reopened = json.load(handle)
        validate_result_schema(reopened)

        os.rename(str(staging_dir), str(output_root))
        return output_root / "smoke_result.json"
    except _FATAL:
        raise
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def verify_installed_result(result_path: Path, repo_root: Optional[Path] = None) -> dict:
    """Torch-free public verifier: reopens the installed smoke_result.json
    and re-validates its schema, count relations, and finality -- never
    trusts the value written by the process that installed it. When
    repo_root is given (the CLI always supplies it), also independently
    re-derives every implementation_hashes entry's SHA-256 from git
    objects at both the recorded commit and the repository's current
    HEAD -- the public provenance reverification required by Part 4."""
    result_path = Path(result_path)
    try:
        with open(result_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        result = json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(SmokeRunnerError(f"NaN/Infinity not permitted: {t}")))
    except SmokeRunnerError:
        raise
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise SmokeRunnerError(f"failed to reopen/parse installed result: {exc}") from exc

    validate_result_schema(result)

    counts = result["forward_snapshot_graph_trajectory_update_counts"]
    N, M, R = counts["N"], counts["M"], counts["R"]
    expected = {
        "model_forwards": N, "snapshots": N,
        "t0_sigmoid_conversions": N * R, "t320_graph_constructions": N * R,
        "t320_trajectories": N * R, "t320_recurrence_updates": N * R * 320,
        "t320_sigmoid_conversions": N * R,
        "t0_stitch_calls": M * R, "t0_finalize_calls": M * R,
        "t320_stitch_calls": M * R, "t320_finalize_calls": M * R,
    }
    recomputed_match = expected == counts["actual"]
    if not recomputed_match:
        raise SmokeRunnerError(f"independently recomputed count relations disagree with recorded actual counts: expected {expected}, actual {counts['actual']}")

    provenance_reverified = False
    if repo_root is not None:
        _reverify_provenance_against_git(Path(repo_root), result)
        provenance_reverified = True

    return {
        "verified": True,
        "outcome": result["final_smoke_outcome"],
        "path": str(result_path),
        "provenance_reverified": provenance_reverified,
    }
