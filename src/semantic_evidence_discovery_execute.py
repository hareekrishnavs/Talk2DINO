"""48-image semantic-evidence discovery execution engine.

Implements research_specs/semantic_evidence_discovery_execution_v1.toml
exactly. Reuses the committed, unmodified E3 adapter (semantic_evidence_
e3_adapter.py) and isolated T320 implementation (semantic_evidence_t320.py)
directly -- never reimplements graph construction, recurrence, internal
sigmoid/upsample, stitching, or final rescale.

Layer boundaries (mirrors semantic_evidence_t320_real_smoke.py's own
layering, which this module's real-provider function is adapted from):

  Layer 1 -- torch-free preflight (verify-protocol / preflight / verify-result)
  Layer 2 -- real execution core (lazy torch import; run_discovery())
  Layer 3 -- artifact validators, checkpoint/atomic-install helpers

Population is loaded exclusively from the committed discovery authority's
own dependency-bound population manifest; the locked 48-image holdout is
never passed to any provider, loader, log, checkpoint, or CLI argument in
this module.
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


class DiscoveryExecutionError(Exception):
    pass


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        raise exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiscoveryExecutionError(message)


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
AUTHORITY_TOML_PATH = "research_specs/semantic_evidence_discovery_execution_v1.toml"

# ---------------------------------------------------------------------------
# Layer 1: torch-free protocol verification and preflight
# ---------------------------------------------------------------------------

def verify_protocol(repo_root: Path) -> dict:
    """Torch-free: re-runs the committed discovery-authority's own official
    verifier (semantic_evidence_discovery_execution_identity.verify_all)."""
    import sys
    sys.path.insert(0, str(repo_root / "src"))
    import semantic_evidence_discovery_execution_identity as ident  # torch-free

    result = ident.verify_all(repo_root, repo_root / AUTHORITY_TOML_PATH)
    return {
        "verified": result["verified"],
        "prerequisite_gate_head_ok": result["prerequisite_gate"]["head_ok"],
        "dependency_hashes_ok": result["dependency_hashes"]["all_match"],
        "population_ok": result["population"]["all_ok"],
        "scientific_inheritance_ok": result["scientific_inheritance"]["all_ok"],
        "t0_t320_frozen_constants_ok": result["t0_t320_frozen_constants"]["all_ok"],
        "smoke_prerequisite_ok": result["smoke_prerequisite"].get("all_ok"),
        "gate_inputs_exclude_profiling_ok": result["gate_inputs_exclude_profiling"]["all_ok"],
        "artifact_required_fields_ok": result["artifact_required_fields"]["all_ok"],
        "future_implementation_files_ok": result["future_implementation_files"]["all_ok"],
        "resource_contract_ok": result["resource_contract"]["all_ok"],
        "authorization_flags_ok": result["authorization_flags"]["all_ok"],
    }


def _load_authority(repo_root: Path) -> dict:
    with open(repo_root / AUTHORITY_TOML_PATH, "rb") as fh:
        return tomllib.load(fh)


def _git_show_bytes(repo_root: Path, commit: str, relative_path: str) -> bytes:
    proc = subprocess.run(["git", "show", f"{commit}:{relative_path}"], cwd=str(repo_root), capture_output=True)
    _require(proc.returncode == 0, f"git show failed for {commit}:{relative_path}")
    return proc.stdout


def load_discovery_population(repo_root: Path) -> list:
    """Loads exactly the 48 frozen discovery IDs, in frozen order, from the
    committed authority's population_manifest dependency. Never returns or
    references the locked_holdout IDs."""
    data = _load_authority(repo_root)
    required_head = data["prerequisite"]["required_head"]
    manifest_dep = next(e for e in data["dependency"] if e["role"] == "population_manifest")
    raw = _git_show_bytes(repo_root, required_head, manifest_dep["path"])
    manifest = json.loads(raw)

    pop = data["population"]
    dr_lo, dr_hi = pop["discovery_rank_range"]
    records = sorted(manifest["records"], key=lambda r: r["selection_rank"])
    discovery_ids = [r["canonical_image_id"] for r in records if dr_lo <= r["selection_rank"] <= dr_hi]
    _require(len(discovery_ids) == pop["discovery_count"], "loaded discovery population count mismatch")

    digest = hashlib.sha256("".join(i + "\n" for i in discovery_ids).encode("ascii")).hexdigest()
    _require(digest == pop["discovery_digest"], "loaded discovery population digest mismatch")

    return discovery_ids


_COCO_ID_RE = re.compile(r"^[0-9]{12}$")


def validate_discovery_image_id(image_id: str, authorized_ids: tuple) -> str:
    """Independently rejects holdout IDs, arbitrary COCO IDs, malformed
    IDs, and traversal/absolute/option-like strings BEFORE touching the
    filesystem -- mirrors semantic_evidence_t320_real_smoke.py's own
    _validate_image_id pattern."""
    if type(image_id) is not str:
        raise DiscoveryExecutionError(f"image_id must be a str, got {type(image_id).__name__}")
    if image_id != image_id.strip() or any(ch.isspace() for ch in image_id):
        raise DiscoveryExecutionError(f"image_id must not contain whitespace: {image_id!r}")
    if "/" in image_id or "\\" in image_id or ".." in image_id or image_id.startswith("-"):
        raise DiscoveryExecutionError(f"image_id must not contain path separators, traversal, or look option-like: {image_id!r}")
    if not _COCO_ID_RE.fullmatch(image_id):
        raise DiscoveryExecutionError(f"image_id must be exactly 12 decimal digits: {image_id!r}")
    if image_id not in authorized_ids:
        raise DiscoveryExecutionError(f"image_id is not one of the 48 authorized discovery IDs: {image_id!r}")
    return image_id


def preflight(repo_root: Path, *, weight_dir: str, image_root: Path, gt_root: Path, output_root: Path, mode: str = "fresh") -> dict:
    """Torch-free readiness check. Never imports torch/torchvision/mmcv/
    mmseg/numpy/PIL and never calls torch.load; checkpoint verification is
    raw-byte SHA-256 only; GT verification is path/metadata only -- never
    opens or decodes a GT annotation file. `gt_root` is required and has no
    default or environment-variable fallback: an operator must supply it
    explicitly, exactly as `execute` already requires."""
    _require(mode in ("fresh", "resume", "finalize_only"), f"unknown mode: {mode!r}")

    import sys
    sys.path.insert(0, str(repo_root / "src"))
    import semantic_evidence_discovery_execution_identity as ident

    data = _load_authority(repo_root)
    deps = ident.verify_dependency_hashes(repo_root, data)
    _require(deps["all_match"], "committed discovery-authority dependency hashes do not match")

    pop = ident.verify_population(repo_root, data)
    _require(pop["all_ok"], "committed discovery population verification failed")

    discovery_ids = load_discovery_population(repo_root)
    _require(len(discovery_ids) == 48, "expected exactly 48 discovery IDs")

    image_root = Path(image_root)
    missing_images = [i for i in discovery_ids if not (image_root / f"{i}.jpg").exists()]
    _require(not missing_images, f"missing discovery image files: {missing_images[:3]}{'...' if len(missing_images) > 3 else ''}")

    gt_root = Path(gt_root)
    _require(gt_root.exists(), f"gt_root does not exist: {gt_root}")
    _require(gt_root.is_dir(), f"gt_root is not a directory: {gt_root}")
    gt_checks = []
    gt_problems = []
    for image_id in discovery_ids:
        gt_path = gt_root / f"{image_id}.png"
        if not gt_path.exists():
            gt_problems.append(f"{image_id}: missing ({gt_path})")
            present = False
        elif not gt_path.is_file():
            gt_problems.append(f"{image_id}: exists but is not a regular file ({gt_path})")
            present = False
        else:
            present = True
        gt_checks.append({"image_id": image_id, "path": str(gt_path), "present": present})
    _require(
        not gt_problems,
        f"invalid or missing discovery GT annotation paths ({len(gt_problems)} of 48): "
        f"{gt_problems[:3]}{'...' if len(gt_problems) > 3 else ''} "
        f"(each discovery image requires exactly gt_root/{{image_id}}.png)",
    )

    checkpoint_checks = []
    for key, filename, expected_sha in _external_checkpoint_bindings():
        path = Path(weight_dir) / filename if key != "e3_projection_checkpoint" else repo_root / "weights" / filename
        exists = path.exists()
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
        checkpoint_checks.append({
            "key": key, "path": str(path), "sha256": actual_sha,
            "match": exists and actual_sha == expected_sha,
        })
    _require(all(c["match"] for c in checkpoint_checks), "external checkpoint raw-byte hash mismatch")

    output_root = Path(output_root)
    _require("annotations" not in str(image_root).lower() and "ann_dir" not in str(image_root).lower(),
             "image_root looks like an annotation directory")
    if mode == "fresh":
        _require(not output_root.exists(), f"fresh mode requires an absent output_root: {output_root}")
    else:
        _require(output_root.exists(), f"{mode} mode requires an existing output_root: {output_root}")

    return {
        "status": "READY_FOR_48_IMAGE_DISCOVERY_EXECUTION",
        "mode": mode,
        "discovery_image_count": len(discovery_ids),
        "discovery_id_prefix_sample": discovery_ids[:2],
        "checkpoint_checks": checkpoint_checks,
        "gt_checks_count": len(gt_checks),
        "output_root": str(output_root),
        "authorizes": {
            "holdout": False, "full_96_image_study": False, "full_evaluation": False,
            "gt": True,  # GT is authorized, but only in Phase B, after GT-free persistence
        },
    }


def _external_checkpoint_bindings() -> list:
    """Raw-byte SHA-256 bindings for the three external checkpoints, taken
    verbatim from the committed feasibility contract's [external_weights]
    table (the same three checkpoints the real-smoke authority already
    verified in job 22640114)."""
    return [
        ("dino_backbone_checkpoint", "dinov2_vitb14_reg4_pretrain.pth", "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71"),
        ("clip_text_checkpoint", "ViT-B-16.pt", "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"),
        ("e3_projection_checkpoint", "vitb_mlp_infonce_paired_soft_routing_tau010.pth", "9632117c18613674eeddcb35f86193d37da203a9cbd837cf500c4121b5fa9942"),
    ]


# ---------------------------------------------------------------------------
# Layer 2: provider interfaces
# ---------------------------------------------------------------------------

class GTProvider:
    """Separate object, never reachable from semantic_evidence_intervention_
    features.py. Only constructed/called in Phase B, after Phase A's GT-free
    persistence boundary succeeds for the image in question."""

    def __init__(self, load_fn: Callable[[str], Any]):
        self._load_fn = load_fn

    def load(self, image_id: str) -> Any:
        return self._load_fn(image_id)


def _default_real_image_loader(image_root: Path, image_id: str, authorized_ids: tuple):
    """Lazy-imports torch/PIL only inside this function. Mirrors
    semantic_evidence_t320_real_smoke.py's own _default_real_image_loader:
    independently re-validates the whitelist before touching the filesystem,
    BGR channel order (the adapter performs its own BGR->RGB swap). Returns
    a contiguous [1, 3, H, W] float tensor -- the batch dimension is applied
    exactly once, matching what process_window_gt_free's 4-index crop
    slicing (image_tensor[:, :, top:bottom, left:right]) requires. Any
    failure to validate/read/decode an authorized image raises
    DiscoveryExecutionError with a concise message -- never a raw
    FileNotFoundError/OSError/UnidentifiedImageError/ValueError. Fatal
    exceptions (KeyboardInterrupt, SystemExit, MemoryError) always
    propagate unchanged."""
    validate_discovery_image_id(image_id, authorized_ids)

    import torch
    from PIL import Image, UnidentifiedImageError
    import numpy as np

    path = image_root / f"{image_id}.jpg"
    try:
        if path.is_dir():
            raise DiscoveryExecutionError(f"authorized discovery image path is a directory, not a file: {path}")
        if not path.is_file():
            raise DiscoveryExecutionError(f"authorized discovery image file not found: {path}")
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.array(im)[:, :, ::-1]  # RGB -> BGR
    except DiscoveryExecutionError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        _reraise_if_fatal(exc)
        raise DiscoveryExecutionError(f"failed to decode authorized discovery image {image_id}: {exc}") from exc

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise DiscoveryExecutionError(f"authorized discovery image {image_id} decoded to an invalid shape {arr.shape}: {path}")
    h, w = arr.shape[0], arr.shape[1]
    if h <= 0 or w <= 0:
        raise DiscoveryExecutionError(f"authorized discovery image {image_id} decoded to a non-positive size ({h}, {w}): {path}")

    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).float()

    if tensor.ndim != 4:
        raise DiscoveryExecutionError(f"authorized discovery image {image_id} produced tensor rank {tensor.ndim}, expected 4")
    if tensor.shape[0] != 1:
        raise DiscoveryExecutionError(f"authorized discovery image {image_id} produced batch size {tensor.shape[0]}, expected 1")
    if tensor.shape[1] != 3:
        raise DiscoveryExecutionError(f"authorized discovery image {image_id} produced {tensor.shape[1]} channels, expected 3")
    if not torch.isfinite(tensor).all():
        raise DiscoveryExecutionError(f"authorized discovery image {image_id} decoded to non-finite pixel values")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    return tensor, h, w


def _default_real_gt_loader(gt_root: Path, image_id: str):
    """Lazy-imports PIL only inside this function. Reads a COCO-Stuff-style
    single-channel PNG semantic annotation at gt_root/{image_id}.png, class
    ids 0-170 with 255 reserved for ignore, matching
    [gt_targets.primary_correctness].ignore_gt_value. Returns a plain
    nested-list raster (row-major, [H][W] of int) so the majority-vote
    helper below stays usable from fake-provider tests without PIL/numpy."""
    from PIL import Image

    path = Path(gt_root) / f"{image_id}.png"
    if not path.exists():
        raise DiscoveryExecutionError(f"authorized discovery GT annotation not found: {path}")
    try:
        with Image.open(path) as im:
            _require(im.mode in ("L", "P", "I"), f"unexpected GT annotation PIL mode: {im.mode}")
            return [list(row) for row in _pil_image_to_rows(im)]
    except Exception as exc:
        _reraise_if_fatal(exc)
        raise DiscoveryExecutionError(f"failed to decode authorized discovery GT annotation {image_id}: {exc}") from exc


def _pil_image_to_rows(im) -> list:
    w, h = im.size
    flat = list(im.getdata())
    return [flat[r * w:(r + 1) * w] for r in range(h)]


def _default_real_model_and_text_provider(repo_root: Path, weight_dir: str):
    """Real E3 model + text-embedding construction. Reuses exactly the
    already-debugged construction path from semantic_evidence_t320_real_
    smoke.py's own _default_real_model_and_text_provider (flat import
    convention, mmseg.datasets.COCOStuffDataset, 171-class assertion) --
    this function is a direct adaptation, not a reimplementation, and
    carries none of the smoke runner's two-image/R=2/profiling-specific
    behavior. Lazy imports live only inside this function; never called by
    verify-protocol, preflight, or verify-result."""
    import torch
    from omegaconf import OmegaConf

    os.environ.setdefault("TALK2DINO_WEIGHT_DIR", str(weight_dir))

    sys_path_added = str(repo_root / "src" / "open_vocabulary_segmentation")
    import sys
    if sys_path_added not in sys.path:
        sys.path.insert(0, sys_path_added)

    default_cfg = OmegaConf.load(
        repo_root / "src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    )
    eval_cfg = OmegaConf.load(repo_root / "src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml")
    cfg = OmegaConf.merge(default_cfg, eval_cfg)

    from models import build_model
    model = build_model(cfg.model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    from mmseg.datasets import COCOStuffDataset
    classnames = list(COCOStuffDataset.CLASSES)
    if classnames and classnames[0] == "background":
        classnames = classnames[1:]
    if len(classnames) != 171:
        raise DiscoveryExecutionError(f"expected exactly 171 ordered COCOStuffDataset class names, got {len(classnames)}")

    text_tokens = model.build_dataset_class_tokens(cfg.evaluate.template, classnames)
    text_embedding = model.build_text_embedding(text_tokens)
    if text_embedding.shape[0] != len(classnames):
        raise DiscoveryExecutionError("text_embedding row count does not match the 171-class list used to build it")

    return model, text_embedding, classnames


# ---------------------------------------------------------------------------
# GT-free per-image processing (Phase A) -- calls into
# semantic_evidence_intervention_features.py for all pure geometry/formula
# logic; never opens a GT file, never imports a GT loader.
# ---------------------------------------------------------------------------

GT_FOOTPRINT_MIN_VALID_FRACTION = 0.75  # [gt_targets.primary_correctness].min_valid_footprint_fraction
GT_FOOTPRINT_MIN_MAJORITY_SHARE = 0.80  # [gt_targets.primary_correctness].min_majority_share
GT_IGNORE_VALUE = 255  # [gt_targets.primary_correctness].ignore_gt_value
AMBIGUOUS_GT = "AMBIGUOUS_GT"  # [gt_targets.primary_correctness].ambiguous_label

_PROPAGATION_CATEGORIES = (
    "beneficial", "harmful", "unchanged_correct", "changed_but_same_correctness", "unchanged_incorrect",
)


def compute_canonical_windows(real_h: int, real_w: int, crop_hw: tuple, stride_hw: tuple) -> list:
    """Delegates entirely to the committed, real-image-proven
    `semantic_evidence_t320_real_smoke.compute_canonical_windows` (already a
    hash-bound `[dependency].smoke_runner` of this authority) instead of
    reimplementing canonical-window tiling locally. The prior local
    reimplementation incorrectly required the canvas to be at least
    crop-sized in both dimensions (`crop_h <= real_h and crop_w <= real_w`);
    real COCO-Stuff images are not guaranteed to satisfy that -- e.g. this
    population's own `000000567197.jpg` is 500x407, narrower than the
    448px crop in height -- and the real, already-proven algorithm never
    required it: it clamps each window to the actual image bounds instead
    of rejecting the image (`semantic_evidence_t320_real_smoke.py:342-365`,
    matching `dinotext_seg.py:78-94`). Job 22640114 already exercised this
    exact clamping behavior successfully on this exact image. Delegating
    here (rather than re-deriving the same arithmetic a second time) is
    what eliminates the reimplementation-drift risk that caused this
    defect in the first place. Any `SmokeRunnerError` the delegation raises
    is translated into `DiscoveryExecutionError` to preserve this module's
    own exception-hygiene convention; the smoke function performs only pure
    arithmetic (no I/O, no torch), so no fatal exception can originate
    inside it."""
    import semantic_evidence_t320_real_smoke as smoke_runner
    try:
        return smoke_runner.compute_canonical_windows(real_h, real_w, crop_hw, stride_hw)
    except smoke_runner.SmokeRunnerError as exc:
        raise DiscoveryExecutionError(str(exc)) from exc


def _build_full_grid(s0, grid_h: int, grid_w: int):
    """s0: [grid_h*grid_w, class_count] row-major tensor. Returns
    {(row,col): PatchInfo} for the entire local window grid."""
    import torch as _torch
    from semantic_evidence_intervention_features import PatchInfo

    top2 = _torch.topk(s0, k=2, dim=1)
    top1_idx = top2.indices[:, 0]
    top2_idx = top2.indices[:, 1]
    top1_val = top2.values[:, 0]
    top2_val = top2.values[:, 1]

    patches_by_rc = {}
    for row in range(grid_h):
        for col in range(grid_w):
            idx = row * grid_w + col
            patches_by_rc[(row, col)] = PatchInfo(
                row=row, col=col, patch_index=idx,
                unary_top1=int(top1_idx[idx].item()), unary_top2=int(top2_idx[idx].item()),
                unary_margin=float((top1_val[idx] - top2_val[idx]).item()),
            )
    return patches_by_rc


def _classify_raw_score_rejection(raw) -> str:
    """Diagnostic-only classifier, never used to decide accept/reject (that
    remains validate_raw_scores's own frozen semantics) -- only to attribute
    an already-rejected row to a specific Gate-A counter. 'non_finite' takes
    priority over 'out_of_range' when a value is both (NaN/Inf are never
    "in range" either, but non-finiteness is the more specific defect)."""
    values = (raw.s_original, raw.s_context, raw.s_region, raw.s_control)
    if any(not math.isfinite(v) for v in values):
        return "non_finite"
    return "out_of_range"


def process_window_gt_free(model, text_embedding, image_tensor, window: tuple, real_h: int, real_w: int,
                            image_id: str, patch_hw: tuple, forward_fn) -> dict:
    """Phase A, one canonical source window. Performs exactly ONE canonical
    E3 forward for this window (the reused ORIGINAL forward, per
    [one_forward_reuse_contract]), then exactly 3 additional E3 forwards
    per accepted anchor (one per non-ORIGINAL intervention view, never
    reused across anchors or across the canonical forward). Returns the
    window's ORIGINAL WindowSnapshot (needed by the caller to build the
    per-image T0/T320 target maps), the assembled GT-free rows WITH full
    per-row scientific detail (raw scores, Model-A feature vector, anchor
    grid position -- everything needed to persist and later re-derive GT
    labels without recomputing any GT-free science), and a Gate-A counter
    breakdown using anchor-level counts throughout (never mixed with image
    counts by this function or any caller)."""
    import semantic_evidence_intervention_features as ifeat

    ordinal, top, left, bottom, right = window
    crop = image_tensor[:, :, top:bottom, left:right]
    window_box = (top, left, bottom, right)

    original_snapshot, ordinary_mask = forward_fn(
        model, crop, text_embedding, window_id=f"{image_id}:{ordinal}:ORIGINAL", window_ordinal=ordinal,
        top=top, left=left, bottom=bottom, right=right, canvas_h=real_h, canvas_w=real_w,
        patch_h=patch_hw[0], patch_w=patch_hw[1],
    )
    canonical_forward_count = 1

    full_grid = _build_full_grid(original_snapshot.s0, original_snapshot.grid_h, original_snapshot.grid_w)
    eligible_patches = [p for (r, c), p in full_grid.items() if ifeat._is_eligible(r, c)]
    boundary_patches = [p for p in eligible_patches if ifeat.is_boundary_patch(p, full_grid)]

    anchors = ifeat.select_anchors(image_id, ordinal, eligible_patches, full_grid)

    query_correctness_rows = []
    propagation_rows = []
    query_row_details = {}        # row_key "ordinal:anchor_id:query_rank" -> detail dict
    propagation_row_details = {}  # row_key "ordinal:anchor_id" -> detail dict (q1 only)
    excluded_no_valid_control = []
    intervention_forward_count = 0

    counters = {
        "attempted_anchor_count": len(anchors),
        "accepted_anchor_count": 0,
        "excluded_no_valid_control_count": 0,
        "structurally_invalid_anchor_count": 0,
        "raw_score_non_finite_count": 0,
        "raw_score_out_of_range_count": 0,
        "derived_score_invalid_count": 0,
        "query_row_count": 0,
        "propagation_row_count": 0,
    }

    for anchor in anchors:
        query = ifeat.construct_queries(anchor)
        anchor_bounds = intervention_core_native_bounds(window_box, anchor.row, anchor.col, ifeat.GRID_SIDE, patch_hw)
        control_center = ifeat.select_control_center(image_id, ordinal, anchor.row, anchor.col, eligible_patches)
        if control_center is None:
            excluded_no_valid_control.append(anchor.patch_index)
            counters["excluded_no_valid_control_count"] += 1
            counters["structurally_invalid_anchor_count"] += 1
            continue
        counters["accepted_anchor_count"] += 1
        control_bounds = intervention_core_native_bounds(window_box, control_center.row, control_center.col, ifeat.GRID_SIDE, patch_hw)

        view_snapshots = {"ORIGINAL": original_snapshot}
        for view in ("CONTEXT_BLURRED", "REGION_BLURRED", "AREA_MATCHED_CONTROL"):
            view_crop = ifeat.build_intervention_view(view, crop, anchor_bounds, control_bounds)
            if not ifeat.verify_unchanged_domain(view, crop, view_crop, anchor_bounds, control_bounds):
                raise DiscoveryExecutionError(f"unchanged-domain verification failed for {image_id}:{ordinal}:{anchor.patch_index}:{view}")
            view_snapshot, _ = forward_fn(
                model, view_crop, text_embedding, window_id=f"{image_id}:{ordinal}:{anchor.patch_index}:{view}",
                window_ordinal=ordinal, top=top, left=left, bottom=bottom, right=right,
                canvas_h=real_h, canvas_w=real_w, patch_h=patch_hw[0], patch_w=patch_hw[1],
            )
            intervention_forward_count += 1
            view_snapshots[view] = view_snapshot

        control_4 = ifeat.control_4_dino_neighbor_cosine_similarity(
            original_snapshot.dino_features[anchor.patch_index],
            _torch_stack_neighbor_dino(original_snapshot.dino_features, anchor.row, anchor.col, original_snapshot.grid_w),
        )
        control_5 = ifeat.control_5_normalized_distance_to_nearest_boundary(anchor.row, anchor.col, boundary_patches)
        control_6 = ifeat.control_6_local_top1_agreement_fraction(anchor, full_grid)

        anchor_produced_a_valid_row = False

        for query_rank, query_class in (("q1", query.q1), ("q2", query.q2)):
            raw = ifeat.RawScores(
                s_original=float(view_snapshots["ORIGINAL"].s0[anchor.patch_index, query_class].item()),
                s_context=float(view_snapshots["CONTEXT_BLURRED"].s0[anchor.patch_index, query_class].item()),
                s_region=float(view_snapshots["REGION_BLURRED"].s0[anchor.patch_index, query_class].item()),
                s_control=float(view_snapshots["AREA_MATCHED_CONTROL"].s0[anchor.patch_index, query_class].item()),
            )
            if not ifeat.validate_raw_scores(raw):
                # structurally invalid row: excluded from all fitting, never laundered
                reason = _classify_raw_score_rejection(raw)
                counters["raw_score_non_finite_count" if reason == "non_finite" else "raw_score_out_of_range_count"] += 1
                continue
            derived = ifeat.compute_derived_features(raw)
            if not ifeat.validate_derived_output(derived):
                counters["derived_score_invalid_count"] += 1
                continue

            control_1 = ifeat.control_1_top1_unary_score(raw.s_original)
            control_2 = ifeat.control_2_top1_minus_top2_margin(anchor.unary_margin)
            control_3 = ifeat.control_3_unary_entropy(view_snapshots["ORIGINAL"].s0[anchor.patch_index])
            control_8 = ifeat.control_8_area_matched_control_change(derived.control_change)
            feature_vector = [control_1, control_2, control_3, control_4, control_5, control_6, control_8]

            row = ifeat.assemble_query_correctness_row(
                image_id, ordinal, str(anchor.patch_index), query_rank, query_class,
                control_1, control_2, control_3, control_4, control_5, control_6, control_8,
                derived.intervention_reliability,
            )
            query_correctness_rows.append(row)
            counters["query_row_count"] += 1
            anchor_produced_a_valid_row = True

            row_key = f"{ordinal}:{anchor.patch_index}:{query_rank}"
            query_row_details[row_key] = {
                "query_class": query_class,
                "anchor_row": anchor.row, "anchor_col": anchor.col,
                "raw_scores": {
                    "s_original": raw.s_original, "s_context": raw.s_context,
                    "s_region": raw.s_region, "s_control": raw.s_control,
                },
                "features": feature_vector,
                "intervention_reliability": derived.intervention_reliability,
            }

            if query_rank == "q1":
                prop_row = ifeat.assemble_propagation_outcome_row(
                    image_id, ordinal, str(anchor.patch_index), query_class,
                    control_1, control_2, control_3, control_4, control_5, control_6, control_8,
                    derived.intervention_reliability,
                )
                propagation_rows.append(prop_row)
                counters["propagation_row_count"] += 1

                prop_key = f"{ordinal}:{anchor.patch_index}"
                propagation_row_details[prop_key] = {
                    "query_class": query_class,
                    "anchor_row": anchor.row, "anchor_col": anchor.col,
                    "features": feature_vector,
                    "intervention_reliability": derived.intervention_reliability,
                }

        if not anchor_produced_a_valid_row:
            counters["structurally_invalid_anchor_count"] += 1

    return {
        "window_ordinal": ordinal, "window_box": (top, left, bottom, right),
        "original_snapshot": original_snapshot,
        "query_correctness_rows": query_correctness_rows,
        "propagation_rows": propagation_rows,
        "query_row_details": query_row_details,
        "propagation_row_details": propagation_row_details,
        "excluded_no_valid_control_anchor_ids": excluded_no_valid_control,
        "canonical_forward_count": canonical_forward_count,
        "intervention_forward_count": intervention_forward_count,
        "counters": counters,
    }


def _torch_stack_neighbor_dino(dino_features, row: int, col: int, grid_w: int):
    import torch as _torch
    idxs = [
        (row + dr) * grid_w + (col + dc)
        for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)
    ]
    return _torch.stack([dino_features[i] for i in idxs], dim=0)


def compute_gt_majority(footprint_rows: list, ignore_value: int = GT_IGNORE_VALUE,
                         min_valid_footprint_fraction: float = GT_FOOTPRINT_MIN_VALID_FRACTION,
                         min_majority_share: float = GT_FOOTPRINT_MIN_MAJORITY_SHARE):
    """[gt_targets.primary_correctness]: footprint_rows is a list of rows
    (each a list of int class ids, 255=ignore) covering the anchor patch's
    14x14 pixel footprint. Returns (AMBIGUOUS_GT, None) or
    (majority_class:int, majority_class). Pure Python -- no torch/numpy
    dependency, so fake-provider tests can call it directly with plain
    nested lists."""
    from collections import Counter

    total = 0
    valid = 0
    counts = Counter()
    for r in footprint_rows:
        for v in r:
            total += 1
            if v != ignore_value:
                valid += 1
                counts[v] += 1
    if total == 0:
        return AMBIGUOUS_GT, None
    valid_fraction = valid / total
    if valid_fraction < min_valid_footprint_fraction or valid == 0:
        return AMBIGUOUS_GT, None
    majority_class, majority_count = counts.most_common(1)[0]
    majority_share = majority_count / valid
    if majority_share < min_majority_share:
        return AMBIGUOUS_GT, None
    return majority_class, majority_class


def _center_pixel_class(footprint_rows: list) -> int:
    """Returns the single center-pixel value of a patch-sized footprint
    (row-major nested list). Used for T0/T320 predicted-class identifiers,
    which are always well-defined at every pixel -- unlike GT majority
    voting, there is no ignore/ambiguous concept to apply here."""
    h = len(footprint_rows)
    w = len(footprint_rows[0])
    return footprint_rows[h // 2][w // 2]


def compute_propagation_category(t0_predicted_class: int, t320_predicted_class: int, gt_majority_class: int) -> str:
    """[gt_targets.propagation_outcome].*_definition, evaluated mechanically
    over the exact frozen 5-category MECE taxonomy. Caller must have
    already excluded AMBIGUOUS_GT anchors before calling this."""
    t0_correct = t0_predicted_class == gt_majority_class
    t320_correct = t320_predicted_class == gt_majority_class
    if not t0_correct and t320_correct:
        return "beneficial"
    if t0_correct and not t320_correct:
        return "harmful"
    if t0_correct and t320_correct:
        return "unchanged_correct"
    # both wrong
    if t0_predicted_class != t320_predicted_class:
        return "changed_but_same_correctness"
    return "unchanged_incorrect"


def _round_half_away_from_zero(x: float) -> int:
    """Explicit tie rule: 0.5 always rounds away from zero. Never Python's
    built-in `round()` (banker's rounding, round-half-to-even), which is a
    different, unstated convention -- this pipeline pins one explicitly.
    All values this module ever calls it with are >= 0 (native pixel/patch
    coordinates), but both signs are handled for completeness."""
    if x >= 0:
        return math.floor(x + 0.5)
    return math.ceil(x - 0.5)


def _input_tokenization_inverse_point(canvas_index: int, native_size: int, canvas_size: int) -> int:
    """Selected scientific contract (native-window anchor-content geometry,
    successor to the historical always-448 convention): maps a PIXEL INDEX
    (a discrete, indexed position -- e.g. the historical patch-center pixel
    `i*14+7`) in the fixed 448-wide backbone-tokenization canvas back to a
    native window pixel index, by inverting the exact, real resize
    `model.image_transforms` applies before tokenization
    (`src/open_vocabulary_segmentation/models/dinotext/dinotext.py:73-77`,
    `T.Resize((resize_dim, resize_dim))`, no interpolation/antialias
    kwargs, on an already-tensor input -- mirrored at
    `semantic_evidence_e3_adapter.py:244-246`). Empirically confirmed
    (this repo's own production venv, torch 2.12.0 / torchvision 0.27.0)
    numerically identical (max abs diff ~2e-6) to `torch.nn.functional.
    interpolate(mode="bilinear", align_corners=False)` for a tensor input.

    This is a PIXEL-INDEX (point) mapping, using the standard half-pixel
    formula `x_native = (x_canvas + 0.5) * (native/canvas) - 0.5`, distinct
    from `_input_tokenization_inverse_boundary` below (a half-open
    INTERVAL EDGE mapping, which uses pure linear boundary-space scaling
    with no +/-0.5 offset -- the two are not interchangeable; a boundary is
    not a pixel index). This distinction, and the choice of
    round-half-away-from-zero for discretization, are explicit contract
    choices, not automatic consequences of bilinear interpolation.

    Deliberately NOT self-consistent with `semantic_evidence_t320.
    internal_upsample`'s own align_corners=True grid->native OUTPUT
    placement (dinotext.py:340-346): that is a different, later transform
    governing prediction *rendering*, not anchor *content* -- conflating
    the two was considered and rejected because it does not extend the
    historical `i*14+7` patch-center meaning continuously as native size
    passes through 448 (see this pipeline's geometry-contract records).

    At native_size == canvas_size this reduces EXACTLY to `canvas_index`
    (scale factor 1), matching the historical, pre-clamping-repair value
    bit-for-bit with no special-casing."""
    if native_size == canvas_size:
        return canvas_index
    x = (canvas_index + 0.5) * native_size / canvas_size - 0.5
    point = _round_half_away_from_zero(x)
    return max(0, min(native_size - 1, point))


def _input_tokenization_inverse_boundary(canvas_edge: int, native_size: int, canvas_size: int) -> int:
    """Selected scientific contract: maps a half-open INTERVAL EDGE (e.g.
    one boundary of the historical patch interval `[i*14,(i+1)*14)`, or of
    the frozen intervention core interval) in the fixed 448-wide backbone-
    tokenization canvas back to a native window boundary, via PURE LINEAR
    boundary-space scaling: `x_native = canvas_edge * (native/canvas)`.

    A boundary is not a pixel index (it is the line between pixel k-1 and
    pixel k), so it does not take the +0.5/-0.5 pixel-center offset that
    `_input_tokenization_inverse_point` uses -- boundary-space is already
    linear and origin-preserving between the two images. Empirically
    verified (this repo's own production venv) against the REAL `T.Resize`
    forward operation: for synthetic step-function boundaries at native
    positions 100/205/300/393 (native_size=407), the observed 0.5-crossing
    of the real resized output agreed with this pure linear-scaling
    prediction to within <0.4 pixels in every case -- distinctly better
    agreement than the pixel-index formula would give for a boundary
    (confirmed by direct comparison during this pipeline's geometry-
    contract review).

    This computes a ROUNDED GEOMETRIC BOUNDARY POSITION, not a claim of
    the exact nonzero bilinear-kernel support (which, for a 2-tap bilinear
    kernel, extends up to ~1 native pixel further on each side than this
    rounded boundary) -- align_corners=True output upsampling remains a
    separate, unrelated transform and is not what this function computes.

    At native_size == canvas_size this reduces EXACTLY to `canvas_edge`
    (scale factor 1), matching the historical, pre-clamping-repair
    interval bit-for-bit with no special-casing."""
    if native_size == canvas_size:
        return canvas_edge
    x = canvas_edge * native_size / canvas_size
    edge = _round_half_away_from_zero(x)
    return max(0, min(native_size, edge))


def t0_t320_anchor_pixel_bounds(window_box: tuple, anchor_row: int, anchor_col: int, grid_side: int,
                                 patch_hw: tuple) -> tuple:
    """Phase A: the single native pixel (returned as a 1x1 `crop_raster`-
    compatible bounds tuple) representing anchor grid cell (anchor_row,
    anchor_col)'s historical patch-center pixel (`i*patch_h + patch_h//2`),
    mapped through the selected input-tokenization-inverse point contract
    (see `_input_tokenization_inverse_point`). Uses the WINDOW's own native
    height/width (computed from `window_box`), never the full image size.
    Fails closed if the mapped point ever falls outside the window's own
    native extent (structurally impossible for the frozen eligible anchor
    range on any real discovery-population image, but guarded rather than
    assumed)."""
    window_top, window_left, window_bottom, window_right = window_box
    native_h = window_bottom - window_top
    native_w = window_right - window_left
    patch_h, patch_w = patch_hw
    canvas_h = grid_side * patch_h
    canvas_w = grid_side * patch_w

    center_row_448 = anchor_row * patch_h + patch_h // 2
    center_col_448 = anchor_col * patch_w + patch_w // 2
    local_row = _input_tokenization_inverse_point(center_row_448, native_h, canvas_h)
    local_col = _input_tokenization_inverse_point(center_col_448, native_w, canvas_w)
    _require(
        0 <= local_row < native_h and 0 <= local_col < native_w,
        f"t0_t320_anchor_pixel_bounds: mapped point ({local_row},{local_col}) falls outside "
        f"the window's own native extent ({native_h},{native_w}) for anchor ({anchor_row},{anchor_col})",
    )
    top = window_top + local_row
    left = window_left + local_col
    return top, left, top + 1, left + 1


def footprint_pixel_bounds(window_box: tuple, anchor_row: int, anchor_col: int, patch_hw: tuple, grid_side: int) -> tuple:
    """Phase B (GT-only): maps a single anchor patch's majority-vote
    footprint REGION through the canonical window geometry to native image
    pixel coordinates -- see [gt_targets.primary_correctness].
    footprint_mapping. Uses the WINDOW's own native height/width (computed
    from `window_box`), never the full image size.

    Exact, unchanged behavior when the window's native size equals the
    fixed `grid_side*patch_h` x `grid_side*patch_w` backbone-tokenization
    canvas (the historical, always-448, unclamped case): the ViT patch
    embedding tokenizes that canvas via an EXACT, non-overlapping
    `[row*patch_h, (row+1)*patch_h)` block partition (ordinary
    Conv2d/unfold patchify -- not an interpolation of any kind), so native
    pixels equal canvas pixels 1:1 and the footprint is exactly that block.

    For a genuinely clamped window (native size != the fixed canvas, in
    either axis independently): each of the interval's two edges is mapped
    via the selected input-tokenization-inverse boundary contract (see
    `_input_tokenization_inverse_boundary`), per axis. Fails closed
    (raises DiscoveryExecutionError) if the mapped region ever collapses
    to zero width/height (does not occur for any real discovery-population
    image at the frozen eligible anchor range, but guarded rather than
    assumed)."""
    window_top, window_left, window_bottom, window_right = window_box
    native_h = window_bottom - window_top
    native_w = window_right - window_left
    patch_h, patch_w = patch_hw
    canvas_h = grid_side * patch_h
    canvas_w = grid_side * patch_w

    top448, bottom448 = anchor_row * patch_h, (anchor_row + 1) * patch_h
    left448, right448 = anchor_col * patch_w, (anchor_col + 1) * patch_w

    top = _input_tokenization_inverse_boundary(top448, native_h, canvas_h)
    bottom = _input_tokenization_inverse_boundary(bottom448, native_h, canvas_h)
    left = _input_tokenization_inverse_boundary(left448, native_w, canvas_w)
    right = _input_tokenization_inverse_boundary(right448, native_w, canvas_w)
    _require(
        bottom > top and right > left,
        f"footprint_pixel_bounds: mapped footprint collapsed to an empty region for anchor "
        f"({anchor_row},{anchor_col}) on a window with native size ({native_h}, {native_w})",
    )
    return window_top + top, window_left + left, window_top + bottom, window_left + right


def intervention_core_native_bounds(window_box: tuple, anchor_row: int, anchor_col: int, grid_side: int,
                                      patch_hw: tuple) -> tuple:
    """Discovery-local wrapper (NOT in the protected feature module): maps
    the protected, frozen `semantic_evidence_intervention_features.
    core_pixel_bounds`'s 448-canvas 3x3-patch core interval to the
    WINDOW's own native pixel coordinates, via the same selected
    input-tokenization-inverse boundary contract `footprint_pixel_bounds`
    uses (`_input_tokenization_inverse_boundary`), so ordinary tensor
    slicing against the native, pre-resize `crop` tensor in
    `process_window_gt_free` cannot silently truncate the intended core
    for a clamped window. `core_pixel_bounds` itself is never edited or
    reimplemented here -- only its already-computed 448-canvas interval is
    remapped. Uses the WINDOW's own native height/width (from
    `window_box`), never the full image size. Returns LOCAL (window-
    relative) bounds, matching what `core_pixel_bounds` itself returns
    (both are consumed directly against the window-local `crop` tensor,
    never with an absolute window offset added -- see
    `ifeat.build_intervention_view`'s call sites)."""
    import semantic_evidence_intervention_features as ifeat

    window_top, window_left, window_bottom, window_right = window_box
    native_h = window_bottom - window_top
    native_w = window_right - window_left
    patch_h, patch_w = patch_hw
    canvas_h = grid_side * patch_h
    canvas_w = grid_side * patch_w

    top448, left448, bottom448, right448 = ifeat.core_pixel_bounds(anchor_row, anchor_col)

    top = _input_tokenization_inverse_boundary(top448, native_h, canvas_h)
    bottom = _input_tokenization_inverse_boundary(bottom448, native_h, canvas_h)
    left = _input_tokenization_inverse_boundary(left448, native_w, canvas_w)
    right = _input_tokenization_inverse_boundary(right448, native_w, canvas_w)
    _require(
        bottom > top and right > left,
        f"intervention_core_native_bounds: mapped core collapsed to an empty region for anchor "
        f"({anchor_row},{anchor_col}) on a window with native size ({native_h}, {native_w})",
    )
    return top, left, bottom, right


def crop_raster(raster: list, bounds: tuple) -> list:
    top, left, bottom, right = bounds
    return [row[left:right] for row in raster[top:bottom]]


def compute_gt_label_content_digest(artifact: dict) -> str:
    """Recomputable from ONLY the persisted gt_dependent_labels artifact
    dict (with the digest field itself excluded) -- mirrors
    compute_gt_free_content_digest's same reasoning, so digest relations
    can be independently re-verified from a reloaded JSON file at any
    point (staging, reopen, install, public verify-result)."""
    content = {k: v for k, v in artifact.items() if k != "gt_dependent_label_digest"}
    return _content_digest(content)


# ---------------------------------------------------------------------------
# GT-free / GT-label phase event log (for testable phase-ordering proof)
# ---------------------------------------------------------------------------

class PhaseEventLog:
    def __init__(self):
        self.events: list = []

    def record(self, event: str) -> None:
        self.events.append(event)

    def gt_accessed_before(self, start_event: str, boundary_event: str) -> bool:
        """True iff a gt_access* event was recorded strictly between
        `start_event` and `boundary_event` (both scoped to the same image,
        so one image's legitimate Phase B access never poisons the isolation
        check for a later image sharing this same log)."""
        start_index = self.events.index(start_event) if start_event in self.events else 0
        boundary_index = self.events.index(boundary_event) if boundary_event in self.events else len(self.events)
        return any(e.startswith("gt_access") for e in self.events[start_index:boundary_index])


# ---------------------------------------------------------------------------
# Checkpoint metadata (closed schema per artifact_contract.checkpoint_metadata)
# ---------------------------------------------------------------------------

_CHECKPOINT_REQUIRED_FIELDS = frozenset({
    "schema", "complete", "final", "mode", "discovery_image_count",
    "completed_image_ids_ordered", "completed_image_prefix_digest",
    "identity_and_dependency_bindings", "checkpoint_array_hashes",
})


def build_checkpoint_metadata(mode: str, completed_image_ids_ordered: list, discovery_ids: list,
                               checkpoint_array_hashes: dict, identity_bindings: dict,
                               complete: bool, final: bool) -> dict:
    _require(all(i in discovery_ids for i in completed_image_ids_ordered), "completed IDs must be a subset of the discovery population")
    _require(completed_image_ids_ordered == discovery_ids[:len(completed_image_ids_ordered)],
              "completed_image_ids_ordered must be an exact prefix of the frozen discovery order")
    digest = hashlib.sha256("".join(i + "\n" for i in completed_image_ids_ordered).encode("ascii")).hexdigest()
    meta = {
        "schema": {"name": "semantic_evidence_discovery_checkpoint", "version": "1.0.0"},
        "complete": complete, "final": final, "mode": mode,
        "discovery_image_count": len(discovery_ids),
        "completed_image_ids_ordered": completed_image_ids_ordered,
        "completed_image_prefix_digest": digest,
        "identity_and_dependency_bindings": identity_bindings,
        "checkpoint_array_hashes": checkpoint_array_hashes,
    }
    validate_checkpoint_metadata(meta, discovery_ids)  # construction must never produce an invalid checkpoint
    return meta


def validate_checkpoint_metadata(meta: dict, discovery_ids: list) -> None:
    _require(isinstance(meta, dict), "checkpoint metadata must be a dict")
    actual_keys = frozenset(meta.keys())
    _require(actual_keys == _CHECKPOINT_REQUIRED_FIELDS,
              f"checkpoint metadata key mismatch: missing={_CHECKPOINT_REQUIRED_FIELDS - actual_keys}, extra={actual_keys - _CHECKPOINT_REQUIRED_FIELDS}")
    _require(isinstance(meta["complete"], bool), "checkpoint.complete must be a bool")
    _require(isinstance(meta["final"], bool), "checkpoint.final must be a bool")
    _require(meta["mode"] in ("fresh", "resume", "finalize_only"), "checkpoint.mode must be a valid lifecycle mode")
    completed = meta["completed_image_ids_ordered"]
    _require(isinstance(completed, list), "completed_image_ids_ordered must be a list")
    _require(len(completed) == len(set(completed)), "completed_image_ids_ordered must not contain duplicates")
    _require(completed == discovery_ids[:len(completed)], "completed_image_ids_ordered must be an exact prefix of the frozen discovery order")
    _require(not any(i not in discovery_ids for i in completed), "completed_image_ids_ordered must not contain a non-discovery (e.g. holdout) ID")
    digest = hashlib.sha256("".join(i + "\n" for i in completed).encode("ascii")).hexdigest()
    _require(digest == meta["completed_image_prefix_digest"], "completed_image_prefix_digest mismatch")
    if meta["final"]:
        _require(meta["complete"], "final=true requires complete=true")
        _require(len(completed) == meta["discovery_image_count"], "final checkpoint requires all discovery images completed")


# ---------------------------------------------------------------------------
# Statistical model fitting (the four frozen logistic fits, discovery-only)
# ---------------------------------------------------------------------------

_MODEL_SETTINGS = dict(C=1.0, penalty="l2", class_weight="balanced", solver="liblinear", max_iter=1000, random_state=0)


_MODEL_A_FEATURE_ORDER = (1, 2, 3, 4, 5, 6, 8)  # control ids, frozen [[statistics.model_fits]] feature_order
_MODEL_B_EXTRA_FEATURE = "intervention_reliability"


def fit_logistic_model(X: list, y: list, *, task_id: str, model_id: str, model_b: bool,
                        row_keys: list) -> dict:
    """Fits exactly one of the four frozen logistic-regression models on the
    discovery partition only. Never touches holdout data (the caller is
    responsible for never passing holdout rows in). Standardizes features
    using discovery-row statistics only, per [statistics.model_settings].

    Fails closed (raises DiscoveryExecutionError) on: empty population,
    a one-class target (both classes are required to fit), a row-key/X/y
    length mismatch, a duplicate row key, a wrong feature-vector length for
    the declared model (7 for Model A, 8 for Model B), any non-finite
    feature value, solver non-convergence, and any non-finite fitted
    coefficient/intercept or non-finite prediction on the fitted rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    _require(model_id in ("model_a", "model_b"), f"unknown model_id: {model_id!r}")
    _require(model_b == (model_id == "model_b"), "model_b flag must match model_id")
    expected_feature_count = len(_MODEL_A_FEATURE_ORDER) + (1 if model_b else 0)

    _require(len(X) == len(y) == len(row_keys), "X, y, and row_keys must have matching length")
    _require(len(row_keys) == len(set(row_keys)), "duplicate row_key passed to fit_logistic_model")
    _require(len(X) >= 2, f"insufficient rows to fit {task_id}/{model_id}: got {len(X)}")

    for i, row in enumerate(X):
        _require(len(row) == expected_feature_count,
                  f"{task_id}/{model_id} row {row_keys[i]!r} has {len(row)} features, expected {expected_feature_count}")
        _require(all(math.isfinite(v) for v in row), f"{task_id}/{model_id} row {row_keys[i]!r} contains a non-finite feature value")

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    _require(len(set(y_arr.tolist())) == 2, f"degenerate label set for {task_id}/{model_id}: both classes are required to fit a logistic model")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    clf = LogisticRegression(**_MODEL_SETTINGS)
    clf.fit(X_scaled, y_arr)

    n_iter = int(clf.n_iter_[0]) if hasattr(clf, "n_iter_") else None
    if n_iter is not None:
        _require(n_iter < _MODEL_SETTINGS["max_iter"], f"{task_id}/{model_id} solver did not converge within max_iter={_MODEL_SETTINGS['max_iter']} (n_iter={n_iter})")

    coefficients = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])
    _require(all(math.isfinite(c) for c in coefficients), f"{task_id}/{model_id} produced non-finite coefficients")
    _require(math.isfinite(intercept), f"{task_id}/{model_id} produced a non-finite intercept")

    predictions = clf.predict_proba(X_scaled)[:, 1]
    _require(all(math.isfinite(p) for p in predictions), f"{task_id}/{model_id} produced non-finite predictions on its own fitted rows")

    class_counts = {int(c): int((y_arr == c).sum()) for c in sorted(set(y_arr.tolist()))}
    feature_order_labels = [f"control_{c}" for c in _MODEL_A_FEATURE_ORDER] + ([_MODEL_B_EXTRA_FEATURE] if model_b else [])

    summary = {
        "task_id": task_id, "model_id": model_id,
        "feature_order": feature_order_labels,
        "row_count": int(X_arr.shape[0]),
        "class_counts": class_counts,
        "coefficients": coefficients,
        "intercept": intercept,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "solver": _MODEL_SETTINGS["solver"],
        "converged": True,
        "converged_n_iter": n_iter,
    }
    summary["fit_digest"] = _content_digest({
        "task_id": task_id, "model_id": model_id, "feature_order": feature_order_labels,
        "row_keys": sorted(row_keys), "row_count": summary["row_count"], "class_counts": class_counts,
        "coefficients": coefficients, "intercept": intercept,
    })
    return summary


# ---------------------------------------------------------------------------
# Discovery gates (A and E only; B/C/D remain NOT_COMPUTABLE_WITHOUT_HOLDOUT)
# ---------------------------------------------------------------------------

def compute_gate_a(total_anchors: int, structurally_invalid_anchor_count: int, non_finite_scores_count: int,
                    gt_dependent_intervention_records: int) -> dict:
    """[gate_contract.A]: min_structural_validity_fraction=0.95,
    max_gt_dependent_intervention_records=0, max_non_finite_scores=0.

    Unit discipline: `total_anchors` and `structurally_invalid_anchor_count`
    are both ANCHOR-level counts (attempted_anchor_count and
    structurally_invalid_anchor_count aggregated across every window of
    every processed image -- see process_window_gt_free's own per-window
    counters). This function never accepts an image count in either
    argument, so it is structurally impossible for a caller to divide an
    anchor count by an image count here -- the parameter name itself is the
    guard. `non_finite_scores_count` must be an observed count (see
    raw_score_non_finite_count in process_window_gt_free's counters), never
    a literal constant."""
    _require(total_anchors >= 0, "total_anchors must be non-negative")
    _require(structurally_invalid_anchor_count >= 0, "structurally_invalid_anchor_count must be non-negative")
    _require(structurally_invalid_anchor_count <= total_anchors, "structurally_invalid_anchor_count cannot exceed total_anchors")
    valid_fraction = 1.0 if total_anchors == 0 else (total_anchors - structurally_invalid_anchor_count) / total_anchors
    passed = (
        valid_fraction >= 0.95
        and gt_dependent_intervention_records <= 0
        and non_finite_scores_count <= 0
    )
    return {
        "gate": "A", "computable_under_this_contract": True,
        "total_anchors": total_anchors,
        "structurally_invalid_anchor_count": structurally_invalid_anchor_count,
        "structural_validity_fraction": valid_fraction,
        "gt_dependent_intervention_records": gt_dependent_intervention_records,
        "non_finite_scores_count": non_finite_scores_count,
        "passed": passed,
    }


def compute_gate_e(discovery_wall_time_seconds: float) -> dict:
    """[gate_contract.E]: max_h100_hours_96_image_feasibility_run=2.0. Uses
    only Slurm elapsed time as non-scientific operational metadata,
    projected for the 96-image feasibility run; never uses smoke profiling."""
    projected_96_image_hours = (discovery_wall_time_seconds / 3600.0) * (96.0 / 48.0)
    passed = projected_96_image_hours <= 2.0
    return {
        "gate": "E", "computable_under_this_contract": True,
        "discovery_wall_time_seconds": discovery_wall_time_seconds,
        "projected_96_image_hours": projected_96_image_hours,
        "passed": passed,
    }


def gate_b_c_d_not_computable() -> dict:
    return {
        "B": {"gate": "B", "computable_under_this_contract": False, "status": "NOT_COMPUTABLE_WITHOUT_HOLDOUT"},
        "C": {"gate": "C", "computable_under_this_contract": False, "status": "NOT_COMPUTABLE_WITHOUT_HOLDOUT"},
        "D": {"gate": "D", "computable_under_this_contract": False, "status": "NOT_COMPUTABLE_WITHOUT_HOLDOUT"},
    }


def discovery_decision(gate_a: dict, gate_e: dict, degenerate_sample: bool) -> str:
    """[gate_contract.outcome].discovery_decision_definition, evaluated
    mechanically. This is NEVER the predecessor's gates.outcome
    proceed/stop decision -- only a discovery-phase readiness signal."""
    if degenerate_sample:
        return "DISCOVERY_INCONCLUSIVE"
    if not gate_e["passed"]:
        return "DISCOVERY_INCONCLUSIVE"
    if not gate_a["passed"]:
        return "DISCOVERY_STOP_GATE_A_FAILED"
    return "DISCOVERY_FEATURES_READY_FOR_HOLDOUT_AUTHORITY"


# ---------------------------------------------------------------------------
# Layer 3: artifact validators
# ---------------------------------------------------------------------------

_GT_FREE_REQUIRED_FIELDS = frozenset({
    "schema", "image_id", "window_ids_ordered", "anchor_ids_ordered",
    "query_ranks", "intervention_view_scores", "control_features",
    "intervention_reliability", "structural_validity_flags", "gt_free_feature_digest",
})

_GT_LABEL_REQUIRED_FIELDS = frozenset({
    "schema", "image_id", "query_correctness_labels", "propagation_outcome_categories",
    "ambiguous_gt_anchor_ids", "gt_dependent_label_digest",
})

_FINAL_RESULT_REQUIRED_FIELDS = frozenset({
    "schema", "complete", "final", "identity_and_dependency_bindings",
    "population_binding", "implementation_binding", "discovery_image_ids_ordered",
    "gt_free_feature_digest", "gt_dependent_label_digest", "scientific_collection_digest",
    "model_fit_summaries", "primary_metrics", "bootstrap_intervals", "gate_inputs",
    "gate_pass_fail", "final_frozen_outcome", "holdout_accessed", "full_evaluation_accessed",
})

_DECISION_REQUIRED_FIELDS = frozenset({
    "schema", "gate_a_discovery_side", "gate_e_discovery_side", "gate_b_status",
    "gate_c_status", "gate_d_status", "overall_gates_outcome_status",
    "discovery_decision", "next_step",
})


def _validate_closed(d: dict, required: frozenset, label: str) -> None:
    _require(isinstance(d, dict), f"{label} must be a dict")
    actual = frozenset(d.keys())
    _require(actual == required, f"{label} key mismatch: missing={required - actual}, extra={actual - required}")


_PROPAGATION_TAXONOMY = frozenset({
    "beneficial", "harmful", "unchanged_correct", "changed_but_same_correctness", "unchanged_incorrect",
})
_QUERY_RANK_SET = ("q1", "q2")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_RE = re.compile(r"^[0-9]{12}$")


def _is_strict_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite_number(v) -> bool:
    return _is_strict_int(v) or (isinstance(v, float) and math.isfinite(v))


def _require_schema_block(schema, label: str) -> None:
    _require(isinstance(schema, dict), f"{label}.schema must be a dict")
    _require(frozenset(schema.keys()) == {"name", "version"}, f"{label}.schema must have exactly name/version keys")
    _require(isinstance(schema["name"], str) and schema["name"] != "", f"{label}.schema.name must be a non-empty str")
    _require(isinstance(schema["version"], str) and schema["version"] != "", f"{label}.schema.version must be a non-empty str")


def _require_window_box(box, label: str) -> None:
    _require(isinstance(box, list) and len(box) == 4, f"{label} must be a 4-element list")
    _require(all(_is_strict_int(v) and v >= 0 for v in box), f"{label} entries must be non-negative ints")
    top, left, bottom, right = box
    _require(bottom > top and right > left, f"{label} must have bottom>top and right>left")


def _require_query_correctness_key(key: str, window_boxes: dict, label: str) -> tuple:
    parts = key.split(":")
    _require(len(parts) == 3, f"{label} key {key!r} must have exactly 3 colon-separated parts (ordinal:anchor_id:query_rank)")
    ordinal_str, anchor_id, query_rank = parts
    _require(ordinal_str in window_boxes, f"{label} key {key!r} references an undeclared window ordinal")
    _require(anchor_id.isdigit(), f"{label} key {key!r} has a non-numeric anchor_id")
    _require(query_rank in _QUERY_RANK_SET, f"{label} key {key!r} has an invalid query_rank {query_rank!r}")
    return ordinal_str, anchor_id, query_rank


def _require_propagation_key(key: str, window_boxes: dict, label: str) -> tuple:
    parts = key.split(":")
    _require(len(parts) == 2, f"{label} key {key!r} must have exactly 2 colon-separated parts (ordinal:anchor_id) -- q1-only, never q2, never a 3-part row key")
    ordinal_str, anchor_id = parts
    _require(ordinal_str in window_boxes, f"{label} key {key!r} references an undeclared window ordinal")
    _require(anchor_id.isdigit(), f"{label} key {key!r} has a non-numeric anchor_id")
    return ordinal_str, anchor_id


def _require_anchor_position(row: int, col: int, label: str) -> None:
    _require(_is_strict_int(row) and _is_strict_int(col), f"{label} anchor_row/anchor_col must be ints")
    _require(1 <= row <= 30 and 1 <= col <= 30, f"{label} anchor_row/anchor_col must be in the eligible [1,30] range")


def _require_reliability(v, label: str) -> None:
    _require(_is_finite_number(v) and not _is_strict_int(v), f"{label} intervention_reliability must be a finite float")
    _require(0.0 <= float(v) <= 1.0, f"{label} intervention_reliability must be in [0,1]")


def _require_features_vector(features, label: str) -> None:
    _require(isinstance(features, list) and len(features) == len(_MODEL_A_FEATURE_ORDER),
              f"{label} features must be a {len(_MODEL_A_FEATURE_ORDER)}-element list")
    _require(all(_is_finite_number(v) for v in features), f"{label} features must all be finite numbers")


def validate_gt_free_artifact(d: dict) -> None:
    _validate_closed(d, _GT_FREE_REQUIRED_FIELDS, "gt_free_features artifact")
    _require_schema_block(d["schema"], "gt_free_features")
    _require(isinstance(d["image_id"], str) and _IMAGE_ID_RE.match(d["image_id"]), "gt_free_features.image_id must be a 12-digit str")

    _require(isinstance(d["query_ranks"], list) and d["query_ranks"] == list(_QUERY_RANK_SET),
              "gt_free_features.query_ranks must be exactly ['q1','q2']")

    cf = d["control_features"]
    _require(isinstance(cf, dict) and frozenset(cf.keys()) == {"window_boxes", "query_correctness", "propagation_outcome"},
              "gt_free_features.control_features must have exactly window_boxes/query_correctness/propagation_outcome")

    window_boxes = cf["window_boxes"]
    _require(isinstance(window_boxes, dict), "control_features.window_boxes must be a dict")
    for k, box in window_boxes.items():
        _require(k.isdigit(), f"control_features.window_boxes key {k!r} must be a decimal ordinal string")
        _require_window_box(box, f"control_features.window_boxes[{k!r}]")
    expected_window_ids = [f"{d['image_id']}:{k}" for k in sorted(window_boxes.keys(), key=int)]
    _require(d["window_ids_ordered"] == expected_window_ids,
              "gt_free_features.window_ids_ordered must exactly match control_features.window_boxes in ascending ordinal order")

    qc = cf["query_correctness"]
    _require(isinstance(qc, dict), "control_features.query_correctness must be a dict")
    anchor_row_counts = {}
    for key, detail in qc.items():
        ordinal_str, anchor_id, query_rank = _require_query_correctness_key(key, window_boxes, "control_features.query_correctness")
        _require(isinstance(detail, dict) and frozenset(detail.keys()) == {"query_class", "anchor_row", "anchor_col", "features", "intervention_reliability"},
                  f"control_features.query_correctness[{key!r}] has the wrong key set")
        _require(_is_strict_int(detail["query_class"]) and detail["query_class"] >= 0,
                  f"control_features.query_correctness[{key!r}].query_class must be a non-negative int")
        _require_anchor_position(detail["anchor_row"], detail["anchor_col"], f"control_features.query_correctness[{key!r}]")
        _require_features_vector(detail["features"], f"control_features.query_correctness[{key!r}]")
        _require_reliability(detail["intervention_reliability"], f"control_features.query_correctness[{key!r}]")
        anchor_key = f"{ordinal_str}:{anchor_id}"
        anchor_row_counts.setdefault(anchor_key, set()).add(query_rank)
    for anchor_key, ranks_seen in anchor_row_counts.items():
        _require(ranks_seen <= set(_QUERY_RANK_SET) and len(ranks_seen) <= 2,
                  f"anchor {anchor_key!r} has an invalid query_rank set {ranks_seen!r} (at most one q1 row and one q2 row)")

    po = cf["propagation_outcome"]
    _require(isinstance(po, dict), "control_features.propagation_outcome must be a dict")
    for key, detail in po.items():
        _require_propagation_key(key, window_boxes, "control_features.propagation_outcome")
        _require(isinstance(detail, dict) and frozenset(detail.keys()) == {
            "query_class", "anchor_row", "anchor_col", "features", "intervention_reliability",
            "t0_predicted_class", "t320_predicted_class",
        }, f"control_features.propagation_outcome[{key!r}] has the wrong key set")
        _require(_is_strict_int(detail["query_class"]) and detail["query_class"] >= 0,
                  f"control_features.propagation_outcome[{key!r}].query_class must be a non-negative int")
        _require_anchor_position(detail["anchor_row"], detail["anchor_col"], f"control_features.propagation_outcome[{key!r}]")
        _require_features_vector(detail["features"], f"control_features.propagation_outcome[{key!r}]")
        _require_reliability(detail["intervention_reliability"], f"control_features.propagation_outcome[{key!r}]")
        for cls_field in ("t0_predicted_class", "t320_predicted_class"):
            _require(_is_strict_int(detail[cls_field]) and 0 <= detail[cls_field] <= 170,
                      f"control_features.propagation_outcome[{key!r}].{cls_field} must be an int in [0,170]")
        # q1-only, by construction of the key format (2 parts, never a query_rank suffix) -- defensive re-check:
        _require(detail["query_class"] == qc.get(f"{key}:q1", {}).get("query_class", detail["query_class"]),
                  f"control_features.propagation_outcome[{key!r}].query_class must match its q1 query_correctness row")

    ivs = d["intervention_view_scores"]
    _require(isinstance(ivs, dict) and frozenset(ivs.keys()) == {"query_correctness"},
              "gt_free_features.intervention_view_scores must have exactly one key: query_correctness")
    raw_scores = ivs["query_correctness"]
    _require(isinstance(raw_scores, dict) and frozenset(raw_scores.keys()) == frozenset(qc.keys()),
              "intervention_view_scores.query_correctness keys must exactly match control_features.query_correctness keys")
    for key, scores in raw_scores.items():
        _require(isinstance(scores, dict) and frozenset(scores.keys()) == {"s_original", "s_context", "s_region", "s_control"},
                  f"intervention_view_scores.query_correctness[{key!r}] has the wrong key set")
        for field, v in scores.items():
            _require(_is_finite_number(v) and not _is_strict_int(v), f"intervention_view_scores.query_correctness[{key!r}].{field} must be a finite float")
            _require(-1.0 <= float(v) <= 1.0, f"intervention_view_scores.query_correctness[{key!r}].{field} must be in [-1,1]")

    ir = d["intervention_reliability"]
    _require(isinstance(ir, dict) and frozenset(ir.keys()) == {"query_correctness", "propagation_outcome"},
              "gt_free_features.intervention_reliability must have exactly query_correctness/propagation_outcome")
    _require(frozenset(ir["query_correctness"].keys()) == frozenset(qc.keys()),
              "intervention_reliability.query_correctness keys must exactly match control_features.query_correctness keys")
    _require(frozenset(ir["propagation_outcome"].keys()) == frozenset(po.keys()),
              "intervention_reliability.propagation_outcome keys must exactly match control_features.propagation_outcome keys")
    for namespace, keyset in (("query_correctness", qc), ("propagation_outcome", po)):
        for key in keyset:
            _require(ir[namespace][key] == keyset[key]["intervention_reliability"],
                      f"intervention_reliability.{namespace}[{key!r}] must match control_features.{namespace}[{key!r}].intervention_reliability exactly")

    anchor_ids_from_rows = sorted({k.split(":")[1] for k in qc} | {k.split(":")[1] for k in po})
    _require(d["anchor_ids_ordered"] == anchor_ids_from_rows,
              "gt_free_features.anchor_ids_ordered must be the sorted, deduplicated anchor_ids seen across all rows")
    _require(len(d["anchor_ids_ordered"]) == len(set(d["anchor_ids_ordered"])), "anchor_ids_ordered must not contain duplicates")

    flags = d["structural_validity_flags"]
    expected_flag_keys = frozenset(_GATE_A_COUNTER_KEYS) | {"canonical_forward_count", "intervention_forward_count"}
    _require(isinstance(flags, dict) and frozenset(flags.keys()) == expected_flag_keys,
              "structural_validity_flags has the wrong key set")
    for k, v in flags.items():
        _require(_is_strict_int(v) and v >= 0, f"structural_validity_flags[{k!r}] must be a non-negative int")
    _require(flags["query_row_count"] == len(qc), "structural_validity_flags.query_row_count must equal the number of persisted query_correctness rows")
    _require(flags["propagation_row_count"] == len(po), "structural_validity_flags.propagation_row_count must equal the number of persisted propagation_outcome rows")
    _require(flags["accepted_anchor_count"] + flags["excluded_no_valid_control_count"] == flags["attempted_anchor_count"],
              "structural_validity_flags: accepted_anchor_count + excluded_no_valid_control_count must equal attempted_anchor_count")

    digest = d["gt_free_feature_digest"]
    _require(isinstance(digest, str) and _SHA256_HEX_RE.match(digest), "gt_free_feature_digest must be a 64-char lowercase hex SHA256")
    _require(digest == compute_gt_free_content_digest(d), "gt_free_feature_digest does not match the recomputed content digest")


def validate_gt_label_artifact(d: dict) -> None:
    _validate_closed(d, _GT_LABEL_REQUIRED_FIELDS, "gt_dependent_labels artifact")
    _require_schema_block(d["schema"], "gt_dependent_labels")
    _require(isinstance(d["image_id"], str) and _IMAGE_ID_RE.match(d["image_id"]), "gt_dependent_labels.image_id must be a 12-digit str")

    qcl = d["query_correctness_labels"]
    _require(isinstance(qcl, dict), "query_correctness_labels must be a dict")
    for key, label in qcl.items():
        _require(len(key.split(":")) == 3, f"query_correctness_labels key {key!r} must have exactly 3 colon-separated parts")
        _require(_is_strict_int(label) and label in (0, 1), f"query_correctness_labels[{key!r}] must be exactly 0 or 1 (int, not bool)")

    poc = d["propagation_outcome_categories"]
    _require(isinstance(poc, dict), "propagation_outcome_categories must be a dict")
    for key, category in poc.items():
        _require(len(key.split(":")) == 2, f"propagation_outcome_categories key {key!r} must have exactly 2 colon-separated parts (q1-only)")
        _require(category in _PROPAGATION_TAXONOMY, f"propagation_outcome_categories[{key!r}]={category!r} is not one of the 5 frozen taxonomy values")

    ambiguous = d["ambiguous_gt_anchor_ids"]
    _require(isinstance(ambiguous, list), "ambiguous_gt_anchor_ids must be a list")
    _require(ambiguous == sorted(ambiguous), "ambiguous_gt_anchor_ids must be sorted")
    _require(len(ambiguous) == len(set(ambiguous)), "ambiguous_gt_anchor_ids must not contain duplicates")
    for key in ambiguous:
        _require(len(key.split(":")) == 2, f"ambiguous_gt_anchor_ids entry {key!r} must be an anchor-level (ordinal:anchor_id) key")

    for key in qcl:
        anchor_key = ":".join(key.split(":")[:2])
        _require(anchor_key not in ambiguous, f"query_correctness_labels[{key!r}] is labeled but its anchor {anchor_key!r} is also marked ambiguous")
    for key in poc:
        _require(key not in ambiguous, f"propagation_outcome_categories[{key!r}] is labeled but also marked ambiguous")

    digest = d["gt_dependent_label_digest"]
    _require(isinstance(digest, str) and _SHA256_HEX_RE.match(digest), "gt_dependent_label_digest must be a 64-char lowercase hex SHA256")
    _require(digest == compute_gt_label_content_digest(d), "gt_dependent_label_digest does not match the recomputed content digest")


def validate_gt_pair_linkage(gt_free: dict, gt_label: dict) -> None:
    """Cross-artifact relational check: every GT-label row/key must trace
    back to a real GT-free row, and every GT-free row must be accounted
    for by either a label or an ambiguous-anchor exclusion -- never
    silently missing from both."""
    _require(gt_free["image_id"] == gt_label["image_id"], "gt_free/gt_label image_id mismatch")

    qc_keys = frozenset(gt_free["control_features"]["query_correctness"].keys())
    po_keys = frozenset(gt_free["control_features"]["propagation_outcome"].keys())
    qcl_keys = frozenset(gt_label["query_correctness_labels"].keys())
    poc_keys = frozenset(gt_label["propagation_outcome_categories"].keys())
    ambiguous = frozenset(gt_label["ambiguous_gt_anchor_ids"])

    _require(qcl_keys <= qc_keys, "query_correctness_labels references a row absent from the GT-free artifact")
    _require(poc_keys <= po_keys, "propagation_outcome_categories references a row absent from the GT-free artifact")

    for key in qc_keys - qcl_keys:
        anchor_key = ":".join(key.split(":")[:2])
        _require(anchor_key in ambiguous, f"GT-free row {key!r} is neither labeled nor marked ambiguous")
    for key in po_keys - poc_keys:
        _require(key in ambiguous, f"GT-free propagation row {key!r} is neither categorized nor marked ambiguous")


def validate_final_result(d: dict, discovery_ids: list) -> None:
    _validate_closed(d, _FINAL_RESULT_REQUIRED_FIELDS, "final_result artifact")
    _require(d["complete"] is True and d["final"] is True, "final_result must be complete and final")
    _require(d["discovery_image_ids_ordered"] == discovery_ids, "final_result must list the exact frozen discovery order")
    _require(d["holdout_accessed"] is False, "final_result.holdout_accessed must be false")
    _require(d["full_evaluation_accessed"] is False, "final_result.full_evaluation_accessed must be false")
    _require(d["gt_free_feature_digest"] != d["gt_dependent_label_digest"], "GT-free and GT-label digests must not be conflated")


def validate_decision_document(d: dict) -> None:
    _validate_closed(d, _DECISION_REQUIRED_FIELDS, "decision document")
    _require(d["gate_b_status"] == "NOT_COMPUTABLE_WITHOUT_HOLDOUT", "Gate B must remain not computable on discovery")
    _require(d["gate_c_status"] == "NOT_COMPUTABLE_WITHOUT_HOLDOUT", "Gate C must remain not computable on discovery")
    _require(d["gate_d_status"] == "NOT_COMPUTABLE_WITHOUT_HOLDOUT", "Gate D must remain not computable on discovery")
    _require(d["overall_gates_outcome_status"] == "PENDING_HOLDOUT_EVALUATION", "overall gates outcome must remain pending holdout evaluation")
    _require(
        d["discovery_decision"] in ("DISCOVERY_FEATURES_READY_FOR_HOLDOUT_AUTHORITY", "DISCOVERY_STOP_GATE_A_FAILED", "DISCOVERY_INCONCLUSIVE"),
        "discovery_decision must be one of the three frozen outcome values",
    )


# ---------------------------------------------------------------------------
# Atomic installation
# ---------------------------------------------------------------------------

def atomic_install(staging_dir: Path, output_root: Path) -> None:
    """Stages under output_root's parent, then atomically renames into
    place. No overwrite: output_root must not already exist."""
    _require(not output_root.exists(), f"atomic_install refuses to overwrite an existing output_root: {output_root}")
    _require(staging_dir.exists(), f"staging directory does not exist: {staging_dir}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.rename(str(staging_dir), str(output_root))


def install_discovery_result(output_root: Path, checkpoint_meta: dict, gt_free_artifacts: list,
                              gt_label_artifacts: list, final_result: dict, decision: dict,
                              discovery_ids: list) -> dict:
    """Full staged-write -> staged-validate -> atomic-rename -> reopen ->
    public-verify sequence. Returns the public verify-result-equivalent
    summary. Raises before any success state if any step fails."""
    validate_checkpoint_metadata(checkpoint_meta, discovery_ids)
    _require(len(gt_free_artifacts) == len(gt_label_artifacts), "gt_free_artifacts and gt_label_artifacts must have matching length")
    for a in gt_free_artifacts:
        validate_gt_free_artifact(a)
    for a in gt_label_artifacts:
        validate_gt_label_artifact(a)
    for free, label in zip(gt_free_artifacts, gt_label_artifacts):
        validate_gt_pair_linkage(free, label)
    validate_final_result(final_result, discovery_ids)
    validate_decision_document(decision)

    staging_parent = output_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_parent / f".staging-{output_root.name}-{os.getpid()}-{int(time.time() * 1000)}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        (staging_dir / "checkpoint.json").write_text(json.dumps(checkpoint_meta, indent=2))
        (staging_dir / "gt_free_features.json").write_text(json.dumps(gt_free_artifacts, indent=2))
        (staging_dir / "gt_dependent_labels.json").write_text(json.dumps(gt_label_artifacts, indent=2))
        (staging_dir / "discovery_result.json").write_text(json.dumps(final_result, indent=2))
        (staging_dir / "discovery_decision.json").write_text(json.dumps(decision, indent=2))

        # staged reopen + validation -- every artifact, not just the top-level two
        reopened_gt_free = json.loads((staging_dir / "gt_free_features.json").read_text())
        reopened_gt_label = json.loads((staging_dir / "gt_dependent_labels.json").read_text())
        for a in reopened_gt_free:
            validate_gt_free_artifact(a)
        for a in reopened_gt_label:
            validate_gt_label_artifact(a)
        for free, label in zip(reopened_gt_free, reopened_gt_label):
            validate_gt_pair_linkage(free, label)
        reopened_checkpoint = json.loads((staging_dir / "checkpoint.json").read_text())
        validate_checkpoint_metadata(reopened_checkpoint, discovery_ids)
        reopened_result = json.loads((staging_dir / "discovery_result.json").read_text())
        validate_final_result(reopened_result, discovery_ids)
        reopened_decision = json.loads((staging_dir / "discovery_decision.json").read_text())
        validate_decision_document(reopened_decision)

        atomic_install(staging_dir, output_root)
    except BaseException:
        _reraise_if_fatal_and_cleanup(staging_dir)
        raise
    else:
        pass

    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)

    installed_gt_free = json.loads((output_root / "gt_free_features.json").read_text())
    installed_gt_label = json.loads((output_root / "gt_dependent_labels.json").read_text())
    for a in installed_gt_free:
        validate_gt_free_artifact(a)
    for a in installed_gt_label:
        validate_gt_label_artifact(a)
    for free, label in zip(installed_gt_free, installed_gt_label):
        validate_gt_pair_linkage(free, label)
    installed_checkpoint = json.loads((output_root / "checkpoint.json").read_text())
    validate_checkpoint_metadata(installed_checkpoint, discovery_ids)
    installed_result = json.loads((output_root / "discovery_result.json").read_text())
    validate_final_result(installed_result, discovery_ids)
    installed_decision = json.loads((output_root / "discovery_decision.json").read_text())
    validate_decision_document(installed_decision)

    return {"verified": True, "outcome": installed_decision["discovery_decision"], "path": str(output_root)}


def _reraise_if_fatal_and_cleanup(staging_dir: Path) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)


def _checkpoint_dir_for(output_root: Path) -> Path:
    return output_root.parent / f"{output_root.name}.inprogress"


_GATE_A_COUNTER_KEYS = (
    "attempted_anchor_count", "accepted_anchor_count", "excluded_no_valid_control_count",
    "structurally_invalid_anchor_count", "raw_score_non_finite_count", "raw_score_out_of_range_count",
    "derived_score_invalid_count", "query_row_count", "propagation_row_count",
)


def _content_digest(obj) -> str:
    """Deterministic SHA256 over a canonical (sorted-key, compact) JSON
    encoding. Used to fold structured GT-free content (window boxes, T0/T320
    predicted-class identifiers) into the artifact digest alongside the
    already-independently-verified row digest -- never includes
    timestamps/host/Slurm/profiling data, since none of `obj` ever contains
    any of those (this function is only ever called on scientific content)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _run_phase_a(image_id: str, image_loader, forward_fn, model, text_embedding,
                  crop_hw: tuple, stride_hw: tuple, patch_hw: tuple, event_log: PhaseEventLog) -> dict:
    """Phase A only: GT-free. Builds and validates the complete gt_free
    artifact, including per-row Model-A feature vectors, raw intervention
    scores, and T0/T320 PREDICTED-CLASS IDENTIFIERS for every
    propagation-eligible anchor (a model-vs-model comparison -- GT-free by
    construction, since no ground truth is read here). Never calls
    gt_loader; never receives one. `forward_fn` defaults to the real,
    committed semantic_evidence_e3_adapter.extract_e3_window_snapshot."""
    import semantic_evidence_t320 as t320
    import semantic_evidence_intervention_features as ifeat

    if forward_fn is None:
        import semantic_evidence_e3_adapter as e3a
        forward_fn = e3a.extract_e3_window_snapshot

    event_log.record(f"phase_a_start:{image_id}")

    image_tensor, real_h, real_w = image_loader(image_id)
    windows = compute_canonical_windows(real_h, real_w, crop_hw, stride_hw)

    per_window_results = []
    for window in windows:
        per_window_results.append(process_window_gt_free(
            model, text_embedding, image_tensor, window, real_h, real_w, image_id, patch_hw,
            forward_fn=forward_fn,
        ))

    original_snapshots = [w["original_snapshot"] for w in per_window_results]
    boxes = [w["window_box"] for w in per_window_results]

    targets = t320.generate_t0_t320_targets(original_snapshots, boxes, canvas_h=real_h, canvas_w=real_w, ori_h=real_h, ori_w=real_w)
    t0_rows_full = _class_map_to_rows(targets["t0"][0])
    t320_rows_full = _class_map_to_rows(targets["t320"][0])

    all_query_rows = []
    all_propagation_rows = []
    counters = {k: 0 for k in _GATE_A_COUNTER_KEYS}
    canonical_forward_total = 0
    intervention_forward_total = 0
    window_boxes = {}
    query_correctness_features = {}
    propagation_outcome_features = {}
    raw_scores_by_key = {}

    for w in per_window_results:
        all_query_rows.extend(w["query_correctness_rows"])
        all_propagation_rows.extend(w["propagation_rows"])
        for k in _GATE_A_COUNTER_KEYS:
            counters[k] += w["counters"][k]
        canonical_forward_total += w["canonical_forward_count"]
        intervention_forward_total += w["intervention_forward_count"]

        ordinal = w["window_ordinal"]
        window_boxes[str(ordinal)] = list(w["window_box"])

        for row_key, detail in w["query_row_details"].items():
            raw_scores_by_key[row_key] = detail["raw_scores"]
            query_correctness_features[row_key] = {
                "query_class": detail["query_class"],
                "anchor_row": detail["anchor_row"], "anchor_col": detail["anchor_col"],
                "features": detail["features"],
                "intervention_reliability": detail["intervention_reliability"],
            }

        for prop_key, detail in w["propagation_row_details"].items():
            bounds = t0_t320_anchor_pixel_bounds(w["window_box"], detail["anchor_row"], detail["anchor_col"], ifeat.GRID_SIDE, patch_hw)
            t0_footprint = crop_raster(t0_rows_full, bounds)
            t320_footprint = crop_raster(t320_rows_full, bounds)
            # Model predictions are always defined everywhere (unlike GT,
            # which has a real ignore-label concept), so there is no
            # "ambiguous T0/T320" case to handle -- the anchor patch's own
            # center pixel is used directly, never a majority vote (a
            # majority vote can legitimately have no majority winner, which
            # a GT-free identifier must never be allowed to become).
            t0_predicted_class = _center_pixel_class(t0_footprint)
            t320_predicted_class = _center_pixel_class(t320_footprint)
            propagation_outcome_features[prop_key] = {
                "query_class": detail["query_class"],
                "anchor_row": detail["anchor_row"], "anchor_col": detail["anchor_col"],
                "features": detail["features"],
                "intervention_reliability": detail["intervention_reliability"],
                "t0_predicted_class": t0_predicted_class,
                "t320_predicted_class": t320_predicted_class,
            }

    # Independent internal consistency check: the already-verified,
    # row-object-based digest algorithm (semantic_evidence_intervention_
    # features.gt_free_feature_digest) must be deterministic and non-empty
    # for this image's rows -- exercised here as a sanity check, but the
    # artifact's OWN persisted digest is computed directly from the
    # persisted content below, so it can be independently recomputed and
    # re-verified from a reloaded JSON file alone (no transient row objects
    # required), which the row-object digest cannot support once installed.
    _require(
        ifeat.gt_free_feature_digest(sorted(all_query_rows, key=lambda r: (r.image_id, r.window_ordinal, r.anchor_id, r.query_rank))),
        "row-level GT-free digest computation unexpectedly failed",
    )

    gt_free_artifact = {
        "schema": {"name": "gt_free_features", "version": "1.0.0"},
        "image_id": image_id,
        "window_ids_ordered": [f"{image_id}:{w['window_ordinal']}" for w in per_window_results],
        "anchor_ids_ordered": sorted({r.anchor_id for r in all_query_rows}),
        "query_ranks": ["q1", "q2"],
        "intervention_view_scores": {"query_correctness": raw_scores_by_key},
        "control_features": {
            "window_boxes": window_boxes,
            "query_correctness": query_correctness_features,
            "propagation_outcome": propagation_outcome_features,
        },
        "intervention_reliability": {
            "query_correctness": {k: v["intervention_reliability"] for k, v in query_correctness_features.items()},
            "propagation_outcome": {k: v["intervention_reliability"] for k, v in propagation_outcome_features.items()},
        },
        "structural_validity_flags": dict(counters, canonical_forward_count=canonical_forward_total,
                                           intervention_forward_count=intervention_forward_total),
        "gt_free_feature_digest": "",
    }
    gt_free_artifact["gt_free_feature_digest"] = compute_gt_free_content_digest(gt_free_artifact)
    validate_gt_free_artifact(gt_free_artifact)
    event_log.record(f"phase_a_complete:{image_id}")

    return gt_free_artifact


def compute_gt_free_content_digest(artifact: dict) -> str:
    """Recomputable from ONLY the persisted artifact dict (with the digest
    field itself excluded), so any reader -- staged, reopened, installed,
    or reached via the public verify-result path -- can independently
    confirm `artifact["gt_free_feature_digest"] == compute_gt_free_content_
    digest(artifact)` without needing the transient row objects that
    produced it. Never includes timestamps, host data, Slurm metadata, or
    profiling -- none of those fields exist anywhere in this content."""
    content = {k: v for k, v in artifact.items() if k != "gt_free_feature_digest"}
    return _content_digest(content)


def _run_phase_b(gt_free_artifact: dict, gt_loader, image_id: str, event_log: PhaseEventLog) -> dict:
    """Phase B only: GT-dependent. Rebuildable from ONLY an already-
    installed-or-staged, already-validated gt_free_artifact plus a GT
    loader -- never touches image_loader, model, text_embedding, or
    forward_fn, and therefore never recomputes original/intervention
    views, E3 features, T0/T320, controls, or intervention reliability.
    This is what makes finalize-only recovery and per-image
    resume-after-GT-label-write-failure possible without any real
    model/image access."""
    import semantic_evidence_intervention_features as ifeat

    validate_gt_free_artifact(gt_free_artifact)
    _require(gt_free_artifact["image_id"] == image_id, "gt_free_artifact image_id mismatch")
    patch_hw = (ifeat.PATCH_SIZE, ifeat.PATCH_SIZE)

    event_log.record(f"gt_access_start:{image_id}")
    gt_raster = gt_loader(image_id)

    window_boxes = gt_free_artifact["control_features"]["window_boxes"]
    query_correctness_features = gt_free_artifact["control_features"]["query_correctness"]
    propagation_outcome_features = gt_free_artifact["control_features"]["propagation_outcome"]

    query_labels = {}
    propagation_labels = {}
    ambiguous_anchor_ids = set()

    for row_key, detail in query_correctness_features.items():
        ordinal_str, anchor_id, query_rank = row_key.split(":")
        bounds = footprint_pixel_bounds(window_boxes[ordinal_str], detail["anchor_row"], detail["anchor_col"], patch_hw, ifeat.GRID_SIDE)
        footprint = crop_raster(gt_raster, bounds)
        majority_label, majority_class = compute_gt_majority(footprint)
        if majority_label == AMBIGUOUS_GT:
            ambiguous_anchor_ids.add(f"{ordinal_str}:{anchor_id}")
            continue
        query_labels[row_key] = 1 if detail["query_class"] == majority_class else 0

    for prop_key, detail in propagation_outcome_features.items():
        if prop_key in ambiguous_anchor_ids:
            continue
        ordinal_str, anchor_id = prop_key.split(":")
        bounds = footprint_pixel_bounds(window_boxes[ordinal_str], detail["anchor_row"], detail["anchor_col"], patch_hw, ifeat.GRID_SIDE)
        footprint = crop_raster(gt_raster, bounds)
        majority_label, majority_class = compute_gt_majority(footprint)
        if majority_label == AMBIGUOUS_GT:
            ambiguous_anchor_ids.add(prop_key)
            continue
        category = compute_propagation_category(detail["t0_predicted_class"], detail["t320_predicted_class"], majority_class)
        propagation_labels[prop_key] = category

    gt_label_artifact = {
        "schema": {"name": "gt_dependent_labels", "version": "1.0.0"},
        "image_id": image_id,
        "query_correctness_labels": query_labels,
        "propagation_outcome_categories": propagation_labels,
        "ambiguous_gt_anchor_ids": sorted(ambiguous_anchor_ids),
        "gt_dependent_label_digest": "",
    }
    gt_label_artifact["gt_dependent_label_digest"] = compute_gt_label_content_digest(gt_label_artifact)
    validate_gt_label_artifact(gt_label_artifact)
    event_log.record(f"phase_b_complete:{image_id}")

    _require(not event_log.gt_accessed_before(f"phase_a_start:{image_id}", f"phase_a_complete:{image_id}"),
              f"GT-isolation violation: GT accessed before Phase A completed for {image_id}")

    return gt_label_artifact


def _process_image(repo_root: Path, image_id: str, image_loader, forward_fn, model, text_embedding,
                    gt_loader, crop_hw: tuple, stride_hw: tuple, patch_hw: tuple, event_log: PhaseEventLog) -> dict:
    """Convenience wrapper: runs Phase A then Phase B back to back for one
    image (used by fresh-mode processing and by tests that want the full
    per-image pipeline in one call). See _run_phase_a/_run_phase_b for the
    split that enables per-image resumability."""
    gt_free_artifact = _run_phase_a(
        image_id, image_loader, forward_fn, model, text_embedding, crop_hw, stride_hw, patch_hw, event_log,
    )
    gt_label_artifact = _run_phase_b(gt_free_artifact, gt_loader, image_id, event_log)
    return {"gt_free_artifact": gt_free_artifact, "gt_label_artifact": gt_label_artifact}


def _class_map_to_rows(class_map) -> list:
    return class_map.tolist()


def run_discovery(repo_root: Path, *, weight_dir: str, image_root: Path, gt_root: Path, output_root: Path,
                   mode: str, slurm_job_id: str = "", node: str = "",
                   image_loader=None, model_and_text_provider=None, gt_loader=None, forward_fn=None) -> dict:
    """Top-level discovery execution orchestrator (Layer 2). Lazily imports
    torch and the E3/T320/intervention-feature modules; never called by
    verify_protocol/preflight/verify_result. `image_loader`,
    `model_and_text_provider`, and `gt_loader` default to the real
    providers but accept fake/synthetic callables for tests -- this is the
    ONLY function in this module that performs real execution."""
    _require(mode in ("fresh", "resume", "finalize_only"), f"unknown mode: {mode!r}")

    image_root = Path(image_root)
    gt_root = Path(gt_root)
    output_root = Path(output_root)
    checkpoint_dir = _checkpoint_dir_for(output_root)

    data = _load_authority(repo_root)
    # window geometry constants are frozen by [scientific_inheritance.e3_geometry]
    e3_geometry = data["scientific_inheritance"]["e3_geometry"]
    crop_hw = tuple(e3_geometry["crop_size_pixels"])
    stride_hw = tuple(e3_geometry["stride_pixels"])
    patch_hw = (e3_geometry["patch_size_pixels"], e3_geometry["patch_size_pixels"])

    discovery_ids = load_discovery_population(repo_root)

    if image_loader is None:
        authorized_ids = tuple(discovery_ids)
        image_loader = lambda image_id: _default_real_image_loader(image_root, image_id, authorized_ids)
    if gt_loader is None:
        gt_loader = lambda image_id: _default_real_gt_loader(gt_root, image_id)
    if model_and_text_provider is None:
        model_and_text_provider = lambda: _default_real_model_and_text_provider(repo_root, weight_dir)[:2]

    if mode == "fresh":
        _require(not output_root.exists(), f"fresh mode requires an absent output_root: {output_root}")
        _require(not checkpoint_dir.exists(), f"fresh mode requires an absent checkpoint dir: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        # Written immediately, before any image is processed, so the
        # invariant "checkpoint_dir exists => checkpoint.json exists" holds
        # even if the very first image crashes before its own checkpoint
        # update -- resume must never have to special-case a checkpoint_dir
        # with no checkpoint.json in it yet.
        initial_checkpoint = build_checkpoint_metadata(
            mode=mode, completed_image_ids_ordered=[], discovery_ids=discovery_ids,
            checkpoint_array_hashes={}, identity_bindings={"required_head": data["prerequisite"]["required_head"]},
            complete=(len(discovery_ids) == 0), final=False,
        )
        _atomic_write_json(checkpoint_dir / "checkpoint.json", initial_checkpoint)
        completed_ids = []
    elif mode == "resume":
        _require(not output_root.exists(), f"resume mode requires an absent output_root: {output_root}")
        _require(checkpoint_dir.exists(), f"resume mode requires an existing checkpoint dir: {checkpoint_dir}")
        _require((checkpoint_dir / "checkpoint.json").exists(), f"resume mode requires an existing checkpoint.json in {checkpoint_dir}")
        existing_checkpoint = json.loads((checkpoint_dir / "checkpoint.json").read_text())
        validate_checkpoint_metadata(existing_checkpoint, discovery_ids)
        completed_ids = existing_checkpoint["completed_image_ids_ordered"]
    else:  # finalize_only
        _require(not output_root.exists(), f"finalize_only mode requires an absent output_root: {output_root}")
        _require(checkpoint_dir.exists(), f"finalize_only mode requires an existing checkpoint dir: {checkpoint_dir}")
        existing_checkpoint = json.loads((checkpoint_dir / "checkpoint.json").read_text())
        validate_checkpoint_metadata(existing_checkpoint, discovery_ids)
        completed_ids = existing_checkpoint["completed_image_ids_ordered"]
        _require(len(completed_ids) == len(discovery_ids), "finalize_only requires every discovery image already completed")

    event_log = PhaseEventLog()
    t_start = time.perf_counter()

    if mode != "finalize_only":
        model_and_text = None  # constructed lazily: only if some image actually needs Phase A
        for image_id in discovery_ids:
            if image_id in completed_ids:
                continue

            gt_free_path = checkpoint_dir / f"gt_free_{image_id}.json"
            gt_label_path = checkpoint_dir / f"gt_label_{image_id}.json"

            gt_free_artifact = _load_valid_artifact_or_none(gt_free_path, validate_gt_free_artifact, image_id)
            if gt_free_artifact is None:
                if model_and_text is None:
                    model_and_text = model_and_text_provider()
                model, text_embedding = model_and_text
                gt_free_artifact = _run_phase_a(
                    image_id, image_loader, forward_fn, model, text_embedding, crop_hw, stride_hw, patch_hw, event_log,
                )
                _atomic_write_json(gt_free_path, gt_free_artifact)
            else:
                # Phase A was reused from a leftover, already-valid artifact
                # rather than run in this process -- record synthetic
                # boundary markers anyway, so _run_phase_b's own GT-isolation
                # self-check (which looks for this image's phase_a_start/
                # phase_a_complete events) has a boundary to reference. This
                # is bookkeeping only; it does not re-run or re-verify any
                # GT-free science, which the artifact's own digest already covers.
                event_log.record(f"phase_a_start:{image_id}")
                event_log.record(f"phase_a_complete:{image_id}")

            gt_label_artifact = _load_valid_artifact_or_none(gt_label_path, validate_gt_label_artifact, image_id)
            if gt_label_artifact is not None:
                try:
                    validate_gt_pair_linkage(gt_free_artifact, gt_label_artifact)
                except DiscoveryExecutionError:
                    gt_label_artifact = None  # stale label from a since-regenerated gt_free: redo Phase B
            if gt_label_artifact is None:
                gt_label_artifact = _run_phase_b(gt_free_artifact, gt_loader, image_id, event_log)
                _atomic_write_json(gt_label_path, gt_label_artifact)
            validate_gt_pair_linkage(gt_free_artifact, gt_label_artifact)

            completed_ids = completed_ids + [image_id]
            checkpoint_meta = build_checkpoint_metadata(
                mode=mode, completed_image_ids_ordered=completed_ids, discovery_ids=discovery_ids,
                checkpoint_array_hashes={
                    "gt_free": gt_free_artifact["gt_free_feature_digest"],
                    "gt_label": gt_label_artifact["gt_dependent_label_digest"],
                },
                identity_bindings={"required_head": data["prerequisite"]["required_head"]},
                complete=(len(completed_ids) == len(discovery_ids)), final=False,
            )
            _atomic_write_json(checkpoint_dir / "checkpoint.json", checkpoint_meta)

    # --- aggregate across all completed images for fitting/gates -----------
    gt_free_digests = []
    gt_label_digests = []
    gate_a_totals = {k: 0 for k in _GATE_A_COUNTER_KEYS}
    non_finite_raw_total = 0

    query_X_a, query_X_b, query_y, query_row_keys = [], [], [], []
    prop_X_a, prop_X_b, prop_y, prop_row_keys = [], [], [], []

    for image_id in discovery_ids:
        gt_free = json.loads((checkpoint_dir / f"gt_free_{image_id}.json").read_text())
        gt_label = json.loads((checkpoint_dir / f"gt_label_{image_id}.json").read_text())
        validate_gt_free_artifact(gt_free)
        validate_gt_label_artifact(gt_label)
        gt_free_digests.append(gt_free["gt_free_feature_digest"])
        gt_label_digests.append(gt_label["gt_dependent_label_digest"])

        flags = gt_free["structural_validity_flags"]
        for k in _GATE_A_COUNTER_KEYS:
            gate_a_totals[k] += flags[k]
        non_finite_raw_total += flags["raw_score_non_finite_count"]

        qc_features = gt_free["control_features"]["query_correctness"]
        qc_reliability = gt_free["intervention_reliability"]["query_correctness"]
        for row_key, label in gt_label["query_correctness_labels"].items():
            detail = qc_features[row_key]
            full_key = f"{image_id}:{row_key}"
            query_X_a.append(list(detail["features"]))
            query_X_b.append(list(detail["features"]) + [qc_reliability[row_key]])
            query_y.append(label)
            query_row_keys.append(full_key)

        po_features = gt_free["control_features"]["propagation_outcome"]
        po_reliability = gt_free["intervention_reliability"]["propagation_outcome"]
        for row_key, category in gt_label["propagation_outcome_categories"].items():
            if category not in ("beneficial", "harmful"):
                continue  # unchanged_correct/unchanged_incorrect/changed_but_same_correctness: excluded from the binary fit
            detail = po_features[row_key]
            full_key = f"{image_id}:{row_key}"
            prop_X_a.append(list(detail["features"]))
            prop_X_b.append(list(detail["features"]) + [po_reliability[row_key]])
            prop_y.append(1 if category == "beneficial" else 0)
            prop_row_keys.append(full_key)

    scientific_collection_digest = hashlib.sha256(
        ("".join(gt_free_digests) + "|" + "".join(gt_label_digests)).encode("ascii")
    ).hexdigest()

    total_wall_time = time.perf_counter() - t_start

    gate_a = compute_gate_a(
        total_anchors=gate_a_totals["attempted_anchor_count"],
        structurally_invalid_anchor_count=gate_a_totals["structurally_invalid_anchor_count"],
        non_finite_scores_count=non_finite_raw_total,
        gt_dependent_intervention_records=0,
    )
    gate_e = compute_gate_e(total_wall_time)

    model_fit_summaries = []
    fit_error = None
    try:
        model_fit_summaries.append(fit_logistic_model(
            query_X_a, query_y, task_id="query_correctness", model_id="model_a", model_b=False, row_keys=query_row_keys,
        ))
        model_fit_summaries.append(fit_logistic_model(
            query_X_b, query_y, task_id="query_correctness", model_id="model_b", model_b=True, row_keys=query_row_keys,
        ))
        model_fit_summaries.append(fit_logistic_model(
            prop_X_a, prop_y, task_id="propagation_outcome", model_id="model_a", model_b=False, row_keys=prop_row_keys,
        ))
        model_fit_summaries.append(fit_logistic_model(
            prop_X_b, prop_y, task_id="propagation_outcome", model_id="model_b", model_b=True, row_keys=prop_row_keys,
        ))
    except DiscoveryExecutionError as exc:
        fit_error = str(exc)
        model_fit_summaries = []

    degenerate = fit_error is not None or len(model_fit_summaries) != 4
    decision_value = discovery_decision(gate_a, gate_e, degenerate_sample=degenerate)

    final_result = {
        "schema": {"name": "discovery_result", "version": "1.0.0"},
        "complete": True, "final": True,
        "identity_and_dependency_bindings": {"required_head": data["prerequisite"]["required_head"]},
        "population_binding": {"discovery_count": len(discovery_ids)},
        "implementation_binding": {"module": "semantic_evidence_discovery_execute", "mode": mode},
        "discovery_image_ids_ordered": discovery_ids,
        "gt_free_feature_digest": hashlib.sha256("".join(gt_free_digests).encode("ascii")).hexdigest(),
        "gt_dependent_label_digest": hashlib.sha256("".join(gt_label_digests).encode("ascii")).hexdigest(),
        "scientific_collection_digest": scientific_collection_digest,
        "model_fit_summaries": model_fit_summaries,
        "primary_metrics": {"gate_a": gate_a, "gate_e": gate_e},
        "bootstrap_intervals": {},
        "gate_inputs": {"gate_a": gate_a, "gate_e": gate_e},
        "gate_pass_fail": {"A": gate_a["passed"], "E": gate_e["passed"]},
        "final_frozen_outcome": decision_value,
        "holdout_accessed": False, "full_evaluation_accessed": False,
    }
    decision_doc = {
        "schema": {"name": "discovery_decision", "version": "1.0.0"},
        "gate_a_discovery_side": gate_a, "gate_e_discovery_side": gate_e,
        **{f"gate_{g.lower()}_status": v["status"] for g, v in gate_b_c_d_not_computable().items()},
        "overall_gates_outcome_status": "PENDING_HOLDOUT_EVALUATION",
        "discovery_decision": decision_value,
        "next_step": "CREATE_HOLDOUT_EVALUATION_AUTHORITY" if decision_value == "DISCOVERY_FEATURES_READY_FOR_HOLDOUT_AUTHORITY" else "STOP_OR_REVISE_HYPOTHESIS",
    }

    finalize_checkpoint = build_checkpoint_metadata(
        mode=mode, completed_image_ids_ordered=discovery_ids, discovery_ids=discovery_ids,
        checkpoint_array_hashes={
            "gt_free": final_result["gt_free_feature_digest"], "gt_label": final_result["gt_dependent_label_digest"],
        },
        identity_bindings={"required_head": data["prerequisite"]["required_head"]},
        complete=True, final=True,
    )

    gt_free_artifacts = [json.loads((checkpoint_dir / f"gt_free_{iid}.json").read_text()) for iid in discovery_ids]
    gt_label_artifacts = [json.loads((checkpoint_dir / f"gt_label_{iid}.json").read_text()) for iid in discovery_ids]

    install_summary = install_discovery_result(
        output_root, finalize_checkpoint, gt_free_artifacts, gt_label_artifacts,
        final_result, decision_doc, discovery_ids,
    )
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return install_summary


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(str(tmp), str(path))


def _load_valid_artifact_or_none(path: Path, validator, image_id: str):
    """Reuse support for per-image resumability: if a leftover artifact
    file from a prior interrupted attempt exists AND is well-formed AND
    belongs to this image_id, return it so the caller can skip redoing
    that phase's real work (Phase A's model/image access or Phase B's GT
    access). Any parse/validation failure or a JSON well-formed-but-wrong-
    image_id file is treated as absent -- the caller will safely
    regenerate it (atomic writes mean a genuinely partial write can never
    be read back as valid JSON in the first place)."""
    if not path.exists():
        return None
    try:
        candidate = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    try:
        validator(candidate)
    except DiscoveryExecutionError:
        return None
    if candidate.get("image_id") != image_id:
        return None
    return candidate


def verify_result(output_root: Path, discovery_ids: list) -> dict:
    """Torch-free public post-install verification, independently
    reopening and re-validating the complete installed artifact set --
    checkpoint, every gt_free/gt_label pair (including their cross-artifact
    linkage), the final result, and the decision document."""
    output_root = Path(output_root)
    _require(output_root.exists(), f"output_root does not exist: {output_root}")
    checkpoint_meta = json.loads((output_root / "checkpoint.json").read_text())
    validate_checkpoint_metadata(checkpoint_meta, discovery_ids)

    gt_free_artifacts = json.loads((output_root / "gt_free_features.json").read_text())
    gt_label_artifacts = json.loads((output_root / "gt_dependent_labels.json").read_text())
    _require(len(gt_free_artifacts) == len(gt_label_artifacts), "gt_free_features.json/gt_dependent_labels.json length mismatch")
    for a in gt_free_artifacts:
        validate_gt_free_artifact(a)
    for a in gt_label_artifacts:
        validate_gt_label_artifact(a)
    for free, label in zip(gt_free_artifacts, gt_label_artifacts):
        validate_gt_pair_linkage(free, label)

    result = json.loads((output_root / "discovery_result.json").read_text())
    validate_final_result(result, discovery_ids)
    decision = json.loads((output_root / "discovery_decision.json").read_text())
    validate_decision_document(decision)
    return {"verified": True, "outcome": decision["discovery_decision"], "path": str(output_root)}
