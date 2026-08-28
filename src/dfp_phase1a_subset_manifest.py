"""DFP Phase 1A Step 2: deterministic evaluation-subset selection and
strict manifest generation/verification for the two frozen 500-image
subsets (COCO-Stuff, ADE20K).

CPU/data-read-only. Never initializes CUDA, never imports torch, never
loads a model or checkpoint into a model, never runs inference or
propagation. Ground truth is read for exactly three purposes: source
validation (decoding images/masks that already have their own frozen
identity contract), image-level class-presence-vector construction, and
deterministic multilabel subset selection -- never for prediction
construction.

Selection algorithm
--------------------
``dfp_phase1a_deterministic_iterative_stratification v1`` -- a
self-contained, fully deterministic two-fold specialization of the
iterative-stratification procedure (Sechidis, Tsoumakas & Vlahavas,
"On the Stratification of Multi-Label Data", ECML/PKDD 2011), adapted so
every decision that the published algorithm leaves to an unspecified or
random tie-break is instead resolved by one frozen SHA-256 tie-break
scheme (see ``tie_break_hexdigest``/``tie_break_int`` below). No
``random``, no NumPy/PyTorch RNG, no dependence on set/dict iteration
order, filesystem traversal order, timestamps, hostnames, or process
IDs.

Two folds: ``selected`` (exactly 500) and ``remainder`` (source_count -
500). Steps, precisely:

1. For each class ``c``, ``class_totals[c]`` = number of candidate images
   with at least one evaluated non-ignore pixel of class ``c``.
2. ``desired_selected[c]`` = the nearest integer to
   ``class_totals[c] * 500 / source_count``, computed with exact
   ``fractions.Fraction`` arithmetic (never floating point). An exact
   one-half boundary is broken by the LABEL tie-break parity (round up
   iff the label's tie-break integer is odd). ``desired_remainder[c] =
   class_totals[c] - desired_selected[c]`` (so the two folds' desired
   counts for every class always sum to ``class_totals[c]`` exactly, by
   construction -- no separate reconciliation pass is needed).
3. Repeatedly: among all classes with at least one still-unassigned
   candidate carrying that class (its "remaining supply"), pick the
   class with the smallest positive remaining supply -- the LABEL
   tie-break integer (ascending) breaks a tie. For every
   still-unassigned candidate carrying that class, in ascending IMAGE
   tie-break order, choose its fold by a HARD CAPACITY GATE FIRST: if
   exactly one fold still has remaining capacity
   (``already_assigned[fold] < target_size[fold]``), it is chosen
   unconditionally, regardless of any label's remaining demand -- this
   is the rule that guarantees the exact frozen fold sizes for
   genuinely multilabel data, since per-class desired counts are
   rounded independently and are not constrained to sum to
   ``selected_count`` in aggregate. Only when both folds still have
   capacity does the label-demand comparison apply: assign to whichever
   fold currently has the larger remaining desired count for the
   triggering class (``argmax``); a tie is broken by the fold with the
   larger remaining overall assignment target
   (``target_size[fold] - already_assigned[fold]``); a further tie is
   broken by the IMAGE tie-break integer's parity. Assigning a
   candidate decrements the chosen fold's remaining desired count for
   *every* class it carries (not only the class that triggered its
   selection).
4. Once every candidate with at least one class label has been
   assigned, the previously set-aside label-empty candidates (an
   all-ignore or all-background mask) are assigned last, in ascending
   image tie-break order, each to whichever fold has the larger
   remaining overall assignment target; ties broken by image tie-break
   parity.
5. The selected 500 IDs are reordered into the authoritative canonical
   validation order (the same order ``canonical_validation_ids``
   returns) before being written to the manifest -- selection order is
   never evaluation order, and feeding the selector a reversed or
   otherwise reordered candidate list produces the identical selected
   SET and, after this final reordering step, the identical output
   order.

Tie-break scheme
-----------------
``tie_break_namespace = "talk2dino-dfp-phase1a-subset-tie-break-v1"``.
Every tie-break integer is::

    SHA256(
        canonical_namespace.encode("utf-8") + b"\\x00"
        + dataset_key.encode("utf-8") + b"\\x00"
        + canonical_key.encode("utf-8")
    )

interpreted as a big-endian integer (``int(hexdigest, 16)``). For an
IMAGE-level tie, ``canonical_key`` is the image's own canonical ID
(exactly as the task's frozen formula specifies). For a LABEL-level
tie (deciding label-processing order, and the exact-one-half rounding
boundary), ``canonical_key`` is the literal string
``f"class_index={class_index:04d}"`` -- a format that can never collide
with a real COCO-Stuff (all-digit) or ADE20K (``ADE_val_########``)
image ID, reusing the identical namespace/encoding rather than
inventing a second scheme.
"""

from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

TIE_BREAK_NAMESPACE = "talk2dino-dfp-phase1a-subset-tie-break-v1"
TIE_BREAK_ENCODING = "utf8_nul_joined_sha256_hexdigest"

ALGORITHM_NAME = "dfp_phase1a_deterministic_iterative_stratification"
ALGORITHM_VERSION = "v1"

MANIFEST_SCHEMA_NAME = "talk2dino-dfp-phase1a-subset-manifest-v1"
MANIFEST_SCHEMA_VERSION = 1

# The single production authority for this field's exact text -- used by
# both the generator (build_manifest) and the verifier
# (verify_manifest_document), so the two can never silently drift apart.
# Exact byte-for-byte equality only: no stripping, case-folding, or
# normalization. Relationally confirmed against both datasets' own
# authoritative order convention before being frozen here: ADE20K's
# committed e12-ade20k-dataset-source identity states this exact
# convention in its own [source].split_order_authority field, and the
# live installed mmseg.datasets.custom.CustomDataset.load_annotations
# (split=None, the actual COCO-Stuff configuration) sorts by filename --
# both datasets' canonical order is lexicographic sort of their
# fixed-width image ID strings.
CANONICAL_SOURCE_ORDER_ASSERTION = "sorted validation image filenames (dataset's own frozen split_order_authority / directory-listing sort)"

SELECTED_COUNT = 500

PROTOCOL_IDENTITY_RELATIVE_PATH = "evaluation_identities/e13_dfp_phase1a_protocol.toml"


class DfpPhase1aSubsetManifestError(ValueError):
    """Raised on any DFP Phase 1A subset-manifest invariant violation.
    Always fail closed."""


# ---------------------------------------------------------------------------
# Environment / root resolution
# ---------------------------------------------------------------------------


def require_env_root(var_name: str, *, override: str | None = None) -> Path:
    """Fail-closed resolution of a dataset-root environment variable (or
    an explicit CLI override, which takes precedence when given). Never
    falls back to ``Path("")`` or any other default."""
    raw = override if override is not None else os.environ.get(var_name)
    if raw is None:
        raise DfpPhase1aSubsetManifestError(f"{var_name} is not set and no --*-root override was given")
    if not raw.strip():
        raise DfpPhase1aSubsetManifestError(f"{var_name} is whitespace-only")
    path = Path(raw)
    if not path.exists():
        raise DfpPhase1aSubsetManifestError(f"{var_name} points to a path that does not exist: {path}")
    if not path.is_dir():
        raise DfpPhase1aSubsetManifestError(f"{var_name} points to a regular file, not a directory: {path}")
    return path


def _reject_escaping_symlink(resolved: Path, *, label: str) -> None:
    if resolved.is_symlink() and resolved.resolve().parent != resolved.parent.resolve():
        raise DfpPhase1aSubsetManifestError(f"resolved {label} {resolved} is a symlink escaping its containing directory")


def resolve_coco_stuff_root(data_root: Path) -> Path:
    """Fail-closed COCO-Stuff root resolution. Accepts either the
    directory directly containing ``images/val2017`` and
    ``annotations/val2017``, or an unambiguous immediate child that
    does."""
    data_root = Path(data_root)
    if not data_root.exists():
        raise DfpPhase1aSubsetManifestError(f"COCO-Stuff dataset root does not exist: {data_root}")
    if not data_root.is_dir():
        raise DfpPhase1aSubsetManifestError(f"COCO-Stuff dataset root is not a directory: {data_root}")

    required = ("images/val2017", "annotations/val2017")

    def _satisfies(candidate: Path) -> bool:
        return candidate.is_dir() and all((candidate / sub).is_dir() for sub in required)

    candidates: list[Path] = []
    if _satisfies(data_root):
        candidates.append(data_root)
    for child in sorted((p for p in data_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        if _satisfies(child) and child not in candidates:
            candidates.append(child)

    if len(candidates) == 0:
        raise DfpPhase1aSubsetManifestError(
            f"cannot resolve a COCO-Stuff root under {data_root}: neither it nor an immediate child contains "
            f"both of {list(required)}"
        )
    if len(candidates) > 1:
        raise DfpPhase1aSubsetManifestError(
            f"ambiguous COCO-Stuff root under {data_root}: multiple candidates satisfy the root contract: "
            f"{[str(c) for c in candidates]}"
        )
    resolved = candidates[0]
    _reject_escaping_symlink(resolved, label="COCO-Stuff root")
    return resolved


# ---------------------------------------------------------------------------
# COCO-Stuff candidate universe (no separate source-manifest module exists
# for COCO-Stuff in this repository -- constructed directly here, tracing
# its label/ignore policy from the live evaluation_identities/
# e12_matched_k11_k12_t320.toml [dataset] section and the installed
# mmseg.datasets.coco_stuff.COCOStuffDataset class: COCO-Stuff-164k ships
# pre-converted 0..170 trainId masks with ignore_index=255 baked in --
# reduce_zero_label is NOT applied (that flag is specific to the older
# COCO-Stuff-10k variant).
# ---------------------------------------------------------------------------

COCO_STUFF_CLASS_COUNT = 171
COCO_STUFF_IGNORE_INDEX = 255
COCO_STUFF_IMAGE_SUFFIX = ".jpg"
COCO_STUFF_ANNOTATION_SUFFIX = "_labelTrainIds.png"
COCO_STUFF_LEGAL_RAW_LABELS = frozenset(range(0, COCO_STUFF_CLASS_COUNT)) | {COCO_STUFF_IGNORE_INDEX}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_digest(rows: Any) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=True).encode("utf-8")).hexdigest()


def _scan_ids(directory: Path, *, suffix: str) -> list[str]:
    if not directory.is_dir():
        raise DfpPhase1aSubsetManifestError(f"expected directory does not exist: {directory}")
    stems: list[str] = []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.name.endswith(suffix) and entry.is_file(follow_symlinks=False):
                stems.append(entry.name[: -len(suffix)])
    return sorted(stems)


def coco_stuff_canonical_validation_ids(coco_root: Path, *, expected_count: int) -> list[str]:
    img_dir = coco_root / "images/val2017"
    mask_dir = coco_root / "annotations/val2017"
    image_ids = _scan_ids(img_dir, suffix=COCO_STUFF_IMAGE_SUFFIX)
    mask_ids = _scan_ids(mask_dir, suffix=COCO_STUFF_ANNOTATION_SUFFIX)

    if len(set(image_ids)) != len(image_ids):
        raise DfpPhase1aSubsetManifestError(f"{img_dir} contains a duplicate image ID")
    if image_ids != mask_ids:
        only_images = sorted(set(image_ids) - set(mask_ids))
        only_masks = sorted(set(mask_ids) - set(image_ids))
        raise DfpPhase1aSubsetManifestError(
            f"validation image IDs and trainId-mask IDs disagree: {len(only_images)} image-only, "
            f"{len(only_masks)} mask-only (examples: {only_images[:3]}, {only_masks[:3]})"
        )
    if len(image_ids) != expected_count:
        raise DfpPhase1aSubsetManifestError(f"{img_dir} has {len(image_ids)} entries, expected exactly {expected_count}")
    for image_id in image_ids:
        if not image_id.isdigit() or len(image_id) != 12:
            raise DfpPhase1aSubsetManifestError(f"validation image ID {image_id!r} does not match the 12-digit COCO syntax")
    return image_ids


def _decode_image_size(path: Path, *, label: str) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size
    except (OSError, UnidentifiedImageError) as error:
        raise DfpPhase1aSubsetManifestError(f"cannot decode {label} {path}: {error}") from error


def _decode_mask_array(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as im:
            mode = im.mode
            arr = np.array(im)
    except (OSError, UnidentifiedImageError) as error:
        raise DfpPhase1aSubsetManifestError(f"cannot decode {label} {path}: {error}") from error
    if mode not in ("L", "P", "I"):
        raise DfpPhase1aSubsetManifestError(f"{label} {path} has mode {mode!r}, expected a single-channel mask")
    if arr.ndim != 2:
        raise DfpPhase1aSubsetManifestError(f"{label} {path} decoded to {arr.ndim} dimensions, expected 2D")
    return arr


def coco_stuff_class_presence(coco_root: Path, image_ids: Sequence[str]) -> dict[str, frozenset[int]]:
    """y_i[c] = 1 iff class c occurs as an evaluated (non-ignore) pixel
    value anywhere in image i's trainId mask. COCO-Stuff-164k trainId
    masks already ARE the evaluation label (0..170); 255 is ignore.
    Binary image-level presence only -- no pixel-frequency weighting."""
    img_dir = coco_root / "images/val2017"
    mask_dir = coco_root / "annotations/val2017"
    presence: dict[str, frozenset[int]] = {}
    for image_id in image_ids:
        image_path = img_dir / f"{image_id}{COCO_STUFF_IMAGE_SUFFIX}"
        mask_path = mask_dir / f"{image_id}{COCO_STUFF_ANNOTATION_SUFFIX}"
        if not image_path.is_file():
            raise DfpPhase1aSubsetManifestError(f"missing validation image for id {image_id!r}: {image_path}")
        if not mask_path.is_file():
            raise DfpPhase1aSubsetManifestError(f"missing validation trainId mask for id {image_id!r}: {mask_path}")
        image_wh = _decode_image_size(image_path, label="validation image")
        mask_arr = _decode_mask_array(mask_path, label="validation trainId mask")
        mask_wh = (mask_arr.shape[1], mask_arr.shape[0])
        if image_wh != mask_wh:
            raise DfpPhase1aSubsetManifestError(f"image/mask dimension mismatch for id {image_id!r}: image={image_wh} mask={mask_wh}")
        raw_labels = set(np.unique(mask_arr).tolist())
        illegal = raw_labels - COCO_STUFF_LEGAL_RAW_LABELS
        if illegal:
            raise DfpPhase1aSubsetManifestError(f"trainId mask {mask_path} contains illegal label value(s): {sorted(illegal)}")
        evaluated = frozenset(c for c in raw_labels if c != COCO_STUFF_IGNORE_INDEX)
        presence[image_id] = evaluated
    return presence


# ---------------------------------------------------------------------------
# ADE20K candidate universe -- reuses the existing, already-frozen
# src.ade20k_dataset_manifest module (resolve_dataset_root,
# canonical_validation_ids, _decode_mask) rather than duplicating a second
# authority for a dataset that already has one.
# ---------------------------------------------------------------------------

ADE20K_CLASS_COUNT = 150
ADE20K_RAW_IGNORE_VALUE = 0
ADE20K_IGNORE_INDEX = 255


def ade20k_class_presence(ade_root: Path, identity: Mapping[str, Any], image_ids: Sequence[str]) -> dict[str, frozenset[int]]:
    """y_i[c] = 1 iff evaluation class c (0..149, after the identity's own
    frozen reduce_zero_label transform: raw 0 -> ignore, raw k in
    [1,150] -> eval k-1) occurs anywhere in image i's mask. Traces the
    transform from the live, committed e12-ade20k-dataset-source
    identity's [label_policy] block -- never inferred from class count
    alone."""
    label_policy = identity["label_policy"]
    if label_policy["reduce_zero_label"] is not True:
        raise DfpPhase1aSubsetManifestError("ADE20K label_policy.reduce_zero_label is not true; the frozen identity's transform assumption no longer holds")
    if label_policy["raw_ignore_value"] != ADE20K_RAW_IGNORE_VALUE:
        raise DfpPhase1aSubsetManifestError("ADE20K label_policy.raw_ignore_value disagrees with the frozen raw-0-ignored convention")
    if label_policy["ignore_index"] != ADE20K_IGNORE_INDEX:
        raise DfpPhase1aSubsetManifestError("ADE20K label_policy.ignore_index disagrees with the frozen 255 convention")

    img_dir = ade_root / identity["source"]["image_relative_root"]
    mask_dir = ade_root / identity["source"]["annotation_relative_root"]
    image_suffix = identity["source"]["image_suffix"]
    annotation_suffix = identity["source"]["annotation_suffix"]
    legal_raw = frozenset(range(0, ADE20K_CLASS_COUNT + 1)) | {ADE20K_IGNORE_INDEX}

    presence: dict[str, frozenset[int]] = {}
    for image_id in image_ids:
        image_path = img_dir / f"{image_id}{image_suffix}"
        mask_path = mask_dir / f"{image_id}{annotation_suffix}"
        if not image_path.is_file():
            raise DfpPhase1aSubsetManifestError(f"missing validation image for id {image_id!r}: {image_path}")
        if not mask_path.is_file():
            raise DfpPhase1aSubsetManifestError(f"missing validation mask for id {image_id!r}: {mask_path}")
        image_wh = _decode_image_size(image_path, label="validation image")
        mask_arr = _decode_mask_array(mask_path, label="validation mask")
        mask_wh = (mask_arr.shape[1], mask_arr.shape[0])
        if image_wh != mask_wh:
            raise DfpPhase1aSubsetManifestError(f"image/mask dimension mismatch for id {image_id!r}: image={image_wh} mask={mask_wh}")
        raw_labels = set(np.unique(mask_arr).tolist())
        illegal = raw_labels - legal_raw
        if illegal:
            raise DfpPhase1aSubsetManifestError(f"mask {mask_path} contains illegal label value(s): {sorted(illegal)}")
        evaluated = frozenset(
            (c - 1) for c in raw_labels if c != ADE20K_RAW_IGNORE_VALUE and c != ADE20K_IGNORE_INDEX
        )
        presence[image_id] = evaluated
    return presence


# ---------------------------------------------------------------------------
# Tie-break scheme
# ---------------------------------------------------------------------------


def tie_break_hexdigest(dataset_key: str, canonical_key: str) -> str:
    payload = TIE_BREAK_NAMESPACE.encode("utf-8") + b"\x00" + dataset_key.encode("utf-8") + b"\x00" + canonical_key.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tie_break_int(dataset_key: str, canonical_key: str) -> int:
    return int(tie_break_hexdigest(dataset_key, canonical_key), 16)


def _label_tie_key(class_index: int) -> str:
    return f"class_index={class_index:04d}"


# ---------------------------------------------------------------------------
# Deterministic multilabel iterative stratification
# ---------------------------------------------------------------------------


def _deterministic_round_half(value: Fraction, *, dataset_key: str, class_index: int) -> int:
    floor_val = value.numerator // value.denominator
    remainder = value - floor_val
    half = Fraction(1, 2)
    if remainder > half:
        return floor_val + 1
    if remainder < half:
        return floor_val
    # exact one-half boundary: label tie-break parity
    return floor_val + 1 if (tie_break_int(dataset_key, _label_tie_key(class_index)) % 2 == 1) else floor_val


def compute_desired_selected_counts(
    class_totals: Sequence[int], *, source_count: int, selected_count: int, dataset_key: str,
) -> list[int]:
    return [
        _deterministic_round_half(Fraction(total * selected_count, source_count), dataset_key=dataset_key, class_index=c)
        for c, total in enumerate(class_totals)
    ]


def deterministic_iterative_stratify(
    *, canonical_ids: Sequence[str], presence: Mapping[str, frozenset[int]], class_count: int,
    selected_count: int, dataset_key: str,
) -> dict[str, Any]:
    """Returns a dict with: selected_ids (canonical order), remainder_ids
    (canonical order), class_totals, desired_selected_counts,
    selected_counts, remainder_counts. Depends only on the SET of
    canonical_ids and their presence vectors -- never on input traversal
    order (the caller may pass canonical_ids in any order; the returned
    selected/remainder ID lists are always reordered to the TRUE
    canonical order supplied via `true_canonical_order`)."""
    n = len(canonical_ids)
    if selected_count > n:
        raise DfpPhase1aSubsetManifestError(f"selected_count {selected_count} exceeds source_count {n}")

    class_totals = [0] * class_count
    for image_id in canonical_ids:
        for c in presence[image_id]:
            class_totals[c] += 1

    desired_selected = compute_desired_selected_counts(
        class_totals, source_count=n, selected_count=selected_count, dataset_key=dataset_key,
    )
    remaining_desired: dict[tuple[str, int], int] = {}
    for c in range(class_count):
        remaining_desired[("selected", c)] = desired_selected[c]
        remaining_desired[("remainder", c)] = class_totals[c] - desired_selected[c]

    target_size = {"selected": selected_count, "remainder": n - selected_count}
    assigned_count = {"selected": 0, "remainder": 0}
    fold_of: dict[str, str] = {}

    labeled_ids = [i for i in canonical_ids if presence[i]]
    empty_ids = [i for i in canonical_ids if not presence[i]]

    # Precompute image tie-break ints once.
    image_tie = {i: tie_break_int(dataset_key, i) for i in canonical_ids}

    # supply[c] = ids (as a set) with class c still unassigned.
    supply: dict[int, set[str]] = {c: set() for c in range(class_count)}
    for image_id in labeled_ids:
        for c in presence[image_id]:
            supply[c].add(image_id)

    unassigned = set(labeled_ids)

    def _pick_label() -> int | None:
        candidates = [(len(supply[c]), tie_break_int(dataset_key, _label_tie_key(c)), c) for c in range(class_count) if len(supply[c]) > 0]
        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[0][2]

    def _choose_fold(image_id: str, demand_selected: int, demand_remainder: int) -> str:
        # Hard capacity gate FIRST: this is the rule that guarantees the
        # exact frozen fold sizes for genuinely multilabel data. Per-class
        # desired counts are rounded independently and are not
        # constrained to sum to `selected_count` in aggregate (a sample
        # can carry several labels at once), so the published
        # iterative-stratification demand comparison alone only
        # approximately balances fold totals. Once a fold has received
        # its full target count it is excluded from all further
        # candidacy, regardless of any label's remaining demand --
        # capacity is checked before demand, every time.
        selected_has_capacity = assigned_count["selected"] < target_size["selected"]
        remainder_has_capacity = assigned_count["remainder"] < target_size["remainder"]
        if selected_has_capacity and not remainder_has_capacity:
            return "selected"
        if remainder_has_capacity and not selected_has_capacity:
            return "remainder"
        if not selected_has_capacity and not remainder_has_capacity:
            raise DfpPhase1aSubsetManifestError("internal stratification invariant violated: both folds report zero remaining capacity with unassigned samples left")
        if demand_selected > demand_remainder:
            return "selected"
        if demand_remainder > demand_selected:
            return "remainder"
        need_selected = target_size["selected"] - assigned_count["selected"]
        need_remainder = target_size["remainder"] - assigned_count["remainder"]
        if need_selected > need_remainder:
            return "selected"
        if need_remainder > need_selected:
            return "remainder"
        return "selected" if (image_tie[image_id] % 2 == 0) else "remainder"

    while True:
        label = _pick_label()
        if label is None:
            break
        members = sorted(supply[label], key=lambda i: image_tie[i])
        for image_id in members:
            if image_id not in unassigned:
                continue  # already assigned while processing an earlier label in this same round is impossible, but guard anyway
            chosen = _choose_fold(image_id, remaining_desired[("selected", label)], remaining_desired[("remainder", label)])

            fold_of[image_id] = chosen
            assigned_count[chosen] += 1
            unassigned.discard(image_id)
            for c in presence[image_id]:
                remaining_desired[(chosen, c)] -= 1
                supply[c].discard(image_id)

    for image_id in sorted(empty_ids, key=lambda i: image_tie[i]):
        chosen = _choose_fold(image_id, 0, 0)
        fold_of[image_id] = chosen
        assigned_count[chosen] += 1

    if assigned_count["selected"] != selected_count or assigned_count["remainder"] != (n - selected_count):
        raise DfpPhase1aSubsetManifestError(
            f"internal stratification invariant violated: assigned selected={assigned_count['selected']} "
            f"remainder={assigned_count['remainder']}, expected {selected_count}/{n - selected_count}"
        )

    selected_set = {i for i, f in fold_of.items() if f == "selected"}
    remainder_set = {i for i, f in fold_of.items() if f == "remainder"}

    selected_counts = [0] * class_count
    remainder_counts = [0] * class_count
    for image_id in selected_set:
        for c in presence[image_id]:
            selected_counts[c] += 1
    for image_id in remainder_set:
        for c in presence[image_id]:
            remainder_counts[c] += 1

    return {
        "selected_set": selected_set,
        "remainder_set": remainder_set,
        "class_totals": class_totals,
        "desired_selected_counts": desired_selected,
        "selected_counts": selected_counts,
        "remainder_counts": remainder_counts,
    }


def reorder_to_canonical(ids: set[str] | Sequence[str], true_canonical_order: Sequence[str]) -> list[str]:
    id_set = set(ids)
    return [i for i in true_canonical_order if i in id_set]


# ---------------------------------------------------------------------------
# Manifest schema: build, strict-parse, strict-verify, atomic install.
# ---------------------------------------------------------------------------

DATASET_CONTRACTS: dict[str, dict[str, Any]] = {
    "coco_stuff": {
        "run_mode": "coco_stuff_500",
        "split": "val2017",
        "source_identity_relative_path": "evaluation_identities/e12_matched_k11_k12_t320.toml",
        "class_count": COCO_STUFF_CLASS_COUNT,
        "source_validation_count": 5000,
        "ignore_policy": "raw_pixel_value_is_evaluation_label_0_170_ignore_255_no_reduce_zero_label",
        "source_root_logical_label": "coco_stuff164k",
        "source_manifest_kind": "logical_identity_section",
        "source_manifest_logical_label": "e12_matched_k11_k12_t320.toml [dataset]",
    },
    "ade20k": {
        "run_mode": "ade20k_500",
        "split": "validation",
        "source_identity_relative_path": "evaluation_identities/e12_ade20k_dataset_source.toml",
        "class_count": ADE20K_CLASS_COUNT,
        "source_validation_count": 2000,
        "ignore_policy": "reduce_zero_label_raw_0_ignored_raw_1_150_map_to_eval_0_149",
        "source_root_logical_label": "ADEChallengeData2016",
        "source_manifest_kind": "file_manifest",
        "source_manifest_logical_label": "ade20k_source_manifest.json (talk2dino-ade20k-dataset-manifest-v1)",
    },
}

MANIFEST_TOP_KEYS = frozenset(
    {
        "schema", "schema_version", "dataset_key", "split", "run_mode",
        "protocol_identity_relative_path", "protocol_identity_sha256",
        "source_identity_relative_path", "source_identity_sha256",
        "source_manifest_binding",
        "source_validation_count", "selected_count", "remainder_count",
        "class_count", "class_names_digest",
        "ignore_policy",
        "selection_algorithm_name", "selection_algorithm_version",
        "tie_break_namespace", "tie_break_encoding",
        "canonical_source_order_assertion",
        "selected_ids", "selected_ids_digest",
        "candidate_universe_digest", "presence_input_digest",
        "full_source_class_presence_counts", "selected_class_presence_counts", "remainder_class_presence_counts",
        "desired_selected_class_presence_counts",
        "selection_trace_digest",
        "no_predictions_or_features_used",
        "source_root_logical_label",
    }
)

SOURCE_MANIFEST_BINDING_KEYS = frozenset({"kind", "relative_path", "logical_label", "sha256"})


def presence_input_digest(canonical_ids: Sequence[str], presence: Mapping[str, frozenset[int]]) -> str:
    rows = [[image_id, sorted(presence[image_id])] for image_id in canonical_ids]
    return _ordered_digest(rows)


def candidate_universe_digest(canonical_ids: Sequence[str]) -> str:
    return _ordered_digest(list(canonical_ids))


def selected_ids_digest(selected_ids: Sequence[str]) -> str:
    return _ordered_digest(list(selected_ids))


def selection_trace_digest(
    *, desired_selected_counts: Sequence[int], selected_counts: Sequence[int], remainder_counts: Sequence[int],
) -> str:
    return _ordered_digest([list(desired_selected_counts), list(selected_counts), list(remainder_counts)])


def build_manifest(
    *, dataset_key: str, protocol_identity_sha256: str, source_identity_sha256: str,
    source_manifest_sha256: str | None, class_names_digest: str,
    canonical_ids: Sequence[str], presence: Mapping[str, frozenset[int]],
    stratify_result: Mapping[str, Any],
) -> dict[str, Any]:
    if dataset_key not in DATASET_CONTRACTS:
        raise DfpPhase1aSubsetManifestError(f"unknown dataset_key {dataset_key!r}")
    contract = DATASET_CONTRACTS[dataset_key]

    selected_ids = reorder_to_canonical(stratify_result["selected_set"], canonical_ids)
    if len(selected_ids) != SELECTED_COUNT:
        raise DfpPhase1aSubsetManifestError(f"internal error: selected {len(selected_ids)} ids, expected {SELECTED_COUNT}")
    remainder_ids = reorder_to_canonical(stratify_result["remainder_set"], canonical_ids)

    # Neither binding kind corresponds to a repository-tracked file: a
    # "logical_identity_section" binding IS the committed source
    # identity's own [dataset] section (no separate file), and a
    # "file_manifest" binding (ADE20K) is an externally-verified source
    # manifest that lives outside the repository -- its physical location
    # is deliberately never recorded, only its logical label and content
    # hash.
    source_manifest_binding: dict[str, Any] = {
        "kind": contract["source_manifest_kind"],
        "relative_path": "",
        "logical_label": contract["source_manifest_logical_label"],
        "sha256": source_manifest_sha256 if source_manifest_sha256 is not None else source_identity_sha256,
    }

    return {
        "schema": MANIFEST_SCHEMA_NAME,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_key": dataset_key,
        "split": contract["split"],
        "run_mode": contract["run_mode"],
        "protocol_identity_relative_path": PROTOCOL_IDENTITY_RELATIVE_PATH,
        "protocol_identity_sha256": protocol_identity_sha256,
        "source_identity_relative_path": contract["source_identity_relative_path"],
        "source_identity_sha256": source_identity_sha256,
        "source_manifest_binding": source_manifest_binding,
        "source_validation_count": contract["source_validation_count"],
        "selected_count": SELECTED_COUNT,
        "remainder_count": contract["source_validation_count"] - SELECTED_COUNT,
        "class_count": contract["class_count"],
        "class_names_digest": class_names_digest,
        "ignore_policy": contract["ignore_policy"],
        "selection_algorithm_name": ALGORITHM_NAME,
        "selection_algorithm_version": ALGORITHM_VERSION,
        "tie_break_namespace": TIE_BREAK_NAMESPACE,
        "tie_break_encoding": TIE_BREAK_ENCODING,
        "canonical_source_order_assertion": CANONICAL_SOURCE_ORDER_ASSERTION,
        "selected_ids": selected_ids,
        "selected_ids_digest": selected_ids_digest(selected_ids),
        "candidate_universe_digest": candidate_universe_digest(canonical_ids),
        "presence_input_digest": presence_input_digest(canonical_ids, presence),
        "full_source_class_presence_counts": list(stratify_result["class_totals"]),
        "selected_class_presence_counts": list(stratify_result["selected_counts"]),
        "remainder_class_presence_counts": list(stratify_result["remainder_counts"]),
        "desired_selected_class_presence_counts": list(stratify_result["desired_selected_counts"]),
        "selection_trace_digest": selection_trace_digest(
            desired_selected_counts=stratify_result["desired_selected_counts"],
            selected_counts=stratify_result["selected_counts"],
            remainder_counts=stratify_result["remainder_counts"],
        ),
        "no_predictions_or_features_used": True,
        "source_root_logical_label": contract["source_root_logical_label"],
    }


# ---------------------------------------------------------------------------
# Strict JSON I/O
# ---------------------------------------------------------------------------


def _closed_object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DfpPhase1aSubsetManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> Any:
    raise DfpPhase1aSubsetManifestError(f"document contains non-finite JSON constant {token}")


def parse_strict_json_document(path: Path, *, label: str) -> Any:
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except IsADirectoryError as error:
        raise DfpPhase1aSubsetManifestError(f"cannot read {label} {path}: path is a directory") from error
    except PermissionError as error:
        raise DfpPhase1aSubsetManifestError(f"cannot read {label} {path}: permission denied") from error
    except FileNotFoundError as error:
        raise DfpPhase1aSubsetManifestError(f"cannot read {label} {path}: file does not exist") from error
    except OSError as error:
        raise DfpPhase1aSubsetManifestError(f"cannot read {label} {path}: {error}") from error

    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise DfpPhase1aSubsetManifestError(f"{label} {path} is not valid UTF-8: {error}") from error
    if not text.strip():
        raise DfpPhase1aSubsetManifestError(f"{label} {path} is empty")

    try:
        value = json.loads(text, object_pairs_hook=_closed_object_pairs_hook, parse_constant=_reject_non_finite)
    except DfpPhase1aSubsetManifestError:
        raise
    except json.JSONDecodeError as error:
        raise DfpPhase1aSubsetManifestError(f"cannot parse {label} {path}: {error}") from error

    if not isinstance(value, Mapping):
        raise DfpPhase1aSubsetManifestError(f"{label} {path} must contain one JSON object at its root, observed {type(value).__name__}")
    return value


def write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp_path = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Strict manifest verification -- never mutates its input.
# ---------------------------------------------------------------------------


def _require_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise DfpPhase1aSubsetManifestError(f"{label} must be an exact string")
    if nonempty and not value.strip():
        raise DfpPhase1aSubsetManifestError(f"{label} must be a non-empty, non-whitespace string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DfpPhase1aSubsetManifestError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1aSubsetManifestError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1aSubsetManifestError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aSubsetManifestError(f"{label} must be a lowercase SHA256")
    return token


def _require_relative_path(value: Any, label: str, *, allow_empty: bool = False) -> str:
    token = _require_string(value, label, nonempty=not allow_empty)
    if token == "" and allow_empty:
        return token
    path = Path(token)
    if path.is_absolute() or ".." in path.parts or "\\" in token:
        raise DfpPhase1aSubsetManifestError(f"{label} must be a safe repository-relative path")
    return token


def _require_int_list(value: Any, label: str, *, length: int | None = None) -> list[int]:
    if type(value) is not list:
        raise DfpPhase1aSubsetManifestError(f"{label} must be an exact list")
    if length is not None and len(value) != length:
        raise DfpPhase1aSubsetManifestError(f"{label} must have exactly {length} elements, observed {len(value)}")
    for index, item in enumerate(value):
        if type(item) is not int:
            raise DfpPhase1aSubsetManifestError(f"{label}[{index}] must be an exact non-boolean integer")
    return value


def _require_string_list(value: Any, label: str) -> list[str]:
    if type(value) is not list:
        raise DfpPhase1aSubsetManifestError(f"{label} must be an exact list")
    for index, item in enumerate(value):
        if type(item) is not str:
            raise DfpPhase1aSubsetManifestError(f"{label}[{index}] must be an exact string")
    return value


def verify_manifest_document(
    manifest: Any, *, dataset_key_expected: str, protocol_identity_sha256: str, source_identity_sha256: str,
    class_names_digest: str,
) -> None:
    """Strict, non-mutating verification of a subset manifest document
    against the live parent identities. Does NOT recompute the selection
    from raw pixel data (that is `verify_manifest_reconciles_with_selection`
    below) -- this function checks internal/type/schema/relational
    consistency only."""
    if not isinstance(manifest, Mapping) or set(manifest) != MANIFEST_TOP_KEYS:
        missing = MANIFEST_TOP_KEYS - set(manifest) if isinstance(manifest, Mapping) else MANIFEST_TOP_KEYS
        unknown = set(manifest) - MANIFEST_TOP_KEYS if isinstance(manifest, Mapping) else set()
        raise DfpPhase1aSubsetManifestError(f"manifest has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")

    if manifest["schema"] != MANIFEST_SCHEMA_NAME:
        raise DfpPhase1aSubsetManifestError("manifest.schema must be exactly the frozen schema name")
    if _require_int(manifest["schema_version"], "manifest.schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DfpPhase1aSubsetManifestError("manifest.schema_version must be exactly the frozen schema version")

    dataset_key = _require_string(manifest["dataset_key"], "manifest.dataset_key")
    if dataset_key != dataset_key_expected:
        raise DfpPhase1aSubsetManifestError(f"manifest.dataset_key {dataset_key!r} disagrees with expected {dataset_key_expected!r}")
    if dataset_key not in DATASET_CONTRACTS:
        raise DfpPhase1aSubsetManifestError(f"manifest.dataset_key {dataset_key!r} is not a recognized dataset")
    contract = DATASET_CONTRACTS[dataset_key]

    if manifest["split"] != contract["split"]:
        raise DfpPhase1aSubsetManifestError("manifest.split disagrees with the frozen dataset contract")
    if manifest["run_mode"] != contract["run_mode"]:
        raise DfpPhase1aSubsetManifestError("manifest.run_mode disagrees with the frozen dataset contract")

    if manifest["protocol_identity_relative_path"] != PROTOCOL_IDENTITY_RELATIVE_PATH:
        raise DfpPhase1aSubsetManifestError("manifest.protocol_identity_relative_path disagrees with the frozen path")
    if _require_sha256(manifest["protocol_identity_sha256"], "manifest.protocol_identity_sha256") != protocol_identity_sha256:
        raise DfpPhase1aSubsetManifestError("manifest.protocol_identity_sha256 does not match the live protocol identity")

    if manifest["source_identity_relative_path"] != contract["source_identity_relative_path"]:
        raise DfpPhase1aSubsetManifestError("manifest.source_identity_relative_path disagrees with the frozen dataset contract")
    if _require_sha256(manifest["source_identity_sha256"], "manifest.source_identity_sha256") != source_identity_sha256:
        raise DfpPhase1aSubsetManifestError("manifest.source_identity_sha256 does not match the live source identity")

    binding = manifest["source_manifest_binding"]
    if not isinstance(binding, Mapping) or set(binding) != SOURCE_MANIFEST_BINDING_KEYS:
        raise DfpPhase1aSubsetManifestError("manifest.source_manifest_binding has an unexpected schema")
    if binding["kind"] != contract["source_manifest_kind"]:
        raise DfpPhase1aSubsetManifestError("manifest.source_manifest_binding.kind disagrees with the frozen dataset contract")
    _require_relative_path(binding["relative_path"], "manifest.source_manifest_binding.relative_path", allow_empty=True)
    if binding["logical_label"] != contract["source_manifest_logical_label"]:
        raise DfpPhase1aSubsetManifestError("manifest.source_manifest_binding.logical_label disagrees with the frozen dataset contract")
    _require_sha256(binding["sha256"], "manifest.source_manifest_binding.sha256")

    if _require_int(manifest["source_validation_count"], "manifest.source_validation_count", minimum=1) != contract["source_validation_count"]:
        raise DfpPhase1aSubsetManifestError("manifest.source_validation_count disagrees with the frozen dataset contract")
    if _require_int(manifest["selected_count"], "manifest.selected_count", minimum=1) != SELECTED_COUNT:
        raise DfpPhase1aSubsetManifestError(f"manifest.selected_count must be exactly {SELECTED_COUNT}")
    remainder_count = _require_int(manifest["remainder_count"], "manifest.remainder_count", minimum=0)
    if manifest["selected_count"] + remainder_count != manifest["source_validation_count"]:
        raise DfpPhase1aSubsetManifestError("manifest.selected_count + manifest.remainder_count != manifest.source_validation_count")
    if remainder_count != contract["source_validation_count"] - SELECTED_COUNT:
        raise DfpPhase1aSubsetManifestError("manifest.remainder_count disagrees with the frozen dataset contract")

    class_count = _require_int(manifest["class_count"], "manifest.class_count", minimum=1)
    if class_count != contract["class_count"]:
        raise DfpPhase1aSubsetManifestError("manifest.class_count disagrees with the frozen dataset contract")
    if _require_sha256(manifest["class_names_digest"], "manifest.class_names_digest") != class_names_digest:
        raise DfpPhase1aSubsetManifestError("manifest.class_names_digest does not match the live source identity")

    if manifest["ignore_policy"] != contract["ignore_policy"]:
        raise DfpPhase1aSubsetManifestError("manifest.ignore_policy disagrees with the frozen dataset contract")

    if manifest["selection_algorithm_name"] != ALGORITHM_NAME:
        raise DfpPhase1aSubsetManifestError("manifest.selection_algorithm_name must be exactly the frozen algorithm name")
    if manifest["selection_algorithm_version"] != ALGORITHM_VERSION:
        raise DfpPhase1aSubsetManifestError("manifest.selection_algorithm_version must be exactly the frozen algorithm version")
    if manifest["tie_break_namespace"] != TIE_BREAK_NAMESPACE:
        raise DfpPhase1aSubsetManifestError("manifest.tie_break_namespace must be exactly the frozen namespace")
    if manifest["tie_break_encoding"] != TIE_BREAK_ENCODING:
        raise DfpPhase1aSubsetManifestError("manifest.tie_break_encoding must be exactly the frozen encoding")
    if _require_string(manifest["canonical_source_order_assertion"], "manifest.canonical_source_order_assertion") != CANONICAL_SOURCE_ORDER_ASSERTION:
        raise DfpPhase1aSubsetManifestError("manifest.canonical_source_order_assertion disagrees with the frozen canonical text")

    selected_ids = _require_string_list(manifest["selected_ids"], "manifest.selected_ids")
    if len(selected_ids) != SELECTED_COUNT:
        raise DfpPhase1aSubsetManifestError(f"manifest.selected_ids must contain exactly {SELECTED_COUNT} entries")
    if len(set(selected_ids)) != len(selected_ids):
        raise DfpPhase1aSubsetManifestError("manifest.selected_ids contains a duplicate ID")
    if _require_sha256(manifest["selected_ids_digest"], "manifest.selected_ids_digest") != selected_ids_digest(selected_ids):
        raise DfpPhase1aSubsetManifestError("manifest.selected_ids_digest does not match manifest.selected_ids")
    # Canonical order enforcement: both frozen dataset schemas' image IDs
    # are fixed-width, so lexicographic string order is exactly their
    # authoritative canonical validation order (confirmed relationally --
    # ADE20K's own committed source identity states this convention in
    # [source].split_order_authority; the live installed mmseg
    # CustomDataset.load_annotations, with split=None as COCO-Stuff is
    # actually configured, sorts by filename). selected_ids must already
    # BE in that order -- never silently sorted here; a reordered list
    # (even with an internally-recomputed digest, correct counts, and no
    # duplicates) fails closed, because selection order must never become
    # evaluation order.
    if selected_ids != sorted(selected_ids):
        raise DfpPhase1aSubsetManifestError(
            "manifest.selected_ids is not in canonical source order (lexicographic sort of the fixed-width image IDs)"
        )

    _require_sha256(manifest["candidate_universe_digest"], "manifest.candidate_universe_digest")
    _require_sha256(manifest["presence_input_digest"], "manifest.presence_input_digest")

    full_counts = _require_int_list(manifest["full_source_class_presence_counts"], "manifest.full_source_class_presence_counts", length=class_count)
    selected_counts = _require_int_list(manifest["selected_class_presence_counts"], "manifest.selected_class_presence_counts", length=class_count)
    remainder_counts = _require_int_list(manifest["remainder_class_presence_counts"], "manifest.remainder_class_presence_counts", length=class_count)
    desired_counts = _require_int_list(manifest["desired_selected_class_presence_counts"], "manifest.desired_selected_class_presence_counts", length=class_count)
    for c in range(class_count):
        if selected_counts[c] + remainder_counts[c] != full_counts[c]:
            raise DfpPhase1aSubsetManifestError(f"manifest per-class counts disagree at class {c}: selected+remainder != full_source")
        if selected_counts[c] < 0 or remainder_counts[c] < 0 or full_counts[c] < 0:
            raise DfpPhase1aSubsetManifestError(f"manifest per-class counts must be non-negative at class {c}")

    if _require_sha256(manifest["selection_trace_digest"], "manifest.selection_trace_digest") != selection_trace_digest(
        desired_selected_counts=desired_counts, selected_counts=selected_counts, remainder_counts=remainder_counts,
    ):
        raise DfpPhase1aSubsetManifestError("manifest.selection_trace_digest disagrees with the manifest's own per-class counts")

    if _require_bool(manifest["no_predictions_or_features_used"], "manifest.no_predictions_or_features_used") is not True:
        raise DfpPhase1aSubsetManifestError("manifest.no_predictions_or_features_used must be true")

    source_root_label = _require_string(manifest["source_root_logical_label"], "manifest.source_root_logical_label")
    if "/" in source_root_label or "\\" in source_root_label or Path(source_root_label).is_absolute():
        raise DfpPhase1aSubsetManifestError("manifest.source_root_logical_label must not embed a filesystem path")
    if source_root_label != contract["source_root_logical_label"]:
        raise DfpPhase1aSubsetManifestError("manifest.source_root_logical_label disagrees with the frozen dataset contract")


def verify_manifest_reconciles_with_selection(
    manifest: Mapping[str, Any], *, canonical_ids: Sequence[str], presence: Mapping[str, frozenset[int]],
) -> None:
    """Independent recomputation check: given the raw candidate universe
    and presence vectors, re-run the deterministic algorithm from scratch
    and require every digest and the selected-ID list itself to match the
    manifest exactly. This is the authoritative defense against a
    manifest containing a selected ID that deterministic recomputation
    would not itself have produced."""
    dataset_key = manifest["dataset_key"]
    class_count = manifest["class_count"]

    if candidate_universe_digest(canonical_ids) != manifest["candidate_universe_digest"]:
        raise DfpPhase1aSubsetManifestError("recomputed candidate_universe_digest disagrees with the manifest")
    if presence_input_digest(canonical_ids, presence) != manifest["presence_input_digest"]:
        raise DfpPhase1aSubsetManifestError("recomputed presence_input_digest disagrees with the manifest")

    result = deterministic_iterative_stratify(
        canonical_ids=canonical_ids, presence=presence, class_count=class_count,
        selected_count=manifest["selected_count"], dataset_key=dataset_key,
    )
    recomputed_selected = reorder_to_canonical(result["selected_set"], canonical_ids)
    if recomputed_selected != manifest["selected_ids"]:
        raise DfpPhase1aSubsetManifestError(
            "deterministic recomputation from the candidate universe/presence data does not reproduce "
            "manifest.selected_ids -- at least one selected ID was not produced by the frozen algorithm"
        )


__all__ = [
    "TIE_BREAK_NAMESPACE",
    "TIE_BREAK_ENCODING",
    "ALGORITHM_NAME",
    "ALGORITHM_VERSION",
    "MANIFEST_SCHEMA_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "SELECTED_COUNT",
    "PROTOCOL_IDENTITY_RELATIVE_PATH",
    "COCO_STUFF_CLASS_COUNT",
    "COCO_STUFF_IGNORE_INDEX",
    "COCO_STUFF_IMAGE_SUFFIX",
    "COCO_STUFF_ANNOTATION_SUFFIX",
    "COCO_STUFF_LEGAL_RAW_LABELS",
    "ADE20K_CLASS_COUNT",
    "ADE20K_RAW_IGNORE_VALUE",
    "ADE20K_IGNORE_INDEX",
    "DfpPhase1aSubsetManifestError",
    "require_env_root",
    "resolve_coco_stuff_root",
    "coco_stuff_canonical_validation_ids",
    "coco_stuff_class_presence",
    "ade20k_class_presence",
    "sha256_file",
    "tie_break_hexdigest",
    "tie_break_int",
    "compute_desired_selected_counts",
    "deterministic_iterative_stratify",
    "reorder_to_canonical",
    "DATASET_CONTRACTS",
    "MANIFEST_TOP_KEYS",
    "SOURCE_MANIFEST_BINDING_KEYS",
    "presence_input_digest",
    "candidate_universe_digest",
    "selected_ids_digest",
    "selection_trace_digest",
    "build_manifest",
    "parse_strict_json_document",
    "write_json_atomically",
    "verify_manifest_document",
    "verify_manifest_reconciles_with_selection",
]
