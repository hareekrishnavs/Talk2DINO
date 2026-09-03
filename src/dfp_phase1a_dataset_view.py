"""DFP Phase 1A runtime dataset adapter: maps the frozen 500-image
manifest order onto the native (5,000-image) mmseg validation dataset.

**Root cause this module closes.** The real mmseg dataset (``build_seg_
dataset``, e.g. ``mmseg.datasets.coco_stuff.COCOStuffDataset``) enumerates
the *complete* COCO-Stuff val2017 universe (5,000 images) in its own
native, on-disk-listing order. The frozen DFP Phase 1A subset manifest
(``evaluation_subsets/dfp_phase1a_coco_stuff_500.json``) selects exactly
500 of those images, in its own canonical selection order -- a
*different* order from the dataset's native one. Manifest position ``i``
is not, and must never be assumed to be, native dataset index ``i``.

Before this module existed, ``src/dfp_phase1a_execute.py`` passed the
manifest position directly as the native dataset index (``dataset[i]``,
``dataset.get_gt_seg_map_by_idx(i)``), which happened to be caught --
not silently corrupted -- by the pre-existing pipeline/canonical-ID
reconciliation guard in ``diagnostics.run_k11_k12_stability._extract_
prepared_image`` (via ``src.dataset_image_identity.
reconcile_canonical_image_id``): SLURM job 21055181 failed cleanly with
exactly this mismatch (manifest position 0 -> expected canonical ID
``000000000724``, but native dataset index 0 is actually
``000000000139.jpg``). This module does not touch, weaken, or duplicate
that guard -- it supplies it with the CORRECT native index in the first
place, so the reconciliation it already performs succeeds against the
real dataset instead of failing against a wrong index.

**Canonical-ID/suffix authority.** The bare, frozen COCO-Stuff manifest
ID (e.g. ``"000000000724"``, no extension) and the mmseg dataset's own
``img_infos[i]["filename"]`` (e.g. ``"000000000139.jpg"``, WITH
extension) are related by exactly one authority: the live, already-
identity-bound dataset object's own ``img_suffix`` attribute (mmseg's
``CustomDataset.img_suffix``, set by the E3 identity's own hash-pinned
dataset config file -- see ``diagnostics.run_k11_k12_stability.
resolve_e3_dataset_config_path``). This module reads ``dataset.
img_suffix`` LIVE off the constructed object -- it is never
hardcoded here (never assumes ``.jpg``), and it is never independently
re-derived from a filesystem listing, a manifest field (the manifest
carries no path/suffix field at all), or a second parse of the dataset
config file.

**What this module does not do.** It does not construct, filter, wrap,
or mutate the underlying mmseg dataset -- ``dataset[native_index]`` and
``dataset.get_gt_seg_map_by_idx(native_index)`` are still called
directly, by the driver, on the exact same unwrapped object ``_build_
inference`` already returns. It does not change the frozen 500 selected
IDs, their order, or their digest. It does not implement a new sampling
algorithm -- selection already happened once, in Step 2
(``dfp_phase1a_subset_manifest.py``); this module only resolves already-
selected IDs to native positions in an already-built dataset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.dataset_image_identity import reconcile_canonical_image_id


class DfpPhase1aDatasetViewError(ValueError):
    """Raised on any native-index mapping, canonical-ID, or path-safety
    violation while building or validating the manifest-ordered
    execution view. A `ValueError` subclass, so it is caught by the
    unified CLI's existing exception boundary without any change there."""


_CANONICAL_ID_LENGTH = 12  # frozen COCO image-ID format: exactly 12 zero-padded digits


def _require_canonical_id_format(canonical_id: str, *, label: str) -> None:
    if type(canonical_id) is not str:
        raise DfpPhase1aDatasetViewError(f"{label} must be an exact str, observed {type(canonical_id).__name__}")
    if len(canonical_id) != _CANONICAL_ID_LENGTH or not canonical_id.isdigit():
        raise DfpPhase1aDatasetViewError(
            f"{label} {canonical_id!r} does not match the frozen COCO identifier format "
            f"(exactly {_CANONICAL_ID_LENGTH} decimal digits, zero-padded)"
        )


def _resolve_image_root(dataset: Any) -> str:
    """Identical fallback to ``diagnostics.run_k11_k12_stability.
    _extract_prepared_image``'s own (undocumented-elsewhere) derivation,
    deliberately mirrored here rather than imported (that function is
    private, shared infrastructure used by several unrelated diagnostics
    outside DFP's scope) -- never independently invented: ``dataset.
    img_dir`` when present and non-empty, else ``"."`` for a dataset that
    models no directory structure at all. Both call sites therefore
    agree on the same image root for the same dataset object, and the
    per-image runtime guard (which calls ``_extract_prepared_image``
    with no ``image_root`` override) always reconciles against the
    identical root this module used to build the map."""
    if hasattr(dataset, "img_dir"):
        candidate_root = dataset.img_dir
        if candidate_root is None or (isinstance(candidate_root, str) and not candidate_root.strip()):
            raise DfpPhase1aDatasetViewError("dataset.img_dir is present but empty/None; cannot resolve an image root")
        return candidate_root
    return "."


@dataclass(frozen=True)
class NativeImageRecord:
    """One native mmseg dataset entry, identity-reconciled."""

    canonical_id: str
    native_index: int
    relative_path: str  # e.g. "000000000139.jpg" -- WITH the authoritative suffix
    resolved_path: str  # lexically resolved image_root / relative_path


def build_native_image_index(dataset: Any) -> dict[str, NativeImageRecord]:
    """Scan the already-constructed mmseg dataset's own metadata, in
    native dataset-index order, and return ``{canonical_id:
    NativeImageRecord}`` for every entry. Reads only identity metadata
    (``img_infos``/``data_infos`` filenames, ``img_suffix``, ``img_dir``)
    -- never runs the pipeline, never touches pixel data, never loads a
    model. Fails closed (``DfpPhase1aDatasetViewError``) on any
    malformed, ambiguous, duplicate, unsafe, or wrongly-suffixed entry;
    never silently skips or coerces one."""
    infos = getattr(dataset, "img_infos", None)
    if infos is None:
        infos = getattr(dataset, "data_infos", None)
    if infos is None:
        raise DfpPhase1aDatasetViewError(
            "dataset exposes neither 'img_infos' nor 'data_infos' -- cannot derive the native image-identity index"
        )
    if not isinstance(infos, (list, tuple)):
        raise DfpPhase1aDatasetViewError(f"dataset metadata container must be a list or tuple, observed {type(infos).__name__}")

    img_suffix = getattr(dataset, "img_suffix", None)
    if type(img_suffix) is not str or not img_suffix:
        raise DfpPhase1aDatasetViewError(
            f"dataset.img_suffix must be a non-empty exact str (the sole suffix authority for this dataset); "
            f"observed {img_suffix!r}"
        )

    image_root = _resolve_image_root(dataset)

    by_id: dict[str, NativeImageRecord] = {}
    seen_paths: dict[str, int] = {}
    for native_index, info in enumerate(infos):
        if not hasattr(info, "get") and not hasattr(info, "__getitem__"):
            raise DfpPhase1aDatasetViewError(f"dataset metadata[{native_index}] is not a mapping")
        try:
            filename = info["filename"]
        except (KeyError, TypeError) as error:
            raise DfpPhase1aDatasetViewError(f"dataset metadata[{native_index}] is missing a 'filename' entry") from error

        if type(filename) is not str or not filename.strip():
            raise DfpPhase1aDatasetViewError(
                f"dataset metadata[{native_index}]['filename'] must be a non-empty exact str, observed {filename!r}"
            )
        if not filename.endswith(img_suffix):
            raise DfpPhase1aDatasetViewError(
                f"dataset metadata[{native_index}]['filename']={filename!r} does not end with the dataset's own "
                f"authoritative img_suffix {img_suffix!r}"
            )
        canonical_id = filename[: -len(img_suffix)]
        _require_canonical_id_format(canonical_id, label=f"dataset metadata[{native_index}]-derived canonical ID")

        # Reuse the existing, already-verified path-safety/traversal
        # authority verbatim (never a second, independently reimplemented
        # traversal check): this call is self-consistent by construction
        # (the "pipeline-resolved" argument is the same joined-and-
        # resolved path this function itself would compute), so it can
        # only ever raise on a genuinely unsafe filename (absolute,
        # '..'-containing, or resolving outside image_root) -- exactly
        # the same rejection this function's own docstring documents.
        resolved_path = str(Path(image_root) / filename)
        reconcile_canonical_image_id(
            canonical_relative_id=filename, image_root=image_root, pipeline_resolved_filename=resolved_path,
        )

        if canonical_id in by_id:
            raise DfpPhase1aDatasetViewError(
                f"duplicate canonical ID {canonical_id!r}: native indices {by_id[canonical_id].native_index} "
                f"and {native_index} both resolve to it"
            )
        normalized_resolved = str(Path(resolved_path).resolve(strict=False))
        if normalized_resolved in seen_paths:
            raise DfpPhase1aDatasetViewError(
                f"duplicate resolved physical path {normalized_resolved!r}: native indices "
                f"{seen_paths[normalized_resolved]} and {native_index} both resolve to it"
            )
        seen_paths[normalized_resolved] = native_index

        by_id[canonical_id] = NativeImageRecord(
            canonical_id=canonical_id, native_index=native_index, relative_path=filename, resolved_path=normalized_resolved,
        )

    return by_id


@dataclass(frozen=True)
class ManifestExecutionPosition:
    """One frozen-manifest row, resolved to its native dataset position."""

    manifest_position: int
    canonical_id: str
    native_index: int
    relative_path: str
    resolved_path: str


@dataclass(frozen=True)
class ManifestExecutionView:
    """The complete, order-preserving manifest -> native-index mapping,
    fully validated before any inference is authorized to start."""

    positions: tuple[ManifestExecutionPosition, ...]
    mapping_digest: str

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, manifest_position: int) -> ManifestExecutionPosition:
        return self.positions[manifest_position]


def compute_mapping_digest(positions: Sequence[ManifestExecutionPosition]) -> str:
    """Deterministic SHA-256 over the ordered mapping -- independent of
    ``PYTHONHASHSEED`` (plain list-of-dicts JSON with fixed key order,
    never a set/dict-iteration-order-dependent structure). Used for logs
    and independent verification only; not a schema-v2 member."""
    payload = [
        {
            "manifest_position": p.manifest_position,
            "canonical_id": p.canonical_id,
            "native_index": p.native_index,
            "relative_path": p.relative_path,
        }
        for p in positions
    ]
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest_execution_view(dataset: Any, selected_ids: Sequence[str]) -> ManifestExecutionView:
    """Build and fully validate the manifest-ordered execution view: for
    every ``selected_ids[position]`` (already frozen, already canonical-
    order, never re-selected or re-sorted here), resolve its native
    dataset index via :func:`build_native_image_index`, in manifest
    order. Raises :class:`DfpPhase1aDatasetViewError` -- before returning
    anything, i.e. before any image is authorized to be processed -- on:
    a missing selected ID (not present in the native dataset at all), a
    duplicate selected ID in the input sequence itself, or any native-
    index-mapping violation surfaced by :func:`build_native_image_index`.
    Never mutates ``selected_ids`` or ``dataset``."""
    if type(selected_ids) not in (list, tuple):
        raise DfpPhase1aDatasetViewError(f"selected_ids must be a list or tuple, observed {type(selected_ids).__name__}")

    native_index_by_id = build_native_image_index(dataset)

    seen_selected: set[str] = set()
    positions: list[ManifestExecutionPosition] = []
    for manifest_position, canonical_id in enumerate(selected_ids):
        _require_canonical_id_format(canonical_id, label=f"selected_ids[{manifest_position}]")
        if canonical_id in seen_selected:
            raise DfpPhase1aDatasetViewError(f"selected_ids[{manifest_position}]={canonical_id!r} is a duplicate of an earlier manifest position")
        seen_selected.add(canonical_id)

        record = native_index_by_id.get(canonical_id)
        if record is None:
            raise DfpPhase1aDatasetViewError(
                f"selected_ids[{manifest_position}]={canonical_id!r} has no corresponding entry in the native dataset "
                f"(native dataset has {len(native_index_by_id)} unique canonical IDs)"
            )
        positions.append(
            ManifestExecutionPosition(
                manifest_position=manifest_position, canonical_id=canonical_id, native_index=record.native_index,
                relative_path=record.relative_path, resolved_path=record.resolved_path,
            )
        )

    positions_tuple = tuple(positions)
    return ManifestExecutionView(positions=positions_tuple, mapping_digest=compute_mapping_digest(positions_tuple))


def verify_position_identity(
    dataset: Any, *, manifest_position: int, canonical_id: str, native_index: int,
) -> None:
    """Per-image runtime re-verification (Section 7's "three identities
    agree" guard), called immediately before ``_extract_prepared_image``
    for this exact image, every image, never skipped: independently re-
    derives the canonical ID from ``dataset.img_infos[native_index]``
    (or ``data_infos``) at CALL time -- not merely trusting the value
    already computed once when the view was built -- and requires it to
    still equal ``canonical_id`` exactly. Cheap (one dict/attribute
    lookup, no I/O); catches the dataset object having been mutated or
    substituted between view construction and this specific image's
    processing. Accepts raw fields (not a :class:`ManifestExecutionPosition`)
    so it can be called equally from the view-construction call site and
    from ``_process_one_image`` itself using its own already-validated
    parameters. Does not itself check the pipeline-resolved physical
    filename -- that reconciliation already happens, unconditionally,
    inside ``_extract_prepared_image`` via ``reconcile_canonical_image_id``,
    immediately after this guard returns and strictly before any window/
    graph/recurrence work."""
    infos = getattr(dataset, "img_infos", None)
    if infos is None:
        infos = getattr(dataset, "data_infos", None)
    if infos is None or native_index >= len(infos):
        raise DfpPhase1aDatasetViewError(
            f"manifest position {manifest_position}: native index {native_index} is no longer "
            f"valid against the current dataset metadata (length {0 if infos is None else len(infos)})"
        )
    img_suffix = getattr(dataset, "img_suffix", None)
    try:
        filename = infos[native_index]["filename"]
    except (KeyError, TypeError) as error:
        raise DfpPhase1aDatasetViewError(
            f"manifest position {manifest_position}: dataset metadata[{native_index}] is missing 'filename'"
        ) from error
    if type(img_suffix) is not str or type(filename) is not str or not filename.endswith(img_suffix):
        raise DfpPhase1aDatasetViewError(
            f"manifest position {manifest_position}: dataset metadata[{native_index}]['filename']="
            f"{filename!r} disagrees with the dataset's own img_suffix {img_suffix!r} at re-verification time"
        )
    metadata_canonical_id = filename[: -len(img_suffix)]
    if metadata_canonical_id != canonical_id:
        raise DfpPhase1aDatasetViewError(
            f"manifest position {manifest_position}: expected canonical ID {canonical_id!r} "
            f"(frozen manifest) but native index {native_index} now yields metadata filename {filename!r} "
            f"(canonical ID {metadata_canonical_id!r}) -- dataset ordering changed since the execution view was built"
        )


__all__ = [
    "DfpPhase1aDatasetViewError",
    "NativeImageRecord",
    "ManifestExecutionPosition",
    "ManifestExecutionView",
    "build_native_image_index",
    "build_manifest_execution_view",
    "compute_mapping_digest",
    "verify_position_identity",
]
