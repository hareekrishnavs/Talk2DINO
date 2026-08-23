"""Canonical dataset-relative image-identity reconciliation.

The canonical scientific image ID for any dataset sample is always
``dataset.img_infos[index]["filename"]`` (or ``data_infos``) -- the raw,
dataset-relative annotation filename mmseg's dataset classes load directly
from the annotation index. It is never re-derived from pipeline metadata.

The mmseg test pipeline's own ``img_meta["filename"]`` is a *resolved
physical path* (``dataset.img_dir`` joined with that same relative
filename by ``LoadImageFromFile``) -- useful only as provenance to confirm
the tensor a given dataset index actually produced was loaded from the
expected file, never as a second, independent source of persisted
identity. Treating it as one (as ``diagnostics.run_k11_k12_stability``
previously did) silently persists an absolute/resolved filesystem path as
``image_id`` instead of the canonical relative one, and on the real
COCO-Stuff dataset (``img_dir='./data/coco_stuff164k/images/val2017'``)
that resolved path never matches the bare-filename canonical order the
evaluator's own ``_expected_image_ids`` derives -- surfacing as a
same-image false-mismatch the very first time this code ran against the
real dataset rather than a synthetic fixture (job 20340482).

:func:`reconcile_canonical_image_id` is shared by both
``diagnostics/run_matched_k11_k12_evaluation.py`` and
``diagnostics/run_k11_k12_stability.py`` (via ``_extract_prepared_image``)
so this reconciliation is never independently reimplemented at either call
site.
"""

from __future__ import annotations

import os
import posixpath
from pathlib import Path
from typing import Union

from src.k11_k12_stability_gate_identity import K11K12StabilityGateError

PathLike = Union[str, "os.PathLike[str]"]


def _require_path_text(value: PathLike, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise K11K12StabilityGateError(
            f"{label} must be a string or path-like value, observed {type(value).__name__}"
        )
    text = os.fspath(value)
    if not isinstance(text, str):
        raise K11K12StabilityGateError(f"{label} must resolve to a str path, not bytes")
    if not text.strip():
        raise K11K12StabilityGateError(f"{label} must be a non-empty, non-whitespace string")
    if "\x00" in text:
        raise K11K12StabilityGateError(f"{label} must not contain a NUL byte")
    return text


def _resolve_lexical(path_text: str) -> Path:
    # strict=False: never requires the file to exist -- this is a pure
    # path-syntax reconciliation, exercised identically in unit tests
    # against nonexistent synthetic paths and in production against real
    # files already guaranteed to exist by the pipeline having just loaded
    # them.
    return Path(path_text).resolve(strict=False)


def reconcile_canonical_image_id(
    *,
    canonical_relative_id: PathLike,
    image_root: PathLike,
    pipeline_resolved_filename: PathLike,
) -> str:
    """Verify a pipeline-resolved physical path actually corresponds to the
    canonical dataset-relative image ID, and return that canonical ID
    completely unchanged -- never the resolved physical path, never a
    basename-reduced or otherwise normalized variant.

    Fails closed with :class:`K11K12StabilityGateError` (already caught by
    both the evaluator's and the stability-gate script's own top-level
    exception boundary, since it is a `ValueError` subclass and this
    module is already a dependency of both) on: a non-string/non-path-like
    input, an empty or whitespace-only input, a NUL byte, an absolute or
    ``..``-containing canonical ID, or a pipeline-resolved path that does
    not lexically resolve to the same physical location as
    ``image_root / canonical_relative_id``.
    """
    canonical_id = _require_path_text(canonical_relative_id, "canonical_relative_id")
    root_text = _require_path_text(image_root, "image_root")
    pipeline_text = _require_path_text(pipeline_resolved_filename, "pipeline_resolved_filename")

    canonical_posix = canonical_id.replace("\\", "/")
    if canonical_posix.startswith("/"):
        raise K11K12StabilityGateError(
            f"canonical_relative_id must be dataset-relative, observed an absolute path {canonical_id!r}"
        )
    if any(part == ".." for part in canonical_posix.split("/")):
        raise K11K12StabilityGateError(
            f"canonical_relative_id must not contain '..' traversal components, observed {canonical_id!r}"
        )
    normalized_canonical = posixpath.normpath(canonical_posix)
    if normalized_canonical in (".", ""):
        raise K11K12StabilityGateError(
            f"canonical_relative_id must not be empty after normalization, observed {canonical_id!r}"
        )

    resolved_root = _resolve_lexical(root_text)
    expected_physical = _resolve_lexical(str(Path(root_text) / canonical_posix))

    # Belt-and-suspenders on top of the raw '..'/absolute-path rejection
    # above (which already makes this unreachable for any conforming
    # canonical_relative_id): the joined-and-resolved expected path must
    # still land under image_root.
    if not (expected_physical == resolved_root or resolved_root in expected_physical.parents):
        raise K11K12StabilityGateError(
            f"canonical_relative_id {canonical_id!r} resolves outside image_root {image_root!r} "
            f"(resolved to {expected_physical})"
        )

    observed_physical = _resolve_lexical(pipeline_text)
    if observed_physical != expected_physical:
        raise K11K12StabilityGateError(
            f"pipeline-resolved filename {pipeline_resolved_filename!r} (resolved {observed_physical}) "
            f"disagrees with the expected physical path {expected_physical} derived from "
            f"image_root={image_root!r} + canonical_relative_id={canonical_id!r}"
        )

    return canonical_id


__all__ = ["reconcile_canonical_image_id"]
