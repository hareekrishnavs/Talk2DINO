"""Deterministic palette, panel rendering, and atomic qualitative-bundle
installation for DFP Phase 1A Step 11.

Every color assignment here is a PURE function of a stable identifier
(class index, category name, or component id) and a fixed namespace/
formula bound in ``evaluation_identities/e13_dfp_phase1a_qualitative_report.toml``
-- never Python ``hash()``, never a runtime RNG, never adapted after
viewing results. Rendering never touches ground truth for anything other
than an explicitly-labeled GT panel; nothing here constructs a
prediction or reaches model/graph/recurrence code (see the purity tests
in ``tests/test_dfp_phase1a_qualitative_report.py``).
"""

from __future__ import annotations

import colorsys
import hashlib
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from src.dfp_phase1a_artifact_schema import (
    DfpPhase1aArtifactSchemaError,
    parse_strict_json_document,
    require_safe_basename,
    sha256_file,
    write_json_atomically,
)
from src.dfp_phase1a_qualitative_selection import (
    CATEGORY_ORDER,
    DfpPhase1aQualitativeReportError,
)

GOLDEN_ANGLE_DEGREES = 137.507764
PALETTE_SATURATION = 0.65
PALETTE_VALUE = 0.95

BUNDLE_SCHEMA_NAME = "talk2dino-dfp-phase1a-qualitative-bundle-v1"
BUNDLE_SCHEMA_VERSION = 1

TOP_BUNDLE_MANIFEST_KEYS = frozenset(
    {
        "schema", "schema_version", "reporting_identity_sha256", "selection_manifest_sha256", "source_result_sha256",
        "run_mode", "panel_order", "palette_digest", "panels", "created_at_utc",
    }
)
TOP_PANEL_RECORD_KEYS = frozenset(
    {"image_id", "category_name", "panel_name", "png_filename", "png_sha256", "decoded_pixel_sha256", "width", "height"}
)


# ---------------------------------------------------------------------------
# Deterministic palette.
# ---------------------------------------------------------------------------


def generate_class_palette(class_count: int) -> np.ndarray:
    """For class index c: HSV(hue=(c * golden_angle_degrees) mod 360,
    saturation=0.65, value=0.95) converted to sRGB uint8. Pure function
    of `class_count` alone -- deterministic, no RNG, no Python hash().
    Class 0 is always pure-formula-derived like every other class (no
    special-cased "background" color; ignore/background colors are
    separate, explicitly bound constants applied only at render time)."""
    if type(class_count) is not int or isinstance(class_count, bool) or class_count < 1:
        raise DfpPhase1aQualitativeReportError("class_count must be an exact positive integer")
    palette = np.zeros((class_count, 3), dtype=np.uint8)
    for c in range(class_count):
        hue = ((c * GOLDEN_ANGLE_DEGREES) % 360.0) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, PALETTE_SATURATION, PALETTE_VALUE)
        palette[c] = (round(r * 255), round(g * 255), round(b * 255))
    return palette


def palette_digest(palette: np.ndarray) -> str:
    if not isinstance(palette, np.ndarray) or palette.dtype != np.uint8 or palette.ndim != 2 or palette.shape[1] != 3:
        raise DfpPhase1aQualitativeReportError("palette must be a numpy.ndarray of shape [class_count, 3] and dtype uint8")
    return hashlib.sha256(palette.tobytes()).hexdigest()


def component_color(component_id: int, *, namespace: str) -> tuple[int, int, int]:
    """Deterministic component-diagnostic color: a pure function of
    `component_id` and `namespace` alone via SHA-256 -- independent of
    chunk size, dictionary iteration order, or PYTHONHASHSEED. Never
    Python hash(), never RNG. Diagnostic only -- never confused with a
    semantic class color (a distinct color space/derivation from
    `generate_class_palette`)."""
    if type(component_id) is not int or isinstance(component_id, bool):
        raise DfpPhase1aQualitativeReportError("component_id must be an exact non-boolean integer")
    payload = f"{namespace}\0component\0{component_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return (digest[0], digest[1], digest[2])


# ---------------------------------------------------------------------------
# Panel rendering -- pure array-in, array-out. No file I/O here.
# ---------------------------------------------------------------------------


def render_rgb_panel(rgb: np.ndarray, *, output_height: int, output_width: int) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise DfpPhase1aQualitativeReportError("rgb must be a numpy.ndarray of shape [H, W, 3] and dtype uint8")
    image = Image.fromarray(rgb, mode="RGB")
    resized = image.resize((output_width, output_height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def render_label_panel(labels: np.ndarray, *, palette: np.ndarray, ignore_index: int, ignore_color: tuple[int, int, int], output_height: int, output_width: int) -> np.ndarray:
    """Nearest-neighbor resize ONLY -- no antialiasing may ever be
    applied to a categorical label map (would invent fractional/blended
    class IDs). Ignore pixels render as the fixed ignore color, applied
    AFTER resizing (so a resized ignore boundary can never bleed a real
    class color from interpolation, since none occurred)."""
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise DfpPhase1aQualitativeReportError("labels must be a numpy.ndarray of shape [H, W]")
    class_count = palette.shape[0]
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = labels != ignore_index
    in_range = valid & (labels >= 0) & (labels < class_count)
    if valid.any() and not in_range[valid].all():
        raise DfpPhase1aQualitativeReportError("labels contains a non-ignore value outside [0, class_count)")
    rgb[in_range] = palette[labels[in_range]]
    rgb[~valid] = ignore_color
    image = Image.fromarray(rgb, mode="RGB")
    resized = image.resize((output_width, output_height), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def render_category_overlay_panel(
    base_rgb: np.ndarray, category_masks: Mapping[str, np.ndarray], *, overlay_colors: Mapping[str, tuple[int, int, int]],
    opacity: float, output_height: int, output_width: int,
) -> np.ndarray:
    """Blend each category's boolean mask onto `base_rgb` at the bound
    `opacity`, in `CATEGORY_ORDER` (later categories painted on top of
    earlier ones at overlapping pixels -- a fixed, documented, non-data-
    dependent tie rule). Resizes with nearest-neighbor (the overlay is
    categorical, not continuous)."""
    if not isinstance(base_rgb, np.ndarray) or base_rgb.dtype != np.uint8 or base_rgb.ndim != 3:
        raise DfpPhase1aQualitativeReportError("base_rgb must be a numpy.ndarray of shape [H, W, 3] and dtype uint8")
    result = base_rgb.astype(np.float64).copy()
    for name in CATEGORY_ORDER:
        mask = category_masks.get(name)
        if mask is None:
            continue
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != base_rgb.shape[:2]:
            raise DfpPhase1aQualitativeReportError(f"category_masks[{name!r}] must be a boolean numpy.ndarray matching base_rgb's [H, W]")
        color = np.array(overlay_colors[name], dtype=np.float64)
        result[mask] = (1.0 - opacity) * result[mask] + opacity * color
    result_u8 = np.clip(np.round(result), 0, 255).astype(np.uint8)
    image = Image.fromarray(result_u8, mode="RGB")
    resized = image.resize((output_width, output_height), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def render_component_overlay_panel(component_ids: np.ndarray, *, namespace: str, background_color: tuple[int, int, int], output_height: int, output_width: int) -> np.ndarray:
    """Component-diagnostic panel: each distinct component id gets
    `component_color(id, namespace=namespace)` -- background (id < 0)
    renders as `background_color`. Deterministic regardless of how many
    distinct ids are present or in what order they were discovered."""
    if not isinstance(component_ids, np.ndarray) or component_ids.ndim != 2:
        raise DfpPhase1aQualitativeReportError("component_ids must be a numpy.ndarray of shape [H, W]")
    rgb = np.zeros((*component_ids.shape, 3), dtype=np.uint8)
    rgb[:, :] = background_color
    for cid in sorted(int(v) for v in np.unique(component_ids) if v >= 0):
        rgb[component_ids == cid] = component_color(cid, namespace=namespace)
    image = Image.fromarray(rgb, mode="RGB")
    resized = image.resize((output_width, output_height), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Deterministic PNG I/O: encoded + decoded digests.
# ---------------------------------------------------------------------------


def decoded_pixel_digest(pixel_array: np.ndarray) -> str:
    if not isinstance(pixel_array, np.ndarray) or pixel_array.dtype != np.uint8:
        raise DfpPhase1aQualitativeReportError("pixel_array must be a numpy.ndarray of dtype uint8")
    return hashlib.sha256(pixel_array.tobytes()).hexdigest()


def _decode_rgb_image(path: Path, *, label: str) -> np.ndarray:
    """The one shared, narrowly-scoped strict image-decoding helper every
    public path in this module uses to read back a PNG from disk. Opens
    the file inside a context manager, forces full decoding
    (`image.load()`) while still inside that protected scope, converts
    to the exact required RGB mode, and detaches an independent decoded
    copy (`.copy()`) before the context manager closes the underlying
    file handle. Catches only the PIL decode failures this represents --
    `OSError` (which covers `PIL.UnidentifiedImageError`, a documented
    `OSError` subclass, plus truncated/unreadable files) and `ValueError`
    (malformed/unsupported image data PIL rejects during decode/convert)
    -- and re-raises as :class:`DfpPhase1aQualitativeReportError` with
    the original exception preserved as `__cause__`. Never catches
    `KeyboardInterrupt`/`SystemExit`/`MemoryError`. Alters nothing about
    the decoded pixel values themselves -- this is purely a strict I/O
    boundary, not a content transform."""
    try:
        with Image.open(path) as image:
            image.load()
            decoded = image.convert("RGB").copy()
    except (OSError, ValueError) as error:
        raise DfpPhase1aQualitativeReportError(f"cannot decode {label} {path}: {error}") from error
    return np.asarray(decoded, dtype=np.uint8)


def write_panel_png(pixel_array: np.ndarray, path: Path, *, compress_level: int = 6, optimize: bool = False) -> tuple[str, str]:
    """Write `pixel_array` as a deterministic PNG (no timestamps, no
    nondeterministic metadata chunks, explicit compression settings,
    explicit mode) to `path` via write-temp-then-`os.replace`. Returns
    `(encoded_png_sha256, decoded_pixel_sha256)`. Never mutates
    `pixel_array`. Re-decodes the just-written file to independently
    confirm the decoded digest matches the pre-encode array -- proving
    the encoder round-trips losslessly."""
    if not isinstance(pixel_array, np.ndarray) or pixel_array.dtype != np.uint8 or pixel_array.ndim != 3 or pixel_array.shape[2] != 3:
        raise DfpPhase1aQualitativeReportError("pixel_array must be a numpy.ndarray of shape [H, W, 3] and dtype uint8")
    path = Path(path)
    temp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    image = Image.fromarray(pixel_array, mode="RGB")
    try:
        # PIL's PNG encoder does not embed a timestamp or locale-dependent
        # metadata by default; pnginfo is explicitly omitted (None) so no
        # tEXt/tIME chunk is ever written.
        image.save(temp_path, format="PNG", optimize=optimize, compress_level=compress_level, pnginfo=None)
        os.replace(temp_path, path)
    except OSError as error:
        raise DfpPhase1aQualitativeReportError(f"cannot write panel PNG to {path}: {error}") from error
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    encoded_sha256 = sha256_file(path)
    reread = _decode_rgb_image(path, label="just-written panel PNG")
    decoded_sha256 = decoded_pixel_digest(reread)
    expected_decoded_sha256 = decoded_pixel_digest(pixel_array)
    if decoded_sha256 != expected_decoded_sha256:
        raise DfpPhase1aQualitativeReportError(
            f"PNG round-trip mismatch writing {path}: decoded content does not match the source array"
        )
    return encoded_sha256, decoded_sha256


# ---------------------------------------------------------------------------
# Strict bundle manifest.
# ---------------------------------------------------------------------------


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        raise DfpPhase1aQualitativeReportError(f"{label} must be an exact non-empty string")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DfpPhase1aQualitativeReportError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise DfpPhase1aQualitativeReportError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise DfpPhase1aQualitativeReportError(f"{label} must be a lowercase 64-hex-character SHA-256")
    return token


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        missing = expected_keys - set(value if isinstance(value, Mapping) else {})
        unknown = set(value if isinstance(value, Mapping) else {}) - expected_keys
        raise DfpPhase1aQualitativeReportError(f"{label} has an unexpected schema (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


@dataclass(frozen=True)
class PanelRecord:
    image_id: str
    category_name: str
    panel_name: str
    png_filename: str
    png_sha256: str
    decoded_pixel_sha256: str
    width: int
    height: int


def build_bundle_manifest(
    *, reporting_identity_sha256: str, selection_manifest_sha256: str, source_result_sha256: str, run_mode: str,
    panel_order: Sequence[str], palette_digest_value: str, panels: Sequence[PanelRecord], created_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA_NAME, "schema_version": BUNDLE_SCHEMA_VERSION,
        "reporting_identity_sha256": reporting_identity_sha256, "selection_manifest_sha256": selection_manifest_sha256,
        "source_result_sha256": source_result_sha256, "run_mode": run_mode, "panel_order": list(panel_order),
        "palette_digest": palette_digest_value,
        "panels": [
            {
                "image_id": p.image_id, "category_name": p.category_name, "panel_name": p.panel_name,
                "png_filename": p.png_filename, "png_sha256": p.png_sha256, "decoded_pixel_sha256": p.decoded_pixel_sha256,
                "width": p.width, "height": p.height,
            }
            for p in panels
        ],
        "created_at_utc": created_at_utc,
    }


def validate_bundle_manifest_structure(manifest: Mapping[str, Any]) -> None:
    _require_closed_mapping(manifest, TOP_BUNDLE_MANIFEST_KEYS, "bundle manifest")
    if _require_exact_string(manifest["schema"], "bundle_manifest.schema") != BUNDLE_SCHEMA_NAME:
        raise DfpPhase1aQualitativeReportError(f"bundle_manifest.schema must be exactly {BUNDLE_SCHEMA_NAME!r}")
    if _require_exact_int(manifest["schema_version"], "bundle_manifest.schema_version", minimum=1) != BUNDLE_SCHEMA_VERSION:
        raise DfpPhase1aQualitativeReportError(f"bundle_manifest.schema_version must be exactly {BUNDLE_SCHEMA_VERSION}")
    for field in ("reporting_identity_sha256", "selection_manifest_sha256", "source_result_sha256", "palette_digest"):
        _require_sha256(manifest[field], f"bundle_manifest.{field}")
    _require_exact_string(manifest["run_mode"], "bundle_manifest.run_mode")
    panel_order = manifest["panel_order"]
    if type(panel_order) is not list or not panel_order or any(type(p) is not str for p in panel_order):
        raise DfpPhase1aQualitativeReportError("bundle_manifest.panel_order must be a non-empty exact list of strings")
    panels = manifest["panels"]
    if type(panels) is not list:
        raise DfpPhase1aQualitativeReportError("bundle_manifest.panels must be an exact list")
    seen_filenames: set[str] = set()
    for entry in panels:
        _require_closed_mapping(entry, TOP_PANEL_RECORD_KEYS, "bundle_manifest.panels[]")
        _require_exact_string(entry["image_id"], "panels[].image_id")
        _require_exact_string(entry["category_name"], "panels[].category_name")
        panel_name = _require_exact_string(entry["panel_name"], "panels[].panel_name")
        if panel_name not in panel_order:
            raise DfpPhase1aQualitativeReportError(f"panels[].panel_name {panel_name!r} is not in the declared panel_order")
        try:
            filename = require_safe_basename(entry["png_filename"], "panels[].png_filename")
        except DfpPhase1aArtifactSchemaError as error:
            raise DfpPhase1aQualitativeReportError(str(error)) from error
        if filename in seen_filenames:
            raise DfpPhase1aQualitativeReportError(f"duplicate panel PNG filename {filename!r}")
        seen_filenames.add(filename)
        _require_sha256(entry["png_sha256"], "panels[].png_sha256")
        _require_sha256(entry["decoded_pixel_sha256"], "panels[].decoded_pixel_sha256")
        _require_exact_int(entry["width"], "panels[].width", minimum=1)
        _require_exact_int(entry["height"], "panels[].height", minimum=1)
    _require_exact_string(manifest["created_at_utc"], "bundle_manifest.created_at_utc")


# ---------------------------------------------------------------------------
# Atomic bundle installation: build in a fresh sibling staging directory,
# fully self-verify, then atomically rename into place. Rejects cleanly if
# the destination already exists (no independently-hardened overwrite
# policy exists for this bundle kind in this step).
# ---------------------------------------------------------------------------


def install_qualitative_bundle(
    *, destination_dir: Path, selection_manifest: Mapping[str, Any], bundle_manifest: Mapping[str, Any],
    panel_png_bytes: Mapping[str, bytes],
) -> Path:
    """Build the complete qualitative bundle (selection manifest, bundle
    manifest, every panel PNG) in a fresh sibling staging directory,
    fully self-verify it (decode + re-hash every panel, cross-check
    every hash the bundle manifest declares, confirm the exact expected
    file set with no missing/extra files), then atomically rename the
    staging directory into `destination_dir`. Rejects cleanly if
    `destination_dir` already exists -- no partial multi-file
    replacement. On any failure, `destination_dir` is absent and no
    staging directory remains."""
    destination_dir = Path(destination_dir)
    if destination_dir.exists():
        raise DfpPhase1aQualitativeReportError(f"qualitative bundle destination {destination_dir} already exists; refusing to overwrite an existing bundle")

    validate_bundle_manifest_structure(bundle_manifest)
    from src.dfp_phase1a_qualitative_selection import validate_selection_manifest_structure

    validate_selection_manifest_structure(selection_manifest)

    expected_filenames = {"selection_manifest.json", "bundle_manifest.json"} | {p["png_filename"] for p in bundle_manifest["panels"]}
    if set(panel_png_bytes) | {"selection_manifest.json", "bundle_manifest.json"} != expected_filenames:
        missing = expected_filenames - (set(panel_png_bytes) | {"selection_manifest.json", "bundle_manifest.json"})
        unexpected = (set(panel_png_bytes) | {"selection_manifest.json", "bundle_manifest.json"}) - expected_filenames
        raise DfpPhase1aQualitativeReportError(f"panel PNG file set disagrees with the bundle manifest (missing={sorted(missing)}, unexpected={sorted(unexpected)})")

    staging_dir = destination_dir.parent / f"{destination_dir.name}.staging-{secrets.token_hex(8)}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    try:
        staging_dir.mkdir(parents=True)
        write_json_atomically(staging_dir / "selection_manifest.json", selection_manifest, label="selection manifest")
        write_json_atomically(staging_dir / "bundle_manifest.json", bundle_manifest, label="bundle manifest")
        for filename, png_bytes in panel_png_bytes.items():
            (staging_dir / filename).write_bytes(png_bytes)

        # Full self-verification of the staged bundle before install.
        staged_selection = parse_strict_json_document(staging_dir / "selection_manifest.json", label="staged selection manifest")
        validate_selection_manifest_structure(staged_selection)
        staged_bundle = parse_strict_json_document(staging_dir / "bundle_manifest.json", label="staged bundle manifest")
        validate_bundle_manifest_structure(staged_bundle)
        if staged_bundle["selection_manifest_sha256"] != sha256_file(staging_dir / "selection_manifest.json"):
            raise DfpPhase1aQualitativeReportError("staged bundle_manifest.selection_manifest_sha256 does not match the staged selection manifest's own hash")

        staged_files = {p.name for p in staging_dir.iterdir()}
        if staged_files != expected_filenames:
            raise DfpPhase1aQualitativeReportError("staged bundle file set does not match the expected file set")

        for entry in staged_bundle["panels"]:
            panel_path = staging_dir / entry["png_filename"]
            if sha256_file(panel_path) != entry["png_sha256"]:
                raise DfpPhase1aQualitativeReportError(f"staged panel {entry['png_filename']!r} PNG bytes do not match its declared hash")
            decoded = _decode_rgb_image(panel_path, label=f"staged panel {entry['png_filename']!r}")
            if decoded_pixel_digest(decoded) != entry["decoded_pixel_sha256"]:
                raise DfpPhase1aQualitativeReportError(f"staged panel {entry['png_filename']!r} decoded pixel content does not match its declared digest")
            if decoded.shape[0] != entry["height"] or decoded.shape[1] != entry["width"]:
                raise DfpPhase1aQualitativeReportError(f"staged panel {entry['png_filename']!r} dimensions disagree with its manifest entry")
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    try:
        os.rename(staging_dir, destination_dir)
    except OSError as error:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise DfpPhase1aQualitativeReportError(f"cannot install qualitative bundle at {destination_dir}: {error}") from error

    return destination_dir


def verify_qualitative_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Read-only re-verification of an installed qualitative bundle:
    exact expected file set, every panel decodes and re-hashes
    correctly, selection manifest structurally valid, bundle manifest
    cross-references the selection manifest's own hash. Never mutates
    anything."""
    bundle_dir = Path(bundle_dir)
    from src.dfp_phase1a_qualitative_selection import validate_selection_manifest_structure

    try:
        selection_manifest = parse_strict_json_document(bundle_dir / "selection_manifest.json", label="selection manifest")
        bundle_manifest = parse_strict_json_document(bundle_dir / "bundle_manifest.json", label="bundle manifest")
    except DfpPhase1aArtifactSchemaError as error:
        raise DfpPhase1aQualitativeReportError(str(error)) from error

    validate_selection_manifest_structure(selection_manifest)
    validate_bundle_manifest_structure(bundle_manifest)

    if bundle_manifest["selection_manifest_sha256"] != sha256_file(bundle_dir / "selection_manifest.json"):
        raise DfpPhase1aQualitativeReportError("bundle_manifest.selection_manifest_sha256 disagrees with the installed selection manifest")

    expected_filenames = {"selection_manifest.json", "bundle_manifest.json"} | {p["png_filename"] for p in bundle_manifest["panels"]}
    actual_filenames = {p.name for p in bundle_dir.iterdir()}
    if actual_filenames != expected_filenames:
        missing = expected_filenames - actual_filenames
        unexpected = actual_filenames - expected_filenames
        raise DfpPhase1aQualitativeReportError(f"installed bundle file set disagrees with the manifest (missing={sorted(missing)}, unexpected={sorted(unexpected)})")

    for entry in bundle_manifest["panels"]:
        panel_path = bundle_dir / entry["png_filename"]
        if sha256_file(panel_path) != entry["png_sha256"]:
            raise DfpPhase1aQualitativeReportError(f"installed panel {entry['png_filename']!r} PNG bytes do not match its declared hash")
        decoded = _decode_rgb_image(panel_path, label=f"installed panel {entry['png_filename']!r}")
        if decoded_pixel_digest(decoded) != entry["decoded_pixel_sha256"]:
            raise DfpPhase1aQualitativeReportError(f"installed panel {entry['png_filename']!r} decoded pixel content does not match its declared digest")

    return {"selection_manifest": selection_manifest, "bundle_manifest": bundle_manifest}


__all__ = [
    "GOLDEN_ANGLE_DEGREES",
    "BUNDLE_SCHEMA_NAME",
    "BUNDLE_SCHEMA_VERSION",
    "TOP_BUNDLE_MANIFEST_KEYS",
    "TOP_PANEL_RECORD_KEYS",
    "generate_class_palette",
    "palette_digest",
    "component_color",
    "render_rgb_panel",
    "render_label_panel",
    "render_category_overlay_panel",
    "render_component_overlay_panel",
    "decoded_pixel_digest",
    "write_panel_png",
    "PanelRecord",
    "build_bundle_manifest",
    "validate_bundle_manifest_structure",
    "install_qualitative_bundle",
    "verify_qualitative_bundle",
]
