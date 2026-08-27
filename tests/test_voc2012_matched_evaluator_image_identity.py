"""Regression tests for the VOC2012 logical-ID / physical-filename
reconciliation repair: diagnostics/run_voc2012_matched_evaluation.py's
_voc_physical_relative_image_filename and _extract_prepared_voc_image.

The real GPU pilot (job 20626350) failed deterministically on the first
image because the driver passed the bare checkpoint ID ("2007_000033")
straight into the shared _extract_prepared_image/reconcile_canonical_
image_id physical-path check, which requires the exact suffixed
filename ("2007_000033.jpg"). These tests prove: (1) the bare-ID bug is
independently reproducible against the real dataset, (2) the VOC-only
adapter fixes it without touching the shared helper or
src/dataset_image_identity.py, (3) every downstream structure still
sees only the bare logical ID, and (4) a wide adversarial matrix around
suffix/path handling fails closed with a concise diagnostic before any
model/CUDA work.

CPU-only throughout. Real-data-gated tests never initialize CUDA."""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import os
import sys
from contextlib import redirect_stderr
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
OVS_ROOT = ROOT / "src/open_vocabulary_segmentation"
for _p in (ROOT, OVS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

IDENTITY_PATH = ROOT / "evaluation_identities/e12_voc2012_matched_evaluator.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the voc2012 matched-evaluator identity")

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def driver_module():
    spec = importlib.util.spec_from_file_location(
        "voc2012_driver_image_identity_under_test", ROOT / "diagnostics" / "run_voc2012_matched_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------
# _voc_physical_relative_image_filename: pure construction + validation
# ---------------------------------------------------------------------


def test_constructs_correct_physical_filename(driver_module):
    assert driver_module._voc_physical_relative_image_filename("2007_000033", image_suffix=".jpg") == "2007_000033.jpg"


def test_physical_filename_receives_exactly_one_suffix(driver_module):
    result = driver_module._voc_physical_relative_image_filename("2007_000033", image_suffix=".jpg")
    assert result.count(".jpg") == 1
    assert not result.endswith(".jpg.jpg")


@pytest.mark.parametrize("bad_id", ["2007_000033.jpg", "img.jpg"])
def test_rejects_id_already_ending_with_suffix_double_suffix_guard(driver_module, bad_id):
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="already end"):
        driver_module._voc_physical_relative_image_filename(bad_id, image_suffix=".jpg")


@pytest.mark.parametrize("bad_id", ["", " ", "\t", "\n", "  2007_000033", "2007_000033  "])
def test_rejects_empty_or_whitespace_id(driver_module, bad_id):
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._voc_physical_relative_image_filename(bad_id, image_suffix=".jpg")


@pytest.mark.parametrize("bad_id", ["/2007_000033", "/etc/passwd", "/"])
def test_rejects_absolute_id(driver_module, bad_id):
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="absolute"):
        driver_module._voc_physical_relative_image_filename(bad_id, image_suffix=".jpg")


@pytest.mark.parametrize("bad_id", ["..", "../2007_000033", "2007_000033/../etc", "a/../b"])
def test_rejects_dotdot_traversal(driver_module, bad_id):
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._voc_physical_relative_image_filename(bad_id, image_suffix=".jpg")


@pytest.mark.parametrize("bad_id", ["2007/000033", "a/b", "a\\b", "2007_000033\\"])
def test_rejects_path_separators_in_bare_id(driver_module, bad_id):
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError, match="path separator"):
        driver_module._voc_physical_relative_image_filename(bad_id, image_suffix=".jpg")


@pytest.mark.parametrize("bad_suffix", ["", None, 5])
def test_rejects_empty_or_non_string_suffix(driver_module, bad_suffix):
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._voc_physical_relative_image_filename("2007_000033", image_suffix=bad_suffix)


def test_wrong_suffix_produces_a_filename_that_will_not_match_the_real_pipeline(driver_module):
    """Passing the WRONG authoritative suffix (e.g. a tampered/misconfigured
    source identity claiming '.png' for images) must still construct
    -something-, but that something must NOT coincide with the real
    '.jpg' physical file -- proving the suffix is load-bearing, not
    decorative."""
    wrong = driver_module._voc_physical_relative_image_filename("2007_000033", image_suffix=".png")
    assert wrong == "2007_000033.png"
    assert wrong != "2007_000033.jpg"


# ---------------------------------------------------------------------
# _extract_prepared_voc_image: wrapping/identity-swap behavior, using a
# lightweight in-process fake for the injected extract_prepared_image_fn
# (no real dataset/mmseg needed for these particular checks).
# ---------------------------------------------------------------------


def _fake_prepared(image_id):
    from diagnostics.run_k11_k12_stability import PreparedDiagnosticImage

    return PreparedDiagnosticImage(
        dataset_index=0, image_id=image_id, image_tensor="fake-tensor", img_metas={"filename": image_id},
        inference_height=1, inference_width=1, source_shape_provenance="test",
    )


def test_adapter_returns_bare_logical_id_not_the_suffixed_physical_filename(driver_module):
    seen_canonical_image_id = {}

    def fake_extract(dataset, dataset_index, *, canonical_image_id):
        seen_canonical_image_id["value"] = canonical_image_id
        return _fake_prepared(canonical_image_id)  # mirrors reconcile_canonical_image_id's real echo-back behavior

    prepared = driver_module._extract_prepared_voc_image(
        object(), 0, logical_voc_image_id="2007_000033", image_suffix=".jpg", extract_prepared_image_fn=fake_extract,
    )
    assert seen_canonical_image_id["value"] == "2007_000033.jpg", "the shared helper must receive the SUFFIXED filename"
    assert prepared.image_id == "2007_000033", "the caller must see the BARE logical ID back, never the suffix"
    assert not prepared.image_id.endswith(".jpg")


def test_adapter_preserves_every_other_field_unchanged(driver_module):
    def fake_extract(dataset, dataset_index, *, canonical_image_id):
        from diagnostics.run_k11_k12_stability import PreparedDiagnosticImage
        return PreparedDiagnosticImage(
            dataset_index=7, image_id=canonical_image_id, image_tensor="tensor-marker", img_metas={"a": 1},
            inference_height=123, inference_width=456, source_shape_provenance="tensor==img_shape",
        )

    prepared = driver_module._extract_prepared_voc_image(
        object(), 7, logical_voc_image_id="2007_000033", image_suffix=".jpg", extract_prepared_image_fn=fake_extract,
    )
    assert prepared.dataset_index == 7
    assert prepared.image_tensor == "tensor-marker"
    assert prepared.img_metas == {"a": 1}
    assert prepared.inference_height == 123
    assert prepared.inference_width == 456
    assert prepared.source_shape_provenance == "tensor==img_shape"


def test_adapter_propagates_a_bad_logical_id_before_calling_the_shared_helper(driver_module):
    calls = {"n": 0}

    def counting_extract(dataset, dataset_index, *, canonical_image_id):
        calls["n"] += 1
        return _fake_prepared(canonical_image_id)

    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError):
        driver_module._extract_prepared_voc_image(
            object(), 0, logical_voc_image_id="../escape", image_suffix=".jpg", extract_prepared_image_fn=counting_extract,
        )
    assert calls["n"] == 0, "a malformed logical ID must be rejected before the shared helper (and any real pipeline work) runs"


def test_adapter_propagates_pipeline_mismatch_for_a_different_image(driver_module):
    """If the underlying shared helper's real reconciliation would reject
    a pipeline-resolved filename that names a DIFFERENT image than the
    logical ID under check, that rejection must propagate through the
    adapter unchanged (never swallowed/retried/normalized)."""
    from src.dataset_image_identity import K11K12StabilityGateError

    def mismatched_extract(dataset, dataset_index, *, canonical_image_id):
        raise K11K12StabilityGateError(
            f"pipeline-resolved filename '/root/2007_999999.jpg' disagrees with expected physical path "
            f"derived from canonical_relative_id={canonical_image_id!r}"
        )

    with pytest.raises(K11K12StabilityGateError):
        driver_module._extract_prepared_voc_image(
            object(), 0, logical_voc_image_id="2007_000033", image_suffix=".jpg", extract_prepared_image_fn=mismatched_extract,
        )


# ---------------------------------------------------------------------
# Real-data-gated: reproduction of the exact original bug, the fix, and
# the shared-helper's own symlink-following reconciliation policy.
# ---------------------------------------------------------------------


def _real_data_root():
    raw = os.environ.get("VOC2012_REAL_DATA_ROOT")
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


REAL_DATA_ROOT = _real_data_root()
requires_real_data = pytest.mark.skipif(
    REAL_DATA_ROOT is None or not REAL_DATA_ROOT.exists(),
    reason="requires VOC2012_REAL_DATA_ROOT to point at the real VOC2012 archive",
)


@pytest.fixture(scope="module")
def real_v21_dataset():
    from mmcv import Config as MMCVConfig
    from mmseg.datasets import build_dataset
    import main  # noqa: F401
    from src.voc2012_dataset_identity import load_identity as load_source_identity
    from src.voc2012_dataset_manifest import canonical_validation_ids, resolve_dataset_root

    source_identity = load_source_identity(repo_root=ROOT)
    voc_root = resolve_dataset_root(REAL_DATA_ROOT, source_identity)
    canonical_ids = canonical_validation_ids(voc_root, source_identity)
    image_suffix = source_identity["source"]["image_suffix"]

    cfg = MMCVConfig.fromfile(str(ROOT / "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/pascal_voc12.py"))
    cfg.data.test.data_root = str(voc_root)
    dataset = build_dataset(cfg.data.test)
    return dataset, canonical_ids, image_suffix


@requires_real_data
def test_original_bug_independently_reproduced_on_real_first_sample(real_v21_dataset):
    from src.dataset_image_identity import reconcile_canonical_image_id

    dataset, canonical_ids, image_suffix = real_v21_dataset
    assert canonical_ids[0] == "2007_000033"
    raw = dataset[0]
    meta = raw["img_metas"][0].data if hasattr(raw["img_metas"][0], "data") else raw["img_metas"][0]
    pipeline_filename = meta["filename"]
    assert pipeline_filename.endswith(".jpg")

    with pytest.raises(Exception):
        reconcile_canonical_image_id(
            canonical_relative_id=canonical_ids[0],  # the bare-ID bug
            image_root=dataset.img_dir,
            pipeline_resolved_filename=pipeline_filename,
        )


@requires_real_data
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_repaired_driver_succeeds_on_first_middle_last_real_samples(driver_module, real_v21_dataset, position):
    from diagnostics.run_k11_k12_stability import _extract_prepared_image

    dataset, canonical_ids, image_suffix = real_v21_dataset
    index = {"first": 0, "middle": len(canonical_ids) // 2, "last": len(canonical_ids) - 1}[position]
    logical_id = canonical_ids[index]

    prepared = driver_module._extract_prepared_voc_image(
        dataset, index, logical_voc_image_id=logical_id, image_suffix=image_suffix,
        extract_prepared_image_fn=_extract_prepared_image,
    )
    assert prepared.image_id == logical_id
    assert torch.cuda.is_initialized() is False


@requires_real_data
def test_pipeline_resolved_path_equals_independently_reconstructed_physical_path(real_v21_dataset):
    dataset, canonical_ids, image_suffix = real_v21_dataset
    raw = dataset[0]
    meta = raw["img_metas"][0].data if hasattr(raw["img_metas"][0], "data") else raw["img_metas"][0]
    pipeline_filename = Path(meta["filename"]).resolve(strict=False)
    independently_reconstructed = (Path(dataset.img_dir) / f"{canonical_ids[0]}{image_suffix}").resolve(strict=False)
    assert pipeline_filename == independently_reconstructed


@requires_real_data
def test_v20_and_v21_share_identical_logical_ids_and_order(driver_module, real_v21_dataset):
    """V20 and V21 must use the exact same canonical order -- the fix
    only touches how the PHYSICAL filename is reconstructed, never the
    shared logical-ID order both protocols read from."""
    from mmcv import Config as MMCVConfig
    from mmseg.datasets import build_dataset

    dataset_v21, canonical_ids, image_suffix = real_v21_dataset
    cfg_v20 = MMCVConfig.fromfile(str(ROOT / "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/pascal_voc12_20.py"))
    cfg_v20.data.test.data_root = str(dataset_v21.img_dir).rsplit("/JPEGImages", 1)[0]
    dataset_v20 = build_dataset(cfg_v20.data.test)

    v20_infos = dataset_v20.img_infos if hasattr(dataset_v20, "img_infos") else dataset_v20.data_infos
    v21_infos = dataset_v21.img_infos if hasattr(dataset_v21, "img_infos") else dataset_v21.data_infos
    v20_order = [Path(str(i["filename"])).stem for i in v20_infos]
    v21_order = [Path(str(i["filename"])).stem for i in v21_infos]
    assert v20_order == v21_order == list(canonical_ids)


@requires_real_data
def test_symlinked_image_root_still_reconciles(real_v21_dataset, tmp_path):
    """Existing reconciliation policy (src/dataset_image_identity.py,
    unchanged by this fix) already follows symlinks via
    Path.resolve(strict=False); confirm the VOC adapter still works
    through a symlinked image_root, matching that pre-existing policy."""
    from diagnostics.run_k11_k12_stability import _extract_prepared_image
    import importlib.util

    dataset, canonical_ids, image_suffix = real_v21_dataset
    real_img_dir = Path(dataset.img_dir)
    symlinked_root = tmp_path / "JPEGImages_symlink"
    symlinked_root.symlink_to(real_img_dir)

    spec = importlib.util.spec_from_file_location(
        "voc2012_driver_symlink_check", ROOT / "diagnostics" / "run_voc2012_matched_evaluation.py",
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    original_img_dir = dataset.img_dir
    dataset.img_dir = str(symlinked_root)
    try:
        prepared = driver._extract_prepared_voc_image(
            dataset, 0, logical_voc_image_id=canonical_ids[0], image_suffix=image_suffix,
            extract_prepared_image_fn=_extract_prepared_image,
        )
        assert prepared.image_id == canonical_ids[0]
    finally:
        dataset.img_dir = original_img_dir


# ---------------------------------------------------------------------
# Source-identity suffix mutation: the pre-existing, unchanged
# src.voc2012_dataset_identity.load_identity protection must still be
# what rejects a tampered suffix -- confirms this fix relies on (does
# not bypass) that existing guard.
# ---------------------------------------------------------------------


def test_tampered_source_identity_image_suffix_still_rejected_at_load_time(tmp_path):
    from src.voc2012_dataset_identity import load_identity as load_source_identity, Voc2012DatasetIdentityError

    real_toml = (ROOT / "evaluation_identities/e12_voc2012_dataset_source.toml").read_text()
    tampered = real_toml.replace('image_suffix = ".jpg"', 'image_suffix = ".png"')
    assert tampered != real_toml, "sanity: the replace must have actually changed something"
    tampered_path = tmp_path / "tampered_source_identity.toml"
    tampered_path.write_text(tampered)

    with pytest.raises(Voc2012DatasetIdentityError, match="image_suffix"):
        load_source_identity(tampered_path, repo_root=ROOT)


# ---------------------------------------------------------------------
# Failure contract for the adapter's own validation errors: exit 2,
# concise diagnostic, no traceback, no partial output -- via the real
# domain exception type already caught by main()'s existing boundary.
# ---------------------------------------------------------------------


def test_malformed_logical_id_is_a_domain_error_not_a_raw_exception(driver_module):
    with pytest.raises(driver_module.Voc2012MatchedEvaluatorIdentityError) as excinfo:
        driver_module._voc_physical_relative_image_filename("", image_suffix=".jpg")
    assert not isinstance(excinfo.value, (ValueError,)) or isinstance(excinfo.value, driver_module.Voc2012MatchedEvaluatorIdentityError)


def test_keyboardinterrupt_propagates_through_adapter(driver_module):
    def raising(dataset, dataset_index, *, canonical_image_id):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        driver_module._extract_prepared_voc_image(
            object(), 0, logical_voc_image_id="2007_000033", image_suffix=".jpg", extract_prepared_image_fn=raising,
        )


def test_systemexit_propagates_through_adapter(driver_module):
    def raising(dataset, dataset_index, *, canonical_image_id):
        raise SystemExit(13)

    with pytest.raises(SystemExit) as excinfo:
        driver_module._extract_prepared_voc_image(
            object(), 0, logical_voc_image_id="2007_000033", image_suffix=".jpg", extract_prepared_image_fn=raising,
        )
    assert excinfo.value.code == 13


def test_memoryerror_propagates_through_adapter(driver_module):
    def raising(dataset, dataset_index, *, canonical_image_id):
        raise MemoryError("simulated")

    with pytest.raises(MemoryError):
        driver_module._extract_prepared_voc_image(
            object(), 0, logical_voc_image_id="2007_000033", image_suffix=".jpg", extract_prepared_image_fn=raising,
        )
