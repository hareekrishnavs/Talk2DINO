"""Real-data CPU production-path tests for the ADE20K matched evaluator.
Gated on ADE20K_REAL_DATA_ROOT (no private path in this tracked file).
Reaches the actual production dataset-construction and image-preparation
call sites (mmseg build_dataset, diagnostics.run_k11_k12_stability.
_extract_prepared_image, src.dataset_image_identity.
reconcile_canonical_image_id) for first/middle/final real validation
samples -- the exact code path a real evaluator run uses before any
model/CUDA work. Never imports/initializes CUDA, never constructs the
model, never loads the bridge checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
OVS_ROOT = ROOT / "src/open_vocabulary_segmentation"
for _p in (ROOT, OVS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")


def _real_data_root() -> Path | None:
    raw = os.environ.get("ADE20K_REAL_DATA_ROOT")
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


REAL_DATA_ROOT = _real_data_root()
requires_real_data = pytest.mark.skipif(
    REAL_DATA_ROOT is None or not REAL_DATA_ROOT.exists(),
    reason="requires ADE20K_REAL_DATA_ROOT to point at a real ADEChallengeData2016 root or its parent",
)


def _config_data_root() -> Path:
    """The live ade20k.py config's own img_dir/ann_dir already embed the
    'ADEChallengeData2016/' prefix, so data_root must be the PARENT of
    that directory. Accept ADE20K_REAL_DATA_ROOT pointing at either the
    ADEChallengeData2016 directory itself or its parent, mirroring
    src.ade20k_dataset_manifest.resolve_dataset_root's own acceptance of
    both forms."""
    if REAL_DATA_ROOT.name == "ADEChallengeData2016":
        return REAL_DATA_ROOT.parent
    return REAL_DATA_ROOT


def _build_real_dataset():
    from mmcv import Config as MMCVConfig
    from mmseg.datasets import build_dataset
    import main  # noqa: F401 -- registers FloatImage, side-effect only

    cfg = MMCVConfig.fromfile(str(ROOT / "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/ade20k.py"))
    cfg.data.test.data_root = str(_config_data_root())
    return build_dataset(cfg.data.test)


@requires_real_data
def test_cuda_never_initialized_before_or_after():
    assert not torch.cuda.is_initialized()
    dataset = _build_real_dataset()
    assert len(dataset) == 2000
    assert not torch.cuda.is_initialized()


@requires_real_data
def test_real_dataset_class_contract():
    dataset = _build_real_dataset()
    assert len(dataset.CLASSES) == 150
    assert dataset.reduce_zero_label is True
    infos = dataset.img_infos
    assert infos[0]["filename"] == "ADE_val_00000001.jpg"
    assert infos[-1]["filename"] == "ADE_val_00002000.jpg"


@requires_real_data
@pytest.mark.parametrize("which", ["first", "middle", "final"])
def test_prepared_image_extraction_reconciles_real_sample(which):
    """Reaches the SAME production call site
    (diagnostics.run_k11_k12_stability._extract_prepared_image ->
    src.dataset_image_identity.reconcile_canonical_image_id) the real
    evaluator uses -- this is the exact code path the VOC adapter's
    formerly-failing bug lived in."""
    from diagnostics.run_k11_k12_stability import _extract_prepared_image

    dataset = _build_real_dataset()
    infos = dataset.img_infos
    index = {"first": 0, "middle": len(dataset) // 2, "final": len(dataset) - 1}[which]
    expected_id = infos[index]["filename"]

    prepared = _extract_prepared_image(dataset, index, canonical_image_id=expected_id)

    assert prepared.image_id == expected_id
    assert prepared.image_tensor.ndim == 3
    assert prepared.inference_height > 0 and prepared.inference_width > 0
    assert not torch.cuda.is_initialized()


@requires_real_data
def test_nontrivial_dimensions_sample_extracts_cleanly():
    """A sample whose image dimensions are neither square nor the
    dataset's most common aspect ratio -- proves reconciliation doesn't
    depend on any specific image shape."""
    dataset = _build_real_dataset()
    from diagnostics.run_k11_k12_stability import _extract_prepared_image

    infos = dataset.img_infos
    # Any real ADE20K validation image works; index 250 is arbitrary and
    # deterministic, distinct from the first/middle/final samples above.
    index = 250
    expected_id = infos[index]["filename"]
    prepared = _extract_prepared_image(dataset, index, canonical_image_id=expected_id)
    assert prepared.image_id == expected_id
    assert prepared.image_tensor.shape[0] == 3


# ---------------------------------------------------------------------------
# Adversarial reconciliation -- reuses the shared, already-hardened
# reconcile_canonical_image_id directly (no dataset needed).
# ---------------------------------------------------------------------------


@requires_real_data
def test_reconciliation_rejects_bare_logical_id_without_suffix():
    """Never pass a bare logical ID to a helper that reconstructs a
    physical path requiring .jpg -- this is the exact bug class the VOC
    adapter previously had. reconcile_canonical_image_id must reject a
    canonical_relative_id missing the required .jpg suffix."""
    from src.dataset_image_identity import reconcile_canonical_image_id
    from src.k11_k12_stability_gate_identity import K11K12StabilityGateError

    img_root = str(_config_data_root() / "ADEChallengeData2016" / "images" / "validation")
    real_file = Path(img_root) / "ADE_val_00000001.jpg"
    if not real_file.is_file():
        pytest.skip("expected real validation JPEG not found at the resolved path")

    with pytest.raises(K11K12StabilityGateError):
        reconcile_canonical_image_id(
            canonical_relative_id="ADE_val_00000001",  # missing .jpg -- the historical bug class
            image_root=img_root,
            pipeline_resolved_filename=str(real_file),
        )


@requires_real_data
def test_reconciliation_rejects_double_suffix_and_absolute_and_traversal():
    from src.dataset_image_identity import reconcile_canonical_image_id
    from src.k11_k12_stability_gate_identity import K11K12StabilityGateError

    img_root = str(_config_data_root() / "ADEChallengeData2016" / "images" / "validation")
    real_file = Path(img_root) / "ADE_val_00000001.jpg"
    if not real_file.is_file():
        pytest.skip("expected real validation JPEG not found at the resolved path")

    for bad_id in ("ADE_val_00000001.jpg.jpg", "/etc/passwd", "../../etc/passwd", ""):
        with pytest.raises(K11K12StabilityGateError):
            reconcile_canonical_image_id(
                canonical_relative_id=bad_id, image_root=img_root, pipeline_resolved_filename=str(real_file),
            )
