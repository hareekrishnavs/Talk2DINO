"""Direct CPU coverage of the real-dataset adapter used by the bounded
k11/k12 stability gate: ``PreparedDiagnosticImage``,
``_extract_prepared_image``, and ``_iter_prepared_images`` in
``diagnostics.run_k11_k12_stability``.

Regression coverage for the "'height' KeyError" defect: the real
COCOStuffDataset's ``img_infos``/``data_infos`` entries never carry
height/width (only ``filename``/``ann`` -- confirmed by reading mmseg's
own ``CustomDataset.load_annotations`` source directly). Image dimensions
only exist after the canonical test pipeline has actually loaded and
resized the image. Every synthetic fixture below reproduces the *exact*
real structure ``dataset[i]`` returns for this project's canonical test
pipeline (``LoadImageFromFile`` -> ``MultiScaleFlipAug`` with a single
``img_scale``/``flip=False`` -> ``Resize``/``RandomFlip``/``FloatImage``/
``ImageToTensor``/``Collect``), confirmed by reading mmcv/mmseg's real
pipeline source directly (``MultiScaleFlipAug.__call__``,
``Collect.__call__``, ``ImageToTensor.__call__``, ``mmcv.parallel.DataContainer``):

- ``results["img"]``       -> a Python ``list`` of exactly one ``(C, H, W)``
  ``torch.Tensor`` (the single canonical augmentation).
- ``results["img_metas"]`` -> a Python ``list`` of exactly one
  ``mmcv.parallel.DataContainer``-like object whose ``.data`` is a plain
  dict containing (among other keys) ``img_shape``, ``pad_shape``,
  ``filename``.

This pipeline has no ``Pad`` transform, so ``pad_shape`` -- when present
-- is expected to equal the processed tensor's own shape exactly for
every sample; the adapter treats any disagreement as a fail-closed
defect, never silently trusting one disagreeing source over another.

Importing ``diagnostics.run_k11_k12_stability`` requires no mmcv, mmseg,
or CUDA (its heavy dependencies are all lazy, function-local imports);
these tests call ``_extract_prepared_image``/``_iter_prepared_images``
directly against synthetic dataset stand-ins, never the real dataset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]

spec = importlib.util.spec_from_file_location(
    "diagnostics.run_k11_k12_stability", ROOT / "diagnostics/run_k11_k12_stability.py"
)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

from src.k11_k12_stability_gate_identity import K11K12StabilityGateError


class FakeDataContainer:
    """Stand-in for ``mmcv.parallel.DataContainer``: exposes exactly the
    ``.data`` property real code relies on to unwrap it."""

    def __init__(self, data):
        self.data = data


def _img_meta(height=100, width=100, *, pad=True, filename="img000.jpg"):
    meta = {
        "filename": filename,
        "ori_filename": filename,
        "ori_shape": (200, 200, 3),
        "img_shape": (height, width, 3),
        "scale_factor": 1.0,
        "flip": False,
        "flip_direction": "horizontal",
        "img_norm_cfg": {},
    }
    if pad:
        meta["pad_shape"] = (height, width, 3)
    return meta


def _tensor(height=100, width=100, channels=3, fill=0.5):
    return torch.full((channels, height, width), fill, dtype=torch.float32)


class FakeDataset:
    """A dataset stand-in whose ``img_infos`` deliberately lack height/
    width (matching the real COCOStuffDataset), and whose ``__getitem__``
    returns the real canonical pipeline's exact wrapped structure."""

    def __init__(self, samples):
        # samples: list of dicts with keys "img" (Tensor) and "img_meta" (dict)
        self._samples = samples
        self.img_infos = [{"filename": s.get("filename", f"img{i}.jpg"), "ann": {"seg_map": "x.png"}} for i, s in enumerate(samples)]
        self.getitem_calls = []

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        self.getitem_calls.append(index)
        sample = self._samples[index]
        img = sample.get("img_list", [sample["img"]] if "img" in sample else None)
        meta = sample.get("meta_list", [FakeDataContainer(sample["img_meta"])] if "img_meta" in sample else None)
        result = {}
        if img is not None:
            result["img"] = img
        if meta is not None:
            result["img_metas"] = meta
        return result


def _good_sample(height=100, width=100, filename="img0.jpg"):
    return {"img": _tensor(height, width), "img_meta": _img_meta(height, width, filename=filename)}


# ---------------------------------------------------------------------------
# 1-4: realistic fixtures, happy path
# ---------------------------------------------------------------------------


def test_real_img_infos_have_no_height_or_width_keys():
    dataset = FakeDataset([_good_sample()])
    assert "height" not in dataset.img_infos[0]
    assert "width" not in dataset.img_infos[0]
    assert set(dataset.img_infos[0]) == {"filename", "ann"}


def test_happy_path_extracts_verified_geometry():
    dataset = FakeDataset([_good_sample(height=448, width=448, filename="a.jpg")])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert prepared.inference_height == 448
    assert prepared.inference_width == 448
    assert prepared.image_id == "a.jpg"
    assert prepared.source_shape_provenance == "tensor==img_shape==pad_shape"
    assert tuple(prepared.image_tensor.shape) == (3, 448, 448)


def test_raw_header_size_ignored_processed_tensor_size_used():
    # The dataset's raw img_infos entry (if it had a size at all) would be
    # irrelevant; only the processed tensor/meta shape must be used. Here
    # ori_shape (a stand-in for a "raw header" size) is deliberately
    # different from the processed img_shape/tensor shape.
    meta = _img_meta(height=448, width=448)
    meta["ori_shape"] = (900, 900, 3)  # deliberately different "raw" size
    dataset = FakeDataset([{"img": _tensor(448, 448), "img_meta": meta}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert (prepared.inference_height, prepared.inference_width) == (448, 448)
    assert (prepared.inference_height, prepared.inference_width) != (900, 900)


def test_tensor_and_img_shape_agree_passes():
    dataset = FakeDataset([_good_sample(height=200, width=300)])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert (prepared.inference_height, prepared.inference_width) == (200, 300)


# ---------------------------------------------------------------------------
# 5: tensor/img_shape disagreement fails closed
# ---------------------------------------------------------------------------


def test_tensor_and_img_shape_disagree_fails_closed():
    meta = _img_meta(height=200, width=200)
    dataset = FakeDataset([{"img": _tensor(199, 200), "img_meta": meta}])  # tensor height differs
    with pytest.raises(K11K12StabilityGateError, match="disagrees"):
        runner._extract_prepared_image(dataset, 0)


# ---------------------------------------------------------------------------
# 6: pad_shape handling
# ---------------------------------------------------------------------------


def test_pad_shape_absent_is_accepted_with_img_shape_only_provenance():
    meta = _img_meta(height=150, width=150, pad=False)
    dataset = FakeDataset([{"img": _tensor(150, 150), "img_meta": meta}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert prepared.source_shape_provenance == "tensor==img_shape"


def test_pad_shape_disagreement_fails_closed():
    meta = _img_meta(height=150, width=150)
    meta["pad_shape"] = (160, 160, 3)  # unexpected: this pipeline has no Pad transform
    dataset = FakeDataset([{"img": _tensor(150, 150), "img_meta": meta}])
    with pytest.raises(K11K12StabilityGateError, match="pad_shape"):
        runner._extract_prepared_image(dataset, 0)


# ---------------------------------------------------------------------------
# 7: multiple augmentation wrappers rejected
# ---------------------------------------------------------------------------


def test_multiple_img_augmentations_rejected():
    dataset = FakeDataset([{
        "img_list": [_tensor(100, 100), _tensor(100, 100)],
        "meta_list": [FakeDataContainer(_img_meta()), FakeDataContainer(_img_meta())],
    }])
    with pytest.raises(K11K12StabilityGateError, match="exactly one canonical"):
        runner._extract_prepared_image(dataset, 0)


def test_multiple_img_metas_augmentations_rejected():
    dataset = FakeDataset([{
        "img_list": [_tensor(100, 100)],
        "meta_list": [FakeDataContainer(_img_meta()), FakeDataContainer(_img_meta())],
    }])
    with pytest.raises(K11K12StabilityGateError, match="exactly one canonical"):
        runner._extract_prepared_image(dataset, 0)


def test_zero_augmentations_rejected():
    dataset = FakeDataset([{"img_list": [], "meta_list": [FakeDataContainer(_img_meta())]}])
    with pytest.raises(K11K12StabilityGateError, match="exactly one canonical"):
        runner._extract_prepared_image(dataset, 0)


# ---------------------------------------------------------------------------
# 8: DataContainer/list/tensor extraction
# ---------------------------------------------------------------------------


def test_data_container_unwrapped_correctly():
    meta_dict = _img_meta(height=64, width=64)
    dataset = FakeDataset([{"img_list": [_tensor(64, 64)], "meta_list": [FakeDataContainer(meta_dict)]}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert (prepared.inference_height, prepared.inference_width) == (64, 64)


def test_plain_mapping_meta_without_data_container_also_accepted():
    # If a caller ever passes an already-unwrapped mapping instead of a
    # DataContainer, extraction must still work via the hasattr(.data) guard.
    dataset = FakeDataset([{"img_list": [_tensor(64, 64)], "meta_list": [_img_meta(64, 64)]}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert (prepared.inference_height, prepared.inference_width) == (64, 64)


# ---------------------------------------------------------------------------
# 9-11: missing/malformed structural fields
# ---------------------------------------------------------------------------


def test_missing_img_field_rejected():
    dataset = FakeDataset([{"meta_list": [FakeDataContainer(_img_meta())]}])
    with pytest.raises(K11K12StabilityGateError, match="missing an 'img' field"):
        runner._extract_prepared_image(dataset, 0)


def test_missing_img_metas_field_rejected():
    dataset = FakeDataset([{"img_list": [_tensor(100, 100)]}])
    with pytest.raises(K11K12StabilityGateError, match="missing an 'img_metas' field"):
        runner._extract_prepared_image(dataset, 0)


def test_img_not_a_list_rejected():
    # A dataset whose 'img' field is a bare tensor (not wrapped in the
    # canonical length-1 list MultiScaleFlipAug always produces) must be
    # rejected, not silently accepted as if it were the single augmentation.
    class BareImgDataset:
        img_infos = [{"filename": "x.jpg", "ann": {}}]

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return {"img": _tensor(100, 100), "img_metas": [FakeDataContainer(_img_meta())]}

    with pytest.raises(K11K12StabilityGateError, match="exactly one canonical"):
        runner._extract_prepared_image(BareImgDataset(), 0)


# ---------------------------------------------------------------------------
# 12: invalid/nonpositive dimensions
# ---------------------------------------------------------------------------


def test_zero_dimension_tensor_rejected():
    class ZeroDimDataset:
        img_infos = [{"filename": "x.jpg", "ann": {}}]
        def __len__(self): return 1
        def __getitem__(self, i):
            meta = _img_meta(height=0, width=100)
            return {"img": [torch.zeros(3, 0, 100)], "img_metas": [FakeDataContainer(meta)]}
    with pytest.raises(K11K12StabilityGateError, match="nonpositive"):
        runner._extract_prepared_image(ZeroDimDataset(), 0)


# ---------------------------------------------------------------------------
# 13: non-finite tensor rejected
# ---------------------------------------------------------------------------


def test_non_finite_tensor_rejected():
    tensor = _tensor(50, 50)
    tensor[0, 0, 0] = float("nan")
    dataset = FakeDataset([{"img": tensor, "img_meta": _img_meta(50, 50)}])
    with pytest.raises(K11K12StabilityGateError, match="non-finite"):
        runner._extract_prepared_image(dataset, 0)


def test_infinite_tensor_rejected():
    tensor = _tensor(50, 50)
    tensor[0, 0, 0] = float("inf")
    dataset = FakeDataset([{"img": tensor, "img_meta": _img_meta(50, 50)}])
    with pytest.raises(K11K12StabilityGateError, match="non-finite"):
        runner._extract_prepared_image(dataset, 0)


# ---------------------------------------------------------------------------
# 14, 20: dataset order preserved; one __getitem__ call per retained image
# ---------------------------------------------------------------------------


def test_iter_prepared_images_preserves_dataset_order():
    dataset = FakeDataset([_good_sample(filename=f"img{i}.jpg") for i in range(4)])
    prepared_list = list(runner._iter_prepared_images(dataset))
    assert [p.dataset_index for p in prepared_list] == [0, 1, 2, 3]
    assert [p.image_id for p in prepared_list] == ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]


def test_iter_prepared_images_calls_getitem_exactly_once_per_index():
    dataset = FakeDataset([_good_sample(filename=f"img{i}.jpg") for i in range(3)])
    list(runner._iter_prepared_images(dataset))
    assert dataset.getitem_calls == [0, 1, 2]


def test_iter_prepared_images_is_lazy():
    dataset = FakeDataset([_good_sample(filename=f"img{i}.jpg") for i in range(5)])
    generator = runner._iter_prepared_images(dataset)
    next(generator)
    next(generator)
    assert dataset.getitem_calls == [0, 1]  # not all 5 pulled eagerly


# ---------------------------------------------------------------------------
# 21: input dataset records and tensors unmodified
# ---------------------------------------------------------------------------


def test_extraction_does_not_mutate_input_meta_or_tensor():
    meta = _img_meta(100, 100)
    tensor = _tensor(100, 100)
    tensor_before = tensor.clone()
    meta_before = dict(meta)
    dataset = FakeDataset([{"img": tensor, "img_meta": meta}])
    runner._extract_prepared_image(dataset, 0)
    assert torch.equal(tensor, tensor_before)
    assert meta == meta_before


def test_extraction_returns_a_detached_cpu_copy_not_the_same_tensor_object():
    tensor = _tensor(100, 100)
    dataset = FakeDataset([{"img": tensor, "img_meta": _img_meta(100, 100)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert torch.equal(prepared.image_tensor, tensor)
    assert not prepared.image_tensor.requires_grad


# ---------------------------------------------------------------------------
# Tensor-ownership contract: PreparedDiagnosticImage.image_tensor must
# independently own its storage, never alias the source pipeline tensor
# (regression coverage for the confirmed .detach().cpu()-without-.clone()
# aliasing defect: on an already-CPU, ungraded tensor -- which every
# sample from this pipeline is -- .detach().cpu() alone can return an
# object that shares the exact same underlying storage as its input).
# ---------------------------------------------------------------------------


def test_ownership_1_prepared_tensor_values_equal_source():
    source = _tensor(40, 40, fill=0.42)
    dataset = FakeDataset([{"img": source, "img_meta": _img_meta(40, 40)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert torch.equal(prepared.image_tensor, source)


def test_ownership_2_data_ptr_differs_from_source():
    source = _tensor(40, 40)
    dataset = FakeDataset([{"img": source, "img_meta": _img_meta(40, 40)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert prepared.image_tensor.data_ptr() != source.data_ptr()


def test_ownership_2b_data_ptr_differs_from_list_wrapped_source():
    # The real pipeline always wraps the tensor in a length-1 list
    # (MultiScaleFlipAug); confirm the prepared tensor doesn't alias that
    # wrapped element either, not just a bare unwrapped variable.
    source = _tensor(40, 40)
    dataset = FakeDataset([{"img_list": [source], "meta_list": [FakeDataContainer(_img_meta(40, 40))]}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert prepared.image_tensor.data_ptr() != source.data_ptr()


def test_ownership_3_mutating_source_after_extraction_does_not_change_prepared():
    source = _tensor(40, 40, fill=0.1)
    dataset = FakeDataset([{"img": source, "img_meta": _img_meta(40, 40)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    prepared_before = prepared.image_tensor.clone()
    source[0, 0, 0] = 777.0
    assert torch.equal(prepared.image_tensor, prepared_before)
    assert prepared.image_tensor[0, 0, 0].item() != 777.0


def test_ownership_4_mutating_prepared_does_not_change_source():
    source = _tensor(40, 40, fill=0.1)
    source_before = source.clone()
    dataset = FakeDataset([{"img": source, "img_meta": _img_meta(40, 40)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    prepared.image_tensor[0, 0, 0] = 888.0
    assert torch.equal(source, source_before)
    assert source[0, 0, 0].item() != 888.0


def test_ownership_5_prepared_tensor_is_cpu():
    dataset = FakeDataset([{"img": _tensor(20, 20), "img_meta": _img_meta(20, 20)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert prepared.image_tensor.device.type == "cpu"


def test_ownership_6_prepared_tensor_requires_grad_false():
    source = _tensor(20, 20).requires_grad_(False)
    dataset = FakeDataset([{"img": source, "img_meta": _img_meta(20, 20)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert prepared.image_tensor.requires_grad is False


def test_ownership_7_dtype_and_shape_preserved():
    source = _tensor(30, 50, channels=3).to(torch.float32)
    dataset = FakeDataset([{"img": source, "img_meta": _img_meta(30, 50)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert prepared.image_tensor.dtype == torch.float32
    assert tuple(prepared.image_tensor.shape) == (3, 30, 50)


def test_ownership_8_non_contiguous_source_handled_correctly():
    # A permuted view is non-contiguous; the documented contract is that
    # the prepared tensor is an independently-owned copy with correct
    # values regardless of the source's contiguity.
    base = torch.arange(3 * 20 * 20, dtype=torch.float32).reshape(20, 20, 3)
    non_contiguous = base.permute(2, 0, 1)  # (3, 20, 20), non-contiguous
    assert not non_contiguous.is_contiguous()
    dataset = FakeDataset([{"img": non_contiguous, "img_meta": _img_meta(20, 20)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    assert torch.equal(prepared.image_tensor, non_contiguous)
    assert prepared.image_tensor.data_ptr() != non_contiguous.data_ptr()


def test_ownership_9_crop_mutation_does_not_corrupt_retained_canonical_tensor():
    # _process_window's crop must not be a view sharing storage with the
    # retained PreparedDiagnosticImage tensor.
    dataset = FakeDataset([{"img": _tensor(100, 100, fill=0.2), "img_meta": _img_meta(100, 100)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    canonical_before = prepared.image_tensor.clone()

    class FakeSnapshot:
        unary_scores = [None]
        dino_features = [None]
        grid_hw = None

    class FakeModel:
        def generate_patch_snapshot(self, crop, text_embedding):
            crop += 999.0  # simulate a downstream in-place mutation
            return FakeSnapshot()

    class FakeInference:
        model = FakeModel()
        text_embedding = None

    manifest_entry = {"crop_origin": [0, 0], "crop_end": [50, 50]}
    runner._process_window(FakeInference(), prepared.image_tensor.unsqueeze(0), manifest_entry)
    assert torch.equal(prepared.image_tensor, canonical_before)


def test_ownership_10_metadata_mutation_does_not_cross_aliases():
    meta = _img_meta(30, 30)
    dataset = FakeDataset([{"img": _tensor(30, 30), "img_meta": meta}])
    prepared = runner._extract_prepared_image(dataset, 0)
    prepared.img_metas["filename"] = "mutated.jpg"
    assert meta["filename"] != "mutated.jpg"

    meta2 = _img_meta(30, 30)
    dataset2 = FakeDataset([{"img": _tensor(30, 30), "img_meta": meta2}])
    prepared2 = runner._extract_prepared_image(dataset2, 0)
    meta2["filename"] = "mutated_source.jpg"
    assert prepared2.img_metas["filename"] != "mutated_source.jpg"


def test_ownership_11_repeated_preparation_is_deterministic():
    dataset = FakeDataset([{"img": _tensor(35, 45, fill=0.6), "img_meta": _img_meta(35, 45)}])
    prepared_a = runner._extract_prepared_image(dataset, 0)
    prepared_b = runner._extract_prepared_image(dataset, 0)
    assert torch.equal(prepared_a.image_tensor, prepared_b.image_tensor)
    assert prepared_a.inference_height == prepared_b.inference_height
    assert prepared_a.inference_width == prepared_b.inference_width
    assert prepared_a.source_shape_provenance == prepared_b.source_shape_provenance


def test_ownership_12_input_sample_unchanged_after_repeated_preparation():
    source = _tensor(25, 25, fill=0.3)
    source_before = source.clone()
    meta = _img_meta(25, 25)
    meta_before = dict(meta)
    dataset = FakeDataset([{"img": source, "img_meta": meta}])
    runner._extract_prepared_image(dataset, 0)
    runner._extract_prepared_image(dataset, 0)
    assert torch.equal(source, source_before)
    assert meta == meta_before


def test_ownership_13_manifest_geometry_and_digest_unchanged_after_downstream_processing():
    from src.k11_k12_stability_manifest import ManifestGeometry, build_bounded_manifest

    dataset = FakeDataset([{"img": _tensor(100, 100, fill=0.2), "img_meta": _img_meta(100, 100)}])
    prepared = runner._extract_prepared_image(dataset, 0)
    geometry = ManifestGeometry(crop_height=100, crop_width=100, stride_height=50, stride_width=50)
    manifest_before, digest_before = build_bounded_manifest([prepared.geometry], canonical_window_count=1, geometry=geometry)

    # simulate downstream window processing that could (incorrectly)
    # mutate the retained tensor if ownership were broken
    class FakeSnapshot:
        unary_scores = [None]
        dino_features = [None]
        grid_hw = None

    class FakeModel:
        def generate_patch_snapshot(self, crop, text_embedding):
            crop += 1234.0
            return FakeSnapshot()

    class FakeInference:
        model = FakeModel()
        text_embedding = None

    for entry in manifest_before:
        runner._process_window(FakeInference(), prepared.image_tensor.unsqueeze(0), entry)

    manifest_after, digest_after = build_bounded_manifest([prepared.geometry], canonical_window_count=1, geometry=geometry)
    assert digest_before == digest_after
    assert manifest_before == manifest_after


# ---------------------------------------------------------------------------
# 24-25: no PIL/header dependency; no img_infos["height"]/["width"] access
# ---------------------------------------------------------------------------


def test_no_pil_import_in_diagnostics_module():
    # AST-based: only real import statements count, never a mention of
    # "PIL" inside prose (docstrings/comments explaining what is NOT used).
    import ast

    tree = ast.parse((ROOT / "diagnostics/run_k11_k12_stability.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] == "PIL" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.split(".")[0] == "PIL"


def test_no_executable_access_to_img_infos_height_or_width_in_adapter():
    import ast
    import inspect

    source = inspect.getsource(runner._extract_prepared_image)
    tree = ast.parse(source)
    function_node = tree.body[0]
    # exclude the function's own docstring (which legitimately mentions
    # "img_infos" in prose explaining why it is NOT used) before scanning
    body = function_node.body[1:] if (
        function_node.body and isinstance(function_node.body[0], ast.Expr)
        and isinstance(function_node.body[0].value, ast.Constant)
        and isinstance(function_node.body[0].value.value, str)
    ) else function_node.body
    body_source = ast.unparse(ast.Module(body=body, type_ignores=[]))
    assert "img_infos" not in body_source
    assert "data_infos" not in body_source
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in ("height", "width"):
            raise AssertionError("_extract_prepared_image must not subscript ['height']/['width'] from raw metadata")


# ---------------------------------------------------------------------------
# Integration: adapter geometries feed correctly into build_bounded_manifest
# ---------------------------------------------------------------------------


def test_prepared_image_geometry_matches_image_geometry_record_contract():
    from src.k11_k12_stability_manifest import ImageGeometryRecord

    dataset = FakeDataset([_good_sample(height=448, width=448, filename="a.jpg")])
    prepared = runner._extract_prepared_image(dataset, 0)
    record = prepared.geometry
    assert isinstance(record, ImageGeometryRecord)
    assert record.dataset_index == 0
    assert record.image_id == "a.jpg"
    assert record.inference_height == 448
    assert record.inference_width == 448


def test_end_to_end_generator_to_manifest_stops_early_and_matches_geometry():
    from src.k11_k12_stability_manifest import ManifestGeometry, build_bounded_manifest

    dataset = FakeDataset([_good_sample(height=100, width=100, filename=f"img{i}.jpg") for i in range(5)])
    geometry = ManifestGeometry(crop_height=100, crop_width=100, stride_height=50, stride_width=50)

    def _geometry_records():
        for prepared in runner._iter_prepared_images(dataset):
            yield prepared.geometry

    manifest, digest = build_bounded_manifest(_geometry_records(), canonical_window_count=2, geometry=geometry)
    assert len(manifest) == 2
    assert [e["dataset_index"] for e in manifest] == [0, 1]
    # only the first 2 images (1 window each) were needed -- images 2-4
    # must never have been visited
    assert dataset.getitem_calls == [0, 1]
