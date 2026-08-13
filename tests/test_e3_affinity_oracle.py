import inspect
import io
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import src.e3_affinity_oracle as oracle


def normalized(rows, columns, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(rows, columns, generator=generator), dim=-1)


def loop_graph(features, k, power):
    indices, weights = [], []
    for row in range(features.shape[0]):
        candidates = []
        for column in range(features.shape[0]):
            if row == column:
                continue
            cosine = float(features[row] @ features[column])
            candidates.append((max(cosine, 0) ** power, column))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen = candidates[:k]
        total = sum(value for value, _ in chosen)
        if total == 0:
            indices.append([row] * k)
            weights.append([1.0] + [0.0] * (k - 1))
        else:
            indices.append([index for _, index in chosen])
            weights.append([value / total for value, _ in chosen])
    return torch.tensor(indices), torch.tensor(weights)


def test_affinity_graph_matches_independent_loop_and_excludes_self():
    features = normalized(9, 7)
    indices, weights, zero = oracle.build_knn_graph(
        features.float(), knn_k=4, affinity_power=3.0
    )
    expected_indices, expected_weights = loop_graph(features, 4, 3.0)
    assert zero == 0
    torch.testing.assert_close(indices.long(), expected_indices)
    torch.testing.assert_close(weights.float(), expected_weights, atol=5e-4, rtol=5e-4)
    assert not torch.any(indices.long() == torch.arange(9)[:, None])


def test_exact_affinity_ties_use_smaller_patch_index():
    features = F.normalize(torch.ones(6, 4), dim=-1)
    indices, _, _ = oracle.build_knn_graph(features, knn_k=3)
    assert indices[4].tolist() == [0, 1, 2]
    assert indices[0].tolist() == [1, 2, 3]


def test_zero_neighbour_rows_use_identity_fallback():
    features = torch.eye(5, dtype=torch.float32)
    indices, weights, zero = oracle.build_knn_graph(features, knn_k=3)
    assert zero == 5
    assert indices[:, 0].long().tolist() == list(range(5))
    torch.testing.assert_close(weights[:, 0].float(), torch.ones(5))
    torch.testing.assert_close(weights[:, 1:].float(), torch.zeros(5, 2))


def test_k12_shapes_dtypes_range_and_int16_decode():
    indices, weights, _ = oracle.build_knn_graph(normalized(32, 16), knn_k=12)
    assert indices.shape == (32, 12) and indices.dtype == torch.int16
    assert weights.shape == (32, 12) and weights.dtype == torch.float16
    assert int(indices.min()) >= 0 and int(indices.max()) < 32
    torch.testing.assert_close(
        indices.long().to(torch.int16), indices
    )


def test_invalid_graph_cache_and_nonfinite_scores_are_rejected():
    indices, weights, _ = oracle.build_knn_graph(normalized(6, 4), knn_k=2)
    invalid_indices = indices.clone()
    invalid_indices[0, 0] = 7
    with pytest.raises(oracle.AffinityOracleError, match="patch range"):
        oracle.validate_graph(
            invalid_indices, weights, patch_count=6, knn_k=2
        )
    invalid_weights = weights.clone()
    invalid_weights[0, 0] = float("nan")
    with pytest.raises(oracle.AffinityOracleError, match="finite"):
        oracle.validate_graph(
            indices, invalid_weights, patch_count=6, knn_k=2
        )
    with pytest.raises(oracle.AffinityOracleError, match="non-finite"):
        oracle.propagate_scores(
            torch.tensor([[float("inf")] * 6]), indices, weights, 0.2
        )


def loop_propagation(scores, indices, weights, alpha, steps):
    base = scores.float()
    current = base.clone()
    alpha_vector = (
        torch.full((scores.shape[0],), alpha)
        if isinstance(alpha, float) else alpha.float()
    )
    for _ in range(steps):
        spread = torch.zeros_like(current)
        for c in range(scores.shape[0]):
            for patch in range(scores.shape[1]):
                for neighbour in range(indices.shape[1]):
                    spread[c, patch] += (
                        weights[patch, neighbour]
                        * current[c, indices[patch, neighbour]]
                    )
        current = (1 - alpha_vector[:, None]) * base + alpha_vector[:, None] * spread
    return current


def test_propagation_matches_loop_scalar_and_ten_step_restart():
    features = normalized(7, 5)
    indices, weights, _ = oracle.build_knn_graph(features, knn_k=3)
    scores = torch.randn(4, 7)
    actual = oracle.propagate_scores(scores, indices, weights, 0.4, propagation_steps=10)
    expected = loop_propagation(scores, indices.long(), weights.float(), 0.4, 10)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_per_class_alpha_broadcast_matches_loop():
    indices, weights, _ = oracle.build_knn_graph(normalized(6, 4), knn_k=2)
    scores = torch.randn(3, 6)
    alpha = torch.tensor([0.1, 0.5, 0.9])
    actual = oracle.propagate_scores(scores, indices, weights, alpha, propagation_steps=3)
    expected = loop_propagation(scores, indices.long(), weights.float(), alpha, 3)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_alpha_zero_is_bitwise_same_tensor():
    indices, weights, _ = oracle.build_knn_graph(normalized(6, 4), knn_k=2)
    scores = torch.randn(3, 6, dtype=torch.float64)
    result = oracle.propagate_scores(scores, indices, weights, 0.0)
    assert result is scores
    assert torch.equal(result, scores)


def test_two_fully_separated_cliques_saturate_to_their_own_class():
    # 16 patches in two 8-patch cliques with no cross-clique edges; one
    # informative patch per clique (a strong, correct signal for that
    # clique's class), the rest exactly zero for both classes. alpha=0 must
    # reproduce the input exactly; alpha->1 with enough steps must saturate
    # every patch in a clique to that clique's class, since propagation can
    # never leak across a fully-separated graph boundary.
    patches, classes, k = 16, 2, 3
    clique_a, clique_b = list(range(8)), list(range(8, 16))
    indices = torch.zeros(patches, k, dtype=torch.int64)
    weights = torch.full((patches, k), 1.0 / k)
    for p in range(patches):
        clique = clique_a if p in clique_a else clique_b
        indices[p] = torch.tensor([c for c in clique if c != p][:k])
    indices16, weights16 = indices.to(torch.int16), weights.to(torch.float16)

    raw = torch.zeros(classes, patches)
    raw[0, 0], raw[1, 0] = 5.0, -5.0
    raw[1, 8], raw[0, 8] = 5.0, -5.0

    identity = oracle.propagate_scores(raw, indices16, weights16, 0.0, propagation_steps=10)
    assert torch.equal(identity, raw)

    saturated = oracle.propagate_scores(
        raw, indices16, weights16, 0.999, propagation_steps=200
    )
    predicted = saturated.argmax(dim=0)
    assert predicted.tolist() == [0] * 8 + [1] * 8


@pytest.mark.parametrize("alpha", (-0.1, 1.0, float("nan"), float("inf")))
def test_invalid_alpha_is_rejected(alpha):
    indices, weights, _ = oracle.build_knn_graph(normalized(6, 4), knn_k=2)
    with pytest.raises(oracle.AffinityOracleError):
        oracle.propagate_scores(torch.randn(2, 6), indices, weights, alpha)


def test_pre_sigmoid_interpolation_applies_sigmoid_exactly_once():
    scores = torch.tensor([[0.0, 1.0, -1.0, 2.0]])
    result = oracle.interpolate_window_scores(
        scores, patch_grid=(2, 2), crop_size=(2, 2)
    )
    torch.testing.assert_close(result, scores.sigmoid().reshape(1, 2, 2))


def test_stitch_and_rescale_match_independent_e3_operations():
    first = torch.ones(2, 2, 3)
    second = torch.full((2, 2, 3), 3.0)
    coordinates = [(0, 0, 2, 3), (0, 2, 2, 5)]
    actual = oracle.stitch_windows(
        [first, second], coordinates, image_shape=(2, 5)
    )
    expected = torch.tensor(
        [[[1, 1, 2, 3, 3], [1, 1, 2, 3, 3]]] * 2,
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)
    rescaled = oracle.rescale_logits(actual, img_shape=(2, 4), ori_shape=(4, 8))
    reference = F.interpolate(
        actual[:, :2, :4].unsqueeze(0), (4, 8),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    torch.testing.assert_close(rescaled, reference)


def test_background_and_final_argmax_match_e3():
    masks = torch.tensor([[[0.3, 0.8]], [[0.2, 0.1]]])
    logits = oracle.add_background_channel(
        masks, with_background=True, background_threshold=0.4
    )
    expected = torch.cat((torch.full((1, 1, 2), 0.4), masks))
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(
        oracle.final_prediction(logits), expected.softmax(0).argmax(0)
    )


def test_confusion_metrics_match_independent_pixel_loop():
    prediction = torch.tensor([[0, 1, 2], [1, 2, 0]])
    target = torch.tensor([[0, 2, 2], [1, 255, 0]])
    actual = oracle.confusion_from_prediction(
        prediction, target, num_classes=3, ignore_index=255
    )
    expected = torch.zeros(3, 3, dtype=torch.int64)
    for pred, truth in zip(prediction.flatten(), target.flatten()):
        if truth != 255:
            expected[truth, pred] += 1
    assert torch.equal(actual, expected)
    metrics = oracle.metrics_from_confusion(actual)
    assert metrics["aAcc"] == pytest.approx(80.0)


def test_fp16_control_treats_one_sided_class_support_as_failure():
    assert oracle.per_class_iou_differences(
        [None, None, 25.0], [None, 0.0, 25.02]
    ) == pytest.approx([0.0, 100.0, 0.02])


def brute_candidate(scores, candidate, class_index):
    changed = scores.clone()
    changed[class_index] = candidate
    return changed.argmax(0)


@pytest.mark.parametrize("class_index", (0, 1, 3))
def test_stable_best_other_matches_bruteforce_under_random_and_ties(class_index):
    scores = torch.randn(5, 8, 9)
    scores[0, 0, 0] = scores[1, 0, 0] = 2.0
    top = oracle.stable_top2(scores)
    for candidate in (scores[class_index] + 0.2, scores[class_index] - 0.2, scores[class_index]):
        actual = oracle.candidate_class_prediction(candidate, class_index, *top)
        assert torch.equal(actual, brute_candidate(scores, candidate, class_index))


def test_coupled_candidate_iou_matches_full_channel_competition():
    scores = torch.randn(171, 5, 7)
    target = torch.randint(0, 171, (5, 7))
    class_index = 83
    candidate = torch.randn(5, 7)
    prediction = oracle.candidate_class_prediction(
        candidate, class_index, *oracle.stable_top2(scores)
    )
    brute = brute_candidate(scores, candidate, class_index)
    assert torch.equal(prediction, brute)
    assert torch.equal(
        oracle.confusion_from_prediction(prediction, target, num_classes=171),
        oracle.confusion_from_prediction(brute, target, num_classes=171),
    )


def test_joint_vector_is_exact_full_argmax_on_synthetic_window():
    features = normalized(10, 6)
    indices, weights, _ = oracle.build_knn_graph(features, knn_k=3)
    scores = torch.randn(171, 10)
    alpha = torch.linspace(0, 0.95, 171)
    spread = oracle.propagate_scores(scores, indices, weights, alpha, propagation_steps=2)
    brute = loop_propagation(scores, indices.long(), weights.float(), alpha, 2)
    assert torch.equal(spread.argmax(0), brute.argmax(0))


def test_subset_and_split_are_deterministic_disjoint_complete():
    classes = [{index % 7, (index * 3) % 11} for index in range(100)]
    first = oracle.coverage_subset(classes, 40, seed=42)
    assert first == oracle.coverage_subset(classes, 40, seed=42)
    a, b = oracle.split_balanced_halves(classes, seed=42)
    assert len(a) == len(b) == 50
    assert set(a).isdisjoint(b)
    assert sorted(a + b) == list(range(100))


def test_greedy_tie_break_is_deterministic():
    alpha, reason = oracle.choose_alpha(
        [(0.1, 3.0), (0.3, 3.0), (0.8, 2.0)], global_alpha=0.2
    )
    assert alpha == 0.1
    assert reason == "distance_then_smallest_alpha"


def test_cache_shard_contract_and_no_patch_features(tmp_path):
    scores = torch.zeros(2, 171, 32, 32, dtype=torch.float16)
    indices = torch.zeros(2, 1024, 12, dtype=torch.int16)
    weights = torch.zeros(2, 1024, 12, dtype=torch.float16)
    weights[:, :, 0] = 1
    shard = {
        "raw_scores": scores, "knn_indices": indices,
        "knn_weights": weights,
        "window_image_index": torch.tensor([0, 1], dtype=torch.int64),
        "window_coordinates": torch.tensor([[0, 0, 448, 448]] * 2, dtype=torch.int32),
        "window_grid_indices": torch.tensor([[0, 0]] * 2, dtype=torch.int32),
    }
    validated = oracle.validate_cache_shard(
        shard, class_count=171, patch_grid=(32, 32), knn_k=12,
        score_dtype=torch.float16,
    )
    assert set(validated) == oracle.SHARD_KEYS
    bad = dict(shard, patch_features=torch.zeros(1))
    with pytest.raises(oracle.AffinityOracleError, match="closed schema"):
        oracle.validate_cache_shard(
            bad, class_count=171, patch_grid=(32, 32), knn_k=12,
            score_dtype=torch.float16,
        )


def test_online_capture_publishes_manifest_last_and_fp16_control(
    tmp_path, monkeypatch
):
    identity_paths = {}
    for name in (
        "e3_config", "projection_config", "e3_checkpoint",
        "dino_checkpoint", "clip_checkpoint", "dataset_config",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        identity_paths[name] = str(path)
    provenance = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": True,
        "source_git_diff_sha256": "b" * 64,
    }
    monkeypatch.setattr(oracle, "git_provenance", lambda *a, **k: provenance)
    annotation = tmp_path / "annotation.png"
    from PIL import Image
    import numpy as np
    Image.fromarray(np.zeros((448, 448), dtype=np.uint8)).save(annotation)
    capture = oracle.OnlineAffinityOracleCapture(
        output_dir=tmp_path / "cache",
        protocol=oracle.OracleProtocol(with_background=False),
        class_names=[f"class-{i}" for i in range(171)],
        source_image_count=5000, max_images=1, windows_per_shard=2,
        identity_paths=identity_paths, dino_identity="dinov2_vitb14_reg",
        text_embedding_sha256="d" * 64,
        allow_dirty_source=True, overwrite=False, commands=["synthetic"],
    )
    metadata = {
        "filename": "image.jpg", "ori_filename": "image.jpg",
        "img_shape": (448, 448, 3), "ori_shape": (448, 448, 3),
        "pad_shape": (448, 448, 3), "flip": False,
    }
    capture.begin_image(metadata, dataset_index=0, annotation_path=str(annotation))
    capture.set_window(coordinates=(0, 0, 448, 448), grid_indices=(0, 0))
    features = normalized(1024, 768).T.reshape(1, 768, 32, 32)
    raw = torch.zeros(1, 171, 32, 32)
    capture.observe(features, raw)
    capture.end_image(
        resized_input_shape=(448, 448),
        reference_prediction=torch.zeros(448, 448, dtype=torch.int64),
    )
    assert not (tmp_path / "cache" / "manifest.json").exists()
    changed_provenance = dict(provenance, source_git_commit="c" * 40)
    monkeypatch.setattr(
        oracle, "git_provenance", lambda *args, **kwargs: changed_provenance
    )
    with pytest.raises(oracle.AffinityOracleError, match="Git identity changed"):
        capture.finalize()
    assert not (tmp_path / "cache" / "manifest.json").exists()
    assert list((tmp_path / "cache" / "shards").glob("windows-*.pth"))
    assert not list((tmp_path / "cache").rglob("*.tmp"))
    monkeypatch.setattr(oracle, "git_provenance", lambda *args, **kwargs: provenance)
    manifest = capture.finalize()
    assert manifest["fp16_suitable"]
    assert manifest["fp16_control"]["differing_valid_pixels"] == 0
    assert oracle.cache_summary(tmp_path / "cache")["windows"] == 1
    loaded = torch.load(
        tmp_path / "cache" / manifest["shards"][0]["name"],
        map_location="cpu", weights_only=True,
    )
    assert set(loaded) == oracle.SHARD_KEYS
    assert all("feature" not in key for key in loaded)

    import argparse
    import run_e3_affinity_oracle as runner
    baseline_path = tmp_path / "baseline_control.json"
    runner.baseline(argparse.Namespace(
        cache=tmp_path / "cache", output=baseline_path,
        device="cpu", overwrite=False,
    ))
    baseline = runner._load_result(baseline_path, "baseline")
    runner._compatible(tmp_path / "cache", baseline)
    assert baseline["payload"]["metrics"]["mIoU"] == pytest.approx(100.0)

    # Closed manifests reject unknown/missing/non-finite content before a
    # sweep can deserialize data under ambiguous semantics.
    manifest_path = tmp_path / "cache" / "manifest.json"
    original = manifest_path.read_text()
    mutated = json.loads(original)
    mutated["unknown"] = 1
    manifest_path.write_text(json.dumps(mutated))
    with pytest.raises(oracle.AffinityOracleError, match="closed schema"):
        oracle.load_cache_manifest(tmp_path / "cache")
    mutated = json.loads(original)
    del mutated["knn_k"]
    manifest_path.write_text(json.dumps(mutated))
    with pytest.raises(oracle.AffinityOracleError, match="closed schema"):
        oracle.load_cache_manifest(tmp_path / "cache")
    mutated = json.loads(original)
    mutated["construction_finished_at"] = float("nan")
    manifest_path.write_text(json.dumps(mutated))
    with pytest.raises(oracle.AffinityOracleError, match="non-finite"):
        oracle.load_cache_manifest(tmp_path / "cache")
    manifest_path.write_text(original)

    clean_provenance = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    monkeypatch.setattr(oracle, "git_provenance", lambda *a, **k: clean_provenance)
    full_output = tmp_path / "full-cache"
    with pytest.raises(oracle.AffinityOracleError, match="at least 50"):
        oracle.OnlineAffinityOracleCapture(
            output_dir=full_output,
            protocol=oracle.OracleProtocol(),
            class_names=[f"class-{i}" for i in range(171)],
            source_image_count=5000, max_images=None, windows_per_shard=2,
            identity_paths=identity_paths, dino_identity="dinov2_vitb14_reg",
            text_embedding_sha256="d" * 64,
            allow_dirty_source=False, overwrite=False, commands=["synthetic"],
            fp16_control_manifest=manifest_path,
        )
    assert not full_output.exists()


def test_capture_preserves_patch_raster_order(tmp_path, monkeypatch):
    captured = {}

    def graph(features, **kwargs):
        captured["features"] = features.clone()
        indices = torch.zeros(1024, 12, dtype=torch.int16)
        weights = torch.zeros(1024, 12, dtype=torch.float16)
        weights[:, 0] = 1
        return indices, weights, 0

    monkeypatch.setattr(oracle, "build_knn_graph", graph)
    writer = oracle.AffinityOracleCacheWriter(
        tmp_path / "cache", protocol=oracle.OracleProtocol(),
        windows_per_shard=2,
    )
    feature_map = normalized(1024, 768).T.reshape(1, 768, 32, 32)
    writer.add_window(
        torch.zeros(1, 171, 32, 32), feature_map,
        image_index=0, coordinates=(0, 0, 448, 448), grid_indices=(0, 0),
    )
    torch.testing.assert_close(
        captured["features"], feature_map.flatten(2).transpose(1, 2)[0]
    )


def test_half_b_annotations_are_not_read_during_half_a_greedy_fit(monkeypatch):
    manifest = {
        "class_count": 171,
        "class_names": [f"class-{index}" for index in range(171)],
        "protocol": {"ignore_index": 255},
        "images": [
            {"dataset_index": 0, "annotation_path": "half-a"},
            {"dataset_index": 1, "annotation_path": "half-b"},
        ],
    }
    images = [
        ({"dataset_index": 0, "annotation_path": "half-a"}, [{}]),
        ({"dataset_index": 1, "annotation_path": "half-b"}, [{}]),
    ]
    accessed = []

    monkeypatch.setattr(
        oracle, "load_cache_manifest", lambda path, **kwargs: manifest
    )
    monkeypatch.setattr(oracle, "iter_cached_images", lambda *args: iter(images))

    def annotation(path):
        accessed.append(path)
        if path == "half-b":
            raise AssertionError("half-B label was accessed during fitting")
        return torch.zeros(2, 2, dtype=torch.int64)

    monkeypatch.setattr(oracle, "load_annotation", annotation)
    monkeypatch.setattr(
        oracle, "replay_cached_image_logits",
        lambda *args, **kwargs: torch.zeros(171, 2, 2),
    )
    monkeypatch.setattr(
        oracle, "replay_cached_class_channel",
        lambda *args, **kwargs: torch.zeros(2, 2),
    )
    image_support, pixel_support = oracle.support_for_indices(
        Path("unused"), {0}
    )
    assert accessed == ["half-a"]
    assert image_support[0] == 1 and pixel_support[0] == 4
    accessed.clear()
    result = oracle.greedy_fit(
        Path("unused"), global_alpha=0.0, selected_indices={0}, device="cpu"
    )
    assert accessed == ["half-a"]
    assert result[0]["positive_fitting_image_count"] == 1


def test_result_payload_schema_rejects_unknown_and_nonfinite():
    import run_e3_affinity_oracle as runner

    with pytest.raises(oracle.AffinityOracleError, match="closed baseline"):
        runner._validate_payload("baseline", {"unknown": 1})
    payload = {
        "label": "alpha_zero_control", "alpha": float("inf"), "metrics": {},
        "canonical_e3": {}, "canonical_reported_precision_matches": False,
    }
    with pytest.raises(oracle.AffinityOracleError, match="non-finite"):
        runner._validate_payload("baseline", payload)


def test_atomic_json_rejects_overwrite_and_nonfinite(tmp_path):
    path = tmp_path / "result.json"
    oracle.atomic_json(path, {"value": 1.0})
    with pytest.raises(FileExistsError):
        oracle.atomic_json(path, {"value": 2.0})
    with pytest.raises(ValueError):
        oracle.atomic_json(tmp_path / "bad.json", {"value": float("nan")})


def test_protocol_rejects_noncanonical_modes():
    for protocol in (
        oracle.OracleProtocol(pamr=True),
        oracle.OracleProtocol(flip=True),
        oracle.OracleProtocol(crop_size=(224, 224)),
        oracle.OracleProtocol(class_count=170),
    ):
        with pytest.raises(oracle.AffinityOracleError):
            protocol.validate()


def test_disabled_config_and_observer_preserve_original_e3_path():
    config = Path(
        "src/open_vocabulary_segmentation/configs/stuff/"
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    ).read_text()
    assert "enabled: false" in config
    source = Path(
        "src/open_vocabulary_segmentation/models/dinotext/masker.py"
    ).read_text()
    start = source.rindex("def forward_seg")
    normalize = source.index("image_feat = us.normalize(image_feat, dim=1)", start)
    dot = source.index('simmap = torch.einsum("b c h w, n c -> b n h w"', start)
    observer = source.index("self.affinity_oracle_observer(image_feat, simmap)", start)
    sigmoid = source.index("hard_mask, soft_mask = self.sim2mask(simmap", observer)
    assert normalize < dot < observer < sigmoid


def test_oracle_module_does_not_import_e4_through_e9():
    source = inspect.getsource(oracle)
    for experiment in ("e4_", "e5_", "e6_", "e7_", "e8_", "e9_"):
        assert f"import {experiment}" not in source


# --- Part E: feature-capture verification (E3/E4) ---------------------------

def clustered_features(n_blocks, block_size, dim=768, seed=7):
    """n_blocks*block_size unit vectors s.t. every vector's k<block_size-1
    nearest neighbours are EXACTLY its (block_size-1) block-mates (cosine==1,
    self excluded) with a huge, well-separated gap to every other block's
    near-zero cross-block cosine. This makes the resulting knn graph immune
    to fp16-storage rounding noise, so E3 round-trip tests can assert exact
    agreement deterministically instead of a statistical threshold."""
    generator = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(n_blocks, dim, generator=generator), dim=-1)
    return centers.repeat_interleave(block_size, dim=0)


def build_synthetic_cache(tmp_path, monkeypatch, per_image_features):
    """Writes a real, fully-validated v1 oracle cache (via
    OnlineAffinityOracleCapture, the exact class the online E3 path uses) --
    one image per entry of per_image_features, each with a single
    448x448 window. raw_scores are set so class 0 always wins, matching a
    trivial all-zero annotation, so alpha=0 mIoU is deterministically 100."""

    identity_paths = {}
    for name in (
        "e3_config", "projection_config", "e3_checkpoint",
        "dino_checkpoint", "clip_checkpoint", "dataset_config",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        identity_paths[name] = str(path)
    provenance = {
        "source_git_commit": "a" * 40, "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    monkeypatch.setattr(oracle, "git_provenance", lambda *a, **k: provenance)
    from PIL import Image
    import numpy as np
    annotation = tmp_path / "annotation.png"
    Image.fromarray(np.zeros((448, 448), dtype=np.uint8)).save(annotation)

    cache_dir = tmp_path / "cache"
    count = len(per_image_features)
    capture = oracle.OnlineAffinityOracleCapture(
        output_dir=cache_dir,
        protocol=oracle.OracleProtocol(with_background=False),
        class_names=[f"class-{i}" for i in range(171)],
        source_image_count=count, max_images=count, windows_per_shard=8,
        identity_paths=identity_paths, dino_identity="dinov2_vitb14_reg",
        text_embedding_sha256="d" * 64,
        allow_dirty_source=True, overwrite=False, commands=["synthetic"],
    )
    raw = torch.full((1, 171, 32, 32), -10.0)
    raw[0, 0] = 10.0
    image_ids = []
    for image_index, features in enumerate(per_image_features):
        image_id = f"image_{image_index}.jpg"
        image_ids.append(image_id)
        metadata = {
            "filename": image_id, "ori_filename": image_id,
            "img_shape": (448, 448, 3), "ori_shape": (448, 448, 3),
            "pad_shape": (448, 448, 3), "flip": False,
        }
        capture.begin_image(metadata, dataset_index=image_index, annotation_path=str(annotation))
        capture.set_window(coordinates=(0, 0, 448, 448), grid_indices=(0, 0))
        capture.observe(features.T.reshape(1, 768, 32, 32), raw)
        capture.end_image(
            resized_input_shape=(448, 448),
            reference_prediction=torch.zeros(448, 448, dtype=torch.int64),
        )
    manifest = capture.finalize()
    return cache_dir, manifest, image_ids


def write_synthetic_capture(capture_dir, cache_dir, image_ids, per_image_features, *, limit=None):
    """Writes a feature-capture directory matching load_capture_manifest's
    schema: one shard holding every window's raw fp16 [1024,768] features,
    plus a manifest cross-referencing the real cache built above."""

    capture_dir.mkdir(parents=True)
    shard_dir = capture_dir / "shards"
    shard_dir.mkdir()
    shard_tensor = torch.stack([f.to(torch.float16) for f in per_image_features])
    buffer = io.BytesIO()
    torch.save(shard_tensor, buffer)
    payload = buffer.getvalue()
    (shard_dir / "windows-000000.pt").write_bytes(payload)
    shard_meta = {
        "name": "shards/windows-000000.pt", "bytes": len(payload),
        "sha256": oracle.sha256_bytes(payload),
        "window_start": 0, "window_end": len(per_image_features),
    }
    images = [
        {
            "dataset_index": index, "image_id": image_id,
            "resized_input_shape": [448, 448], "window_start": index, "window_end": index + 1,
            "windows": [{
                "window_index": 0, "coordinates": [0, 0, 448, 448],
                "grid_indices": [0, 0], "global_window_index": index,
            }],
        }
        for index, image_id in enumerate(image_ids)
    ]
    manifest = {
        "format_version": oracle.CAPTURE_FORMAT, "split": "val", "limit": limit, "seed": 42,
        "source_git_commit": "a" * 40, "source_git_dirty": False, "source_git_diff_sha256": None,
        "existing_cache_manifest_sha256": oracle.sha256_file(cache_dir / "manifest.json"),
        "selected_image_count": len(image_ids), "selected_window_count": len(per_image_features),
        "shards": [shard_meta], "total_bytes": len(payload), "images": images,
        "commands": ["synthetic"],
    }
    (capture_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_verify_feature_capture_exact_agreement_when_features_match(tmp_path, monkeypatch):
    per_image_features = [
        clustered_features(64, 16, seed=seed) for seed in (11, 12, 13)
    ]
    cache_dir, _cache_manifest, image_ids = build_synthetic_cache(
        tmp_path, monkeypatch, per_image_features
    )
    capture_dir = tmp_path / "capture"
    write_synthetic_capture(capture_dir, cache_dir, image_ids, per_image_features)

    report = oracle.verify_feature_capture(capture_dir, cache_dir, device="cpu")
    assert report["images_matched"] == 3
    assert report["images_skipped_no_capture"] == 0
    assert report["windows_compared"] == 3
    assert report["window_exact_index_set_match_fraction"] == 1.0
    assert report["edge_match_fraction"] == 1.0
    assert report["edges_total"] == 3 * 1024 * 12
    assert report["max_abs_weight_diff_where_indices_match"] < 1e-3
    assert report["disagreeing_row_count"] == 0
    assert report["gate_passed"] is True


def test_verify_feature_capture_detects_a_perturbed_feature(tmp_path, monkeypatch):
    per_image_features = [clustered_features(64, 16, seed=21)]
    cache_dir, _cache_manifest, image_ids = build_synthetic_cache(
        tmp_path, monkeypatch, per_image_features
    )
    captured = [per_image_features[0].clone()]
    # Replace one patch's captured feature with an unrelated direction: its
    # true (cache-side) 15 block-mates will no longer agree with what gets
    # rebuilt from the corrupted capture for that row.
    captured[0][0] = F.normalize(torch.randn(768), dim=-1)
    capture_dir = tmp_path / "capture"
    write_synthetic_capture(capture_dir, cache_dir, image_ids, captured)

    report = oracle.verify_feature_capture(capture_dir, cache_dir, device="cpu")
    assert report["windows_compared"] == 1
    assert report["window_exact_index_set_match_fraction"] == 0.0
    assert 0.0 < report["edge_match_fraction"] < 1.0
    assert report["disagreeing_row_count"] > 0
    # The corrupted row's block-mates lost a real (cosine==1) neighbour and
    # gained a near-zero one -- not a fp16 tie. Taken over the whole window,
    # this dominates the max tie-gap even though the corrupted row itself
    # only swaps among near-zero candidates.
    assert report["max_tie_affinity_gap"] > 0.3


def test_verify_feature_capture_skips_images_absent_from_capture(tmp_path, monkeypatch):
    per_image_features = [
        clustered_features(64, 16, seed=seed) for seed in (31, 32)
    ]
    cache_dir, _cache_manifest, image_ids = build_synthetic_cache(
        tmp_path, monkeypatch, per_image_features
    )
    capture_dir = tmp_path / "capture"
    write_synthetic_capture(
        capture_dir, cache_dir, image_ids[:1], per_image_features[:1], limit=1,
    )
    report = oracle.verify_feature_capture(capture_dir, cache_dir, device="cpu")
    assert report["images_matched"] == 1
    assert report["images_skipped_no_capture"] == 1
    assert report["gate_passed"] is True


def test_verify_feature_capture_rejects_stale_cache_reference(tmp_path, monkeypatch):
    per_image_features = [clustered_features(64, 16, seed=41)]
    cache_dir, _cache_manifest, image_ids = build_synthetic_cache(
        tmp_path, monkeypatch, per_image_features
    )
    capture_dir = tmp_path / "capture"
    manifest = write_synthetic_capture(capture_dir, cache_dir, image_ids, per_image_features)
    mutated = dict(manifest, existing_cache_manifest_sha256="0" * 64)
    (capture_dir / "manifest.json").write_text(json.dumps(mutated))

    import argparse
    import run_e3_affinity_oracle as runner
    with pytest.raises(oracle.AffinityOracleError, match="cache manifest changed"):
        runner.verify_feature_capture_cli(argparse.Namespace(
            capture_dir=capture_dir, cache=cache_dir, device="cpu",
            output=tmp_path / "verify.json", overwrite=False, max_images=None,
            assert_anchors=False,
        ))


def test_evaluate_with_rebuilt_graph_matches_evaluate_cache_when_features_match(
    tmp_path, monkeypatch,
):
    per_image_features = [
        clustered_features(64, 16, seed=seed) for seed in (51, 52)
    ]
    cache_dir, _cache_manifest, image_ids = build_synthetic_cache(
        tmp_path, monkeypatch, per_image_features
    )
    capture_dir = tmp_path / "capture"
    write_synthetic_capture(capture_dir, cache_dir, image_ids, per_image_features)

    direct = oracle.evaluate_cache(cache_dir, 0.0, device="cpu")
    rebuilt = oracle.evaluate_with_rebuilt_graph(
        capture_dir, cache_dir, 0.0, device="cpu", propagation_steps=10,
    )
    assert rebuilt["evaluated_images"] == direct["evaluated_images"] == 2
    assert rebuilt["images_skipped_no_capture"] == 0
    assert rebuilt["mIoU"] == pytest.approx(direct["mIoU"])
    assert rebuilt["aAcc"] == pytest.approx(direct["aAcc"])
    assert rebuilt["mAcc"] == pytest.approx(direct["mAcc"])
    assert rebuilt["mIoU"] == pytest.approx(100.0)

    direct_mid = oracle.evaluate_cache(cache_dir, 0.5, device="cpu")
    rebuilt_mid = oracle.evaluate_with_rebuilt_graph(
        capture_dir, cache_dir, 0.5, device="cpu", propagation_steps=10,
    )
    assert rebuilt_mid["mIoU"] == pytest.approx(direct_mid["mIoU"])


def test_assert_feature_capture_anchors_requires_complete_coverage(tmp_path, monkeypatch):
    per_image_features = [
        clustered_features(64, 16, seed=seed) for seed in (61, 62)
    ]
    cache_dir, _cache_manifest, image_ids = build_synthetic_cache(
        tmp_path, monkeypatch, per_image_features
    )
    capture_dir = tmp_path / "capture"
    write_synthetic_capture(
        capture_dir, cache_dir, image_ids[:1], per_image_features[:1], limit=1,
    )
    with pytest.raises(oracle.AffinityOracleError, match="does not cover the full cache"):
        oracle.assert_feature_capture_anchors(capture_dir, cache_dir, device="cpu")


def test_load_capture_manifest_rejects_unknown_and_noncanonical_shard_names(tmp_path, monkeypatch):
    per_image_features = [clustered_features(64, 16, seed=71)]
    cache_dir, _cache_manifest, image_ids = build_synthetic_cache(
        tmp_path, monkeypatch, per_image_features
    )
    capture_dir = tmp_path / "capture"
    write_synthetic_capture(capture_dir, cache_dir, image_ids, per_image_features)

    manifest_path = capture_dir / "manifest.json"
    original = manifest_path.read_text()
    mutated = json.loads(original)
    mutated["unknown"] = 1
    manifest_path.write_text(json.dumps(mutated))
    with pytest.raises(oracle.AffinityOracleError, match="closed schema"):
        oracle.load_capture_manifest(capture_dir)
    manifest_path.write_text(original)

    mutated = json.loads(original)
    mutated["shards"][0]["name"] = "shards/windows-000001.pt"
    manifest_path.write_text(json.dumps(mutated))
    with pytest.raises(oracle.AffinityOracleError, match="noncanonical"):
        oracle.load_capture_manifest(capture_dir)
    manifest_path.write_text(original)
