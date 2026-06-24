import importlib.util
import pathlib
import unittest
from types import SimpleNamespace

import torch


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "src/open_vocabulary_segmentation/segmentation/evaluation/sg_gate.py"
)
SPEC = importlib.util.spec_from_file_location("sg_gate", MODULE_PATH)
sg_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sg_gate)


def gate_config(**overrides):
    values = {
        "top_percent": 25.0,
        "score_anchor_threshold": 0.3,
        "tau_pos_thing": 0.45,
        "tau_ignore_thing": 0.30,
        "tau_pos_stuff": 0.35,
        "tau_ignore_stuff": 0.22,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SGGateTest(unittest.TestCase):
    def test_robust_cluster_norm_low_count_and_constant(self):
        values = torch.tensor([2.0, 4.0, 6.0, 8.0])
        normalized = sg_gate.robust_cluster_norm(values)
        self.assertTrue(torch.allclose(
            normalized,
            torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
            atol=1e-5,
        ))
        self.assertTrue(torch.equal(
            sg_gate.robust_cluster_norm(torch.ones(5)),
            torch.zeros(5),
        ))

    def test_sparse_clustering_covers_grid_and_honors_cap(self):
        features = torch.zeros(16, 2)
        features[:8, 0] = 1
        features[8:, 1] = 1
        cluster_map, clusters = sg_gate.build_structural_clusters(
            features,
            (4, 4),
            tau_edge=0.5,
            max_clusters=4,
            min_cluster_area=2,
        )
        self.assertEqual(tuple(cluster_map.shape), (4, 4))
        self.assertLessEqual(len(clusters), 4)
        self.assertEqual(sum(cluster["area"] for cluster in clusters), 16)
        self.assertGreaterEqual(int(cluster_map.min()), 0)

    def test_structural_gate_returns_trusted_sg_overwrites(self):
        score_maps = torch.tensor([
            [[0.9, 0.8], [0.2, 0.1]],
            [[0.9, 0.8], [0.2, 0.1]],
        ])
        features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        cluster_map = torch.tensor([[0, 0], [1, 1]])
        positive, ignored = sg_gate.apply_structural_gate(
            score_maps,
            features,
            cluster_map,
            gate_config(tau_pos_thing=-1.0, tau_ignore_thing=-2.0),
        )
        self.assertTrue((positive[0] > 0).any())
        self.assertFalse((positive[1] > 0).any())
        self.assertFalse(ignored.any())

    def test_structural_gate_returns_no_overwrite_when_no_cluster_is_positive(self):
        score_maps = torch.tensor([[[0.9, 0.8], [0.2, 0.1]]])
        features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        cluster_map = torch.tensor([[0, 0], [1, 1]])
        positive, ignored = sg_gate.apply_structural_gate(
            score_maps,
            features,
            cluster_map,
            gate_config(tau_pos_thing=99.0, tau_ignore_thing=-1.0),
        )
        self.assertFalse((positive > 0).any())
        self.assertTrue(ignored.any())

    def test_raw_supported_soft_candidates_require_cluster_topk_support(self):
        score_maps = torch.tensor([
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.9, 0.9], [0.1, 0.1]],
        ])
        features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        cluster_map = torch.tensor([[0, 0], [1, 1]])
        cfg = gate_config(
            top_percent=50.0,
            tau_pos_thing=-1.0,
            tau_ignore_thing=99.0,
            raw_fallback=SimpleNamespace(enabled=False),
            soft_reweight=SimpleNamespace(
                enabled=True,
                candidate_source="raw_supported_clusters",
                min_soft_gate_score=-1.0,
                require_semantic_anchor=False,
                raw_support_topk=1,
                min_cluster_raw_topk_support=0.5,
            ),
        )
        positive, ignored, stats = sg_gate.apply_structural_gate(
            score_maps,
            features,
            cluster_map,
            cfg,
            return_stats=True,
        )
        self.assertTrue((positive[0] > 0).any())
        self.assertFalse((positive[1] > 0).any())
        self.assertEqual(stats["raw_supported_candidate_clusters"], 1)
        self.assertEqual(stats["raw_supported_candidate_classes"], 1)
        self.assertGreater(stats["raw_supported_topk_support_count"], 1)
        self.assertFalse(ignored.any())

    def test_raw_fallback_recovers_too_small_sg_mask(self):
        score_maps = torch.tensor([
            [[0.9, 0.8], [0.7, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
        ])
        features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        cluster_map = torch.tensor([[0, 0], [1, 1]])
        cfg = gate_config(
            top_percent=25.0,
            tau_pos_thing=99.0,
            tau_ignore_thing=99.0,
            raw_fallback=SimpleNamespace(
                enabled=True,
                min_area_ratio=0.75,
                max_area_ratio=2.50,
                min_agreement_iou=0.35,
                min_raw_area_pixels=1,
                fallback_if_sg_empty=True,
                fallback_if_raw_empty_and_no_anchor=True,
            ),
        )
        positive, ignored, stats = sg_gate.apply_structural_gate(
            score_maps,
            features,
            cluster_map,
            cfg,
            return_stats=True,
        )
        self.assertFalse((positive > 0).any())
        self.assertEqual(stats["fallback_sg_empty"], 1)
        self.assertEqual(stats["fallback_used"], 2)
        self.assertFalse(ignored.any())

    def test_raw_fallback_recovers_low_agreement_iou(self):
        score_maps = torch.tensor([
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.1, 0.8], [0.2, 0.9]],
        ])
        features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        cluster_map = torch.tensor([[0, 0], [1, 1]])
        cfg = gate_config(
            top_percent=25.0,
            tau_pos_thing=0.5,
            tau_ignore_thing=99.0,
            raw_fallback=SimpleNamespace(
                enabled=True,
                min_area_ratio=0.60,
                max_area_ratio=1.60,
                min_agreement_iou=0.35,
                min_raw_area_pixels=1,
                fallback_if_sg_empty=True,
                fallback_if_raw_empty_and_no_anchor=True,
            ),
        )
        positive, ignored, stats = sg_gate.apply_structural_gate(
            score_maps,
            features,
            cluster_map,
            cfg,
            return_stats=True,
        )
        self.assertFalse((positive > 0).any())
        self.assertEqual(stats["fallback_low_agreement_iou"], 2)
        self.assertEqual(stats["fallback_used"], 2)
        self.assertEqual(stats["raw_sg_agreement_iou_count"], 2)
        self.assertFalse(ignored.any())

    def test_raw_too_small_blocks_sg_mask_by_default(self):
        score_maps = torch.tensor([
            [[0.9, 0.8], [0.7, 0.6]],
            [[0.4, 0.3], [0.2, 0.1]],
        ])
        features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        cluster_map = torch.tensor([[0, 0], [1, 1]])
        cfg = gate_config(
            top_percent=25.0,
            tau_pos_thing=-1.0,
            tau_ignore_thing=99.0,
            raw_fallback=SimpleNamespace(
                enabled=True,
                min_area_ratio=0.60,
                max_area_ratio=1.60,
                min_agreement_iou=0.35,
                min_raw_area_pixels=1,
                allow_sg_when_raw_empty=False,
                fallback_if_sg_empty=True,
                fallback_if_raw_empty_and_no_anchor=True,
            ),
        )
        positive, ignored, stats = sg_gate.apply_structural_gate(
            score_maps,
            features,
            cluster_map,
            cfg,
            return_stats=True,
        )
        self.assertFalse((positive[1] > 0).any())
        self.assertEqual(stats["raw_too_small_block_sg_count"], 1)
        self.assertEqual(stats["fallback_raw_too_small_block_sg"], 1)
        self.assertFalse(ignored.any())

    def test_alignment_audit_populates_scalar_stats(self):
        score_maps = torch.tensor([
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.1, 0.8], [0.2, 0.9]],
        ])
        features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        cluster_map = torch.tensor([[0, 0], [1, 1]])
        raw_label_img = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ])
        cfg = gate_config(
            top_percent=25.0,
            tau_pos_thing=0.5,
            tau_ignore_thing=99.0,
            raw_fallback=SimpleNamespace(
                enabled=True,
                min_area_ratio=0.60,
                max_area_ratio=1.60,
                min_agreement_iou=0.35,
                min_raw_area_pixels=1,
                allow_sg_when_raw_empty=False,
                fallback_if_sg_empty=True,
                fallback_if_raw_empty_and_no_anchor=True,
            ),
            alignment_audit=SimpleNamespace(
                enabled=True,
                max_logged_examples=0,
                save_debug_images=False,
                save_debug_tensors=False,
            ),
        )
        positive, ignored, stats = sg_gate.apply_structural_gate(
            score_maps,
            features,
            cluster_map,
            cfg,
            return_stats=True,
            raw_label_img=raw_label_img,
            image_size=(4, 4),
        )
        self.assertFalse((positive > 0).any())
        self.assertFalse(ignored.any())
        self.assertEqual(stats["alignment_seed_inside_sg_patch_rate_count"], 2)
        self.assertEqual(len(stats["alignment_sg_roundtrip_iou_values"]), 2)
        self.assertGreater(stats["alignment_num_image_class_mask_decisions"], 0)
        self.assertGreater(len(stats["alignment_raw_sg_iou_pre_fallback_values"]), 0)


if __name__ == "__main__":
    unittest.main()
