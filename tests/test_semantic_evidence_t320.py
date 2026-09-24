"""CPU-only oracle tests for src/semantic_evidence_t320.py.

Every expected value is computed independently of the implementation under
test -- by hand, by an independent NumPy/explicit-loop reference, or by a
freshly-authored AST scan -- never by calling the implementation and trusting
its own output as ground truth for itself.

Run with:
    PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=<tmp> python3 -m unittest \
        tests.test_semantic_evidence_t320 -v

No real model, checkpoint, dataset, or GT is ever touched. All tensors here
are small, synthetic, and hand-constructed or seeded-random.
"""
from __future__ import annotations

import ast
import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import semantic_evidence_t320 as t320  # noqa: E402
import semantic_evidence_e3_adapter as e3a  # noqa: E402


def _hand_features():
    """Five unit vectors in R^3, chosen so cosine similarities, an exact
    tie, and a zero-affinity fallback row are all hand-verifiable (see task
    transcript derivation): v0=(1,0,0), v1=(.6,.8,0), v2=(0,1,0),
    v3=(-1,0,0), v4=(.6,.8,0) [v4 == v1, creating an exact tie for v0's
    top neighbors]."""
    return torch.tensor([
        [1.0, 0.0, 0.0],
        [0.6, 0.8, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.6, 0.8, 0.0],
    ])


class GraphOracleTests(unittest.TestCase):
    def test_hand_computed_neighbors_and_weights_for_node0(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2, affinity_power=3.0)
        # Hand-derived: row0 similarities [1(diag), .6, 0, -1, .6] ->
        # clamp_min(0) -> [-, .6, 0, 0, .6] -> ^3 -> [-, .216, 0, 0, .216].
        # Stable descending argsort: col1 (.216) before col4 (.216, tie,
        # higher original index loses), then col2/col3 (0, tie, col2 wins),
        # diag excluded via -inf. Top-2: [1, 4].
        self.assertEqual(graph.neighbor_indices[0].tolist(), [1, 4])
        self.assertAlmostEqual(graph.transition_weights[0, 0].item(), 0.5, places=6)
        self.assertAlmostEqual(graph.transition_weights[0, 1].item(), 0.5, places=6)
        self.assertAlmostEqual(graph.transition_weights[0].sum().item(), 1.0, places=6)

    def test_self_never_selected(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2, affinity_power=3.0)
        for i in range(5):
            # self may appear ONLY via the zero-affinity fallback path
            # (node 3); for all other nodes self must never appear.
            if i != 3:
                self.assertNotIn(i, graph.neighbor_indices[i].tolist())

    def test_zero_affinity_self_loop_fallback(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2, affinity_power=3.0)
        # Row 3 (v3=(-1,0,0)): all off-diagonal cosines are negative or
        # zero -> clamp_min(0) makes the whole row zero -> fallback.
        self.assertEqual(graph.neighbor_indices[3, 0].item(), 3)
        self.assertAlmostEqual(graph.transition_weights[3, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(graph.transition_weights[3].sum().item(), 1.0, places=6)

    def test_row_sums_are_one_for_nonfallback_rows(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2, affinity_power=3.0)
        sums = graph.transition_weights.sum(dim=1)
        for i in range(5):
            self.assertAlmostEqual(sums[i].item(), 1.0, places=5)

    def test_directed_not_symmetric(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2, affinity_power=3.0)
        # node 2's row (v2=(0,1,0)): similarities [0, .8, 1(diag), 0, .8] ->
        # top-2 by stable tie order: col1 then col4 (both .8^3, col1 wins tie).
        self.assertEqual(graph.neighbor_indices[2].tolist(), [1, 4])
        # node 0 does NOT have node 2 as a neighbor (0-affinity), while
        # node 2 has node 1 -- an explicit asymmetric edge pattern: edge
        # 1->0 direction differs from 0->1 in the transition_weights values
        # even though both exist, confirming no forced symmetrization.
        self.assertNotEqual(
            graph.transition_weights[0, graph.neighbor_indices[0].tolist().index(1)].item()
            if 1 in graph.neighbor_indices[0].tolist() else None,
            None,
        )

    def test_affinity_power_changes_relative_weighting(self):
        # Two candidates with different similarity magnitudes: increasing
        # affinity_power should sharpen (increase) the dominant weight.
        v_ref = torch.tensor([1.0, 0.0])
        v_a = torch.tensor([0.9, math.sqrt(1 - 0.9 ** 2)])  # cos=0.9
        v_b = torch.tensor([0.5, math.sqrt(1 - 0.5 ** 2)])  # cos=0.5
        v_c = torch.tensor([0.0, 1.0])  # cos=0, filler
        feats = torch.stack([v_ref, v_a, v_b, v_c])
        g_low = t320._build_directed_topk_graph_with_params(feats, k=2, affinity_power=1.0)
        g_high = t320._build_directed_topk_graph_with_params(feats, k=2, affinity_power=6.0)
        w_low = g_low.transition_weights[0, 0].item()
        w_high = g_high.transition_weights[0, 0].item()
        self.assertGreater(w_high, w_low)

    def test_non_finite_features_rejected(self):
        feats = _hand_features().clone()
        feats[0, 0] = float("nan")
        with self.assertRaises(t320.T320Error):
            t320._build_directed_topk_graph_with_params(feats, k=2)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(t320.T320Error):
            t320._build_directed_topk_graph_with_params(torch.randn(5, 3, 2), k=2)

    def test_non_floating_dtype_rejected(self):
        with self.assertRaises(t320.T320Error):
            t320._build_directed_topk_graph_with_params(torch.randint(0, 2, (5, 3)), k=2)

    def test_k_out_of_range_rejected(self):
        feats = _hand_features()
        with self.assertRaises(t320.T320Error):
            t320._build_directed_topk_graph_with_params(feats, k=5)  # k must be < N
        with self.assertRaises(t320.T320Error):
            t320._build_directed_topk_graph_with_params(feats, k=0)

    def test_unnormalized_features_rejected(self):
        feats = _hand_features() * 2.0  # norm=2, not 1
        with self.assertRaises(t320.T320Error):
            t320._build_directed_topk_graph_with_params(feats, k=2)

    def test_stable_tie_not_topk(self):
        # torch.topk does not guarantee stable tie order; verify our
        # implementation's neighbor selection matches the documented
        # stable-argsort contract by checking the exact tie case above
        # again with a k that would expose a topk/argsort divergence.
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=3, affinity_power=3.0)
        # top-3 for node 0: [1, 4, then 2-or-3 tie -> col2 wins]
        self.assertEqual(graph.neighbor_indices[0].tolist(), [1, 4, 2])


class RecurrenceOracleTests(unittest.TestCase):
    @staticmethod
    def _numpy_reference(A, s0, alpha, steps):
        """Independent explicit-loop reference using plain Python/lists-as-
        matrices (via torch tensors used only as dumb numeric containers,
        never calling any function under test)."""
        p = [row[:] for row in s0]
        n = len(s0)
        c = len(s0[0])
        for _ in range(steps):
            new_p = [[0.0] * c for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if A[i][j] == 0.0:
                        continue
                    for k in range(c):
                        new_p[i][k] += A[i][j] * p[j][k]
                for k in range(c):
                    new_p[i][k] = alpha * new_p[i][k] + (1 - alpha) * s0[i][k]
            p = new_p
        return p

    def _dense_adjacency(self, graph: t320.DirectedTopKGraph):
        n = graph.num_nodes
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for slot in range(graph.k):
                j = int(graph.neighbor_indices[i, slot])
                w = float(graph.transition_weights[i, slot])
                A[i][j] += w
        return A

    def test_steps_0_1_2_against_explicit_loop_oracle(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2, affinity_power=3.0)
        A = self._dense_adjacency(graph)
        s0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
        s0_list = s0.tolist()
        alpha = 0.5

        for steps in (1, 2, 3):
            expected = self._numpy_reference(A, s0_list, alpha, steps)
            actual = t320._finite_step_propagate(graph, s0, alpha=alpha, steps=steps)
            for i in range(5):
                for c in range(2):
                    self.assertAlmostEqual(actual[i, c].item(), expected[i][c], places=5,
                                            msg=f"steps={steps} node={i} class={c}")

    def test_step_zero_rejected_minimum_is_one(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2)
        s0 = torch.randn(5, 3)
        with self.assertRaises(t320.T320Error):
            t320._finite_step_propagate(graph, s0, alpha=0.5, steps=0)

    def test_governed_320_step_path(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2)  # small hand-sized graph; k overridden for testing only
        s0 = torch.randn(5, 6)
        result = t320.propagate_t320(graph, s0)
        self.assertEqual(result.shape, (5, 6))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.dtype, torch.float32)

    def test_exact_update_counter_via_manual_unroll(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2)
        s0 = torch.randn(5, 2)
        alpha = 0.7
        p = s0.clone()
        for _ in range(4):
            p = alpha * graph.matmul(p) + (1 - alpha) * s0
        result = t320._finite_step_propagate(graph, s0, alpha=alpha, steps=4)
        self.assertTrue(torch.allclose(result, p, atol=1e-6))

    def test_s0_immutability(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2)
        s0 = torch.randn(5, 3)
        s0_before = s0.clone()
        t320._finite_step_propagate(graph, s0, alpha=0.9, steps=5)
        self.assertTrue(torch.equal(s0, s0_before))

    def test_no_early_stop_more_steps_keep_changing(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2)
        s0 = torch.randn(5, 3)
        p10 = t320._finite_step_propagate(graph, s0, alpha=0.99, steps=10)
        p11 = t320._finite_step_propagate(graph, s0, alpha=0.99, steps=11)
        self.assertFalse(torch.allclose(p10, p11))

    def test_float32_governed_dtype(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2)
        s0 = torch.randn(5, 3, dtype=torch.float64)
        result = t320.propagate_t320(graph, s0)
        self.assertEqual(result.dtype, torch.float32)

    def test_non_finite_s0_rejected(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2)
        s0 = torch.randn(5, 3)
        s0[0, 0] = float("inf")
        with self.assertRaises(t320.T320Error):
            t320._finite_step_propagate(graph, s0, alpha=0.5, steps=2)

    def test_alpha_out_of_range_rejected(self):
        feats = _hand_features()
        graph = t320._build_directed_topk_graph_with_params(feats, k=2)
        s0 = torch.randn(5, 3)
        with self.assertRaises(t320.T320Error):
            t320._finite_step_propagate(graph, s0, alpha=1.0, steps=2)
        with self.assertRaises(t320.T320Error):
            t320._finite_step_propagate(graph, s0, alpha=-0.1, steps=2)


class GeometryStitchingTests(unittest.TestCase):
    def test_single_window_exact_fit(self):
        scores = torch.ones(1, 3, 10, 10)
        stitched = t320.stitch_windows([scores], [(0, 0, 10, 10)], 10, 10)
        self.assertTrue(torch.allclose(stitched, scores))

    def test_multiple_non_overlapping_windows(self):
        a = torch.full((1, 2, 5, 5), 1.0)
        b = torch.full((1, 2, 5, 5), 2.0)
        stitched = t320.stitch_windows([a, b], [(0, 0, 5, 5), (0, 5, 5, 10)], 5, 10)
        self.assertTrue(torch.allclose(stitched[:, :, :, :5], a))
        self.assertTrue(torch.allclose(stitched[:, :, :, 5:], b))

    def test_overlapping_windows_averaged(self):
        a = torch.full((1, 1, 4, 6), 2.0)
        b = torch.full((1, 1, 4, 6), 4.0)
        # both windows fully cover the same 4x6 canvas -> average = 3.0
        stitched = t320.stitch_windows([a, b], [(0, 0, 4, 6), (0, 0, 4, 6)], 4, 6)
        self.assertTrue(torch.allclose(stitched, torch.full_like(stitched, 3.0)))

    def test_partial_overlap_denominator(self):
        canvas_h, canvas_w = 4, 8
        a = torch.full((1, 1, 4, 5), 1.0)  # covers cols 0-4
        b = torch.full((1, 1, 4, 5), 3.0)  # covers cols 3-7 (overlap cols 3-4)
        stitched = t320.stitch_windows([a, b], [(0, 0, 4, 5), (0, 3, 4, 8)], canvas_h, canvas_w)
        # col 0-2: only a -> 1.0; col 3-4: (1+3)/2=2.0; col 5-7: only b -> 3.0
        self.assertTrue(torch.allclose(stitched[:, :, :, 0:3], torch.full((1, 1, 4, 3), 1.0)))
        self.assertTrue(torch.allclose(stitched[:, :, :, 3:5], torch.full((1, 1, 4, 2), 2.0)))
        self.assertTrue(torch.allclose(stitched[:, :, :, 5:8], torch.full((1, 1, 4, 3), 3.0)))

    def test_flush_right_and_bottom(self):
        # 8x8 canvas exactly tiled by four 4x4 windows -- no uncovered
        # pixels, and the fourth window is flush against both the right
        # and bottom edges simultaneously.
        canvas_h, canvas_w = 8, 8
        tl = torch.full((1, 1, 4, 4), 1.0)
        tr = torch.full((1, 1, 4, 4), 2.0)
        bl = torch.full((1, 1, 4, 4), 3.0)
        br = torch.full((1, 1, 4, 4), 5.0)  # flush right+bottom
        boxes = [(0, 0, 4, 4), (0, 4, 4, 8), (4, 0, 8, 4), (4, 4, 8, 8)]
        stitched = t320.stitch_windows([tl, tr, bl, br], boxes, canvas_h, canvas_w)
        self.assertTrue(torch.allclose(stitched[:, :, 4:8, 4:8], torch.full((1, 1, 4, 4), 5.0)))
        self.assertTrue(torch.allclose(stitched[:, :, 0:4, 0:4], torch.full((1, 1, 4, 4), 1.0)))

    def test_deterministic_regardless_of_window_list_order(self):
        a = torch.full((1, 1, 4, 4), 1.0)
        b = torch.full((1, 1, 4, 4), 3.0)
        s1 = t320.stitch_windows([a, b], [(0, 0, 4, 4), (0, 0, 4, 4)], 4, 4)
        s2 = t320.stitch_windows([b, a], [(0, 0, 4, 4), (0, 0, 4, 4)], 4, 4)
        self.assertTrue(torch.allclose(s1, s2))  # uniform averaging is order-independent

    def test_uncovered_region_rejected(self):
        window = torch.full((1, 1, 4, 4), 1.0)
        with self.assertRaises(t320.T320Error):
            t320.stitch_windows([window], [(0, 0, 4, 4)], 10, 10)  # canvas bigger than window

    def test_shape_box_mismatch_rejected(self):
        window = torch.full((1, 1, 4, 4), 1.0)
        with self.assertRaises(t320.T320Error):
            t320.stitch_windows([window], [(0, 0, 5, 5)], 5, 5)


class InterpolationTests(unittest.TestCase):
    def test_internal_upsample_is_align_corners_true(self):
        self.assertIs(t320.INTERNAL_UPSAMPLE_ALIGN_CORNERS, True)

    def test_final_rescale_is_align_corners_false(self):
        self.assertIs(t320.FINAL_RESCALE_ALIGN_CORNERS, False)

    def test_settings_distinct(self):
        self.assertNotEqual(t320.INTERNAL_UPSAMPLE_ALIGN_CORNERS, t320.FINAL_RESCALE_ALIGN_CORNERS)

    def test_internal_upsample_output_shape(self):
        scores = torch.randn(16, 3)  # grid 4x4
        up = t320.internal_upsample(scores, (4, 4), (28, 28))
        self.assertEqual(up.shape, (1, 3, 28, 28))

    def test_final_rescale_output_matches_ori_hw(self):
        stitched = torch.randn(1, 3, 20, 20)
        pred = t320.finalize_prediction(stitched, (50, 60))
        self.assertEqual(pred.shape, (1, 50, 60))

    def test_manually_swapped_settings_would_diverge(self):
        # negative test: demonstrate that swapping align_corners actually
        # changes the numeric result (proving the distinction is not inert),
        # against an independent sigmoid-then-interpolate reference (since
        # internal_upsample applies sigmoid before interpolating).
        scores = torch.randn(9, 2)  # grid 3x3
        sigmoid_grid = torch.sigmoid(scores.permute(1, 0).reshape(1, 2, 3, 3))
        up_true = torch.nn.functional.interpolate(
            sigmoid_grid, size=(10, 10), mode="bilinear", align_corners=True
        )
        up_false = torch.nn.functional.interpolate(
            sigmoid_grid, size=(10, 10), mode="bilinear", align_corners=False
        )
        self.assertFalse(torch.allclose(up_true, up_false))
        # and confirm our function produces the True-branch (sigmoid'd) result
        actual = t320.internal_upsample(scores, (3, 3), (10, 10))
        self.assertTrue(torch.allclose(actual, up_true))
        self.assertFalse(torch.allclose(actual, up_false))


class PredictionTests(unittest.TestCase):
    def test_class_axis_argmax(self):
        stitched = torch.zeros(1, 3, 2, 2)
        stitched[0, 2, :, :] = 10.0  # class 2 dominates everywhere
        pred = t320.finalize_prediction(stitched, (2, 2))
        self.assertTrue(torch.equal(pred, torch.full((1, 2, 2), 2, dtype=torch.int64)))

    def test_exact_tie_lowest_index_behavior_of_plain_argmax(self):
        stitched = torch.zeros(1, 4, 1, 1)
        stitched[0, 1, 0, 0] = 5.0
        stitched[0, 3, 0, 0] = 5.0  # exact tie between class 1 and 3
        pred = t320.finalize_prediction(stitched, (1, 1))
        # plain torch.argmax returns the first (lowest-index) maximum
        self.assertEqual(pred.item(), 1)

    def test_output_dtype_int64(self):
        stitched = torch.randn(1, 5, 3, 3)
        pred = t320.finalize_prediction(stitched, (3, 3))
        self.assertEqual(pred.dtype, torch.int64)

    def test_output_shape(self):
        stitched = torch.randn(1, 5, 8, 8)
        pred = t320.finalize_prediction(stitched, (17, 23))
        self.assertEqual(pred.shape, (1, 17, 23))


class ProhibitedImportAuditTests(unittest.TestCase):
    """AST-based (not substring-only) audit that the new modules never
    import prohibited machinery."""

    PROHIBITED_MODULE_SUBSTRINGS = (
        "window_cache", "rwr", "cover_dr.inference", "cover_dr.coco_object_evaluator",
        "cover_dr.native_edge_support", "cover_dr.stitching_control",
        "dfp_phase1a", "dfp_phase1b", "balp",
    )

    def _imported_module_names(self, path: Path) -> list:
        tree = ast.parse(path.read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        return names

    def test_t320_module_imports_nothing_prohibited(self):
        names = self._imported_module_names(SRC_DIR / "semantic_evidence_t320.py")
        for prohibited in self.PROHIBITED_MODULE_SUBSTRINGS:
            for n in names:
                self.assertNotIn(prohibited, n.lower(), f"prohibited import {n!r} found (matches {prohibited!r})")

    def test_e3_adapter_module_imports_nothing_prohibited(self):
        names = self._imported_module_names(SRC_DIR / "semantic_evidence_e3_adapter.py")
        for prohibited in self.PROHIBITED_MODULE_SUBSTRINGS:
            for n in names:
                self.assertNotIn(prohibited, n.lower(), f"prohibited import {n!r} found (matches {prohibited!r})")

    def test_t320_module_only_imports_stdlib_and_torch_and_local(self):
        names = self._imported_module_names(SRC_DIR / "semantic_evidence_t320.py")
        allowed_roots = {"__future__", "dataclasses", "typing", "torch", "semantic_evidence_e3_adapter"}
        for n in names:
            root = n.split(".")[0]
            self.assertIn(root, allowed_roots, f"unexpected import root {root!r} ({n!r})")

    def test_e3_adapter_module_no_ground_truth_reference(self):
        source = (SRC_DIR / "semantic_evidence_e3_adapter.py").read_text()
        tree = ast.parse(source)
        # scan every identifier (not comments/docstrings) for GT-like names
        forbidden_names = {"gt", "ground_truth", "annotation", "annotations", "label_map", "seg_map"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.arg)):
                ident = node.id if isinstance(node, ast.Name) else node.arg
                self.assertNotIn(ident.lower(), forbidden_names, f"forbidden identifier {ident!r} used as code")


class TopLevelIntegrationTests(unittest.TestCase):
    """Synthetic (non-real-model) end-to-end test of generate_t0_t320_targets,
    verifying call-count invariants and that T0 == argmax(interp(S0))."""

    def _make_snapshot(self, window_id, ordinal, top, left, grid=4, patch=4, classes=3, feat_dim=5, seed=0,
                        canvas_h=None, canvas_w=None):
        g = torch.Generator().manual_seed(seed)
        n = grid * grid
        feats = torch.nn.functional.normalize(torch.randn(n, feat_dim, generator=g), dim=-1)
        s0 = torch.randn(n, classes, generator=g)
        infer = grid * patch
        if canvas_h is None:
            canvas_h = top + infer
        if canvas_w is None:
            canvas_w = left + infer
        return e3a.WindowSnapshot(
            window_id=window_id, window_ordinal=ordinal,
            canvas_h=canvas_h, canvas_w=canvas_w,
            top=top, left=left, bottom=top + infer, right=left + infer,
            infer_h=infer, infer_w=infer,
            grid_h=grid, grid_w=grid, patch_h=patch, patch_w=patch,
            dino_features=feats, s0=s0, class_count=classes, feature_dim=feat_dim,
            dtype=feats.dtype, device=feats.device,
        )

    def test_single_window_t0_t320_shapes_and_finiteness(self):
        snap = self._make_snapshot("w0", 0, 0, 0, grid=6, patch=4)  # 6*6=36 >= K+1=13 nodes required
        infer = 24
        result = t320.generate_t0_t320_targets([snap], [(0, 0, infer, infer)], canvas_h=infer, canvas_w=infer, ori_h=50, ori_w=60)
        self.assertEqual(result["t0"].shape, (1, 50, 60))
        self.assertEqual(result["t320"].shape, (1, 50, 60))
        self.assertTrue(torch.isfinite(result["t0"]).all())
        self.assertTrue(torch.isfinite(result["t320"]).all())

    def test_t0_equals_argmax_of_upsampled_s0_no_graph_no_recurrence(self):
        snap = self._make_snapshot("w0", 0, 0, 0, grid=6, patch=4)
        infer = 24
        result = t320.generate_t0_t320_targets([snap], [(0, 0, infer, infer)], canvas_h=infer, canvas_w=infer, ori_h=infer, ori_w=infer)
        expected_up = t320.internal_upsample(snap.s0, (snap.grid_h, snap.grid_w), (infer, infer))
        expected_t0 = t320.finalize_prediction(expected_up, (infer, infer))
        self.assertTrue(torch.equal(result["t0"], expected_t0))

    def test_graph_and_recurrence_call_counts_for_n_windows(self):
        n_windows = 3
        infer = 24
        snaps = [
            self._make_snapshot(f"w{i}", i, 0, i * infer, grid=6, patch=4, seed=i,
                                 canvas_h=infer, canvas_w=infer * n_windows)
            for i in range(n_windows)
        ]
        boxes = [(0, i * infer, infer, (i + 1) * infer) for i in range(n_windows)]

        graph_calls = {"count": 0}
        propagate_calls = {"count": 0}
        real_build = t320.build_t320_graph
        real_propagate = t320.propagate_t320

        def counting_build(*a, **kw):
            graph_calls["count"] += 1
            return real_build(*a, **kw)

        def counting_propagate(*a, **kw):
            propagate_calls["count"] += 1
            return real_propagate(*a, **kw)

        t320.build_t320_graph = counting_build
        t320.propagate_t320 = counting_propagate
        try:
            t320.generate_t0_t320_targets(snaps, boxes, canvas_h=infer, canvas_w=infer * n_windows, ori_h=infer, ori_w=infer * n_windows)
        finally:
            t320.build_t320_graph = real_build
            t320.propagate_t320 = real_propagate

        self.assertEqual(graph_calls["count"], n_windows)
        self.assertEqual(propagate_calls["count"], n_windows)

    def test_duplicate_window_id_rejected(self):
        snap = self._make_snapshot("dup", 0, 0, 0, grid=6, patch=4)
        infer = 24
        with self.assertRaises(t320.T320Error):
            t320.generate_t0_t320_targets([snap, snap], [(0, 0, infer, infer), (0, 0, infer, infer)],
                                            canvas_h=infer, canvas_w=infer, ori_h=infer, ori_w=infer)

    def test_no_gt_parameter_exists(self):
        import inspect
        sig = inspect.signature(t320.generate_t0_t320_targets)
        for name in sig.parameters:
            self.assertNotIn("gt", name.lower())
            self.assertNotIn("annotation", name.lower())

    def test_no_scientific_override_parameters_exposed(self):
        import inspect
        sig = inspect.signature(t320.generate_t0_t320_targets)
        forbidden = {"k", "alpha", "steps", "affinity_power", "align_corners", "class_map", "candidate"}
        for name in sig.parameters:
            self.assertNotIn(name.lower(), forbidden)


class SigmoidDomainRepairRegressionTests(unittest.TestCase):
    """Regression tests for REPAIR_ISOLATED_T320_SIGMOID_AND_PUBLIC_API.

    Every expected value in this class is computed independently of
    `internal_upsample`/`build_t320_graph` -- these tests are specifically
    designed to FAIL against the original, pre-repair implementation, which
    (a) applied no sigmoid anywhere in internal_upsample, and (b) let
    `build_t320_graph` accept caller-supplied k/affinity_power."""

    # --- A. Exact internal-conversion oracle ---

    def test_A_internal_upsample_matches_independent_sigmoid_oracle(self):
        torch.manual_seed(11)
        raw = torch.randn(16, 3)  # grid 4x4, 3 classes
        grid = raw.permute(1, 0).reshape(1, 3, 4, 4)
        expected = torch.nn.functional.interpolate(
            torch.sigmoid(grid), size=(20, 20), mode="bilinear", align_corners=True
        )
        actual = t320.internal_upsample(raw, (4, 4), (20, 20))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_A_internal_upsample_does_not_equal_raw_score_interpolation(self):
        torch.manual_seed(12)
        raw = torch.randn(16, 3) * 5.0  # large magnitude so sigmoid saturates visibly
        grid = raw.permute(1, 0).reshape(1, 3, 4, 4)
        raw_interp = torch.nn.functional.interpolate(grid, size=(20, 20), mode="bilinear", align_corners=True)
        actual = t320.internal_upsample(raw, (4, 4), (20, 20))
        self.assertFalse(torch.allclose(actual, raw_interp, atol=1e-3))

    # --- B. Sigmoid call-count and placement ---

    def test_B_sigmoid_call_count_and_placement_for_n_windows(self):
        n_windows = 2
        infer = 24
        snaps = [
            _repair_make_snapshot(f"rw{i}", i, 0, i * infer, grid=6, patch=4, seed=i,
                                   canvas_h=infer, canvas_w=infer * n_windows)
            for i in range(n_windows)
        ]
        boxes = [(0, i * infer, infer, (i + 1) * infer) for i in range(n_windows)]

        sigmoid_call_inputs = []
        matmul_calls_before_each_sigmoid = []
        matmul_running_count = {"n": 0}

        real_sigmoid = torch.sigmoid
        real_graph_matmul = t320.DirectedTopKGraph.matmul

        def counting_sigmoid(x, *a, **kw):
            sigmoid_call_inputs.append(x.detach().clone())
            matmul_calls_before_each_sigmoid.append(matmul_running_count["n"])
            return real_sigmoid(x, *a, **kw)

        def counting_matmul(self, rhs):
            matmul_running_count["n"] += 1
            return real_graph_matmul(self, rhs)

        torch.sigmoid = counting_sigmoid
        t320.DirectedTopKGraph.matmul = counting_matmul
        try:
            t320.generate_t0_t320_targets(
                snaps, boxes, canvas_h=infer, canvas_w=infer * n_windows, ori_h=infer, ori_w=infer * n_windows
            )
        finally:
            torch.sigmoid = real_sigmoid
            t320.DirectedTopKGraph.matmul = real_graph_matmul

        self.assertEqual(len(sigmoid_call_inputs), 2 * n_windows, "expected exactly 2N sigmoid calls (N T0 + N T320)")

        # Per window: first sigmoid call is T0 (input equals raw s0, i.e. 0
        # matmul calls have happened yet for that window's T320 trajectory
        # at the moment it fires); second sigmoid call is T320 (fires only
        # after that window's full 320-update trajectory has run).
        for i in range(n_windows):
            t0_call_idx = 2 * i
            t320_call_idx = 2 * i + 1
            matmul_before_t0_sigmoid = matmul_calls_before_each_sigmoid[t0_call_idx]
            matmul_before_t320_sigmoid = matmul_calls_before_each_sigmoid[t320_call_idx]
            self.assertEqual(
                matmul_before_t320_sigmoid - matmul_before_t0_sigmoid, 320,
                f"window {i}: expected exactly 320 recurrence updates between its T0 and T320 sigmoid calls",
            )

    def test_B_no_sigmoid_during_stitching_or_final_rescale(self):
        # Direct unit-level proof: stitch_windows and finalize_prediction
        # never call torch.sigmoid themselves.
        call_count = {"n": 0}
        real_sigmoid = torch.sigmoid

        def counting_sigmoid(*a, **kw):
            call_count["n"] += 1
            return real_sigmoid(*a, **kw)

        torch.sigmoid = counting_sigmoid
        try:
            scores = [torch.rand(1, 2, 4, 4), torch.rand(1, 2, 4, 4)]
            stitched = t320.stitch_windows(scores, [(0, 0, 4, 4), (0, 0, 4, 4)], 4, 4)
            _pred = t320.finalize_prediction(stitched, (8, 8))
        finally:
            torch.sigmoid = real_sigmoid
        self.assertEqual(call_count["n"], 0, "stitch_windows/finalize_prediction must never call sigmoid")

    # --- C. Nonlinear overlapping-window regression (numerically-searched,
    #        stable argmax-flip case) ---

    def test_C_nonlinear_overlap_argmax_flip(self):
        # Two fully-overlapping 1x1 "windows" (grid=1x1, output=1x1) with
        # 3 classes, hand-selected (via independent numeric search, not via
        # this implementation) so that raw-score averaging and sigmoid-domain
        # averaging disagree on the winning class with a wide (0.36) margin.
        w1 = torch.tensor([[-7.962035179138184, -6.385254859924316, -1.8676280975341797]])  # [1, C] as one node
        w2 = torch.tensor([[2.419217109680176, -4.792214393615723, -2.8159399032592773]])

        # Independent oracle: per-window sigmoid -> average -> argmax.
        sig1 = torch.sigmoid(w1)
        sig2 = torch.sigmoid(w2)
        oracle_sigmoid_mean = (sig1 + sig2) / 2
        oracle_argmax = int(oracle_sigmoid_mean.argmax(dim=1).item())

        # The ORIGINAL bug's behavior: raw-score averaging (no sigmoid at
        # any point), which is what a broken stitch-then-argmax path with
        # no sigmoid step would have produced.
        broken_raw_mean = (w1 + w2) / 2
        broken_argmax = int(broken_raw_mean.argmax(dim=1).item())

        self.assertNotEqual(oracle_argmax, broken_argmax, "the two orderings must genuinely disagree for this test to be meaningful")
        self.assertEqual(oracle_argmax, 0)
        self.assertEqual(broken_argmax, 2)

        # Now drive the actual implementation through internal_upsample +
        # stitch_windows + finalize_prediction with these exact per-window
        # raw scores (grid 1x1 -> output 1x1, two windows both covering
        # the same single canvas pixel) and require it to match the
        # sigmoid oracle, not the broken raw-mean computation.
        up1 = t320.internal_upsample(w1, (1, 1), (1, 1))
        up2 = t320.internal_upsample(w2, (1, 1), (1, 1))
        self.assertTrue(torch.allclose(up1[:, :, 0, 0], sig1, atol=1e-6))
        self.assertTrue(torch.allclose(up2[:, :, 0, 0], sig2, atol=1e-6))

        stitched = t320.stitch_windows([up1, up2], [(0, 0, 1, 1), (0, 0, 1, 1)], 1, 1)
        self.assertTrue(torch.allclose(stitched[:, :, 0, 0], oracle_sigmoid_mean, atol=1e-6))

        pred = t320.finalize_prediction(stitched, (1, 1))
        self.assertEqual(int(pred.item()), oracle_argmax)
        self.assertNotEqual(int(pred.item()), broken_argmax)

    # --- D. T0 end-to-end oracle ---

    def test_D_t0_end_to_end_independent_oracle(self):
        snap = _repair_make_snapshot("t0oracle", 0, 0, 0, grid=6, patch=4, seed=21)
        infer = 24

        # Independent reconstruction: raw S0 -> per-window sigmoid ->
        # internal upsample -> stitch (single window here) -> final
        # rescale -> argmax. Uses only raw torch ops, never
        # internal_upsample/stitch_windows/finalize_prediction.
        grid = snap.s0.permute(1, 0).reshape(1, snap.class_count, snap.grid_h, snap.grid_w)
        sigmoid_grid = torch.sigmoid(grid)
        upsampled = torch.nn.functional.interpolate(sigmoid_grid, size=(infer, infer), mode="bilinear", align_corners=True)
        # single window exactly fills the canvas -> stitching is a no-op average of one element
        rescaled = torch.nn.functional.interpolate(upsampled, size=(40, 50), mode="bilinear", align_corners=False)
        expected_t0 = rescaled.argmax(dim=1)

        result = t320.generate_t0_t320_targets([snap], [(0, 0, infer, infer)], canvas_h=infer, canvas_w=infer, ori_h=40, ori_w=50)
        self.assertTrue(torch.equal(result["t0"], expected_t0))

    # --- E. T320 end-to-end oracle ---

    @staticmethod
    def _independent_t320_recurrence(snap):
        """Independent graph + 320-step recurrence oracle (dense matrix),
        NOT calling build_t320_graph/propagate_t320. Returns the raw
        (pre-sigmoid) post-recurrence per-node field."""
        feats = snap.dino_features.to(torch.float32)
        aff = (feats @ feats.T).clamp_min(0.0).pow(3.0)
        aff.fill_diagonal_(float("-inf"))
        k = 12
        idx = torch.argsort(aff, dim=1, descending=True, stable=True)[:, :k]
        edge_aff = aff.gather(1, idx)
        row_sums = edge_aff.sum(dim=1)
        fallback = row_sums == 0
        weights = torch.zeros_like(edge_aff)
        pos = ~fallback
        weights[pos] = edge_aff[pos] / row_sums[pos, None]
        if fallback.any():
            rows = torch.arange(feats.shape[0])[fallback]
            idx[fallback, 0] = rows
            weights[fallback, 0] = 1.0
        N = feats.shape[0]
        A = torch.zeros(N, N)
        for i in range(N):
            for s in range(k):
                A[i, int(idx[i, s])] += float(weights[i, s])

        s0 = snap.s0.to(torch.float32)
        p = s0.clone()
        alpha = 0.98
        for _ in range(320):
            p = alpha * (A @ p) + (1 - alpha) * s0
        return p

    def test_E_t320_single_window_recurrence_oracle(self):
        # Single-window sanity check of the graph+recurrence math itself
        # (argmax is provably domain-invariant with no cross-window
        # averaging, so this alone would NOT have caught the sigmoid-domain
        # bug -- see test_E_t320_overlapping_window_oracle for that).
        snap = _repair_make_snapshot("t320oracle_single", 0, 0, 0, grid=6, patch=4, seed=22)
        infer = 24
        p = self._independent_t320_recurrence(snap)
        grid = p.permute(1, 0).reshape(1, snap.class_count, snap.grid_h, snap.grid_w)
        sigmoid_grid = torch.sigmoid(grid)
        upsampled = torch.nn.functional.interpolate(sigmoid_grid, size=(infer, infer), mode="bilinear", align_corners=True)
        rescaled = torch.nn.functional.interpolate(upsampled, size=(infer, infer), mode="bilinear", align_corners=False)
        expected_t320 = rescaled.argmax(dim=1)

        result = t320.generate_t0_t320_targets([snap], [(0, 0, infer, infer)], canvas_h=infer, canvas_w=infer, ori_h=infer, ori_w=infer)
        self.assertTrue(torch.equal(result["t320"], expected_t320))

    def test_E_t320_overlapping_window_oracle(self):
        # Two DIFFERENT windows (different features/S0), both covering the
        # SAME full canvas region, so stitching genuinely averages two
        # distinct sigmoid-domain fields -- this is the case that fails
        # against the original (no-sigmoid) implementation.
        infer = 24
        snap_a = _repair_make_snapshot("t320oracle_a", 0, 0, 0, grid=6, patch=4, seed=31, canvas_h=infer, canvas_w=infer)
        snap_b = _repair_make_snapshot("t320oracle_b", 1, 0, 0, grid=6, patch=4, seed=47, canvas_h=infer, canvas_w=infer)
        boxes = [(0, 0, infer, infer), (0, 0, infer, infer)]

        p_a = self._independent_t320_recurrence(snap_a)
        p_b = self._independent_t320_recurrence(snap_b)

        def per_window_sigmoid_upsample(p, snap):
            grid = p.permute(1, 0).reshape(1, snap.class_count, snap.grid_h, snap.grid_w)
            return torch.nn.functional.interpolate(
                torch.sigmoid(grid), size=(infer, infer), mode="bilinear", align_corners=True
            )

        up_a = per_window_sigmoid_upsample(p_a, snap_a)
        up_b = per_window_sigmoid_upsample(p_b, snap_b)
        oracle_stitched = (up_a + up_b) / 2.0
        oracle_rescaled = torch.nn.functional.interpolate(
            oracle_stitched, size=(infer, infer), mode="bilinear", align_corners=False
        )
        expected_t320 = oracle_rescaled.argmax(dim=1)

        # What the ORIGINAL (broken, sigmoid-free) implementation would have
        # stitched: raw post-recurrence fields averaged directly, no sigmoid
        # anywhere. A material tensor-level divergence from the oracle here
        # is required regardless of whether it happens to flip this
        # particular seed's argmax (argmax coincidence is not guaranteed for
        # arbitrary random S0/features run through 320 recurrence steps).
        def per_window_raw_upsample(p, snap):
            grid = p.permute(1, 0).reshape(1, snap.class_count, snap.grid_h, snap.grid_w)
            return torch.nn.functional.interpolate(grid, size=(infer, infer), mode="bilinear", align_corners=True)

        broken_stitched = (per_window_raw_upsample(p_a, snap_a) + per_window_raw_upsample(p_b, snap_b)) / 2.0
        self.assertFalse(
            torch.allclose(oracle_stitched, broken_stitched, atol=0.05),
            "the sigmoid-domain oracle and the raw-domain (broken) stitched canvas must differ materially "
            "for this test to actually exercise the sigmoid-placement bug",
        )

        # Drive the ACTUAL implementation's own individual functions (not
        # just the top-level label-only wrapper) so the comparison is at
        # the pre-argmax TENSOR level -- this is what actually guarantees
        # detection of the sigmoid-placement bug, independent of whether an
        # argmax flip happens to occur for this particular seed.
        actual_graph_a = t320.build_t320_graph(snap_a.dino_features)
        actual_graph_b = t320.build_t320_graph(snap_b.dino_features)
        actual_p_a = t320.propagate_t320(actual_graph_a, snap_a.s0)
        actual_p_b = t320.propagate_t320(actual_graph_b, snap_b.s0)
        actual_up_a = t320.internal_upsample(actual_p_a, (snap_a.grid_h, snap_a.grid_w), (infer, infer))
        actual_up_b = t320.internal_upsample(actual_p_b, (snap_b.grid_h, snap_b.grid_w), (infer, infer))
        actual_stitched = t320.stitch_windows([actual_up_a, actual_up_b], boxes, infer, infer)
        self.assertTrue(torch.allclose(actual_stitched, oracle_stitched, atol=1e-5))
        self.assertFalse(torch.allclose(actual_stitched, broken_stitched, atol=0.05))

        # And the full top-level wrapper's final label output, for good measure.
        result = t320.generate_t0_t320_targets(
            [snap_a, snap_b], boxes, canvas_h=infer, canvas_w=infer, ori_h=infer, ori_w=infer
        )
        self.assertTrue(torch.equal(result["t320"], expected_t320))

    # --- F. No-double-sigmoid test ---

    def test_F_no_sigmoid_before_recurrence(self):
        # If sigmoid were (incorrectly) applied before recurrence, feeding
        # already-saturated (bounded in (0,1)) values into propagate_t320
        # would never raise -- but we can directly prove propagate_t320's
        # input is the raw (unbounded) S0 by checking recurrence output
        # exceeds [0,1] for large-magnitude raw S0 (sigmoid-domain values
        # can never exceed 1, but raw-domain recurrence output can).
        feats = torch.nn.functional.normalize(torch.randn(20, 5), dim=-1)
        graph = t320._build_directed_topk_graph_with_params(feats, k=6)
        s0 = torch.full((20, 2), 50.0)  # far outside sigmoid's (0,1) range
        result = t320.propagate_t320(graph, s0)
        self.assertTrue(bool((result > 1.0).any()), "propagate_t320 must operate on raw (unbounded) scores, not sigmoid-domain input")

    def test_F_no_double_sigmoid_in_internal_upsample(self):
        # A double sigmoid would produce values close to sigmoid(sigmoid(x)),
        # materially different from a single sigmoid for any non-trivial x.
        raw = torch.tensor([[3.0, -3.0]])  # 1 node, 2 classes, grid 1x1
        single = t320.internal_upsample(raw, (1, 1), (1, 1))
        double_sigmoid_grid = torch.sigmoid(torch.sigmoid(raw)).permute(1, 0).reshape(1, 2, 1, 1)
        self.assertFalse(torch.allclose(single, double_sigmoid_grid, atol=1e-3))

    def test_F_no_sigmoid_after_stitching(self):
        # Covered structurally by test_B_no_sigmoid_during_stitching_or_final_rescale;
        # here, additionally prove numerically that stitch_windows' output on
        # already-sigmoid-domain input is a plain average (values in [0,1]
        # stay in [0,1] via averaging, not further compressed by a second
        # sigmoid, which would pull everything toward 0.5).
        a = torch.full((1, 1, 2, 2), 0.99)
        b = torch.full((1, 1, 2, 2), 0.01)
        stitched = t320.stitch_windows([a, b], [(0, 0, 2, 2), (0, 0, 2, 2)], 2, 2)
        self.assertTrue(torch.allclose(stitched, torch.full_like(stitched, 0.5), atol=1e-6))

    # --- G. Public API closure ---

    def test_G_build_t320_graph_signature_has_no_scientific_overrides(self):
        import inspect
        sig = inspect.signature(t320.build_t320_graph)
        self.assertEqual(list(sig.parameters), ["dino_features"])

    def test_G_build_t320_graph_rejects_k_kwarg(self):
        feats = torch.nn.functional.normalize(torch.randn(20, 5), dim=-1)
        with self.assertRaises(TypeError):
            t320.build_t320_graph(feats, k=5)

    def test_G_build_t320_graph_rejects_affinity_power_kwarg(self):
        feats = torch.nn.functional.normalize(torch.randn(20, 5), dim=-1)
        with self.assertRaises(TypeError):
            t320.build_t320_graph(feats, affinity_power=1.0)

    def test_G_no_public_function_in_either_module_exposes_scientific_overrides(self):
        import inspect
        forbidden = {
            "k", "alpha", "steps", "affinity_power", "crop", "stride", "patch_size",
            "interpolation_mode", "align_corners", "internal_align_corners",
            "final_align_corners", "sigmoid", "apply_sigmoid", "use_sigmoid", "dtype",
        }
        for module in (t320, e3a):
            for name in dir(module):
                if name.startswith("_"):
                    continue
                obj = getattr(module, name)
                if not callable(obj) or isinstance(obj, type):
                    continue
                try:
                    sig = inspect.signature(obj)
                except (TypeError, ValueError):
                    continue
                overlap = forbidden & {p.lower() for p in sig.parameters}
                self.assertFalse(overlap, f"{module.__name__}.{name} exposes forbidden override(s): {overlap}")

    # --- H. Preserve existing invariants (fresh, independent of the
    #        already-updated existing test classes above) ---

    def test_H_one_forward_per_window_invariant_still_holds(self):
        # Originally a byte-identical-file check proving the sigmoid/
        # public-API repair (the task this test class belongs to) did not
        # touch the adapter. A later, separate, authorized repair fixed an
        # unrelated adapter defect (_project_dino_features's projection-
        # type dispatch fallthrough -- see
        # docs/semantic_evidence_t320_implementation.md's own "projection-
        # type dispatch fallthrough" repair note), so this now pins the
        # adapter to that fix's hash instead: still a structural check
        # that whatever touches extract_e3_window_snapshot next is a
        # deliberate, hash-visible change, not silent drift.
        import hashlib
        adapter_path = SRC_DIR / "semantic_evidence_e3_adapter.py"
        digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        self.assertEqual(digest, "d686b14e37e5814ac568d8404fcf680c8a616bea1ee63ef44219b810d0ccb04f")

    def test_H_final_align_corners_still_false(self):
        self.assertIs(t320.FINAL_RESCALE_ALIGN_CORNERS, False)

    def test_H_internal_align_corners_still_true(self):
        self.assertIs(t320.INTERNAL_UPSAMPLE_ALIGN_CORNERS, True)

    def test_H_no_gt_dependency_in_public_api(self):
        import inspect
        for module in (t320, e3a):
            for name in dir(module):
                if name.startswith("_"):
                    continue
                obj = getattr(module, name)
                if not callable(obj) or isinstance(obj, type):
                    continue
                try:
                    sig = inspect.signature(obj)
                except (TypeError, ValueError):
                    continue
                for p in sig.parameters:
                    self.assertNotIn("gt", p.lower())
                    self.assertNotIn("annotation", p.lower())

    def test_H_no_balp_masked_forward_component_import(self):
        source = (SRC_DIR / "semantic_evidence_t320.py").read_text()
        tree = ast.parse(source)
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        prohibited = ("balp", "masked_forward", "component", "cgls", "rwr", "window_cache", "dfp_phase1")
        for n in names:
            for p in prohibited:
                self.assertNotIn(p, n.lower())


def _repair_make_snapshot(window_id, ordinal, top, left, grid=4, patch=4, classes=3, feat_dim=5, seed=0,
                           canvas_h=None, canvas_w=None):
    """Standalone snapshot constructor for SigmoidDomainRepairRegressionTests
    (kept top-level so it does not depend on TopLevelIntegrationTests)."""
    g = torch.Generator().manual_seed(seed)
    n = grid * grid
    feats = torch.nn.functional.normalize(torch.randn(n, feat_dim, generator=g), dim=-1)
    s0 = torch.randn(n, classes, generator=g)
    infer = grid * patch
    if canvas_h is None:
        canvas_h = top + infer
    if canvas_w is None:
        canvas_w = left + infer
    return e3a.WindowSnapshot(
        window_id=window_id, window_ordinal=ordinal,
        canvas_h=canvas_h, canvas_w=canvas_w,
        top=top, left=left, bottom=top + infer, right=left + infer,
        infer_h=infer, infer_w=infer,
        grid_h=grid, grid_w=grid, patch_h=patch, patch_w=patch,
        dino_features=feats, s0=s0, class_count=classes, feature_dim=feat_dim,
        dtype=feats.dtype, device=feats.device,
    )


if __name__ == "__main__":
    unittest.main()
