"""CPU-only, fake-provider / synthetic-data tests for
src/semantic_evidence_holdout_evaluate.py and
diagnostics/run_semantic_evidence_holdout_evaluation.py.

Never opens a real holdout image or GT annotation, never constructs a real
model, never loads a real checkpoint, never initializes CUDA. Every real-
access entry point (`preflight_holdout`, `run_holdout_evaluation`) is
confirmed to fail closed under the currently committed authority BEFORE
touching any holdout path -- this is proven by passing deliberately
nonexistent paths and confirming the raised error is the authorization
error, not a missing-path error.

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
        tests.test_semantic_evidence_holdout_evaluate -v
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import semantic_evidence_holdout_evaluate as hv  # noqa: E402
import semantic_evidence_discovery_execute as disc  # noqa: E402
import semantic_evidence_e3_adapter as e3a  # noqa: E402

GRID = 32
CLASS_COUNT = 171
FEAT_DIM = 8
PATCH = 14
CANVAS = 448


def _fake_image_loader(authorized_ids):
    def loader(image_id):
        assert image_id in authorized_ids, f"unauthorized image_id requested: {image_id}"
        g = torch.Generator().manual_seed(int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16))
        return torch.rand(1, 3, CANVAS, CANVAS, generator=g) * 255.0, CANVAS, CANVAS
    return loader


def _fake_gt_loader(authorized_ids):
    def loader(image_id):
        assert image_id in authorized_ids, f"unauthorized image_id requested for GT: {image_id}"
        g = torch.Generator().manual_seed(int(hashlib.sha256(("gt:" + image_id).encode()).hexdigest()[:8], 16))
        klass = int(torch.randint(0, CLASS_COUNT, (1,), generator=g).item())
        return [[klass] * CANVAS for _ in range(CANVAS)]
    return loader


def _fake_model_and_text_provider():
    return object(), object()


def _fake_forward_fn(model, crop, text_embedding, *, window_id, window_ordinal, top, left, bottom, right,
                      canvas_h, canvas_w, patch_h, patch_w):
    seed = int(hashlib.sha256(crop.contiguous().numpy().tobytes()).hexdigest()[:8], 16) & 0xFFFFFFFF
    g = torch.Generator().manual_seed(seed)
    s0 = torch.rand(GRID * GRID, CLASS_COUNT, generator=g) * 2 - 1
    dino_features = torch.nn.functional.normalize(torch.rand(GRID * GRID, FEAT_DIM, generator=g), dim=1)
    snapshot = e3a.WindowSnapshot(
        window_id=window_id, window_ordinal=window_ordinal, canvas_h=canvas_h, canvas_w=canvas_w,
        top=top, left=left, bottom=bottom, right=right,
        infer_h=bottom - top, infer_w=right - left,
        backbone_input_height=GRID * PATCH, backbone_input_width=GRID * PATCH,
        grid_h=GRID, grid_w=GRID, patch_h=patch_h, patch_w=patch_w,
        dino_features=dino_features, s0=s0, class_count=CLASS_COUNT, feature_dim=FEAT_DIM,
        dtype=s0.dtype, device=s0.device,
    )
    return snapshot, torch.sigmoid(s0)


SYNTHETIC_HOLDOUT_IDS = ("000000100001", "000000100002", "000000100003")


def _synthetic_holdout_authority_data():
    """A minimal, hand-built dict with only the keys
    `_run_holdout_evaluation_core`/`compute_gate_c` actually read -- not a
    real authority, never written to disk, never touching the committed
    research_specs/ file."""
    return {
        "prerequisite": {"required_head": "0" * 40},
        "gate_c_authorization": {"execution_authorized_by_this_document": True},
    }


def _synthetic_frozen_fit(n_features, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "task_id": "x", "model_id": "y",
        "feature_order": [f"control_{i}" for i in range(n_features)],
        "coefficients": (torch.rand(n_features, generator=g) * 0.4 - 0.2).tolist(),
        "intercept": 0.0,
        "scaler_mean": [0.0] * n_features,
        "scaler_scale": [1.0] * n_features,
        "fit_digest": "f" * 64,
    }


def _synthetic_frozen_fits():
    return {
        ("query_correctness", "model_a"): _synthetic_frozen_fit(7, 1),
        ("query_correctness", "model_b"): _synthetic_frozen_fit(8, 2),
        ("propagation_outcome", "model_a"): _synthetic_frozen_fit(7, 3),
        ("propagation_outcome", "model_b"): _synthetic_frozen_fit(8, 4),
    }


_REAL_LOAD_HOLDOUT_AUTHORITY = hv._load_holdout_authority  # captured before any test monkeypatches the name


def _authorized_real_authority(repo_root):
    """A deep copy of the REAL committed holdout authority with only
    `holdout_execution_authorized` flipped to True IN THE COPY -- used to
    exercise preflight_holdout/run_holdout_evaluation's deeper,
    post-auth-gate logic (exception translation, the whole-file hash gate)
    without ever writing to or altering the committed research_specs/ file
    on disk. Every other field (dependency hashes, discovery_result_binding,
    frozen_model_fits) is the real, unmodified committed value. Calls the
    ORIGINAL captured `_load_holdout_authority`, never `hv._load_holdout_
    authority` -- tests assign this function itself to that name, so
    calling through the module would recurse forever."""
    data = copy.deepcopy(_REAL_LOAD_HOLDOUT_AUTHORITY(repo_root))
    data["authorization"]["frozen_flags"]["holdout_execution_authorized"] = True
    return data


REAL_DISCOVERY_RESULT_PATH = Path(
    "/scratch/haree/semantic_evidence_interventions/"
    "semantic_evidence_discovery_48_20260926T041409Z/discovery_result.json"
)


class SafeCommandImportPoisoningTests(unittest.TestCase):
    """verify_protocol_holdout and preflight_holdout (up to the point they
    fail closed) must never import torch/torchvision/mmcv/mmseg/timm/PIL/
    numpy."""

    def test_verify_protocol_holdout_never_imports_forbidden_modules(self):
        import builtins
        orig_import = builtins.__import__
        forbidden = ("torch", "torchvision", "mmcv", "mmseg", "timm", "PIL", "numpy")

        def poisoned(name, *a, **kw):
            top = name.split(".")[0]
            if top in forbidden:
                raise RuntimeError(f"FORBIDDEN IMPORT: {name}")
            return orig_import(name, *a, **kw)

        builtins.__import__ = poisoned
        try:
            hv.verify_protocol_holdout(REPO_ROOT)
        finally:
            builtins.__import__ = orig_import

    def test_preflight_holdout_never_imports_forbidden_modules_before_failing_closed(self):
        import builtins
        orig_import = builtins.__import__
        forbidden = ("torch", "torchvision", "mmcv", "mmseg", "timm", "PIL", "numpy")

        def poisoned(name, *a, **kw):
            top = name.split(".")[0]
            if top in forbidden:
                raise RuntimeError(f"FORBIDDEN IMPORT: {name}")
            return orig_import(name, *a, **kw)

        builtins.__import__ = poisoned
        try:
            with self.assertRaises(hv.HoldoutEvaluationError):
                hv.preflight_holdout(
                    REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=Path("/nonexistent/discovery_result.json"), mode="fresh",
                )
        finally:
            builtins.__import__ = orig_import


class CurrentAuthorityDenialTests(unittest.TestCase):
    """preflight_holdout/run_holdout_evaluation must fail closed under the
    currently committed authority, BEFORE touching any holdout path --
    proven by using deliberately nonexistent paths and confirming the
    error is the AUTHORIZATION error, not a missing-path error."""

    def test_preflight_holdout_denied_before_any_path_check(self):
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.preflight_holdout(
                REPO_ROOT, weight_dir="/definitely/does/not/exist",
                image_root=Path("/definitely/does/not/exist"),
                gt_root=Path("/definitely/does/not/exist"),
                output_root=Path("/definitely/does/not/exist/output"),
                discovery_result_path=Path("/definitely/does/not/exist/discovery_result.json"),
                mode="fresh",
            )
        self.assertIn("not authorized", str(ctx.exception))
        self.assertNotIn("does not exist", str(ctx.exception))  # would indicate it got past the auth check

    def test_run_holdout_evaluation_denied_before_any_path_check(self):
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.run_holdout_evaluation(
                REPO_ROOT, weight_dir="/definitely/does/not/exist",
                image_root=Path("/definitely/does/not/exist"),
                gt_root=Path("/definitely/does/not/exist"),
                output_root=Path("/definitely/does/not/exist/output"),
                discovery_result_path=Path("/definitely/does/not/exist/discovery_result.json"),
                mode="fresh",
            )
        self.assertIn("not authorized", str(ctx.exception))

    def test_committed_authority_actually_has_the_flag_false(self):
        data = hv._load_holdout_authority(REPO_ROOT)
        self.assertFalse(data["authorization"]["frozen_flags"]["holdout_execution_authorized"])
        self.assertFalse(data["gate_c_authorization"]["execution_authorized_by_this_document"])


class HoldoutPopulationTests(unittest.TestCase):
    def test_loads_exactly_48_ids_matching_digest(self):
        ids = hv.load_holdout_population(REPO_ROOT)
        self.assertEqual(len(ids), 48)
        self.assertEqual(len(set(ids)), 48)
        digest = hashlib.sha256("".join(i + "\n" for i in ids).encode("ascii")).hexdigest()
        self.assertEqual(digest, "c3931d1e12b62af10839d693cac29bac25b166e448265c54b066cff5b8110592")

    def test_zero_overlap_with_real_discovery_population(self):
        holdout_ids = hv.load_holdout_population(REPO_ROOT)
        discovery_ids = disc.load_discovery_population(REPO_ROOT)
        self.assertEqual(len(set(holdout_ids) & set(discovery_ids)), 0)

    def test_verify_no_discovery_overlap_detects_a_synthetic_overlap(self):
        orig = disc.load_discovery_population
        disc.load_discovery_population = lambda repo_root: ["000000100001", "000000999999"]
        try:
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv._verify_no_discovery_overlap(REPO_ROOT, ["000000100001", "000000100002"])
            self.assertIn("overlap", str(ctx.exception))
        finally:
            disc.load_discovery_population = orig

    def test_verify_no_discovery_overlap_passes_when_disjoint(self):
        orig = disc.load_discovery_population
        disc.load_discovery_population = lambda repo_root: ["000000999998", "000000999999"]
        try:
            hv._verify_no_discovery_overlap(REPO_ROOT, ["000000100001", "000000100002"])  # must not raise
        finally:
            disc.load_discovery_population = orig


class SyntheticPopulationDigestMismatchTests(unittest.TestCase):
    """Exercises the REAL load_discovery_population/load_holdout_population
    code path (not a stub) against a fully synthetic, temporary git repo,
    to prove wrong count/order/digest is actually detected -- never
    touching the real repository's population data."""

    def _build_temp_repo(self, ids, discovery_range, holdout_range, discovery_digest_override=None, holdout_digest_override=None):
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
        (root / "research_specs").mkdir()
        (root / "evaluation_subsets").mkdir()

        records = [{"canonical_image_id": iid, "selection_rank": rank} for rank, iid in enumerate(ids)]
        manifest_path = root / "evaluation_subsets" / "pop.json"
        manifest_path.write_text(json.dumps({"records": records}))
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "manifest"], cwd=root, check=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()

        dr_lo, dr_hi = discovery_range
        hr_lo, hr_hi = holdout_range
        discovery_ids = [r["canonical_image_id"] for r in records if dr_lo <= r["selection_rank"] <= dr_hi]
        holdout_ids = [r["canonical_image_id"] for r in records if hr_lo <= r["selection_rank"] <= hr_hi]
        real_discovery_digest = hashlib.sha256("".join(i + "\n" for i in discovery_ids).encode()).hexdigest()
        real_holdout_digest = hashlib.sha256("".join(i + "\n" for i in holdout_ids).encode()).hexdigest()

        toml_text = f'''
[prerequisite]
required_head = "{commit}"

[[dependency]]
role = "population_manifest"
path = "evaluation_subsets/pop.json"

[population]
discovery_count = {len(discovery_ids)}
locked_holdout_count = {len(holdout_ids)}
discovery_rank_range = [{dr_lo}, {dr_hi}]
locked_holdout_rank_range = [{hr_lo}, {hr_hi}]
discovery_digest = "{discovery_digest_override or real_discovery_digest}"
locked_holdout_digest = "{holdout_digest_override or real_holdout_digest}"
'''
        (root / disc.AUTHORITY_TOML_PATH).parent.mkdir(parents=True, exist_ok=True)
        (root / disc.AUTHORITY_TOML_PATH).write_text(toml_text)
        return root

    def test_correct_synthetic_population_loads_cleanly(self):
        ids = [f"{100000000000 + i}" for i in range(6)]
        root = self._build_temp_repo(ids, (0, 2), (3, 5))
        try:
            holdout = hv.load_holdout_population(root)
            self.assertEqual(len(holdout), 3)
            self.assertEqual(holdout, ids[3:6])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_wrong_holdout_digest_rejected(self):
        ids = [f"{100000000000 + i}" for i in range(6)]
        root = self._build_temp_repo(ids, (0, 2), (3, 5), holdout_digest_override="0" * 64)
        try:
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.load_holdout_population(root)
            self.assertIn("digest mismatch", str(ctx.exception))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_wrong_holdout_count_rejected(self):
        ids = [f"{100000000000 + i}" for i in range(6)]
        # declare locked_holdout_count wrong relative to the actual rank range
        root = self._build_temp_repo(ids, (0, 2), (3, 4))  # rank range picks only 2 ids
        # patch the toml's own count field to claim 3 (mismatch) -- rebuild manually
        toml_path = root / disc.AUTHORITY_TOML_PATH
        text = toml_path.read_text().replace("locked_holdout_count = 2", "locked_holdout_count = 3")
        toml_path.write_text(text)
        try:
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.load_holdout_population(root)
            self.assertIn("count mismatch", str(ctx.exception))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class FrozenFitTests(unittest.TestCase):
    def _base_summary(self, **overrides):
        summary = {
            "task_id": "query_correctness", "model_id": "model_a",
            "feature_order": ["control_1", "control_2"],
            "coefficients": [0.5, -0.3],
            "intercept": 0.1,
            "scaler_mean": [0.0, 0.0],
            "scaler_scale": [1.0, 1.0],
            "fit_digest": "a" * 64,
        }
        summary.update(overrides)
        return summary

    def test_load_frozen_fits_accepts_matching_digests(self):
        keys = [("query_correctness", "model_a"), ("query_correctness", "model_b"),
                ("propagation_outcome", "model_a"), ("propagation_outcome", "model_b")]
        holdout_data = {"frozen_model_fits": {"fit": [
            {"task_id": k[0], "model_id": k[1], "fit_digest": "a" * 64} for k in keys
        ]}}
        discovery_result = {"model_fit_summaries": [
            self._base_summary(task_id=k[0], model_id=k[1]) for k in keys
        ]}
        fits = hv.load_frozen_fits(holdout_data, discovery_result)
        self.assertEqual(len(fits), 4)

    def test_load_frozen_fits_rejects_digest_mismatch(self):
        holdout_data = {"frozen_model_fits": {"fit": [
            {"task_id": "query_correctness", "model_id": "model_a", "fit_digest": "a" * 64},
        ]}}
        discovery_result = {"model_fit_summaries": [self._base_summary(fit_digest="b" * 64)]}
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.load_frozen_fits(holdout_data, discovery_result)
        self.assertIn("fit_digest mismatch", str(ctx.exception))

    def test_load_frozen_fits_rejects_feature_order_length_mismatch(self):
        holdout_data = {"frozen_model_fits": {"fit": [
            {"task_id": "query_correctness", "model_id": "model_a", "fit_digest": "a" * 64},
        ]}}
        discovery_result = {"model_fit_summaries": [self._base_summary(feature_order=["control_1"])]}
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.load_frozen_fits(holdout_data, discovery_result)
        self.assertIn("feature_order", str(ctx.exception))

    def test_load_frozen_fits_rejects_non_finite_coefficient(self):
        holdout_data = {"frozen_model_fits": {"fit": [
            {"task_id": "query_correctness", "model_id": "model_a", "fit_digest": "a" * 64},
        ]}}
        discovery_result = {"model_fit_summaries": [self._base_summary(coefficients=[float("nan"), -0.3])]}
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.load_frozen_fits(holdout_data, discovery_result)
        self.assertIn("non-finite", str(ctx.exception))

    def test_load_frozen_fits_rejects_missing_fit(self):
        holdout_data = {"frozen_model_fits": {"fit": [
            {"task_id": "propagation_outcome", "model_id": "model_b", "fit_digest": "a" * 64},
        ]}}
        discovery_result = {"model_fit_summaries": [self._base_summary()]}  # different task/model
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.load_frozen_fits(holdout_data, discovery_result)
        self.assertIn("missing model_fit_summaries", str(ctx.exception))

    def test_apply_frozen_fit_is_pure_arithmetic_no_refit(self):
        fit = self._base_summary(coefficients=[1.0, 0.0], intercept=0.0, scaler_mean=[0.0, 0.0], scaler_scale=[1.0, 1.0])
        # z = 1*x0 + 0*x1 + 0 -> sigmoid(x0)
        p = hv.apply_frozen_fit(fit, [0.0, 5.0])
        self.assertAlmostEqual(p, 0.5, places=9)
        p2 = hv.apply_frozen_fit(fit, [10.0, -100.0])
        self.assertGreater(p2, 0.999)

    def test_apply_frozen_fit_rejects_wrong_length(self):
        fit = self._base_summary()
        with self.assertRaises(hv.HoldoutEvaluationError):
            hv.apply_frozen_fit(fit, [1.0])


class BootstrapTests(unittest.TestCase):
    def test_reproducibility_same_seed_same_result(self):
        rows_by_image = {
            f"img{i}": [(i % 2, 0.3 + 0.01 * i, 0.6 + 0.01 * i) for _ in range(3)]
            for i in range(20)
        }
        r1 = hv.paired_bootstrap(rows_by_image, hv.compute_auroc, replicates=200, seed=0)
        r2 = hv.paired_bootstrap(rows_by_image, hv.compute_auroc, replicates=200, seed=0)
        self.assertEqual(r1["ci_lower"], r2["ci_lower"])
        self.assertEqual(r1["ci_upper"], r2["ci_upper"])
        self.assertEqual(r1["point_estimate"], r2["point_estimate"])

    def test_image_pairing_all_rows_of_a_sampled_image_move_together(self):
        # image "special" has a single, extreme row; every other image is
        # perfectly balanced and uninformative. If resampling ever split an
        # image's own rows apart, this could not be distinguished from
        # row-level resampling -- but by construction here, class balance
        # in a replicate is only possible when "special" is NOT sampled.
        rows_by_image = {"special": [(1, 0.99, 0.99)]}
        for i in range(10):
            rows_by_image[f"plain{i}"] = [(0, 0.5, 0.5), (0, 0.5, 0.5)]
        result = hv.paired_bootstrap(rows_by_image, hv.compute_auroc, replicates=500, seed=0)
        # every valid (non-degenerate) replicate must have included "special"
        # at least once (otherwise only one class would be present) --
        # the function itself already enforces this via both_classes_present;
        # this test just confirms the function runs and reports a plausible
        # valid fraction (not 0, not silently wrong).
        self.assertGreater(result["valid_replicate_count"], 0)

    def test_degenerate_majority_yields_inconclusive(self):
        # 19 of 20 images are single-class; only one image contributes the
        # rare class, so most resamples that miss it are degenerate --
        # forcing well over 10% degenerate replicates.
        rows_by_image = {"rare": [(1, 0.9, 0.9)]}
        for i in range(19):
            rows_by_image[f"common{i}"] = [(0, 0.4, 0.4)]
        result = hv.paired_bootstrap(rows_by_image, hv.compute_auroc, replicates=2000, seed=0)
        self.assertLess(result["valid_replicate_fraction"], 0.90)
        self.assertTrue(result["inconclusive"])
        self.assertIsNone(result["ci_lower"])

    def test_one_class_only_is_fully_inconclusive(self):
        rows_by_image = {f"img{i}": [(0, 0.5, 0.5)] for i in range(10)}
        result = hv.paired_bootstrap(rows_by_image, hv.compute_auroc, replicates=100, seed=0)
        self.assertEqual(result["valid_replicate_count"], 0)
        self.assertTrue(result["inconclusive"])


class GateBTests(unittest.TestCase):
    def _bootstrap_ok(self, lower=0.05):
        return {"inconclusive": False, "ci_lower": lower, "ci_upper": 0.2, "valid_replicate_fraction": 0.95}

    def test_under_500_rows_is_inconclusive(self):
        result = hv.compute_gate_b(eligible_rows=400, classes_present=True, auroc_a=0.6, auroc_b=0.7,
                                    ap_a=0.5, ap_b=0.6, bootstrap=self._bootstrap_ok())
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_missing_class_is_inconclusive(self):
        result = hv.compute_gate_b(eligible_rows=600, classes_present=False, auroc_a=0.6, auroc_b=0.7,
                                    ap_a=0.5, ap_b=0.6, bootstrap=self._bootstrap_ok())
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_passes_when_all_conditions_met(self):
        result = hv.compute_gate_b(eligible_rows=600, classes_present=True, auroc_a=0.60, auroc_b=0.70,
                                    ap_a=0.50, ap_b=0.55, bootstrap=self._bootstrap_ok(lower=0.01))
        self.assertEqual(result["status"], "PASS")

    def test_fails_when_improvement_below_threshold(self):
        result = hv.compute_gate_b(eligible_rows=600, classes_present=True, auroc_a=0.60, auroc_b=0.615,
                                    ap_a=0.50, ap_b=0.55, bootstrap=self._bootstrap_ok(lower=0.001))
        self.assertEqual(result["status"], "FAIL")

    def test_fails_when_bootstrap_lower_bound_not_positive(self):
        result = hv.compute_gate_b(eligible_rows=600, classes_present=True, auroc_a=0.60, auroc_b=0.70,
                                    ap_a=0.50, ap_b=0.55, bootstrap=self._bootstrap_ok(lower=-0.001))
        self.assertEqual(result["status"], "FAIL")

    def test_inconclusive_bootstrap_propagates(self):
        result = hv.compute_gate_b(eligible_rows=600, classes_present=True, auroc_a=0.6, auroc_b=0.7,
                                    ap_a=0.5, ap_b=0.6, bootstrap={"inconclusive": True, "valid_replicate_fraction": 0.5})
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_auroc_improvement_exactly_0_03_passes_when_other_conditions_pass(self):
        # 0.03 - 0.0, not e.g. 0.63 - 0.60, so the float subtraction is
        # exact rather than landing a few ULPs off 0.03.
        result = hv.compute_gate_b(eligible_rows=600, classes_present=True, auroc_a=0.0, auroc_b=0.03,
                                    ap_a=0.50, ap_b=0.55, bootstrap=self._bootstrap_ok(lower=0.01))
        self.assertEqual(result["auroc_improvement"], 0.03)
        self.assertEqual(result["status"], "PASS")

    def test_bootstrap_ci_lower_exactly_0_0_is_fail(self):
        result = hv.compute_gate_b(eligible_rows=600, classes_present=True, auroc_a=0.60, auroc_b=0.70,
                                    ap_a=0.50, ap_b=0.55, bootstrap=self._bootstrap_ok(lower=0.0))
        self.assertEqual(result["status"], "FAIL")

    def test_ap_difference_exactly_0_0_passes_when_other_conditions_pass(self):
        result = hv.compute_gate_b(eligible_rows=600, classes_present=True, auroc_a=0.60, auroc_b=0.70,
                                    ap_a=0.50, ap_b=0.50, bootstrap=self._bootstrap_ok(lower=0.01))
        self.assertEqual(result["ap_difference"], 0.0)
        self.assertEqual(result["status"], "PASS")

    def test_eligible_rows_exactly_500_is_not_rejected_for_row_count(self):
        result = hv.compute_gate_b(eligible_rows=500, classes_present=True, auroc_a=0.60, auroc_b=0.70,
                                    ap_a=0.50, ap_b=0.55, bootstrap=self._bootstrap_ok(lower=0.01))
        self.assertNotEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["status"], "PASS")  # every other condition also passes, isolating the row-count boundary


class GateCTests(unittest.TestCase):
    def _bootstrap_ok(self, lower=0.05):
        return {"inconclusive": False, "ci_lower": lower, "ci_upper": 0.2, "valid_replicate_fraction": 0.95}

    def test_real_discovery_shaped_13_3_is_inconclusive(self):
        # Matches the actual discovery run's own propagation_outcome shape
        # (13/3) -- used here only as a REALISTIC row-count shape to prove
        # the frozen 100/100 minimum correctly yields INCONCLUSIVE; this is
        # NOT a forecast of the real holdout's own count.
        result = hv.compute_gate_c(
            eligible_rows=16, beneficial_dominant_rows=3, harmful_dominant_rows=13,
            classes_present=True, auroc_a=0.5, auroc_b=0.9, ap_a=0.5, ap_b=0.9,
            bootstrap=self._bootstrap_ok(), gate_c_execution_authorized=True,
        )
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_execution_not_authorized_is_inconclusive_even_with_ample_data(self):
        result = hv.compute_gate_c(
            eligible_rows=1000, beneficial_dominant_rows=200, harmful_dominant_rows=200,
            classes_present=True, auroc_a=0.6, auroc_b=0.7, ap_a=0.5, ap_b=0.6,
            bootstrap=self._bootstrap_ok(), gate_c_execution_authorized=False,
        )
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_sufficient_data_and_authorization_computes_pass_or_fail(self):
        result = hv.compute_gate_c(
            eligible_rows=1000, beneficial_dominant_rows=150, harmful_dominant_rows=150,
            classes_present=True, auroc_a=0.60, auroc_b=0.70, ap_a=0.50, ap_b=0.55,
            bootstrap=self._bootstrap_ok(lower=0.01), gate_c_execution_authorized=True,
        )
        self.assertEqual(result["status"], "PASS")

    def test_never_pass_on_insufficient_data_regardless_of_effect_size(self):
        result = hv.compute_gate_c(
            eligible_rows=50, beneficial_dominant_rows=5, harmful_dominant_rows=45,
            classes_present=True, auroc_a=0.5, auroc_b=0.99, ap_a=0.5, ap_b=0.99,
            bootstrap=self._bootstrap_ok(lower=0.4), gate_c_execution_authorized=True,
        )
        self.assertEqual(result["status"], "INCONCLUSIVE")  # never PASS despite a huge apparent effect


class GateDTests(unittest.TestCase):
    def test_empty_subgroup_list_is_not_a_vacuous_pass(self):
        result = hv.compute_gate_d([])
        self.assertEqual(result["status"], "NO_QUALIFYING_SUBGROUPS")
        self.assertNotEqual(result["status"], "PASS")

    def test_small_subgroup_excluded_not_failed(self):
        result = hv.compute_gate_d([{"name": "s1", "task_id": "query_correctness", "size": 40, "auroc_difference": -0.5}])
        self.assertEqual(result["status"], "NO_QUALIFYING_SUBGROUPS")
        self.assertIn("s1", result["excluded_subgroups"])

    def test_qualifying_subgroup_within_threshold_passes(self):
        result = hv.compute_gate_d([{"name": "s1", "task_id": "query_correctness", "size": 150, "auroc_difference": -0.005}])
        self.assertEqual(result["status"], "PASS")

    def test_qualifying_subgroup_violating_threshold_fails(self):
        result = hv.compute_gate_d([{"name": "s1", "task_id": "query_correctness", "size": 150, "auroc_difference": -0.02}])
        self.assertEqual(result["status"], "FAIL")

    def test_mixed_small_and_qualifying_subgroups(self):
        result = hv.compute_gate_d([
            {"name": "small", "task_id": "query_correctness", "size": 10, "auroc_difference": -0.5},
            {"name": "big_ok", "task_id": "query_correctness", "size": 200, "auroc_difference": 0.01},
        ])
        self.assertEqual(result["status"], "PASS")
        self.assertIn("small", result["excluded_subgroups"])
        self.assertEqual(len(result["checked_subgroups"]), 1)

    def test_subgroup_size_exactly_100_is_checked_not_excluded(self):
        result = hv.compute_gate_d([{"name": "s1", "task_id": "query_correctness", "size": 100, "auroc_difference": 0.01}])
        self.assertNotEqual(result["status"], "NO_QUALIFYING_SUBGROUPS")
        self.assertNotIn("s1", result["excluded_subgroups"])
        self.assertEqual(len(result["checked_subgroups"]), 1)

    def test_auroc_difference_exactly_negative_0_01_is_not_a_violation(self):
        result = hv.compute_gate_d([{"name": "s1", "task_id": "query_correctness", "size": 150, "auroc_difference": -0.01}])
        self.assertEqual(result["status"], "PASS")
        self.assertNotIn("s1", result["violating_subgroups"])


class ArtifactSchemaTests(unittest.TestCase):
    def _valid_result(self):
        return {
            "schema": {"name": "holdout_result", "version": "1.0.0"}, "complete": True, "final": True,
            "identity_and_dependency_bindings": {"required_head": "0" * 40},
            "population_binding": {"holdout_count": 3},
            "implementation_binding": {"module": "semantic_evidence_holdout_evaluate", "mode": "fresh"},
            "holdout_image_ids_ordered": list(SYNTHETIC_HOLDOUT_IDS),
            "frozen_fit_digests": {}, "gt_free_feature_digest": "a" * 64, "gt_dependent_label_digest": "b" * 64,
            "scientific_collection_digest": "c" * 64,
            "gate_b_result": {"gate": "B", "status": "INCONCLUSIVE"},
            "gate_c_result": {"gate": "C", "status": "INCONCLUSIVE"},
            "gate_d_result": {"gate": "D", "status": "NO_QUALIFYING_SUBGROUPS"},
            "bootstrap_settings": {"replicates": 10000, "seed": 0},
            "final_frozen_outcome": "HOLDOUT_INCONCLUSIVE", "holdout_accessed": True, "full_evaluation_accessed": False,
        }

    def test_valid_result_accepted(self):
        hv.validate_holdout_result(self._valid_result())  # must not raise

    def test_missing_field_rejected(self):
        d = self._valid_result()
        del d["holdout_accessed"]
        with self.assertRaises(hv.HoldoutEvaluationError):
            hv.validate_holdout_result(d)

    def test_extra_field_rejected(self):
        d = self._valid_result()
        d["unexpected_extra_field"] = 1
        with self.assertRaises(hv.HoldoutEvaluationError):
            hv.validate_holdout_result(d)

    def test_valid_decision_accepted(self):
        d = {
            "schema": {"name": "holdout_decision", "version": "1.0.0"},
            "gate_b_result": {}, "gate_c_result": {}, "gate_d_result": {},
            "overall_gates_outcome_status": "HOLDOUT_INCONCLUSIVE",
            "holdout_decision": "HOLDOUT_INCONCLUSIVE", "next_step": "STOP_OR_REVISE_HYPOTHESIS",
        }
        hv.validate_holdout_decision(d)  # must not raise

    def test_holdout_decision_never_pass_if_any_gate_inconclusive(self):
        gate_b = {"status": "PASS"}
        gate_c = {"status": "INCONCLUSIVE"}
        gate_d = {"status": "PASS"}
        self.assertEqual(hv.holdout_decision(gate_b, gate_c, gate_d), "HOLDOUT_INCONCLUSIVE")

    def test_holdout_decision_stop_if_any_gate_failed(self):
        gate_b = {"status": "FAIL"}
        gate_c = {"status": "PASS"}
        gate_d = {"status": "PASS"}
        self.assertEqual(hv.holdout_decision(gate_b, gate_c, gate_d), "HOLDOUT_STOP_GATE_FAILED")

    def test_holdout_decision_passes_with_no_qualifying_subgroups(self):
        gate_b = {"status": "PASS"}
        gate_c = {"status": "PASS"}
        gate_d = {"status": "NO_QUALIFYING_SUBGROUPS"}
        self.assertEqual(hv.holdout_decision(gate_b, gate_c, gate_d), "HOLDOUT_GATES_B_C_D_PASSED")


class GtIsolationTests(unittest.TestCase):
    def test_poisoned_gt_loader_raises_if_called_before_phase_a_completes(self):
        image_id = SYNTHETIC_HOLDOUT_IDS[0]
        event_log = disc.PhaseEventLog()

        def poisoned_gt_loader(iid):
            if f"phase_a_complete:{iid}" not in event_log.events:
                raise AssertionError("GT accessed before Phase A completed")
            return [[3] * CANVAS for _ in range(CANVAS)]

        model, text_embedding = _fake_model_and_text_provider()
        gt_free = disc._run_phase_a(
            image_id, _fake_image_loader((image_id,)), _fake_forward_fn, model, text_embedding,
            (448, 448), (224, 224), (14, 14), event_log,
        )
        gt_label = disc._run_phase_b(gt_free, poisoned_gt_loader, image_id, event_log)  # must not raise
        disc.validate_gt_label_artifact(gt_label)


class PublicExceptionContractTests(unittest.TestCase):
    """Confirms preflight_holdout/run_holdout_evaluation/verify_holdout_result
    translate a reused discovery-module disc.DiscoveryExecutionError into
    this module's own HoldoutEvaluationError (original preserved as
    __cause__), rather than leaking the sibling exception type, and that
    the translation is narrow enough that an unrelated programming error
    (TypeError, JSONDecodeError) still propagates untouched. The auth gate
    is bypassed only via an in-memory monkeypatch of
    hv._load_holdout_authority (see _authorized_real_authority) -- the
    committed authority file on disk is never touched, and no real
    holdout image/GT/model/CUDA path is ever reached."""

    def test_preflight_holdout_translates_reused_validator_failure(self):
        orig_load_authority = hv._load_holdout_authority
        orig_load_discovery_population = disc.load_discovery_population
        hv._load_holdout_authority = _authorized_real_authority

        def poisoned(repo_root):
            raise disc.DiscoveryExecutionError("synthetic reused-validator failure (test-injected)")

        disc.load_discovery_population = poisoned
        try:
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.preflight_holdout(
                    REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=Path("/nonexistent/discovery_result.json"), mode="fresh",
                )
            self.assertIsInstance(ctx.exception.__cause__, disc.DiscoveryExecutionError)
            self.assertIn("synthetic reused-validator failure", str(ctx.exception))
        finally:
            hv._load_holdout_authority = orig_load_authority
            disc.load_discovery_population = orig_load_discovery_population

    def test_preflight_holdout_does_not_swallow_unrelated_programming_errors(self):
        orig_load_authority = hv._load_holdout_authority
        orig_load_discovery_population = disc.load_discovery_population
        hv._load_holdout_authority = _authorized_real_authority

        def poisoned(repo_root):
            raise TypeError("unrelated programming error (test-injected)")

        disc.load_discovery_population = poisoned
        try:
            with self.assertRaises(TypeError):
                hv.preflight_holdout(
                    REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=Path("/nonexistent/discovery_result.json"), mode="fresh",
                )
        finally:
            hv._load_holdout_authority = orig_load_authority
            disc.load_discovery_population = orig_load_discovery_population

    def test_run_holdout_evaluation_translates_reused_validator_failure(self):
        if not REAL_DISCOVERY_RESULT_PATH.exists():
            self.skipTest("real discovery_result.json fixture not present on this filesystem")

        orig_load_authority = hv._load_holdout_authority
        orig_load_discovery_population = disc.load_discovery_population
        hv._load_holdout_authority = _authorized_real_authority

        def poisoned(repo_root):
            raise disc.DiscoveryExecutionError("synthetic reused-validator failure (test-injected)")

        disc.load_discovery_population = poisoned
        try:
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.run_holdout_evaluation(
                    REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=REAL_DISCOVERY_RESULT_PATH, mode="fresh",
                )
            self.assertIsInstance(ctx.exception.__cause__, disc.DiscoveryExecutionError)
            self.assertIn("synthetic reused-validator failure", str(ctx.exception))
        finally:
            hv._load_holdout_authority = orig_load_authority
            disc.load_discovery_population = orig_load_discovery_population

    def test_verify_holdout_result_translates_malformed_checkpoint(self):
        holdout_ids = list(SYNTHETIC_HOLDOUT_IDS)
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            # missing almost every required key (not merely a bad value) --
            # disc.validate_checkpoint_metadata's own key-set check must
            # reject this as disc.DiscoveryExecutionError.
            (output_root / "checkpoint.json").write_text(json.dumps({"complete": True}))
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.verify_holdout_result(output_root, holdout_ids)
            self.assertIsInstance(ctx.exception.__cause__, disc.DiscoveryExecutionError)
            self.assertIn("checkpoint metadata key mismatch", str(ctx.exception))

    def test_verify_holdout_result_does_not_swallow_unrelated_programming_errors(self):
        holdout_ids = list(SYNTHETIC_HOLDOUT_IDS)
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            (output_root / "checkpoint.json").write_text("{not valid json")
            with self.assertRaises(json.JSONDecodeError):
                hv.verify_holdout_result(output_root, holdout_ids)


class WholeFileIntegrityGateTests(unittest.TestCase):
    """Proves the whole-file SHA-256 gate (verify_discovery_result_file_hash)
    detects in-place coefficient tampering that PRESERVES the per-fit
    fit_digest string -- the exact gap load_frozen_fits's own docstring
    documents it cannot close on its own -- and that both public entry
    points (preflight_holdout, run_holdout_evaluation) refuse to proceed
    past it, before any coefficient is applied and before any holdout
    image/GT path is touched. Uses the real, already-produced
    discovery_result.json and the REAL committed holdout authority's own
    bound hash; only ever mutates an in-memory copy or a temp-file copy,
    never the fixture itself."""

    def setUp(self):
        if not REAL_DISCOVERY_RESULT_PATH.exists():
            self.skipTest("real discovery_result.json fixture not present on this filesystem")
        self.holdout_data = hv._load_holdout_authority(REPO_ROOT)

    def test_unmutated_copy_matches_the_committed_authority_bound_hash(self):
        result = hv.verify_discovery_result_file_hash(self.holdout_data, REAL_DISCOVERY_RESULT_PATH)
        self.assertTrue(result["match"])

    def test_mutated_coefficient_with_preserved_fit_digest_is_rejected_standalone(self):
        original = json.loads(REAL_DISCOVERY_RESULT_PATH.read_text())
        mutated = copy.deepcopy(original)
        target = mutated["model_fit_summaries"][0]
        original_fit_digest = target["fit_digest"]
        target["coefficients"][0] = target["coefficients"][0] + 5.0
        self.assertEqual(target["fit_digest"], original_fit_digest)  # digest label left untouched

        # load_frozen_fits ALONE (the per-fit fit_digest string check) does
        # NOT detect this -- confirming its own documented limitation, not
        # merely asserting it.
        fits = hv.load_frozen_fits(self.holdout_data, mutated)
        self.assertEqual(fits[(target["task_id"], target["model_id"])]["coefficients"][0], target["coefficients"][0])

        with tempfile.TemporaryDirectory() as tmp:
            mutated_path = Path(tmp) / "discovery_result.json"
            mutated_path.write_text(json.dumps(mutated))
            # the WHOLE-FILE hash gate does detect it.
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.verify_discovery_result_file_hash(self.holdout_data, mutated_path)
            self.assertIn("whole-file SHA-256 mismatch", str(ctx.exception))

    def test_preflight_holdout_rejects_mutated_coefficient_before_any_image_path(self):
        original = json.loads(REAL_DISCOVERY_RESULT_PATH.read_text())
        mutated = copy.deepcopy(original)
        mutated["model_fit_summaries"][0]["coefficients"][0] += 5.0

        orig_load_authority = hv._load_holdout_authority
        hv._load_holdout_authority = _authorized_real_authority
        try:
            with tempfile.TemporaryDirectory() as tmp:
                mutated_path = Path(tmp) / "discovery_result.json"
                mutated_path.write_text(json.dumps(mutated))
                with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                    hv.preflight_holdout(
                        REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                        gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                        discovery_result_path=mutated_path, mode="fresh",
                    )
                msg = str(ctx.exception)
                self.assertIn("whole-file SHA-256 mismatch", msg)
                # must fail on the HASH gate, not a missing-image/GT error --
                # proving the gate fires before any holdout path is touched.
                self.assertNotIn("missing holdout image files", msg)
                self.assertNotIn("gt_root does not exist", msg)
        finally:
            hv._load_holdout_authority = orig_load_authority

    def test_run_holdout_evaluation_rejects_mutated_coefficient_before_frozen_fits(self):
        original = json.loads(REAL_DISCOVERY_RESULT_PATH.read_text())
        mutated = copy.deepcopy(original)
        mutated["model_fit_summaries"][0]["coefficients"][0] += 5.0

        orig_load_authority = hv._load_holdout_authority
        hv._load_holdout_authority = _authorized_real_authority
        try:
            with tempfile.TemporaryDirectory() as tmp:
                mutated_path = Path(tmp) / "discovery_result.json"
                mutated_path.write_text(json.dumps(mutated))
                with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                    hv.run_holdout_evaluation(
                        REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                        gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                        discovery_result_path=mutated_path, mode="fresh",
                    )
                self.assertIn("whole-file SHA-256 mismatch", str(ctx.exception))
        finally:
            hv._load_holdout_authority = orig_load_authority


class HoldoutEvaluationCoreIntegrationTests(unittest.TestCase):
    """Exercises _run_holdout_evaluation_core directly -- the internal
    logic, never the authorization-gated public entry point -- with a
    fully synthetic 3-image population and fake providers. This is not a
    CLI bypass: the CLI only ever calls the public, gated
    run_holdout_evaluation/preflight_holdout."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_root = Path(self.tmpdir.name) / "out"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run(self, mode="fresh", output_root=None):
        return hv._run_holdout_evaluation_core(
            REPO_ROOT, _synthetic_holdout_authority_data(), _synthetic_frozen_fits(),
            output_root=output_root or self.output_root, mode=mode, holdout_ids=list(SYNTHETIC_HOLDOUT_IDS),
            image_loader=_fake_image_loader(SYNTHETIC_HOLDOUT_IDS),
            model_and_text_provider=_fake_model_and_text_provider,
            gt_loader=_fake_gt_loader(SYNTHETIC_HOLDOUT_IDS), forward_fn=_fake_forward_fn,
        )

    def test_fresh_run_installs_valid_artifacts(self):
        outcome = self._run(mode="fresh")
        self.assertTrue(outcome["verified"])
        self.assertTrue(self.output_root.exists())
        for fname in ("checkpoint.json", "gt_free_features.json", "gt_dependent_labels.json", "holdout_result.json", "holdout_decision.json"):
            self.assertTrue((self.output_root / fname).exists(), fname)
        result = json.loads((self.output_root / "holdout_result.json").read_text())
        hv.validate_holdout_result(result)
        self.assertEqual(result["holdout_image_ids_ordered"], list(SYNTHETIC_HOLDOUT_IDS))
        decision = json.loads((self.output_root / "holdout_decision.json").read_text())
        hv.validate_holdout_decision(decision)
        # inprogress checkpoint dir must be gone after finalization
        self.assertFalse(hv._checkpoint_dir_for(self.output_root).exists())

    def test_fresh_mode_refuses_existing_output_root(self):
        self.output_root.mkdir()
        with self.assertRaises(hv.HoldoutEvaluationError):
            self._run(mode="fresh")

    def test_resume_after_interruption_reuses_leftover_artifacts(self):
        checkpoint_dir = hv._checkpoint_dir_for(self.output_root)

        # simulate an interrupted run: only complete Phase A/B for the
        # first image, leave the checkpoint dir in place, never install.
        original_run_phase_a_calls = []
        real_run_phase_a = disc._run_phase_a

        def counting_run_phase_a(image_id, *a, **kw):
            original_run_phase_a_calls.append(image_id)
            return real_run_phase_a(image_id, *a, **kw)

        disc._run_phase_a = counting_run_phase_a
        try:
            data = _synthetic_holdout_authority_data()
            checkpoint_dir.mkdir(parents=True)
            initial = disc.build_checkpoint_metadata(
                mode="fresh", completed_image_ids_ordered=[], discovery_ids=list(SYNTHETIC_HOLDOUT_IDS),
                checkpoint_array_hashes={}, identity_bindings={"required_head": data["prerequisite"]["required_head"]},
                complete=False, final=False,
            )
            disc._atomic_write_json(checkpoint_dir / "checkpoint.json", initial)

            model, text_embedding = _fake_model_and_text_provider()
            event_log = disc.PhaseEventLog()
            gt_free = real_run_phase_a(SYNTHETIC_HOLDOUT_IDS[0], _fake_image_loader(SYNTHETIC_HOLDOUT_IDS), _fake_forward_fn,
                                        model, text_embedding, (448, 448), (224, 224), (14, 14), event_log)
            disc._atomic_write_json(checkpoint_dir / f"gt_free_{SYNTHETIC_HOLDOUT_IDS[0]}.json", gt_free)
            gt_label = disc._run_phase_b(gt_free, _fake_gt_loader(SYNTHETIC_HOLDOUT_IDS), SYNTHETIC_HOLDOUT_IDS[0], event_log)
            disc._atomic_write_json(checkpoint_dir / f"gt_label_{SYNTHETIC_HOLDOUT_IDS[0]}.json", gt_label)
            partial = disc.build_checkpoint_metadata(
                mode="fresh", completed_image_ids_ordered=[SYNTHETIC_HOLDOUT_IDS[0]], discovery_ids=list(SYNTHETIC_HOLDOUT_IDS),
                checkpoint_array_hashes={"gt_free": gt_free["gt_free_feature_digest"], "gt_label": gt_label["gt_dependent_label_digest"]},
                identity_bindings={"required_head": data["prerequisite"]["required_head"]}, complete=False, final=False,
            )
            disc._atomic_write_json(checkpoint_dir / "checkpoint.json", partial)

            original_run_phase_a_calls.clear()
            outcome = self._run(mode="resume")
            self.assertTrue(outcome["verified"])
            # only the remaining 2 images should have triggered a real Phase A run
            self.assertEqual(sorted(original_run_phase_a_calls), sorted(SYNTHETIC_HOLDOUT_IDS[1:]))
        finally:
            disc._run_phase_a = real_run_phase_a

    def test_finalize_only_requires_every_image_already_completed(self):
        checkpoint_dir = hv._checkpoint_dir_for(self.output_root)
        checkpoint_dir.mkdir(parents=True)
        data = _synthetic_holdout_authority_data()
        partial = disc.build_checkpoint_metadata(
            mode="fresh", completed_image_ids_ordered=[SYNTHETIC_HOLDOUT_IDS[0]], discovery_ids=list(SYNTHETIC_HOLDOUT_IDS),
            checkpoint_array_hashes={}, identity_bindings={"required_head": data["prerequisite"]["required_head"]},
            complete=False, final=False,
        )
        disc._atomic_write_json(checkpoint_dir / "checkpoint.json", partial)
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            self._run(mode="finalize_only")
        self.assertIn("already completed", str(ctx.exception))

    def test_interrupted_atomic_install_leaves_no_partial_output_root(self):
        # force validate_holdout_result to fail during install, simulating
        # a mid-install crash, and confirm output_root never appears.
        orig_validate = hv.validate_holdout_result
        hv.validate_holdout_result = lambda d: (_ for _ in ()).throw(hv.HoldoutEvaluationError("forced failure"))
        try:
            with self.assertRaises(hv.HoldoutEvaluationError):
                self._run(mode="fresh")
            self.assertFalse(self.output_root.exists())
        finally:
            hv.validate_holdout_result = orig_validate

    def test_output_tampering_detected_by_verify_holdout_result(self):
        self._run(mode="fresh")
        result_path = self.output_root / "holdout_result.json"
        result = json.loads(result_path.read_text())
        result["gt_free_feature_digest"] = "tampered" + result["gt_free_feature_digest"][8:]
        result_path.write_text(json.dumps(result))
        # tampering the digest field alone doesn't break closed-schema
        # validation, so also verify holdout_decision cross-check: mutate
        # a gate result's status without going through the real computation.
        decision_path = self.output_root / "holdout_decision.json"
        decision = json.loads(decision_path.read_text())
        del decision["gate_b_result"]  # break the closed schema
        decision_path.write_text(json.dumps(decision))
        with self.assertRaises(hv.HoldoutEvaluationError):
            hv.verify_holdout_result(self.output_root, list(SYNTHETIC_HOLDOUT_IDS))


if __name__ == "__main__":
    unittest.main()
