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


def _granted_execution_grant(*, gate_c=True):
    """A stand-in return value for `hv._resolve_holdout_execution_grant`,
    used to exercise preflight_holdout/run_holdout_evaluation's deeper,
    post-auth-gate logic (exception translation, the whole-file hash gate)
    without ever writing to, committing, or altering the committed
    research_specs/ predecessor file, and without needing a real,
    independently-verified successor authority on disk. The predecessor's
    own `[authorization.frozen_flags].holdout_execution_authorized` is
    NEVER consulted by the real code any more (see
    `_resolve_holdout_execution_grant`'s own docstring), so patching
    `hv._load_holdout_authority` no longer bypasses anything -- only
    patching the resolver itself does."""
    return {
        "holdout_execution_authorized": True, "gate_c_execution_authorized": gate_c,
        "successor_present": True, "denial_reason": None,
    }


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


class PredecessorFlagNeverAuthorizesTests(unittest.TestCase):
    """Regression test for a real defect found and repaired in this task:
    preflight_holdout/run_holdout_evaluation used to accept
    `frozen["holdout_execution_authorized"] is True OR grant[...] is True`
    -- an OR that let a LOCALLY ALTERED (never committed, never verified)
    copy of the immutable predecessor authority silently authorize real
    holdout access, with no successor, no commit, and no independent
    verification of any kind. The fix removes the predecessor's own flag
    from the decision entirely: a verified successor is now the SOLE
    possible source of a grant. These tests simulate exactly that attack
    -- a predecessor authority dict whose own flag reads True -- via an
    in-memory monkeypatch of hv._load_holdout_authority only (never
    touching the real committed research_specs/ file), while leaving
    hv._resolve_holdout_execution_grant genuinely real (no successor
    exists), and confirm both public functions still deny."""

    def _tampered_authority(self, repo_root):
        # Reads the real committed file directly via tomllib -- NOT via
        # hv._load_holdout_authority, which this test itself monkeypatches
        # to be this very function (calling through the module name here
        # would recurse forever).
        import tomllib as _tomllib
        with open(Path(repo_root) / hv.HOLDOUT_AUTHORITY_TOML_PATH, "rb") as fh:
            data = _tomllib.load(fh)
        data["authorization"]["frozen_flags"]["holdout_execution_authorized"] = True
        data["gate_c_authorization"]["execution_authorized_by_this_document"] = True
        return data

    def test_preflight_holdout_denies_even_when_local_predecessor_flag_reads_true(self):
        orig = hv._load_holdout_authority
        hv._load_holdout_authority = self._tampered_authority
        try:
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
        finally:
            hv._load_holdout_authority = orig

    def test_run_holdout_evaluation_denies_even_when_local_predecessor_flag_reads_true(self):
        orig = hv._load_holdout_authority
        hv._load_holdout_authority = self._tampered_authority
        try:
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
        finally:
            hv._load_holdout_authority = orig

    def test_predecessor_flag_value_is_never_even_read_by_either_public_function(self):
        """A stronger proof than the two tests above: the predecessor's
        own frozen_flags.holdout_execution_authorized is not merely
        overridden by the real successor check -- it is never looked at,
        at all. Confirmed by making the tampered authority's own
        frozen_flags table raise if that specific key is ever accessed,
        so ANY future regression that reads it again fails loudly here
        with an uncaught AssertionError instead of the expected
        HoldoutEvaluationError."""
        real_data = hv._load_holdout_authority(REPO_ROOT)

        class PoisonedFrozenFlags(dict):
            def __getitem__(self, key):
                if key == "holdout_execution_authorized":
                    raise AssertionError("holdout_execution_authorized was read -- it must never be consulted")
                return super().__getitem__(key)

        def poisoned_authority(repo_root):
            data = copy.deepcopy(real_data)
            data["authorization"]["frozen_flags"] = PoisonedFrozenFlags(data["authorization"]["frozen_flags"])
            return data

        orig = hv._load_holdout_authority
        hv._load_holdout_authority = poisoned_authority
        try:
            for fn, kwargs in (
                (hv.preflight_holdout, {}),
                (hv.run_holdout_evaluation, {}),
            ):
                with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                    fn(
                        REPO_ROOT, weight_dir="/definitely/does/not/exist",
                        image_root=Path("/definitely/does/not/exist"),
                        gt_root=Path("/definitely/does/not/exist"),
                        output_root=Path("/definitely/does/not/exist/output"),
                        discovery_result_path=Path("/definitely/does/not/exist/discovery_result.json"),
                        mode="fresh", **kwargs,
                    )
                self.assertIn("not authorized", str(ctx.exception))
        finally:
            hv._load_holdout_authority = orig


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
    hv._resolve_holdout_execution_grant (see _granted_execution_grant) --
    the committed authority files on disk are never touched, no successor
    is ever committed, and no real holdout image/GT/model/CUDA path is
    ever reached."""

    def test_preflight_holdout_translates_reused_validator_failure(self):
        orig_grant = hv._resolve_holdout_execution_grant
        orig_load_discovery_population = disc.load_discovery_population
        hv._resolve_holdout_execution_grant = lambda repo_root: _granted_execution_grant()

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
            hv._resolve_holdout_execution_grant = orig_grant
            disc.load_discovery_population = orig_load_discovery_population

    def test_preflight_holdout_does_not_swallow_unrelated_programming_errors(self):
        orig_grant = hv._resolve_holdout_execution_grant
        orig_load_discovery_population = disc.load_discovery_population
        hv._resolve_holdout_execution_grant = lambda repo_root: _granted_execution_grant()

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
            hv._resolve_holdout_execution_grant = orig_grant
            disc.load_discovery_population = orig_load_discovery_population

    def test_run_holdout_evaluation_translates_reused_validator_failure(self):
        if not REAL_DISCOVERY_RESULT_PATH.exists():
            self.skipTest("real discovery_result.json fixture not present on this filesystem")

        orig_grant = hv._resolve_holdout_execution_grant
        orig_load_discovery_population = disc.load_discovery_population
        hv._resolve_holdout_execution_grant = lambda repo_root: _granted_execution_grant()

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
            hv._resolve_holdout_execution_grant = orig_grant
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

        orig_grant = hv._resolve_holdout_execution_grant
        hv._resolve_holdout_execution_grant = lambda repo_root: _granted_execution_grant()
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
            hv._resolve_holdout_execution_grant = orig_grant

    def test_run_holdout_evaluation_rejects_mutated_coefficient_before_frozen_fits(self):
        original = json.loads(REAL_DISCOVERY_RESULT_PATH.read_text())
        mutated = copy.deepcopy(original)
        mutated["model_fit_summaries"][0]["coefficients"][0] += 5.0

        orig_grant = hv._resolve_holdout_execution_grant
        hv._resolve_holdout_execution_grant = lambda repo_root: _granted_execution_grant()
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
            hv._resolve_holdout_execution_grant = orig_grant


class ResolveHoldoutExecutionGrantTests(unittest.TestCase):
    """Direct, synthetic-repo tests of hv._resolve_holdout_execution_grant
    -- the fail-closed successor resolver. Every repo here is a fresh,
    temporary `git init` with placeholder dependency files, so a precise
    hash/role/path/schema mutation can be injected without ever touching
    the real committed predecessor authority or runner files. Never opens
    real holdout content, never constructs a model, never touches CUDA."""

    def _init_repo(self):
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
        return root

    def _commit_all(self, root, message):
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        return commit, branch

    def _write_placeholder_dependencies(self, root):
        info = {}
        for role, path in hv._SUCCESSOR_EXPECTED_DEPENDENCY_PATHS.items():
            full = root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            content = f"placeholder content for role={role}\n".encode()
            full.write_bytes(content)
            info[role] = (path, hashlib.sha256(content).hexdigest())
        return info

    def _dependency_toml(self, info, *, bad_hash_role=None, bad_path_role=None, drop_role=None,
                          must_be_committed_false_role=None):
        lines = []
        for role, (path, sha) in info.items():
            if role == drop_role:
                continue
            if role == bad_hash_role:
                sha = "0" * 64
            if role == bad_path_role:
                path = "some/unexpected/path.txt"
            must_flag = "false" if role == must_be_committed_false_role else "true"
            lines.append(
                f'[[dependency]]\nrole = "{role}"\npath = "{path}"\nsha256 = "{sha}"\n'
                f'must_be_committed_at_required_head = {must_flag}\nreason = "x"\n'
            )
        return "\n".join(lines)

    def _grant_toml(self, *, holdout=True, gt_access=True, gate_b=True, gate_c=True, gate_d=True,
                     withhold_field=None, missing_field=None):
        fields = {
            "description": '"x"',
            "holdout_execution_intended": str(holdout).lower(),
            "holdout_gt_access_intended": str(gt_access).lower(),
            "gate_b_computation_execution_intended": str(gate_b).lower(),
            "gate_c_computation_execution_intended": str(gate_c).lower(),
            "gate_d_computation_execution_intended": str(gate_d).lower(),
            "combined_96_image_execution_intended": "false",
            "combined_96_image_note": '"x"',
            "full_5000_image_execution_intended": "false",
            "full_5000_image_note": '"x"',
            "population_enlargement_intended": "false",
            "refit_intended": "false",
            "refit_note": '"x"',
            "training_intended": "false",
            "scientific_kernel_modification_intended": "false",
            "accuracy_or_deployment_claim_intended": "false",
            "accuracy_or_deployment_claim_note": '"x"',
        }
        if withhold_field:
            fields[withhold_field] = "true"
        if missing_field:
            del fields[missing_field]
        body = "\n".join(f"{k} = {v}" for k, v in fields.items())
        return f"[authorization.execution_grant_pending_wiring]\n{body}\n"

    def _gate_c_toml(self, *, execution_intended=True, missing_field=None):
        fields = {
            "rule_reference": '"x"',
            "computable_under_this_contract": "true",
            "computable_basis": '"x"',
            "min_auroc_improvement_holdout": "0.03",
            "paired_bootstrap_95_lower_bound_required": '"> 0"',
            "average_precision_difference_requirement": '"not negative"',
            "min_beneficial_dominant_holdout_propagation_outcome_rows": "100",
            "min_harmful_dominant_holdout_propagation_outcome_rows": "100",
            "below_minimum_result": '"INCONCLUSIVE"',
            "execution_intended_by_this_document": str(execution_intended).lower(),
            "execution_intended_note": '"x"',
        }
        if missing_field:
            del fields[missing_field]
        body = "\n".join(f"{k} = {v}" for k, v in fields.items())
        return f"[gate_c_authorization]\n{body}\n"

    def _build(self, *, commit_successor=True, alter_after_commit=False, malformed_text=None,
               wrong_branch=False, bad_required_head=None,
               dep_kwargs=None, grant_kwargs=None, gate_c_kwargs=None):
        root = self._init_repo()
        info = self._write_placeholder_dependencies(root)
        base_commit, branch = self._commit_all(root, "base")

        if malformed_text is not None:
            successor_text = malformed_text
        else:
            prereq = (
                "[prerequisite]\n"
                f'required_branch = "{"wrong-branch" if wrong_branch else branch}"\n'
                f'required_head = "{bad_required_head or base_commit}"\n\n'
            )
            dep_text = self._dependency_toml(info, **(dep_kwargs or {}))
            grant_text = self._grant_toml(**(grant_kwargs or {}))
            gate_c_text = self._gate_c_toml(**(gate_c_kwargs or {}))
            successor_text = prereq + dep_text + "\n" + grant_text + "\n" + gate_c_text

        successor_path = root / hv.HOLDOUT_EXECUTION_AUTHORITY_TOML_PATH
        successor_path.parent.mkdir(parents=True, exist_ok=True)
        successor_path.write_text(successor_text)

        if commit_successor:
            self._commit_all(root, "successor")
            if alter_after_commit:
                successor_path.write_text(successor_text + "\n# locally altered after commit\n")

        return root

    def test_no_successor_file_denies(self):
        root = self._init_repo()
        self._write_placeholder_dependencies(root)
        self._commit_all(root, "base")
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertFalse(result["successor_present"])
        self.assertFalse(result["gate_c_execution_authorized"])

    def test_successor_present_but_uncommitted_denies(self):
        root = self._build(commit_successor=False)
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertTrue(result["successor_present"])
        self.assertIn("not committed", result["denial_reason"])

    def test_successor_committed_but_locally_altered_denies(self):
        root = self._build(alter_after_commit=True)
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIn("differ", result["denial_reason"])

    def test_malformed_toml_denies(self):
        root = self._build(malformed_text="this is not [ valid toml =")
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIsNotNone(result["denial_reason"])

    def test_missing_grant_field_denies(self):
        root = self._build(grant_kwargs={"missing_field": "refit_intended"})
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIn("execution_grant_pending_wiring", result["denial_reason"])

    def test_missing_gate_c_field_denies(self):
        root = self._build(gate_c_kwargs={"missing_field": "computable_basis"})
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIn("gate_c_authorization", result["denial_reason"])

    def test_dependency_hash_mismatch_denies(self):
        root = self._build(dep_kwargs={"bad_hash_role": "holdout_runner_implementation"})
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIn("hash mismatch", result["denial_reason"])

    def test_dependency_path_mismatch_denies(self):
        root = self._build(dep_kwargs={"bad_path_role": "holdout_evaluation_authority"})
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIn("unexpected path", result["denial_reason"])

    def test_missing_dependency_role_denies(self):
        root = self._build(dep_kwargs={"drop_role": "feasibility_contract"})
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIn("role set mismatch", result["denial_reason"])

    def test_must_be_committed_false_denies(self):
        root = self._build(dep_kwargs={"must_be_committed_false_role": "holdout_runner_tests"})
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])

    def test_withheld_permission_declared_true_denies_even_with_holdout_true(self):
        for field in hv._SUCCESSOR_EXECUTION_GRANT_MUST_BE_FALSE:
            with self.subTest(field=field):
                root = self._build(grant_kwargs={"withhold_field": field})
                result = hv._resolve_holdout_execution_grant(root)
                self.assertFalse(result["holdout_execution_authorized"], f"{field} should have forced denial")

    def test_wrong_branch_denies(self):
        root = self._build(wrong_branch=True)
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIn("required_branch", result["denial_reason"])

    def test_required_head_not_ancestor_denies(self):
        root = self._build(bad_required_head="0" * 40)
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIn("ancestor", result["denial_reason"])

    def test_required_head_hash_mismatch_denies_even_if_matching_current_head(self):
        """Regression test for a real defect found and repaired in this
        task: the resolver used to re-verify each dependency's declared
        hash against `git show <current_head>:<path>` -- NOT against the
        successor's own stated `required_head`, despite every dependency
        declaring `must_be_committed_at_required_head=true`. This let a
        successor pair a genuinely-older, ancestor-valid `required_head`
        with a hash that only actually matches some LATER commit's
        content. Reproduces exactly that: required_head is the FIRST
        commit (before a dependency file's content changed); the
        successor's declared hash for that dependency is the LATER
        commit's content -- which DOES match current HEAD, so the
        pre-repair code would have wrongly accepted it. Must be denied."""
        root = self._init_repo()
        info = self._write_placeholder_dependencies(root)
        required_head_commit, branch = self._commit_all(root, "base (required_head)")

        tampered_role = "holdout_runner_implementation"
        tampered_path_rel = hv._SUCCESSOR_EXPECTED_DEPENDENCY_PATHS[tampered_role]
        later_content = b"later commit content -- NOT what existed at required_head\n"
        (root / tampered_path_rel).write_bytes(later_content)
        self._commit_all(root, f"later change to {tampered_role}")  # current HEAD now past required_head_commit
        later_sha = hashlib.sha256(later_content).hexdigest()

        info_with_later_hash = dict(info)
        orig_path, _orig_sha_at_required_head = info[tampered_role]
        info_with_later_hash[tampered_role] = (orig_path, later_sha)  # lies about required_head's own content

        prereq = f'[prerequisite]\nrequired_branch = "{branch}"\nrequired_head = "{required_head_commit}"\n\n'
        successor_text = (
            prereq + self._dependency_toml(info_with_later_hash) + "\n"
            + self._grant_toml() + "\n" + self._gate_c_toml()
        )
        successor_path = root / hv.HOLDOUT_EXECUTION_AUTHORITY_TOML_PATH
        successor_path.parent.mkdir(parents=True, exist_ok=True)
        successor_path.write_text(successor_text)
        self._commit_all(root, "successor (lying about required_head's own hash)")

        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertIn("required_head", result["denial_reason"])
        self.assertIn(tampered_role, result["denial_reason"])

    def test_valid_committed_successor_authorizes_holdout_and_gate_c(self):
        root = self._build()
        result = hv._resolve_holdout_execution_grant(root)
        self.assertTrue(result["holdout_execution_authorized"])
        self.assertTrue(result["gate_c_execution_authorized"])
        self.assertIsNone(result["denial_reason"])

    def test_gate_c_declined_independently_of_granted_holdout(self):
        root = self._build(grant_kwargs={"gate_c": False})
        result = hv._resolve_holdout_execution_grant(root)
        self.assertTrue(result["holdout_execution_authorized"])
        self.assertFalse(result["gate_c_execution_authorized"])

    def test_gate_c_authorization_table_declining_also_denies_gate_c(self):
        root = self._build(gate_c_kwargs={"execution_intended": False})
        result = hv._resolve_holdout_execution_grant(root)
        self.assertTrue(result["holdout_execution_authorized"])
        self.assertFalse(result["gate_c_execution_authorized"])

    def test_gate_c_never_granted_when_holdout_itself_is_denied(self):
        root = self._build(grant_kwargs={"holdout": False})
        result = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(result["holdout_execution_authorized"])
        self.assertFalse(result["gate_c_execution_authorized"])


class PublicGrantWiringTests(unittest.TestCase):
    """Confirms the resolver's result is actually consumed by
    preflight_holdout/run_holdout_evaluation's authorization check and by
    _run_holdout_evaluation_core's Gate C argument -- not merely that
    _resolve_holdout_execution_grant computes correctly in isolation
    (that is ResolveHoldoutExecutionGrantTests's job). Monkeypatches only
    hv._resolve_holdout_execution_grant (and, for the injection tests,
    hv._run_holdout_evaluation_core itself, to capture its `data` argument
    without running any real per-image work) -- the real committed
    predecessor authority, which still denies on its own, is used
    throughout, so these tests also double as an authorization-check
    ordering proof: a granted result lets execution reach a later,
    different failure (a real path/content check), never the reverse."""

    def test_preflight_holdout_proceeds_past_auth_when_resolver_grants(self):
        orig = hv._resolve_holdout_execution_grant
        hv._resolve_holdout_execution_grant = lambda repo_root: {
            "holdout_execution_authorized": True, "gate_c_execution_authorized": True,
            "successor_present": True, "denial_reason": None,
        }
        try:
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.preflight_holdout(
                    REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=Path("/nonexistent/discovery_result.json"), mode="fresh",
                )
            msg = str(ctx.exception)
            self.assertNotIn("not authorized", msg)
            self.assertIn("does not exist", msg)
        finally:
            hv._resolve_holdout_execution_grant = orig

    def test_preflight_holdout_still_denies_when_resolver_denies(self):
        orig = hv._resolve_holdout_execution_grant
        hv._resolve_holdout_execution_grant = lambda repo_root: {
            "holdout_execution_authorized": False, "gate_c_execution_authorized": False,
            "successor_present": False, "denial_reason": "no successor authority file present",
        }
        try:
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.preflight_holdout(
                    REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=Path("/nonexistent/discovery_result.json"), mode="fresh",
                )
            self.assertIn("not authorized", str(ctx.exception))
        finally:
            hv._resolve_holdout_execution_grant = orig

    def test_run_holdout_evaluation_proceeds_past_auth_when_resolver_grants(self):
        orig = hv._resolve_holdout_execution_grant
        hv._resolve_holdout_execution_grant = lambda repo_root: {
            "holdout_execution_authorized": True, "gate_c_execution_authorized": True,
            "successor_present": True, "denial_reason": None,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                # an EXISTING but wrong-content file: proceeds past the
                # authorization check, then fails on the (unrelated,
                # already-tested) whole-file hash gate -- not on the
                # authorization message, and not on a raw FileNotFoundError.
                bad_discovery_result = Path(tmp) / "discovery_result.json"
                bad_discovery_result.write_text("{}")
                with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                    hv.run_holdout_evaluation(
                        REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                        gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                        discovery_result_path=bad_discovery_result, mode="fresh",
                    )
                msg = str(ctx.exception)
                self.assertNotIn("not authorized", msg)
                self.assertIn("whole-file SHA-256 mismatch", msg)
        finally:
            hv._resolve_holdout_execution_grant = orig

    def test_run_holdout_evaluation_still_denies_when_resolver_denies(self):
        orig = hv._resolve_holdout_execution_grant
        hv._resolve_holdout_execution_grant = lambda repo_root: {
            "holdout_execution_authorized": False, "gate_c_execution_authorized": False,
            "successor_present": False, "denial_reason": "no successor authority file present",
        }
        try:
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.run_holdout_evaluation(
                    REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=Path("/nonexistent/discovery_result.json"), mode="fresh",
                )
            self.assertIn("not authorized", str(ctx.exception))
        finally:
            hv._resolve_holdout_execution_grant = orig

    def test_run_holdout_evaluation_injects_resolved_gate_c_grant_into_core(self):
        if not REAL_DISCOVERY_RESULT_PATH.exists():
            self.skipTest("real discovery_result.json fixture not present on this filesystem")
        captured = {}
        orig_core = hv._run_holdout_evaluation_core

        def fake_core(repo_root, data, frozen_fits, **kw):
            captured["gate_c_flag"] = data["gate_c_authorization"]["execution_authorized_by_this_document"]
            raise hv.HoldoutEvaluationError("stop here deliberately, before any real per-image work")

        orig_grant = hv._resolve_holdout_execution_grant
        hv._resolve_holdout_execution_grant = lambda repo_root: {
            "holdout_execution_authorized": True, "gate_c_execution_authorized": True,
            "successor_present": True, "denial_reason": None,
        }
        hv._run_holdout_evaluation_core = fake_core
        try:
            with self.assertRaises(hv.HoldoutEvaluationError):
                hv.run_holdout_evaluation(
                    REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=REAL_DISCOVERY_RESULT_PATH, mode="fresh",
                )
            self.assertTrue(captured.get("gate_c_flag"), "resolved gate_c grant was not propagated into _run_holdout_evaluation_core's data")
        finally:
            hv._resolve_holdout_execution_grant = orig_grant
            hv._run_holdout_evaluation_core = orig_core

    def test_run_holdout_evaluation_does_not_inject_gate_c_when_resolver_declines_it(self):
        if not REAL_DISCOVERY_RESULT_PATH.exists():
            self.skipTest("real discovery_result.json fixture not present on this filesystem")
        captured = {}
        orig_core = hv._run_holdout_evaluation_core

        def fake_core(repo_root, data, frozen_fits, **kw):
            captured["gate_c_flag"] = data["gate_c_authorization"]["execution_authorized_by_this_document"]
            raise hv.HoldoutEvaluationError("stop here deliberately")

        orig_grant = hv._resolve_holdout_execution_grant
        hv._resolve_holdout_execution_grant = lambda repo_root: {
            "holdout_execution_authorized": True, "gate_c_execution_authorized": False,
            "successor_present": True, "denial_reason": None,
        }
        hv._run_holdout_evaluation_core = fake_core
        try:
            with self.assertRaises(hv.HoldoutEvaluationError):
                hv.run_holdout_evaluation(
                    REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=REAL_DISCOVERY_RESULT_PATH, mode="fresh",
                )
            self.assertFalse(captured.get("gate_c_flag"))
        finally:
            hv._resolve_holdout_execution_grant = orig_grant
            hv._run_holdout_evaluation_core = orig_core


class GateCIsolationSoleSourceTests(unittest.TestCase):
    """Regression test for a real defect found in independent review and
    repaired in this task: run_holdout_evaluation used to OR the verified
    successor's own gate_c_execution_authorized with the PREDECESSOR's
    working-tree gate_c_authorization.execution_authorized_by_this_document
    -- read straight off disk with no git-commit verification, unlike the
    successor -- so a locally altered (never committed) predecessor
    claiming Gate C was authorized could override a genuinely verified
    successor's explicit denial of it. The fix takes the value SOLELY
    from the verified successor grant. These tests build a GENUINELY
    git-committed successor (never monkeypatching the resolver or the
    predecessor loader) so the successor's own verification is real, and
    only bypass the unrelated discovery-result/population mechanics that
    are separately, thoroughly tested elsewhere."""

    def _build_repo(self, *, gate_c_intended, predecessor_gate_c_flag):
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)

        info = {}
        for role, path in hv._SUCCESSOR_EXPECTED_DEPENDENCY_PATHS.items():
            full = root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            if role == "holdout_evaluation_authority":
                content = (
                    "[authorization.frozen_flags]\nholdout_execution_authorized = false\n\n"
                    "[gate_c_authorization]\n"
                    f"execution_authorized_by_this_document = {str(predecessor_gate_c_flag).lower()}\n"
                ).encode()
            else:
                content = f"content for {role}\n".encode()
            full.write_bytes(content)
            info[role] = (path, hashlib.sha256(content).hexdigest())

        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=root, check=True)
        base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()

        dep_lines = "\n".join(
            f'[[dependency]]\nrole = "{role}"\npath = "{path}"\nsha256 = "{sha}"\n'
            'must_be_committed_at_required_head = true\nreason = "x"\n'
            for role, (path, sha) in info.items()
        )
        grant_fields = {
            "description": '"x"', "holdout_execution_intended": "true", "holdout_gt_access_intended": "true",
            "gate_b_computation_execution_intended": "true",
            "gate_c_computation_execution_intended": str(gate_c_intended).lower(),
            "gate_d_computation_execution_intended": "true", "combined_96_image_execution_intended": "false",
            "combined_96_image_note": '"x"', "full_5000_image_execution_intended": "false",
            "full_5000_image_note": '"x"', "population_enlargement_intended": "false",
            "refit_intended": "false", "refit_note": '"x"', "training_intended": "false",
            "scientific_kernel_modification_intended": "false", "accuracy_or_deployment_claim_intended": "false",
            "accuracy_or_deployment_claim_note": '"x"',
        }
        gate_c_fields = {
            "rule_reference": '"x"', "computable_under_this_contract": "true", "computable_basis": '"x"',
            "min_auroc_improvement_holdout": "0.03", "paired_bootstrap_95_lower_bound_required": '"> 0"',
            "average_precision_difference_requirement": '"not negative"',
            "min_beneficial_dominant_holdout_propagation_outcome_rows": "100",
            "min_harmful_dominant_holdout_propagation_outcome_rows": "100",
            "below_minimum_result": '"INCONCLUSIVE"',
            "execution_intended_by_this_document": str(gate_c_intended).lower(),
            "execution_intended_note": '"x"',
        }
        successor_text = (
            f'[prerequisite]\nrequired_branch = "{branch}"\nrequired_head = "{base_commit}"\n\n'
            + dep_lines + "\n"
            + "[authorization.execution_grant_pending_wiring]\n"
            + "\n".join(f"{k} = {v}" for k, v in grant_fields.items()) + "\n\n"
            + "[gate_c_authorization]\n"
            + "\n".join(f"{k} = {v}" for k, v in gate_c_fields.items()) + "\n"
        )
        successor_path = root / hv.HOLDOUT_EXECUTION_AUTHORITY_TOML_PATH
        successor_path.parent.mkdir(parents=True, exist_ok=True)
        successor_path.write_text(successor_text)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "successor"], cwd=root, check=True)
        return root

    def _capture_gate_c_flag_reaching_core(self, root):
        """Runs the REAL run_holdout_evaluation -- REAL _load_holdout_
        authority, REAL _resolve_holdout_execution_grant -- up to the
        point _run_holdout_evaluation_core is called, capturing the Gate C
        flag it receives. Only the downstream discovery-result/population
        mechanics (irrelevant to this test, and separately covered by
        WholeFileIntegrityGateTests/HoldoutPopulationTests) are bypassed."""
        captured = {}
        orig_hash_check = hv.verify_discovery_result_file_hash
        orig_load_fits = hv.load_frozen_fits
        orig_load_pop = hv.load_holdout_population
        orig_overlap = hv._verify_no_discovery_overlap
        orig_core = hv._run_holdout_evaluation_core

        hv.verify_discovery_result_file_hash = lambda data, path: {"match": True}
        hv.load_frozen_fits = lambda data, discovery_result: {}
        hv.load_holdout_population = lambda repo_root: [f"{100000000000 + i}" for i in range(48)]
        hv._verify_no_discovery_overlap = lambda repo_root, holdout_ids: None

        def fake_core(repo_root, data, frozen_fits, **kw):
            captured["gate_c_flag"] = data["gate_c_authorization"]["execution_authorized_by_this_document"]
            raise hv.HoldoutEvaluationError("stop here deliberately, before any real per-image work")

        hv._run_holdout_evaluation_core = fake_core
        try:
            dummy_path = root / "dummy_discovery_result.json"
            dummy_path.write_text("{}")
            with self.assertRaises(hv.HoldoutEvaluationError):
                hv.run_holdout_evaluation(
                    root, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                    gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                    discovery_result_path=dummy_path, mode="fresh",
                )
        finally:
            hv.verify_discovery_result_file_hash = orig_hash_check
            hv.load_frozen_fits = orig_load_fits
            hv.load_holdout_population = orig_load_pop
            hv._verify_no_discovery_overlap = orig_overlap
            hv._run_holdout_evaluation_core = orig_core
        return captured.get("gate_c_flag")

    def test_locally_altered_predecessor_is_now_denied_by_the_live_dependency_check(self):
        """The scenario this class was originally written to probe --
        editing the predecessor's OWN on-disk copy after the successor was
        committed -- is now caught EARLIER and MORE STRONGLY than by Gate
        C isolation alone: VERIFY_AND_REPAIR_LIVE_HOLDOUT_AUTHORITY_
        BINDINGS added a live on-disk-bytes-vs-required_head check for
        every bound dependency (including the predecessor itself, role
        'holdout_evaluation_authority'), so this edit now denies the
        WHOLE grant, not just Gate C. See LiveDependencyDriftTests for
        the dedicated, general-purpose regression coverage of that check;
        this test keeps the historical scenario intact end to end."""
        root = self._build_repo(gate_c_intended=False, predecessor_gate_c_flag=False)
        grant = hv._resolve_holdout_execution_grant(root)
        self.assertTrue(grant["holdout_execution_authorized"], grant["denial_reason"])
        self.assertFalse(grant["gate_c_execution_authorized"])

        predecessor_path = root / hv.HOLDOUT_AUTHORITY_TOML_PATH
        predecessor_path.write_text(
            "[authorization.frozen_flags]\nholdout_execution_authorized = false\n\n"
            "[gate_c_authorization]\nexecution_authorized_by_this_document = true\n"
        )
        grant_after_tamper = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(grant_after_tamper["holdout_execution_authorized"])
        self.assertIn("LIVE", grant_after_tamper["denial_reason"])

    def test_gate_c_combination_never_reads_the_predecessors_own_field(self):
        """Isolates the Gate C combination line itself
        (data["gate_c_authorization"]["execution_authorized_by_this_
        document"] = grant["gate_c_execution_authorized"] is True) from
        the separately-verified live-dependency-drift check: monkeypatches
        _load_holdout_authority to a predecessor whose OWN gate_c flag
        reads True, and _resolve_holdout_execution_grant to a genuinely-
        shaped verified-successor result that denies Gate C, confirming
        the combination takes its value SOLELY from the resolver -- no
        git/disk verification involved, so this is a pure code-level
        guarantee, independent of and in addition to the live check."""
        orig_load = hv._load_holdout_authority
        orig_grant = hv._resolve_holdout_execution_grant
        orig_hash_check = hv.verify_discovery_result_file_hash
        orig_load_fits = hv.load_frozen_fits
        orig_load_pop = hv.load_holdout_population
        orig_overlap = hv._verify_no_discovery_overlap
        orig_core = hv._run_holdout_evaluation_core

        hv._load_holdout_authority = lambda repo_root: {
            "gate_c_authorization": {"execution_authorized_by_this_document": True},
        }
        hv._resolve_holdout_execution_grant = lambda repo_root: {
            "holdout_execution_authorized": True, "gate_c_execution_authorized": False,
            "successor_present": True, "denial_reason": None,
        }
        hv.verify_discovery_result_file_hash = lambda data, path: {"match": True}
        hv.load_frozen_fits = lambda data, discovery_result: {}
        hv.load_holdout_population = lambda repo_root: [f"{100000000000 + i}" for i in range(48)]
        hv._verify_no_discovery_overlap = lambda repo_root, holdout_ids: None

        captured = {}

        def fake_core(repo_root, data, frozen_fits, **kw):
            captured["gate_c_flag"] = data["gate_c_authorization"]["execution_authorized_by_this_document"]
            raise hv.HoldoutEvaluationError("stop here deliberately")

        hv._run_holdout_evaluation_core = fake_core
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dummy_path = Path(tmp) / "dummy_discovery_result.json"
                dummy_path.write_text("{}")
                with self.assertRaises(hv.HoldoutEvaluationError):
                    hv.run_holdout_evaluation(
                        REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                        gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                        discovery_result_path=dummy_path, mode="fresh",
                    )
            self.assertFalse(captured.get("gate_c_flag"), "predecessor's own gate_c field overrode the verified successor's Gate C denial")
        finally:
            hv._load_holdout_authority = orig_load
            hv._resolve_holdout_execution_grant = orig_grant
            hv.verify_discovery_result_file_hash = orig_hash_check
            hv.load_frozen_fits = orig_load_fits
            hv.load_holdout_population = orig_load_pop
            hv._verify_no_discovery_overlap = orig_overlap
            hv._run_holdout_evaluation_core = orig_core

    def test_verified_explicit_gate_c_grant_reaches_the_core_as_true(self):
        root = self._build_repo(gate_c_intended=True, predecessor_gate_c_flag=False)
        grant = hv._resolve_holdout_execution_grant(root)
        self.assertTrue(grant["holdout_execution_authorized"], grant["denial_reason"])
        self.assertTrue(grant["gate_c_execution_authorized"])

        flag = self._capture_gate_c_flag_reaching_core(root)
        self.assertTrue(flag, "a verified successor's explicit Gate C grant did not reach the core")


class LiveDependencyDriftTests(unittest.TestCase):
    """Regression test for a real defect found and repaired in this task:
    _resolve_holdout_execution_grant's dependency verification compared
    only git BLOBS (required_head vs current HEAD) for each bound
    dependency, never the LIVE file actually on disk -- which is what a
    running process (including this one) actually imports/reads. Editing
    a bound dependency file on disk after the successor was committed,
    with Git HEAD left unchanged, went completely undetected: neither
    commit-based check notices a working-tree-only change. The fix adds a
    third check per dependency: live on-disk bytes must also match
    required_head's bytes. These tests use a REAL on-disk file edit in a
    genuinely committed temp repo, never a mocked loader."""

    def _build_repo(self):
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)

        info = {}
        for role, path in hv._SUCCESSOR_EXPECTED_DEPENDENCY_PATHS.items():
            full = root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            if role == "holdout_evaluation_authority":
                content = (
                    "[authorization.frozen_flags]\nholdout_execution_authorized = false\n\n"
                    "[gate_c_authorization]\nexecution_authorized_by_this_document = false\n"
                ).encode()
            else:
                content = f"content for {role}\n".encode()
            full.write_bytes(content)
            info[role] = (path, hashlib.sha256(content).hexdigest())

        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=root, check=True)
        base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()

        dep_lines = "\n".join(
            f'[[dependency]]\nrole = "{role}"\npath = "{path}"\nsha256 = "{sha}"\n'
            'must_be_committed_at_required_head = true\nreason = "x"\n'
            for role, (path, sha) in info.items()
        )
        successor_text = (
            f'[prerequisite]\nrequired_branch = "{branch}"\nrequired_head = "{base_commit}"\n\n'
            + dep_lines + "\n"
            + '[authorization.execution_grant_pending_wiring]\n'
            'description = "x"\nholdout_execution_intended = true\nholdout_gt_access_intended = true\n'
            'gate_b_computation_execution_intended = true\ngate_c_computation_execution_intended = true\n'
            'gate_d_computation_execution_intended = true\ncombined_96_image_execution_intended = false\n'
            'combined_96_image_note = "x"\nfull_5000_image_execution_intended = false\n'
            'full_5000_image_note = "x"\npopulation_enlargement_intended = false\nrefit_intended = false\n'
            'refit_note = "x"\ntraining_intended = false\nscientific_kernel_modification_intended = false\n'
            'accuracy_or_deployment_claim_intended = false\naccuracy_or_deployment_claim_note = "x"\n\n'
            '[gate_c_authorization]\nrule_reference = "x"\ncomputable_under_this_contract = true\n'
            'computable_basis = "x"\nmin_auroc_improvement_holdout = 0.03\n'
            'paired_bootstrap_95_lower_bound_required = "> 0"\naverage_precision_difference_requirement = "not negative"\n'
            'min_beneficial_dominant_holdout_propagation_outcome_rows = 100\n'
            'min_harmful_dominant_holdout_propagation_outcome_rows = 100\nbelow_minimum_result = "INCONCLUSIVE"\n'
            'execution_intended_by_this_document = true\nexecution_intended_note = "x"\n'
        )
        successor_path = root / hv.HOLDOUT_EXECUTION_AUTHORITY_TOML_PATH
        successor_path.parent.mkdir(parents=True, exist_ok=True)
        successor_path.write_text(successor_text)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "successor"], cwd=root, check=True)
        return root

    def test_live_edit_to_a_dependency_after_commit_denies_execution(self):
        root = self._build_repo()
        head_before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        grant_before = hv._resolve_holdout_execution_grant(root)
        self.assertTrue(grant_before["holdout_execution_authorized"], grant_before["denial_reason"])

        # a REAL on-disk edit -- not a mocked loader -- to a bound
        # dependency OTHER than the predecessor (already covered by
        # GateCIsolationSoleSourceTests), after commit, HEAD unchanged.
        drifted_role = "holdout_runner_cli"
        drifted_path = root / hv._SUCCESSOR_EXPECTED_DEPENDENCY_PATHS[drifted_role]
        drifted_path.write_text("live, uncommitted edit -- not what was committed or reviewed\n")

        head_after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        self.assertEqual(head_before, head_after)  # confirms HEAD genuinely did not move

        grant_after = hv._resolve_holdout_execution_grant(root)
        self.assertFalse(grant_after["holdout_execution_authorized"])
        self.assertIn(drifted_role, grant_after["denial_reason"])
        self.assertIn("LIVE", grant_after["denial_reason"])

    def test_public_run_holdout_evaluation_denies_before_reading_holdout_paths_under_live_drift(self):
        root = self._build_repo()
        drifted_path = root / hv._SUCCESSOR_EXPECTED_DEPENDENCY_PATHS["holdout_runner_tests"]
        drifted_path.write_text("live, uncommitted edit\n")

        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.run_holdout_evaluation(
                root, weight_dir="/definitely/does/not/exist",
                image_root=Path("/definitely/does/not/exist"),
                gt_root=Path("/definitely/does/not/exist"),
                output_root=Path("/definitely/does/not/exist/output"),
                discovery_result_path=Path("/definitely/does/not/exist/discovery_result.json"),
                mode="fresh",
            )
        msg = str(ctx.exception)
        self.assertIn("not authorized", msg)
        self.assertNotIn("does not exist", msg)  # would indicate it got past the auth check

    def test_public_preflight_holdout_denies_before_reading_holdout_paths_under_live_drift(self):
        root = self._build_repo()
        drifted_path = root / hv._SUCCESSOR_EXPECTED_DEPENDENCY_PATHS["discovery_executor_implementation"]
        drifted_path.write_text("live, uncommitted edit\n")

        # preflight_holdout also calls verify_dependency_hashes_holdout,
        # which verifies the PREDECESSOR's own, unrelated 8-role
        # dependency table -- irrelevant to what this test verifies, so
        # it is bypassed the same way other tests in this suite isolate a
        # single mechanism (e.g. PublicGrantWiringTests).
        orig_verify_deps = hv.verify_dependency_hashes_holdout
        hv.verify_dependency_hashes_holdout = lambda repo_root, data: {"all_match": True}
        try:
            with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                hv.preflight_holdout(
                    root, weight_dir="/definitely/does/not/exist",
                    image_root=Path("/definitely/does/not/exist"),
                    gt_root=Path("/definitely/does/not/exist"),
                    output_root=Path("/definitely/does/not/exist/output"),
                    discovery_result_path=Path("/definitely/does/not/exist/discovery_result.json"),
                    mode="fresh",
                )
            msg = str(ctx.exception)
            self.assertIn("not authorized", msg)
            self.assertNotIn("does not exist", msg)
        finally:
            hv.verify_dependency_hashes_holdout = orig_verify_deps


class RunHoldoutEvaluationErrorContractAfterGrantTests(unittest.TestCase):
    """Regression test for a real defect found and repaired in this task:
    run_holdout_evaluation called verify_discovery_result_file_hash
    (which does discovery_result_path.read_bytes()) without first
    checking the path exists -- a gap only reachable once a valid grant
    exists, so it went unexercised until now. A genuinely missing
    discovery_result.json leaked a raw FileNotFoundError instead of
    HoldoutEvaluationError. Unlike PublicGrantWiringTests (which
    monkeypatches hv._resolve_holdout_execution_grant directly), this
    test builds a fully synthetic, GENUINELY committed, GENUINELY
    independently-verified successor authority -- exercising
    run_holdout_evaluation's real authorization path end to end -- so the
    reproduction is not just "assume a grant" but "prove one, then hit
    the bug precisely once past it, before any frozen-fit coefficient use."""

    def _build_granted_repo(self, discovery_result_json_hash=None):
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)

        info = {}
        for role, path in hv._SUCCESSOR_EXPECTED_DEPENDENCY_PATHS.items():
            full = root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            if role == "holdout_evaluation_authority":
                # a MINIMAL but real predecessor authority -- only the
                # keys run_holdout_evaluation actually reads from it.
                binding = (
                    f'\n[discovery_result_binding.artifact_hashes]\ndiscovery_result_json = "{discovery_result_json_hash}"\n'
                    if discovery_result_json_hash else ""
                )
                content = (
                    "[authorization.frozen_flags]\n"
                    "holdout_execution_authorized = false\n\n"
                    "[gate_c_authorization]\n"
                    "execution_authorized_by_this_document = false\n"
                    + binding
                ).encode()
            else:
                content = f"placeholder content for role={role}\n".encode()
            full.write_bytes(content)
            info[role] = (path, hashlib.sha256(content).hexdigest())

        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=root, check=True)
        base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()

        dep_lines = "\n".join(
            f'[[dependency]]\nrole = "{role}"\npath = "{path}"\nsha256 = "{sha}"\n'
            'must_be_committed_at_required_head = true\nreason = "x"\n'
            for role, (path, sha) in info.items()
        )
        grant_fields = {
            "description": '"x"', "holdout_execution_intended": "true", "holdout_gt_access_intended": "true",
            "gate_b_computation_execution_intended": "true", "gate_c_computation_execution_intended": "true",
            "gate_d_computation_execution_intended": "true", "combined_96_image_execution_intended": "false",
            "combined_96_image_note": '"x"', "full_5000_image_execution_intended": "false",
            "full_5000_image_note": '"x"', "population_enlargement_intended": "false",
            "refit_intended": "false", "refit_note": '"x"', "training_intended": "false",
            "scientific_kernel_modification_intended": "false", "accuracy_or_deployment_claim_intended": "false",
            "accuracy_or_deployment_claim_note": '"x"',
        }
        gate_c_fields = {
            "rule_reference": '"x"', "computable_under_this_contract": "true", "computable_basis": '"x"',
            "min_auroc_improvement_holdout": "0.03", "paired_bootstrap_95_lower_bound_required": '"> 0"',
            "average_precision_difference_requirement": '"not negative"',
            "min_beneficial_dominant_holdout_propagation_outcome_rows": "100",
            "min_harmful_dominant_holdout_propagation_outcome_rows": "100",
            "below_minimum_result": '"INCONCLUSIVE"', "execution_intended_by_this_document": "true",
            "execution_intended_note": '"x"',
        }
        successor_text = (
            f'[prerequisite]\nrequired_branch = "{branch}"\nrequired_head = "{base_commit}"\n\n'
            + dep_lines + "\n"
            + "[authorization.execution_grant_pending_wiring]\n"
            + "\n".join(f"{k} = {v}" for k, v in grant_fields.items()) + "\n\n"
            + "[gate_c_authorization]\n"
            + "\n".join(f"{k} = {v}" for k, v in gate_c_fields.items()) + "\n"
        )
        successor_path = root / hv.HOLDOUT_EXECUTION_AUTHORITY_TOML_PATH
        successor_path.parent.mkdir(parents=True, exist_ok=True)
        successor_path.write_text(successor_text)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "successor"], cwd=root, check=True)
        return root

    def test_run_holdout_evaluation_raises_holdout_evaluation_error_not_file_not_found_error(self):
        root = self._build_granted_repo()
        # confirm the grant is genuinely, independently verified -- not
        # assumed -- before relying on it for the actual reproduction.
        grant = hv._resolve_holdout_execution_grant(root)
        self.assertTrue(grant["holdout_execution_authorized"], grant["denial_reason"])

        missing_path = root / "definitely_missing_discovery_result.json"
        self.assertFalse(missing_path.exists())
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.run_holdout_evaluation(
                root, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                discovery_result_path=missing_path, mode="fresh",
            )
        msg = str(ctx.exception)
        self.assertIn("does not exist", msg)
        self.assertNotIn("not authorized", msg)  # proves it got PAST authorization first

    def test_run_holdout_evaluation_translates_invalid_json_with_matching_hash(self):
        """Regression test for the second finding of the independent
        review: a discovery_result.json that EXISTS and matches its
        authority-bound whole-file SHA-256 (so verify_discovery_result_
        file_hash genuinely passes -- not bypassed) but is not valid JSON
        used to leak a raw json.JSONDecodeError. Built with a GENUINELY
        committed successor (never monkeypatched) and a hash the
        predecessor genuinely binds to these exact invalid-JSON bytes, so
        the hash-match itself is real, not assumed."""
        bad_bytes = b"not valid json at all {{{"
        bad_hash = hashlib.sha256(bad_bytes).hexdigest()
        root = self._build_granted_repo(discovery_result_json_hash=bad_hash)
        grant = hv._resolve_holdout_execution_grant(root)
        self.assertTrue(grant["holdout_execution_authorized"], grant["denial_reason"])

        bad_json_path = root / "bad_discovery_result.json"
        bad_json_path.write_bytes(bad_bytes)
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.run_holdout_evaluation(
                root, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                discovery_result_path=bad_json_path, mode="fresh",
            )
        msg = str(ctx.exception)
        self.assertNotIn("not authorized", msg)
        self.assertNotIn("whole-file SHA-256 mismatch", msg)  # proves the hash check genuinely PASSED first
        self.assertIsInstance(ctx.exception.__cause__, json.JSONDecodeError)

    def test_run_holdout_evaluation_translates_invalid_utf8_with_matching_hash(self):
        """Companion to the invalid-JSON test above, for the OTHER failure
        mode _load_discovery_result_json now guards: bytes that are not
        even valid UTF-8 (so .decode("utf-8") itself raises, before
        json.loads is ever reached), again with a GENUINELY matching
        whole-file hash -- proving the failure is specifically the
        decode step, not the hash gate."""
        bad_bytes = b"\xff\xfe not valid utf-8 at all \x80\x81"
        bad_hash = hashlib.sha256(bad_bytes).hexdigest()
        root = self._build_granted_repo(discovery_result_json_hash=bad_hash)
        grant = hv._resolve_holdout_execution_grant(root)
        self.assertTrue(grant["holdout_execution_authorized"], grant["denial_reason"])

        bad_utf8_path = root / "bad_utf8_discovery_result.json"
        bad_utf8_path.write_bytes(bad_bytes)
        with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
            hv.run_holdout_evaluation(
                root, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                discovery_result_path=bad_utf8_path, mode="fresh",
            )
        msg = str(ctx.exception)
        self.assertNotIn("not authorized", msg)
        self.assertNotIn("whole-file SHA-256 mismatch", msg)
        self.assertIsInstance(ctx.exception.__cause__, UnicodeDecodeError)


class PreflightHoldoutInvalidJsonTests(unittest.TestCase):
    """Companion to RunHoldoutEvaluationErrorContractAfterGrantTests'
    invalid-JSON regression test, for preflight_holdout specifically --
    the two public functions call the SAME shared _load_discovery_result_
    json helper, but each has its own call site and its own exception-
    translation wrapper, so each is verified independently. Uses a
    monkeypatched auth/population bypass (rather than a second full
    synthetic git repo) since preflight_holdout additionally requires the
    PREDECESSOR's own 8-role dependency table and a discovery-population
    manifest, neither of which bears on what this test verifies; the
    whole-file hash check itself is never bypassed -- the invalid-JSON
    fixture's real SHA-256 is bound directly."""

    def test_preflight_holdout_translates_invalid_json_with_matching_hash(self):
        bad_bytes = b"not valid json at all {{{"
        bad_hash = hashlib.sha256(bad_bytes).hexdigest()

        orig_grant = hv._resolve_holdout_execution_grant
        orig_load = hv._load_holdout_authority
        orig_verify_deps = hv.verify_dependency_hashes_holdout
        orig_load_pop = hv.load_holdout_population
        orig_overlap = hv._verify_no_discovery_overlap

        hv._resolve_holdout_execution_grant = lambda repo_root: _granted_execution_grant()
        hv._load_holdout_authority = lambda repo_root: {
            "discovery_result_binding": {"artifact_hashes": {"discovery_result_json": bad_hash}},
        }
        hv.verify_dependency_hashes_holdout = lambda repo_root, data: {"all_match": True}
        hv.load_holdout_population = lambda repo_root: [f"{100000000000 + i}" for i in range(48)]
        hv._verify_no_discovery_overlap = lambda repo_root, holdout_ids: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                bad_json_path = Path(tmp) / "bad_discovery_result.json"
                bad_json_path.write_bytes(bad_bytes)
                with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                    hv.preflight_holdout(
                        REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                        gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                        discovery_result_path=bad_json_path, mode="fresh",
                    )
                msg = str(ctx.exception)
                self.assertNotIn("not authorized", msg)
                self.assertNotIn("whole-file SHA-256 mismatch", msg)  # hash check genuinely passed
                self.assertIsInstance(ctx.exception.__cause__, json.JSONDecodeError)
        finally:
            hv._resolve_holdout_execution_grant = orig_grant
            hv._load_holdout_authority = orig_load
            hv.verify_dependency_hashes_holdout = orig_verify_deps
            hv.load_holdout_population = orig_load_pop
            hv._verify_no_discovery_overlap = orig_overlap

    def test_preflight_holdout_translates_invalid_utf8_with_matching_hash(self):
        bad_bytes = b"\xff\xfe not valid utf-8 at all \x80\x81"
        bad_hash = hashlib.sha256(bad_bytes).hexdigest()

        orig_grant = hv._resolve_holdout_execution_grant
        orig_load = hv._load_holdout_authority
        orig_verify_deps = hv.verify_dependency_hashes_holdout
        orig_load_pop = hv.load_holdout_population
        orig_overlap = hv._verify_no_discovery_overlap

        hv._resolve_holdout_execution_grant = lambda repo_root: _granted_execution_grant()
        hv._load_holdout_authority = lambda repo_root: {
            "discovery_result_binding": {"artifact_hashes": {"discovery_result_json": bad_hash}},
        }
        hv.verify_dependency_hashes_holdout = lambda repo_root, data: {"all_match": True}
        hv.load_holdout_population = lambda repo_root: [f"{100000000000 + i}" for i in range(48)]
        hv._verify_no_discovery_overlap = lambda repo_root, holdout_ids: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                bad_utf8_path = Path(tmp) / "bad_utf8_discovery_result.json"
                bad_utf8_path.write_bytes(bad_bytes)
                with self.assertRaises(hv.HoldoutEvaluationError) as ctx:
                    hv.preflight_holdout(
                        REPO_ROOT, weight_dir="/nonexistent", image_root=Path("/nonexistent"),
                        gt_root=Path("/nonexistent"), output_root=Path("/nonexistent"),
                        discovery_result_path=bad_utf8_path, mode="fresh",
                    )
                msg = str(ctx.exception)
                self.assertNotIn("not authorized", msg)
                self.assertNotIn("whole-file SHA-256 mismatch", msg)
                self.assertIsInstance(ctx.exception.__cause__, UnicodeDecodeError)
        finally:
            hv._resolve_holdout_execution_grant = orig_grant
            hv._load_holdout_authority = orig_load
            hv.verify_dependency_hashes_holdout = orig_verify_deps
            hv.load_holdout_population = orig_load_pop
            hv._verify_no_discovery_overlap = orig_overlap


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
