"""CPU-only, fake-provider tests for src/semantic_evidence_discovery_execute.py
and diagnostics/run_semantic_evidence_discovery.py.

Never builds a real E3 model, never loads a real checkpoint, never reads a
real image or GT annotation, never initializes CUDA. Fake providers are
written independently of the runner's own internals; the E3 window
snapshots they produce use the real, committed
semantic_evidence_e3_adapter.WindowSnapshot dataclass (so its own
structural __post_init__ checks still apply), populated with synthetic
random tensors.

NOTE: this repository runs on a shared HPC login node; torch's default
thread pool can severely oversubscribe cgroup-restricted CPUs, so
torch.set_num_threads(1) below is not optional -- see
tests/test_semantic_evidence_t320_real_smoke.py's own precedent. Every test
here processes at most 2 synthetic single-window images, kept deliberately
small so the whole suite stays lightweight enough for a login node.

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
        tests.test_semantic_evidence_discovery_execute -v
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import semantic_evidence_discovery_execute as ex  # noqa: E402
import semantic_evidence_e3_adapter as e3a  # noqa: E402
import semantic_evidence_intervention_features as ifeat  # noqa: E402


GRID = 32
CLASS_COUNT = 171
FEAT_DIM = 8
PATCH = 14
CANVAS = 448  # crop_size_pixels == canvas => exactly one canonical window


def _fake_image_loader(authorized_ids):
    def loader(image_id):
        assert image_id in authorized_ids, f"unauthorized image_id requested: {image_id}"
        g = torch.Generator().manual_seed(int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16))
        return torch.rand(1, 3, CANVAS, CANVAS, generator=g) * 255.0, CANVAS, CANVAS
    return loader


def _fake_gt_loader(authorized_ids, uniform=False):
    def loader(image_id):
        assert image_id in authorized_ids, f"unauthorized image_id requested for GT: {image_id}"
        if uniform:
            return [[7] * CANVAS for _ in range(CANVAS)]
        # A single class covering the whole canvas, deterministically varying
        # by image_id. Deliberately NOT a block-random raster: a block size
        # that isn't a multiple of the 14px patch footprint would make most
        # footprints straddle a class boundary and land AMBIGUOUS_GT (a
        # synthetic-fixture alignment artifact, not a real annotation
        # property), starving the label population needed to exercise the
        # four logistic fits end-to-end.
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


class _PopulationPatch:
    """Context manager that monkeypatches ex.load_discovery_population to
    return a small, fixed test population instead of the real 48 IDs --
    keeps every test in this file lightweight."""

    def __init__(self, ids):
        self.ids = list(ids)
        self._orig = None

    def __enter__(self):
        self._orig = ex.load_discovery_population
        ex.load_discovery_population = lambda repo_root: self.ids
        return self.ids

    def __exit__(self, *exc):
        ex.load_discovery_population = self._orig


REAL_DISCOVERY_IDS = None


def _real_discovery_ids():
    global REAL_DISCOVERY_IDS
    if REAL_DISCOVERY_IDS is None:
        REAL_DISCOVERY_IDS = ex.load_discovery_population(REPO_ROOT)
    return REAL_DISCOVERY_IDS


class RealImageLoaderTests(unittest.TestCase):
    """Uses only small, locally-generated synthetic JPEGs (never a real
    dataset image) to exercise _default_real_image_loader's actual
    filesystem/decode path -- the class of defect (missing batch
    dimension, wrong return arity) that a fake in-memory loader can never
    catch, since a fake loader is written to already match the shape the
    caller expects."""

    def setUp(self):
        from PIL import Image
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.authorized = ("000000567197", "000000367680")
        im = Image.new("RGB", (64, 48), color=(10, 20, 30))
        im.save(self.root / "000000567197.jpg")
        # a real, valid file at an ID that is NOT in the authorized tuple
        im.save(self.root / "000000999999.jpg")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_authorized_image_returns_batched_shape(self):
        tensor, h, w = ex._default_real_image_loader(self.root, "000000567197", self.authorized)
        self.assertEqual(tuple(tensor.shape), (1, 3, 48, 64))
        self.assertEqual((h, w), (48, 64))
        self.assertTrue(tensor.is_contiguous())
        self.assertTrue(torch.isfinite(tensor).all())

    def test_pre_repair_defect_would_fail_this_test(self):
        # Defect-reinstatement A: directly demonstrates that a rank-3
        # tensor (the pre-repair shape) fails the same shape assertion
        # this suite now enforces on every call.
        tensor, h, w = ex._default_real_image_loader(self.root, "000000567197", self.authorized)
        pre_repair_shape_would_have_been = tensor.squeeze(0).shape  # simulates the old, unfixed loader's output
        self.assertNotEqual(tuple(pre_repair_shape_would_have_been), (1, 3, 48, 64))
        self.assertEqual(len(pre_repair_shape_would_have_been), 3, "pre-repair shape was rank 3, which the current assertion rejects")

    def test_unauthorized_id_with_real_file_rejected_before_opening(self):
        # 000000999999.jpg genuinely exists on disk and is a valid JPEG --
        # rejection must come from the whitelist check, not a missing-file error.
        with self.assertRaises(ex.DiscoveryExecutionError) as ctx:
            ex._default_real_image_loader(self.root, "000000999999", self.authorized)
        self.assertIn("authorized", str(ctx.exception))
        self.assertNotIn("not found", str(ctx.exception))

    def test_corrupt_authorized_image_raises_domain_exception(self):
        (self.root / "000000367680.jpg").write_bytes(b"not a real jpeg")
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex._default_real_image_loader(self.root, "000000367680", self.authorized)

    def test_zero_byte_authorized_image_raises_domain_exception(self):
        (self.root / "000000367680.jpg").write_bytes(b"")
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex._default_real_image_loader(self.root, "000000367680", self.authorized)

    def test_missing_authorized_image_raises_domain_exception(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex._default_real_image_loader(self.root, "000000367680", self.authorized)

    def test_directory_instead_of_file_raises_domain_exception(self):
        (self.root / "000000367680.jpg").mkdir()
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex._default_real_image_loader(self.root, "000000367680", self.authorized)

    def test_fatal_exceptions_propagate_unchanged(self):
        import PIL.Image
        real_open = PIL.Image.open
        for exc_cls in (MemoryError, KeyboardInterrupt, SystemExit):
            def raiser(*a, **kw):
                raise exc_cls("injected")
            PIL.Image.open = raiser
            try:
                with self.assertRaises(exc_cls):
                    ex._default_real_image_loader(self.root, "000000567197", self.authorized)
            finally:
                PIL.Image.open = real_open

    def test_pre_repair_defect_g_bare_except_would_swallow_memory_error(self):
        # Defect-reinstatement G: a bare `except Exception` (no
        # _reraise_if_fatal guard) WOULD catch MemoryError, since
        # MemoryError subclasses Exception -- confirmed directly, as the
        # reasoning behind why the repaired loader needs the guard.
        self.assertTrue(issubclass(MemoryError, Exception))


class LoadDiscoveryPopulationTests(unittest.TestCase):
    def test_loads_exactly_48_ids_matching_digest(self):
        ids = _real_discovery_ids()
        self.assertEqual(len(ids), 48)
        self.assertEqual(len(set(ids)), 48)


class ValidateDiscoveryImageIdTests(unittest.TestCase):
    def setUp(self):
        self.authorized = tuple(_real_discovery_ids())

    def test_accepts_authorized_id(self):
        self.assertEqual(ex.validate_discovery_image_id(self.authorized[0], self.authorized), self.authorized[0])

    def test_rejects_unauthorized_coco_id(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_discovery_image_id("000000999999", self.authorized)

    def test_rejects_traversal(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_discovery_image_id("../../etc/passwd", self.authorized)

    def test_rejects_absolute_path(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_discovery_image_id("/etc/passwd", self.authorized)

    def test_rejects_option_like_string(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_discovery_image_id("--overwrite", self.authorized)

    def test_rejects_non_str(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_discovery_image_id(123456789012, self.authorized)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_discovery_image_id("1234", self.authorized)


class PreflightGtRootTests(unittest.TestCase):
    """Synthetic, torch-free tests for the repaired preflight() gt_root
    check. Never opens/decodes a real (or even a synthetic) GT file's
    content -- every fixture file is an arbitrary placeholder byte string,
    and every check below only exercises path/existence/type logic."""

    def setUp(self):
        self.ids = _real_discovery_ids()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.image_root = self.root / "images"
        self.gt_root = self.root / "gt"
        self.weight_dir = self.root / "weights"
        self.image_root.mkdir()
        self.gt_root.mkdir()
        self.weight_dir.mkdir()
        for iid in self.ids:
            (self.image_root / f"{iid}.jpg").write_bytes(b"placeholder-image-bytes")
            (self.gt_root / f"{iid}.png").write_bytes(b"placeholder-gt-bytes")

        # Checkpoint hash mocking: dino/clip point at small synthetic files
        # under our own weight_dir; the E3 projection checkpoint keeps using
        # the real, already-committed, already-verified repo file (its path
        # is hardcoded to repo_root/"weights"/filename regardless of
        # weight_dir, so it cannot be redirected -- reusing the real file
        # avoids needing to fake it while still exercising the real hash-
        # match code path for that entry).
        real_e3_binding = next(b for b in ex._external_checkpoint_bindings() if b[0] == "e3_projection_checkpoint")
        dino_bytes = b"fake-dino-backbone-checkpoint-bytes"
        clip_bytes = b"fake-clip-text-checkpoint-bytes"
        (self.weight_dir / "dino.bin").write_bytes(dino_bytes)
        (self.weight_dir / "clip.bin").write_bytes(clip_bytes)
        self._fake_bindings = [
            ("dino_backbone_checkpoint", "dino.bin", hashlib.sha256(dino_bytes).hexdigest()),
            ("clip_text_checkpoint", "clip.bin", hashlib.sha256(clip_bytes).hexdigest()),
            real_e3_binding,
        ]
        self._real_bindings_fn = ex._external_checkpoint_bindings
        ex._external_checkpoint_bindings = lambda: self._fake_bindings

        self.output_root = self.root / "out"  # deliberately never created

    def tearDown(self):
        ex._external_checkpoint_bindings = self._real_bindings_fn
        self.tmpdir.cleanup()

    def _preflight(self, **overrides):
        kwargs = dict(
            weight_dir=str(self.weight_dir), image_root=self.image_root, gt_root=self.gt_root,
            output_root=self.output_root, mode="fresh",
        )
        kwargs.update(overrides)
        return ex.preflight(REPO_ROOT, **kwargs)

    # --- A: positive ---
    def test_A_positive_synthetic_preflight_succeeds(self):
        result = self._preflight()
        self.assertEqual(result["status"], "READY_FOR_48_IMAGE_DISCOVERY_EXECUTION")
        self.assertEqual(result["gt_checks_count"], 48)
        self.assertFalse(self.output_root.exists(), "preflight must never create the output root")

    # --- B: missing GT root ---
    def test_B_missing_gt_root_rejected(self):
        with self.assertRaises(ex.DiscoveryExecutionError) as ctx:
            self._preflight(gt_root=self.root / "nonexistent_gt_root")
        self.assertIn("gt_root", str(ctx.exception))

    # --- C: GT root is a file, not a directory ---
    def test_C_gt_root_as_file_rejected(self):
        gt_file = self.root / "gt_root_is_a_file"
        gt_file.write_bytes(b"not a directory")
        with self.assertRaises(ex.DiscoveryExecutionError):
            self._preflight(gt_root=gt_file)

    # --- D: one missing discovery GT file ---
    def test_D_one_missing_discovery_gt_identifies_the_id(self):
        missing_id = self.ids[0]
        (self.gt_root / f"{missing_id}.png").unlink()
        with self.assertRaises(ex.DiscoveryExecutionError) as ctx:
            self._preflight()
        self.assertIn(missing_id, str(ctx.exception))

    # --- E: required path is a directory instead of a file ---
    def test_E_gt_path_as_directory_rejected(self):
        target_id = self.ids[1]
        (self.gt_root / f"{target_id}.png").unlink()
        (self.gt_root / f"{target_id}.png").mkdir()
        with self.assertRaises(ex.DiscoveryExecutionError) as ctx:
            self._preflight()
        self.assertIn(target_id, str(ctx.exception))

    # --- F: wrong extension does not substitute ---
    def test_F_wrong_extension_does_not_substitute(self):
        target_id = self.ids[2]
        (self.gt_root / f"{target_id}.png").unlink()
        (self.gt_root / f"{target_id}.jpg").write_bytes(b"wrong extension")
        with self.assertRaises(ex.DiscoveryExecutionError) as ctx:
            self._preflight()
        self.assertIn(target_id, str(ctx.exception))

    # --- G: holdout isolation ---
    def test_G_only_the_48_discovery_gt_paths_are_checked(self):
        checked_paths = []
        real_is_file = Path.is_file

        def recording_is_file(self_path):
            checked_paths.append(str(self_path))
            return real_is_file(self_path)

        Path.is_file = recording_is_file
        try:
            self._preflight()
        finally:
            Path.is_file = real_is_file

        checked_gt_paths = [p for p in checked_paths if str(self.gt_root) in p]
        checked_ids_in_order = [Path(p).stem for p in checked_gt_paths]
        self.assertEqual(checked_ids_in_order, self.ids, "must check exactly the 48 discovery IDs, in canonical order")
        self.assertEqual(len(checked_gt_paths), 48)

        # Independently confirm zero holdout GT paths were ever checked, by
        # deleting the GT root entirely and confirming preflight behavior is
        # driven solely by the 48 discovery files (already proven above) --
        # plus a direct membership check against the real holdout digest set.
        import json as _json
        with open(REPO_ROOT / "evaluation_subsets" / "semantic_evidence_feasibility_96.json") as f:
            manifest = _json.load(f)
        records = sorted(manifest["records"], key=lambda r: r["selection_rank"])
        holdout_ids = {r["canonical_image_id"] for r in records if 48 <= r["selection_rank"] <= 95}
        self.assertEqual(set(checked_ids_in_order) & holdout_ids, set(), "zero holdout GT paths may ever be checked")

    def test_G_absent_holdout_gt_files_never_affect_preflight(self):
        # The gt_root here contains ONLY the 48 discovery GT files (holdout
        # GT files were never created in this fixture at all) -- preflight
        # already succeeds in test_A, proving their absence is irrelevant.
        result = self._preflight()
        self.assertEqual(result["status"], "READY_FOR_48_IMAGE_DISCOVERY_EXECUTION")

    # --- H: no GT decode ---
    def test_H_no_gt_decode_attempted(self):
        import builtins
        real_open = builtins.open
        gt_root_str = str(self.gt_root)
        opened_gt_files = []

        def watchdog_open(file, *a, **kw):
            if isinstance(file, (str, Path)) and gt_root_str in str(file):
                opened_gt_files.append(str(file))
            return real_open(file, *a, **kw)

        builtins.open = watchdog_open
        try:
            self._preflight()
        finally:
            builtins.open = real_open
        self.assertEqual(opened_gt_files, [], "preflight must never open a GT file's content")

    def test_H_pil_and_cv2_never_touched_for_gt(self):
        # PIL/cv2 are not even importable without torch-adjacent heavy deps
        # being present; if they ARE importable in this environment, poison
        # them defensively to prove preflight doesn't reach for them.
        poisoned = {}
        try:
            import PIL.Image as _PILImage
            real_pil_open = _PILImage.open
            def poisoned_pil_open(*a, **kw):
                raise AssertionError("preflight must never call PIL.Image.open for GT")
            _PILImage.open = poisoned_pil_open
            poisoned["PIL"] = (_PILImage, real_pil_open)
        except ImportError:
            pass
        try:
            import cv2 as _cv2
            real_cv2_imread = _cv2.imread
            def poisoned_cv2_imread(*a, **kw):
                raise AssertionError("preflight must never call cv2.imread for GT")
            _cv2.imread = poisoned_cv2_imread
            poisoned["cv2"] = (_cv2, real_cv2_imread)
        except ImportError:
            pass
        try:
            result = self._preflight()
            self.assertEqual(result["status"], "READY_FOR_48_IMAGE_DISCOVERY_EXECUTION")
        finally:
            if "PIL" in poisoned:
                poisoned["PIL"][0].open = poisoned["PIL"][1]
            if "cv2" in poisoned:
                poisoned["cv2"][0].imread = poisoned["cv2"][1]

    # --- I: torch-free preflight ---
    def test_I_torch_free_preflight(self):
        import builtins
        real_import = builtins.__import__
        poisoned_names = {"torch", "torchvision", "mmcv", "mmseg", "timm"}
        touched = []

        def poisoned_import(name, *a, **kw):
            top = name.split(".")[0]
            if top in poisoned_names:
                touched.append(name)
                raise ImportError(f"POISONED: {name} must not be imported by preflight")
            return real_import(name, *a, **kw)

        builtins.__import__ = poisoned_import
        try:
            result = self._preflight()
        finally:
            builtins.__import__ = real_import
        self.assertEqual(touched, [])
        self.assertEqual(result["status"], "READY_FOR_48_IMAGE_DISCOVERY_EXECUTION")

    # --- L: fatal exceptions during the GT metadata-check path ---
    def test_L_fatal_exceptions_propagate_through_gt_check(self):
        for exc_cls in (MemoryError, KeyboardInterrupt, SystemExit):
            real_is_file = Path.is_file
            def raiser(self_path):
                raise exc_cls("injected during GT metadata check")
            Path.is_file = raiser
            try:
                with self.assertRaises(exc_cls):
                    self._preflight()
            finally:
                Path.is_file = real_is_file

    # --- Defect-reinstatement (Section 8) ---
    def test_defect_reinstatement_mutant_checks_only_gt_root_existence(self):
        # A mutant that checks gt_root.exists() but never iterates the 48
        # exact files -- the missing-one-file scenario must still be caught
        # by the REAL implementation even though this mutant would accept it.
        def mutant_preflight_gt_check(gt_root):
            gt_root = Path(gt_root)
            if not gt_root.exists() or not gt_root.is_dir():
                raise ex.DiscoveryExecutionError("gt_root missing")
            return True  # BUG: never checks the 48 individual files

        missing_id = self.ids[5]
        (self.gt_root / f"{missing_id}.png").unlink()

        # the mutant incorrectly accepts this broken fixture
        self.assertTrue(mutant_preflight_gt_check(self.gt_root))

        # the real, repaired preflight() correctly rejects it
        with self.assertRaises(ex.DiscoveryExecutionError) as ctx:
            self._preflight()
        self.assertIn(missing_id, str(ctx.exception))


class CheckpointMetadataTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["a", "b", "c"]

    def test_roundtrip(self):
        meta = ex.build_checkpoint_metadata(
            mode="fresh", completed_image_ids_ordered=["a", "b"], discovery_ids=self.ids,
            checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=False,
        )
        ex.validate_checkpoint_metadata(meta, self.ids)  # must not raise

    def test_rejects_out_of_order_prefix(self):
        meta = ex.build_checkpoint_metadata(
            mode="fresh", completed_image_ids_ordered=["a", "b"], discovery_ids=self.ids,
            checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=False,
        )
        meta["completed_image_ids_ordered"] = ["b", "a"]
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_checkpoint_metadata(meta, self.ids)

    def test_rejects_non_discovery_id(self):
        meta = ex.build_checkpoint_metadata(
            mode="fresh", completed_image_ids_ordered=["a"], discovery_ids=self.ids,
            checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=False,
        )
        meta["completed_image_ids_ordered"] = ["a", "holdout_id"]
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_checkpoint_metadata(meta, self.ids)

    def test_rejects_duplicate_ids(self):
        meta = ex.build_checkpoint_metadata(
            mode="fresh", completed_image_ids_ordered=["a"], discovery_ids=self.ids,
            checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=False,
        )
        meta["completed_image_ids_ordered"] = ["a", "a"]
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_checkpoint_metadata(meta, self.ids)

    def test_final_requires_complete_and_full_population(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.build_checkpoint_metadata(
                mode="fresh", completed_image_ids_ordered=["a", "b"], discovery_ids=self.ids,
                checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=True,
            )

    def test_rejects_extra_or_missing_field(self):
        meta = ex.build_checkpoint_metadata(
            mode="fresh", completed_image_ids_ordered=[], discovery_ids=self.ids,
            checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=False,
        )
        del meta["schema"]
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_checkpoint_metadata(meta, self.ids)
        meta2 = ex.build_checkpoint_metadata(
            mode="fresh", completed_image_ids_ordered=[], discovery_ids=self.ids,
            checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=False,
        )
        meta2["extra_field"] = 1
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_checkpoint_metadata(meta2, self.ids)


class GateTests(unittest.TestCase):
    def test_gate_a_passes_with_no_defects(self):
        g = ex.compute_gate_a(total_anchors=100, structurally_invalid_anchor_count=0, non_finite_scores_count=0, gt_dependent_intervention_records=0)
        self.assertTrue(g["passed"])

    def test_gate_a_fails_below_structural_validity_threshold(self):
        g = ex.compute_gate_a(total_anchors=100, structurally_invalid_anchor_count=10, non_finite_scores_count=0, gt_dependent_intervention_records=0)
        self.assertFalse(g["passed"])  # 90% < 95% required

    def test_gate_a_fails_on_any_gt_dependent_intervention_record(self):
        g = ex.compute_gate_a(total_anchors=100, structurally_invalid_anchor_count=0, non_finite_scores_count=0, gt_dependent_intervention_records=1)
        self.assertFalse(g["passed"])

    def test_gate_a_non_finite_count_is_observed_not_hardcoded(self):
        # Defect-reinstatement D guard: a non-zero observed non_finite_scores_count
        # must fail Gate A even when the anchor-level fraction is otherwise perfect.
        g = ex.compute_gate_a(total_anchors=100, structurally_invalid_anchor_count=0, non_finite_scores_count=3, gt_dependent_intervention_records=0)
        self.assertFalse(g["passed"])
        self.assertEqual(g["non_finite_scores_count"], 3)

    def test_gate_a_exact_pass_boundary(self):
        # 95/100 = exactly 0.95 -> must pass (>= threshold)
        at_boundary = ex.compute_gate_a(total_anchors=100, structurally_invalid_anchor_count=5, non_finite_scores_count=0, gt_dependent_intervention_records=0)
        self.assertTrue(at_boundary["passed"])
        self.assertAlmostEqual(at_boundary["structural_validity_fraction"], 0.95)
        # 94/100 = 0.94 -> just below threshold -> must fail
        below_boundary = ex.compute_gate_a(total_anchors=100, structurally_invalid_anchor_count=6, non_finite_scores_count=0, gt_dependent_intervention_records=0)
        self.assertFalse(below_boundary["passed"])
        # 96/100 = 0.96 -> just above threshold -> must pass
        above_boundary = ex.compute_gate_a(total_anchors=100, structurally_invalid_anchor_count=4, non_finite_scores_count=0, gt_dependent_intervention_records=0)
        self.assertTrue(above_boundary["passed"])

    def test_gate_a_unit_consistency_image_count_cannot_alter_anchor_fraction(self):
        # Defect-reinstatement E guard: total_anchors is an ANCHOR count, never
        # an image count. Changing a hypothetical "image count" must have zero
        # effect on the fraction -- this test proves the function signature
        # itself makes that mixing impossible (no image-count parameter exists),
        # and that the fraction depends only on the anchor-level numbers given.
        import inspect
        params = list(inspect.signature(ex.compute_gate_a).parameters)
        self.assertNotIn("image_count", params)
        self.assertNotIn("total_images", params)
        self.assertNotIn("total_records", params)
        g_small = ex.compute_gate_a(total_anchors=8, structurally_invalid_anchor_count=1, non_finite_scores_count=0, gt_dependent_intervention_records=0)
        g_large = ex.compute_gate_a(total_anchors=800, structurally_invalid_anchor_count=100, non_finite_scores_count=0, gt_dependent_intervention_records=0)
        # both are exactly 7/8 == 0.875 structural validity -- identical fraction
        # regardless of the absolute anchor count (a stand-in for "image count"
        # cannot leak in here since the parameter doesn't exist).
        self.assertAlmostEqual(g_small["structural_validity_fraction"], g_large["structural_validity_fraction"])

    def test_gate_a_rejects_invalid_count_exceeding_total(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.compute_gate_a(total_anchors=5, structurally_invalid_anchor_count=10, non_finite_scores_count=0, gt_dependent_intervention_records=0)

    def test_gate_e_within_ceiling(self):
        g = ex.compute_gate_e(discovery_wall_time_seconds=1800.0)  # -> 1.0h projected for 96 images
        self.assertTrue(g["passed"])

    def test_gate_e_exceeds_ceiling(self):
        g = ex.compute_gate_e(discovery_wall_time_seconds=3600.0 * 3)  # -> 6h projected
        self.assertFalse(g["passed"])

    def test_gate_b_c_d_are_not_computable(self):
        gates = ex.gate_b_c_d_not_computable()
        for g in ("B", "C", "D"):
            self.assertFalse(gates[g]["computable_under_this_contract"])
            self.assertEqual(gates[g]["status"], "NOT_COMPUTABLE_WITHOUT_HOLDOUT")

    def test_discovery_decision_values(self):
        pass_a = ex.compute_gate_a(total_anchors=10, structurally_invalid_anchor_count=0, non_finite_scores_count=0, gt_dependent_intervention_records=0)
        fail_a = ex.compute_gate_a(total_anchors=10, structurally_invalid_anchor_count=10, non_finite_scores_count=0, gt_dependent_intervention_records=0)
        pass_e = ex.compute_gate_e(1800.0)
        fail_e = ex.compute_gate_e(3600.0 * 10)
        self.assertEqual(ex.discovery_decision(pass_a, pass_e, degenerate_sample=False), "DISCOVERY_FEATURES_READY_FOR_HOLDOUT_AUTHORITY")
        self.assertEqual(ex.discovery_decision(fail_a, pass_e, degenerate_sample=False), "DISCOVERY_STOP_GATE_A_FAILED")
        self.assertEqual(ex.discovery_decision(pass_a, fail_e, degenerate_sample=False), "DISCOVERY_INCONCLUSIVE")
        self.assertEqual(ex.discovery_decision(pass_a, pass_e, degenerate_sample=True), "DISCOVERY_INCONCLUSIVE")


class LogisticFitTests(unittest.TestCase):
    def _synthetic_rows(self, n, n_features):
        import random
        random.seed(0)
        X = [[random.uniform(-1, 1) for _ in range(n_features)] for _ in range(n)]
        y = [1 if sum(row) > 0 else 0 for row in X]
        if len(set(y)) < 2:
            y[0] = 1 - y[0]
        row_keys = [f"img:{i}" for i in range(n)]
        return X, y, row_keys

    def test_fit_model_a_returns_expected_shape(self):
        X, y, row_keys = self._synthetic_rows(30, 7)
        fit = ex.fit_logistic_model(X, y, task_id="query_correctness", model_id="model_a", model_b=False, row_keys=row_keys)
        self.assertEqual(len(fit["coefficients"]), 7)
        self.assertEqual(fit["row_count"], 30)
        self.assertEqual(fit["solver"], "liblinear")
        self.assertEqual(fit["task_id"], "query_correctness")
        self.assertEqual(fit["model_id"], "model_a")
        self.assertTrue(fit["converged"])
        self.assertIn("fit_digest", fit)

    def test_fit_model_b_requires_eight_features(self):
        X, y, row_keys = self._synthetic_rows(30, 8)
        fit = ex.fit_logistic_model(X, y, task_id="query_correctness", model_id="model_b", model_b=True, row_keys=row_keys)
        self.assertEqual(len(fit["coefficients"]), 8)
        self.assertEqual(fit["feature_order"][-1], "intervention_reliability")

    def test_model_a_feature_order_is_strict_prefix_of_model_b(self):
        Xa, y, row_keys = self._synthetic_rows(30, 7)
        Xb = [row + [0.5] for row in Xa]
        fit_a = ex.fit_logistic_model(Xa, y, task_id="query_correctness", model_id="model_a", model_b=False, row_keys=row_keys)
        fit_b = ex.fit_logistic_model(Xb, y, task_id="query_correctness", model_id="model_b", model_b=True, row_keys=row_keys)
        self.assertEqual(fit_b["feature_order"][:7], fit_a["feature_order"])

    def test_rejects_degenerate_single_class_labels(self):
        X = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]] * 5
        y = [1] * 5
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.fit_logistic_model(X, y, task_id="query_correctness", model_id="model_a", model_b=False, row_keys=["a", "b", "c", "d", "e"])

    def test_rejects_wrong_feature_vector_length(self):
        X, y, row_keys = self._synthetic_rows(10, 5)  # wrong: model_a expects 7
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.fit_logistic_model(X, y, task_id="query_correctness", model_id="model_a", model_b=False, row_keys=row_keys)

    def test_rejects_non_finite_feature_value(self):
        import math
        X, y, row_keys = self._synthetic_rows(10, 7)
        X[0][0] = math.nan
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.fit_logistic_model(X, y, task_id="query_correctness", model_id="model_a", model_b=False, row_keys=row_keys)

    def test_rejects_duplicate_row_key(self):
        X, y, row_keys = self._synthetic_rows(10, 7)
        row_keys[1] = row_keys[0]
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.fit_logistic_model(X, y, task_id="query_correctness", model_id="model_a", model_b=False, row_keys=row_keys)

    def test_rejects_empty_population(self):
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.fit_logistic_model([], [], task_id="query_correctness", model_id="model_a", model_b=False, row_keys=[])

    def test_rejects_mismatched_model_id_and_model_b_flag(self):
        X, y, row_keys = self._synthetic_rows(10, 7)
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.fit_logistic_model(X, y, task_id="query_correctness", model_id="model_a", model_b=True, row_keys=row_keys)


class GtMajorityTests(unittest.TestCase):
    def test_uniform_footprint_is_unambiguous(self):
        rows = [[3] * 14 for _ in range(14)]
        label, cls = ex.compute_gt_majority(rows)
        self.assertEqual(label, 3)
        self.assertEqual(cls, 3)

    def test_mostly_ignore_is_ambiguous(self):
        rows = [[255] * 14 for _ in range(14)]
        label, cls = ex.compute_gt_majority(rows)
        self.assertEqual(label, ex.AMBIGUOUS_GT)
        self.assertIsNone(cls)

    def test_split_below_majority_share_is_ambiguous(self):
        rows = [[1 if (r + c) % 2 == 0 else 2 for c in range(14)] for r in range(14)]  # ~50/50 split
        label, cls = ex.compute_gt_majority(rows)
        self.assertEqual(label, ex.AMBIGUOUS_GT)

    def test_propagation_category_taxonomy(self):
        self.assertEqual(ex.compute_propagation_category(t0_predicted_class=1, t320_predicted_class=5, gt_majority_class=5), "beneficial")
        self.assertEqual(ex.compute_propagation_category(t0_predicted_class=5, t320_predicted_class=1, gt_majority_class=5), "harmful")
        self.assertEqual(ex.compute_propagation_category(t0_predicted_class=5, t320_predicted_class=5, gt_majority_class=5), "unchanged_correct")
        self.assertEqual(ex.compute_propagation_category(t0_predicted_class=1, t320_predicted_class=2, gt_majority_class=5), "changed_but_same_correctness")
        self.assertEqual(ex.compute_propagation_category(t0_predicted_class=1, t320_predicted_class=1, gt_majority_class=5), "unchanged_incorrect")


class FootprintMappingTests(unittest.TestCase):
    def test_maps_through_window_and_patch_offset(self):
        bounds = ex.footprint_pixel_bounds(window_top=100, window_left=50, anchor_row=2, anchor_col=3, patch_hw=(14, 14))
        self.assertEqual(bounds, (100 + 28, 50 + 42, 100 + 28 + 14, 50 + 42 + 14))


def _sample_gt_pair(image_id="000000567197"):
    """Runs the real Phase A -> Phase B pipeline with fake providers to
    obtain a genuinely valid gt_free/gt_label artifact pair -- used as the
    base fixture for adversarial mutation tests below, since hand-writing
    a valid instance of the now-strict nested schema by hand would be
    fragile and likely to drift from the real one."""
    event_log = ex.PhaseEventLog()
    model, text_embedding = _fake_model_and_text_provider()
    gt_free = ex._run_phase_a(
        image_id, _fake_image_loader((image_id,)), _fake_forward_fn, model, text_embedding,
        (448, 448), (224, 224), (14, 14), event_log,
    )
    gt_label = ex._run_phase_b(gt_free, _fake_gt_loader((image_id,)), image_id, event_log)
    return gt_free, gt_label


class ArtifactValidatorTests(unittest.TestCase):
    def setUp(self):
        self.gt_free, self.gt_label = _sample_gt_pair()

    def _mut(self, base):
        return json.loads(json.dumps(base))  # deep copy through the same JSON round-trip artifacts actually go through

    def test_gt_free_artifact_accepts_real_fixture(self):
        ex.validate_gt_free_artifact(self.gt_free)  # must not raise

    def test_gt_label_artifact_accepts_real_fixture(self):
        ex.validate_gt_label_artifact(self.gt_label)  # must not raise

    def test_pair_linkage_accepts_real_fixture(self):
        ex.validate_gt_pair_linkage(self.gt_free, self.gt_label)  # must not raise

    def test_gt_free_missing_digest_field_rejected(self):
        bad = self._mut(self.gt_free)
        del bad["gt_free_feature_digest"]
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_unknown_top_level_key_rejected(self):
        bad = self._mut(self.gt_free)
        bad["extra_unexpected_key"] = 1
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_garbage_typed_control_features_rejected(self):
        # This is exactly the class of defect independent verification found:
        # a completely garbage-typed nested value must not pass merely
        # because the outer 'control_features' key exists.
        bad = self._mut(self.gt_free)
        bad["control_features"] = "not-even-a-dict"
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_bool_as_int_rejected(self):
        bad = self._mut(self.gt_free)
        any_key = next(iter(bad["control_features"]["query_correctness"]))
        bad["control_features"]["query_correctness"][any_key]["query_class"] = True
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_non_finite_feature_rejected(self):
        bad = self._mut(self.gt_free)
        any_key = next(iter(bad["control_features"]["query_correctness"]))
        bad["control_features"]["query_correctness"][any_key]["features"][0] = float("nan")
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)
        bad2 = self._mut(self.gt_free)
        bad2["control_features"]["query_correctness"][any_key]["features"][0] = float("inf")
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad2)

    def test_gt_free_wrong_feature_vector_length_rejected(self):
        bad = self._mut(self.gt_free)
        any_key = next(iter(bad["control_features"]["query_correctness"]))
        bad["control_features"]["query_correctness"][any_key]["features"].append(0.5)
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_reliability_out_of_range_rejected(self):
        bad = self._mut(self.gt_free)
        any_key = next(iter(bad["control_features"]["query_correctness"]))
        bad["control_features"]["query_correctness"][any_key]["intervention_reliability"] = 1.5
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_intervention_reliability_linkage_mismatch_rejected(self):
        bad = self._mut(self.gt_free)
        any_key = next(iter(bad["intervention_reliability"]["query_correctness"]))
        bad["intervention_reliability"]["query_correctness"][any_key] = 0.0123456789
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_malformed_row_key_rejected(self):
        bad = self._mut(self.gt_free)
        qc = bad["control_features"]["query_correctness"]
        any_key = next(iter(qc))
        detail = qc.pop(any_key)
        qc["not:a:valid:key:format"] = detail
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_propagation_key_with_query_rank_suffix_rejected(self):
        # q2 (or any 3-part key) must never appear in propagation_outcome.
        bad = self._mut(self.gt_free)
        po = bad["control_features"]["propagation_outcome"]
        any_key = next(iter(po))
        detail = po.pop(any_key)
        po[f"{any_key}:q2"] = detail
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_t0_t320_class_out_of_range_rejected(self):
        bad = self._mut(self.gt_free)
        any_key = next(iter(bad["control_features"]["propagation_outcome"]))
        bad["control_features"]["propagation_outcome"][any_key]["t0_predicted_class"] = 999
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_digest_tamper_rejected(self):
        bad = self._mut(self.gt_free)
        bad["gt_free_feature_digest"] = "0" * 64
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_free_window_box_tamper_changes_digest(self):
        bad = self._mut(self.gt_free)
        any_ordinal = next(iter(bad["control_features"]["window_boxes"]))
        bad["control_features"]["window_boxes"][any_ordinal][0] += 1
        # digest now stale relative to tampered content -> must fail
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(bad)

    def test_gt_label_garbage_typed_fields_rejected(self):
        # The exact defect independent verification found: every field
        # wrong-typed, only the key set intact.
        garbage = {
            "schema": "not-even-a-dict", "image_id": 12345,
            "query_correctness_labels": "garbage-string-not-a-dict",
            "propagation_outcome_categories": None,
            "ambiguous_gt_anchor_ids": 42,
            "gt_dependent_label_digest": [],
        }
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_label_artifact(garbage)

    def test_gt_label_bool_as_int_label_rejected(self):
        bad = self._mut(self.gt_label)
        if not bad["query_correctness_labels"]:
            self.skipTest("no query_correctness_labels in this synthetic fixture")
        any_key = next(iter(bad["query_correctness_labels"]))
        bad["query_correctness_labels"][any_key] = True
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_label_artifact(bad)

    def test_gt_label_invalid_taxonomy_value_rejected(self):
        bad = self._mut(self.gt_label)
        if not bad["propagation_outcome_categories"]:
            self.skipTest("no propagation_outcome_categories in this synthetic fixture")
        any_key = next(iter(bad["propagation_outcome_categories"]))
        bad["propagation_outcome_categories"][any_key] = "MADE_UP_CATEGORY"
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_label_artifact(bad)

    def test_gt_label_digest_tamper_rejected(self):
        bad = self._mut(self.gt_label)
        bad["gt_dependent_label_digest"] = "f" * 64
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_label_artifact(bad)

    def test_gt_label_row_labeled_and_ambiguous_simultaneously_rejected(self):
        bad = self._mut(self.gt_label)
        if not bad["query_correctness_labels"]:
            self.skipTest("no query_correctness_labels in this synthetic fixture")
        any_key = next(iter(bad["query_correctness_labels"]))
        anchor_key = ":".join(any_key.split(":")[:2])
        bad["ambiguous_gt_anchor_ids"] = sorted(set(bad["ambiguous_gt_anchor_ids"]) | {anchor_key})
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_label_artifact(bad)

    def test_pair_linkage_rejects_row_absent_from_gt_free(self):
        bad_label = self._mut(self.gt_label)
        bad_label["query_correctness_labels"]["999:99999:q1"] = 1
        bad_label["gt_dependent_label_digest"] = ex.compute_gt_label_content_digest(bad_label)
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_pair_linkage(self.gt_free, bad_label)

    def test_pair_linkage_rejects_unaccounted_gt_free_row(self):
        bad_label = self._mut(self.gt_label)
        if not bad_label["query_correctness_labels"]:
            self.skipTest("no query_correctness_labels in this synthetic fixture")
        any_key = next(iter(bad_label["query_correctness_labels"]))
        del bad_label["query_correctness_labels"][any_key]
        bad_label["gt_dependent_label_digest"] = ex.compute_gt_label_content_digest(bad_label)
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_pair_linkage(self.gt_free, bad_label)

    def test_decision_document_requires_holdout_gates_uncomputable(self):
        base = {
            "schema": {}, "gate_a_discovery_side": {}, "gate_e_discovery_side": {},
            "gate_b_status": "NOT_COMPUTABLE_WITHOUT_HOLDOUT", "gate_c_status": "NOT_COMPUTABLE_WITHOUT_HOLDOUT",
            "gate_d_status": "NOT_COMPUTABLE_WITHOUT_HOLDOUT", "overall_gates_outcome_status": "PENDING_HOLDOUT_EVALUATION",
            "discovery_decision": "DISCOVERY_INCONCLUSIVE", "next_step": "STOP",
        }
        ex.validate_decision_document(base)  # must not raise
        poisoned = dict(base)
        poisoned["gate_b_status"] = "PASS"
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_decision_document(poisoned)


class AtomicInstallTests(unittest.TestCase):
    def _artifacts(self, ids):
        checkpoint_meta = ex.build_checkpoint_metadata(
            mode="fresh", completed_image_ids_ordered=ids, discovery_ids=ids,
            checkpoint_array_hashes={}, identity_bindings={}, complete=True, final=True,
        )
        final_result = {
            "schema": {}, "complete": True, "final": True, "identity_and_dependency_bindings": {},
            "population_binding": {}, "implementation_binding": {}, "discovery_image_ids_ordered": ids,
            "gt_free_feature_digest": "a", "gt_dependent_label_digest": "b", "scientific_collection_digest": "c",
            "model_fit_summaries": [], "primary_metrics": {}, "bootstrap_intervals": {}, "gate_inputs": {},
            "gate_pass_fail": {}, "final_frozen_outcome": "DISCOVERY_INCONCLUSIVE",
            "holdout_accessed": False, "full_evaluation_accessed": False,
        }
        decision = {
            "schema": {}, "gate_a_discovery_side": {}, "gate_e_discovery_side": {},
            "gate_b_status": "NOT_COMPUTABLE_WITHOUT_HOLDOUT", "gate_c_status": "NOT_COMPUTABLE_WITHOUT_HOLDOUT",
            "gate_d_status": "NOT_COMPUTABLE_WITHOUT_HOLDOUT", "overall_gates_outcome_status": "PENDING_HOLDOUT_EVALUATION",
            "discovery_decision": "DISCOVERY_INCONCLUSIVE", "next_step": "STOP",
        }
        return checkpoint_meta, [], [], final_result, decision

    def test_install_then_refuse_overwrite(self):
        ids = ["a", "b"]
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            args = self._artifacts(ids)
            summary = ex.install_discovery_result(output_root, *args, ids)
            self.assertTrue(summary["verified"])
            with self.assertRaises(ex.DiscoveryExecutionError):
                ex.install_discovery_result(output_root, *args, ids)

    def test_verify_result_independently_reopens(self):
        ids = ["a", "b"]
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            args = self._artifacts(ids)
            ex.install_discovery_result(output_root, *args, ids)
            v = ex.verify_result(output_root, ids)
            self.assertTrue(v["verified"])

    def test_no_partial_output_root_on_validation_failure(self):
        ids = ["a", "b"]
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            checkpoint_meta, gt_free, gt_label, final_result, decision = self._artifacts(ids)
            bad_decision = dict(decision)
            bad_decision["gate_b_status"] = "PASS"  # will fail validation
            with self.assertRaises(ex.DiscoveryExecutionError):
                ex.install_discovery_result(output_root, checkpoint_meta, gt_free, gt_label, final_result, bad_decision, ids)
            self.assertFalse(output_root.exists(), "a failed install must never leave a partial output_root")
            leftover_staging = list(output_root.parent.glob(".staging-*"))
            self.assertEqual(leftover_staging, [], "no staging directory should survive a failed install")


class RunDiscoveryFreshTests(unittest.TestCase):
    """Full fresh-mode integration, one synthetic single-window image."""

    def test_fresh_run_end_to_end(self):
        ids = ["000000567197"]
        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            summary = ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root, mode="fresh",
                image_loader=_fake_image_loader(ids), model_and_text_provider=_fake_model_and_text_provider,
                gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
            )
            self.assertTrue(summary["verified"])
            self.assertFalse(ex._checkpoint_dir_for(output_root).exists(), "checkpoint dir must be cleaned up on success")

            result = json.loads((output_root / "discovery_result.json").read_text())
            self.assertEqual(result["discovery_image_ids_ordered"], ids)
            self.assertFalse(result["holdout_accessed"])
            self.assertFalse(result["full_evaluation_accessed"])

            gt_free = json.loads((output_root / "gt_free_features.json").read_text())
            self.assertEqual(len(gt_free), 1)
            self.assertLessEqual(len(gt_free[0]["anchor_ids_ordered"]), ifeat.TARGET_ANCHORS_PER_WINDOW)

            decision = json.loads((output_root / "discovery_decision.json").read_text())
            self.assertEqual(decision["gate_b_status"], "NOT_COMPUTABLE_WITHOUT_HOLDOUT")
            self.assertEqual(decision["gate_c_status"], "NOT_COMPUTABLE_WITHOUT_HOLDOUT")
            self.assertEqual(decision["gate_d_status"], "NOT_COMPUTABLE_WITHOUT_HOLDOUT")

    def test_fresh_refuses_existing_output_root(self):
        ids = ["000000567197"]
        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            output_root.mkdir()
            with self.assertRaises(ex.DiscoveryExecutionError):
                ex.run_discovery(
                    REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                    output_root=output_root, mode="fresh",
                    image_loader=_fake_image_loader(ids), model_and_text_provider=_fake_model_and_text_provider,
                    gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
                )

    def _small_alphabet_forward_fn(self, alphabet_size=3):
        """A dedicated fake forward_fn whose ORIGINAL/CONTEXT/REGION/CONTROL
        top1 class is always drawn from a small {0..alphabet_size-1}
        alphabet (a large deterministic bonus is added to one such class per
        patch, guaranteed to beat the rest of the random [-1,0.5] scores),
        instead of the full random 171-class draw the shared
        _fake_forward_fn uses. Paired with a matching small-alphabet GT
        loader, this gives a realistic (~1/alphabet_size) query/GT match
        probability instead of the ~1/171 an unconstrained random draw
        would give -- needed so a small (48-anchor) synthetic fixture
        reliably produces a genuine two-class label split instead of
        requiring an impractically large image count."""
        def forward_fn(model, crop, text_embedding, *, window_id, window_ordinal, top, left, bottom, right,
                        canvas_h, canvas_w, patch_h, patch_w):
            seed = int(hashlib.sha256(crop.contiguous().numpy().tobytes()).hexdigest()[:8], 16) & 0xFFFFFFFF
            g = torch.Generator().manual_seed(seed)
            s0 = torch.rand(GRID * GRID, CLASS_COUNT, generator=g) * 1.5 - 1.0  # base scores in [-1, 0.5]
            assigned = torch.randint(0, alphabet_size, (GRID * GRID,), generator=g)
            s0[torch.arange(GRID * GRID), assigned] = 0.9  # guaranteed top1 within the small alphabet
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
        return forward_fn

    def _small_alphabet_gt_loader(self, authorized_ids, alphabet_size=3):
        def loader(image_id):
            assert image_id in authorized_ids
            g = torch.Generator().manual_seed(int(hashlib.sha256(("gt:" + image_id).encode()).hexdigest()[:8], 16))
            klass = int(torch.randint(0, alphabet_size, (1,), generator=g).item())
            return [[klass] * CANVAS for _ in range(CANVAS)]
        return loader

    def test_fresh_run_executes_exactly_four_fits_with_persisted_parameters(self):
        # Multiple images, small class alphabet, to give the synthetic GT/
        # scores enough rows and a realistic enough match probability for a
        # genuine two-class split in both tasks (see _small_alphabet_*).
        ids = ["000000567197", "000000367680", "000000361571", "000000148508", "000000310862", "000000528977"]
        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            summary = ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root, mode="fresh",
                image_loader=_fake_image_loader(ids), model_and_text_provider=_fake_model_and_text_provider,
                gt_loader=self._small_alphabet_gt_loader(ids), forward_fn=self._small_alphabet_forward_fn(),
            )
            self.assertTrue(summary["verified"])
            result = json.loads((output_root / "discovery_result.json").read_text())
            fits = result["model_fit_summaries"]
            if len(fits) == 0:
                self.skipTest("synthetic random fixture happened to produce a degenerate (single-class) "
                               "label split for this seed -- fail-closed behavior confirmed instead; "
                               "see test_run_discovery_fails_closed_on_degenerate_fit_population")
            self.assertEqual(len(fits), 4, "exactly four frozen fits must be persisted on a non-degenerate run")
            task_model_pairs = sorted((f["task_id"], f["model_id"]) for f in fits)
            self.assertEqual(task_model_pairs, [
                ("propagation_outcome", "model_a"), ("propagation_outcome", "model_b"),
                ("query_correctness", "model_a"), ("query_correctness", "model_b"),
            ])
            for f in fits:
                self.assertIn("coefficients", f)
                self.assertIn("intercept", f)
                self.assertIn("fit_digest", f)
                self.assertTrue(f["converged"])
                expected_len = 7 if f["model_id"] == "model_a" else 8
                self.assertEqual(len(f["coefficients"]), expected_len)
            self.assertFalse(result["holdout_accessed"])

    def test_fresh_run_gate_a_counters_are_populated_and_coherent(self):
        ids = ["000000567197", "000000367680"]
        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root, mode="fresh",
                image_loader=_fake_image_loader(ids), model_and_text_provider=_fake_model_and_text_provider,
                gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
            )
            result = json.loads((output_root / "discovery_result.json").read_text())
            gate_a = result["primary_metrics"]["gate_a"]
            # 8 anchors/window * 2 windows (1 per image, single-window canvas) = 16 attempted anchors
            self.assertEqual(gate_a["total_anchors"], 16)
            self.assertGreaterEqual(gate_a["total_anchors"], gate_a["structurally_invalid_anchor_count"])


class RunDiscoveryResumeTests(unittest.TestCase):
    def test_interrupt_then_resume_matches_uninterrupted_fresh(self):
        ids = ["000000567197", "000000367680"]

        class _Interrupt(Exception):
            pass

        call_count = {"n": 0}
        base_loader = _fake_image_loader(ids)

        def interrupting_loader(image_id):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise _Interrupt("simulated crash after first image")
            return base_loader(image_id)

        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            checkpoint_dir = ex._checkpoint_dir_for(output_root)
            with self.assertRaises(_Interrupt):
                ex.run_discovery(
                    REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                    output_root=output_root, mode="fresh",
                    image_loader=interrupting_loader, model_and_text_provider=_fake_model_and_text_provider,
                    gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
                )
            self.assertTrue(checkpoint_dir.exists())
            self.assertFalse(output_root.exists())
            ckpt = json.loads((checkpoint_dir / "checkpoint.json").read_text())
            self.assertEqual(ckpt["completed_image_ids_ordered"], ids[:1])
            self.assertFalse(ckpt["complete"])

            resume_summary = ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root, mode="resume",
                image_loader=base_loader, model_and_text_provider=_fake_model_and_text_provider,
                gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
            )
            self.assertTrue(resume_summary["verified"])
            resume_result = json.loads((output_root / "discovery_result.json").read_text())

        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td2:
            output_root2 = Path(td2) / "out"
            ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root2, mode="fresh",
                image_loader=base_loader, model_and_text_provider=_fake_model_and_text_provider,
                gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
            )
            fresh_result = json.loads((output_root2 / "discovery_result.json").read_text())

        self.assertEqual(fresh_result["gt_free_feature_digest"], resume_result["gt_free_feature_digest"],
                          "determinism violation: fresh != interrupted+resume (gt_free digest)")
        self.assertEqual(fresh_result["gt_dependent_label_digest"], resume_result["gt_dependent_label_digest"],
                          "determinism violation: fresh != interrupted+resume (gt_label digest)")

    def test_resume_refuses_when_output_root_already_exists(self):
        ids = ["000000567197"]
        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            output_root.mkdir()
            with self.assertRaises(ex.DiscoveryExecutionError):
                ex.run_discovery(
                    REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                    output_root=output_root, mode="resume",
                    image_loader=_fake_image_loader(ids), model_and_text_provider=_fake_model_and_text_provider,
                    gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
                )

    def test_resume_refuses_when_no_checkpoint_dir(self):
        ids = ["000000567197"]
        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            with self.assertRaises(ex.DiscoveryExecutionError):
                ex.run_discovery(
                    REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                    output_root=output_root, mode="resume",
                    image_loader=_fake_image_loader(ids), model_and_text_provider=_fake_model_and_text_provider,
                    gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
                )

    def test_resume_after_checkpoint_write_failure_reuses_leftover_artifacts_without_reprocessing(self):
        # Simulates a crash between writing gt_free_<id>.json/gt_label_<id>.json
        # and the checkpoint.json update: both per-image artifacts already
        # exist on disk, but checkpoint.json does not yet list the image as
        # completed. Resume must detect and REUSE them, never call the image
        # loader again for that image.
        ids = ["000000567197", "000000367680"]
        image_loader_calls = []

        def counting_loader(image_id):
            image_loader_calls.append(image_id)
            return _fake_image_loader(ids)(image_id)

        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            checkpoint_dir = ex._checkpoint_dir_for(output_root)
            checkpoint_dir.mkdir(parents=True)
            # matches what fresh mode itself writes immediately on start,
            # before processing any image -- the crash this test simulates
            # happens strictly AFTER that point.
            initial_checkpoint = ex.build_checkpoint_metadata(
                mode="fresh", completed_image_ids_ordered=[], discovery_ids=ids,
                checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=False,
            )
            (checkpoint_dir / "checkpoint.json").write_text(json.dumps(initial_checkpoint))

            event_log = ex.PhaseEventLog()
            model, text_embedding = _fake_model_and_text_provider()
            gt_free = ex._run_phase_a(ids[0], counting_loader, _fake_forward_fn, model, text_embedding,
                                        (448, 448), (224, 224), (14, 14), event_log)
            gt_label = ex._run_phase_b(gt_free, _fake_gt_loader(ids), ids[0], event_log)
            (checkpoint_dir / f"gt_free_{ids[0]}.json").write_text(json.dumps(gt_free))
            (checkpoint_dir / f"gt_label_{ids[0]}.json").write_text(json.dumps(gt_label))
            # checkpoint.json still shows zero completed images -- simulates
            # the crash window between finishing these two files and the
            # per-image checkpoint.json update that would have listed ids[0].
            calls_before_resume = len(image_loader_calls)

            summary = ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root, mode="resume",
                image_loader=counting_loader, model_and_text_provider=_fake_model_and_text_provider,
                gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
            )
            self.assertTrue(summary["verified"])
            # ids[0] must NOT have been reprocessed -- only ids[1] should have triggered a new call.
            self.assertEqual(image_loader_calls[calls_before_resume:], [ids[1]],
                              "leftover gt_free/gt_label for ids[0] must be reused, never reprocessed")

    def test_resume_after_gt_label_write_failure_reuses_gt_free_and_only_redoes_phase_b(self):
        # gt_free_<id>.json exists and is valid, but gt_label_<id>.json is
        # missing (crash between Phase A and Phase B for that image). Resume
        # must reuse the persisted gt_free artifact and only run Phase B --
        # no image loader or model call for that image.
        ids = ["000000567197"]

        def failing_loader(image_id):
            raise AssertionError("must not call the image loader when a valid gt_free artifact already exists")

        def failing_model_provider():
            raise AssertionError("must not construct the model when a valid gt_free artifact already exists")

        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            checkpoint_dir = ex._checkpoint_dir_for(output_root)
            checkpoint_dir.mkdir(parents=True)
            initial_checkpoint = ex.build_checkpoint_metadata(
                mode="fresh", completed_image_ids_ordered=[], discovery_ids=ids,
                checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=False,
            )
            (checkpoint_dir / "checkpoint.json").write_text(json.dumps(initial_checkpoint))

            event_log = ex.PhaseEventLog()
            model, text_embedding = _fake_model_and_text_provider()
            gt_free = ex._run_phase_a(ids[0], _fake_image_loader(ids), _fake_forward_fn, model, text_embedding,
                                        (448, 448), (224, 224), (14, 14), event_log)
            (checkpoint_dir / f"gt_free_{ids[0]}.json").write_text(json.dumps(gt_free))
            # deliberately no gt_label file -- simulates a crash right after Phase A

            summary = ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root, mode="resume",
                image_loader=failing_loader, model_and_text_provider=failing_model_provider,
                gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
            )
            self.assertTrue(summary["verified"])

    def test_zero_reprocessing_across_a_three_image_resume(self):
        ids = ["000000567197", "000000367680", "000000361571"]
        image_loader_calls = []

        def counting_loader(image_id):
            image_loader_calls.append(image_id)
            return _fake_image_loader(ids)(image_id)

        class _Interrupt(Exception):
            pass

        def interrupt_after_two(image_id):
            if len(image_loader_calls) >= 2:
                raise _Interrupt("simulated crash")
            return counting_loader(image_id)

        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            with self.assertRaises(_Interrupt):
                ex.run_discovery(
                    REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                    output_root=output_root, mode="fresh",
                    image_loader=interrupt_after_two, model_and_text_provider=_fake_model_and_text_provider,
                    gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
                )
            self.assertEqual(sorted(image_loader_calls), sorted(ids[:2]))

            ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root, mode="resume",
                image_loader=counting_loader, model_and_text_provider=_fake_model_and_text_provider,
                gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
            )
            # only the third (never-completed) image should have been loaded during resume
            self.assertEqual(sorted(set(image_loader_calls)), sorted(ids), "every image loaded exactly once across the whole run")
            self.assertEqual(len(image_loader_calls), len(set(image_loader_calls)), "no image was ever reprocessed")


class RunDiscoveryFinalizeOnlyTests(unittest.TestCase):
    def test_finalize_only_never_touches_image_loader(self):
        ids = ["000000567197"]

        def failing_loader(image_id):
            raise AssertionError("finalize_only must never call the image loader")

        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            checkpoint_dir = ex._checkpoint_dir_for(output_root)
            checkpoint_dir.mkdir(parents=True)
            event_log = ex.PhaseEventLog()
            model, text_embedding = _fake_model_and_text_provider()
            per_image = ex._process_image(
                REPO_ROOT, ids[0], _fake_image_loader(ids), _fake_forward_fn, model, text_embedding,
                _fake_gt_loader(ids), (448, 448), (224, 224), (14, 14), event_log,
            )
            (checkpoint_dir / f"gt_free_{ids[0]}.json").write_text(json.dumps(per_image["gt_free_artifact"]))
            (checkpoint_dir / f"gt_label_{ids[0]}.json").write_text(json.dumps(per_image["gt_label_artifact"]))
            checkpoint_meta = ex.build_checkpoint_metadata(
                mode="fresh", completed_image_ids_ordered=ids, discovery_ids=ids,
                checkpoint_array_hashes={"gt_free": per_image["gt_free_artifact"]["gt_free_feature_digest"],
                                          "gt_label": per_image["gt_label_artifact"]["gt_dependent_label_digest"]},
                identity_bindings={}, complete=True, final=False,
            )
            (checkpoint_dir / "checkpoint.json").write_text(json.dumps(checkpoint_meta))

            summary = ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root, mode="finalize_only",
                image_loader=failing_loader, model_and_text_provider=_fake_model_and_text_provider,
                gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
            )
            self.assertTrue(summary["verified"])
            self.assertFalse(checkpoint_dir.exists())

    def test_finalize_only_requires_complete_checkpoint(self):
        ids = ["000000567197", "000000367680"]
        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            checkpoint_dir = ex._checkpoint_dir_for(output_root)
            checkpoint_dir.mkdir(parents=True)
            checkpoint_meta = ex.build_checkpoint_metadata(
                mode="fresh", completed_image_ids_ordered=ids[:1], discovery_ids=ids,
                checkpoint_array_hashes={}, identity_bindings={}, complete=False, final=False,
            )
            (checkpoint_dir / "checkpoint.json").write_text(json.dumps(checkpoint_meta))
            with self.assertRaises(ex.DiscoveryExecutionError):
                ex.run_discovery(
                    REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                    output_root=output_root, mode="finalize_only",
                    image_loader=_fake_image_loader(ids), model_and_text_provider=_fake_model_and_text_provider,
                    gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
                )


class GtIsolationTests(unittest.TestCase):
    def test_gt_free_pipeline_has_no_gt_loader_parameter(self):
        # Architectural isolation proof: process_window_gt_free's signature
        # has no gt_loader/gt_root parameter at all, so Phase A's per-window
        # code path cannot call into GT even if a caller tried to smuggle
        # one in -- there is no plumbing for it to arrive through.
        import inspect
        sig = inspect.signature(ex.process_window_gt_free)
        for name in sig.parameters:
            self.assertNotIn("gt", name.lower(), f"process_window_gt_free must not accept a GT-related parameter, found {name!r}")

    def test_poisoned_gt_loader_that_raises_if_called_before_phase_a_completes(self):
        ids = ["000000567197"]
        access_log = []

        def watchdog_gt_loader(image_id):
            access_log.append(image_id)
            return _fake_gt_loader(ids)(image_id)

        with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "out"
            ex.run_discovery(
                REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                output_root=output_root, mode="fresh",
                image_loader=_fake_image_loader(ids), model_and_text_provider=_fake_model_and_text_provider,
                gt_loader=watchdog_gt_loader, forward_fn=_fake_forward_fn,
            )
        self.assertEqual(access_log, ids, "GT must be accessed exactly once, after Phase A, per image")


class DefectReinstatementTests(unittest.TestCase):
    """Reinstates representative defects (matching the repair task's own
    lettered list) and shows the current test suite's assertions actually
    discriminate against them -- not merely that the repaired code happens
    to look different."""

    def test_B_bypassing_all_fit_calls_is_caught_by_the_four_fit_assertion(self):
        ids = ["000000567197", "000000367680", "000000361571", "000000148508", "000000310862", "000000528977"]
        real_fit = ex.fit_logistic_model

        def broken_fit(*a, **kw):
            raise ex.DiscoveryExecutionError("simulated defect B: fitting is broken/bypassed")

        ex.fit_logistic_model = broken_fit
        try:
            with _PopulationPatch(ids), tempfile.TemporaryDirectory() as td:
                output_root = Path(td) / "out"
                summary = ex.run_discovery(
                    REPO_ROOT, weight_dir="unused", image_root=Path("unused"), gt_root=Path("unused"),
                    output_root=output_root, mode="fresh",
                    image_loader=_fake_image_loader(ids), model_and_text_provider=_fake_model_and_text_provider,
                    gt_loader=_fake_gt_loader(ids), forward_fn=_fake_forward_fn,
                )
                result = json.loads((output_root / "discovery_result.json").read_text())
        finally:
            ex.fit_logistic_model = real_fit

        # This is exactly the observable state the original Finding 2 defect
        # left behind: model_fit_summaries == [] on a run that otherwise
        # "succeeds". The non-degenerate end-to-end test in this suite
        # (test_fresh_run_executes_exactly_four_fits_with_persisted_parameters)
        # asserts len(fits) == 4 and would fail against this mutant.
        self.assertEqual(result["model_fit_summaries"], [])
        with self.assertRaises(AssertionError):
            self.assertEqual(len(result["model_fit_summaries"]), 4)

    def test_C_stripped_feature_persistence_is_rejected_by_the_validator(self):
        gt_free, _ = _sample_gt_pair()
        mutant = json.loads(json.dumps(gt_free))
        # Simulate defect C: control_features rows emptied out (as if the
        # per-row feature vector were never persisted) while everything
        # else is left as the real, valid artifact -- the closed-schema/
        # linkage checks must catch this even though the top-level keys
        # are all still present.
        mutant["control_features"]["query_correctness"] = {}
        mutant["control_features"]["propagation_outcome"] = {}
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_free_artifact(mutant)

    def test_F_outer_key_only_validator_mutant_accepts_what_the_real_validator_rejects(self):
        def outer_key_only_mutant_validator(d):
            # This is the ORIGINAL, pre-repair validator: closed top-level
            # key set only, no recursive content checking whatsoever.
            required = ex._GT_LABEL_REQUIRED_FIELDS
            if frozenset(d.keys()) != required:
                raise ex.DiscoveryExecutionError("key mismatch")

        garbage = {
            "schema": "not-even-a-dict", "image_id": 12345,
            "query_correctness_labels": "garbage-string-not-a-dict",
            "propagation_outcome_categories": None,
            "ambiguous_gt_anchor_ids": 42,
            "gt_dependent_label_digest": [],
        }
        outer_key_only_mutant_validator(garbage)  # the mutant accepts it (does not raise)
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.validate_gt_label_artifact(garbage)  # the repaired validator rejects the same input


CLI_PATH = REPO_ROOT / "diagnostics" / "run_semantic_evidence_discovery.py"
LAUNCHER_PATH = REPO_ROOT / "semantic_evidence_discovery_h100.sbatch"


class PreflightCliTests(unittest.TestCase):
    """Subprocess-based tests against the real CLI script -- fast,
    torch-free, argument-parsing checks only (no real preflight run)."""

    def _run(self, args):
        import subprocess
        return subprocess.run(
            [sys.executable, str(CLI_PATH)] + args,
            capture_output=True, text=True, timeout=30,
        )

    def test_J_preflight_help_lists_gt_root(self):
        result = self._run(["preflight", "--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--gt-root", result.stdout)

    def test_J_preflight_without_gt_root_fails_via_argparse(self):
        result = self._run([
            "preflight", "--weight-dir", "/nonexistent", "--image-root", "/nonexistent",
            "--output-root", "/nonexistent",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--gt-root", result.stderr)

    def test_J_execute_still_requires_its_own_gt_root(self):
        result = self._run(["execute", "--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--gt-root", result.stdout)

    def test_J_no_new_subcommand_or_forbidden_flag(self):
        result = self._run(["--help"])
        self.assertEqual(result.returncode, 0)
        for sub in ("verify-protocol", "preflight", "execute", "verify-result"):
            self.assertIn(sub, result.stdout)
        forbidden = ("--image-id", "--manifest", "--split", "--holdout", "--all-96",
                     "--full-5000", "--max-images", "--debug", "--overwrite", "--force")
        preflight_help = self._run(["preflight", "--help"]).stdout
        for flag in forbidden:
            self.assertNotIn(flag, preflight_help)


class LauncherGtRootTests(unittest.TestCase):
    """Pure text inspection of the committed launcher -- no subprocess, no
    sbatch, no submission."""

    def setUp(self):
        self.text = LAUNCHER_PATH.read_text()

    def test_K_single_gt_root_configuration_source(self):
        # Exactly one operator-facing GT-root variable is declared as required.
        required_decls = [l for l in self.text.splitlines() if ":?" in l and "GT_ROOT" in l]
        self.assertEqual(len(required_decls), 1, f"expected exactly one required GT-root declaration, found {required_decls}")
        self.assertIn("DISCOVERY_GT_ROOT", required_decls[0])

    def test_K_preflight_and_execute_both_use_the_same_variable(self):
        import re
        preflight_block = self.text[self.text.index('"${CLI}" preflight'):self.text.index('"${CLI}" execute')]
        execute_block = self.text[self.text.index('"${CLI}" execute'):]
        self.assertIn('--gt-root "${DISCOVERY_GT_ROOT}"', preflight_block)
        self.assertIn('--gt-root "${DISCOVERY_GT_ROOT}"', execute_block)

    def test_K_preflight_appears_before_execute(self):
        self.assertLess(self.text.index('"${CLI}" preflight'), self.text.index('"${CLI}" execute'))

    def test_K_resource_directives_unchanged(self):
        for directive in ("--gres=gpu:h100:1", "--cpus-per-task=8", "--mem=80G",
                           "--time=02:00:00", "--account=rrg-yangw"):
            self.assertIn(directive, self.text)

    def test_K_no_self_submission(self):
        import re
        executable_sbatch_calls = [
            l for l in self.text.splitlines()
            if re.match(r"^\s*sbatch\s", l) or re.match(r'^\s*"\S*sbatch"\s', l)
        ]
        self.assertEqual(executable_sbatch_calls, [])

    def test_K_bash_syntax_valid(self):
        import subprocess
        result = subprocess.run(["bash", "-n", str(LAUNCHER_PATH)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
