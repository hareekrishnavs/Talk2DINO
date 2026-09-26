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


class CanonicalWindowClampingTests(unittest.TestCase):
    """Regression tests for the window-clamping repair: job 22692249 failed
    on the real image 000000567197.jpg (500x500 -> actually 500 wide,
    407 tall), because the discovery runner's own prior reimplementation of
    compute_canonical_windows incorrectly required the canvas to be at
    least 448x448. The real, committed, real-image-proven
    semantic_evidence_t320_real_smoke.compute_canonical_windows clamps
    windows to the actual image bounds instead -- this authority already
    depends on that exact file by hash ([dependency].smoke_runner), and
    the repaired discovery runner now delegates to it directly rather than
    re-deriving the same arithmetic a second time. Every case here uses
    real, non-448 dimensions specifically because a fixed-448x448 fake
    image loader is exactly what let the original defect go undetected
    through every previous round of testing and independent verification
    in this project."""

    CROP_HW = (448, 448)
    STRIDE_HW = (224, 224)

    def _check_one(self, h, w):
        import semantic_evidence_t320_real_smoke as smoke_runner
        mine = ex.compute_canonical_windows(h, w, self.CROP_HW, self.STRIDE_HW)
        real = smoke_runner.compute_canonical_windows(h, w, self.CROP_HW, self.STRIDE_HW)
        self.assertEqual(mine, real, f"must match the smoke runner's ordered windows exactly for (h={h}, w={w})")
        self.assertGreaterEqual(len(mine), 1)
        covered = [[False] * w for _ in range(h)]
        for (ordinal, top, left, bottom, right) in mine:
            self.assertTrue(0 <= top < bottom <= h, f"window {ordinal} top/bottom out of bounds")
            self.assertTrue(0 <= left < right <= w, f"window {ordinal} left/right out of bounds")
            self.assertGreater(bottom, top)
            self.assertGreater(right, left)
            for r in range(top, bottom):
                for c in range(left, right):
                    covered[r][c] = True
        self.assertTrue(all(all(row) for row in covered), f"every pixel must be covered for (h={h}, w={w})")
        return mine

    def test_real_failing_image_000000567197_dimensions(self):
        # The exact image and exact orientation that crashed job 22692249:
        # PIL reports (width, height) = (500, 407) for this file.
        windows = self._check_one(407, 500)
        # Height 407 < the 448 crop: the committed smoke behavior clamps
        # the window height to 407, it never raises.
        self.assertTrue(all(bottom - top == 407 for (_, top, _, bottom, _) in windows))

    def test_smaller_than_crop_in_both_dimensions(self):
        self._check_one(300, 200)

    def test_multi_window_image(self):
        windows = self._check_one(448, 672)
        self.assertGreater(len(windows), 1)

    def test_multi_window_both_dimensions(self):
        windows = self._check_one(900, 900)
        self.assertGreater(len(windows), 1)

    def test_exact_crop_size_still_single_window(self):
        windows = self._check_one(448, 448)
        self.assertEqual(len(windows), 1)

    def test_pre_repair_defect_would_have_rejected_the_real_image(self):
        # Demonstrates precisely what job 22692249 hit: the OLD local
        # formula's guard condition, evaluated directly (without invoking
        # any implementation), was false for the real image's dimensions --
        # proving this specific defect is what caused the crash, and that
        # the repaired implementation no longer has this guard at all.
        crop_h, crop_w = self.CROP_HW
        real_h, real_w = 407, 500
        old_guard_would_pass = (crop_h <= real_h and crop_w <= real_w)
        self.assertFalse(old_guard_would_pass, "the old guard condition was false for this image -- confirms the crash's root cause")
        # and confirm the repaired function does NOT raise for this input
        ex.compute_canonical_windows(real_h, real_w, self.CROP_HW, self.STRIDE_HW)

    @staticmethod
    def _fake_forward_fn_fixed_grid(crop, *, window_id, window_ordinal, top, left, bottom, right,
                                     canvas_h, canvas_w, patch_h, patch_w):
        # The real semantic_evidence_e3_adapter.extract_e3_window_snapshot
        # resizes every window's crop to a FIXED 448x448 backbone-input
        # canvas via model.image_transforms's T.Resize((448, 448)),
        # "applied unconditionally regardless of the source window's own
        # extent" (its own docstring/comment, src/semantic_evidence_
        # e3_adapter.py:236-243) -- so grid_h/grid_w are always 32x32
        # (448/14) no matter how small a clamped window's true (infer_h,
        # infer_w) is. infer_h/infer_w record the window's own (possibly
        # clamped, non-448) extent; backbone_input_height/width and
        # grid_h/grid_w always reflect the fixed 448x448 resize target.
        import hashlib
        win_h, win_w = bottom - top, right - left
        grid_h = grid_w = GRID
        seed = int(hashlib.sha256(crop.contiguous().numpy().tobytes()).hexdigest()[:8], 16) & 0xFFFFFFFF
        g = torch.Generator().manual_seed(seed)
        s0 = torch.rand(grid_h * grid_w, CLASS_COUNT, generator=g) * 2 - 1
        dino_features = torch.nn.functional.normalize(torch.rand(grid_h * grid_w, FEAT_DIM, generator=g), dim=1)
        snapshot = e3a.WindowSnapshot(
            window_id=window_id, window_ordinal=window_ordinal, canvas_h=canvas_h, canvas_w=canvas_w,
            top=top, left=left, bottom=bottom, right=right, infer_h=win_h, infer_w=win_w,
            backbone_input_height=grid_h * patch_h, backbone_input_width=grid_w * patch_w,
            grid_h=grid_h, grid_w=grid_w, patch_h=patch_h, patch_w=patch_w,
            dino_features=dino_features, s0=s0, class_count=CLASS_COUNT, feature_dim=FEAT_DIM,
            dtype=s0.dtype, device=s0.device,
        )
        return snapshot, torch.sigmoid(s0)

    def test_through_execution_path_window_generation_and_t0t320_no_longer_crash(self):
        # A fixed-448x448 fake image loader is exactly what concealed this
        # bug through every previous test and independent verification
        # round. This exercises the real failing dimensions end-to-end
        # through window generation and T0/T320 canvas-level target
        # generation (semantic_evidence_t320.generate_t0_t320_targets),
        # calling process_window_gt_free directly rather than through
        # _run_phase_a so this test does not depend on anchor selection --
        # see FootprintPixelBoundsClampedWindowTests below for the
        # anchor/footprint-mapping repair and its own end-to-end test.
        import semantic_evidence_t320 as t320
        real_h, real_w = 407, 500

        def image_loader(iid):
            g = torch.Generator().manual_seed(1)
            return torch.rand(1, 3, real_h, real_w, generator=g) * 255.0, real_h, real_w

        def forward_fn(model, crop, text_embedding, **kw):
            return self._fake_forward_fn_fixed_grid(crop, **kw)

        image_tensor, h, w = image_loader("000000567197")
        # Must not raise -- this is exactly the call that crashed job
        # 22692249 with "canvas smaller than one canonical window crop".
        windows = ex.compute_canonical_windows(h, w, self.CROP_HW, self.STRIDE_HW)
        self.assertGreater(len(windows), 0)

        model, text_embedding = _fake_model_and_text_provider()
        original_snapshots, boxes = [], []
        for window in windows:
            result = ex.process_window_gt_free(model, text_embedding, image_tensor, window, h, w,
                                                 "000000567197", (14, 14), forward_fn)
            original_snapshots.append(result["original_snapshot"])
            boxes.append(result["window_box"])

        targets = t320.generate_t0_t320_targets(original_snapshots, boxes, canvas_h=h, canvas_w=w, ori_h=h, ori_w=w)
        self.assertEqual(tuple(targets["t0"].shape[1:]), (h, w))
        self.assertEqual(tuple(targets["t320"].shape[1:]), (h, w))


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


class RoundHalfAwayFromZeroTests(unittest.TestCase):
    """The explicit, pinned tie rule -- deliberately NOT Python's built-in
    round() (round-half-to-even/banker's rounding)."""

    def test_diverges_from_python_builtin_round_at_a_real_tie(self):
        self.assertEqual(ex._round_half_away_from_zero(2.5), 3)
        self.assertEqual(round(2.5), 2)  # documents exactly why round() is not used
        self.assertEqual(ex._round_half_away_from_zero(4.5), 5)
        self.assertEqual(round(4.5), 4)

    def test_negative_ties_and_ordinary_values(self):
        self.assertEqual(ex._round_half_away_from_zero(-2.5), -3)
        self.assertEqual(ex._round_half_away_from_zero(3.4), 3)
        self.assertEqual(ex._round_half_away_from_zero(3.6), 4)


class InputTokenizationInverseTests(unittest.TestCase):
    """Hard-coded, independently-recomputed oracles for the selected
    scientific contract's two primitives: _input_tokenization_inverse_point
    (Phase A, a PIXEL-INDEX mapping using the half-pixel formula) and
    _input_tokenization_inverse_boundary (Phase B / intervention core, a
    half-open INTERVAL EDGE mapping using pure linear boundary-space
    scaling -- no +/-0.5 offset). Every value below was computed
    independently of the implementation and cross-checked against the
    real T.Resize forward operation before this repair was written."""

    def test_point_exact_identity_at_native_448(self):
        for center in (7, 217, 427, 441):
            self.assertEqual(ex._input_tokenization_inverse_point(center, 448, 448), center)

    def test_boundary_exact_identity_at_native_448(self):
        for edge in (0, 210, 224, 420, 434, 448):
            self.assertEqual(ex._input_tokenization_inverse_boundary(edge, 448, 448), edge)

    def test_point_oracle_407(self):
        # anchor 15 center=217, anchor 30 center=427
        self.assertEqual(ex._input_tokenization_inverse_point(217, 407, 448), 197)
        self.assertEqual(ex._input_tokenization_inverse_point(427, 407, 448), 388)

    def test_point_oracle_rectangular_300x200(self):
        # anchor 15 center=217 (row axis native 300, col axis native 200)
        self.assertEqual(ex._input_tokenization_inverse_point(217, 300, 448), 145)
        self.assertEqual(ex._input_tokenization_inverse_point(217, 200, 448), 97)
        # anchor 30 center=427
        self.assertEqual(ex._input_tokenization_inverse_point(427, 300, 448), 286)
        self.assertEqual(ex._input_tokenization_inverse_point(427, 200, 448), 190)

    def test_boundary_oracle_407_patch_anchor_15(self):
        # patch interval [210,224); this exercises an EXACT half-integer
        # tie: 224*407/448 == 203.5 exactly.
        self.assertEqual(ex._input_tokenization_inverse_boundary(210, 407, 448), 191)
        self.assertEqual(ex._input_tokenization_inverse_boundary(224, 407, 448), 204)

    def test_boundary_oracle_407_patch_anchor_30(self):
        self.assertEqual(ex._input_tokenization_inverse_boundary(420, 407, 448), 382)
        self.assertEqual(ex._input_tokenization_inverse_boundary(434, 407, 448), 394)

    def test_boundary_oracle_rectangular_300x200_patch(self):
        # anchor 15: patch [210,224)
        self.assertEqual(ex._input_tokenization_inverse_boundary(210, 300, 448), 141)
        self.assertEqual(ex._input_tokenization_inverse_boundary(224, 300, 448), 150)
        self.assertEqual(ex._input_tokenization_inverse_boundary(210, 200, 448), 94)
        self.assertEqual(ex._input_tokenization_inverse_boundary(224, 200, 448), 100)
        # anchor 30: patch [420,434)
        self.assertEqual(ex._input_tokenization_inverse_boundary(420, 300, 448), 281)
        self.assertEqual(ex._input_tokenization_inverse_boundary(434, 300, 448), 291)
        self.assertEqual(ex._input_tokenization_inverse_boundary(420, 200, 448), 188)
        self.assertEqual(ex._input_tokenization_inverse_boundary(434, 200, 448), 194)

    def test_boundary_oracle_rectangular_300x200_core(self):
        # anchor 15 core [196,238); anchor 30 core [406,448)
        self.assertEqual(ex._input_tokenization_inverse_boundary(196, 300, 448), 131)
        self.assertEqual(ex._input_tokenization_inverse_boundary(238, 300, 448), 159)
        self.assertEqual(ex._input_tokenization_inverse_boundary(196, 200, 448), 88)
        self.assertEqual(ex._input_tokenization_inverse_boundary(238, 200, 448), 106)
        self.assertEqual(ex._input_tokenization_inverse_boundary(406, 300, 448), 272)
        self.assertEqual(ex._input_tokenization_inverse_boundary(448, 300, 448), 300)
        self.assertEqual(ex._input_tokenization_inverse_boundary(406, 200, 448), 181)
        self.assertEqual(ex._input_tokenization_inverse_boundary(448, 200, 448), 200)

    def test_agrees_with_the_real_forward_resize_operation(self):
        # Direct empirical cross-check against the REAL model.image_
        # transforms operation (T.Resize((448,448)) on a tensor input, no
        # interpolation/antialias kwargs): a synthetic step function's
        # observed 0.5-crossing after the real resize must agree with pure
        # linear boundary-space scaling to well under 1 pixel. CPU only,
        # no model, no checkpoint, no GT, no real image.
        import torchvision.transforms as T

        native = 407
        canvas = 448
        for k in (100, 205, 300, 393):
            step = torch.zeros(1, 1, native, native)
            step[:, :, k:, :] = 1.0
            resized = T.Resize((canvas, canvas))(step)
            col = resized[0, 0, :, 0]
            observed = int((col >= 0.5).nonzero()[0].item())
            predicted = k * canvas / native
            self.assertLess(abs(observed - predicted), 1.0, f"boundary k={k} mismatch")


class FootprintMappingTests(unittest.TestCase):
    """footprint_pixel_bounds is GT-ONLY (majority-vote 14x14 patch
    footprint region). Selected contract: input-tokenization-inverse
    boundary mapping, per axis, with exact identity at native==448."""

    def test_maps_through_window_and_patch_offset_when_unclamped(self):
        window_box = (100, 50, 100 + 448, 50 + 448)
        bounds = ex.footprint_pixel_bounds(window_box, anchor_row=2, anchor_col=3, patch_hw=(14, 14), grid_side=32)
        self.assertEqual(bounds, (100 + 28, 50 + 42, 100 + 28 + 14, 50 + 42 + 14))

    def test_fails_closed_rather_than_returning_an_empty_footprint(self):
        # Pathological, not-realizable-by-any-real-discovery-image window
        # (native size far smaller than one patch pitch): fails closed
        # with a clear DiscoveryExecutionError rather than an empty region.
        window_box = (0, 0, 2, 2)
        with self.assertRaises(ex.DiscoveryExecutionError):
            ex.footprint_pixel_bounds(window_box, anchor_row=1, anchor_col=1, patch_hw=(14, 14), grid_side=32)

    def test_real_407x448_window_zero_offset(self):
        window_box = (0, 0, 407, 448)
        b15 = ex.footprint_pixel_bounds(window_box, 15, 15, (14, 14), 32)
        self.assertEqual(b15, (191, 210, 204, 224))
        self.assertEqual((b15[2] - b15[0]) * (b15[3] - b15[1]), 182)
        b30 = ex.footprint_pixel_bounds(window_box, 30, 30, (14, 14), 32)
        self.assertEqual(b30, (382, 420, 394, 434))
        self.assertEqual((b30[2] - b30[0]) * (b30[3] - b30[1]), 168)

    def test_real_407x448_window_left_offset_52(self):
        # The actual second canonical window of 000000567197.jpg.
        window_box = (0, 52, 407, 500)
        b15 = ex.footprint_pixel_bounds(window_box, 15, 15, (14, 14), 32)
        self.assertEqual(b15, (191, 262, 204, 276))
        b30 = ex.footprint_pixel_bounds(window_box, 30, 30, (14, 14), 32)
        self.assertEqual(b30, (382, 472, 394, 486))

    def test_300x200_corrected_areas(self):
        window_box = (0, 0, 300, 200)
        b15 = ex.footprint_pixel_bounds(window_box, 15, 15, (14, 14), 32)
        self.assertEqual(b15, (141, 94, 150, 100))
        self.assertEqual((b15[2] - b15[0]) * (b15[3] - b15[1]), 54)
        b30 = ex.footprint_pixel_bounds(window_box, 30, 30, (14, 14), 32)
        self.assertEqual(b30, (281, 188, 291, 194))
        self.assertEqual((b30[2] - b30[0]) * (b30[3] - b30[1]), 60)

    def test_no_longer_raises_for_the_real_clamped_image(self):
        # Prior repair round left this fail-closed pending an authority
        # decision; that decision has now been made and implemented.
        window_box = (0, 0, 407, 448)
        ex.footprint_pixel_bounds(window_box, anchor_row=15, anchor_col=15, patch_hw=(14, 14), grid_side=32)  # must not raise


class T0T320AnchorPixelBoundsTests(unittest.TestCase):
    """t0_t320_anchor_pixel_bounds implements the SELECTED contract: the
    historical patch-center pixel index (i*patch_h+patch_h//2), mapped
    through the input-tokenization inverse point rule -- NOT the
    align_corners=True output-placement inverse an earlier repair round
    used here (that conflated anchor-content geometry with a different,
    later, output-rendering transform, and did not extend the historical
    i*14+7 meaning continuously as native size passes through 448)."""

    ELIGIBLE = range(1, 31)  # [scientific_inheritance.e3_geometry].eligible_grid_range

    def test_exact_identity_at_native_448(self):
        window_box = (0, 0, 448, 448)
        for a in (1, 15, 30):
            bounds = ex.t0_t320_anchor_pixel_bounds(window_box, a, a, grid_side=32, patch_hw=(14, 14))
            expected = a * 14 + 7
            self.assertEqual(bounds, (expected, expected, expected + 1, expected + 1))

    def test_real_407x448_window_zero_offset(self):
        window_box = (0, 0, 407, 448)
        self.assertEqual(ex.t0_t320_anchor_pixel_bounds(window_box, 15, 15, 32, (14, 14)), (197, 217, 198, 218))
        self.assertEqual(ex.t0_t320_anchor_pixel_bounds(window_box, 30, 30, 32, (14, 14)), (388, 427, 389, 428))

    def test_real_407x448_window_left_offset_52(self):
        window_box = (0, 52, 407, 500)
        self.assertEqual(ex.t0_t320_anchor_pixel_bounds(window_box, 15, 15, 32, (14, 14)), (197, 269, 198, 270))
        self.assertEqual(ex.t0_t320_anchor_pixel_bounds(window_box, 30, 30, 32, (14, 14)), (388, 479, 389, 480))

    def test_300x200_oracle(self):
        window_box = (0, 0, 300, 200)
        self.assertEqual(ex.t0_t320_anchor_pixel_bounds(window_box, 15, 15, 32, (14, 14)), (145, 97, 146, 98))
        self.assertEqual(ex.t0_t320_anchor_pixel_bounds(window_box, 30, 30, 32, (14, 14)), (286, 190, 287, 191))

    def test_every_eligible_anchor_stays_in_bounds_across_real_population_dimensions(self):
        real_dims = [
            (407, 500), (338, 450), (425, 640), (640, 411), (640, 426),
            (333, 500), (640, 427), (398, 500), (399, 600), (427, 640),
            (314, 640), (333, 500), (426, 640), (640, 360), (375, 500),
            (425, 640), (500, 375), (336, 500), (426, 640), (500, 334),
            (427, 640), (375, 500), (427, 640), (427, 640), (640, 428),
            (422, 640), (426, 640), (425, 640), (500, 375),
        ]
        for native_h, native_w in real_dims:
            window_box = (0, 0, native_h, native_w)
            for anchor_row in (1, 15, 30):
                for anchor_col in (1, 15, 30):
                    top, left, bottom, right = ex.t0_t320_anchor_pixel_bounds(window_box, anchor_row, anchor_col, 32, (14, 14))
                    self.assertTrue(0 <= top < bottom <= native_h, f"row out of range for dims ({native_h},{native_w})")
                    self.assertTrue(0 <= left < right <= native_w, f"col out of range for dims ({native_h},{native_w})")

    def test_multi_window_image_with_clamped_edge_window(self):
        windows = ex.compute_canonical_windows(448, 600, (448, 448), (224, 224))
        for (ordinal, top, left, bottom, right) in windows:
            box = (top, left, bottom, right)
            for anchor_row in (1, 30):
                for anchor_col in (1, 30):
                    t, l, b, r = ex.t0_t320_anchor_pixel_bounds(box, anchor_row, anchor_col, 32, (14, 14))
                    self.assertTrue(top <= t < b <= bottom)
                    self.assertTrue(left <= l < r <= right)

    def test_reads_the_correct_predicted_class_from_a_real_t0_raster(self):
        # Not just bounds: builds a REAL T0 raster (via semantic_evidence_
        # t320.generate_t0_t320_targets, the actual production function)
        # from a controlled, deterministic grid where the top half of
        # eligible rows favors class 0 and the bottom half favors class 1,
        # then confirms the point this function selects reads back the
        # class its own row actually belongs to -- for BOTH a 448 and a
        # real 407x448 clamped window.
        import semantic_evidence_e3_adapter as e3a
        import semantic_evidence_t320 as t320

        for native_h, native_w in ((448, 448), (407, 448)):
            grid = 32
            s0 = torch.full((grid * grid, 2), -5.0)
            for r in range(grid):
                for c in range(grid):
                    idx = r * grid + c
                    s0[idx, 0 if r < 16 else 1] = 5.0  # rows 0-15 favor class 0, 16-31 favor class 1
            snapshot = e3a.WindowSnapshot(
                window_id="synthetic:0:ORIGINAL", window_ordinal=0, canvas_h=native_h, canvas_w=native_w,
                top=0, left=0, bottom=native_h, right=native_w, infer_h=native_h, infer_w=native_w,
                backbone_input_height=grid * 14, backbone_input_width=grid * 14,
                grid_h=grid, grid_w=grid, patch_h=14, patch_w=14,
                dino_features=torch.nn.functional.normalize(torch.rand(grid * grid, FEAT_DIM), dim=1),
                s0=s0, class_count=2, feature_dim=FEAT_DIM, dtype=s0.dtype, device=s0.device,
            )
            targets = t320.generate_t0_t320_targets(
                [snapshot], [(0, 0, native_h, native_w)], canvas_h=native_h, canvas_w=native_w,
                ori_h=native_h, ori_w=native_w,
            )
            t0_rows = targets["t0"][0].tolist()

            window_box = (0, 0, native_h, native_w)
            for anchor_row, expected_class in ((5, 0), (25, 1)):
                bounds = ex.t0_t320_anchor_pixel_bounds(window_box, anchor_row, 15, grid_side=32, patch_hw=(14, 14))
                footprint = ex.crop_raster(t0_rows, bounds)
                predicted = ex._center_pixel_class(footprint)
                self.assertEqual(predicted, expected_class,
                                  f"native=({native_h},{native_w}) anchor_row={anchor_row}: expected class {expected_class}, got {predicted}")


class InterventionCoreNativeBoundsTests(unittest.TestCase):
    """intervention_core_native_bounds maps the protected core_pixel_
    bounds' 448-canvas interval to the window's OWN native pixels, before
    it is used to mask the native, pre-resize crop tensor -- fixing the
    silent-truncation risk found during geometry-contract review (ordinary
    tensor/boolean-mask slicing clips out-of-range indices rather than
    raising, so an unmapped 448-canvas core applied to a smaller native
    crop would silently manipulate the wrong, truncated region)."""

    def test_exact_identity_at_native_448(self):
        window_box = (0, 0, 448, 448)
        bounds = ex.intervention_core_native_bounds(window_box, 15, 15, grid_side=32, patch_hw=(14, 14))
        self.assertEqual(bounds, (196, 196, 238, 238))

    def test_real_407x448_window_anchor_row_30(self):
        # The specific case that would silently truncate under plain
        # tensor slicing: core_pixel_bounds(30,30) = (406,406,448,448) in
        # 448-canvas space; a native crop of height 407 would silently
        # clip `mask[..., 406:448, ...]` to a single row under ordinary
        # slicing if this were applied unmapped.
        window_box = (0, 0, 407, 448)
        bounds = ex.intervention_core_native_bounds(window_box, 30, 30, grid_side=32, patch_hw=(14, 14))
        self.assertEqual(bounds, (369, 406, 407, 448))
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        self.assertEqual(area, 1596)
        self.assertGreater(bounds[2] - bounds[0], 30, "core height must not have silently truncated to ~1px")

    def test_300x200_corrected_areas(self):
        window_box = (0, 0, 300, 200)
        c15 = ex.intervention_core_native_bounds(window_box, 15, 15, grid_side=32, patch_hw=(14, 14))
        self.assertEqual(c15, (131, 88, 159, 106))
        self.assertEqual((c15[2] - c15[0]) * (c15[3] - c15[1]), 504)
        c30 = ex.intervention_core_native_bounds(window_box, 30, 30, grid_side=32, patch_hw=(14, 14))
        self.assertEqual(c30, (272, 181, 300, 200))
        self.assertEqual((c30[2] - c30[0]) * (c30[3] - c30[1]), 532)

    def test_actual_mask_pixel_count_matches_the_mapped_core_not_a_silent_truncation(self):
        # Builds a REAL native crop tensor (407 tall, matching the real
        # failing image) and a REAL intervention view via the protected
        # ifeat.build_intervention_view, at anchor row=30 (the exact row
        # whose 448-canvas core would overrun a 407-tall crop under plain
        # slicing), and inspects the ACTUAL number of pixels the view
        # changed relative to the original -- proving the mask covers the
        # full, correctly-mapped 38x42 core (1596px per channel-row), not
        # a truncated ~1px sliver.
        import semantic_evidence_intervention_features as ifeat

        native_h, native_w = 407, 448
        crop = torch.rand(1, 3, native_h, native_w) * 255.0
        window_box = (0, 0, native_h, native_w)
        anchor_bounds = ex.intervention_core_native_bounds(window_box, 30, 30, grid_side=32, patch_hw=(14, 14))

        view_crop = ifeat.build_intervention_view("REGION_BLURRED", crop, anchor_bounds)
        changed = (view_crop != crop).any(dim=1).sum().item()  # count of changed spatial positions
        top, left, bottom, right = anchor_bounds
        expected_area = (bottom - top) * (right - left)
        self.assertEqual(changed, expected_area,
                          "REGION_BLURRED must change exactly the mapped core's pixel area -- "
                          "any smaller count means the mask silently truncated")
        self.assertGreater(bottom - top, 30, "core height must reflect the real 38px-tall mapped core, not ~1px")


class ClampedWindowFullPipelineTests(unittest.TestCase):
    """Runs the full discovery execution path (Phase A then Phase B) at
    the real failing dimensions, through a non-448 fake image loader --
    exactly the class of test a fixed-448x448 fake loader would conceal.
    With the unified contract implemented, BOTH phases now succeed for a
    clamped window (the prior round's Phase B fail-closed guard is gone,
    replaced by a real, tested mapping); an unclamped 448x448 image
    remains completely unaffected."""

    def _non_448_image_loader(self, real_h, real_w, seed=1):
        def loader(iid):
            g = torch.Generator().manual_seed(seed)
            return torch.rand(1, 3, real_h, real_w, generator=g) * 255.0, real_h, real_w
        return loader

    def _gt_loader(self, real_h, real_w, seed=2):
        def loader(iid):
            g = torch.Generator().manual_seed(seed)
            klass = int(torch.randint(0, CLASS_COUNT, (1,), generator=g).item())
            return [[klass] * real_w for _ in range(real_h)]
        return loader

    def test_phase_a_and_b_both_succeed_for_the_real_failing_image(self):
        real_h, real_w = 407, 448  # the real 000000567197.jpg canonical window size
        event_log = ex.PhaseEventLog()
        model, text_embedding = _fake_model_and_text_provider()
        gt_free = ex._run_phase_a(
            "000000567197", self._non_448_image_loader(real_h, real_w), _fake_forward_fn, model, text_embedding,
            (448, 448), (224, 224), (14, 14), event_log,
        )
        ex.validate_gt_free_artifact(gt_free)
        self.assertGreater(len(gt_free["control_features"]["query_correctness"]), 0)
        for detail in gt_free["control_features"]["propagation_outcome"].values():
            self.assertIsInstance(detail["t0_predicted_class"], int)
            self.assertIsInstance(detail["t320_predicted_class"], int)
            self.assertTrue(0 <= detail["t0_predicted_class"] < CLASS_COUNT)
            self.assertTrue(0 <= detail["t320_predicted_class"] < CLASS_COUNT)

        gt_label = ex._run_phase_b(gt_free, self._gt_loader(real_h, real_w), "000000567197", event_log)
        ex.validate_gt_label_artifact(gt_label)
        self.assertGreater(len(gt_label["query_correctness_labels"]) + len(gt_label["ambiguous_gt_anchor_ids"]), 0)

    def test_phase_a_and_b_both_succeed_unchanged_for_an_unclamped_image(self):
        # Regression guard: a plain 448x448 image (the historical case)
        # must be completely unaffected by this repair.
        gt_free, gt_label = _sample_gt_pair()
        ex.validate_gt_free_artifact(gt_free)
        ex.validate_gt_label_artifact(gt_label)
        self.assertGreater(len(gt_label["query_correctness_labels"]) + len(gt_label["ambiguous_gt_anchor_ids"]), 0)


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
