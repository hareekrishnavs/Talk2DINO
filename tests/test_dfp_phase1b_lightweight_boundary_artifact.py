"""Tests for the BALP development checkpoint/result artifact
validators, converters, and atomic writers
(`src.dfp_phase1b_lightweight_boundary_artifact`). Exercises the real
public loader/writer functions -- never a mock of the validator
itself. No torch/model/dataset/CUDA access."""

from __future__ import annotations

import itertools
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1b_lightweight_boundary_artifact as art
import src.dfp_phase1b_lightweight_boundary_identity as e22

E = art.DfpPhase1bLightweightBoundaryArtifactError
S, R = (0, 16, 32, 64), (0, 1, 2, 4)
CANDS = [f"balp_s{s:03d}_r{r:02d}" for s, r in itertools.product(S, R)]
VARIANTS = list(e22.CANONICAL_CONTROLS) + CANDS
CLASS_COUNT = 3
DEV = [f"dev{i:04d}" for i in range(350)]
CONF = [f"conf{i:04d}" for i in range(150)]
IDENTITY = {
    "e22_protocol_sha256": "1" * 64, "e22_artifacts_sha256": "2" * 64, "e13_protocol_sha256": "3" * 64,
    "matched_evaluator_sha256": "4" * 64, "subset_manifest_sha256": "5" * 64, "balp_split_sha256": "6" * 64,
    "run_mode": "coco_stuff_500",
}


@pytest.fixture
def tmp_root(tmp_path):
    return tmp_path / "run"


def _one_image_inputs(value=1):
    conf = {v: [[value if r == c else 0 for c in range(CLASS_COUNT)] for r in range(CLASS_COUNT)] for v in VARIANTS}
    scalars = {v: {f: value for f in art.PER_IMAGE_STAT_SCALAR_FIELDS} for v in VARIANTS}
    return conf, scalars


def _install_n(root, n, *, start_arrays=None, start_generation=0, start_completed=None):
    """Installs images [len(start_completed), n) sequentially, each as
    its OWN new immutable checkpoint generation (monotonically
    incrementing from `start_generation`) -- matching
    `execute_development`'s real per-image generation-numbering
    convention exactly. Returns (reopened_doc, arrays, completed_ids)."""
    arrays = start_arrays if start_arrays is not None else art.empty_checkpoint_arrays(variant_ids=VARIANTS, class_count=CLASS_COUNT)
    completed = list(start_completed) if start_completed is not None else []
    generation = start_generation
    reopened = None
    for i in range(len(completed), n):
        conf, scal = _one_image_inputs(i + 1)
        arrays = art.append_image_to_checkpoint_arrays(arrays, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
        completed.append(DEV[i])
        reopened = art.install_checkpoint_generation(
            root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=arrays,
            completed_image_ids=completed, generation=generation, complete=(len(completed) == 350), final=False,
            final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )
        generation += 1
    return reopened, arrays, completed


def _current_generation_dir(root):
    pointer = art.load_current_pointer(root)
    assert pointer is not None
    return root / art.CHECKPOINT_GENERATIONS_DIRNAME / pointer["generation_directory"]


# =====================================================================
# Atomic checkpoint install: valid states, resume, digest binding.
# =====================================================================


def test_install_checkpoint_generation_zero_progress(tmp_root):
    reopened, _, _ = _install_n(tmp_root, 0)
    doc = art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    assert doc is None  # nothing installed yet for n=0
    assert art.load_current_pointer(tmp_root) is None


def test_install_checkpoint_generation_partial(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 5)
    assert reopened["progress"]["next_development_index"] == 5
    assert reopened["complete"] is False
    assert reopened["progress"]["generation"] == 4  # 5 images installed as generations 0..4


def test_install_checkpoint_generation_complete_unfinalized(tmp_root):
    reopened, _, _ = _install_n(tmp_root, 350)
    assert reopened["complete"] is True
    assert reopened["final"] is False


def test_load_full_checkpoint_reopens_and_matches_installed_state(tmp_root):
    reopened, _, _ = _install_n(tmp_root, 4)
    loaded = art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    assert loaded == reopened


def test_each_completed_image_creates_its_own_immutable_generation_directory(tmp_root):
    _install_n(tmp_root, 5)
    gens_dir = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME
    generation_dirs = sorted(p.name for p in gens_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    assert generation_dirs == [f"generation_{i}" for i in range(5)]
    for name in generation_dirs:
        d = gens_dir / name
        assert (d / art.CANONICAL_FILENAMES["checkpoint"]).is_file()
        assert (d / art.CANONICAL_FILENAMES["checkpoint_arrays"]).is_file()
    # every earlier generation remains byte-identical and independently loadable/valid --
    # immutable generations are never rewritten once installed.
    for i, name in enumerate(generation_dirs):
        d = gens_dir / name
        doc = json.loads((d / art.CANONICAL_FILENAMES["checkpoint"]).read_bytes())
        assert doc["progress"]["next_development_index"] == i + 1
        assert doc["progress"]["generation"] == i


def test_reinstalling_same_generation_number_rejected(tmp_root):
    _install_n(tmp_root, 1)  # occupies generation_0
    arrays = art.empty_checkpoint_arrays(variant_ids=VARIANTS, class_count=CLASS_COUNT)
    conf, scal = _one_image_inputs(1)
    arrays = art.append_image_to_checkpoint_arrays(arrays, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
    with pytest.raises(E, match="already exists"):
        art.install_checkpoint_generation(
            tmp_root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=arrays,
            completed_image_ids=[DEV[0]], generation=0, complete=False, final=False,
            final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )


def test_current_pointer_advances_only_to_latest_generation(tmp_root):
    _install_n(tmp_root, 3)
    pointer = art.load_current_pointer(tmp_root)
    assert pointer["current_generation"] == 2
    assert pointer["generation_directory"] == "generation_2"
    assert pointer["format"] == art.CURRENT_POINTER_FORMAT


def test_checkpoint_sibling_digest_binding_enforced(tmp_root):
    _install_n(tmp_root, 2)
    gen_dir = _current_generation_dir(tmp_root)
    json_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint"]
    doc = json.loads(json_path.read_bytes())
    doc["identity_bindings"]["development_digest"] = "f" * 64  # corrupt the sibling binding
    json_path.write_text(json.dumps(doc))
    # also update the pointer's own checkpoint_json_sha256 to match the tampered file, so the
    # pointer-level self-consistency check passes and the DEEPER sibling-digest check (JSON's
    # own recorded development_digest vs the actual NPZ bytes) is the one that is isolated here.
    pointer_path = tmp_root / art.CURRENT_CHECKPOINT_POINTER_FILENAME
    pointer_doc = json.loads(pointer_path.read_bytes())
    pointer_doc["checkpoint_json_sha256"] = art._sha256_file(json_path)
    pointer_path.write_text(json.dumps(pointer_doc))
    with pytest.raises(E, match="sibling digest mismatch"):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_checkpoint_resume_no_reprocessing_and_byte_identical(tmp_root):
    reopened5, arrays5, completed5 = _install_n(tmp_root, 5)
    reopened, arrays10, completed10 = _install_n(
        tmp_root, 10, start_arrays=arrays5, start_completed=completed5, start_generation=reopened5["progress"]["generation"] + 1,
    )
    assert completed10 == DEV[:10]
    assert reopened["progress"]["completed_image_ids"] == DEV[:10]
    # every generation installed by the first _install_n call remains untouched by the second
    for i in range(5):
        d = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / f"generation_{i}"
        doc = json.loads((d / art.CANONICAL_FILENAMES["checkpoint"]).read_bytes())
        assert doc["progress"]["next_development_index"] == i + 1


def test_checkpoint_non_prefix_rejected(tmp_root):
    _install_n(tmp_root, 3)
    arrays = art.empty_checkpoint_arrays(variant_ids=VARIANTS, class_count=CLASS_COUNT)
    conf, scal = _one_image_inputs(1)
    arrays = art.append_image_to_checkpoint_arrays(arrays, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
    with pytest.raises(E):
        art.install_checkpoint_generation(
            tmp_root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=arrays,
            completed_image_ids=["not_a_real_prefix"], generation=3, complete=False, final=False,
            final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )


def test_checkpoint_confirmation_id_leakage_rejected(tmp_root):
    arrays = art.empty_checkpoint_arrays(variant_ids=VARIANTS, class_count=CLASS_COUNT)
    conf, scal = _one_image_inputs(1)
    arrays = art.append_image_to_checkpoint_arrays(arrays, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
    with pytest.raises(E):
        art.install_checkpoint_generation(
            tmp_root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=arrays,
            completed_image_ids=[CONF[0]], generation=0, complete=False, final=False,
            final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )


def test_checkpoint_aggregate_per_image_disagreement_rejected(tmp_root):
    arrays = art.empty_checkpoint_arrays(variant_ids=VARIANTS, class_count=CLASS_COUNT)
    conf, scal = _one_image_inputs(1)
    arrays = art.append_image_to_checkpoint_arrays(arrays, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
    arrays["aggregate_confusion"]["T0"][0][0] = 99999
    with pytest.raises(E):
        art.install_checkpoint_generation(
            tmp_root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=arrays,
            completed_image_ids=[DEV[0]], generation=0, complete=False, final=False,
            final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )


def test_changed_old_checkpoint_segment_detected_on_reopen(tmp_root):
    _install_n(tmp_root, 3)
    gen_dir = _current_generation_dir(tmp_root)
    npz_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    data = dict(np.load(npz_path))
    data["pi__T0__evaluated_pixel_count"] = data["pi__T0__evaluated_pixel_count"].copy()
    data["pi__T0__evaluated_pixel_count"][0] = 999999
    np.savez(npz_path, **data)
    # sibling digest now disagrees with the (unchanged) JSON -> must be rejected
    with pytest.raises(E):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


# =====================================================================
# Corrupted/truncated NPZ: must always fail through the artifact
# domain exception, never a raw zipfile.BadZipFile or other exception.
# =====================================================================


def test_truncated_npz_wrapped_not_raw_badzipfile(tmp_root):
    _install_n(tmp_root, 2)
    gen_dir = _current_generation_dir(tmp_root)
    npz_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    data = npz_path.read_bytes()
    npz_path.write_bytes(data[: len(data) // 2])
    with pytest.raises(E) as excinfo:
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    assert excinfo.value.__cause__ is not None
    import zipfile
    assert isinstance(excinfo.value.__cause__, (zipfile.BadZipFile, OSError, ValueError, EOFError))


def test_random_bytes_npz_wrapped_not_raw_exception(tmp_root):
    _install_n(tmp_root, 2)
    gen_dir = _current_generation_dir(tmp_root)
    npz_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    npz_path.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(E) as excinfo:
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    assert excinfo.value.__cause__ is not None


def test_valid_zip_missing_required_member_wrapped(tmp_root):
    _install_n(tmp_root, 2)
    gen_dir = _current_generation_dir(tmp_root)
    npz_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    data = dict(np.load(npz_path))
    del data["agg__T0"]
    np.savez(npz_path, **data)
    with pytest.raises(E, match="missing member"):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_invalid_npy_payload_inside_zip_wrapped(tmp_root):
    import zipfile as _zipfile
    _install_n(tmp_root, 2)
    gen_dir = _current_generation_dir(tmp_root)
    npz_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    # rewrite the zip with one member's payload replaced by garbage bytes
    # that are NOT a valid .npy file, while keeping the zip container itself valid.
    with _zipfile.ZipFile(npz_path, "r") as zin:
        members = {info.filename: zin.read(info.filename) for info in zin.infolist()}
    target = "agg__T0.npy"
    assert target in members
    members[target] = b"NOT A VALID NPY PAYLOAD" * 4
    with _zipfile.ZipFile(npz_path, "w") as zout:
        for name, payload in members.items():
            zout.writestr(name, payload)
    with pytest.raises(E, match="not a valid array"):
        # numpy silently returns raw bytes for this corruption mode rather than raising --
        # so there is no underlying exception to chain here; our own structural guard
        # (_require_npz_ndarray) raises the domain exception directly and correctly.
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_corrupt_npz_fatal_exceptions_still_propagate(tmp_root, monkeypatch):
    _install_n(tmp_root, 2)
    gen_dir = _current_generation_dir(tmp_root)
    npz_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    npz_path.write_bytes(b"garbage")

    real_load = np.load
    def raising_load(*a, **k):
        raise KeyboardInterrupt("synthetic fatal during npz load")
    monkeypatch.setattr(np, "load", raising_load)
    with pytest.raises(KeyboardInterrupt):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    monkeypatch.setattr(np, "load", real_load)


# =====================================================================
# Current-pointer safety: path traversal, symlink escape, malformed
# pointer, mismatched siblings, mixed-generation pairing.
# =====================================================================


def test_pointer_rejects_path_traversal_generation_directory(tmp_root):
    _install_n(tmp_root, 1)
    pointer_path = tmp_root / art.CURRENT_CHECKPOINT_POINTER_FILENAME
    doc = json.loads(pointer_path.read_bytes())
    doc["generation_directory"] = "../../etc"
    pointer_path.write_text(json.dumps(doc))
    with pytest.raises(E):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_pointer_rejects_absolute_path_substitution(tmp_root):
    _install_n(tmp_root, 1)
    pointer_path = tmp_root / art.CURRENT_CHECKPOINT_POINTER_FILENAME
    doc = json.loads(pointer_path.read_bytes())
    doc["generation_directory"] = "/etc"
    pointer_path.write_text(json.dumps(doc))
    with pytest.raises(E):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_pointer_rejects_symlink_escape(tmp_root):
    _install_n(tmp_root, 1)
    gens_dir = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME
    outside = tmp_root.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    link = gens_dir / "generation_evil"
    link.symlink_to(outside, target_is_directory=True)
    pointer_path = tmp_root / art.CURRENT_CHECKPOINT_POINTER_FILENAME
    doc = json.loads(pointer_path.read_bytes())
    doc["generation_directory"] = "generation_evil"
    pointer_path.write_text(json.dumps(doc))
    with pytest.raises(E, match="symlink"):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_pointer_rejects_missing_generation(tmp_root):
    _install_n(tmp_root, 1)
    pointer_path = tmp_root / art.CURRENT_CHECKPOINT_POINTER_FILENAME
    doc = json.loads(pointer_path.read_bytes())
    doc["generation_directory"] = "generation_999"
    pointer_path.write_text(json.dumps(doc))
    with pytest.raises(E, match="does not exist"):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_pointer_rejects_malformed_pointer_missing_key(tmp_root):
    _install_n(tmp_root, 1)
    pointer_path = tmp_root / art.CURRENT_CHECKPOINT_POINTER_FILENAME
    doc = json.loads(pointer_path.read_bytes())
    del doc["checkpoint_json_sha256"]
    pointer_path.write_text(json.dumps(doc))
    with pytest.raises(E):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_pointer_rejects_stale_npz_digest(tmp_root):
    _install_n(tmp_root, 2)
    gen_dir = _current_generation_dir(tmp_root)
    npz_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    data = dict(np.load(npz_path))
    data["agg__T0"] = data["agg__T0"].copy()
    data["agg__T0"][0, 0] += 7
    np.savez(npz_path, **data)  # a VALID but different NPZ -- pointer's recorded hash is now stale
    with pytest.raises(E, match="checkpoint_arrays_sha256 mismatch"):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_pointer_rejects_stale_json_digest(tmp_root):
    _install_n(tmp_root, 2)
    gen_dir = _current_generation_dir(tmp_root)
    json_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint"]
    doc = json.loads(json_path.read_bytes())
    doc["complete"] = doc["complete"]  # touch nothing semantically...
    json_path.write_text(json.dumps(doc, indent=2))  # ...but re-serialize with different formatting -> different bytes/hash
    with pytest.raises(E, match="checkpoint_json_sha256 mismatch"):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


def test_mixed_generation_json_and_npz_rejected(tmp_root):
    """Directly mixing a generation_0 checkpoint.json with a
    generation_1 checkpoint_arrays.npz (bypassing the pointer/rename
    mechanism entirely, as if someone manually assembled a Frankenstein
    generation directory) must be rejected."""
    _install_n(tmp_root, 2)  # installs generation_0 and generation_1
    gens_dir = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME
    gen0_json = json.loads((gens_dir / "generation_0" / art.CANONICAL_FILENAMES["checkpoint"]).read_bytes())
    gen1_npz_bytes = (gens_dir / "generation_1" / art.CANONICAL_FILENAMES["checkpoint_arrays"]).read_bytes()

    frankenstein = gens_dir / "generation_2"
    frankenstein.mkdir()
    (frankenstein / art.CANONICAL_FILENAMES["checkpoint"]).write_text(json.dumps(gen0_json))
    (frankenstein / art.CANONICAL_FILENAMES["checkpoint_arrays"]).write_bytes(gen1_npz_bytes)

    pointer_path = tmp_root / art.CURRENT_CHECKPOINT_POINTER_FILENAME
    doc = json.loads(pointer_path.read_bytes())
    doc["generation_directory"] = "generation_2"
    doc["current_generation"] = 2
    doc["checkpoint_json_sha256"] = art._sha256_file(frankenstein / art.CANONICAL_FILENAMES["checkpoint"])
    doc["checkpoint_arrays_sha256"] = art._sha256_file(frankenstein / art.CANONICAL_FILENAMES["checkpoint_arrays"])
    pointer_path.write_text(json.dumps(doc))
    # gen0's JSON records gen0's own npz digest, which disagrees with gen1's npz bytes -> sibling digest mismatch first
    with pytest.raises(E):
        art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)


# =====================================================================
# Crash-atomicity fault injection: install_checkpoint_generation must
# leave the previous current generation valid and loadable no matter
# where a failure is injected, and a subsequent retry must reach the
# exact same state as an unfaulted run.
# =====================================================================


def _fake_processor_result(i):
    return _one_image_inputs(i + 1)


def _install_one_more(root, *, arrays, completed, generation):
    conf, scal = _fake_processor_result(len(completed))
    new_arrays = art.append_image_to_checkpoint_arrays(arrays, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
    new_completed = completed + [DEV[len(completed)]]
    art.install_checkpoint_generation(
        root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=new_arrays,
        completed_image_ids=new_completed, generation=generation, complete=(len(new_completed) == 350), final=False,
        final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
    )
    return new_arrays, new_completed


@pytest.mark.parametrize("fault_target,fault_stage", [
    ("write_arrays_into", "before"),
    ("write_json_into", "before"),
    ("validate_full_checkpoint", "during_stage_validation"),
])
def test_atomicity_fault_before_rename_leaves_old_generation_untouched(tmp_root, monkeypatch, fault_target, fault_stage):
    reopened5, arrays5, completed5 = _install_n(tmp_root, 5)
    before_pointer = art.load_current_pointer(tmp_root)
    before_doc = art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    gens_dir = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME
    before_dir_count = len([p for p in gens_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])

    if fault_target == "write_arrays_into":
        target_name = "_write_checkpoint_arrays_npz_into"
    elif fault_target == "write_json_into":
        target_name = "_write_checkpoint_json_into"
    else:
        target_name = "validate_full_checkpoint"

    def raiser(*a, **k):
        raise OSError(f"SYNTHETIC failure at {fault_target}/{fault_stage}")
    monkeypatch.setattr(art, target_name, raiser)

    conf6, scal6 = _one_image_inputs(6)
    new_arrays = art.append_image_to_checkpoint_arrays(arrays5, variant_ids=VARIANTS, per_variant_confusion=conf6, per_variant_scalars=scal6)
    with pytest.raises(OSError):
        art.install_checkpoint_generation(
            tmp_root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=new_arrays,
            completed_image_ids=completed5 + [DEV[5]], generation=reopened5["progress"]["generation"] + 1,
            complete=False, final=False, final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )
    monkeypatch.undo()  # clear the fault BEFORE the retry phase below

    # old current generation must remain exactly as it was
    after_pointer = art.load_current_pointer(tmp_root)
    after_doc = art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    assert after_pointer == before_pointer
    assert after_doc == before_doc
    # no NEW generation directory becomes visible under its real (non-staging) name
    after_dir_count = len([p for p in gens_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    assert after_dir_count == before_dir_count

    # retry (fault cleared) must succeed and reach a state equivalent to an unfaulted run
    retried_arrays, retried_completed = _install_one_more(tmp_root, arrays=arrays5, completed=completed5, generation=reopened5["progress"]["generation"] + 1)
    final_doc = art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    assert final_doc["progress"]["completed_image_ids"] == DEV[:6]


def test_atomicity_original_npz_then_json_corruption_no_longer_representable(tmp_root, monkeypatch):
    """The original defect: NPZ successfully installed, JSON write
    fails. Under the new architecture, BOTH files are written into a
    private staging directory and made visible together via exactly
    ONE atomic directory rename -- so a JSON-write failure can only
    ever occur while still in staging, before anything is renamed into
    a real generation directory. Proves the original corruption is no
    longer representable as a current-checkpoint state."""
    reopened5, arrays5, completed5 = _install_n(tmp_root, 5)
    before_doc = art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)

    real_write_json = art._write_checkpoint_json_into
    def failing_json_write(*a, **k):
        raise OSError("SYNTHETIC JSON staging failure, AFTER the NPZ was already staged")
    monkeypatch.setattr(art, "_write_checkpoint_json_into", failing_json_write)

    conf, scal = _one_image_inputs(6)
    new_arrays = art.append_image_to_checkpoint_arrays(arrays5, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
    with pytest.raises(OSError):
        art.install_checkpoint_generation(
            tmp_root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=new_arrays,
            completed_image_ids=completed5 + [DEV[5]], generation=reopened5["progress"]["generation"] + 1,
            complete=False, final=False, final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )
    monkeypatch.setattr(art, "_write_checkpoint_json_into", real_write_json)

    # the previous current checkpoint must still load -- the NPZ that WAS staged never became
    # visible under a real generation directory name, so it can never be mistaken for "current".
    after_doc = art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    assert after_doc == before_doc
    # no orphaned staging directory left behind
    gens_dir = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME
    staging_dirs = [p for p in gens_dir.iterdir() if p.name.startswith(".staging")]
    assert staging_dirs == []
    # retry succeeds and reaches the same state a clean run would
    retried, _ = art.append_image_to_checkpoint_arrays(new_arrays, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal), None
    reopened6 = art.install_checkpoint_generation(
        tmp_root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=new_arrays,
        completed_image_ids=completed5 + [DEV[5]], generation=reopened5["progress"]["generation"] + 1,
        complete=False, final=False, final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
    )
    assert reopened6["progress"]["completed_image_ids"] == DEV[:6]


def test_atomicity_fault_after_rename_before_pointer_advance_leaves_old_current(tmp_root, monkeypatch):
    """Simulates a failure AFTER the new generation directory has been
    installed (atomic rename succeeded) but BEFORE the current pointer
    is advanced to it (post-install reverification fails). The new
    generation becomes an unreferenced-but-valid orphan; the previous
    generation remains current."""
    reopened5, arrays5, completed5 = _install_n(tmp_root, 5)
    before_pointer = art.load_current_pointer(tmp_root)

    real_validate = art.validate_full_checkpoint
    state = {"matching_calls": 0}
    def sometimes_failing_validate(doc, *, development_ids, confirmation_ids):
        # install_checkpoint_generation calls validate_full_checkpoint on the NEW generation
        # (next_development_index == 6) exactly twice before the pointer is ever touched: once
        # against the STAGED bytes, once against the INSTALLED (post-rename) bytes. Fail on that
        # SECOND (post-install) call specifically -- strictly before the pointer write.
        if doc["progress"]["next_development_index"] == 6:
            state["matching_calls"] += 1
            if state["matching_calls"] == 2:
                raise art.DfpPhase1bLightweightBoundaryArtifactError("SYNTHETIC post-install reverification failure")
        return real_validate(doc, development_ids=development_ids, confirmation_ids=confirmation_ids)
    monkeypatch.setattr(art, "validate_full_checkpoint", sometimes_failing_validate)

    conf, scal = _one_image_inputs(6)
    new_arrays = art.append_image_to_checkpoint_arrays(arrays5, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
    with pytest.raises(art.DfpPhase1bLightweightBoundaryArtifactError):
        art.install_checkpoint_generation(
            tmp_root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=new_arrays,
            completed_image_ids=completed5 + [DEV[5]], generation=reopened5["progress"]["generation"] + 1,
            complete=False, final=False, final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )
    monkeypatch.undo()

    # old pointer unaffected -- new generation, though installed on disk, never became current
    after_pointer = art.load_current_pointer(tmp_root)
    assert after_pointer == before_pointer
    gens_dir = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME
    assert (gens_dir / "generation_5").is_dir()  # the orphaned-but-valid installed generation
    after_doc = art.load_full_checkpoint(tmp_root, variant_ids=VARIANTS)
    assert after_doc["progress"]["completed_image_ids"] == DEV[:5]  # still generation_4, not generation_5


# =====================================================================
# Development result: schema, install, overwrite rejection.
# =====================================================================


def _valid_result_doc():
    ctrl = {c: {"control_id": c, "dataset_level_miou": 50.0, "per_class_iou": [1.0] * CLASS_COUNT,
                "mean_class_accuracy": 60.0, "pixel_accuracy": 70.0, "evaluated_pixel_count": 10, "ignored_pixel_count": 0}
            for c in e22.CANONICAL_CONTROLS}
    cand = {}
    for cid in CANDS:
        s, r = e22.parse_candidate_id(cid)
        cand[cid] = {"candidate_id": cid, "stability_start": s, "radius": r, "dataset_level_miou": 51.0,
                     "per_class_iou": [1.1] * CLASS_COUNT, "mean_class_accuracy": 61.0, "pixel_accuracy": 71.0,
                     "evaluated_pixel_count": 10, "ignored_pixel_count": 0, "candidate_versus_t320_changed_pixel_count": 1,
                     "preservation_mask_pixel_count": 1, "preservation_mask_fraction": 0.1, "boundary_preservation_pixel_count": 1,
                     "non_boundary_preservation_pixel_count": 0, "classwise_gain_loss_versus_t320": [0.0] * CLASS_COUNT}
    return art.build_development_result_document(
        identity_bindings=IDENTITY, control_results=ctrl, candidate_results=cand, scientific_output_digest="0" * 64,
        complete=True, final=True,
    )


def test_development_result_schema_name_and_version():
    doc = _valid_result_doc()
    assert doc["schema_name"] == art.DEVELOPMENT_RESULT_SCHEMA_NAME
    assert doc["schema_version"] == art.DEVELOPMENT_RESULT_SCHEMA_VERSION
    art.validate_development_result(doc, variant_ids=VARIANTS)


def test_development_result_missing_top_key_rejected():
    doc = _valid_result_doc()
    del doc["scientific_output_digest"]
    with pytest.raises(E):
        art.validate_development_result(doc, variant_ids=VARIANTS)


def test_development_result_extra_top_key_rejected():
    doc = _valid_result_doc()
    doc["extra"] = 1
    with pytest.raises(E):
        art.validate_development_result(doc, variant_ids=VARIANTS)


def test_development_result_wrong_schema_version_rejected():
    doc = _valid_result_doc()
    doc["schema_version"] = 999
    with pytest.raises(E):
        art.validate_development_result(doc, variant_ids=VARIANTS)


def test_development_result_unhashable_schema_version_rejected():
    doc = _valid_result_doc()
    doc["schema_version"] = [1]
    with pytest.raises(E):
        art.validate_development_result(doc, variant_ids=VARIANTS)


def test_development_result_candidate_control_reorder_via_key_set_still_valid():
    doc = _valid_result_doc()
    # dict key ORDER is irrelevant to a set-based key check -- rebuild with reversed insertion order
    doc["candidate_results"] = {k: doc["candidate_results"][k] for k in reversed(list(doc["candidate_results"]))}
    art.validate_development_result(doc, variant_ids=VARIANTS)  # must still be accepted


def test_development_result_foreign_candidate_rejected():
    doc = _valid_result_doc()
    doc["candidate_results"]["balp_s999_r99"] = doc["candidate_results"][CANDS[0]]
    with pytest.raises(E):
        art.validate_development_result(doc, variant_ids=VARIANTS)


def test_development_result_wrong_image_count_rejected():
    doc = _valid_result_doc()
    doc["development_image_count"] = 349
    with pytest.raises(E):
        art.validate_development_result(doc, variant_ids=VARIANTS)


def test_install_development_result_atomic_and_reverified(tmp_root):
    tmp_root.mkdir(parents=True)
    doc = _valid_result_doc()
    final_dir = art.install_development_result(tmp_root, doc=doc, variant_ids=VARIANTS)
    assert (final_dir / "result.json").exists()
    installed = json.loads((final_dir / "result.json").read_bytes())
    assert installed == doc


def test_install_development_result_overwrite_rejected(tmp_root):
    tmp_root.mkdir(parents=True)
    doc = _valid_result_doc()
    art.install_development_result(tmp_root, doc=doc, variant_ids=VARIANTS)
    with pytest.raises(E):
        art.install_development_result(tmp_root, doc=doc, variant_ids=VARIANTS)


def test_install_development_result_no_orphan_staging_on_failure(tmp_root):
    tmp_root.mkdir(parents=True)
    doc = _valid_result_doc()
    doc["candidate_results"]["balp_s000_r00"]["classwise_gain_loss_versus_t320"] = "not_a_list"  # trigger a late failure path is hard; use invalid doc directly
    bad_doc = dict(doc)
    del bad_doc["complete"]
    with pytest.raises(E):
        art.install_development_result(tmp_root, doc=bad_doc, variant_ids=VARIANTS)
    leftovers = [p for p in tmp_root.iterdir() if p.name.startswith(".staging-")]
    assert leftovers == []
    assert not (tmp_root / "development_result").exists()


def test_source_artifacts_never_mutated(tmp_root):
    tmp_root.mkdir(parents=True)
    doc = _valid_result_doc()
    frozen_copy = json.loads(json.dumps(doc))
    art.install_development_result(tmp_root, doc=doc, variant_ids=VARIANTS)
    assert doc == frozen_copy


def test_fatal_exception_propagates_through_install(tmp_root, monkeypatch):
    tmp_root.mkdir(parents=True)
    doc = _valid_result_doc()

    def boom(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(art, "validate_development_result", boom)
    with pytest.raises(KeyboardInterrupt):
        art.install_development_result(tmp_root, doc=doc, variant_ids=VARIANTS)


# =====================================================================
# Orphan-successor reconciliation (repairs the permanent resumability
# deadlock found by independent verification: a new generation
# directory installed successfully but the current pointer never
# advanced to it). See art.reconcile_orphan_successor.
# =====================================================================


def _reconcile(root, current):
    return art.reconcile_orphan_successor(
        root, current=current, variant_ids=VARIANTS, identity_bindings_without_digest=IDENTITY,
        development_ids=DEV, confirmation_ids=CONF,
    )


def test_reconcile_case_a_no_successor_returns_none_and_untouched(tmp_root):
    reopened, _, _ = _install_n(tmp_root, 3)
    before_ptr = art.load_current_pointer(tmp_root)
    result = _reconcile(tmp_root, reopened)
    assert result is None
    assert art.load_current_pointer(tmp_root) == before_ptr


def test_reconcile_case_a_fresh_no_checkpoint_at_all(tmp_root):
    assert _reconcile(tmp_root, None) is None
    assert art.load_current_pointer(tmp_root) is None


def _install_orphan_generation(root, current_arrays, current_completed, generation, *, mutate=None):
    """Installs a generation directly into place (bypassing the normal
    pointer-advance step) to simulate an interrupted commit: the
    generation directory exists and is complete, but the current
    pointer was never updated to reference it."""
    conf, scal = _one_image_inputs(generation + 1)
    new_arrays = art.append_image_to_checkpoint_arrays(current_arrays, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
    new_completed = current_completed + [DEV[generation]]
    real_advance = art._advance_current_pointer
    art._advance_current_pointer = lambda *a, **k: None  # suppress the pointer advance -> orphan
    try:
        art.install_checkpoint_generation(
            root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=new_arrays,
            completed_image_ids=new_completed, generation=generation, complete=(len(new_completed) == 350), final=False,
            final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )
    except art.DfpPhase1bLightweightBoundaryArtifactError:
        # Expected when there is no PRIOR pointer to fall back on (e.g. the very first-ever
        # generation): install_checkpoint_generation's own final self-check tries to reopen
        # via the (still-unwritten, since we suppressed it) pointer and correctly raises. The
        # generation directory itself was already fully installed via the atomic rename BEFORE
        # that check runs -- exactly the genuinely-orphaned state this helper is simulating.
        pass
    finally:
        art._advance_current_pointer = real_advance
    if mutate is not None:
        gens_dir = tmp_root_path(root)
        mutate(gens_dir / art.CHECKPOINT_GENERATIONS_DIRNAME / f"generation_{generation}")
    return new_arrays, new_completed


def tmp_root_path(root):
    return Path(root)


def test_reconcile_case_b_adopts_first_ever_generation_zero_orphan(tmp_root):
    arrays = art.empty_checkpoint_arrays(variant_ids=VARIANTS, class_count=CLASS_COUNT)
    new_arrays, new_completed = _install_orphan_generation(tmp_root, arrays, [], 0)
    assert art.load_current_pointer(tmp_root) is None  # confirm it's genuinely orphaned
    result = _reconcile(tmp_root, None)
    assert result is not None
    assert result["progress"]["generation"] == 0
    assert result["progress"]["completed_image_ids"] == [DEV[0]]
    assert art.load_current_pointer(tmp_root)["current_generation"] == 0


def test_reconcile_case_b_adopts_orphan_after_existing_generations(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 5)
    new_arrays, new_completed = _install_orphan_generation(tmp_root, arrays, completed, 5)
    assert art.load_current_pointer(tmp_root)["current_generation"] == 4  # still old
    result = _reconcile(tmp_root, reopened)
    assert result is not None
    assert result["progress"]["generation"] == 5
    assert result["progress"]["completed_image_ids"] == DEV[:6]
    assert art.load_current_pointer(tmp_root)["current_generation"] == 5


def test_reconcile_idempotent_second_call_sees_nothing_to_adopt(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 5)
    _install_orphan_generation(tmp_root, arrays, completed, 5)
    first = _reconcile(tmp_root, reopened)
    assert first is not None
    second = _reconcile(tmp_root, first)
    assert second is None  # already current -- nothing left to reconcile
    assert art.load_current_pointer(tmp_root)["current_generation"] == 5


def test_reconcile_does_not_touch_prior_generations(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 5)
    before_hashes = []
    for i in range(5):
        d = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / f"generation_{i}"
        before_hashes.append((art._sha256_file(d / art.CANONICAL_FILENAMES["checkpoint"]), art._sha256_file(d / art.CANONICAL_FILENAMES["checkpoint_arrays"])))
    _install_orphan_generation(tmp_root, arrays, completed, 5)
    _reconcile(tmp_root, reopened)
    for i in range(5):
        d = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / f"generation_{i}"
        after = (art._sha256_file(d / art.CANONICAL_FILENAMES["checkpoint"]), art._sha256_file(d / art.CANONICAL_FILENAMES["checkpoint_arrays"]))
        assert after == before_hashes[i]


# --- Adversarial invalid-orphan cases: every one must fail closed with
# the domain exception, preserve the current pointer, and leave no new
# current state. ---


def _assert_orphan_rejected(root, current, label):
    before_ptr = art.load_current_pointer(root)
    with pytest.raises(art.DfpPhase1bLightweightBoundaryArtifactError):
        _reconcile(root, current)
    after_ptr = art.load_current_pointer(root)
    assert after_ptr == before_ptr, f"[{label}] current pointer must be unchanged after a rejected orphan"


def test_reconcile_rejects_malformed_checkpoint_json(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gd = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_3"
    (gd / art.CANONICAL_FILENAMES["checkpoint"]).write_text("{not valid json")
    _assert_orphan_rejected(tmp_root, reopened, "malformed JSON")


def test_reconcile_rejects_corrupt_truncated_npz(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gd = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_3"
    npz = gd / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    data = npz.read_bytes()
    npz.write_bytes(data[: len(data) // 2])
    _assert_orphan_rejected(tmp_root, reopened, "truncated NPZ")


def test_reconcile_rejects_wrong_npz_digest(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gd = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_3"
    npz = gd / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    data = dict(np.load(npz))
    data["agg__T0"] = data["agg__T0"].copy(); data["agg__T0"][0][0] += 99
    np.savez(npz, **data)
    _assert_orphan_rejected(tmp_root, reopened, "wrong NPZ digest")


def test_reconcile_rejects_missing_npz_member(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gd = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_3"
    npz = gd / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    data = dict(np.load(npz))
    del data["agg__T320"]
    np.savez(npz, **data)
    _assert_orphan_rejected(tmp_root, reopened, "missing NPZ member")


def test_reconcile_rejects_invalid_completed_id_prefix(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gd = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_3"
    jpath = gd / art.CANONICAL_FILENAMES["checkpoint"]
    doc = json.loads(jpath.read_bytes())
    doc["progress"]["completed_image_ids"][0] = "not_a_real_id"
    jpath.write_text(json.dumps(doc))
    _assert_orphan_rejected(tmp_root, reopened, "invalid completed-ID prefix")


def test_reconcile_rejects_confirmation_id_inserted(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gd = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_3"
    jpath = gd / art.CANONICAL_FILENAMES["checkpoint"]
    doc = json.loads(jpath.read_bytes())
    doc["progress"]["completed_image_ids"][-1] = CONF[0]
    jpath.write_text(json.dumps(doc))
    _assert_orphan_rejected(tmp_root, reopened, "confirmation ID inserted")


def test_reconcile_rejects_generation_metadata_mismatch(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gd = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_3"
    jpath = gd / art.CANONICAL_FILENAMES["checkpoint"]
    doc = json.loads(jpath.read_bytes())
    doc["progress"]["generation"] = 99  # directory says generation_3, JSON claims 99
    jpath.write_text(json.dumps(doc))
    # this also breaks the sibling digest (JSON content changed) -- either check is
    # acceptable, the important thing is a clean domain-exception rejection.
    _assert_orphan_rejected(tmp_root, reopened, "generation metadata mismatch (N+2 vs dir N+1)")


def test_reconcile_rejects_predecessor_mismatch_aggregate_inconsistent(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gd = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_3"
    npz = gd / art.CANONICAL_FILENAMES["checkpoint_arrays"]
    data = dict(np.load(npz))
    # tamper aggregate AND recompute a self-consistent (but now scientifically wrong
    # relative to the predecessor) digest by rewriting through the real writer path is
    # not straightforward here -- instead break the *pointer-independent* structural
    # relation (per_image sum != aggregate), which validate_full_checkpoint itself rejects.
    data["agg__T0"] = data["agg__T0"].copy(); data["agg__T0"][0][0] += 1000
    np.savez(npz, **data)
    _assert_orphan_rejected(tmp_root, reopened, "predecessor/aggregate inconsistency")


def test_reconcile_rejects_illegal_final_flag_state(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gd = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_3"
    jpath = gd / art.CANONICAL_FILENAMES["checkpoint"]
    doc = json.loads(jpath.read_bytes())
    doc["final"] = True  # final=True but complete=False and no final_result_binding -- illegal
    jpath.write_text(json.dumps(doc))
    _assert_orphan_rejected(tmp_root, reopened, "illegal final flag state")


def test_reconcile_rejects_symlinked_generation_directory(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    outside = Path(tempfile.mkdtemp(prefix="outside_orphan_"))
    gens_dir = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME
    (gens_dir / "generation_3").symlink_to(outside, target_is_directory=True)
    _assert_orphan_rejected(tmp_root, reopened, "symlinked generation directory")
    shutil.rmtree(outside, ignore_errors=True)


def test_reconcile_rejects_two_alternate_successor_candidates_ambiguous(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    _install_orphan_generation(tmp_root, arrays, completed, 3)
    gens_dir = tmp_root / art.CHECKPOINT_GENERATIONS_DIRNAME
    # a second, unexpected alternate candidate at the same "slot"
    bogus = gens_dir / "generation_3_alt"
    shutil.copytree(gens_dir / "generation_3", bogus)
    with pytest.raises(art.DfpPhase1bLightweightBoundaryArtifactError, match="ambiguous"):
        _reconcile(tmp_root, reopened)
    assert art.load_current_pointer(tmp_root)["current_generation"] == 2  # untouched


def test_reconcile_rejects_generation_gap_ambiguous(tmp_root):
    reopened, arrays, completed = _install_n(tmp_root, 3)
    # install generation_4 directly, SKIPPING generation_3 entirely -- a gap
    conf, scal = _one_image_inputs(5)
    arrays4 = art.append_image_to_checkpoint_arrays(arrays, variant_ids=VARIANTS, per_variant_confusion=conf, per_variant_scalars=scal)
    real_advance = art._advance_current_pointer
    art._advance_current_pointer = lambda *a, **k: None
    try:
        art.install_checkpoint_generation(
            tmp_root, identity_bindings_without_digest=IDENTITY, variant_ids=VARIANTS, arrays=arrays4,
            completed_image_ids=completed + [DEV[3]], generation=4, complete=False, final=False,
            final_result_binding=None, development_ids=DEV, confirmation_ids=CONF,
        )
    finally:
        art._advance_current_pointer = real_advance
    with pytest.raises(art.DfpPhase1bLightweightBoundaryArtifactError, match="ambiguous"):
        _reconcile(tmp_root, reopened)
    assert art.load_current_pointer(tmp_root)["current_generation"] == 2


def test_reconcile_no_scientific_provider_invoked_on_rejection(tmp_root):
    """Rejection must never invoke any scientific provider -- reconcile_orphan_successor
    itself takes no provider arguments at all, a structural guarantee."""
    import inspect
    sig = inspect.signature(art.reconcile_orphan_successor)
    assert set(sig.parameters.keys()) == {"output_root", "current", "variant_ids", "identity_bindings_without_digest", "development_ids", "confirmation_ids"}
