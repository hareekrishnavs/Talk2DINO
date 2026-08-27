"""Tests for the shared VOC2012 V20/V21 matched-evaluator checkpoint
schema/invariant validation. CPU-only synthetic fixtures; no CUDA."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.voc2012_matched_evaluator_checkpoint import (  # noqa: E402
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_structure,
    resume_dataset_index,
)
from src.voc2012_matched_evaluator_identity import (  # noqa: E402
    Voc2012MatchedEvaluatorIdentityError,
    load_identity,
)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_voc2012_matched_evaluator.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the voc2012 matched-evaluator identity")


@pytest.fixture(scope="module")
def identity():
    return load_identity(repo_root=ROOT)


IDENTITY_SHA = "a" * 64


def _valid_checkpoint(identity, *, run_mode="pilot20", next_index=3, complete=False):
    image_count = identity["run_modes"][f"{run_mode}_image_count"]
    completed = [f"img{i}" for i in range(next_index)]
    return {
        "schema": identity["checkpoint"]["schema_name"], "run_mode": run_mode,
        "identity": identity["identity"]["name"], "identity_sha256": IDENTITY_SHA,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "voc2012_source_identity_sha256": identity["parent_identities"]["voc2012_source_identity_sha256"],
        "source_manifest_sha256": "b" * 64,
        "bridge_checkpoint_sha256": identity["model_and_checkpoint"]["projection_checkpoint_sha256"],
        "git_commit": "c" * 40,
        "v20_class_count": identity["v20_protocol"]["class_count"],
        "v21_class_count": identity["v21_protocol"]["class_count"],
        "live_v20_class_names_digest": "d" * 64, "live_v21_class_names_digest": "e" * 64,
        "image_count_expected": image_count, "image_order_digest": "f" * 64,
        "next_dataset_index": next_index if not complete else image_count,
        "completed_image_ids": completed if not complete else [f"img{i}" for i in range(image_count)],
        "images_completed_count": next_index if not complete else image_count,
        "windows_processed_total": (next_index if not complete else image_count) * 10,
        "complete": complete,
        "created_at_utc": "2026-01-01T00:00:00+00:00", "updated_at_utc": "2026-01-01T00:00:00+00:00",
    }


def test_valid_checkpoint_accepted(identity):
    checkpoint = _valid_checkpoint(identity)
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


@pytest.mark.parametrize("next_index", [0, 1, 10, 19])
def test_interrupted_positions_accepted(identity, next_index):
    checkpoint = _valid_checkpoint(identity, next_index=next_index)
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")
    assert resume_dataset_index(checkpoint) == next_index


def test_complete_checkpoint_at_final_minus_one_boundary(identity):
    checkpoint = _valid_checkpoint(identity, next_index=19)
    assert checkpoint["next_dataset_index"] == 19
    assert checkpoint["image_count_expected"] == 20


def test_complete_true_requires_full_prefix(identity):
    checkpoint = _valid_checkpoint(identity, complete=True)
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_false_complete_rejected_when_prefix_incomplete(identity):
    checkpoint = _valid_checkpoint(identity, next_index=5)
    checkpoint["complete"] = True  # 5 completed but claims complete for a 20-image pilot
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_resume_dataset_index_rejects_already_complete_checkpoint(identity):
    checkpoint = _valid_checkpoint(identity, complete=True)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        resume_dataset_index(checkpoint)


def test_next_index_mismatch_rejected(identity):
    checkpoint = _valid_checkpoint(identity, next_index=5)
    checkpoint["next_dataset_index"] = 4  # disagrees with len(completed_image_ids) == 5
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_duplicate_completed_ids_rejected(identity):
    checkpoint = _valid_checkpoint(identity, next_index=3)
    checkpoint["completed_image_ids"] = ["img0", "img0", "img1"]
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_images_completed_count_mismatch_rejected(identity):
    checkpoint = _valid_checkpoint(identity, next_index=3)
    checkpoint["images_completed_count"] = 2
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_wrong_identity_name_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    checkpoint["identity"] = "some-other-identity"
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_wrong_identity_sha256_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256="z" * 64, run_mode="pilot20")


def test_wrong_matched_identity_sha256_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    checkpoint["matched_identity_sha256"] = "0" * 64
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_wrong_voc2012_source_identity_sha256_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    checkpoint["voc2012_source_identity_sha256"] = "0" * 64
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_run_mode_disagreement_rejected(identity):
    checkpoint = _valid_checkpoint(identity, run_mode="pilot20")
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot100")


def test_wrong_image_count_expected_rejected(identity):
    checkpoint = _valid_checkpoint(identity, run_mode="pilot20")
    checkpoint["image_count_expected"] = 5000
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_changed_v20_class_count_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    checkpoint["v20_class_count"] = 19
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_changed_v21_class_count_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    checkpoint["v21_class_count"] = 22
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_live_class_count_mismatch_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(
            checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20", v20_class_count=99,
        )


def test_unknown_field_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    checkpoint["extra_unexpected_field"] = "x"
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_missing_field_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    del checkpoint["windows_processed_total"]
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_bool_as_int_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    checkpoint["next_dataset_index"] = True  # bool is not an acceptable int
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_float_as_int_rejected(identity):
    checkpoint = _valid_checkpoint(identity)
    checkpoint["images_completed_count"] = 3.0
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_reordered_completed_ids_rejected_against_canonical_order(identity):
    checkpoint = _valid_checkpoint(identity, next_index=3)
    canonical = ["img0", "img1", "img2", "img3", "img4"]
    checkpoint["completed_image_ids"] = ["img1", "img0", "img2"]  # reordered
    checkpoint["image_order_digest"] = "f" * 64
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_against_canonical_order(checkpoint, canonical, image_order_digest="f" * 64)


def test_skipped_id_rejected_against_canonical_order(identity):
    checkpoint = _valid_checkpoint(identity, next_index=3)
    canonical = ["img0", "img1", "img2", "img3", "img4"]
    checkpoint["completed_image_ids"] = ["img0", "img1", "img9"]  # img9 not in canonical prefix
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_against_canonical_order(checkpoint, canonical, image_order_digest="f" * 64)


def test_wrong_image_order_digest_rejected(identity):
    checkpoint = _valid_checkpoint(identity, next_index=3)
    canonical = ["img0", "img1", "img2", "img3", "img4"]
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_against_canonical_order(checkpoint, canonical, image_order_digest="different" * 8)


def test_matching_canonical_order_accepted(identity):
    checkpoint = _valid_checkpoint(identity, next_index=3)
    canonical = ["img0", "img1", "img2", "img3", "img4"]
    validate_checkpoint_against_canonical_order(checkpoint, canonical, image_order_digest="f" * 64)


def test_validate_checkpoint_structure_never_mutates_input(identity):
    checkpoint = _valid_checkpoint(identity)
    before = copy.deepcopy(checkpoint)
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")
    assert checkpoint == before


# ---------------------------------------------------------------------
# bridge_checkpoint_sha256 relational validation: the checkpoint's
# recorded value must equal identity["model_and_checkpoint"]["projection_checkpoint_sha256"]
# exactly, not merely be well-formed hex.
# ---------------------------------------------------------------------


def test_bridge_checkpoint_sha256_matching_value_accepted(identity):
    checkpoint = _valid_checkpoint(identity)
    assert checkpoint["bridge_checkpoint_sha256"] == identity["model_and_checkpoint"]["projection_checkpoint_sha256"]
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")


def test_bridge_checkpoint_sha256_wrong_value_rejected_on_incomplete_checkpoint(identity):
    checkpoint = _valid_checkpoint(identity, next_index=3)
    checkpoint["bridge_checkpoint_sha256"] = "0" * 64
    before = copy.deepcopy(checkpoint)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")
    assert checkpoint == before


def test_bridge_checkpoint_sha256_wrong_value_rejected_on_resumable_checkpoint(identity):
    checkpoint = _valid_checkpoint(identity, next_index=10)  # mid-run, resumable
    checkpoint["bridge_checkpoint_sha256"] = "1" * 64
    before = copy.deepcopy(checkpoint)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")
    assert checkpoint == before


def test_bridge_checkpoint_sha256_wrong_value_rejected_on_complete_checkpoint(identity):
    checkpoint = _valid_checkpoint(identity, complete=True)
    checkpoint["bridge_checkpoint_sha256"] = "2" * 64
    before = copy.deepcopy(checkpoint)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=IDENTITY_SHA, run_mode="pilot20")
    assert checkpoint == before


def test_bridge_checkpoint_sha256_wrong_value_rejected_before_dataset_access():
    """AST-based: confirm the bridge_checkpoint_sha256 relational check
    happens inside validate_checkpoint_structure -- the SAME CPU-only
    function the driver calls in PHASE A, strictly before any dataset
    resolution or model/CUDA work (see run_voc2012_matched_evaluation.py:
    validate_checkpoint_structure is called before load_voc2012_source_identity/
    resolve_dataset_root/import torch)."""
    import inspect

    source = inspect.getsource(validate_checkpoint_structure)
    assert "projection_checkpoint_sha256" in source
