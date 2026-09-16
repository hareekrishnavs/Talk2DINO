"""Tests for the DFP Phase 1B BALP development window-universe
materializer/verifier (`src.dfp_phase1b_development_window_universe`).
Exercises the real digest/schema/record-building logic against the
real committed E13 crop/stride contract and BALP split artifact --
never a mock of the validator itself. No torch/model/dataset/CUDA
access: per-image geometry used here is synthetic (a fixed formula,
never claimed to match real image pixels) -- only
`independently_reconstruct_and_verify`, exercised separately against
the real dataset (see `verify_dfp_phase1b_development_windows.py`),
checks that real geometry agrees with a persisted manifest."""

from __future__ import annotations

import copy
import hashlib
import tomllib
from pathlib import Path

import pytest

import src.dfp_phase1b_development_window_universe as wu
from src.dfp_phase1a_horizon_stitching import dfp_stitching_contract_from_protocol
from src.dfp_phase1b_lightweight_boundary_identity import load_and_verify_split
from src.open_vocabulary_segmentation.segmentation.evaluation.sliding_window_geometry import (
    SlidingWindowPlan,
    SpatialSize,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
E = wu.DfpPhase1bDevelopmentWindowUniverseError


def _load_protocol_identity():
    with open(REPO_ROOT / "evaluation_identities/e13_dfp_phase1a_protocol.toml", "rb") as fh:
        return tomllib.load(fh)


def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"hello world")
    assert wu.sha256_file(target) == hashlib.sha256(b"hello world").hexdigest()


def test_window_geometry_to_record_fields():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(597, 448), crop_size=SpatialSize(448, 448), stride=SpatialSize(224, 224),
    )
    record = wu.window_geometry_to_record(plan.windows[0])
    assert record["index"] == 0
    assert record["grid_row"] == 0 and record["grid_col"] == 0
    assert record["origin"] == [0, 0]
    assert record["is_top"] is True and record["is_left"] is True
    assert set(record.keys()) == {
        "index", "grid_row", "grid_col", "nominal_origin", "origin", "end",
        "is_top", "is_bottom", "is_left", "is_right", "clamped_vertical", "clamped_horizontal",
    }


def test_build_image_window_record_window_count_matches_plan():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(597, 448), crop_size=SpatialSize(448, 448), stride=SpatialSize(224, 224),
    )
    record = wu.build_image_window_record(
        rank=5, canonical_id="000000314264", inference_height=597, inference_width=448, patch_size=14, plan=plan,
    )
    assert record["window_count"] == plan.window_count == len(record["windows"])
    assert record["rank"] == 5
    assert record["canonical_id"] == "000000314264"


def test_compute_dependency_bindings_matches_direct_hash():
    bindings = wu.compute_dependency_bindings(REPO_ROOT)
    assert set(bindings.keys()) == set(wu._DEPENDENCY_RELATIVE_PATHS.keys())
    for name, relative_path in wu._DEPENDENCY_RELATIVE_PATHS.items():
        assert bindings[name]["relative_path"] == str(relative_path)
        assert bindings[name]["sha256"] == wu.sha256_file(REPO_ROOT / relative_path)


def _fabricate_valid_document() -> dict:
    """A structurally valid, internally self-consistent document built
    from the real committed 350 development IDs and the real E13
    crop/stride contract, but with SYNTHETIC per-image geometry --
    sufficient to exercise `verify_window_universe_document`'s own
    internal-consistency and dependency-hash checks without touching
    the real dataset."""
    split = load_and_verify_split(repo_root=REPO_ROOT)
    development_ids = list(split["development_image_ids"])
    contract = dfp_stitching_contract_from_protocol(_load_protocol_identity(), repo_root=REPO_ROOT)
    crop = SpatialSize(*contract.crop)
    stride = SpatialSize(*contract.stride)

    images = []
    aggregate = 0
    for rank, canonical_id in enumerate(development_ids):
        height = 448 + (rank % 5) * 37
        width = 448 + ((rank * 3) % 7) * 29
        plan = SlidingWindowPlan.build(image_size=SpatialSize(height, width), crop_size=crop, stride=stride)
        record = wu.build_image_window_record(
            rank=rank, canonical_id=canonical_id, inference_height=height, inference_width=width,
            patch_size=14, plan=plan,
        )
        images.append(record)
        aggregate += plan.window_count

    canonical_record_digest = wu._canonical_json_digest(images)
    per_image_window_counts_digest = wu._canonical_json_digest(
        [{"canonical_id": r["canonical_id"], "window_count": r["window_count"]} for r in images]
    )
    return {
        "schema_name": wu.SCHEMA_NAME,
        "schema_version": wu.SCHEMA_VERSION,
        "namespace": wu.SCHEMA_NAME,
        "source_development_digest": split["development_digest"],
        "development_count": len(development_ids),
        "crop": list(contract.crop),
        "stride": list(contract.stride),
        "align_corners": contract.align_corners,
        "images": images,
        "aggregate_window_count": aggregate,
        "canonical_record_digest": canonical_record_digest,
        "per_image_window_counts_digest": per_image_window_counts_digest,
        "dependency_bindings": wu.compute_dependency_bindings(REPO_ROOT),
    }


@pytest.fixture(scope="module")
def valid_document():
    return _fabricate_valid_document()


def test_verify_window_universe_document_accepts_valid_document(valid_document):
    wu.verify_window_universe_document(valid_document, repo_root=REPO_ROOT)  # must not raise


def test_rejects_wrong_schema_name(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["schema_name"] = "wrong-schema"
    with pytest.raises(E, match="schema_name"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)


def test_rejects_wrong_development_count(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["images"] = doc["images"][:349]
    with pytest.raises(E, match="350"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)


def test_rejects_rank_mismatch(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["images"][3]["rank"] = 999
    with pytest.raises(E, match="rank"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)


def test_rejects_window_count_disagreeing_with_windows_list(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["images"][0]["window_count"] = doc["images"][0]["window_count"] + 1
    with pytest.raises(E, match="window_count"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)


def test_rejects_tampered_aggregate_window_count(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["aggregate_window_count"] += 1
    with pytest.raises(E, match="aggregate_window_count"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)


def test_rejects_tampered_canonical_record_digest(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["canonical_record_digest"] = "0" * 64
    with pytest.raises(E, match="canonical_record_digest"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)


def test_rejects_tampered_per_image_window_counts_digest(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["per_image_window_counts_digest"] = "0" * 64
    with pytest.raises(E, match="per_image_window_counts_digest"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)


def test_rejects_tampered_dependency_bindings(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["dependency_bindings"]["e13_protocol"]["sha256"] = "0" * 64
    with pytest.raises(E, match="dependency_bindings"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)


def test_rejects_tampered_source_development_digest(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["source_development_digest"] = "0" * 64
    with pytest.raises(E, match="source_development_digest"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)


def test_rejects_reordered_images(valid_document):
    doc = copy.deepcopy(valid_document)
    doc["images"][0], doc["images"][1] = doc["images"][1], doc["images"][0]
    doc["images"][0]["rank"], doc["images"][1]["rank"] = doc["images"][1]["rank"], doc["images"][0]["rank"]
    # ranks now self-consistent again but canonical_record_digest was
    # computed over the ORIGINAL order, so this must still be rejected.
    with pytest.raises(E, match="canonical_record_digest"):
        wu.verify_window_universe_document(doc, repo_root=REPO_ROOT)
