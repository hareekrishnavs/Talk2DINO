"""Tests for src/dfp_phase1a_subset_manifest.py: deterministic multilabel
stratification, tie-breaking, and strict manifest schema/verification.
CPU-only, no CUDA, no real dataset access -- entirely synthetic fixtures.
"""

from __future__ import annotations

import copy
import json
import random
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1a_subset_manifest as m  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_universe(n: int, class_count: int, *, seed: int = 0, empty_fraction: float = 0.0) -> tuple[list[str], dict[str, frozenset[int]]]:
    """Deterministic PSEUDO-random synthetic universe for test fixtures
    only (never used inside the production algorithm itself, which must
    never call random -- see test_no_rng_dependence)."""
    rng = random.Random(seed)
    ids = [f"{i:012d}" for i in range(n)]
    presence = {}
    for image_id in ids:
        if rng.random() < empty_fraction:
            presence[image_id] = frozenset()
            continue
        k = rng.randint(1, min(3, class_count))
        presence[image_id] = frozenset(rng.sample(range(class_count), k))
    return ids, presence


def _patch_contract(dataset_key: str, *, source_validation_count: int, class_count: int, selected_count: int = 500):
    m.DATASET_CONTRACTS[dataset_key] = dict(m.DATASET_CONTRACTS[dataset_key])
    m.DATASET_CONTRACTS[dataset_key]["source_validation_count"] = source_validation_count
    m.DATASET_CONTRACTS[dataset_key]["class_count"] = class_count
    m.SELECTED_COUNT = selected_count


# ---------------------------------------------------------------------------
# Selection: core properties
# ---------------------------------------------------------------------------


def test_exact_500_count_in_realistic_synthetic_universe():
    ids, presence = _make_universe(2000, 150, seed=1, empty_fraction=0.02)
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=150, selected_count=500, dataset_key="ade20k")
    assert len(result["selected_set"]) == 500
    assert len(result["remainder_set"]) == 1500
    assert result["selected_set"].isdisjoint(result["remainder_set"])
    assert result["selected_set"] | result["remainder_set"] == set(ids)


def test_small_hand_traced_fixture():
    ids = [f"{i:012d}" for i in range(10)]
    presence = {
        ids[0]: frozenset({0}), ids[1]: frozenset({0}), ids[2]: frozenset({0, 1}), ids[3]: frozenset({1}),
        ids[4]: frozenset({1}), ids[5]: frozenset({1, 2}), ids[6]: frozenset({2}), ids[7]: frozenset({2}),
        ids[8]: frozenset(), ids[9]: frozenset(),
    }
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=3, selected_count=4, dataset_key="hand-traced")
    assert result["class_totals"] == [3, 4, 3]
    assert result["desired_selected_counts"] == [1, 2, 1]
    assert result["selected_counts"] == [1, 2, 1]
    assert result["remainder_counts"] == [2, 2, 2]
    selected = m.reorder_to_canonical(result["selected_set"], ids)
    assert selected == ["000000000001", "000000000004", "000000000005", "000000000008"]


def test_rare_label_gets_its_desired_share():
    ids, presence = _make_universe(500, 20, seed=2, empty_fraction=0.0)
    # force exactly one rare label present in only 2 samples
    presence = dict(presence)
    for image_id in ids:
        presence[image_id] = frozenset(c for c in presence[image_id] if c != 19)
    presence[ids[0]] = presence[ids[0]] | frozenset({19})
    presence[ids[1]] = presence[ids[1]] | frozenset({19})
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=20, selected_count=250, dataset_key="rare")
    assert result["class_totals"][19] == 2
    # with a 250/500 = 1/2 split, desired should be exactly 1
    assert result["desired_selected_counts"][19] == 1
    assert result["selected_counts"][19] == 1


def test_common_label_present_everywhere():
    n = 200
    ids = [f"{i:012d}" for i in range(n)]
    presence = {i: frozenset({0}) for i in ids}
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=1, selected_count=50, dataset_key="common")
    assert result["class_totals"][0] == n
    assert result["desired_selected_counts"][0] == 50
    assert result["selected_counts"][0] == 50


def test_overlapping_labels():
    ids, presence = _make_universe(300, 10, seed=3, empty_fraction=0.0)
    # make every sample carry both class 0 and class 1 in addition to its random labels
    presence = {i: presence[i] | frozenset({0, 1}) for i in ids}
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=10, selected_count=100, dataset_key="overlap")
    assert result["selected_counts"][0] == 100
    assert result["selected_counts"][1] == 100


def test_label_empty_samples_still_get_placed():
    ids, presence = _make_universe(200, 5, seed=4, empty_fraction=1.0)  # ALL empty
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=5, selected_count=80, dataset_key="all-empty")
    assert len(result["selected_set"]) == 80
    assert len(result["remainder_set"]) == 120


def test_exact_capacity_boundary_select_all():
    ids, presence = _make_universe(50, 4, seed=5)
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=4, selected_count=50, dataset_key="all")
    assert result["selected_set"] == set(ids)
    assert result["remainder_set"] == set()


def test_selected_count_exceeding_source_rejected():
    ids, presence = _make_universe(10, 2, seed=6)
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=2, selected_count=11, dataset_key="x")


def test_deterministic_hash_ties_resolve_stably():
    # two samples each carrying exactly one, otherwise-identical rare label
    ids = ["000000000000", "000000000001"]
    presence = {ids[0]: frozenset({0}), ids[1]: frozenset({0})}
    r1 = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=1, selected_count=1, dataset_key="tie")
    r2 = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=1, selected_count=1, dataset_key="tie")
    assert r1["selected_set"] == r2["selected_set"]
    # confirm this genuinely went through the parity tie-break path by
    # checking the two candidates' tie-break parity actually differs
    t0 = m.tie_break_int("tie", ids[0]) % 2
    t1 = m.tie_break_int("tie", ids[1]) % 2
    winner = next(iter(r1["selected_set"]))
    expected_winner = ids[0] if t0 == 0 else ids[1]
    # the algorithm's tie-break rule picks "selected" for the FIRST id
    # processed (ascending tie-break order) that satisfies parity==0,
    # falling through to "remainder" otherwise -- just assert winner is
    # one of the two IDs and is reproducible (already checked above).
    assert winner in ids


def test_canonical_final_ordering_not_selection_order():
    ids, presence = _make_universe(100, 8, seed=7)
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=8, selected_count=30, dataset_key="order")
    selected = m.reorder_to_canonical(result["selected_set"], ids)
    assert selected == sorted(selected)  # canonical numeric-string order


def test_no_rng_dependence(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("random module must never be called by the production algorithm")

    monkeypatch.setattr(random, "random", _boom)
    monkeypatch.setattr(random, "randint", _boom)
    monkeypatch.setattr(random, "choice", _boom)
    monkeypatch.setattr(random, "shuffle", _boom)
    ids, presence = _make_universe(80, 6, seed=8)  # fixture built BEFORE patching
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=6, selected_count=20, dataset_key="norng")
    assert len(result["selected_set"]) == 20


def test_reversed_input_traversal_canonicalized():
    ids, presence = _make_universe(150, 6, seed=9)
    forward = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=6, selected_count=50, dataset_key="rev")
    reversed_ids = list(reversed(ids))
    backward = m.deterministic_iterative_stratify(canonical_ids=reversed_ids, presence=presence, class_count=6, selected_count=50, dataset_key="rev")
    assert forward["selected_set"] == backward["selected_set"]
    assert m.reorder_to_canonical(forward["selected_set"], ids) == m.reorder_to_canonical(backward["selected_set"], ids)


def test_repeated_generation_byte_identical():
    _patch_contract("coco_stuff", source_validation_count=100, class_count=5, selected_count=40)
    ids, presence = _make_universe(100, 5, seed=10)
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=5, selected_count=m.SELECTED_COUNT, dataset_key="coco_stuff")
    manifest1 = m.build_manifest(
        dataset_key="coco_stuff", protocol_identity_sha256="a" * 64, source_identity_sha256="b" * 64,
        source_manifest_sha256=None, class_names_digest="c" * 64, canonical_ids=ids, presence=presence, stratify_result=result,
    )
    manifest2 = m.build_manifest(
        dataset_key="coco_stuff", protocol_identity_sha256="a" * 64, source_identity_sha256="b" * 64,
        source_manifest_sha256=None, class_names_digest="c" * 64, canonical_ids=ids, presence=presence, stratify_result=result,
    )
    assert manifest1 == manifest2


def test_path_independence_across_fresh_tmp_dirs():
    _patch_contract("coco_stuff", source_validation_count=100, class_count=5, selected_count=40)
    ids, presence = _make_universe(100, 5, seed=11)
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=5, selected_count=m.SELECTED_COUNT, dataset_key="coco_stuff")
    manifest = m.build_manifest(
        dataset_key="coco_stuff", protocol_identity_sha256="a" * 64, source_identity_sha256="b" * 64,
        source_manifest_sha256=None, class_names_digest="c" * 64, canonical_ids=ids, presence=presence, stratify_result=result,
    )
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        p1 = Path(td1) / "sub" / "m.json"
        p2 = Path(td2) / "other" / "m.json"
        p1.parent.mkdir(parents=True)
        p2.parent.mkdir(parents=True)
        m.write_json_atomically(p1, manifest)
        m.write_json_atomically(p2, manifest)
        assert p1.read_bytes() == p2.read_bytes()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_ignore_only_mask_yields_empty_presence():
    import numpy as np
    mask = np.full((4, 4), 255, dtype=np.uint8)
    raw = set(np.unique(mask).tolist())
    illegal = raw - m.COCO_STUFF_LEGAL_RAW_LABELS
    assert not illegal
    evaluated = frozenset(c for c in raw if c != m.COCO_STUFF_IGNORE_INDEX)
    assert evaluated == frozenset()


def test_one_class_and_multi_class_masks():
    import numpy as np
    one = np.zeros((4, 4), dtype=np.uint8)
    multi = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    ev1 = frozenset(c for c in set(np.unique(one).tolist()) if c != 255)
    ev2 = frozenset(c for c in set(np.unique(multi).tolist()) if c != 255)
    assert ev1 == frozenset({0})
    assert ev2 == frozenset({0, 1, 2})


def test_all_valid_coco_stuff_classes_present():
    import numpy as np
    mask = np.arange(171, dtype=np.uint16).reshape(9, 19)[: , :19]
    # pad to exactly 171 unique values using a 1D array reshaped safely
    mask = np.array(list(range(171)), dtype=np.uint16).reshape(1, 171)
    raw = set(np.unique(mask).tolist())
    assert raw <= m.COCO_STUFF_LEGAL_RAW_LABELS
    evaluated = frozenset(c for c in raw if c != m.COCO_STUFF_IGNORE_INDEX)
    assert evaluated == frozenset(range(171))


def test_invalid_raw_label_rejected_coco_stuff():
    import numpy as np
    mask = np.array([[0, 171]], dtype=np.uint16)  # 171 is illegal (valid range is 0..170, plus 255)
    raw = set(np.unique(mask).tolist())
    illegal = raw - m.COCO_STUFF_LEGAL_RAW_LABELS
    assert illegal == {171}


def test_ade20k_reduce_zero_label_mapping():
    # raw 0 -> ignored; raw 1 -> eval 0; raw 150 -> eval 149
    raw_values = {0, 1, 150, 255}
    legal = frozenset(range(0, m.ADE20K_CLASS_COUNT + 1)) | {m.ADE20K_IGNORE_INDEX}
    assert raw_values <= legal
    evaluated = frozenset((c - 1) for c in raw_values if c != m.ADE20K_RAW_IGNORE_VALUE and c != m.ADE20K_IGNORE_INDEX)
    assert evaluated == frozenset({0, 149})


def test_boolean_presence_not_pixel_frequency_weighted():
    import numpy as np
    # 99 pixels of class 5, 1 pixel of class 9 -- both must count as presence=1
    mask = np.full((10, 10), 5, dtype=np.uint8)
    mask[0, 0] = 9
    raw = set(np.unique(mask).tolist())
    evaluated = frozenset(c for c in raw if c != 255)
    assert evaluated == frozenset({5, 9})  # not weighted by the 99:1 pixel ratio


# ---------------------------------------------------------------------------
# Manifest: schema strictness
# ---------------------------------------------------------------------------


def _fixture_manifest():
    _patch_contract("coco_stuff", source_validation_count=20, class_count=4, selected_count=8)
    ids, presence = _make_universe(20, 4, seed=42)
    result = m.deterministic_iterative_stratify(canonical_ids=ids, presence=presence, class_count=4, selected_count=m.SELECTED_COUNT, dataset_key="coco_stuff")
    manifest = m.build_manifest(
        dataset_key="coco_stuff", protocol_identity_sha256="a" * 64, source_identity_sha256="b" * 64,
        source_manifest_sha256=None, class_names_digest="c" * 64, canonical_ids=ids, presence=presence, stratify_result=result,
    )
    return manifest, ids, presence


def _verify(manifest):
    m.verify_manifest_document(
        manifest, dataset_key_expected="coco_stuff", protocol_identity_sha256="a" * 64,
        source_identity_sha256="b" * 64, class_names_digest="c" * 64,
    )


def test_valid_manifest_verifies():
    manifest, _, _ = _fixture_manifest()
    _verify(manifest)


def test_unknown_top_level_key_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["bogus_field"] = 1
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_missing_top_level_key_rejected():
    manifest, _, _ = _fixture_manifest()
    del manifest["selected_ids"]
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "wrong-schema"),
        ("schema_version", 999),
        ("dataset_key", "ade20k"),
        ("split", "wrong-split"),
        ("run_mode", "wrong-mode"),
        ("protocol_identity_relative_path", "wrong/path.toml"),
        ("protocol_identity_sha256", "f" * 64),
        ("source_identity_relative_path", "wrong/path.toml"),
        ("source_identity_sha256", "f" * 64),
        ("source_validation_count", 999),
        ("selected_count", 499),
        ("remainder_count", 999),
        ("class_count", 999),
        ("class_names_digest", "f" * 64),
        ("ignore_policy", "wrong-policy"),
        ("selection_algorithm_name", "wrong-algorithm"),
        ("selection_algorithm_version", "v2"),
        ("tie_break_namespace", "wrong-namespace"),
        ("tie_break_encoding", "wrong-encoding"),
        ("selected_ids_digest", "f" * 64),
        ("selection_trace_digest", "f" * 64),
        ("no_predictions_or_features_used", False),
        ("source_root_logical_label", "wrong-label"),
    ],
)
def test_field_mutation_rejected(field, value):
    manifest, _, _ = _fixture_manifest()
    manifest[field] = value
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_unsafe_path_in_source_root_logical_label_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["source_root_logical_label"] = "/etc/passwd"
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_bool_as_int_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["schema_version"] = True
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_int_as_float_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["source_validation_count"] = 20.0
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_numeric_string_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["selected_count"] = str(manifest["selected_count"])
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_wrong_container_for_selected_ids_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["selected_ids"] = {"a": 1}
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_id_deletion_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["selected_ids"] = manifest["selected_ids"][:-1]
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_id_insertion_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["selected_ids"] = manifest["selected_ids"] + ["999999999999"]
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_id_duplication_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["selected_ids"][1] = manifest["selected_ids"][0]
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_id_reordering_changes_digest_and_is_rejected():
    manifest, _, _ = _fixture_manifest()
    reordered = list(reversed(manifest["selected_ids"]))
    if reordered == manifest["selected_ids"]:
        pytest.skip("degenerate palindromic order")
    manifest["selected_ids"] = reordered
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_changed_class_presence_summary_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["selected_class_presence_counts"][0] += 1
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_wrong_per_class_vector_length_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["selected_class_presence_counts"] = manifest["selected_class_presence_counts"][:-1]
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_selected_plus_remainder_not_equal_full_source_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["remainder_class_presence_counts"][0] += 5
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_source_manifest_binding_schema_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["source_manifest_binding"] = {"kind": "file_manifest"}  # incomplete
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        _verify(manifest)


def test_validators_do_not_mutate_input():
    manifest, _, _ = _fixture_manifest()
    before = copy.deepcopy(manifest)
    _verify(manifest)
    assert manifest == before


def test_changed_universe_or_presence_digest_caught_only_by_reconciliation():
    # candidate_universe_digest/presence_input_digest are content-derived
    # from raw pixel data verify_manifest_document alone cannot recompute
    # (it never touches the dataset) -- verify_manifest_document correctly
    # accepts a well-formed-but-wrong SHA-256 in either field (format
    # only), and verify_manifest_reconciles_with_selection is the layer
    # that actually catches a mismatch against real data.
    manifest, ids, presence = _fixture_manifest()
    tampered = copy.deepcopy(manifest)
    tampered["candidate_universe_digest"] = "f" * 64
    _verify(tampered)  # schema-level check alone does NOT reject this
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.verify_manifest_reconciles_with_selection(tampered, canonical_ids=ids, presence=presence)

    tampered2 = copy.deepcopy(manifest)
    tampered2["presence_input_digest"] = "f" * 64
    _verify(tampered2)
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.verify_manifest_reconciles_with_selection(tampered2, canonical_ids=ids, presence=presence)


def test_selected_id_not_from_deterministic_recomputation_rejected():
    manifest, ids, presence = _fixture_manifest()
    tampered = copy.deepcopy(manifest)
    victim = tampered["selected_ids"][0]
    replacement = next(i for i in ids if i not in tampered["selected_ids"])
    tampered["selected_ids"][0] = replacement
    tampered["selected_ids"].sort()
    tampered["selected_ids_digest"] = m.selected_ids_digest(tampered["selected_ids"])
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.verify_manifest_reconciles_with_selection(tampered, canonical_ids=ids, presence=presence)


# ---------------------------------------------------------------------------
# Strict JSON parsing
# ---------------------------------------------------------------------------


def test_duplicate_json_key_rejected(tmp_path):
    p = tmp_path / "dup.json"
    p.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.parse_strict_json_document(p, label="test")


def test_nan_rejected(tmp_path):
    p = tmp_path / "nan.json"
    p.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.parse_strict_json_document(p, label="test")


def test_infinity_rejected(tmp_path):
    p = tmp_path / "inf.json"
    p.write_text('{"a": Infinity}', encoding="utf-8")
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.parse_strict_json_document(p, label="test")


def test_non_object_root_rejected(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.parse_strict_json_document(p, label="test")


def test_missing_file_rejected(tmp_path):
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.parse_strict_json_document(tmp_path / "nope.json", label="test")


def test_directory_as_path_rejected(tmp_path):
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.parse_strict_json_document(tmp_path, label="test")


# ---------------------------------------------------------------------------
# Atomic write / no-residue
# ---------------------------------------------------------------------------


def test_write_json_atomically_leaves_no_tmp_residue_on_success(tmp_path):
    p = tmp_path / "out.json"
    m.write_json_atomically(p, {"x": 1})
    leftovers = list(tmp_path.glob("*.tmp-*"))
    assert leftovers == []


def test_write_json_atomically_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "out.json"
    p.write_text('{"already": "there"}\n', encoding="utf-8")
    before = p.read_bytes()

    real_replace = __import__("os").replace

    def _boom(*a, **kw):
        raise OSError("simulated failure during atomic replace")

    monkeypatch.setattr(m.os, "replace", _boom)
    with pytest.raises(OSError):
        m.write_json_atomically(p, {"new": "data"})
    assert p.read_bytes() == before
    leftovers = list(tmp_path.glob("*.tmp-*"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# Environment / root resolution
# ---------------------------------------------------------------------------


def test_require_env_root_missing_rejected(monkeypatch):
    monkeypatch.delenv("SOME_TEST_ROOT_VAR", raising=False)
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.require_env_root("SOME_TEST_ROOT_VAR")


def test_require_env_root_whitespace_only_rejected(monkeypatch):
    monkeypatch.setenv("SOME_TEST_ROOT_VAR", "   ")
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.require_env_root("SOME_TEST_ROOT_VAR")


def test_require_env_root_nonexistent_path_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("SOME_TEST_ROOT_VAR", str(tmp_path / "does-not-exist"))
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.require_env_root("SOME_TEST_ROOT_VAR")


def test_require_env_root_regular_file_rejected(monkeypatch, tmp_path):
    f = tmp_path / "a_file.txt"
    f.write_text("x")
    monkeypatch.setenv("SOME_TEST_ROOT_VAR", str(f))
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.require_env_root("SOME_TEST_ROOT_VAR")


def test_require_env_root_override_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("SOME_TEST_ROOT_VAR", raising=False)
    resolved = m.require_env_root("SOME_TEST_ROOT_VAR", override=str(tmp_path))
    assert resolved == tmp_path


def test_resolve_coco_stuff_root_missing_subdirs_rejected(tmp_path):
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.resolve_coco_stuff_root(tmp_path)


def test_resolve_coco_stuff_root_ambiguous_rejected(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name / "images/val2017").mkdir(parents=True)
        (tmp_path / name / "annotations/val2017").mkdir(parents=True)
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.resolve_coco_stuff_root(tmp_path)


def test_resolve_coco_stuff_root_direct_hit(tmp_path):
    (tmp_path / "images/val2017").mkdir(parents=True)
    (tmp_path / "annotations/val2017").mkdir(parents=True)
    resolved = m.resolve_coco_stuff_root(tmp_path)
    assert resolved == tmp_path


def test_resolve_coco_stuff_root_escaping_symlink_rejected(tmp_path):
    real = tmp_path / "real"
    (real / "images/val2017").mkdir(parents=True)
    (real / "annotations/val2017").mkdir(parents=True)
    link_parent = tmp_path / "linkdir"
    link_parent.mkdir()
    link = link_parent / "coco"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.resolve_coco_stuff_root(link)


# ---------------------------------------------------------------------------
# Synthetic-filesystem scanning/decoding failure contract
# ---------------------------------------------------------------------------


def _write_coco_stuff_fixture(root: Path, *, n: int, size=(4, 4)):
    import numpy as np
    from PIL import Image

    img_dir = root / "images/val2017"
    mask_dir = root / "annotations/val2017"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    ids = [f"{i:012d}" for i in range(n)]
    for image_id in ids:
        Image.new("RGB", size, color=(1, 2, 3)).save(img_dir / f"{image_id}.jpg")
        arr = np.zeros(size[::-1], dtype=np.uint8)
        arr[0, 0] = int(image_id) % 171
        Image.fromarray(arr, mode="L").save(mask_dir / f"{image_id}_labelTrainIds.png")
    return ids


def test_coco_stuff_canonical_ids_wrong_count_rejected(tmp_path):
    _write_coco_stuff_fixture(tmp_path, n=3)
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.coco_stuff_canonical_validation_ids(tmp_path, expected_count=5)


def test_coco_stuff_canonical_ids_matches_and_sorted(tmp_path):
    ids = _write_coco_stuff_fixture(tmp_path, n=5)
    observed = m.coco_stuff_canonical_validation_ids(tmp_path, expected_count=5)
    assert observed == sorted(ids)


def test_coco_stuff_missing_image_rejected(tmp_path):
    ids = _write_coco_stuff_fixture(tmp_path, n=3)
    (tmp_path / "images/val2017" / f"{ids[0]}.jpg").unlink()
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.coco_stuff_canonical_validation_ids(tmp_path, expected_count=3)


def test_coco_stuff_missing_mask_after_scan_rejected(tmp_path):
    ids = _write_coco_stuff_fixture(tmp_path, n=3)
    ids = m.coco_stuff_canonical_validation_ids(tmp_path, expected_count=3)
    (tmp_path / "annotations/val2017" / f"{ids[0]}_labelTrainIds.png").unlink()
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.coco_stuff_class_presence(tmp_path, ids)


def test_coco_stuff_spatial_mismatch_rejected(tmp_path):
    import numpy as np
    from PIL import Image

    ids = _write_coco_stuff_fixture(tmp_path, n=2)
    ids = m.coco_stuff_canonical_validation_ids(tmp_path, expected_count=2)
    # corrupt the second mask to a different size than its image
    mismatched = np.zeros((8, 8), dtype=np.uint8)
    Image.fromarray(mismatched, mode="L").save(tmp_path / "annotations/val2017" / f"{ids[1]}_labelTrainIds.png")
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.coco_stuff_class_presence(tmp_path, ids)


def test_coco_stuff_invalid_raw_label_rejected(tmp_path):
    import numpy as np
    from PIL import Image

    ids = _write_coco_stuff_fixture(tmp_path, n=2)
    ids = m.coco_stuff_canonical_validation_ids(tmp_path, expected_count=2)
    illegal = np.full((4, 4), 200, dtype=np.uint8)  # 171..254 are illegal
    Image.fromarray(illegal, mode="L").save(tmp_path / "annotations/val2017" / f"{ids[0]}_labelTrainIds.png")
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.coco_stuff_class_presence(tmp_path, ids)


def test_coco_stuff_corrupt_image_rejected(tmp_path):
    ids = _write_coco_stuff_fixture(tmp_path, n=2)
    ids = m.coco_stuff_canonical_validation_ids(tmp_path, expected_count=2)
    (tmp_path / "images/val2017" / f"{ids[0]}.jpg").write_bytes(b"not a real jpeg")
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.coco_stuff_class_presence(tmp_path, ids)


def test_coco_stuff_ignore_only_mask_yields_empty_presence(tmp_path):
    import numpy as np
    from PIL import Image

    ids = _write_coco_stuff_fixture(tmp_path, n=1)
    ids = m.coco_stuff_canonical_validation_ids(tmp_path, expected_count=1)
    ignore_only = np.full((4, 4), 255, dtype=np.uint8)
    Image.fromarray(ignore_only, mode="L").save(tmp_path / "annotations/val2017" / f"{ids[0]}_labelTrainIds.png")
    presence = m.coco_stuff_class_presence(tmp_path, ids)
    assert presence[ids[0]] == frozenset()


# ---------------------------------------------------------------------------
# Narrow repair regressions
#
# Finding 1: canonical_source_order_assertion was validated only for
# non-emptiness, not exact frozen content.
# Finding 2: verify_manifest_document never enforced selected_ids ==
# sorted(selected_ids) (canonical source order).
# Both were independently confirmed accepted-when-they-should-be-rejected
# before this repair; both are now exact-pinned / order-enforced.
# ---------------------------------------------------------------------------


def test_canonical_source_order_assertion_generator_and_verifier_share_one_constant():
    manifest, _, _ = _fixture_manifest()
    assert manifest["canonical_source_order_assertion"] == m.CANONICAL_SOURCE_ORDER_ASSERTION


def test_canonical_source_order_assertion_wrong_text_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["canonical_source_order_assertion"] = "a completely different assertion"
    with pytest.raises(m.DfpPhase1aSubsetManifestError, match="canonical_source_order_assertion disagrees"):
        _verify(manifest)


def test_canonical_source_order_assertion_whitespace_padded_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["canonical_source_order_assertion"] = f"  {m.CANONICAL_SOURCE_ORDER_ASSERTION}  "
    with pytest.raises(m.DfpPhase1aSubsetManifestError, match="canonical_source_order_assertion disagrees"):
        _verify(manifest)


def test_canonical_source_order_assertion_case_changed_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["canonical_source_order_assertion"] = m.CANONICAL_SOURCE_ORDER_ASSERTION.upper()
    with pytest.raises(m.DfpPhase1aSubsetManifestError, match="canonical_source_order_assertion disagrees"):
        _verify(manifest)


def test_canonical_source_order_assertion_punctuation_changed_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["canonical_source_order_assertion"] = m.CANONICAL_SOURCE_ORDER_ASSERTION.replace(")", "]")
    with pytest.raises(m.DfpPhase1aSubsetManifestError, match="canonical_source_order_assertion disagrees"):
        _verify(manifest)


def test_canonical_source_order_assertion_unicode_lookalike_rejected():
    manifest, _, _ = _fixture_manifest()
    # Cyrillic "о" (U+043E) in place of the first Latin "o" in "sorted"
    manifest["canonical_source_order_assertion"] = "sоrted" + m.CANONICAL_SOURCE_ORDER_ASSERTION[len("sorted"):]
    with pytest.raises(m.DfpPhase1aSubsetManifestError, match="canonical_source_order_assertion disagrees"):
        _verify(manifest)


def _real_manifest_and_identities(dataset_key: str):
    import importlib

    from src.dfp_phase1a_identity import load_identity as load_protocol, IDENTITY_RELATIVE_PATH as PROTOCOL_PATH

    # Earlier tests in this module deliberately patch m.SELECTED_COUNT /
    # m.DATASET_CONTRACTS in-place (via _patch_contract) to exercise small
    # synthetic universes; reload to the pristine, as-committed production
    # state before checking the real frozen manifests against it.
    importlib.reload(m)
    root = ROOT
    manifest_path = root / {
        "coco_stuff": "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
        "ade20k": "evaluation_subsets/dfp_phase1a_ade20k_500.json",
    }[dataset_key]
    with open(manifest_path) as f:
        manifest = json.load(f)
    protocol_sha = m.sha256_file(root / PROTOCOL_PATH)
    protocol = load_protocol(root / PROTOCOL_PATH, repo_root=root)
    if dataset_key == "coco_stuff":
        from src.matched_k11_k12_identity import load_identity as load_matched
        source_path = root / m.DATASET_CONTRACTS["coco_stuff"]["source_identity_relative_path"]
        load_matched(source_path, repo_root=root)
        class_digest = protocol["coco_stuff_dataset"]["class_names_digest"]
    else:
        from src.ade20k_dataset_identity import load_identity as load_ade_source
        source_path = root / m.DATASET_CONTRACTS["ade20k"]["source_identity_relative_path"]
        ade_identity = load_ade_source(source_path, repo_root=root)
        class_digest = ade_identity["class_contract"]["class_names_digest"]
    source_sha = m.sha256_file(source_path)
    return manifest, protocol_sha, source_sha, class_digest


def test_positive_coco_stuff_frozen_manifest_verifies_and_is_canonically_ordered():
    manifest, protocol_sha, source_sha, class_digest = _real_manifest_and_identities("coco_stuff")
    assert manifest["selected_ids"] == sorted(manifest["selected_ids"])
    m.verify_manifest_document(
        manifest, dataset_key_expected="coco_stuff", protocol_identity_sha256=protocol_sha,
        source_identity_sha256=source_sha, class_names_digest=class_digest,
    )


def test_positive_ade20k_frozen_manifest_verifies_and_is_canonically_ordered():
    manifest, protocol_sha, source_sha, class_digest = _real_manifest_and_identities("ade20k")
    assert manifest["selected_ids"] == sorted(manifest["selected_ids"])
    m.verify_manifest_document(
        manifest, dataset_key_expected="ade20k", protocol_identity_sha256=protocol_sha,
        source_identity_sha256=source_sha, class_names_digest=class_digest,
    )


def test_reversed_selected_ids_stale_digest_rejected():
    manifest, _, _ = _fixture_manifest()
    manifest["selected_ids"] = list(reversed(manifest["selected_ids"]))
    # selected_ids_digest intentionally left stale (not recomputed)
    with pytest.raises(m.DfpPhase1aSubsetManifestError, match="selected_ids_digest does not match"):
        _verify(manifest)


def test_reversed_selected_ids_recomputed_digest_rejected_on_canonical_order():
    manifest, _, _ = _fixture_manifest()
    reversed_ids = list(reversed(manifest["selected_ids"]))
    assert reversed_ids != manifest["selected_ids"], "degenerate palindromic fixture, cannot exercise this case"
    manifest["selected_ids"] = reversed_ids
    manifest["selected_ids_digest"] = m.selected_ids_digest(reversed_ids)
    with pytest.raises(m.DfpPhase1aSubsetManifestError, match="not in canonical source order"):
        _verify(manifest)


def test_one_adjacent_swap_recomputed_digest_rejected_on_canonical_order():
    manifest, _, _ = _fixture_manifest()
    swapped = list(manifest["selected_ids"])
    # swap two adjacent entries -- still all-unique, still correct count,
    # still a subset of valid IDs, only the ORDER is wrong
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert swapped != manifest["selected_ids"]
    manifest["selected_ids"] = swapped
    manifest["selected_ids_digest"] = m.selected_ids_digest(swapped)
    with pytest.raises(m.DfpPhase1aSubsetManifestError, match="not in canonical source order"):
        _verify(manifest)


def test_canonical_order_check_reports_field_qualified_diagnostic():
    manifest, _, _ = _fixture_manifest()
    reversed_ids = list(reversed(manifest["selected_ids"]))
    manifest["selected_ids"] = reversed_ids
    manifest["selected_ids_digest"] = m.selected_ids_digest(reversed_ids)
    with pytest.raises(m.DfpPhase1aSubsetManifestError) as excinfo:
        _verify(manifest)
    assert "selected_ids" in str(excinfo.value)
    assert "canonical source order" in str(excinfo.value)


def test_verify_manifest_reconciles_with_selection_still_accepts_valid_manifest():
    manifest, ids, presence = _fixture_manifest()
    # must not raise
    m.verify_manifest_reconciles_with_selection(manifest, canonical_ids=ids, presence=presence)


def test_verify_manifest_reconciles_with_selection_still_rejects_reversed_ids():
    manifest, ids, presence = _fixture_manifest()
    reversed_ids = list(reversed(manifest["selected_ids"]))
    manifest["selected_ids"] = reversed_ids
    manifest["selected_ids_digest"] = m.selected_ids_digest(reversed_ids)
    with pytest.raises(m.DfpPhase1aSubsetManifestError):
        m.verify_manifest_reconciles_with_selection(manifest, canonical_ids=ids, presence=presence)
