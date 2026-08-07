import hashlib
import subprocess
from pathlib import Path

import pytest
import torch

import train_e9_sparse_region_alignment as training
from src.e9_sparse_region_alignment import (
    CANONICAL_E9_TRAINING_CONFIG,
    E9_ADAPTER_CHECKPOINT_FORMAT,
    E9ValidationError,
    SparseRegionAlignmentAdapter,
    SparseRegionAlignmentConfig,
    load_e9_adapter,
    validate_e9_checkpoint,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    tracked = path / "source.py"
    tracked.write_text("clean\n")
    subprocess.run(["git", "-C", str(path), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "base"], check=True)
    return tracked


def dataset_identity(split):
    discriminator = "1" if split == "train" else "2"
    return {
        "split": split,
        "source_image_count": 2,
        "source_annotation_count": 2,
        "selected_image_count": 2,
        "selected_annotation_count": 2,
        "image_id_fingerprint": discriminator * 64,
        "annotation_id_fingerprint": ("3" if split == "train" else "4") * 64,
        "annotation_to_image_fingerprint": ("5" if split == "train" else "6") * 64,
    }


def identity(split):
    dataset = dataset_identity(split)
    return {
        "format_version": "talk2dino-e7-training-bank-v1",
        "split_name": split,
        "bank_sha256": ("b" if split == "train" else "f") * 64,
        "source_feature_sha256": "a" * 64,
        **{key: dataset[key] for key in (
            "source_image_count", "source_annotation_count",
            "selected_image_count", "selected_annotation_count",
            "image_id_fingerprint", "annotation_id_fingerprint",
            "annotation_to_image_fingerprint",
        )},
        "e3_config_sha256": "c" * 64,
        "e3_checkpoint_sha256": "d" * 64,
        "routing_temperature": 0.1,
        "source_git_commit": "e" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
        "complete": True,
        "is_pilot": False,
    }


def spatial_identity(split):
    return {
        "format_version": "talk2dino-e9-spatial-bank-v1",
        "split": split,
        "manifest_sha256": ("7" if split == "train" else "8") * 64,
        "source_feature_sha256": [("9" if split == "train" else "a") * 64],
        "dataset_identity": dataset_identity(split),
        "complete": True,
        "is_pilot": False,
        "production_eligible": True,
        "source_git_commit": "e" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }


def payload(*, embedding_dim=8, production=False):
    architecture = (
        SparseRegionAlignmentConfig()
        if production
        else SparseRegionAlignmentConfig(
            embedding_dim=embedding_dim,
            bottleneck_dim=4,
            dropout=0,
            residual_max=0.25,
            gamma_max=0.3,
            gate_hidden_dim=4,
        )
    )
    model = SparseRegionAlignmentAdapter(architecture)
    config = dict(CANONICAL_E9_TRAINING_CONFIG)
    if not production:
        config.update({
            "embedding_dim": embedding_dim,
            "bottleneck_dim": 4,
            "dropout": 0,
            "gate_hidden_dim": 4,
        })
    run = {
        "pilot_training": not production,
        "bounded_training": not production,
        "bounded_validation": not production,
        "train_query_complete": True,
        "train_query_is_pilot": False,
        "validation_query_complete": True,
        "validation_query_is_pilot": False,
        "train_spatial_complete": True,
        "train_spatial_is_pilot": False,
        "validation_spatial_complete": True,
        "validation_spatial_is_pilot": False,
        "source_git_dirty": False,
        "train_validation_overlap": False,
        "production_eligible": production,
    }
    return {
        "format_version": E9_ADAPTER_CHECKPOINT_FORMAT,
        "adapter_state_dict": model.state_dict(),
        "architecture_config": dict(architecture.__dict__),
        "training_config": config,
        "epoch": 1,
        "best_validation_metric": 1.0,
        "query_bank_identities": {
            "train": identity("train"), "validation": identity("val")
        },
        "spatial_bank_identities": {
            "train": spatial_identity("train"),
            "validation": spatial_identity("val"),
        },
        "e3_identity": {
            "config_sha256": "c" * 64,
            "checkpoint_sha256": "d" * 64,
        },
        "source_feature_identities": {
            "train": {
                "query_bank_sha256": "b" * 64,
                "query_source_feature_sha256": "a" * 64,
                "spatial_manifest_sha256": "7" * 64,
                "spatial_source_artifact_sha256": ["9" * 64],
            },
            "validation": {
                "query_bank_sha256": "f" * 64,
                "query_source_feature_sha256": "a" * 64,
                "spatial_manifest_sha256": "8" * 64,
                "spatial_source_artifact_sha256": ["a" * 64],
            },
        },
        "run_identity": run,
        "source_git_provenance": {
            "source_git_commit": "4" * 40,
            "source_git_dirty": False,
            "source_git_diff_sha256": None,
        },
        "diagnostic_summary": {"validation": {"loss": 1.0}},
    }


def test_checkpoint_round_trip_and_contains_only_adapter_state(tmp_path):
    value = payload()
    path = tmp_path / "adapter.pth"
    torch.save(value, path)
    loaded = load_e9_adapter(path, require_production=False)
    assert isinstance(loaded, SparseRegionAlignmentAdapter)
    assert set(value["adapter_state_dict"]) == set(loaded.state_dict())
    serialized_keys = " ".join(value["adapter_state_dict"]).lower()
    for forbidden in ("clip", "dino", "projection", "bank"):
        assert forbidden not in serialized_keys


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda value: value.update({"unknown": 1}), "closed schema"),
        (lambda value: value["architecture_config"].update({"unknown": 1}), "closed schema"),
        (lambda value: value["training_config"].update({"unknown": 1}), "closed schema"),
        (lambda value: value["run_identity"].update({"unknown": False}), "closed schema"),
    ),
)
def test_closed_checkpoint_schemas_reject_unknown_keys(mutation, match):
    value = payload()
    mutation(value)
    with pytest.raises(E9ValidationError, match=match):
        validate_e9_checkpoint(value, require_production=False)


def test_adapter_state_shape_contract_rejects_before_construction(
    tmp_path, monkeypatch
):
    value = payload(production=True)
    key = next(iter(value["adapter_state_dict"]))
    value["adapter_state_dict"][key] = torch.ones(1)
    path = tmp_path / "adapter.pth"
    torch.save(value, path)
    constructions = 0

    class InstrumentedAdapter(SparseRegionAlignmentAdapter):
        def __init__(self, *args, **kwargs):
            nonlocal constructions
            constructions += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "src.e9_sparse_region_alignment.SparseRegionAlignmentAdapter",
        InstrumentedAdapter,
    )
    with pytest.raises(E9ValidationError, match="key/shape contract"):
        load_e9_adapter(
            path, expected_checkpoint_sha256=digest(path),
            expected_source_git_commit="4" * 40,
        )
    assert constructions == 0


@pytest.mark.parametrize(
    "field",
    (
        "pilot_training", "bounded_training", "bounded_validation",
        "train_query_is_pilot", "validation_query_is_pilot",
        "train_spatial_is_pilot", "validation_spatial_is_pilot",
        "source_git_dirty", "train_validation_overlap",
    ),
)
def test_pilot_dirty_bounded_and_overlap_are_production_rejected(field):
    value = payload(production=True)
    value["run_identity"][field] = True
    if field == "source_git_dirty":
        value["source_git_provenance"]["source_git_dirty"] = True
        value["source_git_provenance"]["source_git_diff_sha256"] = "5" * 64
    value["run_identity"]["production_eligible"] = False
    with pytest.raises(
        E9ValidationError,
        match="not production eligible|does not match loaded artifact identities",
    ):
        validate_e9_checkpoint(value)


@pytest.mark.parametrize(
    "field",
    (
        "train_query_complete", "validation_query_complete",
        "train_spatial_complete", "validation_spatial_complete",
    ),
)
def test_incomplete_inputs_are_production_rejected(field):
    value = payload(production=True)
    split = "train" if field.startswith("train") else "validation"
    if "query" in field:
        value["query_bank_identities"][split]["complete"] = False
    else:
        value["spatial_bank_identities"][split]["complete"] = False
        value["spatial_bank_identities"][split]["production_eligible"] = False
    value["run_identity"][field] = False
    value["run_identity"]["production_eligible"] = False
    with pytest.raises(E9ValidationError, match="not production eligible"):
        validate_e9_checkpoint(value)


def test_forged_eligibility_and_non768_production_are_rejected():
    value = payload()
    value["run_identity"]["production_eligible"] = True
    with pytest.raises(E9ValidationError, match="forged"):
        validate_e9_checkpoint(value, require_production=False)


def test_e3_identity_fails_before_adapter_construction(monkeypatch):
    value = payload()
    with pytest.raises(E9ValidationError, match="E3 identity"):
        validate_e9_checkpoint(
            value,
            require_production=False,
            expected_e3_identity={
                "config_sha256": "0" * 64,
                "checkpoint_sha256": "d" * 64,
            },
        )


def test_incompatible_training_identities_are_rejected():
    value = payload()
    value["query_bank_identities"]["validation"]["e3_checkpoint_sha256"] = "0" * 64
    with pytest.raises(E9ValidationError, match="incompatible train/validation"):
        validate_e9_checkpoint(value, require_production=False)


def test_clean_canonical_production_checkpoint_is_accepted():
    value = payload(production=True)
    result = validate_e9_checkpoint(
        value,
        expected_source_git_commit=value["source_git_provenance"][
            "source_git_commit"
        ],
    )
    assert result["production_eligible"] and result["canonical_experiment"]


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda value: value["training_config"].update(
                {"pooled_grid_width": 8}
            ),
            "forged E9 production eligibility",
        ),
        (
            lambda value: value["training_config"].update({"mil_top_k": 15}),
            "forged E9 production eligibility",
        ),
        (
            lambda value: value["training_config"].update(
                {"mil_temperature": 0.2}
            ),
            "forged E9 production eligibility",
        ),
        (
            lambda value: value["training_config"].update(
                {"infonce_temperature": 0.08}
            ),
            "forged E9 production eligibility",
        ),
        (
            lambda value: value["query_bank_identities"]["train"].update(
                {"image_id_fingerprint": "0" * 64}
            ),
            "dataset identity mismatch",
        ),
        (
            lambda value: value["spatial_bank_identities"]["train"].update(
                {"manifest_sha256": "0" * 64}
            ),
            "spatial manifest identity mismatch",
        ),
        (
            lambda value: value["source_feature_identities"]["train"].update(
                {"query_source_feature_sha256": "0" * 64}
            ),
            "source-feature identity mismatch",
        ),
        (
            lambda value: value["e3_identity"].update(
                {"config_sha256": "0" * 64}
            ),
            "incompatible train/validation E3 identities",
        ),
        (
            lambda value: value["source_git_provenance"].update(
                {"source_git_commit": "0" * 40}
            ),
            "source Git identity mismatch",
        ),
    ),
)
def test_production_identity_mutations_fail_before_adapter_construction(
    tmp_path, monkeypatch, mutation, match
):
    value = payload(production=True)
    mutation(value)
    path = tmp_path / "adapter.pth"
    torch.save(value, path)
    counts = {"construct": 0, "load_state": 0}

    class InstrumentedAdapter(SparseRegionAlignmentAdapter):
        def __init__(self, *args, **kwargs):
            counts["construct"] += 1
            super().__init__(*args, **kwargs)

        def load_state_dict(self, *args, **kwargs):
            counts["load_state"] += 1
            return super().load_state_dict(*args, **kwargs)

    monkeypatch.setattr(
        "src.e9_sparse_region_alignment.SparseRegionAlignmentAdapter",
        InstrumentedAdapter,
    )
    with pytest.raises(E9ValidationError, match=match):
        load_e9_adapter(
            path,
            expected_checkpoint_sha256=digest(path),
            expected_source_git_commit="4" * 40,
        )
    assert counts == {"construct": 0, "load_state": 0}


def test_stale_external_checkpoint_digest_fails_before_load_or_construction(
    tmp_path, monkeypatch
):
    path = tmp_path / "adapter.pth"
    torch.save(payload(production=True), path)
    stale = digest(path)
    path.write_bytes(path.read_bytes() + b"mutation")
    counts = {"load": 0, "construct": 0}
    original_load = torch.load

    def counted_load(*args, **kwargs):
        counts["load"] += 1
        return original_load(*args, **kwargs)

    class InstrumentedAdapter(SparseRegionAlignmentAdapter):
        def __init__(self, *args, **kwargs):
            counts["construct"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("src.e9_sparse_region_alignment.torch.load", counted_load)
    monkeypatch.setattr(
        "src.e9_sparse_region_alignment.SparseRegionAlignmentAdapter",
        InstrumentedAdapter,
    )
    with pytest.raises(E9ValidationError, match="SHA256 mismatch"):
        load_e9_adapter(
            path,
            expected_checkpoint_sha256=stale,
            expected_source_git_commit="4" * 40,
        )
    assert counts == {"load": 0, "construct": 0}


def test_atomic_refusal_preserves_existing_and_cleans_temporary(tmp_path, monkeypatch):
    output = tmp_path / "adapter.pth"
    output.write_bytes(b"existing")
    before = digest(output)
    source = tmp_path / "input"
    source.write_bytes(b"input")
    monkeypatch.setattr(
        training,
        "_require_unchanged_git_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Git changed")),
    )
    with pytest.raises(RuntimeError, match="Git changed"):
        training.atomic_publish_checkpoint(
            payload(), output, overwrite=True,
            provenance={
                "source_git_commit": "a" * 40,
                "source_git_dirty": False,
                "source_git_diff_sha256": None,
            },
            repository_root=tmp_path,
            input_hashes={str(source): digest(source)},
            allow_dirty_source=False,
        )
    assert digest(output) == before
    assert not list(tmp_path.glob(".adapter.pth.*.tmp"))


def test_input_mutation_during_serialization_prevents_publication(tmp_path, monkeypatch):
    output = tmp_path / "adapter.pth"
    source = tmp_path / "input"
    source.write_bytes(b"before")
    original = torch.save

    def mutating_save(value, path):
        original(value, path)
        source.write_bytes(b"after")

    monkeypatch.setattr(training.torch, "save", mutating_save)
    monkeypatch.setattr(training, "_require_unchanged_git_provenance", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="input changed"):
        training.atomic_publish_checkpoint(
            payload(), output, overwrite=False,
            provenance={
                "source_git_commit": "a" * 40,
                "source_git_dirty": False,
                "source_git_diff_sha256": None,
            },
            repository_root=tmp_path,
            input_hashes={str(source): hashlib.sha256(b"before").hexdigest()},
            allow_dirty_source=False,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".adapter.pth.*.tmp"))


def test_unchanged_inputs_publish_and_explicit_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "adapter.pth"
    source = tmp_path / "input"
    source.write_bytes(b"input")
    provenance = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    monkeypatch.setattr(
        training, "_require_unchanged_git_provenance",
        lambda *args, **kwargs: None,
    )
    training.atomic_publish_checkpoint(
        payload(), output, overwrite=False, provenance=provenance,
        repository_root=tmp_path, input_hashes={str(source): digest(source)},
        allow_dirty_source=False,
    )
    first = digest(output)
    replacement = payload()
    replacement["epoch"] = 2
    training.atomic_publish_checkpoint(
        replacement, output, overwrite=True, provenance=provenance,
        repository_root=tmp_path, input_hashes={str(source): digest(source)},
        allow_dirty_source=False,
    )
    assert digest(output) != first
    assert torch.load(output, map_location="cpu", weights_only=False)["epoch"] == 2
    assert not list(tmp_path.glob(".adapter.pth.*.tmp"))


def test_concurrent_target_creation_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "adapter.pth"
    source = tmp_path / "input"
    source.write_bytes(b"input")
    original_save = torch.save

    def concurrent_save(value, path):
        original_save(value, path)
        output.write_bytes(b"concurrent-winner")

    monkeypatch.setattr(training.torch, "save", concurrent_save)
    monkeypatch.setattr(
        training, "_require_unchanged_git_provenance",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        training.atomic_publish_checkpoint(
            payload(), output, overwrite=False,
            provenance={
                "source_git_commit": "a" * 40,
                "source_git_dirty": False,
                "source_git_diff_sha256": None,
            },
            repository_root=tmp_path,
            input_hashes={str(source): digest(source)},
            allow_dirty_source=False,
        )
    assert output.read_bytes() == b"concurrent-winner"
    assert not list(tmp_path.glob(".adapter.pth.*.tmp"))


def test_best_and_final_paths_call_the_same_atomic_publication_helper():
    source = Path("train_e9_sparse_region_alignment.py").read_text()
    assert "parser.add_argument(\"--final_output\"" in source
    # One definition plus the best-model and final-model call sites.
    assert source.count("atomic_publish_checkpoint(") == 3


@pytest.mark.parametrize("kind", ("unstaged", "staged", "untracked"))
def test_real_git_provenance_rejects_every_dirty_source_kind(tmp_path, kind):
    root = tmp_path / "repo"
    tracked = git_repo(root)
    if kind == "untracked":
        (root / "new.py").write_text("new\n")
    else:
        tracked.write_text("changed\n")
        if kind == "staged":
            subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    with pytest.raises(ValueError, match="clean Git worktree"):
        training.source_git_provenance(root)


def test_real_git_mutation_during_serialization_prevents_publication(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    tracked = git_repo(root)
    initial = training.source_git_provenance(root)
    output = tmp_path / "adapter.pth"
    source = tmp_path / "input"
    source.write_bytes(b"input")
    original = torch.save

    def mutating_save(value, path):
        original(value, path)
        tracked.write_text("mutated during save\n")

    monkeypatch.setattr(training.torch, "save", mutating_save)
    with pytest.raises(RuntimeError, match="provenance changed"):
        training.atomic_publish_checkpoint(
            payload(), output, overwrite=False, provenance=initial,
            repository_root=root, input_hashes={str(source): digest(source)},
            allow_dirty_source=False,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".adapter.pth.*.tmp"))


@pytest.mark.parametrize("existing", (False, True))
def test_git_mutation_inside_input_hash_recheck_is_caught_by_final_check(
    tmp_path, monkeypatch, existing
):
    root = tmp_path / "repo"
    tracked = git_repo(root)
    initial = training.source_git_provenance(root)
    output = tmp_path / "adapter.pth"
    if existing:
        output.write_bytes(b"existing-checkpoint")
        before = output.read_bytes()
        before_sha = digest(output)
    source = tmp_path / "input"
    source.write_bytes(b"input")
    expected_source_sha = digest(source)
    original_sha = training.sha256_file
    mutated = False

    def mutate_during_hash(path):
        nonlocal mutated
        result = original_sha(path)
        if not mutated:
            tracked.write_text("mutated from input-hash callback\n")
            mutated = True
        return result

    monkeypatch.setattr(training, "sha256_file", mutate_during_hash)
    with pytest.raises(RuntimeError, match="provenance changed"):
        training.atomic_publish_checkpoint(
            payload(), output, overwrite=existing, provenance=initial,
            repository_root=root,
            input_hashes={str(source): expected_source_sha},
            allow_dirty_source=False,
        )
    if existing:
        assert output.read_bytes() == before
        assert digest(output) == before_sha
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".adapter.pth.*.tmp"))
