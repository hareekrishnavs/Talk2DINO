import copy
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import build_e7_training_bank as builder_module
from build_e7_training_bank import (
    _publish_atomic,
    _require_unchanged_git_provenance,
    build_e7_training_bank,
    source_git_provenance,
)
from src.e6_prototype_bank import annotation_id_set_fingerprint, sha256_file
from src.e7_training_bank import (
    FORMAT_VERSION,
    E7BankIdentity,
    E7TrainingBankValidationError,
    load_e7_training_bank,
    route_e7_annotation_batch,
    validate_e7_training_bank,
)
from train_e7_prototype_adapter import _publish_verified_checkpoint


class SyntheticProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.arange(768 * 512, dtype=torch.float32).reshape(768, 512)
        self.register_buffer("weight", torch.sin(weight * 1e-4) * 0.02)
        self.register_buffer("bias", torch.linspace(-0.2, 0.2, 768))

    def project_clip_txt(self, value):
        return torch.tanh(F.linear(value, self.weight, self.bias))


def normalized(rows, dimensions, seed):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(rows, dimensions, generator=generator), dim=-1)


def _git(repository, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def initialized_repository(path):
    path.mkdir()
    _git(path, "init", "-q")
    tracked = path / "implementation.py"
    tracked.write_text("VERSION = 1\n")
    _git(path, "add", "implementation.py")
    _git(
        path,
        "-c",
        "user.name=E7 Test",
        "-c",
        "user.email=e7@example.invalid",
        "commit",
        "-q",
        "-m",
        "initial",
    )
    return tracked


def stub_builder_inputs(tmp_path, monkeypatch):
    source = tmp_path / "source.pth"
    config = tmp_path / "vitb_mlp_infonce_paired_soft_routing_tau010.yaml"
    checkpoint = tmp_path / "vitb_mlp_infonce_paired_soft_routing_tau010.pth"
    for path, contents in (
        (source, b"source"),
        (config, b"config"),
        (checkpoint, b"checkpoint"),
    ):
        path.write_bytes(contents)
    images = [{"id": 7, "disentangled_self_attn": torch.randn(12, 768)}]
    annotations = [{"id": 11, "image_id": 7, "ann_feats": torch.randn(512)}]
    monkeypatch.setattr(
        builder_module,
        "_load_source_archive",
        lambda unused: (images, annotations),
    )
    monkeypatch.setattr(
        builder_module,
        "_load_e3_projection",
        lambda *unused: (object(), 0.10),
    )

    def route(unused_projection, annotation_features, unused_heads, unused_tau):
        rows = len(annotation_features)
        return (
            normalized(rows, 512, 21),
            normalized(rows, 768, 22),
            normalized(rows, 768, 23),
        )

    monkeypatch.setattr(builder_module, "route_e7_annotation_batch", route)
    return source, config, checkpoint


def payload(entries=5, *, split="train", pilot=False, complete=True):
    annotation_ids = torch.arange(100, 100 + entries, dtype=torch.int64)
    source_annotations = entries + 2 if pilot else entries
    return {
        "caption_embeddings": normalized(entries, 512, 1).half(),
        "mapped_query_embeddings": normalized(entries, 768, 2).half(),
        "routed_target_embeddings": normalized(entries, 768, 3).half(),
        "image_ids": torch.arange(entries, dtype=torch.int64),
        "annotation_ids": annotation_ids,
        "metadata": {
            "format_version": FORMAT_VERSION,
            "complete": complete,
            "is_pilot": pilot,
            "split_name": split,
            "source_feature_path": f"/synthetic/{split}.pth",
            "source_feature_sha256": "a" * 64,
            "source_image_count": entries,
            "source_annotation_count": source_annotations,
            "selected_annotation_count": entries,
            "annotation_id_fingerprint": annotation_id_set_fingerprint(annotation_ids),
            "e3_config_name": "vitb_mlp_infonce_paired_soft_routing_tau010.yaml",
            "e3_config_sha256": "b" * 64,
            "e3_checkpoint_name": "vitb_mlp_infonce_paired_soft_routing_tau010.pth",
            "e3_checkpoint_sha256": "c" * 64,
            "routing_temperature": 0.10,
            "source_git_commit": "d" * 40,
            "source_git_dirty": False,
            "source_git_diff_sha256": None,
            "dimensions": {
                "caption_embeddings": 512,
                "mapped_query_embeddings": 768,
                "routed_target_embeddings": 768,
            },
            "dtypes": {
                "caption_embeddings": "float16",
                "mapped_query_embeddings": "float16",
                "routed_target_embeddings": "float16",
                "image_ids": "int64",
                "annotation_ids": "int64",
            },
        },
    }


def test_e7_bank_routing_matches_independent_loop_reference():
    torch.manual_seed(7)
    projection = SyntheticProjection()
    annotations = torch.randn(3, 512)
    heads = torch.randn(3, 12, 768)
    raw, mapped, routed = route_e7_annotation_batch(
        projection, annotations, heads, 0.10
    )
    expected_raw = []
    expected_mapped = []
    expected_routed = []
    for index in range(3):
        raw_i = annotations[index] / annotations[index].norm()
        mapped_i = projection.project_clip_txt(annotations[index : index + 1])[0]
        mapped_i = mapped_i / mapped_i.norm()
        heads_i = F.normalize(heads[index].float(), dim=-1)
        weights = torch.softmax((heads_i @ mapped_i) / 0.10, dim=0)
        routed_i = (weights[:, None] * heads_i).sum(dim=0)
        routed_i = routed_i / routed_i.norm()
        expected_raw.append(raw_i)
        expected_mapped.append(mapped_i)
        expected_routed.append(routed_i)
    torch.testing.assert_close(raw, torch.stack(expected_raw))
    torch.testing.assert_close(mapped, torch.stack(expected_mapped))
    torch.testing.assert_close(routed, torch.stack(expected_routed))


def test_mapped_query_uses_original_not_normalized_annotation_features():
    projection = SyntheticProjection()
    annotations = torch.randn(2, 512) * 9
    heads = torch.randn(2, 12, 768)
    _, actual, _ = route_e7_annotation_batch(projection, annotations, heads)
    expected = F.normalize(projection.project_clip_txt(annotations), dim=-1)
    wrong = F.normalize(
        projection.project_clip_txt(F.normalize(annotations, dim=-1)), dim=-1
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, wrong, atol=1e-4, rtol=1e-4)


def test_closed_schema_and_identity_are_validated():
    bank = payload()
    summary = validate_e7_training_bank(bank, expected_split="train")
    assert summary["entries"] == 5
    identity = E7BankIdentity.from_metadata(bank["metadata"])
    assert identity.split_name == "train"
    unknown = copy.deepcopy(bank)
    unknown["metadata"]["ambiguous"] = True
    with pytest.raises(E7TrainingBankValidationError, match="closed metadata"):
        validate_e7_training_bank(unknown)


def test_pilot_completeness_and_dirty_source_guards():
    pilot = payload(pilot=True)
    with pytest.raises(E7TrainingBankValidationError, match="pilot"):
        validate_e7_training_bank(pilot)
    validate_e7_training_bank(pilot, allow_pilot=True)
    incomplete = payload(complete=False)
    with pytest.raises(E7TrainingBankValidationError, match="incomplete"):
        validate_e7_training_bank(incomplete)
    validate_e7_training_bank(incomplete, require_complete=False)
    dirty = payload(pilot=True)
    dirty["metadata"]["source_git_dirty"] = True
    dirty["metadata"]["source_git_diff_sha256"] = "e" * 64
    with pytest.raises(E7TrainingBankValidationError, match="dirty"):
        validate_e7_training_bank(dirty, allow_pilot=True)
    validate_e7_training_bank(
        dirty, allow_pilot=True, allow_dirty_source=True
    )


def test_tensor_contract_rejects_wrong_shape_dtype_and_nonfinite():
    wrong_shape = payload()
    wrong_shape["mapped_query_embeddings"] = torch.randn(5, 767).half()
    with pytest.raises(E7TrainingBankValidationError, match="mapped_query"):
        validate_e7_training_bank(wrong_shape)
    wrong_dtype = payload()
    wrong_dtype["caption_embeddings"] = wrong_dtype["caption_embeddings"].float()
    with pytest.raises(E7TrainingBankValidationError, match="float16"):
        validate_e7_training_bank(wrong_dtype)
    nonfinite = payload()
    nonfinite["routed_target_embeddings"][0, 0] = torch.inf
    with pytest.raises(E7TrainingBankValidationError, match="non-finite"):
        validate_e7_training_bank(nonfinite)


def test_sha_validation_rejects_same_named_modified_inputs(tmp_path):
    bank = payload()
    config = tmp_path / bank["metadata"]["e3_config_name"]
    checkpoint = tmp_path / bank["metadata"]["e3_checkpoint_name"]
    source = tmp_path / "train.pth"
    config.write_bytes(b"exact config")
    checkpoint.write_bytes(b"exact checkpoint")
    source.write_bytes(b"exact source")
    bank["metadata"]["e3_config_sha256"] = sha256_file(config)
    bank["metadata"]["e3_checkpoint_sha256"] = sha256_file(checkpoint)
    bank["metadata"]["source_feature_sha256"] = sha256_file(source)
    validate_e7_training_bank(
        bank,
        expected_config_path=config,
        expected_checkpoint_path=checkpoint,
        expected_source_features_path=source,
    )
    config.write_bytes(b"modified config")
    with pytest.raises(E7TrainingBankValidationError, match="configuration SHA"):
        validate_e7_training_bank(bank, expected_config_path=config)


def test_atomic_publication_refuses_overwrite_and_round_trips(tmp_path):
    bank = payload()
    path = tmp_path / "bank.pth"
    _publish_atomic(bank, path, overwrite=False)
    loaded = load_e7_training_bank(path)
    assert torch.equal(loaded["annotation_ids"], bank["annotation_ids"])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _publish_atomic(bank, path, overwrite=False)
    assert path.read_bytes() == original
    replacement = payload(entries=3)
    _publish_atomic(replacement, path, overwrite=True)
    assert load_e7_training_bank(path)["metadata"]["selected_annotation_count"] == 3


def test_unchanged_git_provenance_succeeds(tmp_path):
    repository = tmp_path / "repository"
    initialized_repository(repository)
    initial = source_git_provenance(repository)
    assert _require_unchanged_git_provenance(
        repository,
        initial,
        allow_dirty_source=False,
    ) == initial


@pytest.mark.parametrize("mutation", ("tracked", "staged", "untracked"))
def test_git_source_mutation_is_rejected(tmp_path, mutation):
    repository = tmp_path / mutation
    tracked = initialized_repository(repository)
    initial = source_git_provenance(repository)
    if mutation == "untracked":
        (repository / "new_e7_source.py").write_text("NEW = True\n")
    else:
        tracked.write_text("VERSION = 2\n")
        if mutation == "staged":
            _git(repository, "add", "implementation.py")
    with pytest.raises(
        E7TrainingBankValidationError,
        match="Git source provenance changed",
    ):
        _require_unchanged_git_provenance(
            repository,
            initial,
            allow_dirty_source=False,
        )


def test_unchanged_provenance_build_publishes_bank(tmp_path, monkeypatch):
    source, config, checkpoint = stub_builder_inputs(tmp_path, monkeypatch)
    provenance = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    monkeypatch.setattr(
        builder_module,
        "source_git_provenance",
        lambda *args, **kwargs: dict(provenance),
    )
    output = tmp_path / "bank.pth"
    summary = build_e7_training_bank(
        source_features_path=source,
        output_path=output,
        split_name="train",
        model_config_path=config,
        checkpoint_path=checkpoint,
        repository_root=tmp_path,
    )
    assert summary["complete"] is True
    assert output.is_file()


def test_mutated_provenance_publishes_no_new_bank(tmp_path, monkeypatch):
    source, config, checkpoint = stub_builder_inputs(tmp_path, monkeypatch)
    initial = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    changed = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": True,
        "source_git_diff_sha256": "b" * 64,
    }
    calls = iter((initial, changed))
    monkeypatch.setattr(
        builder_module,
        "source_git_provenance",
        lambda *args, **kwargs: dict(next(calls)),
    )
    output = tmp_path / "bank.pth"
    with pytest.raises(
        E7TrainingBankValidationError,
        match="Git source provenance changed",
    ):
        build_e7_training_bank(
            source_features_path=source,
            output_path=output,
            split_name="train",
            model_config_path=config,
            checkpoint_path=checkpoint,
            repository_root=tmp_path,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".bank.pth.*.tmp"))


def test_mutated_provenance_preserves_existing_output(
    tmp_path,
    monkeypatch,
):
    source, config, checkpoint = stub_builder_inputs(tmp_path, monkeypatch)
    initial = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": False,
        "source_git_diff_sha256": None,
    }
    changed = {
        "source_git_commit": "a" * 40,
        "source_git_dirty": True,
        "source_git_diff_sha256": "b" * 64,
    }
    calls = iter((initial, changed))
    monkeypatch.setattr(
        builder_module,
        "source_git_provenance",
        lambda *args, **kwargs: dict(next(calls)),
    )
    output = tmp_path / "bank.pth"
    output.write_bytes(b"existing output")
    with pytest.raises(
        E7TrainingBankValidationError,
        match="Git source provenance changed",
    ):
        build_e7_training_bank(
            source_features_path=source,
            output_path=output,
            split_name="train",
            model_config_path=config,
            checkpoint_path=checkpoint,
            repository_root=tmp_path,
            overwrite=True,
        )
    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".bank.pth.*.tmp"))


def test_unchanged_adapter_training_provenance_allows_publication(tmp_path):
    repository = tmp_path / "adapter-clean"
    initialized_repository(repository)
    initial = source_git_provenance(repository)
    output = tmp_path / "adapter.pth"
    _publish_verified_checkpoint(
        {"adapter": torch.tensor([1.0])},
        output,
        repository_root=repository,
        initial_provenance=initial,
        allow_dirty_source=False,
    )
    assert output.is_file()


@pytest.mark.parametrize("mutation", ("tracked", "staged", "untracked"))
@pytest.mark.parametrize("existing_checkpoint", (False, True))
def test_adapter_training_source_mutation_rejects_publication_and_preserves_output(
    tmp_path,
    mutation,
    existing_checkpoint,
):
    repository = tmp_path / f"adapter-{mutation}-{existing_checkpoint}"
    tracked = initialized_repository(repository)
    initial = source_git_provenance(repository)
    output = tmp_path / f"adapter-{mutation}-{existing_checkpoint}.pth"
    original_sha256 = None
    original_bytes = None
    if existing_checkpoint:
        output.write_bytes(b"previous adapter checkpoint")
        original_bytes = output.read_bytes()
        original_sha256 = sha256_file(output)
    if mutation == "untracked":
        (repository / "new_adapter_source.py").write_text("NEW = True\n")
    else:
        tracked.write_text("VERSION = 2\n")
        if mutation == "staged":
            _git(repository, "add", "implementation.py")
    with pytest.raises(
        E7TrainingBankValidationError,
        match="Git source provenance changed during E7 adapter training",
    ):
        _publish_verified_checkpoint(
            {"adapter": torch.tensor([2.0])},
            output,
            repository_root=repository,
            initial_provenance=initial,
            allow_dirty_source=False,
        )
    if existing_checkpoint:
        assert output.read_bytes() == original_bytes
        assert sha256_file(output) == original_sha256
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_unchanged_dirty_pilot_fingerprint_is_required_for_publication(tmp_path):
    repository = tmp_path / "adapter-dirty-pilot"
    initialized_repository(repository)
    dirty_source = repository / "pilot_source.py"
    dirty_source.write_text("VERSION = 1\n")
    initial = source_git_provenance(repository, allow_dirty_source=True)
    output = tmp_path / "dirty-pilot-adapter.pth"
    _publish_verified_checkpoint(
        {"adapter": torch.tensor([1.0])},
        output,
        repository_root=repository,
        initial_provenance=initial,
        allow_dirty_source=True,
    )
    original = output.read_bytes()
    dirty_source.write_text("VERSION = 2\n")
    with pytest.raises(
        E7TrainingBankValidationError,
        match="Git source provenance changed during E7 adapter training",
    ):
        _publish_verified_checkpoint(
            {"adapter": torch.tensor([2.0])},
            output,
            repository_root=repository,
            initial_provenance=initial,
            allow_dirty_source=True,
        )
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
