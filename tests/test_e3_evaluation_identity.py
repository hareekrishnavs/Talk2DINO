import json
import copy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import src.e3_evaluation_identity as identity_module
from src.e3_evaluation_identity import (
    E3IdentityError,
    load_identity,
    parse_log_metrics,
    parse_structured_metrics,
    validate_static_configuration,
    verify_metrics,
    verify_result,
)


ROOT = Path(__file__).parents[1]
IDENTITY_PATH = ROOT / "evaluation_identities/e3_paired_soft_routing.toml"


def _identity():
    return load_identity(IDENTITY_PATH, repo_root=ROOT)


def _canonical_metrics():
    identity = _identity()
    return {
        "evaluated_images": identity["dataset"]["images"],
        **identity["expected_metrics"],
    }


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _copy_static_repository(tmp_path):
    for relative in (
        "src/open_vocabulary_segmentation/configs/stuff",
        "src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets",
        "configs",
    ):
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    model_source = (
        ROOT
        / "src/open_vocabulary_segmentation/models/dinotext/dinotext.py"
    )
    model_destination = tmp_path / model_source.relative_to(ROOT)
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_source, model_destination)
    return tmp_path


def _mutate_eval_yaml(repo, callback):
    path = (
        repo
        / "src/open_vocabulary_segmentation/configs/stuff/"
        "dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml"
    )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    callback(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _mutate_eval_base_yaml(repo, callback):
    path = repo / "src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    callback(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _append_dataset_config(repo, statement):
    path = repo / _identity()["dataset"]["config_path"]
    path.write_text(
        path.read_text(encoding="utf-8") + f"\n{statement}\n",
        encoding="utf-8",
    )


def _validate_copy(repo, **kwargs):
    return validate_static_configuration(
        repo_root=repo,
        identity_path=IDENTITY_PATH,
        check_git=False,
        **kwargs,
    )


def test_identity_specification_loads_successfully():
    identity = _identity()
    assert identity["identity_name"]
    assert identity["dataset"]["images"] > 0
    assert identity["dataset"]["classes"] > 0
    assert type(identity["dataset"]["background_class"]) is bool
    assert set(identity["expected_metrics"]) == {"aAcc", "mIoU", "mAcc"}


def test_canonical_repository_configuration_passes_static_validation():
    result = validate_static_configuration(repo_root=ROOT)
    assert result["identity_name"] == _identity()["identity_name"]
    assert result["checkpoint_checked"] is False


def test_wrong_crop_size_fails(tmp_path):
    repo = _copy_static_repository(tmp_path)
    path = repo / _identity()["dataset"]["config_path"]
    crop = tuple(_identity()["evaluation"]["crop"])
    wrong_crop = (crop[0] - 1, crop[1])
    text = path.read_text(encoding="utf-8").replace(
        f"crop_size={crop!r}", f"crop_size={wrong_crop!r}"
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(E3IdentityError, match="sliding crop mismatch"):
        _validate_copy(repo)


def test_wrong_stride_fails(tmp_path):
    repo = _copy_static_repository(tmp_path)
    path = repo / _identity()["dataset"]["config_path"]
    stride = tuple(_identity()["evaluation"]["stride"])
    wrong_stride = (stride[0] - 1, stride[1])
    text = path.read_text(encoding="utf-8").replace(
        f"stride={stride!r}", f"stride={wrong_stride!r}"
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(E3IdentityError, match="sliding stride mismatch"):
        _validate_copy(repo)


def test_pamr_enabled_fails(tmp_path):
    repo = _copy_static_repository(tmp_path)
    path = repo / "src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["evaluate"]["pamr"] = True
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(E3IdentityError, match="PAMR setting mismatch"):
        _validate_copy(repo)


def test_omitted_pre_trained_uses_production_default_and_passes(tmp_path):
    repo = _copy_static_repository(tmp_path)
    result = _validate_copy(repo)
    expected = _identity()["model"]["flags"]["pre_trained"]
    assert result["effective_model_flags"]["pre_trained"] is expected
    assert result["projection_checkpoint_loading"] is expected


def test_explicit_canonical_pre_trained_passes(tmp_path):
    repo = _copy_static_repository(tmp_path)
    expected = _identity()["model"]["flags"]["pre_trained"]
    _mutate_eval_yaml(
        repo,
        lambda config: config["model"].update({"pre_trained": expected}),
    )
    result = _validate_copy(repo)
    assert result["projection_checkpoint_loading"] is expected


@pytest.mark.parametrize("disabled_value", [False, 0, None, "false", "off"])
def test_disabled_or_false_like_pre_trained_fails(tmp_path, disabled_value):
    repo = _copy_static_repository(tmp_path)
    _mutate_eval_yaml(
        repo,
        lambda config: config["model"].update({"pre_trained": disabled_value}),
    )
    with pytest.raises(E3IdentityError, match=r"model\.pre_trained"):
        _validate_copy(repo)


def test_checkpoint_existence_cannot_pass_when_loading_is_disabled(tmp_path):
    repo = _copy_static_repository(tmp_path)
    checkpoint = repo / _identity()["projection"]["checkpoint_path"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic checkpoint existence probe")
    _mutate_eval_yaml(
        repo,
        lambda config: config["model"].update({"pre_trained": False}),
    )
    with pytest.raises(E3IdentityError, match="checkpoint would not be loaded"):
        _validate_copy(repo, check_checkpoint=True)


def test_canonical_dataset_class_contract_resolves_without_data():
    result = validate_static_configuration(repo_root=ROOT)
    dataset = _identity()["dataset"]
    assert result["dataset_classes"] == dataset["classes"]
    assert result["dataset_background_class"] is dataset["background_class"]
    assert result["dataset_first_class"].lower() != "background"


def test_direct_classes_list_override_fails(tmp_path):
    repo = _copy_static_repository(tmp_path)
    _append_dataset_config(
        repo,
        'data["test"]["classes"] = ["background", "not-e3"]',
    )
    with pytest.raises(E3IdentityError, match="class metadata override"):
        _validate_copy(repo)


def test_direct_classes_tuple_override_fails(tmp_path):
    repo = _copy_static_repository(tmp_path)
    _append_dataset_config(
        repo,
        'data["test"]["classes"] = ("background", "not-e3")',
    )
    with pytest.raises(E3IdentityError, match="class metadata override"):
        _validate_copy(repo)


def test_palette_override_fails(tmp_path):
    repo = _copy_static_repository(tmp_path)
    _append_dataset_config(repo, 'data["test"]["palette"] = [[0, 0, 0]]')
    with pytest.raises(E3IdentityError, match="class metadata override"):
        _validate_copy(repo)


@pytest.mark.parametrize("metadata_key", ["metainfo", "metadata"])
def test_metadata_class_override_fails(tmp_path, metadata_key):
    repo = _copy_static_repository(tmp_path)
    _append_dataset_config(
        repo,
        f'data["test"][{metadata_key!r}] = {{"classes": ["not-e3"]}}',
    )
    with pytest.raises(E3IdentityError, match="class metadata override"):
        _validate_copy(repo)


def test_nested_dataset_wrapper_class_override_fails(tmp_path):
    repo = _copy_static_repository(tmp_path)
    _append_dataset_config(
        repo,
        'data["test"] = dict(type="RepeatDataset", times=1, '
        'dataset=dict(data["test"], classes=("background", "not-e3")))',
    )
    with pytest.raises(E3IdentityError, match=r"dataset test split\.dataset\.classes"):
        _validate_copy(repo)


def test_dataset_class_metadata_resolution_fails_closed(tmp_path, monkeypatch):
    repo = _copy_static_repository(tmp_path)

    def fail_resolution(_dataset_identity):
        raise E3IdentityError("cannot establish canonical dataset class metadata")

    monkeypatch.setattr(
        identity_module,
        "_resolve_dataset_class_names",
        fail_resolution,
    )
    with pytest.raises(
        E3IdentityError,
        match="cannot establish canonical dataset class metadata",
    ):
        _validate_copy(repo)


def test_canonical_prompt_template_passes():
    result = validate_static_configuration(repo_root=ROOT)
    assert result["evaluation_template"] == _identity()["evaluation"]["template"]


def test_wrong_prompt_template_fails(tmp_path):
    repo = _copy_static_repository(tmp_path)
    expected = _identity()["evaluation"]["template"]
    observed = f"not-{expected}"
    _mutate_eval_base_yaml(
        repo,
        lambda config: config["evaluate"].update({"template": observed}),
    )
    with pytest.raises(
        E3IdentityError,
        match=rf"expected {expected!r}, observed {observed!r}",
    ):
        _validate_copy(repo)


def test_explicit_canonical_text_token_flags_pass(tmp_path):
    repo = _copy_static_repository(tmp_path)
    expected = _identity()["model"]["flags"]
    _mutate_eval_yaml(
        repo,
        lambda config: config["model"].update(
            {
                name: expected[name]
                for name in ("use_avg_text_token", "keep_cls", "keep_end_seq")
            }
        ),
    )
    result = _validate_copy(repo)
    for name in ("use_avg_text_token", "keep_cls", "keep_end_seq"):
        assert result["effective_model_flags"][name] is expected[name]


def test_use_avg_text_token_wrong_value_fails(tmp_path):
    repo = _copy_static_repository(tmp_path)
    expected = _identity()["model"]["flags"]["use_avg_text_token"]
    _mutate_eval_yaml(
        repo,
        lambda config: config["model"].update(
            {"use_avg_text_token": not expected}
        ),
    )
    with pytest.raises(
        E3IdentityError,
        match=rf"model\.use_avg_text_token mismatch: expected {expected!r}",
    ):
        _validate_copy(repo)


@pytest.mark.parametrize("flag", ["keep_cls", "keep_end_seq"])
def test_wrong_text_token_retention_flag_fails(tmp_path, flag):
    repo = _copy_static_repository(tmp_path)
    expected = _identity()["model"]["flags"][flag]
    _mutate_eval_yaml(
        repo,
        lambda config: config["model"].update({flag: not expected}),
    )
    with pytest.raises(
        E3IdentityError,
        match=rf"model\.{flag} mismatch: expected {expected!r}",
    ):
        _validate_copy(repo)


def test_omitted_token_flags_use_production_constructor_defaults(tmp_path):
    repo = _copy_static_repository(tmp_path)
    selected = ("use_avg_text_token", "keep_cls", "keep_end_seq")

    def remove_flags(config):
        for name in selected:
            config["model"].pop(name, None)

    _mutate_eval_yaml(repo, remove_flags)
    result = _validate_copy(repo)
    expected = _identity()["model"]["flags"]
    for name in selected:
        assert result["effective_model_flags"][name] is expected[name]


@pytest.mark.parametrize(
    "flag",
    [
        "avg_self_attn_token",
        "disentangled_self_attn_token",
        "is_eval",
        "with_bg_clean",
    ],
)
def test_wrong_image_or_head_selection_flag_fails(tmp_path, flag):
    repo = _copy_static_repository(tmp_path)
    expected = _identity()["model"]["flags"][flag]
    _mutate_eval_yaml(
        repo,
        lambda config: config["model"].update({flag: not expected}),
    )
    with pytest.raises(
        E3IdentityError,
        match=rf"model\.{flag} mismatch: expected {expected!r}",
    ):
        _validate_copy(repo)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("model_name", "dinov2_vitl14_reg", "DINO backbone mismatch"),
        ("proj_name", "wrong_projection", "projection checkpoint name mismatch"),
    ],
)
def test_wrong_model_or_projection_identity_fails(tmp_path, key, value, message):
    repo = _copy_static_repository(tmp_path)
    _mutate_eval_yaml(repo, lambda config: config["model"].update({key: value}))
    with pytest.raises(E3IdentityError, match=message):
        _validate_copy(repo)


@pytest.mark.parametrize("forbidden_key", ["diffusion", "rwr", "cover_dr"])
def test_forbidden_mode_key_fails_even_when_disabled(tmp_path, forbidden_key):
    repo = _copy_static_repository(tmp_path)
    _mutate_eval_yaml(
        repo,
        lambda config: config.setdefault("evaluate", {}).update(
            {forbidden_key: False}
        ),
    )
    with pytest.raises(E3IdentityError, match="forbidden RWR/diffusion/COVER"):
        _validate_copy(repo)


def test_exact_full_precision_metrics_pass(tmp_path):
    path = _write_json(tmp_path / "metrics.json", _canonical_metrics())
    assert verify_result(path, source_kind="structured").startswith("E3 IDENTITY PASS")


def test_exact_aacc_reference_is_immutable():
    assert _identity()["expected_metrics"]["aAcc"] == 46.614213


def test_rounded_canonical_log_metrics_pass(tmp_path):
    metrics = _identity()["expected_metrics"]
    images = _identity()["dataset"]["images"]
    path = tmp_path / "eval.log"
    path.write_text(
        "+-------+-------+-------+\n"
        "| aAcc | mIoU | mAcc |\n"
        "+-------+-------+-------+\n"
        f"| {metrics['aAcc']:.2f} | {metrics['mIoU']:.2f} | {metrics['mAcc']:.2f} |\n"
        "+-------+-------+-------+\n"
        f"[coco_stuff] mIoU of {images} test images: {metrics['mIoU']:.2f}%\n",
        encoding="utf-8",
    )
    parsed = parse_log_metrics(path)
    assert parsed.image_count == images
    assert verify_metrics(parsed, _identity(), source_kind="log").startswith(
        "E3 IDENTITY PASS"
    )


def test_two_decimal_log_delta_005787_passes():
    identity = _identity()
    observed_aacc = 46.62
    assert observed_aacc - identity["expected_metrics"]["aAcc"] == pytest.approx(
        0.005787
    )
    parsed = identity_module.ParsedMetrics(
        values={
            "aAcc": observed_aacc,
            "mIoU": 28.48,
            "mAcc": 52.08,
        },
        image_count=identity["dataset"]["images"],
        printed_tokens={
            "aAcc": "46.62",
            "mIoU": "28.48",
            "mAcc": "52.08",
        },
    )
    assert verify_metrics(parsed, identity, source_kind="log").startswith(
        "E3 IDENTITY PASS"
    )


def test_two_decimal_log_delta_above_006_fails():
    identity = copy.deepcopy(_identity())
    identity["expected_metrics"]["aAcc"] = 46.613999
    observed_aacc = 46.62
    assert observed_aacc - identity["expected_metrics"]["aAcc"] == pytest.approx(
        0.006001
    )
    parsed = identity_module.ParsedMetrics(
        values={
            "aAcc": observed_aacc,
            "mIoU": identity["expected_metrics"]["mIoU"],
            "mAcc": identity["expected_metrics"]["mAcc"],
        },
        image_count=identity["dataset"]["images"],
        printed_tokens={
            "aAcc": "46.62",
            "mIoU": "28.48",
            "mAcc": "52.08",
        },
    )
    with pytest.raises(E3IdentityError, match=r"absolute_delta=0\.006001"):
        verify_metrics(parsed, identity, source_kind="log")


def test_structured_result_remains_strict_at_one_e_minus_six(tmp_path):
    value = _canonical_metrics()
    value["aAcc"] += 0.000002
    path = _write_json(tmp_path / "strict-structured.json", value)
    with pytest.raises(
        E3IdentityError,
        match=r"tolerance=1e-06.*absolute_delta=2\.0.*e-06",
    ):
        verify_result(path, source_kind="structured")


@pytest.mark.parametrize("metric", ["mIoU", "aAcc", "mAcc"])
def test_metric_mismatch_fails_with_diagnostics(tmp_path, metric):
    value = _canonical_metrics()
    value[metric] += 0.01
    path = _write_json(tmp_path / f"wrong-{metric}.json", value)
    with pytest.raises(
        E3IdentityError,
        match=rf"{metric}: expected=.*observed=.*tolerance=.*absolute_delta=",
    ):
        verify_result(path, source_kind="structured")


def test_missing_metric_fails(tmp_path):
    value = _canonical_metrics()
    del value["mAcc"]
    path = _write_json(tmp_path / "missing.json", value)
    with pytest.raises(E3IdentityError, match="missing: mAcc"):
        parse_structured_metrics(path)


def test_duplicate_or_ambiguous_metric_blocks_fail(tmp_path):
    metrics = _canonical_metrics()
    path = _write_json(tmp_path / "duplicate.json", {"first": metrics, "second": metrics})
    with pytest.raises(E3IdentityError, match="duplicate or ambiguous metric blocks"):
        parse_structured_metrics(path)


def test_fractional_metric_scale_fails(tmp_path):
    value = _canonical_metrics()
    for metric in ("aAcc", "mIoU", "mAcc"):
        value[metric] /= 100
    path = _write_json(tmp_path / "fractional.json", value)
    with pytest.raises(E3IdentityError, match=r"\[0,1\].*\[0,100\]"):
        verify_result(path, source_kind="structured")


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_structured_metric_fails(tmp_path, token):
    text = json.dumps(_canonical_metrics()).replace(
        str(_canonical_metrics()["mIoU"]), token
    )
    path = tmp_path / "nonfinite.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(E3IdentityError, match="non-finite"):
        parse_structured_metrics(path)


def test_wrong_image_count_fails_when_available(tmp_path):
    value = _canonical_metrics()
    value["evaluated_images"] -= 1
    path = _write_json(tmp_path / "wrong-count.json", value)
    with pytest.raises(E3IdentityError, match="image count mismatch"):
        verify_result(path, source_kind="structured")


def test_duplicate_log_metric_blocks_fail(tmp_path):
    metrics = _identity()["expected_metrics"]
    row = (
        "| aAcc | mIoU | mAcc |\n"
        f"| {metrics['aAcc']:.2f} | {metrics['mIoU']:.2f} | {metrics['mAcc']:.2f} |\n"
    )
    path = tmp_path / "duplicate.log"
    path.write_text(row + row, encoding="utf-8")
    with pytest.raises(E3IdentityError, match="duplicate or ambiguous metric blocks"):
        parse_log_metrics(path)


def test_missing_checkpoint_fails_only_when_explicitly_requested(tmp_path):
    repo = _copy_static_repository(tmp_path)
    _validate_copy(repo, check_checkpoint=False)
    with pytest.raises(E3IdentityError, match="missing E3 projection checkpoint"):
        _validate_copy(repo, check_checkpoint=True)


def test_optional_dataset_root_structure_check(tmp_path):
    repo = _copy_static_repository(tmp_path / "repo")
    dataset = tmp_path / "dataset"
    (dataset / "images/val2017").mkdir(parents=True)
    with pytest.raises(E3IdentityError, match="validation annotations"):
        _validate_copy(repo, dataset_root=dataset)
    (dataset / "annotations/val2017").mkdir(parents=True)
    result = _validate_copy(repo, dataset_root=dataset)
    assert result["dataset_checked"] is True


def test_cli_returns_zero_for_match_and_nonzero_for_mismatch(tmp_path):
    passing = _write_json(tmp_path / "passing.json", _canonical_metrics())
    failing_value = _canonical_metrics()
    failing_value["mIoU"] += 1
    failing = _write_json(tmp_path / "failing.json", failing_value)
    command = [sys.executable, str(ROOT / "verify_e3_identity.py"), "verify-result"]
    passed = subprocess.run(
        [*command, "--metrics-json", str(passing)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    failed = subprocess.run(
        [*command, "--metrics-json", str(failing)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert passed.returncode == 0
    assert "E3 IDENTITY PASS" in passed.stdout
    assert failed.returncode != 0
    assert "E3 IDENTITY FAIL" in failed.stderr
