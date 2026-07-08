import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

from eval_xattn_clean import load_clean_config, validate_msa_eval_config


CONFIG = (
    ROOT
    / "src/open_vocabulary_segmentation/configs/stuff/eval_xattn_clean_coco_stuff.yml"
)


def test_msa_disabled_preserves_legacy_protocol():
    cfg = load_clean_config(CONFIG)
    assert validate_msa_eval_config(cfg) is False


def test_single_scale_msa_configuration_is_valid():
    cfg = load_clean_config(
        CONFIG,
        [
            "evaluate.msa_enabled=true",
            "msa.enabled=true",
            "msa.scales=[1.0]",
            "msa.hflip=false",
        ],
    )
    assert validate_msa_eval_config(cfg) is True
    assert list(cfg.msa.scales) == [1.0]


def test_msa_requires_matching_enable_flags():
    cfg = load_clean_config(CONFIG, ["evaluate.msa_enabled=true"])
    with pytest.raises(ValueError, match="to match"):
        validate_msa_eval_config(cfg)


def test_msa_rejects_other_refinement_modules():
    cfg = load_clean_config(
        CONFIG,
        [
            "evaluate.msa_enabled=true",
            "msa.enabled=true",
            "rvs.enabled=true",
        ],
    )
    with pytest.raises(ValueError, match="rvs"):
        validate_msa_eval_config(cfg)
