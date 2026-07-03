import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

from eval_xattn_clean import (
    validate_component_cpa_eval_config,
    validate_rvs_eval_config,
)


def _config():
    cfg = OmegaConf.load(
        ROOT
        / "src/open_vocabulary_segmentation/configs/stuff/xattn_clean_coco_stuff.yml"
    )
    cfg.cpa.topk = 15
    cfg.cpa.residual_scale = 0.50
    cfg.cpa.residual_clip = 0.50
    return cfg


def test_component_cpa_does_not_require_clip_image_encoder():
    cfg = _config()
    cfg.evaluate.component_cpa_enabled = True
    cfg.evaluate.rvs_enabled = False
    cfg.clip_image.enabled = False
    assert validate_component_cpa_eval_config(cfg) is True
    assert validate_rvs_eval_config(cfg) is False


def test_rvs_requires_component_cpa_mode():
    cfg = _config()
    cfg.evaluate.component_cpa_enabled = False
    cfg.evaluate.rvs_enabled = True
    cfg.rvs.enabled = True
    cfg.clip_image.enabled = True
    with pytest.raises(ValueError, match="component_cpa_enabled"):
        validate_rvs_eval_config(cfg)
