# ------------------------------------------------------------------------------
# Talk2DINO
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import argparse
import csv
import datetime
import json
import math
import os
import os.path as osp
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="On January 1, 2023, MMCV will release v2.0.0.*",
    category=UserWarning,
)

import matplotlib.pyplot as plt
import mmcv
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch.distributed.distributed_c10d import _get_default_group
import numpy as np
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import collect_env, get_git_hash
from torch.utils.data import Subset

# from datasets import build_loader, build_text_transform
from models import build_model
from omegaconf import OmegaConf, read_write

from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_dinotext_seg_inference

from timm.utils import AverageMeter
from torchvision.utils import make_grid
from utils import (
    build_optimizer,
    build_scheduler,
    get_config,
    get_grad_norm,
    get_logger,
    parse_losses,
    load_config
)
import us
from utils import (
    build_optimizer,
    build_scheduler,
    get_config,
    get_grad_norm,
    get_logger,
    load_checkpoint,
    parse_losses,
    save_checkpoint,
    CheckpointManager,
    load_config
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

from mmseg.datasets import PIPELINES, PascalVOCDataset, PascalContextDataset, ADE20KDataset, CityscapesDataset, \
    COCOStuffDataset, PascalContextDataset59


@PIPELINES.register_module()
class FloatImage:
    def __call__(self, results):
        results['img'] = results['img'].astype(np.float32)
        return results


def cyclize(loader):
    while True:
        for i in loader:
            yield i


def get_argparser():
    parser = argparse.ArgumentParser("DINOText training and evaluation script")
    parser.add_argument("--cfg", "--config", dest="cfg", type=str, help="path to config file")
    parser.add_argument(
        "--opts", help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs="+"
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument(
        "--output",
        type=str,
        help="root of output folder, " "the full path is <output>/<model_name>/<tag>",
    )
    parser.add_argument("--tag", type=str, help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument(
        "--extract_features",
        action="store_true",
        help="Prepare feature extraction output if the selected method requires it",
    )
    parser.add_argument("--wandb", action="store_true", help="Use W&B to log experiments")
    parser.add_argument("--wandb_name", type=str, help="W&B run name", default="default")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval_cfg', type=str, default="configs/dinotext.yml")
    parser.add_argument(
        "--eval_base_cfg",
        "--eval_base",
        dest="eval_base_cfg",
        type=str,
        default="configs/eval.yml",
    )

    parser.add_argument("--pred_qual_path", type=str, default=None)
    parser.add_argument("--gt_qual_path", type=str, default=None)

    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--num_jobs", type=int, default=1)

    return parser

def log_results(miou, proj_name, bench, result_dir, logger):
    os.makedirs(result_dir, exist_ok=True)
    
    json_path = os.path.join(result_dir, f"{proj_name}.json")
    
    # Load existing data or start with an empty dictionary
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            corrupt_path = f"{json_path}.corrupt"
            os.replace(json_path, corrupt_path)
            logger.warning(
                f"Existing results JSON was corrupt and moved to {corrupt_path}"
            )
            data = {}
    else:
        data = {}
    
    # Add or update the benchmark result
    data[bench] = float(miou)
    
    # Write the updated data to the JSON file in human-readable format
    tmp_json_path = f"{json_path}.tmp"
    with open(tmp_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_json_path, json_path)
    logger.info(f"Saved results at {json_path}")


def load_saved_e0_result(config, benchmark):
    result_path = os.path.join(
        "segmentation_results",
        f"{config.model.proj_name}.json",
    )
    if not os.path.isfile(result_path):
        return None
    try:
        with open(result_path, "r") as result_file:
            value = json.load(result_file).get(benchmark)
        return None if value is None else float(value)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_sg_summary(output_dir, summary, logger):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "summary.json")
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(json_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    with open(csv_path, "w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    logger.info(f"Saved SG-Gate summary at {json_path} and {csv_path}")


def format_metric(value):
    return "not_available" if value is None or value == "not_available" else f"{float(value):.2f}%"


def format_scalar(value, digits=2):
    return (
        "not_available"
        if value is None or value == "not_available"
        else f"{float(value):.{digits}f}"
    )


def log_eval_summary(cfg, results, sg_summary, logger):
    logger.info("=" * 64)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 64)

    if sg_summary is not None:
        logger.info(f"Mode                 : {sg_summary['mode']}")
        logger.info(f"Ablation mode        : {sg_summary['ablation_mode']}")
        logger.info(f"Uses E1              : {sg_summary['uses_e1']}")
        logger.info(f"Uses E2 clusters     : {sg_summary['uses_e2_clusters']}")
        logger.info(f"Uses E3 gate         : {sg_summary['uses_e3_gate']}")
        logger.info(f"Dataset              : {sg_summary['dataset']}")
        logger.info(f"Images evaluated     : {sg_summary['num_images']}")
        logger.info(
            f"Job                  : {sg_summary['job_id'] + 1}/"
            f"{sg_summary['num_jobs']}"
        )
        logger.info(
            f"E0 no-PAMR mIoU      : "
            f"{format_metric(sg_summary['E0_no_pamr_mIoU'])}"
        )
        logger.info(
            f"E0 PAMR mIoU         : "
            f"{format_metric(sg_summary['E0_pamr_mIoU'])}"
        )
        logger.info(
            f"Full SG PAMR best    : "
            f"{format_metric(sg_summary['sg_full_soft_pamr_best_ref'])}"
        )
        logger.info(
            f"SG raw PAMR ref      : "
            f"{format_metric(sg_summary['sg_path_raw_pamr_ref'])}"
        )
        logger.info(
            f"E1 seed PAMR ref     : "
            f"{format_metric(sg_summary['e1_seed_only_pamr_ref'])}"
        )
        logger.info(
            f"Full SG no-PAMR best : "
            f"{format_metric(sg_summary['full_sg_soft_no_pamr_ref'])}"
        )
        logger.info(
            f"SG strict mIoU       : "
            f"{format_metric(sg_summary['SG_strict_mIoU'])}"
        )
        if sg_summary.get("pamr_dual_path_debug_enabled", False):
            logger.info(
                f"Path A raw PAMR mIoU : "
                f"{format_metric(sg_summary['path_a_miou'])}"
            )
            logger.info(
                f"Path B soft0 PAMR mIoU: "
                f"{format_metric(sg_summary['path_b_miou'])}"
            )
            logger.info(
                f"Path B - Path A mIoU : "
                f"{format_metric(sg_summary['path_b_minus_path_a_miou'])}"
            )
            logger.info(
                f"PAMR input max diff  : "
                f"{format_scalar(sg_summary['max_abs_diff_path_a_input_vs_path_b_input'], 6)}"
            )
            logger.info(
                f"PAMR output max diff : "
                f"{format_scalar(sg_summary['max_abs_diff_path_a_pamr_output_vs_path_b_pamr_output'], 6)}"
            )
            logger.info(
                f"Final label disagree : "
                f"{format_metric(sg_summary['final_label_disagreement_percent'])}"
            )
        if sg_summary.get("debug_force_raw_output", False):
            logger.info(
                f"Raw parity expected  : "
                f"{format_metric(sg_summary['raw_parity_expected_miou'])}"
            )
            logger.info(
                f"Raw parity actual    : "
                f"{format_metric(sg_summary['raw_parity_actual_miou'])}"
            )
            logger.info(
                f"Raw parity delta     : "
                f"{format_metric(sg_summary['raw_parity_delta'])}"
            )
        logger.info(
            f"Delta vs E0 no-PAMR  : "
            f"{format_metric(sg_summary['delta_vs_E0_no_pamr'])}"
        )
        logger.info(
            f"Delta vs E0 PAMR     : "
            f"{format_metric(sg_summary['delta_vs_E0_pamr'])}"
        )
        logger.info(
            f"Delta vs full SG best: "
            f"{format_metric(sg_summary['delta_vs_full_sg_soft_pamr_best'])}"
        )
        logger.info(
            f"Delta vs raw PAMR    : "
            f"{format_metric(sg_summary['delta_vs_sg_path_raw_pamr'])}"
        )
        logger.info(
            f"Delta vs E1 PAMR     : "
            f"{format_metric(sg_summary['delta_vs_e1_seed_only_pamr'])}"
        )
        logger.info(
            f"Delta vs full PAMR   : "
            f"{format_metric(sg_summary['delta_vs_full_sg_soft_pamr'])}"
        )
        logger.info(
            f"Delta vs full no-PAMR: "
            f"{format_metric(sg_summary['delta_vs_full_sg_soft_no_pamr'])}"
        )
        logger.info(
            f"SG diagnostic mIoU   : "
            f"{format_metric(sg_summary['SG_diagnostic_mIoU'])}"
        )
        logger.info(
            f"Ignored pixels       : "
            f"{format_metric(sg_summary['ignored_pixels_percent'])}"
        )
        logger.info(
            f"Positive pixels      : "
            f"{format_metric(sg_summary['positive_pixels_percent'])}"
        )
        logger.info(
            f"Fallback used        : "
            f"{format_metric(sg_summary['fallback_used_percent'])}"
        )
        logger.info(
            f"Fallback count       : "
            f"{sg_summary['fallback_used_count']}/"
            f"{sg_summary['total_class_decisions']}"
        )
        logger.info(
            f"Use SG               : "
            f"{format_metric(sg_summary['use_sg_percent'])}"
        )
        logger.info(
            f"Dense raw fallback   : "
            f"{sg_summary['dense_raw_fallback_enabled']}"
        )
        logger.info(
            f"SG raw baseline      : "
            f"{format_metric(sg_summary['sg_path_raw_baseline_miou'])}"
        )
        logger.info(
            f"Delta vs SG raw      : "
            f"{format_metric(sg_summary['delta_vs_sg_path_raw_baseline'])}"
        )
        logger.info(
            f"SG no-PAMR best      : "
            f"{format_metric(sg_summary['SG_no_pamr_best_mIoU'])}"
        )
        logger.info(
            f"Delta vs SG no-PAMR  : "
            f"{format_metric(sg_summary['delta_vs_SG_no_pamr_best'])}"
        )
        logger.info(
            f"Trusted SG classes   : "
            f"{format_metric(sg_summary['trusted_sg_class_percent'])}"
        )
        logger.info(
            f"SG overwrite pixels  : "
            f"{format_metric(sg_summary['sg_overwrite_pixel_percent'])}"
        )
        logger.info(
            f"Soft reweight        : "
            f"{sg_summary['soft_reweight_enabled']}"
        )
        logger.info(
            f"PAMR enabled         : "
            f"{sg_summary['pamr_enabled']}"
        )
        logger.info(
            f"Apply PAMR after soft: "
            f"{sg_summary['apply_pamr_after_soft']}"
        )
        logger.info(
            f"PAMR source          : "
            f"{sg_summary['pamr_source']}"
        )
        logger.info(
            f"Shuffle prior        : "
            f"{sg_summary['shuffle_prior_enabled']}"
        )
        logger.info(
            f"Shuffle mode         : "
            f"{sg_summary['shuffle_prior_mode']}"
        )
        logger.info(
            f"Shuffle seed         : "
            f"{sg_summary['shuffle_prior_seed']}"
        )
        logger.info(
            f"Random prior         : "
            f"{sg_summary['random_prior_enabled']}"
        )
        logger.info(
            f"Random mode          : "
            f"{sg_summary['random_prior_mode']}"
        )
        logger.info(
            f"Random seed          : "
            f"{sg_summary['random_prior_seed']}"
        )
        logger.info(
            f"Lambda boost         : "
            f"{sg_summary['lambda_boost']}"
        )
        logger.info(
            f"Top-k guard          : "
            f"{sg_summary['topk_guard']}"
        )
        logger.info(
            f"Margin guard         : "
            f"{sg_summary['margin_guard']}"
        )
        logger.info(
            f"Guard mode           : "
            f"{sg_summary['guard_mode']}"
        )
        logger.info(
            f"Uncertainty margin   : "
            f"{sg_summary['uncertainty_margin']}"
        )
        logger.info(
            f"Only trusted clusters: "
            f"{sg_summary['only_trusted_clusters']}"
        )
        logger.info(
            f"Candidate source     : "
            f"{sg_summary['soft_candidate_source']}"
        )
        logger.info(
            f"Min soft gate score  : "
            f"{sg_summary['min_soft_gate_score']}"
        )
        logger.info(
            f"Require anchor       : "
            f"{sg_summary['require_semantic_anchor']}"
        )
        logger.info(
            f"Raw support top-k    : "
            f"{sg_summary['raw_support_topk']}"
        )
        logger.info(
            f"Min raw top-k support: "
            f"{sg_summary['min_cluster_raw_topk_support']}"
        )
        logger.info(
            f"Soft candidate class : "
            f"{format_metric(sg_summary['soft_candidate_class_percent'])}"
        )
        logger.info(
            f"Soft candidate clust.: "
            f"{format_metric(sg_summary['soft_candidate_cluster_percent'])}"
        )
        logger.info(
            f"Raw-supported class  : "
            f"{format_metric(sg_summary['raw_supported_candidate_class_percent'])}"
        )
        logger.info(
            f"Raw-supported clust. : "
            f"{format_metric(sg_summary['raw_supported_candidate_cluster_percent'])}"
        )
        logger.info(
            f"Avg raw top-k support: "
            f"{format_metric(sg_summary['avg_cluster_raw_topk_support'])}"
        )
        logger.info(
            f"Normalize prior      : "
            f"{sg_summary['normalize_prior']}"
        )
        logger.info(
            f"Score scale          : "
            f"{sg_summary['score_scale_type']}"
        )
        logger.info(
            f"Hard overwrite       : "
            f"{sg_summary['hard_overwrite_enabled']}"
        )
        logger.info(
            f"Soft prior pixels    : "
            f"{format_metric(sg_summary['soft_prior_pixel_percent'])}"
        )
        logger.info(
            f"Random prior pixels  : "
            f"{format_metric(sg_summary['random_prior_density_percent'])}"
        )
        logger.info(
            f"Raw top-k prior pix. : "
            f"{format_metric(sg_summary['raw_topk_prior_density_percent'])}"
        )
        logger.info(
            f"Boosted pixels       : "
            f"{format_metric(sg_summary['boosted_pixel_percent'])}"
        )
        logger.info(
            f"Max diff before PAMR : "
            f"{format_scalar(sg_summary['max_abs_diff_before_pamr'], 6)}"
        )
        logger.info(
            f"Uncertain pixels     : "
            f"{format_metric(sg_summary['uncertain_pixel_percent'])}"
        )
        logger.info(
            f"Changed vs raw       : "
            f"{format_metric(sg_summary['changed_pixel_percent_vs_raw'])}"
        )
        logger.info(
            f"Raw preserved pixels : "
            f"{format_metric(sg_summary['raw_preserved_pixel_percent'])}"
        )
        logger.info(
            f"Low agreement IoU    : "
            f"{format_metric(sg_summary['fallback_low_agreement_iou_percent'])}"
        )
        logger.info(
            f"Low agreement count  : "
            f"{sg_summary['low_agreement_iou_count']}/"
            f"{sg_summary['total_class_decisions']}"
        )
        logger.info(
            f"Raw too small block  : "
            f"{format_metric(sg_summary['raw_too_small_block_sg_percent'])}"
        )
        logger.info(
            f"Raw block count      : "
            f"{sg_summary['raw_too_small_block_sg_count']}/"
            f"{sg_summary['total_class_decisions']}"
        )
        logger.info(
            f"Avg raw-SG IoU       : "
            f"{format_scalar(sg_summary['avg_raw_sg_agreement_iou'], 2)}"
        )
        logger.info(
            f"Min raw area pixels  : "
            f"{sg_summary['min_raw_area_pixels']}"
        )
        logger.info(
            f"Allow SG raw empty   : "
            f"{sg_summary['allow_sg_when_raw_empty']}"
        )
        logger.info(
            f"Min agreement IoU    : "
            f"{sg_summary['min_agreement_iou_threshold']:.2f}"
        )
        if sg_summary.get("agreement_iou_warning", False):
            logger.warning("WARNING: agreement IoU fallback may not be active.")
        logger.info(
            f"Alignment audit      : "
            f"{sg_summary['alignment_audit_enabled']}"
        )
        if sg_summary["alignment_audit_enabled"] is True:
            logger.info(
                f"Seed inside SG       : "
                f"{sg_summary['avg_seed_inside_sg_patch_rate']:.3f}"
            )
            logger.info(
                f"Seed hit SG          : "
                f"{format_metric(sg_summary['seed_hit_sg_percent'])}"
            )
            logger.info(
                f"Raw/seed patch IoU   : "
                f"{sg_summary['avg_raw_patch_seed_iou']:.3f}"
            )
            logger.info(
                f"SG roundtrip IoU     : "
                f"{sg_summary['avg_sg_roundtrip_iou']:.3f}"
            )
            logger.info(
                f"Raw/SG pre-fallback  : "
                f"{sg_summary['avg_raw_sg_iou_pre_fallback']:.3f}"
            )
            logger.info(
                f"Raw/final post-fb    : "
                f"{sg_summary['avg_raw_final_iou_post_fallback']:.3f}"
            )
        logger.info(
            f"Avg clusters/image   : "
            f"{format_scalar(sg_summary['avg_clusters_per_image'], 2)}"
        )
        logger.info(f"Summary JSON         : {os.path.join(cfg.output, 'summary.json')}")
        logger.info(f"Summary CSV          : {os.path.join(cfg.output, 'summary.csv')}")
    else:
        logger.info("Mode                 : E0 official")
        for key, value in results.items():
            if key.startswith("val/") and key.endswith("_miou"):
                dataset = key[len("val/"):-len("_miou")]
            logger.info(
                f"{dataset + ' mIoU':<21}: {format_metric(value)}"
            )


def log_xattn_bridge_startup(cfg, model, logger):
    model_without_ddp = model.module if hasattr(model, "module") else model
    xattn_cfg = cfg.get("xattn_bridge", cfg.model.get("xattn_bridge", {}))
    if not bool(xattn_cfg.get("enabled", False)):
        return

    if hasattr(model_without_ddp, "assert_frozen_backbones"):
        model_without_ddp.assert_frozen_backbones()
    trainable_names = (
        model_without_ddp.trainable_parameter_names()
        if hasattr(model_without_ddp, "trainable_parameter_names")
        else [name for name, param in model_without_ddp.named_parameters() if param.requires_grad]
    )
    trainable_count = sum(
        param.numel() for param in model_without_ddp.parameters() if param.requires_grad
    )
    if bool(xattn_cfg.get("train_bridge_only", True)):
        bad_names = [name for name in trainable_names if "xattn_bridge" not in name]
        if bad_names:
            raise AssertionError(
                "xattn_bridge.train_bridge_only=true but non-bridge parameters are trainable: "
                f"{bad_names[:12]}"
            )

    logger.info("=" * 64)
    logger.info("Talk2DINO_XAttnBridge startup")
    logger.info("=" * 64)
    logger.info(f"Bridge type           : {xattn_cfg.get('type', 'text_queries_dino')}")
    logger.info("CLIP frozen           : true")
    logger.info("DINO frozen           : true")
    logger.info(f"Train bridge only     : {xattn_cfg.get('train_bridge_only', True)}")
    logger.info(f"Trainable params      : {trainable_count} ({trainable_count / 1e6:.3f}M)")
    logger.info(f"Trainable modules     : {', '.join(trainable_names[:16])}")
    if len(trainable_names) > 16:
        logger.info(f"Trainable modules     : ... +{len(trainable_names) - 16} more")
    logger.info("Feature source        : online frozen CLIP/DINO features")
    logger.info(f"Output directory      : {cfg.output}")
    logger.info(f"Auto-resume           : {cfg.train.get('auto_resume', False)}")
    logger.info(f"Total steps           : {cfg.train.get('total_steps', 'not_available')}")
    logger.info(f"Batch size            : {cfg.data.get('batch_size', 'not_available')}")
    logger.info(f"Learning rate         : {cfg.train.get('base_lr', 'not_available')}")
    logger.info(f"Runtime budget        : {cfg.train.get('max_train_hours', 8)}h")
    logger.info(f"Debug base only       : {xattn_cfg.get('debug_base_only', False)}")
    logger.info(f"Force gamma zero      : {xattn_cfg.get('force_gamma_zero', False)}")
    logger.info(f"Disable correction    : {xattn_cfg.get('disable_correction', False)}")
    logger.info(f"Safe init             : {xattn_cfg.get('safe_init', True)}")
    logger.info(f"Zero init out_proj    : {xattn_cfg.get('zero_init_out_proj', True)}")
    logger.info(f"Normalize correction  : {xattn_cfg.get('normalize_correction', True)}")
    logger.info(f"Ratio clamp           : {xattn_cfg.get('correction_ratio_clamp', True)}")
    logger.info(f"Max correction ratio  : {xattn_cfg.get('max_correction_base_ratio', 0.05)}")
    logger.info(f"Base source           : {xattn_cfg.get('base_source', 'original_talk2dino_proj')}")
    logger.info(
        f"Debug base compare    : "
        f"{xattn_cfg.get('debug_compare_base_projection', False)}"
    )
    logger.info(f"Debug tensor parity   : {xattn_cfg.get('debug_tensor_parity', False)}")
    logger.info(f"Hard return base      : {xattn_cfg.get('hard_return_base', False)}")
    logger.info(f"Gamma max clamp       : {xattn_cfg.get('gamma_max', None)}")
    logger.info(f"Correction ratio warn : {xattn_cfg.get('correction_ratio_warn', 0.20)}")
    logger.info(f"Correction ratio stop : {xattn_cfg.get('correction_ratio_stop', None)}")


def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(max(0, seconds))))


def get_xattn_diagnostics(model):
    model_without_ddp = model.module if hasattr(model, "module") else model
    bridge = getattr(model_without_ddp, "xattn_bridge", None)
    if bridge is None:
        return {}
    return getattr(bridge, "last_diagnostics", {}) or {}


def get_xattn_out_proj_norms(model):
    model_without_ddp = model.module if hasattr(model, "module") else model
    bridge = getattr(model_without_ddp, "xattn_bridge", None)
    if bridge is None:
        return None
    return {
        "weight": float(bridge.out_proj.weight.detach().float().norm().cpu()),
        "bias": float(bridge.out_proj.bias.detach().float().norm().cpu()),
    }


def set_xattn_checkpoint_status(model, loaded, path):
    model_without_ddp = model.module if hasattr(model, "module") else model
    if hasattr(model_without_ddp, "xattn_checkpoint_loaded"):
        model_without_ddp.xattn_checkpoint_loaded = bool(loaded)
        model_without_ddp.xattn_checkpoint_path = path


def format_xattn_diagnostics(diag):
    if not diag:
        return ""
    return (
        f" gamma={diag.get('gamma', 0.0):.4f}"
        f" base_norm={diag.get('base_norm', 0.0):.3f}"
        f" corr_pre={diag.get('correction_norm_before_clamp', diag.get('correction_norm', 0.0)):.3f}"
        f" corr_post={diag.get('correction_norm_after_clamp', diag.get('correction_norm', 0.0)):.3f}"
        f" ratio_pre={diag.get('correction_base_ratio_before_clamp', diag.get('correction_base_ratio', 0.0)):.3f}"
        f" ratio_post={diag.get('correction_base_ratio_after_clamp', diag.get('correction_base_ratio', 0.0)):.3f}"
    )


def parameter_count_by_name(model, predicate):
    model_without_ddp = model.module if hasattr(model, "module") else model
    return sum(
        param.numel()
        for name, param in model_without_ddp.named_parameters()
        if predicate(name, param)
    )


def grad_norm_by_name(model, name_predicate):
    model_without_ddp = model.module if hasattr(model, "module") else model
    sq_sum = 0.0
    found = False
    for name, param in model_without_ddp.named_parameters():
        if name_predicate(name) and param.grad is not None:
            grad = param.grad.detach().float()
            sq_sum += float(grad.pow(2).sum().cpu())
            found = True
    return math.sqrt(sq_sum) if found else 0.0


def log_xattn_gradient_debug(config, model, loss, lr, xattn_diag, logger):
    clip_trainable = parameter_count_by_name(
        model,
        lambda name, param: name.startswith("clip_model.") and param.requires_grad,
    )
    dino_trainable = parameter_count_by_name(
        model,
        lambda name, param: name.startswith("model.") and param.requires_grad,
    )
    proj_trainable = parameter_count_by_name(
        model,
        lambda name, param: name.startswith("proj.") and param.requires_grad,
    )
    bridge_trainable = parameter_count_by_name(
        model,
        lambda name, param: name.startswith("xattn_bridge.") and param.requires_grad,
    )
    logger.info("=" * 64)
    logger.info("XAttnBridge one-step gradient debug")
    logger.info("=" * 64)
    logger.info(f"CLIP trainable params        : {clip_trainable}")
    logger.info(f"DINO trainable params        : {dino_trainable}")
    logger.info(f"original MLP/proj trainable  : {proj_trainable}")
    logger.info(f"XAttnBridge trainable params : {bridge_trainable}")
    logger.info(f"gamma grad norm              : {grad_norm_by_name(model, lambda n: n == 'xattn_bridge.gamma'):.8f}")
    logger.info(f"text_proj grad norm          : {grad_norm_by_name(model, lambda n: n.startswith('xattn_bridge.text_proj.')):.8f}")
    logger.info(f"patch_k_proj grad norm       : {grad_norm_by_name(model, lambda n: n.startswith('xattn_bridge.patch_k_proj.')):.8f}")
    logger.info(f"patch_v_proj grad norm       : {grad_norm_by_name(model, lambda n: n.startswith('xattn_bridge.patch_v_proj.')):.8f}")
    logger.info(f"attention grad norm          : {grad_norm_by_name(model, lambda n: n.startswith('xattn_bridge.attn_layers.')):.8f}")
    logger.info(f"out_proj grad norm           : {grad_norm_by_name(model, lambda n: n.startswith('xattn_bridge.out_proj.')):.8f}")
    logger.info(f"base_norm                    : {float(xattn_diag.get('base_norm', 0.0)):.8f}")
    logger.info(
        f"correction_norm_before_clamp : "
        f"{float(xattn_diag.get('correction_norm_before_clamp', xattn_diag.get('correction_norm', 0.0))):.8f}"
    )
    logger.info(
        f"correction_norm_after_clamp  : "
        f"{float(xattn_diag.get('correction_norm_after_clamp', xattn_diag.get('correction_norm', 0.0))):.8f}"
    )
    logger.info(
        f"correction/base before clamp : "
        f"{float(xattn_diag.get('correction_base_ratio_before_clamp', xattn_diag.get('correction_base_ratio', 0.0))):.8f}"
    )
    logger.info(
        f"correction/base after clamp  : "
        f"{float(xattn_diag.get('correction_base_ratio_after_clamp', xattn_diag.get('correction_base_ratio', 0.0))):.8f}"
    )
    logger.info(f"gamma                        : {float(xattn_diag.get('gamma', 0.0)):.8f}")
    logger.info(f"base_source                  : {xattn_diag.get('base_source', 'not_available')}")
    logger.info(f"loss                         : {float(loss.detach().float().cpu()):.8f}")
    logger.info(f"lr                           : {float(lr):.8e}")


def move_training_batch_to_device(samples, target_device):
    moved = {}
    for key, value in samples.items():
        if torch.is_tensor(value):
            moved[key] = value.to(target_device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def print_train_progress(step, total_steps, loss_meter, lr, start_time, xattn_diag=None):
    total_steps = max(1, int(total_steps))
    step = min(int(step), total_steps)
    width = 30
    frac = step / total_steps
    filled = int(width * frac)
    bar = "=" * filled + ">" + "." * max(0, width - filled - 1)
    elapsed = time.time() - start_time
    rate = step / elapsed if elapsed > 0 and step > 0 else 0.0
    eta = (total_steps - step) / rate if rate > 0 else 0.0
    print(
        "\r"
        f"Train [{bar}] {step}/{total_steps} ({100.0 * frac:5.1f}%) "
        f"loss={loss_meter.avg:.4f} lr={lr:.2e} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
        f"{format_xattn_diagnostics(xattn_diag or {})}",
        end="",
        flush=True,
    )

def config_value(config, key, default="not_available"):
    value = config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)
    return default if value is None else value


def percent(numerator, denominator):
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def mean_value(values):
    return 0.0 if not values else float(sum(values) / len(values))


def median_value(values):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def summarize_sg_stats(config, strict_metric, diagnostic_metric, metric, num_images):
    stats = metric.get("stats", {})
    class_decisions = stats.get("total_class_decisions", stats.get("class_decisions", 0))
    total_pixels = metric.get("total_pixels", 0)
    patch_pixels = stats.get("num_patches", 0)
    e0_no_pamr = config_value(config.sg_gate, "e0_no_pamr_miou")
    e0_pamr = config_value(config.sg_gate, "e0_pamr_miou")
    sg_full_soft_pamr_best = config_value(config.sg_gate, "sg_full_soft_pamr_best_miou")
    sg_path_raw_pamr = config_value(config.sg_gate, "sg_path_raw_pamr_miou")
    e1_seed_only_pamr = config_value(config.sg_gate, "e1_seed_only_pamr_miou")
    full_sg_soft_pamr = config_value(config.sg_gate, "full_sg_soft_pamr_miou")
    sg_strict = float(strict_metric["mIoU"] * 100)
    diagnostic_value = (
        "not_available"
        if diagnostic_metric is None
        else float(diagnostic_metric["mIoU"] * 100)
    )
    path_a_metric = metric.get("pamr_path_a")
    path_b_metric = metric.get("pamr_path_b")
    path_a_miou = (
        "not_available"
        if path_a_metric is None
        else float(path_a_metric["mIoU"] * 100)
    )
    path_b_miou = (
        "not_available"
        if path_b_metric is None
        else float(path_b_metric["mIoU"] * 100)
    )
    path_b_minus_path_a = (
        "not_available"
        if path_a_miou == "not_available" or path_b_miou == "not_available"
        else float(path_b_miou) - float(path_a_miou)
    )

    def delta(reference):
        return "not_available" if reference == "not_available" else sg_strict - float(reference)

    fallback_enabled = bool(config.sg_gate.get("raw_fallback", {}).get("enabled", False))
    debug_force_raw = bool(config.sg_gate.get("debug_force_raw_output", False))
    debug_force_raw_apply_pamr = bool(
        config.sg_gate.get("debug_force_raw_apply_pamr", False)
    )
    ablation_mode = config.sg_gate.get("ablation_mode", "full_e1_e2_e3")
    dense_raw_fallback_enabled = bool(
        config.sg_gate.get("dense_raw_fallback", {}).get("enabled", False)
    )
    soft_cfg = config.sg_gate.get("soft_reweight", {})
    soft_reweight_enabled = bool(soft_cfg.get("enabled", False))
    disable_hard_overwrite = bool(soft_cfg.get("disable_hard_overwrite", False))
    apply_pamr_after_soft = bool(soft_cfg.get("apply_pamr_after_soft", False))
    sg_path_raw_baseline = config_value(config.sg_gate, "sg_path_raw_baseline_miou")
    sg_no_pamr_best = config_value(config.sg_gate, "sg_no_pamr_best_miou")
    fallback_cfg = config.sg_gate.get("raw_fallback", {})
    fallback_used_count = stats.get("fallback_used_count", stats.get("fallback_used", 0))
    low_agreement_iou_count = stats.get(
        "low_agreement_iou_count",
        stats.get("low_agreement_iou", stats.get("fallback_low_agreement_iou", 0)),
    )
    raw_too_small_block_sg_count = stats.get(
        "raw_too_small_block_sg_count",
        stats.get("raw_too_small_block_sg", stats.get("fallback_raw_too_small_block_sg", 0)),
    )
    avg_raw_sg_agreement_iou = (
        0.0
        if stats.get("raw_sg_agreement_iou_count", 0) == 0
        else stats.get("raw_sg_agreement_iou_sum", 0.0)
        / stats["raw_sg_agreement_iou_count"]
    )
    min_agreement_iou_threshold = float(fallback_cfg.get("min_agreement_iou", 0.35))
    min_raw_area_pixels = int(fallback_cfg.get("min_raw_area_pixels", 16))
    allow_sg_when_raw_empty = bool(fallback_cfg.get("allow_sg_when_raw_empty", False))
    low_agreement_iou_percent = percent(low_agreement_iou_count, class_decisions)
    raw_too_small_block_sg_percent = percent(raw_too_small_block_sg_count, class_decisions)
    audit_cfg = config.sg_gate.get("alignment_audit", {})
    audit_enabled = bool(audit_cfg.get("enabled", False))
    raw_patch_seed_iou_values = stats.get("alignment_raw_patch_seed_iou_values", [])
    sg_roundtrip_iou_values = stats.get("alignment_sg_roundtrip_iou_values", [])
    raw_sg_iou_pre_values = stats.get("alignment_raw_sg_iou_pre_fallback_values", [])
    raw_final_iou_post_values = stats.get("alignment_raw_final_iou_post_fallback_values", [])
    seed_inside_count = stats.get("alignment_seed_inside_sg_patch_rate_count", 0)
    raw_patch_seed_count = stats.get("alignment_raw_patch_seed_count", 0)
    raw_parity_reference = e0_pamr if debug_force_raw_apply_pamr else e0_no_pamr
    raw_parity_delta = (
        "not_available"
        if raw_parity_reference == "not_available"
        else sg_strict - float(raw_parity_reference)
    )
    delta_vs_sg_path_raw_baseline = (
        "not_available"
        if sg_path_raw_baseline == "not_available"
        else sg_strict - float(sg_path_raw_baseline)
    )
    delta_vs_sg_no_pamr_best = (
        "not_available"
        if sg_no_pamr_best == "not_available"
        else sg_strict - float(sg_no_pamr_best)
    )
    delta_vs_full_sg_soft_pamr_best = (
        "not_available"
        if sg_full_soft_pamr_best == "not_available"
        else sg_strict - float(sg_full_soft_pamr_best)
    )
    delta_vs_sg_path_raw_pamr = (
        "not_available"
        if sg_path_raw_pamr == "not_available"
        else sg_strict - float(sg_path_raw_pamr)
    )
    delta_vs_e1_seed_only_pamr = (
        "not_available"
        if e1_seed_only_pamr == "not_available"
        else sg_strict - float(e1_seed_only_pamr)
    )
    delta_vs_full_sg_soft_pamr = (
        "not_available"
        if full_sg_soft_pamr == "not_available"
        else sg_strict - float(full_sg_soft_pamr)
    )
    dense_total_pixels = stats.get("dense_total_pixels", total_pixels)
    sg_overwrite_pixels = stats.get(
        "sg_overwrite_pixels",
        stats.get("trusted_sg_pixel_overwrite_sum", 0),
    )
    soft_prior_pixels = stats.get("soft_prior_pixels", 0)
    boosted_pixels = stats.get("boosted_pixels", 0)
    changed_pixels_vs_raw = stats.get("changed_pixels_vs_raw", sg_overwrite_pixels)
    raw_preserved_pixels = stats.get(
        "raw_preserved_pixels",
        max(0, dense_total_pixels - changed_pixels_vs_raw),
    )
    uncertain_pixels = stats.get("uncertain_pixels", 0)
    random_prior_density_pixels = stats.get("random_prior_density_pixels", 0)
    raw_topk_prior_density_pixels = stats.get("raw_topk_prior_density_pixels", 0)
    raw_supported_support_count = stats.get("raw_supported_topk_support_count", 0)
    pamr_output_diff_count = stats.get(
        "mean_abs_diff_path_a_pamr_output_vs_path_b_pamr_output_count",
        0,
    )
    mean_pamr_output_diff = (
        0.0
        if pamr_output_diff_count == 0
        else stats.get(
            "mean_abs_diff_path_a_pamr_output_vs_path_b_pamr_output_sum",
            0.0,
        )
        / pamr_output_diff_count
    )
    avg_cluster_raw_topk_support = (
        0.0
        if raw_supported_support_count == 0
        else stats.get("raw_supported_topk_support_sum", 0.0)
        / raw_supported_support_count
    )
    candidate_source = soft_cfg.get("candidate_source", "trusted_clusters")
    uses_e1 = not debug_force_raw and candidate_source in {
        "seed_only",
        "seed_overlap_clusters",
        "trusted_clusters",
        "anchored_clusters",
        "raw_supported_clusters",
    }
    uses_e2_clusters = not debug_force_raw and candidate_source in {
        "seed_overlap_clusters",
        "trusted_clusters",
        "anchored_clusters",
        "raw_supported_clusters",
    }
    uses_e3_gate = not debug_force_raw and candidate_source in {
        "trusted_clusters",
        "anchored_clusters",
        "raw_supported_clusters",
    }
    min_seed_overlap = config.sg_gate.get("seed_overlap_clusters", {}).get(
        "min_seed_overlap",
        "not_available",
    )
    shuffle_cfg = config.sg_gate.get("shuffle_prior", {})
    shuffle_prior_enabled = bool(shuffle_cfg.get("enabled", False))
    summary = {
        "mode": (
            "SG-Gate debug force raw output"
            if debug_force_raw
            else "SG-Gate + raw_fallback" if fallback_enabled else "SG-Gate"
        ),
        "ablation_mode": ablation_mode,
        "uses_e1": uses_e1,
        "uses_e2_clusters": uses_e2_clusters,
        "uses_e3_gate": uses_e3_gate,
        "debug_force_raw_output": debug_force_raw,
        "debug_force_raw_apply_pamr": debug_force_raw_apply_pamr,
        "raw_parity_expected_miou": raw_parity_reference,
        "raw_parity_actual_miou": sg_strict if debug_force_raw else "not_available",
        "raw_parity_delta": raw_parity_delta if debug_force_raw else "not_available",
        "raw_prediction_source": (
            "official_e0_branch" if debug_force_raw else "sg_gate_branch"
        ),
        "dense_raw_fallback_enabled": dense_raw_fallback_enabled,
        "soft_reweight_enabled": soft_reweight_enabled,
        "apply_pamr_after_soft": apply_pamr_after_soft,
        "pamr_enabled": bool(
            config.evaluate.get("pamr", False)
            or apply_pamr_after_soft
            or debug_force_raw_apply_pamr
        ),
        "evaluate_pamr": bool(config.evaluate.get("pamr", False)),
        "pamr_source": stats.get(
            "pamr_source",
            "official_e0_pamr_branch"
            if (apply_pamr_after_soft or debug_force_raw_apply_pamr)
            else "not_available",
        ),
        "shuffle_prior_enabled": stats.get(
            "shuffle_prior_enabled",
            shuffle_prior_enabled,
        ),
        "shuffle_prior_mode": stats.get(
            "shuffle_prior_mode",
            shuffle_cfg.get(
                "mode",
                "class_permutation" if shuffle_prior_enabled else "not_available",
            ),
        ),
        "shuffle_prior_seed": stats.get(
            "shuffle_prior_seed",
            shuffle_cfg.get("seed", 42 if shuffle_prior_enabled else "not_available"),
        ),
        "random_prior_enabled": bool(config.sg_gate.get("random_prior", {}).get("enabled", False)),
        "random_prior_mode": config.sg_gate.get("random_prior", {}).get(
            "mode",
            "not_available",
        ),
        "random_prior_seed": config.sg_gate.get("random_prior", {}).get(
            "seed",
            "not_available",
        ),
        "lambda_boost": soft_cfg.get("lambda_boost", "not_available"),
        "topk_guard": soft_cfg.get("topk_guard", "not_available"),
        "margin_guard": soft_cfg.get("margin_guard", "not_available"),
        "guard_mode": soft_cfg.get("guard_mode", "current_or"),
        "uncertainty_margin": soft_cfg.get("uncertainty_margin", "not_available"),
        "only_trusted_clusters": soft_cfg.get("only_trusted_clusters", "not_available"),
        "normalize_prior": soft_cfg.get("normalize_prior", "not_available"),
        "score_scale": soft_cfg.get("score_scale", "not_available"),
        "soft_candidate_source": candidate_source,
        "candidate_source": candidate_source,
        "min_soft_gate_score": soft_cfg.get("min_soft_gate_score", "not_available"),
        "require_semantic_anchor": soft_cfg.get("require_semantic_anchor", "not_available"),
        "raw_support_topk": soft_cfg.get("raw_support_topk", "not_available"),
        "min_cluster_raw_topk_support": soft_cfg.get(
            "min_cluster_raw_topk_support",
            "not_available",
        ),
        "hard_overwrite_enabled": (
            False if debug_force_raw else not (soft_reweight_enabled and disable_hard_overwrite)
        ),
        "hard_overwrite": (
            False if debug_force_raw else not (soft_reweight_enabled and disable_hard_overwrite)
        ),
        "score_scale_type": stats.get(
            "score_scale_type",
            "std" if soft_reweight_enabled else "not_available",
        ),
        "sg_path_raw_baseline_miou": sg_path_raw_baseline,
        "sg_path_raw_no_pamr_ref": sg_path_raw_baseline,
        "delta_vs_sg_path_raw_baseline": delta_vs_sg_path_raw_baseline,
        "delta_vs_sg_path_raw_no_pamr": delta_vs_sg_path_raw_baseline,
        "SG_no_pamr_best_mIoU": sg_no_pamr_best,
        "full_sg_soft_no_pamr_ref": sg_no_pamr_best,
        "delta_vs_SG_no_pamr_best": delta_vs_sg_no_pamr_best,
        "delta_vs_full_sg_soft_no_pamr": delta_vs_sg_no_pamr_best,
        "E0_no_pamr_mIoU": e0_no_pamr,
        "E0_pamr_mIoU": e0_pamr,
        "e0_no_pamr_miou_ref": e0_no_pamr,
        "e0_pamr_miou_ref": e0_pamr,
        "expected_e0_pamr_miou": e0_pamr,
        "sg_full_soft_pamr_best_ref": sg_full_soft_pamr_best,
        "sg_path_raw_pamr_ref": sg_path_raw_pamr,
        "sg_raw_pamr_ref": sg_path_raw_pamr,
        "e1_seed_only_pamr_ref": e1_seed_only_pamr,
        "e1_seed_pamr_ref": e1_seed_only_pamr,
        "full_sg_soft_pamr_ref": full_sg_soft_pamr,
        "full_sg_pamr_ref": full_sg_soft_pamr,
        "SG_strict_mIoU": sg_strict,
        "mIoU": sg_strict,
        "miou": sg_strict,
        "actual_miou": sg_strict,
        "pamr_dual_path_debug_enabled": bool(
            stats.get("pamr_dual_path_debug_enabled", False)
        ),
        "path_a_miou": path_a_miou,
        "path_b_miou": path_b_miou,
        "path_b_minus_path_a_miou": path_b_minus_path_a,
        "raw_scores_shape": stats.get("raw_scores_shape", "not_available"),
        "path_a_pamr_input_shape": stats.get(
            "path_a_pamr_input_shape",
            "not_available",
        ),
        "path_b_pamr_input_shape": stats.get(
            "path_b_pamr_input_shape",
            "not_available",
        ),
        "raw_scores_dtype": stats.get("raw_scores_dtype", "not_available"),
        "path_a_input_dtype": stats.get("path_a_input_dtype", "not_available"),
        "path_b_input_dtype": stats.get("path_b_input_dtype", "not_available"),
        "max_abs_diff_raw_vs_path_a_input": stats.get(
            "max_abs_diff_raw_vs_path_a_input",
            0.0,
        ),
        "max_abs_diff_raw_vs_path_b_input": stats.get(
            "max_abs_diff_raw_vs_path_b_input",
            0.0,
        ),
        "max_abs_diff_path_a_input_vs_path_b_input": stats.get(
            "max_abs_diff_path_a_input_vs_path_b_input",
            0.0,
        ),
        "max_abs_diff_path_a_pamr_output_vs_path_b_pamr_output": stats.get(
            "max_abs_diff_path_a_pamr_output_vs_path_b_pamr_output",
            0.0,
        ),
        "mean_abs_diff_path_a_pamr_output_vs_path_b_pamr_output": mean_pamr_output_diff,
        "final_label_disagreement_percent": percent(
            stats.get("final_label_disagreement_pixels", 0),
            stats.get("final_label_disagreement_total", 0),
        ),
        "path_a_pamr_num_iter": stats.get("path_a_pamr_num_iter", "not_available"),
        "path_b_pamr_num_iter": stats.get("path_b_pamr_num_iter", "not_available"),
        "path_a_pamr_dilations": stats.get("path_a_pamr_dilations", "not_available"),
        "path_b_pamr_dilations": stats.get("path_b_pamr_dilations", "not_available"),
        "pamr_kernel_or_neighborhood": stats.get(
            "pamr_kernel_or_neighborhood",
            "not_available",
        ),
        "path_a_image_normalization_input": stats.get(
            "path_a_image_normalization_input",
            "not_available",
        ),
        "path_b_image_normalization_input": stats.get(
            "path_b_image_normalization_input",
            "not_available",
        ),
        "pamr_uses_image_rgb_or_feature": stats.get(
            "pamr_uses_image_rgb_or_feature",
            "not_available",
        ),
        "path_a_scores_are": stats.get("path_a_scores_are", "not_available"),
        "path_b_scores_are": stats.get("path_b_scores_are", "not_available"),
        "path_a_softmax_before_pamr": stats.get(
            "path_a_softmax_before_pamr",
            "not_available",
        ),
        "path_b_softmax_before_pamr": stats.get(
            "path_b_softmax_before_pamr",
            "not_available",
        ),
        "path_a_resize_before_or_after_pamr": stats.get(
            "path_a_resize_before_or_after_pamr",
            "not_available",
        ),
        "path_b_resize_before_or_after_pamr": stats.get(
            "path_b_resize_before_or_after_pamr",
            "not_available",
        ),
        "path_a_final_argmax_resolution": stats.get(
            "path_a_final_argmax_resolution",
            "not_available",
        ),
        "path_b_final_argmax_resolution": stats.get(
            "path_b_final_argmax_resolution",
            "not_available",
        ),
        "evaluator_input_resolution": stats.get(
            "evaluator_input_resolution",
            "not_available",
        ),
        "delta_vs_E0_no_pamr": delta(e0_no_pamr),
        "delta_vs_E0_pamr": delta(e0_pamr),
        "delta_vs_e0_no_pamr": delta(e0_no_pamr),
        "delta_vs_e0_pamr": delta(e0_pamr),
        "delta_vs_full_sg_soft_pamr_best": delta_vs_full_sg_soft_pamr_best,
        "delta_vs_sg_path_raw_pamr": delta_vs_sg_path_raw_pamr,
        "delta_vs_sg_raw_pamr": delta_vs_sg_path_raw_pamr,
        "delta_vs_e1_seed_only_pamr": delta_vs_e1_seed_only_pamr,
        "delta_vs_e1_seed_pamr": delta_vs_e1_seed_only_pamr,
        "delta_vs_full_sg_soft_pamr": delta_vs_full_sg_soft_pamr,
        "delta_vs_full_sg_pamr": delta_vs_full_sg_soft_pamr,
        "SG_diagnostic_mIoU": diagnostic_value,
        "ignored_pixels_percent": percent(metric.get("ignore_pixels", 0), total_pixels),
        "positive_pixels_percent": percent(metric.get("positive_pixels", 0), total_pixels),
        "ignore_pixels_before_strict_percent": percent(
            stats.get("ignore_pixels_before_strict", 0),
            patch_pixels,
        ),
        "ignore_pixels_after_compile_percent": percent(
            metric.get("ignore_pixels_after_compile", 0),
            total_pixels,
        ),
        "total_class_decisions": class_decisions,
        "fallback_used_count": fallback_used_count,
        "low_agreement_iou_count": low_agreement_iou_count,
        "raw_too_small_block_sg_count": raw_too_small_block_sg_count,
        "fallback_used_percent": percent(fallback_used_count, class_decisions),
        "fallback_sg_too_small_percent": percent(stats.get("sg_too_small", stats.get("fallback_sg_too_small", 0)), class_decisions),
        "fallback_sg_too_large_percent": percent(stats.get("sg_too_large", stats.get("fallback_sg_too_large", 0)), class_decisions),
        "fallback_sg_empty_percent": percent(stats.get("sg_empty", stats.get("fallback_sg_empty", 0)), class_decisions),
        "fallback_low_agreement_iou_percent": low_agreement_iou_percent,
        "raw_too_small_block_sg_percent": raw_too_small_block_sg_percent,
        "fallback_raw_empty_no_anchor_percent": percent(
            stats.get("raw_empty_no_anchor", stats.get("fallback_raw_empty_no_anchor", 0)),
            class_decisions,
        ),
        "use_sg_percent": percent(stats.get("use_sg", 0), class_decisions),
        "trusted_sg_class_percent": percent(stats.get("use_sg", 0), class_decisions),
        "soft_candidate_class_percent": percent(
            stats.get("soft_candidate_classes", stats.get("use_sg", 0)),
            class_decisions,
        ),
        "soft_candidate_cluster_percent": percent(
            stats.get("soft_candidate_clusters", 0),
            stats.get("soft_total_clusters", 0),
        ),
        "raw_supported_candidate_class_percent": percent(
            stats.get("raw_supported_candidate_classes", 0),
            class_decisions,
        ),
        "raw_supported_candidate_cluster_percent": percent(
            stats.get("raw_supported_candidate_clusters", 0),
            stats.get("soft_total_clusters", 0),
        ),
        "min_seed_overlap": min_seed_overlap,
        "accepted_seed_overlap_cluster_percent": percent(
            stats.get("seed_overlap_candidate_clusters", 0),
            stats.get("seed_overlap_total_clusters", 0),
        ),
        "avg_cluster_raw_topk_support": avg_cluster_raw_topk_support,
        "trusted_sg_pixel_overwrite_percent": percent(sg_overwrite_pixels, dense_total_pixels),
        "soft_prior_pixel_percent": percent(soft_prior_pixels, dense_total_pixels),
        "random_prior_density_percent": percent(random_prior_density_pixels, dense_total_pixels),
        "raw_topk_prior_density_percent": percent(raw_topk_prior_density_pixels, dense_total_pixels),
        "boosted_pixel_percent": percent(boosted_pixels, dense_total_pixels),
        "uncertain_pixel_percent": percent(uncertain_pixels, dense_total_pixels),
        "max_abs_diff_before_pamr": stats.get("max_abs_diff_before_pamr", 0.0),
        "changed_pixel_percent_vs_raw": percent(changed_pixels_vs_raw, dense_total_pixels),
        "changed_vs_raw_percent": percent(changed_pixels_vs_raw, dense_total_pixels),
        "raw_preserved_pixel_percent": percent(raw_preserved_pixels, dense_total_pixels),
        "sg_overwrite_pixel_percent": percent(sg_overwrite_pixels, dense_total_pixels),
        "avg_raw_area_pixels": (
            0.0 if class_decisions == 0 else stats.get("raw_area_sum", 0.0) / class_decisions
        ),
        "avg_sg_area_pixels": (
            0.0 if class_decisions == 0 else stats.get("sg_area_sum", 0.0) / class_decisions
        ),
        "avg_sg_to_raw_area_ratio": (
            0.0
            if stats.get("sg_to_raw_ratio_count", 0) == 0
            else stats.get("sg_to_raw_ratio_sum", 0.0) / stats["sg_to_raw_ratio_count"]
        ),
        "avg_raw_sg_agreement_iou": avg_raw_sg_agreement_iou,
        "min_agreement_iou_threshold": min_agreement_iou_threshold,
        "min_raw_area_pixels": min_raw_area_pixels,
        "allow_sg_when_raw_empty": allow_sg_when_raw_empty,
        "min_area_ratio": fallback_cfg.get("min_area_ratio", 0.60),
        "max_area_ratio": fallback_cfg.get("max_area_ratio", 1.60),
        "agreement_iou_warning": (
            avg_raw_sg_agreement_iou < min_agreement_iou_threshold
            and low_agreement_iou_percent == 0
        ),
        "alignment_audit_enabled": audit_enabled,
        "avg_seed_inside_sg_patch_rate": (
            0.0
            if seed_inside_count == 0
            else stats.get("alignment_seed_inside_sg_patch_rate_sum", 0.0)
            / seed_inside_count
        ),
        "seed_hit_sg_percent": percent(
            stats.get("alignment_seed_hit_sg_count", 0),
            seed_inside_count,
        ),
        "avg_raw_patch_seed_iou": mean_value(raw_patch_seed_iou_values),
        "median_raw_patch_seed_iou": median_value(raw_patch_seed_iou_values),
        "raw_patch_seed_hit_percent": percent(
            stats.get("alignment_raw_patch_seed_hit_count", 0),
            raw_patch_seed_count,
        ),
        "avg_sg_roundtrip_iou": mean_value(sg_roundtrip_iou_values),
        "median_sg_roundtrip_iou": median_value(sg_roundtrip_iou_values),
        "num_image_class_mask_decisions": stats.get(
            "alignment_num_image_class_mask_decisions",
            0,
        ),
        "avg_raw_sg_iou_pre_fallback": mean_value(raw_sg_iou_pre_values),
        "median_raw_sg_iou_pre_fallback": median_value(raw_sg_iou_pre_values),
        "avg_raw_final_iou_post_fallback": mean_value(raw_final_iou_post_values),
        "median_raw_final_iou_post_fallback": median_value(raw_final_iou_post_values),
        "count_score_map_interpolated_to_dino_grid": stats.get(
            "count_score_map_interpolated_to_dino_grid",
            0,
        ),
        "count_score_map_already_matching_dino_grid": stats.get(
            "count_score_map_already_matching_dino_grid",
            0,
        ),
        "count_shape_mismatch_after_interpolation": stats.get(
            "count_shape_mismatch_after_interpolation",
            0,
        ),
        "count_raw_mask_size_mismatch": stats.get("count_raw_mask_size_mismatch", 0),
        "count_sg_mask_size_mismatch": stats.get("count_sg_mask_size_mismatch", 0),
        "avg_clusters_per_image": (
            0.0 if num_images == 0 else stats.get("num_clusters", 0) / num_images
        ),
    }
    if debug_force_raw:
        for key in [
            "fallback_used_count",
            "low_agreement_iou_count",
            "raw_too_small_block_sg_count",
            "fallback_used_percent",
            "fallback_sg_too_small_percent",
            "fallback_sg_too_large_percent",
            "fallback_sg_empty_percent",
            "fallback_low_agreement_iou_percent",
            "raw_too_small_block_sg_percent",
            "fallback_raw_empty_no_anchor_percent",
            "use_sg_percent",
            "trusted_sg_class_percent",
            "trusted_sg_pixel_overwrite_percent",
            "soft_prior_pixel_percent",
            "boosted_pixel_percent",
            "uncertain_pixel_percent",
            "changed_pixel_percent_vs_raw",
            "changed_vs_raw_percent",
            "raw_preserved_pixel_percent",
            "sg_overwrite_pixel_percent",
            "lambda_boost",
            "topk_guard",
            "margin_guard",
            "guard_mode",
            "uncertainty_margin",
            "only_trusted_clusters",
            "normalize_prior",
            "min_soft_gate_score",
            "require_semantic_anchor",
            "raw_support_topk",
            "min_cluster_raw_topk_support",
            "soft_candidate_class_percent",
            "soft_candidate_cluster_percent",
            "raw_supported_candidate_class_percent",
            "raw_supported_candidate_cluster_percent",
            "min_seed_overlap",
            "accepted_seed_overlap_cluster_percent",
            "avg_cluster_raw_topk_support",
            "score_scale_type",
            "avg_sg_to_raw_area_ratio",
            "avg_raw_sg_agreement_iou",
            "agreement_iou_warning",
            "alignment_audit_enabled",
            "avg_seed_inside_sg_patch_rate",
            "seed_hit_sg_percent",
            "avg_raw_patch_seed_iou",
            "median_raw_patch_seed_iou",
            "raw_patch_seed_hit_percent",
            "avg_sg_roundtrip_iou",
            "median_sg_roundtrip_iou",
            "num_image_class_mask_decisions",
            "avg_raw_sg_iou_pre_fallback",
            "median_raw_sg_iou_pre_fallback",
            "avg_raw_final_iou_post_fallback",
            "median_raw_final_iou_post_fallback",
            "avg_clusters_per_image",
            "SG_no_pamr_best_mIoU",
            "delta_vs_SG_no_pamr_best",
        ]:
            summary[key] = "not_available"
        summary.update({
            "raw_parity_evaluate_pamr": bool(config.evaluate.get("pamr", False)),
            "raw_parity_test_cfg_mode": config.evaluate.get("mode", "from_dataset"),
            "raw_parity_test_cfg_stride": config.evaluate.get("stride", "from_dataset"),
            "raw_parity_test_cfg_crop_size": config.evaluate.get("crop_size", "from_dataset"),
            "raw_parity_with_bg": "from_dataset",
            "raw_parity_bg_thresh": config.evaluate.get("bg_thresh", "not_available"),
            "raw_parity_num_classes": stats.get("num_classes", "official_e0_branch"),
            "raw_parity_ignore_index": "from_dataset",
            "raw_parity_class_mapping_length": "from_dataset",
        })
    return summary


def train(cfg, args):
    if device == "cuda":
        dist.barrier()

    # build datasets
    # dataset_train, data_loader_train = build_loader(cfg.data) # TODO: Ripristinate something like this
    # ___________________________________________
    # TODO
    # ___________________________________________
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    import clip
    import sys
    sys.path.append("src")
    from src.dataset import COCOCaptions

    image_transforms = T.Compose([
        T.Resize(448, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_ann_path = cfg.data.get("train_ann_path", "coco/train.json")
    train_image_dir = cfg.data.get("train_image_dir", "coco/train2014")
    if cfg.evaluate.eval_only or args.extract_features:
        data_loader_train = None
    else:
        if not os.path.exists(train_ann_path):
            raise FileNotFoundError(
                f"Training annotation file not found: {train_ann_path}. "
                "Set data.train_ann_path to the COCO captions annotation .pt file."
            )
        if not os.path.isdir(train_image_dir):
            raise FileNotFoundError(
                f"Training image directory not found: {train_image_dir}. "
                "Set data.train_image_dir to the COCO train image directory."
            )
        dataset_train = COCOCaptions(
            train_ann_path,
            train_image_dir,
            "train",
            image_transforms,
            clip.tokenize,
        )
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True,
        )
    # ___________________________________________
    # End TODO
    # ___________________________________________

    # build validation loaders
    val_loaders = {}
    for key in cfg.evaluate.task:
        if key == "cls":
            continue

        dataset = build_seg_dataset(cfg.evaluate.get(key))
        len_dataset = len(dataset)

        first_sample = args.job_id * len_dataset // args.num_jobs
        last_sample = ((args.job_id + 1) * len_dataset // args.num_jobs)
        if args.job_id == args.num_jobs - 1:
            last_sample = len_dataset

        dataset = Subset(dataset, range(first_sample, last_sample))
        loader = build_seg_dataloader(
            dataset,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        val_loaders[key] = loader

    logger = get_logger()

    # build model & optimizer
    logger.info(f"Creating model:{cfg.model.type}/{cfg.model_name}")
    model = build_model(cfg.model)
    if device == "cuda":
        model.cuda()

        # model.set_train(decoder_only=(cfg.train.ust_steps > 0), config=cfg)
        # optimizer = build_optimizer(cfg.train, model)
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=cfg.train.base_lr) # TODO: Ripristinate
        if dist.get_world_size() > 1:
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            if not hasattr(model, "_use_replicated_tensor_module"):
                model._use_replicated_tensor_module = False

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters} ({n_parameters/1000/1000:.1f}M)")
    log_xattn_bridge_startup(cfg, model, logger)
    xattn_out_norms = get_xattn_out_proj_norms(model)
    if xattn_out_norms is not None:
        logger.info(
            "XAttn out_proj norm after model init: "
            f"weight={xattn_out_norms['weight']:.8f}, "
            f"bias={xattn_out_norms['bias']:.8f}"
        )
    lr_scheduler = build_scheduler(cfg.train, optimizer)

    # fp16 compression
    logger.info(us.dist_info())
    if cfg.train.fp16 and cfg.train.fp16_comm:
        if int(os.getenv("LOCAL_WORLD_SIZE")) < int(os.getenv("WORLD_SIZE")):
            pg = _get_default_group()
            logger.info("!!! Multi-node setting :: turn on fp16 compression hook")
            model.register_comm_hook(pg, fp16_compress_hook)
        else:
            logger.info("!!! Single-node setting :: skip fp16 compression hook")

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.fp16)

    if (
        bool(cfg.train.get("auto_resume", False))
        and not cfg.checkpoint.resume
        and not cfg.evaluate.eval_only
    ):
        for candidate in ("checkpoint_last.pth", "checkpoint.pth"):
            candidate_path = os.path.join(cfg.output, candidate)
            if os.path.isfile(candidate_path):
                with read_write(cfg):
                    cfg.checkpoint.resume = candidate_path
                logger.info(
                    f"Auto-resume found {candidate_path}; continuing from saved state"
                )
                break

    checkpoint_loaded = False
    checkpoint_path_loaded = None
    if cfg.checkpoint.resume:
        model_to_load = model.module if hasattr(model, "module") else model
        load_checkpoint(cfg, model_to_load, optimizer, lr_scheduler, scaler)
        checkpoint_loaded = True
        checkpoint_path_loaded = str(cfg.checkpoint.resume)
    logger.info(
        f"Loaded checkpoint: {checkpoint_path_loaded if checkpoint_loaded else 'None'}"
    )
    if bool(cfg.get("xattn_bridge", {}).get("debug_tensor_parity", False)) and checkpoint_loaded:
        raise AssertionError(
            "xattn_bridge.debug_tensor_parity=true requires checkpoint.resume disabled, "
            f"but loaded {checkpoint_path_loaded}"
        )
    set_xattn_checkpoint_status(model, checkpoint_loaded, checkpoint_path_loaded)
    if cfg.get("xattn_bridge", {}).get("enabled", False):
        with read_write(cfg):
            cfg.xattn_bridge.checkpoint_loaded = checkpoint_loaded
            cfg.xattn_bridge.checkpoint_path = checkpoint_path_loaded
            cfg.model.xattn_bridge = cfg.xattn_bridge
    xattn_out_norms = get_xattn_out_proj_norms(model)
    if xattn_out_norms is not None:
        logger.info(
            "XAttn out_proj norm after checkpoint loading: "
            f"weight={xattn_out_norms['weight']:.8f}, "
            f"bias={xattn_out_norms['bias']:.8f}"
        )

    if args.extract_features:
        if device == "cpu" or dist.get_rank() == 0:
            os.makedirs(cfg.output, exist_ok=True)
            feature_summary = {
                "method": "Talk2DINO_XAttnBridge",
                "feature_source": "online_frozen_clip_dino",
                "reuses_original_extraction_pipeline": True,
                "message": (
                    "No separate feature extraction is required for the current "
                    "lightweight XAttnBridge implementation."
                ),
            }
            path = os.path.join(cfg.output, "feature_extraction_summary.json")
            with open(path, "w") as f:
                json.dump(feature_summary, f, indent=2)
            logger.info("Reusing original Talk2DINO extracted features")
            logger.info(f"Feature extraction summary saved to {path}")
        return

    if cfg.evaluate.eval_only:
        res = evaluate(cfg, model, val_loaders)
        metrics = res.pop("metrics", None)
        sg_summary = res.pop("sg_summary", None)
        # log res on wandb as statics
        if cfg.wandb and metrics:
            import wandb
            wandb.init(
                project="open-vocab-metrics",
                name=args.wandb_name,
                dir=cfg.output,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume=False,
            )
            wandb.log(metrics[0])
        if sg_summary is not None:
            if device == "cpu" or dist.get_rank() == 0:
                save_sg_summary(cfg.output, sg_summary, logger)
        else:
            log_results(res['val/avg_miou'], cfg['model']['proj_name'], cfg['evaluate']['task'][0], 'segmentation_results', logger)
        if device == "cpu" or dist.get_rank() == 0:
            log_eval_summary(cfg, res, sg_summary, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    do_training(cfg, model, data_loader_train, optimizer, lr_scheduler, scaler, val_loaders)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    if device == "cuda":
        dist.barrier()


def do_training(config, model, data_loader, optimizer, lr_scheduler, scaler, val_loaders):
    logger = get_logger()
    dist.barrier()
    model.train()
    optimizer.zero_grad()
    if config.wandb and dist.get_rank() == 0:
        import wandb
    else:
        wandb = None

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)

    total_steps = config.train.total_steps
    org_total_steps = total_steps
    debug_one_step = bool(config.train.get("debug_one_step", False))
    train_max_steps = config.train.get("max_steps", None)
    if train_max_steps is not None:
        total_steps = min(int(total_steps), int(train_max_steps))
    eval_during_training = bool(config.evaluate.get("eval_during_training", True))
    # update training steps by evaluation step (discard non-evaluation steps)
    if eval_during_training:
        total_steps = total_steps - (total_steps % config.evaluate.eval_freq) + 1
    if org_total_steps != total_steps:
        logger.info(f"Total step is updated: {org_total_steps} -> {total_steps}")
    if not eval_during_training:
        logger.info("Train-time evaluation disabled; checkpoints will use training loss only.")
    if debug_one_step:
        logger.info(f"XAttn one-step gradient debug enabled; max_steps={total_steps}")
        
    ckpt_manager = CheckpointManager(config.checkpoint.save_topk, config.output)

    batch_size = config.data.batch_size
    accum_freq = config.train.accum_freq

    if accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    # ust_check = True
    end = time.time()
    train_start_time = time.time()
    runtime_warned = False
    xattn_ratio_warned = False
    best_miou = -float("inf")
    for step, samples in enumerate(cyclize(data_loader), config.train.start_step):
        if step >= total_steps:
            break
        # if ust_check and config.train.ust_steps and step >= config.train.ust_steps:
        #     model.module.set_train(decoder_only=False, config=config)
        #     logger.info(f" -- [{step}] UST stage is DONE; Now fine-tuning stage begins ...")
        #     ust_check = False

        # caption = samples.pop("org_caption")

        optimizer.zero_grad()
        samples = move_training_batch_to_device(samples, device)
        with torch.amp.autocast("cuda", enabled=config.train.fp16):
            # losses = model(**samples)
            img_emb, txt_emb = model(image=samples["image"], text=samples["annotation"])
            losses = model.compute_loss(img_emb, txt_emb)

        loss, log_vars = parse_losses(losses)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if debug_one_step:
            xattn_diag = get_xattn_diagnostics(model)
            lr = optimizer.param_groups[0]["lr"]
            log_xattn_gradient_debug(config, model, loss, lr, xattn_diag, logger)
        # if config.train.clip_grad:
        #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
        # else:
        #     grad_norm = get_grad_norm(model.parameters())

        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), batch_size)
        for loss_name in log_vars:
            log_vars_meters[loss_name].update(log_vars[loss_name].item(), batch_size)
        # norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if step % config.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            xattn_diag = get_xattn_diagnostics(model)
            print_train_progress(
                step + 1,
                total_steps,
                loss_meter,
                lr,
                train_start_time,
                xattn_diag=xattn_diag,
            )
            ratio = float(xattn_diag.get("correction_base_ratio", 0.0))
            ratio_warn = float(config.get("xattn_bridge", {}).get("correction_ratio_warn", 0.20))
            if ratio > ratio_warn and not xattn_ratio_warned:
                xattn_ratio_warned = True
                logger.warning(
                    f"XAttn correction/base ratio {ratio:.3f} exceeds "
                    f"xattn_bridge.correction_ratio_warn={ratio_warn:.3f}. "
                    "CLIP semantics may be drifting."
                )
            ratio_stop = config.get("xattn_bridge", {}).get("correction_ratio_stop", None)
            if ratio_stop is not None and ratio > float(ratio_stop):
                if us.is_global_zero():
                    print("", flush=True)
                    save_checkpoint(
                        config=config,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        filename="checkpoint_last.pth",
                    )
                    logger.warning(
                        f"Stopping training because XAttn correction/base ratio "
                        f"{ratio:.3f} exceeded correction_ratio_stop={float(ratio_stop):.3f}. "
                        "checkpoint_last.pth saved."
                    )
                break
            elapsed_hours = (time.time() - train_start_time) / 3600.0
            projected_hours = elapsed_hours / max(1, step + 1) * total_steps
            if (
                not runtime_warned
                and projected_hours > float(config.train.get("max_train_hours", 8))
            ):
                runtime_warned = True
                logger.warning(
                    f"Projected training time {projected_hours:.2f}h exceeds "
                    f"train.max_train_hours={config.train.get('max_train_hours', 8)}h"
                )
            if elapsed_hours > float(config.train.get("hard_stop_hours", 10)):
                if us.is_global_zero():
                    print("", flush=True)
                    save_checkpoint(
                        config=config,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        filename="checkpoint_last.pth",
                    )
                    logger.warning(
                        f"Hard stop after {elapsed_hours:.2f}h; checkpoint_last.pth saved."
                    )
                break

            if wandb is not None:
                log_stat = {f"iter/train_{n}": m.val for n, m in log_vars_meters.items()}
                log_stat["iter/train_total_loss"] = loss_meter.val
                # log_stat["iter/grad_norm"] = norm_meter.val
                log_stat["iter/learning_rate"] = lr
                log_stat["iter/epoch"] = epoch
                log_stat["iter/grad_scale"] = scaler.get_scale()

                # image & mask logging
                # if "mask" in losses and step % 500 == 0:
                #     N = 3

                #     # un-normalize image
                #     org_img = us.unnorm(samples["image"][:N])
                #     org_img = torch.clamp(org_img, 0.0, 1.0)  # random erasing makes out-of-range value
                #     mask = losses["mask"][:N].repeat(1, 3, 1, 1).cpu().float()
                #     mask = F.interpolate(mask, org_img.shape[2:]) > 0.5
                #     log_images = [org_img, mask, org_img * mask]
                #     if "neg_mask" in losses:
                #         neg_mask = losses["neg_mask"][:N, :1].repeat(1, 3, 1, 1).cpu().float()
                #         neg_mask = F.interpolate(neg_mask, org_img.shape[2:]) > 0.5
                #         log_images.append(neg_mask)

                #     log_images = torch.cat(log_images)
                #     grid = make_grid(log_images, nrow=N, value_range=(0, 1))
                #     cap = "\n".join([f"[{i}] {c}" for i, c in enumerate(caption[:N])])
                #     log_stat["examples"] = wandb.Image(grid, caption=cap)

                wandb.log(log_stat, step=step)

        save_every_steps = int(config.train.get("save_every_steps", 0) or 0)
        if save_every_steps and step and step % save_every_steps == 0 and us.is_global_zero():
            save_checkpoint(
                config=config,
                step=step,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                metrics={"train/loss": loss_meter.val},
                filename="checkpoint_last.pth",
            )

        if eval_during_training and step and step % config.evaluate.eval_freq == 0:
            metrics = evaluate(config, model, val_loaders)

            if us.is_global_zero():
                ckpt_kwargs = {
                    "config": config,
                    "step": step,
                    "model": model,
                    "optimizer": optimizer,
                    "lr_scheduler": lr_scheduler,
                    "scaler": scaler,
                    "metrics": metrics,
                }
                save_checkpoint(**ckpt_kwargs)
                save_checkpoint(**ckpt_kwargs, filename="checkpoint_last.pth")
                if config.checkpoint.save_all:
                    save_checkpoint(**ckpt_kwargs, filename=f"ckpt_{step}.pth")
                # save best
                miou = metrics["val/avg_miou"]
                if miou > best_miou:
                    best_miou = miou
                    save_checkpoint(**ckpt_kwargs, filename="checkpoint_best.pth")
                if config.checkpoint.save_topk:
                    ckpt_manager.add(miou, ckpt_kwargs, step)

            dist.barrier()
            if us.is_global_zero():
                print("", flush=True)

            if wandb is not None:
                wandb.log(metrics, step=step)

            batch_time.reset()
            loss_meter.reset()
            norm_meter.reset()
            for m in log_vars_meters.values():
                m.reset()

    if us.is_global_zero():
        save_checkpoint(
            config=config,
            step=step,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            metrics={"train/loss": loss_meter.val},
            filename="checkpoint_last.pth",
        )

    if us.is_global_zero():
        print("", flush=True)


@torch.no_grad()
def evaluate(cfg, model, val_loaders):
    logger = get_logger()
    ret = {}
    model.eval()

    for key, loader in val_loaders.items():
        if key == "cls":
            continue

        dataset_class = loader.dataset.__class__.__name__
        logger.info(
            f"Evaluating {key} ({dataset_class}, {len(loader.dataset)} images)"
        )

        miou, metrics, sg_metrics = validate_seg(
            cfg,
            cfg.evaluate.get(key),
            loader,
            model,
        )

        ret[f"val/{key}_miou"] = miou
        ret[f"metrics"] = metrics
        if sg_metrics is not None:
            ret["sg_summary"] = {
                "dataset": key,
                "num_images": len(loader.dataset),
                "images_evaluated": len(loader.dataset),
                "job_id": int(cfg.get("eval_job_id", 0)),
                "num_jobs": int(cfg.get("eval_num_jobs", 1)),
                **sg_metrics,
            }

    ret["val/avg_miou"] = np.mean([
        value
        for key, value in ret.items()
        if key.startswith("val/") and key.endswith("_miou")
    ])

    model.train()

    return ret


@torch.no_grad()
def validate_seg(config, seg_config, data_loader, model):
    logger = get_logger()
    if device == "cuda":
        dist.barrier()

    model.eval()

    if hasattr(model, "module"):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    seg_model = build_dinotext_seg_inference(
        model_without_ddp,
        data_loader.dataset,
        config,
        seg_config,
    )
    seg_model.to(device)

    if device == "cuda" and dist.get_world_size() > 1:
        mmddp_model = MMDistributedDataParallel(
            seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
        if not hasattr(mmddp_model, "_use_replicated_tensor_module"):
            mmddp_model._use_replicated_tensor_module = False
    else:
        mmddp_model = seg_model
    mmddp_model.eval()

    # TODO: Use multi-gpu-test from mmseg instead of ours
    results, pred_qualitatives, gt_qualitatives, num_classes = us.multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=device == "cuda",
        efficient_test=False,
        pre_eval=True,
        format_only=False,
        show_progress=bool(config.evaluate.get("show_progress", True)),
        progress_log_interval=int(config.evaluate.get("progress_log_interval", 0)),
        diagnostic_ignore_eval=bool(config.sg_gate.get("diagnostic_ignore_eval", True))
        if bool(config.get("sg_gate", {}).get("enabled", False))
        else True,
    )
    if device == "cpu" or dist.get_rank() == 0:
        bridge = getattr(model_without_ddp, "xattn_bridge", None)
        if bridge is not None:
            logger.info(
                "XAttnBridge eval forward call count: "
                f"{getattr(bridge, 'forward_call_count', 0)}"
            )
            logger.info(
                "XAttnBridge eval base source: "
                f"{getattr(bridge, 'last_diagnostics', {}).get('base_source', 'not_available')}"
            )

    if device == "cpu" or dist.get_rank() == 0:
        sg_enabled = bool(config.get("sg_gate", {}).get("enabled", False))
        debug_force_raw = bool(
            config.get("sg_gate", {}).get("debug_force_raw_output", False)
        )
        if sg_enabled and not debug_force_raw:
            strict_results = [result["strict"] for result in results]
            diagnostic_results = [result["diagnostic"] for result in results]
            strict_metric = data_loader.dataset.dataset.evaluate(
                strict_results,
                metric="mIoU",
                logger="silent",
            )
            path_a_metric = None
            path_b_metric = None
            if (
                results
                and "pamr_path_a" in results[0]
                and "pamr_path_b" in results[0]
            ):
                path_a_results = [result["pamr_path_a"] for result in results]
                path_b_results = [result["pamr_path_b"] for result in results]
                path_a_metric = data_loader.dataset.dataset.evaluate(
                    path_a_results,
                    metric="mIoU",
                    logger="silent",
                )
                path_b_metric = data_loader.dataset.dataset.evaluate(
                    path_b_results,
                    metric="mIoU",
                    logger="silent",
                )
            diagnostic_metric = None
            if config.sg_gate.get("diagnostic_ignore_eval", True):
                diagnostic_metric = data_loader.dataset.dataset.evaluate(
                    diagnostic_results,
                    metric="mIoU",
                    logger="silent",
                )
            metric = [{
                "strict": strict_metric,
                "diagnostic": diagnostic_metric,
                "pamr_path_a": path_a_metric,
                "pamr_path_b": path_b_metric,
                "ignore_pixels": sum(result["ignore_pixels"] for result in results),
                "total_pixels": sum(result["total_pixels"] for result in results),
                "positive_pixels": sum(result.get("positive_pixels", 0) for result in results),
                "ignore_pixels_after_compile": sum(
                    result.get("ignore_pixels_after_compile", result["ignore_pixels"])
                    for result in results
                ),
                "stats": {},
            }]
            for result in results:
                for stat_key, stat_value in result.get("stats", {}).items():
                    if isinstance(stat_value, list):
                        metric[0]["stats"].setdefault(stat_key, [])
                        metric[0]["stats"][stat_key].extend(stat_value)
                    elif isinstance(stat_value, (str, bool)):
                        metric[0]["stats"].setdefault(stat_key, stat_value)
                    elif stat_key in {
                        "max_abs_diff_before_pamr",
                        "max_abs_diff_raw_vs_path_a_input",
                        "max_abs_diff_raw_vs_path_b_input",
                        "max_abs_diff_path_a_input_vs_path_b_input",
                        "max_abs_diff_path_a_pamr_output_vs_path_b_pamr_output",
                    }:
                        metric[0]["stats"][stat_key] = max(
                            float(metric[0]["stats"].get(stat_key, 0.0)),
                            float(stat_value),
                        )
                    else:
                        metric[0]["stats"][stat_key] = (
                            metric[0]["stats"].get(stat_key, 0) + stat_value
                        )
        else:
            metric = [data_loader.dataset.dataset.evaluate(
                results,
                metric="mIoU",
                logger="silent",
            )]
    else:
        metric = [None]

    if device == "cuda":
        dist.broadcast_object_list(metric)
    sg_metrics = None
    if bool(config.get("sg_gate", {}).get("enabled", False)):
        debug_force_raw = bool(
            config.get("sg_gate", {}).get("debug_force_raw_output", False)
        )
        if debug_force_raw:
            strict_metric = metric[0]
            diagnostic_metric = None
            metric_for_summary = {
                "stats": {},
                "ignore_pixels": 0,
                "total_pixels": 0,
                "positive_pixels": 0,
                "ignore_pixels_after_compile": 0,
            }
        else:
            strict_metric = metric[0]["strict"]
            diagnostic_metric = metric[0]["diagnostic"]
            metric_for_summary = metric[0]
        miou_result = strict_metric["mIoU"] * 100
        sg_metrics = summarize_sg_stats(
            config,
            strict_metric,
            diagnostic_metric,
            metric_for_summary,
            len(data_loader.dataset),
        )
        metric = [strict_metric]
    else:
        miou_result = metric[0]["mIoU"] * 100

    torch.cuda.empty_cache()
    if device == "cuda":
        dist.barrier()
    return miou_result, metric, sg_metrics


def main():
    parser = get_argparser()
    args = parser.parse_args()

    if args.eval:
        # update config when resume
        # default config -> org config -> eval config
        default_cfg = load_config(args.eval_cfg)
        # default_cfg = load_config("configs/HOME.yml")
        # default_cfg = load_config("configs/freeda.yml")
        # org_cfg_path = Path(args.resume).parent / "config.json"
        # if org_cfg_path.exists():
        #     org_cfg = OmegaConf.load(Path(args.resume).parent / "config.json")
        # else:
        org_cfg = OmegaConf.create()  # empty container
        eval_cfg = OmegaConf.load(args.eval_base_cfg)
        cfg = OmegaConf.merge(default_cfg, org_cfg, eval_cfg)
        if args.opts is not None:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))

        cfg.wandb = args.wandb
        cfg.evaluate.eval_only = args.eval
        args.output = args.output if args.output is not None else "output/eval"

        assert args.output is not None, "Please specify output folder for evaluation"
        cfg.output = args.output
        cfg.eval_job_id = args.job_id
        cfg.eval_num_jobs = args.num_jobs

        # create output folder if it does not exist
        Path(cfg.output).mkdir(parents=True, exist_ok=True)
    else:
        cfg = get_config(args)


    # TODO: The config can be modified here
    if isinstance(cfg.get("checkpoint", None), str):
        checkpoint_path = cfg.get("checkpoint")
        with read_write(cfg):
            cfg.checkpoint = {
                "resume": checkpoint_path,
                "save_topk": 0,
                "save_all": False,
            }
    if cfg.get("xattn_bridge", {}).get("enabled", False):
        with read_write(cfg):
            cfg.model.xattn_bridge = cfg.xattn_bridge
    if cfg.get("postprocess", {}).get("mode", "none") == "sg_soft_prior":
        with read_write(cfg):
            cfg.sg_gate.enabled = True
            cfg.sg_gate.soft_reweight.enabled = True
            cfg.sg_gate.soft_reweight.disable_hard_overwrite = True
            cfg.sg_gate.soft_reweight.candidate_source = "trusted_clusters"
            cfg.sg_gate.soft_reweight.apply_pamr_after_soft = False

    if device == "cuda":

        # start faster ref: https://github.com/open-mmlab/mmdetection/pull/7036
        mp.set_start_method("fork", force=True)
        init_dist("pytorch")
        rank, world_size = get_dist_info()
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

        dist.barrier()

    else:
        rank = 0
        world_size = 1

    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = cfg.train.base_lr * cfg.data.batch_size * world_size / 4096.0
    # linear_scaled_min_lr = cfg.train.min_lr * cfg.data.batch_size * world_size / 4096.0

    # with read_write(cfg):
    #     logger.info(f"Scale base_lr from {cfg.train.base_lr} to {linear_scaled_lr}")
    #     logger.info(f"Scale min_lr from {cfg.train.min_lr} to {linear_scaled_min_lr}")
    #     cfg.train.base_lr = linear_scaled_lr
    #     cfg.train.min_lr = linear_scaled_min_lr

    if device == "cuda" and dist.get_rank() == 0:
        path = os.path.join(cfg.output, "config.json")
        OmegaConf.save(cfg, path)
        logger.info(f"Full config saved to {path}")

    log_startup_details = bool(cfg.evaluate.get("log_startup_details", True))
    if log_startup_details:
        env_info_dict = collect_env()
        env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
        dash_line = "-" * 60 + "\n"
        logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

        logger.info(f"Git hash: {get_git_hash(digits=7)}")

        # print config
        logger.info(OmegaConf.to_yaml(cfg))
    else:
        mode = "SG-Gate" if cfg.get("sg_gate", {}).get("enabled", False) else "E0"
        tasks = ", ".join(cfg.evaluate.task)
        logger.info(
            f"Startup summary: mode={mode}, tasks={tasks}, "
            f"output={cfg.output}"
        )

    train(cfg, args)
    if device == "cuda":
        dist.barrier()

    # print outputdir
    logger.info(f"Experiment dir: {cfg.output}")


if __name__ == "__main__":
    main()
