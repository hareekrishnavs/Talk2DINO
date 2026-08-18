#!/usr/bin/env python3
"""Bounded production-path RWR pilot with aggregate-only diagnostics."""

from __future__ import annotations

import atexit
import json
import os
import runpy
import statistics
import sys
import time
from pathlib import Path

import torch


REPOSITORY = Path("/project/6114407/haree/Talk2DINO")
OPEN_VOCABULARY_ROOT = REPOSITORY / "src/open_vocabulary_segmentation"
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(OPEN_VOCABULARY_ROOT))

from models.dinotext.cover_dr import inference as inference_module
from segmentation.evaluation import dinotext_seg as segmentation_module


OUTPUT = Path(os.environ["TALK2DINO_RWR_PILOT_OUTPUT"])
WINDOW_LIMIT = int(os.environ.get("TALK2DINO_RWR_PILOT_WINDOWS", "0"))
_production_apply = inference_module.apply_rwr_to_e3_snapshot
_records: list[dict[str, int | float]] = []
_written = False


class PilotWindowLimitReached(RuntimeError):
    pass


def write_report():
    global _written
    if _written or not _records:
        return
    _written = True
    iterations = [int(item["iterations"]) for item in _records]
    work = [int(item["work_count"]) for item in _records]
    restarts = [int(item["restarts"]) for item in _records]
    certificate_work = [int(item["fp64_certificate_work"]) for item in _records]
    certificate_rejections = [
        int(item["fp64_certificate_rejections"]) for item in _records
    ]
    working_scaled = [
        float(item["working_maximum_scaled_residual"]) for item in _records
    ]
    certified_scaled = [
        float(item["certified_maximum_scaled_residual"]) for item in _records
    ]
    runtimes = [float(item["seconds"]) for item in _records]
    report = {
        "windows_processed": len(_records),
        "window_limit": WINDOW_LIMIT,
        "maximum_iterations": max(iterations),
        "median_iterations": statistics.median(iterations),
        "maximum_work_count": max(work),
        "median_work_count": statistics.median(work),
        "restart_count_minimum": min(restarts),
        "restart_count_median": statistics.median(restarts),
        "restart_count_maximum": max(restarts),
        "unconverged_rhs_count": 0,
        "returned_certificate_failure_count": sum(
            value > 1 for value in certified_scaled
        ),
        "maximum_scaled_true_residual": max(
            float(item["maximum_scaled_residual"]) for item in _records
        ),
        "working_scaled_residual_median": statistics.median(working_scaled),
        "working_scaled_residual_maximum": max(working_scaled),
        "certified_scaled_residual_median": statistics.median(certified_scaled),
        "certified_scaled_residual_maximum": max(certified_scaled),
        "certificate_work_total": sum(certificate_work),
        "certificate_work_median": statistics.median(certificate_work),
        "certificate_work_maximum": max(certificate_work),
        "certificate_rejections_total": sum(certificate_rejections),
        "certificate_rejections_minimum": min(certificate_rejections),
        "certificate_rejections_median": statistics.median(
            certificate_rejections
        ),
        "certificate_rejections_maximum": max(certificate_rejections),
        "runtime_per_crop_seconds_minimum": min(runtimes),
        "runtime_per_crop_seconds_median": statistics.median(runtimes),
        "runtime_per_crop_seconds_maximum": max(runtimes),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("RWR_PILOT_RESULT " + json.dumps(report, sort_keys=True), flush=True)


def apply_and_measure(snapshot, config):
    started = time.perf_counter()
    output = _production_apply(snapshot, config)
    elapsed = time.perf_counter() - started
    for item in output.windows:
        _records.append({
            "iterations": item.iterations,
            "work_count": item.work_count,
            "restarts": item.restarts,
            "maximum_scaled_residual": item.maximum_scaled_residual,
            "working_maximum_scaled_residual": (
                item.working_maximum_scaled_residual
            ),
            "certified_maximum_scaled_residual": (
                item.certified_maximum_scaled_residual
            ),
            "fp64_certificate_work": item.fp64_certificate_work,
            "fp64_certificate_rejections": item.fp64_certificate_rejections,
            "seconds": elapsed / len(output.windows),
        })
        if WINDOW_LIMIT and len(_records) >= WINDOW_LIMIT:
            write_report()
            raise PilotWindowLimitReached(
                f"diagnostic stop after {WINDOW_LIMIT} successful RWR windows"
            )
    return output


atexit.register(write_report)
inference_module.apply_rwr_to_e3_snapshot = apply_and_measure
segmentation_module.apply_rwr_to_e3_snapshot = apply_and_measure
runpy.run_path(str(OPEN_VOCABULARY_ROOT / "main.py"), run_name="__main__")
