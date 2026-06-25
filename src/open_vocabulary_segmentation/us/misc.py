# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
from typing import Dict, List, Any
from datetime import datetime
from itertools import chain
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

# ImageNet mean/std (from timm)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_MEAN = IMAGENET_DEFAULT_MEAN
DEFAULT_STD = IMAGENET_DEFAULT_STD

# NOTE Originally CLIP statistics should be used, but the legacy of ImageNet statistics
# from GroupViT is applied. Fortunately, CLIP is quite robust to slightly different
# normalization constants (https://github.com/openai/CLIP/issues/20#issuecomment-764985771).


def unnorm(x):
    mean = torch.as_tensor(DEFAULT_MEAN, device=x.device)[None, ..., None, None]
    std = torch.as_tensor(DEFAULT_STD, device=x.device)[None, ..., None, None]
    return x.mul(std).add(mean)


# DEBUG NaN
def check_nonfinite(x, name=""):
    rank = dist.get_rank()
    n_nan = x.isnan().sum()
    n_inf = x.isinf().sum()
    if n_nan or n_inf:
        print(f"[RANK {rank}] {name} is not finite: #nan={n_nan}, #inf={n_inf}")
        return True

    print(f"[RANK {rank}] {name} is OK ...")
    return False


def normalize(t, dim, eps=1e-6):
    """Large default eps for fp16"""
    return F.normalize(t, dim=dim, eps=eps)


def timestamp(fmt="%y%m%d-%H%M%S"):
    return datetime.now().strftime(fmt)


def merge_dicts_by_key(dics: List[Dict]) -> Dict[Any, List]:
    """Merge dictionaries by key. All of dicts must have same keys."""
    ret = {key: [] for key in dics[0].keys()}
    for dic in dics:
        for key, value in dic.items():
            ret[key].append(value)

    return ret


def flatten_2d_list(list2d):
    return list(chain.from_iterable(list2d))


def num_params(module):
    return sum(p.numel() for p in module.parameters())


def param_trace(name, module, depth=0, max_depth=999, threshold=0, printf=print):
    if depth > max_depth:
        return
    prefix = "  " * depth
    n_params = num_params(module)
    if n_params > threshold:
        printf("{:60s}\t{:10.3f}M".format(prefix + name, n_params / 1024 / 1024))
    for n, m in module.named_children():
        if depth == 0:
            child_name = n
        else:
            child_name = "{}.{}".format(name, n)
        param_trace(child_name, m, depth + 1, max_depth, threshold, printf)


@torch.no_grad()
def hash_bn(module):
    summary = []
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            w = m.weight.detach().mean().item()
            b = m.bias.detach().mean().item()
            rm = m.running_mean.detach().mean().item()
            rv = m.running_var.detach().mean().item()
            summary.append((w, b, rm, rv))

    if not summary:
        return 0.0, 0.0

    w, b, rm, rv = [np.mean(col) for col in zip(*summary)]
    p = np.mean([w, b])
    s = np.mean([rm, rv])

    return p, s


@torch.no_grad()
def hash_params(module):
    return torch.as_tensor([p.mean() for p in module.parameters()]).mean().item()


@torch.no_grad()
def hashm(module):
    p = hash_params(module)
    _, s = hash_bn(module)

    return p, s

import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmseg.core.evaluation import intersect_and_union

device = "cuda" if torch.cuda.is_available() else "cpu"

from typing import Optional


def np2tmp(array, temp_file_name=None, tmpdir=None):
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix=".npy",
            delete=False,
            dir=tmpdir,
        ).name
    np.save(temp_file_name, array)
    return temp_file_name


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _print_eval_progress(
        processed,
        total,
        start_time,
        width=30,
        final=False,
        timing_avgs=None):
    total = max(1, total)
    fraction = min(1.0, processed / total)
    filled = int(width * fraction)
    bar = "=" * filled + "." * (width - filled)
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 and processed > 0 else 0.0
    eta = (total - processed) / rate if rate > 0 else 0.0
    line = (
        f"\rEval [{bar}] {processed}/{total} "
        f"({100.0 * fraction:5.1f}%) "
        f"elapsed {_format_duration(elapsed)} "
        f"eta {_format_duration(eta)} "
        f"{rate:5.2f} img/s"
    )
    if timing_avgs:
        line += (
            f" | avg50 talk {timing_avgs.get('talk2dino_forward_time', 0.0):.2f}s"
            f" e2 {timing_avgs.get('e2_clustering_time', 0.0):.2f}s"
            f" e3 {timing_avgs.get('e3_gate_time', 0.0):.2f}s"
            f" eval {timing_avgs.get('evaluator_time', 0.0):.2f}s"
        )
    print(line, end="\n" if final else "", flush=True)


def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    # rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    part_file = osp.join(tmpdir, f'part_{rank}.pkl')  # type: ignore
    mmcv.dump(result_part, part_file)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            part_result = mmcv.load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={},
                   show_progress=True,
                   progress_log_interval=0,
                   diagnostic_ignore_eval=True):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    eval_dataset = getattr(dataset, "dataset", dataset)

    def absolute_dataset_index(index):
        if hasattr(dataset, "indices"):
            indices = dataset.indices
            if hasattr(indices, "start"):
                return index + indices.start
            return indices[index]
        return index
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    progress_log_interval = int(progress_log_interval or 0)
    processed = 0
    progress_start_time = time.time()
    timing_window = []
    timing_avgs = None
    if rank == 0 and show_progress:
        prog_bar = mmcv.ProgressBar(len(dataset))
    elif rank == 0 and progress_log_interval > 0:
        _print_eval_progress(0, len(dataset), progress_start_time)
    if rank == 0:
        print("Eval debug: waiting for first dataloader batch...", flush=True)

    pred_qualitatives = []
    gt_qualitatives = []

    for batch_id, (batch_indices, data) in enumerate(zip(loader_indices, data_loader)):
        if rank == 0 and batch_id == 0:
            waited = time.time() - progress_start_time
            print(
                f"Eval debug: first batch loaded after {waited:.1f}s; "
                f"indices={list(batch_indices)}",
                flush=True,
            )
        forward_start = time.time()
        with torch.no_grad():
            if device == 'cpu':
                data['img_metas'] = [e.data[0] for e in data['img_metas']]
            result = model(return_loss=False, rescale=True, **data)
        if rank == 0 and batch_id == 0:
            print(
                f"Eval debug: first model forward finished in "
                f"{time.time() - forward_start:.1f}s",
                flush=True,
            )

        for pred_qualitative, index in zip(result, batch_indices):
            displayed_prediction = (
                pred_qualitative["strict"]
                if isinstance(pred_qualitative, dict)
                else pred_qualitative
            )
            pred_qualitatives.append(displayed_prediction + 1)
            seg_map_gt = eval_dataset.get_gt_seg_map_by_idx(
                absolute_dataset_index(index)
            )
            # seg_map_gt[seg_map_gt == 255] = 0
            gt_qualitatives.append(seg_map_gt)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = eval_dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            absolute_indices = [absolute_dataset_index(i) for i in batch_indices]
            if result and isinstance(result[0], dict):
                sg_results = []
                eval_start = time.perf_counter()
                for prediction, absolute_index in zip(result, absolute_indices):
                    strict_pre_eval = eval_dataset.pre_eval(
                        [prediction["strict"]],
                        indices=[absolute_index],
                    )[0]
                    diagnostic_pre_eval = None
                    if diagnostic_ignore_eval:
                        diagnostic_gt = eval_dataset.get_gt_seg_map_by_idx(
                            absolute_index
                        ).copy()
                        diagnostic_gt[prediction["diagnostic"] == 255] = \
                            eval_dataset.ignore_index
                        diagnostic_pre_eval = intersect_and_union(
                            prediction["diagnostic"],
                            diagnostic_gt,
                            len(eval_dataset.CLASSES),
                            eval_dataset.ignore_index,
                            label_map=dict(),
                            reduce_zero_label=eval_dataset.reduce_zero_label,
                        )
                    sg_results.append({
                        "strict": strict_pre_eval,
                        "diagnostic": diagnostic_pre_eval,
                        "ignore_pixels": prediction["ignore_pixels"],
                        "total_pixels": prediction["total_pixels"],
                        "positive_pixels": prediction.get("positive_pixels", 0),
                        "ignore_pixels_after_compile": prediction.get(
                            "ignore_pixels_after_compile",
                            prediction["ignore_pixels"],
                        ),
                        "timing": prediction.get("timing", {}),
                        "stats": prediction.get("stats", {}),
                    })
                evaluator_time = time.perf_counter() - eval_start
                per_item_evaluator_time = evaluator_time / max(1, len(sg_results))
                for sg_result in sg_results:
                    sg_result["timing"]["evaluator_time"] = per_item_evaluator_time
                result = sg_results
            else:
                result = eval_dataset.pre_eval(
                    result,
                    indices=absolute_indices,
                )

        results.extend(result)
        for item in result:
            if isinstance(item, dict) and "timing" in item:
                timing_window.append(item["timing"])
        if rank == 0 and timing_window and len(timing_window) >= 50:
            keys = (
                "talk2dino_forward_time",
                "e2_clustering_time",
                "e3_gate_time",
                "evaluator_time",
            )
            window = timing_window[-50:]
            timing_avgs = {
                key: sum(float(timing.get(key, 0.0)) for timing in window) / len(window)
                for key in keys
            }

        batch_size = len(result) * world_size
        if rank == 0 and show_progress:
            for _ in range(batch_size):
                prog_bar.update()
        elif rank == 0 and progress_log_interval > 0:
            previous = processed
            processed = min(processed + batch_size, len(dataset))
            if (
                processed == len(dataset)
                or processed // progress_log_interval
                > previous // progress_log_interval
            ):
                _print_eval_progress(
                    processed,
                    len(dataset),
                    progress_start_time,
                    final=processed == len(dataset),
                    timing_avgs=timing_avgs,
                )

    # collect results from all ranks
    if world_size > 1:
        if gpu_collect:
            results = collect_results_gpu(results, len(dataset))
        else:
            results = collect_results_cpu(results, len(dataset), tmpdir)
    return results, pred_qualitatives, gt_qualitatives, len(eval_dataset.CLASSES)
