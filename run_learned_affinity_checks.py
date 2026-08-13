#!/usr/bin/env python3
"""F1f/F1g/F1h checks for the learned affinity metric (src/learned_affinity).
Real GPU data (Part E capture + the E3 cache), read-only throughout: never
writes to, moves, or deletes anything in the capture or cache directories.
No training, no loss -- inference/diagnostic only (F1 scope)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from src.e3_affinity_oracle import (
    AffinityOracleError,
    iter_cache_windows,
    load_cache_manifest,
    load_capture_features,
    load_capture_manifest,
    propagate_scores,
)
from src.learned_affinity import (
    LearnedMetric,
    apply_knn,
    build_differentiable_knn_graph,
    implicit_propagate,
)
from src.learned_affinity.evaluate import evaluate_with_learned_metric
from src.learned_affinity.implicit_solve import (
    LAST_SOLVE_STATS,
    solve_fixed_point,
    solve_fixed_point_richardson,
)

ALPHA_ZERO_ANCHOR = 28.480169315747716
ANCHOR_MIOU = 29.878049
ANCHOR_AACC = 48.528202
ANCHOR_MACC = 54.138821
ANCHOR_TOLERANCE = 5e-3  # per F1f: same tolerance established in Part E1


def assert_identity(capture_dir: Path, cache_dir: Path, device: str) -> None:
    print("=== F1f: identity gate (r forced to exactly 0) ===")
    metric = LearnedMetric()

    torch.manual_seed(0)
    f = F.normalize(torch.randn(200, 768, device=device), dim=-1)
    g = metric.to(device)(f, r_override=0.0)
    check1 = torch.allclose(g, f, atol=1e-6)
    print(f"[1] g(f) == f (atol=1e-6, float32 eps): {check1}  "
          f"(max abs diff={float((g-f).abs().max()):.3e})")
    if not check1:
        raise AffinityOracleError("F1f check 1 FAILED: g(f) != f at r=0")

    from src.e3_affinity_oracle import build_knn_graph
    ref_indices, ref_weights, _zero = build_knn_graph(f, knn_k=12, affinity_power=3.0)
    learned_indices, learned_weights = build_differentiable_knn_graph(g, k=12, kappa=3.0)
    indices_match = torch.equal(learned_indices.cpu(), ref_indices.to(torch.int64).cpu())
    weights_match = torch.equal(
        learned_weights.detach().to(torch.float16).cpu(), ref_weights.cpu()
    )
    print(f"[2] rebuilt graph indices identical to build_knn_graph(f): {indices_match}")
    print(f"[2] rebuilt graph weights identical (post fp16 cast) to build_knn_graph(f): {weights_match}")
    if not (indices_match and weights_match):
        raise AffinityOracleError("F1f check 2 FAILED: graph built from g(f) at r=0 diverges from build_knn_graph(f)")

    print("[3] full val evaluation, alpha=0.98,T=320, graph from g(f) at r=0 ...")
    point98 = evaluate_with_learned_metric(
        capture_dir, cache_dir, metric, 0.98, device=device,
        propagation_steps=320, r_override=0.0,
    )
    deviations = {
        "mIoU": abs(point98["mIoU"] - ANCHOR_MIOU),
        "aAcc": abs(point98["aAcc"] - ANCHOR_AACC),
        "mAcc": abs(point98["mAcc"] - ANCHOR_MACC),
    }
    check3 = all(v <= ANCHOR_TOLERANCE for v in deviations.values())
    print(f"    actual mIoU={point98['mIoU']:.6f} aAcc={point98['aAcc']:.6f} mAcc={point98['mAcc']:.6f} "
          f"evaluated_images={point98['evaluated_images']} runtime={point98['runtime_seconds']:.1f}s")
    print(f"    deviations={deviations} tolerance={ANCHOR_TOLERANCE}: {check3}")
    if not check3:
        raise AffinityOracleError(f"F1f check 3 FAILED: {deviations}")

    print("[4] full val evaluation, alpha=0.00 ...")
    zero = evaluate_with_learned_metric(
        capture_dir, cache_dir, metric, 0.0, device=device,
        propagation_steps=10, r_override=0.0,
    )
    check4 = zero["mIoU"] == ALPHA_ZERO_ANCHOR
    print(f"    actual mIoU={zero['mIoU']!r} expected={ALPHA_ZERO_ANCHOR!r}: {check4}")
    if not check4:
        raise AffinityOracleError(f"F1f check 4 FAILED: {zero['mIoU']!r} != {ALPHA_ZERO_ANCHOR!r}")

    print("F1f: ALL FOUR IDENTITY CHECKS PASSED")


def check_solver_equivalence(cache_dir: Path, device: str, n_windows: int) -> None:
    print(f"=== F1g: solver equivalence, alpha=0.98,T=320, n_windows={n_windows} ===")
    print("Reports THREE comparisons: (1) production CGLS vs propagate_scores(T=320) "
          "-- the one that matters now that CGLS is the production solver; (2) the old "
          "Richardson solver vs propagate_scores(T=320), for continuity with the original "
          "F1g/X7 finding; (3) CGLS vs Richardson, isolating how much the solver SWITCH "
          "itself moved the answer.")
    manifest = load_cache_manifest(cache_dir, verify_shards=False)
    device_value = torch.device(device)
    max_diff_cg_vs_power = 0.0
    max_diff_richardson_vs_power = 0.0
    max_diff_cg_vs_richardson = 0.0
    cg_iters_all = []
    compared = 0
    for row in iter_cache_windows(cache_dir, manifest):
        if compared >= n_windows:
            break
        raw_scores = row["raw_scores"].reshape(manifest["class_count"], -1).to(
            device=device_value, dtype=torch.float32,
        )  # [C,P]
        # propagate_scores validates knn_indices/knn_weights as int16/
        # float16 internally (validate_graph); keep the ORIGINAL dtypes for
        # that call and make separate int64/float32 copies for my own
        # solver's tensor indexing / arithmetic.
        indices16 = row["knn_indices"].to(device=device_value)
        weights16 = row["knn_weights"].to(device=device_value)
        indices = indices16.to(dtype=torch.int64)
        weights = weights16.to(dtype=torch.float32)

        power = propagate_scores(
            raw_scores, indices16, weights16, 0.98, propagation_steps=320,
        )  # [C,P]

        s0_pc = raw_scores.T.contiguous()  # [P,C]
        cg, cg_iters = solve_fixed_point(s0_pc, indices, weights, 0.98)  # shipped default
        richardson, _ = solve_fixed_point_richardson(s0_pc, indices, weights, 0.98, tol=1e-6, max_iter=500)
        cg_cp, richardson_cp = cg.T, richardson.T  # [C,P]

        max_diff_cg_vs_power = max(max_diff_cg_vs_power, (power - cg_cp).abs().max().item())
        max_diff_richardson_vs_power = max(
            max_diff_richardson_vs_power, (power - richardson_cp).abs().max().item()
        )
        max_diff_cg_vs_richardson = max(
            max_diff_cg_vs_richardson, (cg_cp - richardson_cp).abs().max().item()
        )
        cg_iters_all.append(cg_iters)
        compared += 1

    print(f"windows_compared={compared}")
    print(f"cg_forward_iterations: min={min(cg_iters_all)} "
          f"mean={sum(cg_iters_all)/len(cg_iters_all):.1f} max={max(cg_iters_all)} "
          f"(all converged -- solve_fixed_point raises otherwise)")
    print(f"(1) max|CGLS - propagate_scores(T=320)|       = {max_diff_cg_vs_power:.3e}  (bar: < 1e-5)")
    print(f"(2) max|Richardson - propagate_scores(T=320)| = {max_diff_richardson_vs_power:.3e}  (bar: < 1e-5)")
    print(f"(3) max|CGLS - Richardson|                    = {max_diff_cg_vs_richardson:.3e}")
    passed = max_diff_cg_vs_power < 1e-5
    print(f"F1g (production solver vs reference): {'PASSED' if passed else 'FAILED'}")
    if not passed:
        raise AffinityOracleError(f"F1g solver equivalence FAILED: max_diff={max_diff_cg_vs_power}")


def report_timing(capture_dir: Path, cache_dir: Path, device: str, n_windows: int) -> None:
    print(f"=== F1h: timing, n_windows={n_windows} ===")
    device_value = torch.device(device)
    metric = LearnedMetric().to(device_value)
    metric.eval()

    capture_manifest = load_capture_manifest(capture_dir)
    sample_globals = [
        w["global_window_index"]
        for image in capture_manifest["images"][: max(1, n_windows // 2 + 1)]
        for w in image["windows"]
    ][:n_windows]
    features = load_capture_features(
        capture_dir, capture_manifest, device=device_value, needed_indices=set(sample_globals),
    )
    windows = [
        F.normalize(features[g].to(device=device_value, dtype=torch.float32), dim=-1)
        for g in sample_globals
    ]
    print(f"loaded {len(windows)} real captured windows")

    def sync():
        if device_value.type == "cuda":
            torch.cuda.synchronize()

    # warm-up (first call pays CUDA kernel compilation / allocator warm-up)
    g0 = metric(windows[0])
    idx0, w0 = build_differentiable_knn_graph(g0, k=metric.k, kappa=metric.kappa)
    s0 = torch.randn(1024, 171, device=device_value)
    _ = implicit_propagate(s0, idx0, w0.detach(), 0.98)  # shipped default, no overrides
    sync()

    graph_times, forward_times, backward_times = [], [], []
    forward_iters_all, backward_iters_all = [], []
    for f in windows:
        sync()
        t0 = time.perf_counter()
        g = metric(f)
        indices, weights = build_differentiable_knn_graph(g, k=metric.k, kappa=metric.kappa)
        sync()
        graph_times.append(time.perf_counter() - t0)

        s0 = torch.randn(1024, 171, device=device_value)
        weights_leaf = weights.detach().requires_grad_(True)
        sync()
        t0 = time.perf_counter()
        s_star = implicit_propagate(s0, indices, weights_leaf, 0.98)  # shipped default
        sync()
        forward_times.append(time.perf_counter() - t0)
        forward_iters_all.append(LAST_SOLVE_STATS.get("forward_iters"))

        loss = s_star.sum()
        sync()
        t0 = time.perf_counter()
        loss.backward()
        sync()
        backward_times.append(time.perf_counter() - t0)
        backward_iters_all.append(LAST_SOLVE_STATS.get("backward_iters"))

    def stats(name, values):
        mean_ms = sum(values) / len(values) * 1000
        max_ms = max(values) * 1000
        print(f"{name}: mean={mean_ms:.3f}ms  max={max_ms:.3f}ms  over {len(values)} windows")

    stats("graph_construction_from_g(f)", graph_times)
    stats("implicit_forward_solve (CGLS, shipped default tol=1e-10/max_iter=200)", forward_times)
    stats("adjoint_backward_solve (CGLS, shipped default)", backward_times)
    print(f"forward solve iterations at alpha=0.98 (CGLS): "
          f"min={min(forward_iters_all)} mean={sum(forward_iters_all)/len(forward_iters_all):.1f} "
          f"max={max(forward_iters_all)}  (all {len(forward_iters_all)} windows converged -- "
          f"solve_fixed_point raises SolverConvergenceError otherwise)")
    print(f"adjoint solve iterations at alpha=0.98 (CGLS): "
          f"min={min(backward_iters_all)} mean={sum(backward_iters_all)/len(backward_iters_all):.1f} "
          f"max={max(backward_iters_all)}")


def report_peak_memory(device: str) -> None:
    print("=== Peak memory independent of iteration count (100 vs 500 iters) ===")
    device_value = torch.device(device)
    torch.manual_seed(0)
    n, k, c = 1024, 12, 171
    indices = torch.randint(0, n, (n, k), device=device_value)
    weights = F.normalize(torch.rand(n, k, device=device_value), p=1, dim=-1).requires_grad_(True)
    s0 = torch.randn(n, c, device=device_value)

    for max_iter in (100, 500):
        if device_value.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device_value)
            torch.cuda.synchronize()
        # tol=0 forces exactly max_iter iterations (never converges early),
        # isolating iteration-count effect on memory. raise_on_nonconvergence
        # must be disabled for this diagnostic -- forcing non-convergence is
        # the whole point here, not a real failure.
        s_star = implicit_propagate(
            s0, indices, weights, 0.98, tol=0.0, max_iter=max_iter, raise_on_nonconvergence=False,
        )
        loss = s_star.sum()
        loss.backward()
        weights.grad = None
        if device_value.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device_value)
            print(f"max_iter={max_iter}: peak_gpu_bytes={peak} ({peak/1e6:.3f} MB)")
        else:
            print(f"max_iter={max_iter}: (device={device}, no CUDA peak-memory counter available)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assert-identity", action="store_true")
    parser.add_argument("--check-solver-equivalence", action="store_true")
    parser.add_argument("--report-timing", action="store_true")
    parser.add_argument("--report-peak-memory", action="store_true")
    parser.add_argument("--n-windows", type=int, default=50)
    args = parser.parse_args()

    if args.assert_identity:
        if args.capture_dir is None or args.cache is None:
            raise AffinityOracleError("--assert-identity requires --capture-dir and --cache")
        assert_identity(args.capture_dir, args.cache, args.device)
    if args.check_solver_equivalence:
        if args.cache is None:
            raise AffinityOracleError("--check-solver-equivalence requires --cache")
        check_solver_equivalence(args.cache, args.device, args.n_windows)
    if args.report_timing:
        if args.capture_dir is None or args.cache is None:
            raise AffinityOracleError("--report-timing requires --capture-dir and --cache")
        report_timing(args.capture_dir, args.cache, args.device, args.n_windows)
    if args.report_peak_memory:
        report_peak_memory(args.device)
    if not any((
        args.assert_identity, args.check_solver_equivalence,
        args.report_timing, args.report_peak_memory,
    )):
        parser.error("pass at least one of --assert-identity/--check-solver-equivalence/--report-timing/--report-peak-memory")


if __name__ == "__main__":
    main()
