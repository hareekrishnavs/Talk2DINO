"""F1e: mandatory gradient correctness for the implicit fixed-point solve.
CPU, float64, small synthetic problem (N=64, C=4, k=4) -- this must catch a
wrong analytic gradient before anything is ever trained on it (F1e)."""
import torch
import torch.nn.functional as F

from src.learned_affinity import LearnedMetric, build_differentiable_knn_graph, implicit_propagate
from src.learned_affinity.implicit_solve import ImplicitPropagate, solve_fixed_point


def _synthetic_graph(n=64, k=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.zeros(n, k, dtype=torch.int64)
    for p in range(n):
        candidates = [i for i in range(n) if i != p]
        chosen = torch.tensor(candidates)[
            torch.randperm(len(candidates), generator=generator)[:k]
        ]
        indices[p] = chosen
    raw = torch.rand(n, k, generator=generator, dtype=torch.float64) + 0.1
    weights = (raw / raw.sum(-1, keepdim=True)).clone().requires_grad_(True)
    return indices, weights


def test_finite_difference_gradient_matches_analytic():
    torch.manual_seed(0)
    n, c, k = 64, 4, 4
    indices, weights = _synthetic_graph(n, k, seed=1)
    s0 = torch.randn(n, c, dtype=torch.float64)
    direction = torch.randn(n, c, dtype=torch.float64)
    alpha = 0.9
    tol = 1e-12
    max_iter = 500

    def loss_of(w):
        s_star = implicit_propagate(s0, indices, w, alpha, tol=tol, max_iter=max_iter)
        return (s_star * direction).sum()

    loss = loss_of(weights)
    loss.backward()
    analytic = weights.grad.clone()
    assert analytic is not None and torch.isfinite(analytic).all()

    eps = 1e-4
    max_rel_error = 0.0
    generator = torch.Generator().manual_seed(2)
    sample_edges = [
        (int(p), int(kk))
        for p, kk in zip(
            torch.randint(0, n, (20,), generator=generator),
            torch.randint(0, k, (20,), generator=generator),
        )
    ]
    for p, kk in sample_edges:
        with torch.no_grad():
            w_plus = weights.detach().clone()
            w_plus[p, kk] += eps
            w_minus = weights.detach().clone()
            w_minus[p, kk] -= eps
        loss_plus = loss_of(w_plus.requires_grad_(False))
        loss_minus = loss_of(w_minus.requires_grad_(False))
        fd = (loss_plus - loss_minus).item() / (2 * eps)
        an = analytic[p, kk].item()
        rel_error = abs(an - fd) / max(abs(an), abs(fd), 1e-12)
        max_rel_error = max(max_rel_error, rel_error)

    assert max_rel_error < 1e-4, f"max relative error {max_rel_error} >= 1e-4"


def test_gradcheck_float64():
    n, c, k = 64, 4, 4
    indices, weights = _synthetic_graph(n, k, seed=3)
    s0 = torch.randn(n, c, dtype=torch.float64, requires_grad=True)
    alpha = 0.9

    def func(s0_in, weights_in):
        return ImplicitPropagate.apply(s0_in, indices, weights_in, alpha, 1e-12, 500, True)

    assert torch.autograd.gradcheck(
        func, (s0, weights), eps=1e-6, atol=1e-5, rtol=1e-3,
    )


def test_shipped_default_no_overrides():
    """Adversarial review finding (ADVERSARIAL_REVIEW_PARTF1.md, X3): the
    OLD Richardson solver's shipped default (tol=1e-6, max_iter=500) gave
    30-70% gradient error against finite differences, but every test in
    this file (before this one was added) hardcoded a much tighter
    non-default tol, so the actual default was never checked. This test
    calls implicit_propagate with NO tol/max_iter/anything override -- the
    exact call an F2 training loop would naturally write -- and checks it
    against finite differences at the same 1e-4 bar as F1e."""
    torch.manual_seed(0)
    n, c, k = 64, 4, 4
    indices, weights = _synthetic_graph(n, k, seed=5)
    s0 = torch.randn(n, c, dtype=torch.float64)
    direction = torch.randn(n, c, dtype=torch.float64)
    alpha = 0.9

    def loss_of(w):
        s_star = implicit_propagate(s0, indices, w, alpha)  # NO overrides
        return (s_star * direction).sum()

    loss = loss_of(weights)
    loss.backward()
    analytic = weights.grad.clone()

    eps = 1e-4
    max_rel_error = 0.0
    generator = torch.Generator().manual_seed(6)
    for p, kk in zip(
        torch.randint(0, n, (20,), generator=generator),
        torch.randint(0, k, (20,), generator=generator),
    ):
        p, kk = int(p), int(kk)
        with torch.no_grad():
            w_plus = weights.detach().clone(); w_plus[p, kk] += eps
            w_minus = weights.detach().clone(); w_minus[p, kk] -= eps
        lp = loss_of(w_plus.requires_grad_(False))
        lm = loss_of(w_minus.requires_grad_(False))
        fd = (lp - lm).item() / (2 * eps)
        an = analytic[p, kk].item()
        rel = abs(an - fd) / max(abs(an), abs(fd), 1e-12)
        max_rel_error = max(max_rel_error, rel)

    assert max_rel_error < 1e-4, (
        f"shipped default (no overrides) gradient check FAILED: "
        f"max relative error {max_rel_error} >= 1e-4"
    )


def test_solver_convergence_assertion_raises():
    """Convergence assertion (per user request following X3): a solve that
    cannot reach `tol` within `max_iter` must raise SolverConvergenceError,
    not silently return an under-converged (and therefore
    under-differentiated) result."""
    from src.learned_affinity import SolverConvergenceError

    n, c, k = 64, 4, 4
    indices, weights = _synthetic_graph(n, k, seed=7)
    weights = weights.detach()
    s0 = torch.randn(n, c, dtype=torch.float64)
    # tol=0 can never be satisfied -- forces max_iter exhaustion every time.
    try:
        solve_fixed_point(s0, indices, weights, 0.9, tol=0.0, max_iter=5)
        raised = False
    except SolverConvergenceError:
        raised = True
    assert raised, "solve_fixed_point did not raise on non-convergence"

    # raise_on_nonconvergence=False is the documented escape hatch for
    # controlled diagnostics (e.g. forcing an exact iteration count).
    s_star, iters = solve_fixed_point(
        s0, indices, weights, 0.9, tol=0.0, max_iter=5, raise_on_nonconvergence=False,
    )
    assert iters == 5
    assert torch.isfinite(s_star).all()


def test_forward_solve_converges_and_is_untracked():
    n, c, k = 64, 4, 4
    indices, weights = _synthetic_graph(n, k, seed=4)
    weights = weights.detach()
    s0 = torch.randn(n, c, dtype=torch.float64)
    s_star, iters = solve_fixed_point(s0, indices, weights, 0.9, tol=1e-10, max_iter=500)
    assert iters < 500
    assert torch.isfinite(s_star).all()
    # solving to convergence: residual of the linear system itself must be tiny
    from src.learned_affinity.implicit_solve import apply_knn
    residual = (1 - 0.9) * s0 + 0.9 * apply_knn(s_star, indices, weights) - s_star
    assert residual.norm() < 1e-8


def test_learned_metric_identity_at_untrained_init():
    torch.manual_seed(0)
    metric = LearnedMetric()
    f = F.normalize(torch.randn(50, 768), dim=-1)
    g = metric(f)
    assert torch.allclose(g, f, atol=1e-6)
    g_forced = metric(f, r_override=0.0)
    assert torch.allclose(g_forced, f, atol=1e-6)


def test_end_to_end_gradient_at_r0_and_r03():
    """End-to-end gradient check through the FULL differentiable chain --
    LearnedMetric parameters -> g(f) -> graph construction -> implicit
    solve -> loss -- at both r=0 and r=0.3, using a RANDOMLY (non-zero)
    initialised MLP so the check actually exercises every layer (the
    default zero-init final layer would make d(loss)/d(first-layer
    weights) trivially zero on both sides, an unfalsifiable test).

    r=0 is forced via `r_override=0.0, bypass_mlp_at_zero=False`: this
    keeps the MLP in the autograd graph (unlike the identity-gate's
    short-circuit) but, by design, an EXPLICIT r_override is a literal
    python float, not `self.r`, so `r` is correctly disconnected from
    the graph there and is excluded from this check (checking MLP params
    only). r=0.3 is instead reached by setting `metric.r` directly (the
    plain scalar parameter, no override), so `r` participates and IS
    checked."""
    P, D, K, C = 48, 32, 6, 3

    def run(r_target, use_override):
        torch.manual_seed(0)
        metric = LearnedMetric(dim=D, hidden=16, r_max=0.5, kappa=3.0, k=K).double()
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for param in metric.parameters():
                param.copy_(torch.randn(param.shape, generator=generator, dtype=torch.float64) * 0.05)
            if not use_override:
                metric.r.copy_(torch.tensor(r_target, dtype=torch.float64))

        f = F.normalize(torch.randn(P, D, generator=generator, dtype=torch.float64), dim=-1)
        s0 = torch.randn(P, C, generator=generator, dtype=torch.float64)
        direction = torch.randn(P, C, generator=generator, dtype=torch.float64)
        alpha = 0.9

        def loss_of():
            g = metric(f, r_override=r_target, bypass_mlp_at_zero=False) if use_override else metric(f)
            indices, weights = build_differentiable_knn_graph(g, k=K, kappa=metric.kappa)
            s_star = implicit_propagate(s0.clone(), indices, weights, alpha, tol=1e-12, max_iter=1000)
            return (s_star * direction).sum()

        if not use_override:
            # sanity check ONLY at the unperturbed baseline -- loss_of() is
            # re-evaluated with r perturbed by +-eps during the sweep
            # below, so metric.r legitimately drifts slightly from r_target then.
            with torch.no_grad():
                assert abs(float(metric.r) - r_target) < 1e-9
        loss = loss_of()
        metric.zero_grad()
        loss.backward()

        checks = [
            ("mlp.0.weight", metric.mlp[0].weight, [(0, 0), (2, 5)]),
            ("mlp.0.bias", metric.mlp[0].bias, [(1,)]),
            ("mlp.2.weight", metric.mlp[2].weight, [(0, 0), (10, 3)]),
            ("mlp.2.bias", metric.mlp[2].bias, [(4,)]),
        ]
        if not use_override:
            checks.append(("r", metric.r, [()]))
        eps = 1e-5
        max_rel_error = 0.0
        for name, tensor, coords in checks:
            assert tensor.grad is not None, f"r={r_target}: {name} received no gradient at all"
            for coord in coords:
                with torch.no_grad():
                    (tensor[coord] if coord else tensor).add_(eps)
                loss_plus = loss_of().item()
                with torch.no_grad():
                    (tensor[coord] if coord else tensor).add_(-2 * eps)
                loss_minus = loss_of().item()
                with torch.no_grad():
                    (tensor[coord] if coord else tensor).add_(eps)  # restore
                fd = (loss_plus - loss_minus) / (2 * eps)
                analytic = (tensor.grad[coord] if coord else tensor.grad).item()
                rel = abs(analytic - fd) / max(abs(analytic), abs(fd), 1e-10)
                max_rel_error = max(max_rel_error, rel)

        assert max_rel_error < 1e-4, (
            f"r={r_target}: end-to-end max relative error {max_rel_error} >= 1e-4"
        )

    run(0.0, use_override=True)
    run(0.3, use_override=False)


def test_differentiable_graph_matches_reference_at_identity():
    """At r=0 (g=f exactly to float32 eps), the differentiable graph
    builder must match the existing, non-differentiable build_knn_graph
    exactly (indices identical, weights identical to float16-cache
    tolerance) -- this is the graph half of F1f's identity gate."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.e3_affinity_oracle import build_knn_graph

    torch.manual_seed(0)
    f = F.normalize(torch.randn(200, 768), dim=-1)
    metric = LearnedMetric()
    g = metric(f, r_override=0.0)

    ref_indices, ref_weights, _zero = build_knn_graph(f, knn_k=12, affinity_power=3.0)
    learned_indices, learned_weights = build_differentiable_knn_graph(g, k=12, kappa=3.0)

    assert torch.equal(learned_indices.cpu(), ref_indices.to(torch.int64))
    # Exact at the precision the graph is actually stored/replayed at
    # (float16) -- see LearnedMetric.forward's r_override==0.0 short
    # circuit, which returns f unmodified so g is f is True, not merely
    # close to it.
    assert torch.equal(learned_weights.detach().to(torch.float16).cpu(), ref_weights.cpu())


def test_gradient_at_true_untrained_init_final_layer_resolves_at_default_eps():
    """F1 regression test. Before the fix, `r` at the true untrained init
    was ~2.3e-5 (r_max * sigmoid(-10)), making the analytic gradient to the
    MLP's final layer real but vanishingly tiny -- finite differences at the
    usual eps=1e-4/1e-5 scale failed (max relative error > 1e-4) not because
    the gradient was wrong, but because the FD step was too small to
    resolve a gradient that tiny against float64 rounding noise from the
    CGLS solve. See RUN_PartF2fix.md for the full diagnosis.

    After F1 (r a plain parameter at 0.1, not gated through a near-zero
    sigmoid), the final-layer gradient is no longer vanishingly small, so
    it MUST now resolve correctly at the DEFAULT eps too -- if this ever
    regresses back to needing a larger eps, that is exactly the old bug
    recurring.

    Still documents (does not just assert away) two exact-zero gradients at
    this starting point, both structural, both UNCHANGED by F1: the first
    MLP layer (blocked in the backward chain by the zero final layer,
    regardless of r) and `r` itself (dg/dr = MLP(f) = 0 identically while
    the final layer is zero, so the loss is completely insensitive to r
    until the final layer moves first) -- these clear within the first few
    real training steps once the final layer's now-much-larger gradient
    moves it, per Gate 1."""
    P, D, K, C = 64, 48, 6, 3
    alpha = 0.9
    torch.manual_seed(0)
    metric = LearnedMetric(dim=D, hidden=24, r_max=0.5, kappa=3.0, k=K).double()
    with torch.no_grad():
        # 1e-7, not 1e-9: `r` is initialised via torch.tensor(0.1) (float32,
        # where 0.1 is not exactly representable) THEN upcast to float64 by
        # .double() -- ~1.5e-9 of float32 rounding survives the upcast, an
        # expected artifact of construction order, not of this test.
        assert abs(float(metric.r) - 0.1) < 1e-7  # the actual shipped init, unchanged by this test

    generator = torch.Generator().manual_seed(11)
    f = F.normalize(torch.randn(P, D, generator=generator, dtype=torch.float64), dim=-1)
    s0 = torch.randn(P, C, generator=generator, dtype=torch.float64)
    direction = torch.randn(P, C, generator=generator, dtype=torch.float64)

    def loss_of():
        g = metric(f)  # no override -- r flows naturally, exactly as training would call it
        indices, weights = build_differentiable_knn_graph(g, k=K, kappa=metric.kappa)
        s_star = implicit_propagate(s0.clone(), indices, weights, alpha)  # shipped default, untouched
        return (s_star * direction).sum()

    loss = loss_of()
    metric.zero_grad()
    loss.backward()

    # Structural exact zeros at this starting point -- confirmed, not skipped.
    # Unaffected by F1: both come from the zero-initialised final layer, not from r's value.
    assert metric.mlp[0].weight.grad.norm().item() == 0.0
    assert metric.r.grad.item() == 0.0

    coord = (0, 0)
    analytic = metric.mlp[2].weight.grad[coord].item()
    w = metric.mlp[2].weight

    def fd_at(eps):
        with torch.no_grad():
            w[coord] += eps
        lp = loss_of().item()
        with torch.no_grad():
            w[coord] -= 2 * eps
        lm = loss_of().item()
        with torch.no_grad():
            w[coord] += eps
        fd = (lp - lm) / (2 * eps)
        return abs(analytic - fd) / max(abs(analytic), abs(fd), 1e-12)

    rel_default_eps = fd_at(1e-5)
    assert rel_default_eps < 1e-4, (
        f"gradient to the MLP's final layer at the true untrained init (r=0.1) failed to "
        f"resolve even at the DEFAULT FD eps ({rel_default_eps} >= 1e-4) -- this is the "
        f"exact signature of the pre-F1 bug (a vanishingly small true gradient) recurring."
    )
