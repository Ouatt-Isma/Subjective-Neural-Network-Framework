"""Experiment 2: two-threshold abstention.

Train SNN head on the 'easy' regime, evaluate on the easy/split/diffuse mixture,
compare 1D rejection rules (H, neg_b, u) against the 2D (max_b, u) frontier.

Persistence:
  results/cache/run_exp2_<hash>_results.json  (keyed by all params; no model saving
  since each seed trains a fresh model — the per-seed AUSCs are the output)

Use --no-cache to re-run all seeds from scratch.

Usage: python -m snn_eval.run_exp2 --seeds 10
       python -m snn_eval.run_exp2 --seeds 10 --no-cache
"""
import argparse
import numpy as np
import torch
from scipy import stats
from . import models, inference, metrics, data, cache
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def ausc_1d(score_reject, correct):
    return metrics.selective_ausc(-score_reject.numpy(), correct)


def ausc_2d(max_b, u, correct, grid=60):
    """Accept iff max_b > tb AND u < tu; take Pareto envelope over grid."""
    max_b, u = max_b.numpy(), u.numpy()
    correct = np.asarray(correct, dtype=float)
    tbs = np.quantile(max_b, np.linspace(0, 1, grid))
    tus = np.quantile(u, np.linspace(0, 1, grid))
    pts = []
    for tb in tbs:
        for tu in tus:
            acc = (max_b > tb) & (u < tu)
            cov = acc.mean()
            if cov < 0.4:
                continue
            ret_acc = correct[acc].mean() if acc.sum() else 0.0
            pts.append((cov, ret_acc))
    if not pts:
        return 0.0
    pts = sorted(pts)
    covs = np.array([p[0] for p in pts]); accs = np.array([p[1] for p in pts])
    grid_cov = np.linspace(0.4, 1.0, 100)
    env = [accs[covs >= c].max() if (covs >= c).any() else 0.0 for c in grid_cov]
    return float(_trapz(env, grid_cov) / 0.6)


def run_seed(seed, args):
    torch.manual_seed(seed)
    Xtr, ytr, _, _, _, protos = data.make_synthetic(
        n_per_class=400, K=args.K, d=args.d, seed=seed)
    snn = models.SubjectiveHead(args.d, args.d_hidden, args.K,
                                prior_a=1, prior_b=1, init_keep=0.5)
    snn = models.train_head(snn, Xtr, ytr, args.K, epochs=args.epochs, is_snn=True,
                            beta_max=args.beta_max, warmup_frac=0.2, device=args.device)
    Xmix, ymix, _ = data.make_regime_mixture(protos, K=args.K, seed=seed)
    raw, pb = inference.snn_nested_samples(snn, Xmix, args.Np, args.Nm, args.device)
    sig = inference.sl_signals(raw, pb)
    correct = (sig["probs"].argmax(1) == ymix).numpy()
    max_b = sig["b"].max(1).values
    return {
        "H":     ausc_1d(sig["H"],     correct),
        "neg_b": ausc_1d(sig["neg_b"], correct),
        "u":     ausc_1d(sig["u"],     correct),
        "2D":    ausc_2d(max_b, sig["u"], correct),
    }


def compute(args):
    keys = ["H", "neg_b", "u", "2D"]
    vals = {k: [] for k in keys}
    for s in range(args.seeds):
        r = run_seed(s, args)
        for k in keys:
            vals[k].append(r[k])
        print(f"seed {s}: " + " ".join(f"{k}={r[k]:.4f}" for k in keys))
    return {"vals": vals}


def display(res):
    vals = res["vals"]
    keys = ["H", "neg_b", "u", "2D"]
    two = np.array(vals["2D"])
    print("\n%-8s %8s %20s %10s" % ("Signal", "MeanAUSC", "95% CI", "p vs 2D"))
    for k in keys:
        a = np.array(vals[k])
        m = a.mean()
        ci = stats.t.interval(0.95, len(a) - 1, loc=m, scale=stats.sem(a)) if len(a) > 1 else (m, m)
        if k == "2D":
            p = float("nan")
        else:
            try:
                p = stats.wilcoxon(two, a, alternative="greater").pvalue
            except Exception:
                p = float("nan")
        print("%-8s %8.4f   [%.4f, %.4f] %10.3f" % (k, m, ci[0], ci[1], p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--d_hidden", type=int, default=12)
    ap.add_argument("--beta_max", type=float, default=40.0)
    ap.add_argument("--Np", type=int, default=20)
    ap.add_argument("--Nm", type=int, default=20)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--no-cache", action="store_true", dest="no_cache",
                    help="Re-run all seeds from scratch")
    args = ap.parse_args()

    all_params = {k: v for k, v in vars(args).items() if k not in ("device", "no_cache")}
    res = None if args.no_cache else cache.load_results("run_exp2", all_params)
    if res is None:
        res = compute(args)
        cache.save_results("run_exp2", all_params, res)
    display(res)


if __name__ == "__main__":
    main()
