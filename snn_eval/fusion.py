"""Subjective Logic fusion operators and scalar fusion baselines (Experiment 3)."""
import numpy as np
import torch

EPS = 1e-8


def probs_to_eta(per_beta):
    """Per-source SNN opinion via moment matching. per_beta: (B, Np, K) -> eta (B,K)."""
    from .inference import dirichlet_moment_match
    eta, _ = dirichlet_moment_match(per_beta)
    return eta


def eta_to_opinion(eta):
    K = eta.shape[1]
    S = eta.sum(1, keepdim=True)
    b = eta / (S + K)
    u = (K / (S.squeeze(1) + K))
    return b, u  # (B,K), (B,)


def opinion_to_eta(b, u):
    K = b.shape[1]
    S = K / u.clamp_min(EPS) - K
    return b * (S.unsqueeze(1) + K)


def sl_cumulative(etas):
    """eta_C = sum eta_i - (N-1)."""
    N = len(etas)
    return torch.stack(etas).sum(0) - (N - 1)


def sl_averaging(etas):
    return torch.stack(etas).mean(0)


def sl_trust_discounted(etas, trusts):
    """Discount each opinion by binomial trust t_i, then cumulative-fuse."""
    disc = []
    for eta, t in zip(etas, trusts):
        b, u = eta_to_opinion(eta)
        b2 = t * b
        u2 = t * u + (1 - t)
        disc.append(opinion_to_eta(b2, u2))
    return sl_cumulative(disc)


def fuse_eval(per_beta_sources, y, trusts, K, metrics_mod):
    """Run all fusion methods. per_beta_sources: list of (B,Np,K). Returns dict."""
    y_np = y.numpy()
    etas = [probs_to_eta(pb) for pb in per_beta_sources]
    probs_sources = [pb.mean(1) for pb in per_beta_sources]  # (B,K) each

    out = {}

    def record(name, probs):
        out[name] = dict(acc=metrics_mod.accuracy(probs, y),
                         nll=metrics_mod.nll(probs, y),
                         ece=metrics_mod.ece(probs, y))

    # single best source (oracle by accuracy)
    accs = [metrics_mod.accuracy(p, y) for p in probs_sources]
    record("single_best", probs_sources[int(np.argmax(accs))])

    # logit averaging (softmax of mean log-prob)
    logp = torch.stack([p.clamp_min(EPS).log() for p in probs_sources]).mean(0)
    record("logit_avg", torch.softmax(logp, dim=1))

    # accuracy-weighted vote
    w = torch.tensor(accs).clamp_min(EPS); w = w / w.sum()
    record("acc_weighted", sum(wi * p for wi, p in zip(w, probs_sources)))

    # SL averaging / cumulative / trust-discounted
    for name, eta in [("sl_averaging", sl_averaging(etas)),
                      ("sl_cumulative", sl_cumulative(etas)),
                      ("sl_trust_disc", sl_trust_discounted(etas, trusts))]:
        S = eta.clamp_min(EPS).sum(1, keepdim=True)
        record(name, eta.clamp_min(EPS) / S)
    return out
