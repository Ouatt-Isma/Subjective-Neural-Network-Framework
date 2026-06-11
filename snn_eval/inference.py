"""Inference: nested sampling, Dirichlet moment matching, SL signals, LoTV split."""
import torch
import torch.nn.functional as F

EPS = 1e-6


@torch.no_grad()
def snn_nested_samples(model, X, Np=10, Nm=10, device="cpu"):
    """Return raw probs (B, Np*Nm, K) and per-beta means (B, Np, K).

    Features are precomputed once (important for CNN/ResNet backbones).
    """
    model.eval().to(device)
    X = X.to(device)
    B = X.shape[0]
    feats = model.extract_features(X)              # (B, H) — computed once
    per_beta = []
    raw = []
    for _ in range(Np):
        p = model.sample_p(B)                      # one trust draw per outer step
        masks = []
        for _ in range(Nm):
            z = torch.bernoulli(p)
            logits = model.fc2(feats * z)
            masks.append(F.softmax(logits, dim=1))
        masks = torch.stack(masks, dim=1)          # (B, Nm, K)
        raw.append(masks)
        per_beta.append(masks.mean(dim=1))         # (B, K)
    raw = torch.cat(raw, dim=1)                    # (B, Np*Nm, K)
    per_beta = torch.stack(per_beta, dim=1)        # (B, Np, K)
    return raw.cpu(), per_beta.cpu()


def dirichlet_moment_match(per_beta):
    """Fit Dirichlet concentration by moment matching across the Np samples.

    per_beta: (B, Np, K). Returns eta (B, K), S (B,).
    """
    m = per_beta.mean(dim=1)                          # (B, K)
    v = per_beta.var(dim=1, unbiased=False).clamp_min(1e-8)
    S_k = m * (1 - m) / v - 1.0                        # (B, K)
    S = S_k.clamp_min(EPS).median(dim=1).values        # (B,)
    K = per_beta.shape[-1]
    eta = m * (S.unsqueeze(1) + K)
    return eta.clamp_min(EPS), S


def sl_signals(raw, per_beta):
    """Compute the three SL signals + LoTV decomposition.

    Returns dict with: probs (mean), u, H, neg_b, aleatoric, epistemic, total.
    """
    B, N, K = raw.shape
    mean = raw.mean(dim=1)                              # (B, K) grand mean (pi_bar)
    eta, S = dirichlet_moment_match(per_beta)
    b = eta / (S.unsqueeze(1) + K)
    u = K / (S + K)
    H = -(mean.clamp_min(EPS) * mean.clamp_min(EPS).log()).sum(dim=1)
    neg_b = 1 - b.max(dim=1).values
    # Law of total variance (trace form) from raw samples
    sq = (raw ** 2).sum(dim=2)                          # ||pi_n||^2, (B, N)
    aleatoric = (1 - sq).mean(dim=1)                    # E_w[1 - ||pi||^2]
    total = 1 - (mean ** 2).sum(dim=1)                  # 1 - ||pi_bar||^2
    epistemic = (total - aleatoric).clamp_min(0)        # = E||pi||^2 - ||pi_bar||^2
    return dict(probs=mean, u=u, H=H, neg_b=neg_b, b=b, eta=eta, S=S,
                aleatoric=aleatoric, epistemic=epistemic, total=total)


@torch.no_grad()
def mc_dropout_probs(model, X, T=100, device="cpu"):
    model.eval().to(device)
    X = X.to(device)
    outs = [F.softmax(model(X, sample=True), dim=1) for _ in range(T)]
    outs = torch.stack(outs, dim=1)                     # (B, T, K)
    return outs.mean(dim=1).cpu(), outs.cpu()


@torch.no_grad()
def edl_opinion(model, X, device="cpu"):
    model.eval().to(device)
    eta = model(X.to(device))
    S = eta.sum(dim=1)
    K = eta.shape[1]
    probs = (eta / S.unsqueeze(1)).cpu()
    u = (K / S).cpu()
    return probs, u


@torch.no_grad()
def deterministic_probs(model, X, device="cpu"):
    model.eval().to(device)
    return F.softmax(model(X.to(device), sample=False), dim=1).cpu()
