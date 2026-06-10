"""Augmented Subjective Logic opinion + unit-level trust opinions.

Exact Python port of the playground's computeOpinion/S_drawAgg pipeline:

  Per trust sample i (outer Beta draw), the Nm inner mask predictions are
  COUNTED into a Dirichlet: alpha_i = counts_i + prior.
    m_i      = alpha_i / S_i                      (per-sample mean)
    var_i,k  = alpha_k (S - alpha_k) / (S^2 (S+1))  (per-sample posterior var)
  Across the Np trust samples:
    varE_k (epistemic)  = Var_i[m_i,k]            (variance of the means)
    eVar_k (aleatoric)  = E_i[var_i,k]            (mean within-sample var)
    overallAlpha_k      = sum_i alpha_i,k          (pooled evidence)
  Augmented opinion:
    u        = sum(varE) / (sum(varE) + sum(eVar))    # epistemic SHARE
    b_k      = (1 - u) * overallAlpha_k / S_total      # direction scaled by 1-u
    P_k      = b_k + u / K                             # projected probability
    S*_k     = max(1, P_k(1-P_k)/eVar_k - 1)           # aleatoric-calibrated
    S*       = mean_k S*_k ;  alpha*_k = P_k * S*      # output Dirichlet

This replaces the old Dirichlet moment-matching/MLE: the law-of-total-variance
split is built INTO the opinion, so vacuity now answers "what fraction of my
uncertainty is epistemic" instead of collapsing when the model is confident.

Per-unit (binomial) opinion, W=2, base rate 1/2:
    u_j = 2/(2+alpha_j+beta_j), b_j = alpha_j/(2+..), d_j = beta_j/(2+..)
Sub-prior alpha,beta < 1 correctly yields HIGHER vacuity than alpha=beta=1.
"""
import torch

EPS = 1e-12


# ----------------------------------------------------------------------
# Augmented opinion from nested samples
# ----------------------------------------------------------------------
def per_sample_dirichlets(raw, prior=1.0, mode="counts"):
    """raw: (B, Np, Nm, K) softmax probs. Returns alpha (B,Np,K), mean, var.

    mode='counts' (playground-faithful): argmax each inner prediction, count.
    mode='soft': use soft probabilities as fractional counts.
    """
    B, Np, Nm, K = raw.shape
    if mode == "counts":
        cls = raw.argmax(dim=-1)                                  # (B,Np,Nm)
        counts = torch.zeros(B, Np, K, dtype=raw.dtype)
        counts.scatter_add_(2, cls, torch.ones_like(cls, dtype=raw.dtype))
    else:  # soft fractional counts
        counts = raw.sum(dim=2)                                   # (B,Np,K)
    alpha = counts + prior
    S = alpha.sum(-1, keepdim=True)
    mean = alpha / S
    var = alpha * (S - alpha) / (S ** 2 * (S + 1))
    return alpha, mean, var


def augmented_opinion(raw, prior=1.0, mode="counts", aleatoric="categorical"):
    """raw: (B, Np, Nm, K). Returns dict of augmented-SL quantities per input.

    aleatoric='categorical' (default, LoTV-faithful): eVar_k = E_i[m_ik(1-m_ik)],
      the within-sample categorical spread — gives the clean regime separation
      (aleatoric regime -> u low; epistemic regime -> u high).
    aleatoric='posterior' (playground-faithful): eVar = mean per-sample Dirichlet
      posterior variance. WARNING: this shrinks as 1/Nm just like the multinomial
      noise in varE, so u -> ~0.5+ even in a purely aleatoric regime.
    """
    B, Np, Nm, K = raw.shape
    alpha, mean, var = per_sample_dirichlets(raw, prior, mode)
    varE = mean.var(dim=1, unbiased=False)                        # (B,K) epistemic
    if aleatoric == "categorical":
        eVar = (mean * (1 - mean)).mean(dim=1)                    # (B,K) LoTV aleatoric
    else:
        eVar = var.mean(dim=1)                                    # posterior variance
    overall = alpha.sum(dim=1)                                    # (B,K) pooled
    u_e = varE.sum(-1)
    u_a = eVar.sum(-1)
    u = u_e / (u_e + u_a + EPS)                                   # epistemic share
    dir_mean = overall / overall.sum(-1, keepdim=True).clamp_min(EPS)
    b = (1 - u).unsqueeze(1) * dir_mean
    P = b + (u / K).unsqueeze(1)
    S_star_k = (P * (1 - P) / eVar.clamp_min(EPS) - 1).clamp_min(1.0)
    S_star = S_star_k.mean(-1)
    alpha_star = P * S_star.unsqueeze(1)
    return dict(u=u, u_e=u_e, u_a=u_a, b=b, P=P, S_star=S_star,
                alpha_star=alpha_star, varE=varE, eVar=eVar,
                probs=P, neg_b=1 - b.max(-1).values,
                H=-(P.clamp_min(EPS) * P.clamp_min(EPS).log()).sum(-1))


def raw_to_4d(raw_flat, Np, Nm):
    """(B, Np*Nm, K) from inference.snn_nested_samples -> (B, Np, Nm, K)."""
    B, NN, K = raw_flat.shape
    assert NN == Np * Nm
    return raw_flat.view(B, Np, Nm, K)


# ----------------------------------------------------------------------
# Unit-level trust opinions and regimes
# ----------------------------------------------------------------------
def unit_opinions(alpha, beta, W=2.0):
    """Binomial SL opinion per unit from Beta(alpha, beta) evidence."""
    R = alpha + beta
    u = W / (W + R)
    b = alpha / (W + R)
    d = beta / (W + R)
    P = b + 0.5 * u                  # = (alpha+1)/(alpha+beta+2)
    return dict(u=u, b=b, d=d, P=P)


def classify_regime(alpha, beta, ratio=4.0, big=4.0, small=0.5):
    """Map each unit's (alpha, beta) to a named regime.

    trusted:      alpha >> beta            -> unit ~always on (no uncertainty)
    distrusted:   alpha << beta            -> unit ~always off
    aleatoric:    alpha ~ beta, both >> 1  -> p concentrates at 0.5; masks flip
                  per-draw -> within-sample (inner) variance -> ALEATORIC
    epistemic_u:  alpha ~ beta, both << 1  -> U-shape; p ~ 0 or 1 per trust draw;
                  consistent within a draw, differs across draws -> ACROSS-sample
                  variance -> EPISTEMIC
    epistemic_flat: alpha ~ beta ~ 1       -> flat p; mixed, epistemic-leaning
    """
    out = []
    for a, bb in zip(alpha.tolist(), beta.tolist()):
        if a > ratio * bb:
            out.append("trusted")
        elif bb > ratio * a:
            out.append("distrusted")
        elif a > big and bb > big:
            out.append("aleatoric")
        elif a < small and bb < small:
            out.append("epistemic_u")
        else:
            out.append("epistemic_flat")
    return out


def regime_summary(model):
    """Summarise learned unit regimes for a SubjectiveHead / SNN net layer."""
    a, b = model.alpha_beta()
    ops = unit_opinions(a.detach(), b.detach())
    regs = classify_regime(a.detach(), b.detach())
    from collections import Counter
    return Counter(regs), ops
