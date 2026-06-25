"""Augmented Subjective Logic opinion + unit-level trust opinions.

Exact Python port of the playground's computeOpinionBALD pipeline (BALD /
mutual-information decomposition — see dirichlet_playground_lastv2.html,
D_drawAgg -> computeOpinionBALD). This REPLACES the earlier law-of-total-
variance (LoTV) formulation: vacuity is now the fraction of predictive
entropy explained by disagreement between trust draws, not a variance ratio.

  Per trust sample i (outer Beta draw), the Nm inner mask predictions are
  COUNTED:
    rawMean_i,k = counts_i,k / Nm                       (no prior; entropy input)
    mean_i,k    = (counts_i,k + prior) / (Nm + prior*K)  (prior-smoothed; belief
                                                           direction input)
  Across the Np trust samples:
    rawGrandMean_k = E_i[rawMean_i,k]      (pooled raw frequency, prior-free)
    grandMean_k    = E_i[mean_i,k]          (pooled, prior-smoothed)
    H_total        = H(rawGrandMean)        (entropy of the mixture = total
                                              predictive entropy)
    eH             = E_i[H(rawMean_i)]      (expected per-draw entropy = aleatoric)
    MI             = max(0, H_total - eH)   (mutual information = epistemic)
  Augmented (BALD) opinion:
    u        = MI / H_total                            # epistemic SHARE
    b_k      = (1 - u) * grandMean_k                    # direction, prior-smoothed
    P_k      = b_k + u / K                              # projected probability
    S*       = clip(K / (eH / log K), K, 1e4)           # aleatoric-calibrated conc.
    alpha*_k = P_k * S*                                 # output Dirichlet

Per-unit (binomial) opinion, W=2, base rate 1/2:
    u_j = 2/(2+alpha_j+beta_j), b_j = alpha_j/(2+..), d_j = beta_j/(2+..)
Sub-prior alpha,beta < 1 correctly yields HIGHER vacuity than alpha=beta=1.
"""
import math
import torch

EPS = 1e-12


# ----------------------------------------------------------------------
# Augmented (BALD / mutual-information) opinion from nested samples
# ----------------------------------------------------------------------
def per_sample_dirichlets(raw, prior=1.0, mode="counts"):
    """raw: (B, Np, Nm, K) softmax probs. Returns counts, raw_mean, mean (B,Np,K).

    mode='counts' (playground-faithful): argmax each inner prediction, count.
    mode='soft': use soft probabilities as fractional counts.
    raw_mean = counts / Nm        (no prior; entropy input)
    mean     = (counts+prior)/S   (prior-smoothed; belief-direction input)
    """
    B, Np, Nm, K = raw.shape
    if mode == "counts":
        cls = raw.argmax(dim=-1)                                  # (B,Np,Nm)
        counts = torch.zeros(B, Np, K, dtype=raw.dtype)
        counts.scatter_add_(2, cls, torch.ones_like(cls, dtype=raw.dtype))
    else:  # soft fractional counts
        counts = raw.sum(dim=2)                                   # (B,Np,K)
    raw_mean = counts / Nm
    alpha = counts + prior
    S = alpha.sum(-1, keepdim=True)
    mean = alpha / S
    return counts, raw_mean, mean


def _cat_entropy(p):
    """Categorical entropy, safe at p=0 (matches JS catEntropy). p: (...,K)."""
    return -(p * p.clamp_min(EPS).log()).sum(-1)


def bald_split(samples):
    """samples: (B, T, K) categorical distributions (or empirical frequency
    vectors) along any flat ensemble axis T -- nested-sampling outer draws,
    MC Dropout forward passes, deep-ensemble members, etc.

    Returns (H_total, eH, MI, u): predictive entropy decomposed via mutual
    information between the ensemble-member index and the predicted class
    (Houlsby BALD / Depeweg 2018) -- H_total = eH (aleatoric) + MI (epistemic),
    u = MI / H_total (epistemic share).
    """
    mean = samples.mean(dim=1)                                      # (B,K)
    H_total = _cat_entropy(mean)                                    # (B,)
    eH = _cat_entropy(samples).mean(dim=1)                          # (B,)
    MI = (H_total - eH).clamp_min(0.0)                              # (B,)
    u = torch.where(H_total > 0,
                     (MI / H_total.clamp_min(EPS)).clamp(max=1.0),
                     torch.zeros_like(H_total))
    return H_total, eH, MI, u


def bald_opinion(samples):
    """samples: (B, T, K) softmax draws from a flat ensemble (e.g. MC Dropout
    forward passes). Same entropy/MI split as augmented_opinion (bald_split)
    without the nested trust structure or Dirichlet-opinion fitting -- there's
    no prior/belief-direction step here, just the prior-free entropy split.
    """
    H_total, eH, MI, u = bald_split(samples)
    return dict(probs=samples.mean(dim=1), u=u, u_e=MI, u_a=eH,
                H_total=H_total, eH=eH, MI=MI, H=H_total)


def augmented_opinion(raw, prior=1.0, mode="counts"):
    """raw: (B, Np, Nm, K). Returns dict of augmented (BALD) SL quantities.

    u = MI / H(rawGrandMean) — epistemic share via mutual information between
    the trust-draw index and the predicted class (BALD). Entropy is computed
    on raw (prior-free) empirical frequencies so the prior cannot inflate
    E[H] and artificially suppress the epistemic signal.
    """
    B, Np, Nm, K = raw.shape
    counts, raw_mean, mean = per_sample_dirichlets(raw, prior, mode)

    grand_mean = mean.mean(dim=1)                                  # (B,K) prior-smoothed
    H_total, eH, MI, u = bald_split(raw_mean)

    b = (1 - u).unsqueeze(1) * grand_mean
    P = b + (u / K).unsqueeze(1)

    Hmax = math.log(K)
    ratio = eH / Hmax
    S_star = torch.where(ratio > 0, K / ratio.clamp_min(EPS),
                          torch.full_like(ratio, 1e4))
    S_star = S_star.clamp(min=K, max=1e4)
    alpha_star = P * S_star.unsqueeze(1)

    return dict(u=u, MI=MI, eH=eH, H_total=H_total, u_e=MI, u_a=eH,
                b=b, P=P, S_star=S_star, alpha_star=alpha_star,
                probs=P, neg_b=1 - b.max(-1).values, H=_cat_entropy(P))


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
