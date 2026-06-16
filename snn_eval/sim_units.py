"""Unit-regime simulation: how unit-level (alpha, beta) translates to output
aleatoric vs epistemic uncertainty. Python replication of the playground.

Minimal 2-unit network with deterministic priority inference:
    m1, m2 ~ Bernoulli(p),  p ~ Beta(alpha, beta)  (one shared trust var)
    m2=1 -> class 1;  m2=0,m1=1 -> class 0;  m2=0,m1=0 -> class 2
Theoretical class probs given p:  P(0)=(1-p)p,  P(1)=p,  P(2)=(1-p)^2.

For each regime we run nested sampling (Np outer trust draws, Nm inner mask
draws), compute the BALD split (MI = epistemic mutual information, eH =
expected per-draw entropy = aleatoric) and the augmented opinion, and check
the mechanism claims:

  trusted   (a>>b): p~1, always class 1            -> H_total, eH ~ 0, u* low
  distrusted(a<<b): p~0, always class 2            -> H_total, eH ~ 0, u* low
  aleatoric (a=b>>1): p~0.5 every draw; classes vary WITHIN a draw
                      -> eH ~ H_total  -> MI small -> u (epistemic share) LOW
  epistemic (a=b<<1): p~0 or 1 PER draw; pure within, differs across
                      -> eH ~ 0, H_total large -> MI ~ H_total -> u HIGH
  flat      (a=b=1):  p uniform per draw           -> mixed, u mid/high

Pooled counts can be IDENTICAL for the aleatoric and epistemic regimes
([0,1,2],[0,1,2],[0,1,2] vs [0,0,0],[1,1,1],[2,2,2]) — a pooled Dirichlet
cannot tell them apart; only the BALD split inside the augmented opinion can.

Usage: python -m snn_eval.sim_units [--Np 200 --Nm 50 --prior 1.0]
"""
import argparse
import torch
from .augmented import augmented_opinion

REGIMES = {
    "trusted":        (20.0, 1.0),
    "distrusted":     (1.0, 20.0),
    "aleatoric":      (10.0, 10.0),
    "epistemic_u":    (0.1, 0.1),
    "epistemic_flat": (1.0, 1.0),
}


def infer_class(m1, m2):
    # priority rule: m2=1 -> 1 ; else m1=1 -> 0 ; else 2
    return torch.where(m2 == 1, torch.ones_like(m1),
                       torch.where(m1 == 1, torch.zeros_like(m1),
                                   torch.full_like(m1, 2)))


def simulate(alpha, beta, Np, Nm, seed=0):
    g = torch.Generator().manual_seed(seed)
    dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta))
    raw = torch.zeros(1, Np, Nm, 3)
    for i in range(Np):
        p = dist.sample()
        m1 = (torch.rand(Nm, generator=g) < p).long()
        m2 = (torch.rand(Nm, generator=g) < p).long()
        cls = infer_class(m1, m2)
        raw[0, i] = torch.nn.functional.one_hot(cls, 3).float()
    return raw


def theory(alpha, beta, n=200000, seed=0):
    """Analytic BALD split in the Nm->inf inner limit (multinomial noise removed).

    Each outer trust draw p yields a deterministic per-draw class distribution
    gk = (P(0),P(1),P(2)); eH = E_i[H(gk_i)] (aleatoric), H_total = H(E_i[gk_i])
    (total), MI = H_total - eH (epistemic).
    """
    g = torch.Generator().manual_seed(seed)
    p = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta)).sample((n,))
    gk = torch.stack([(1 - p) * p, p, (1 - p) ** 2], dim=1)   # (n,3)
    grand = gk.mean(0)
    H_total = -(grand.clamp_min(1e-12) * grand.clamp_min(1e-12).log()).sum().item()
    eH = (-(gk.clamp_min(1e-12) * gk.clamp_min(1e-12).log()).sum(-1)).mean().item()
    MI = max(0.0, H_total - eH)
    return MI, eH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Np", type=int, default=200)
    ap.add_argument("--Nm", type=int, default=50)
    ap.add_argument("--prior", type=float, default=1.0)
    args = ap.parse_args()

    print("Per-class theory: P(0)=(1-p)p  P(1)=p  P(2)=(1-p)^2")
    print("%-15s %8s %8s %8s | %8s %8s | %s" %
          ("regime", "u_BALD", "MI", "eH", "thy MI", "thy eH", "P (projected)"))
    for name, (a, b) in REGIMES.items():
        raw = simulate(a, b, args.Np, args.Nm)
        op = augmented_opinion(raw, prior=args.prior, mode="counts")
        tMI, teH = theory(a, b)
        P = " ".join(f"{x:.2f}" for x in op["P"][0].tolist())
        print("%-15s %8.3f %8.4f %8.4f | %8.4f %8.4f | %s" %
              (name, op["u"].item(), op["MI"].item(), op["eH"].item(), tMI, teH, P))
    print("\nExpected ordering: u(aleatoric) << u(epistemic_u);")
    print("trusted/distrusted: tiny entropy everywhere, direction concentrated on class 1 / 2.")


if __name__ == "__main__":
    main()
