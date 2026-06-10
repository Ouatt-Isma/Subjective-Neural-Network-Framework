"""MNIST small-model evaluation: SNN vs MC Dropout vs EDL (same MLP arch).

1. Calibration table: Acc / NLL / ECE for all three models (+ OOD AUROC).
2. Rotation sweep 0..180 deg: how epistemic (u_e, u*) and aleatoric (u_a)
   uncertainty evolve under covariate shift, per model. For Dropout we compute
   the same LoTV split from its T samples; for EDL u = K/S.

Usage:
    python -m snn_eval.run_mnist                 # real MNIST + FashionMNIST OOD
    python -m snn_eval.run_mnist --synthetic     # no-download code-path check
"""
import argparse, math
import torch
import torch.nn.functional as F
from . import models, inference, metrics
from . import augmented as aug


# ---------------- data (kept as images for rotation) ----------------
def load_mnist(root="./data", ntr=20000, nte=3000):
    import torchvision as tv, torchvision.transforms as T
    tf = T.ToTensor()
    tr = tv.datasets.MNIST(root, train=True, download=True, transform=tf)
    te = tv.datasets.MNIST(root, train=False, download=True, transform=tf)
    ood = tv.datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    def imgs(ds, n):
        X = torch.stack([ds[i][0] for i in range(min(n, len(ds)))])
        y = torch.tensor([ds[i][1] for i in range(min(n, len(ds)))])
        return X, y
    return imgs(tr, ntr), imgs(te, nte), imgs(ood, nte)[0]


def load_synthetic(ntr=3000, nte=600):
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(784, 10, generator=g)
    def mk(n, seed):
        gg = torch.Generator().manual_seed(seed)
        X = torch.rand(n, 1, 28, 28, generator=gg)
        y = (X.flatten(1) @ proj).argmax(1)
        return X, y
    (Xtr, ytr), (Xte, yte) = mk(ntr, 1), mk(nte, 2)
    return (Xtr, ytr), (Xte, yte), torch.rand(nte, 1, 28, 28) * 2

NORM = (0.1307, 0.3081)
def flat(X):  # normalize + flatten
    return ((X - NORM[0]) / NORM[1]).flatten(1)


def rotate_batch(X, angle_deg):
    """Rotate (N,1,28,28) images by angle via affine grid_sample."""
    a = math.radians(angle_deg)
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                          [math.sin(a),  math.cos(a), 0.0]]).unsqueeze(0).repeat(len(X), 1, 1)
    grid = F.affine_grid(theta, X.shape, align_corners=False)
    return F.grid_sample(X, grid, align_corners=False, padding_mode="zeros")


# ---------------- LoTV split for MC Dropout samples ----------------
def lotv_from_samples(samples):
    """samples (B,T,K) -> dict(H, epi, alea) using the trace LoTV."""
    mean = samples.mean(1)
    H = -(mean.clamp_min(1e-12) * mean.clamp_min(1e-12).log()).sum(-1)
    epi = ((samples - mean.unsqueeze(1)) ** 2).sum(-1).mean(1)
    alea = (1 - (samples ** 2).sum(-1)).mean(1)
    return dict(probs=mean, H=H, epi=epi, alea=alea)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--d_hidden", type=int, default=256)
    ap.add_argument("--beta_max", type=float, default=10.0)
    ap.add_argument("--Np", type=int, default=10)
    ap.add_argument("--Nm", type=int, default=10)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--rot_step", type=int, default=15)
    ap.add_argument("--rot_n", type=int, default=1000, help="test digits per angle")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    loader = load_synthetic if args.synthetic else load_mnist
    (Xtr_i, ytr), (Xte_i, yte), Xood_i = loader()
    Xtr, Xte, Xood = flat(Xtr_i), flat(Xte_i), flat(Xood_i)
    K, d = 10, 784
    print(f"train={len(ytr)} test={len(yte)} hidden={args.d_hidden}")

    # ---------------- train the three models ----------------
    print("\n[training SNN]")
    snn = models.SubjectiveHead(d, args.d_hidden, K, prior_a=7, prior_b=3, init_keep=0.7)
    snn = models.train_head(snn, Xtr, ytr, K, epochs=args.epochs, is_snn=True,
                            beta_max=args.beta_max, device=args.device, verbose=True)
    print("[training MC Dropout]")
    mcd = models.MCDropoutHead(d, args.d_hidden, K, p_drop=0.5)
    mcd = models.train_head(mcd, Xtr, ytr, K, epochs=args.epochs, device=args.device,
                            verbose=True)
    print("[training EDL]")
    edl = models.EDLHead(d, args.d_hidden, K)
    edl = models.train_head(edl, Xtr, ytr, K, epochs=max(args.epochs, 25),
                            is_edl=True, device=args.device, verbose=True)

    # ---------------- calibration table ----------------
    raw_id, pb_id = inference.snn_nested_samples(snn, Xte, args.Np, args.Nm, args.device)
    raw_o, pb_o = inference.snn_nested_samples(snn, Xood, args.Np, args.Nm, args.device)
    op_id = aug.augmented_opinion(aug.raw_to_4d(raw_id, args.Np, args.Nm), prior=1.0/K)
    op_o = aug.augmented_opinion(aug.raw_to_4d(raw_o, args.Np, args.Nm), prior=1.0/K)
    opS_id = aug.augmented_opinion(aug.raw_to_4d(raw_id, args.Np, args.Nm),
                                   prior=1.0/K, mode="soft")
    opS_o = aug.augmented_opinion(aug.raw_to_4d(raw_o, args.Np, args.Nm),
                                  prior=1.0/K, mode="soft")
    sig_id = inference.sl_signals(raw_id, pb_id)
    pm, _ = inference.mc_dropout_probs(mcd, Xte, T=args.T, device=args.device)
    pm_o, _ = inference.mc_dropout_probs(mcd, Xood, T=args.T, device=args.device)
    pe, ue = inference.edl_opinion(edl, Xte, args.device)
    pe_o, ue_o = inference.edl_opinion(edl, Xood, args.device)

    print("\n%-18s %6s %6s %6s %9s" % ("Model", "Acc", "NLL", "ECE", "OOD-AUROC"))
    def row(name, probs, s_id, s_ood):
        om = metrics.ood_metrics(s_id, s_ood)
        print("%-18s %6.3f %6.3f %6.3f %9.3f" % (name,
              metrics.accuracy(probs, yte), metrics.nll(probs, yte),
              metrics.ece(probs, yte), om["auroc"]))
    row("MC Dropout", pm, 1 - pm.max(1).values, 1 - pm_o.max(1).values)
    row("EDL", pe, ue, ue_o)
    row("SNN (mean, H)", sig_id["probs"], sig_id["H"],
        inference.sl_signals(raw_o, pb_o)["H"])
    row("SNN (u* counts)", sig_id["probs"], op_id["u"], op_o["u"])
    row("SNN (aug P soft)", opS_id["P"], opS_id["u"], opS_o["u"])

    # ---------------- rotation sweep ----------------
    n = min(args.rot_n, len(yte))
    Xr_base, yr = Xte_i[:n], yte[:n]
    angles = list(range(0, 181, args.rot_step))
    print("\nRotation sweep (means over %d digits)" % n)
    print("%5s | %5s  %-7s %-7s %-7s %-6s | %-6s %-7s %-7s | %-6s %-6s" % (
        "deg", "acc", "SNN u*", "SNN u_e", "SNN u_a", "SNN H",
        "MCD H", "MCD epi", "MCD ale", "EDL u", "EDL H"))
    rows = []
    for ang in angles:
        Xa = flat(rotate_batch(Xr_base, ang))
        raw_a, _ = inference.snn_nested_samples(snn, Xa, args.Np, args.Nm, args.device)
        op = aug.augmented_opinion(aug.raw_to_4d(raw_a, args.Np, args.Nm), prior=1.0/K)
        acc = metrics.accuracy(op["P"], yr)
        pm_a, sm_a = inference.mc_dropout_probs(mcd, Xa, T=max(20, args.T // 5),
                                                device=args.device)
        lm = lotv_from_samples(sm_a)
        pe_a, ue_a = inference.edl_opinion(edl, Xa, args.device)
        He = -(pe_a.clamp_min(1e-12) * pe_a.clamp_min(1e-12).log()).sum(-1)
        r = [ang, acc, op["u"].mean().item(), op["u_e"].mean().item(),
             op["u_a"].mean().item(), op["H"].mean().item(),
             lm["H"].mean().item(), lm["epi"].mean().item(), lm["alea"].mean().item(),
             ue_a.mean().item(), He.mean().item()]
        rows.append(r)
        print("%5d | %5.3f  %-7.3f %-7.4f %-7.3f %-6.3f | %-6.3f %-7.4f %-7.3f | %-6.3f %-6.3f" % tuple(r))

    # CSV + optional plot
    import csv, os
    os.makedirs("results", exist_ok=True)
    with open("results/rotation_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deg", "acc", "snn_u_star", "snn_u_e", "snn_u_a", "snn_H",
                    "mcd_H", "mcd_epi", "mcd_alea", "edl_u", "edl_H"])
        w.writerows(rows)
    print("saved results/rotation_sweep.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        A = np.array(rows)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].plot(A[:, 0], A[:, 1], "k-o", ms=3); ax[0].set_title("Accuracy vs rotation")
        ax[1].plot(A[:, 0], A[:, 2], "-o", ms=3, label="u* (epi share)")
        ax[1].plot(A[:, 0], A[:, 3] / max(A[:, 3].max(), 1e-9), "-s", ms=3, label="u_e (norm)")
        ax[1].plot(A[:, 0], A[:, 4] / max(A[:, 4].max(), 1e-9), "-^", ms=3, label="u_a (norm)")
        ax[1].set_title("SNN decomposition"); ax[1].legend(fontsize=8)
        ax[2].plot(A[:, 0], A[:, 5], "-o", ms=3, label="SNN H")
        ax[2].plot(A[:, 0], A[:, 6], "-s", ms=3, label="MCD H")
        ax[2].plot(A[:, 0], A[:, 10], "-^", ms=3, label="EDL H")
        ax[2].set_title("Entropy comparison"); ax[2].legend(fontsize=8)
        for a in ax: a.set_xlabel("rotation (deg)")
        fig.tight_layout(); fig.savefig("results/rotation_sweep.png", dpi=140)
        print("saved results/rotation_sweep.png")
    except ImportError:
        print("(matplotlib not installed; skipped plot)")


if __name__ == "__main__":
    main()
