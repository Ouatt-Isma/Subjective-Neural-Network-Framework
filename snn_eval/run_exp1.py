"""Experiment 1: predictive/calibration/OOD + decomposition analysis.

Shows the three SNN signals (u, H, neg_b) are not interchangeable, and that
u collapses as an OOD detector while H and neg_b do not (Prop. 1).

Usage:
    python -m snn_eval.run_exp1 --backbone synthetic
    python -m snn_eval.run_exp1 --backbone dinov2_vits14 --dataset cifar10
"""
import argparse
import torch
from . import models, inference, metrics, data, laplace


def build_and_train(kind, d_in, d_hidden, K, Xtr, ytr, device, epochs, beta_max):
    if kind == "linear":
        m = models.LinearHead(d_in, d_hidden, K)
        return models.train_head(m, Xtr, ytr, K, epochs=epochs, device=device)
    if kind == "mc_dropout":
        m = models.MCDropoutHead(d_in, d_hidden, K, p_drop=0.5)
        return models.train_head(m, Xtr, ytr, K, epochs=epochs, device=device)
    if kind == "edl":
        m = models.EDLHead(d_in, d_hidden, K)
        return models.train_head(m, Xtr, ytr, K, epochs=epochs, is_edl=True, device=device)
    if kind == "snn":
        m = models.SubjectiveHead(d_in, d_hidden, K, prior_a=7, prior_b=3, init_keep=0.7)
        return models.train_head(m, Xtr, ytr, K, epochs=epochs, is_snn=True,
                                 beta_max=beta_max, device=device, verbose=True)
    raise ValueError(kind)


def get_data(args):
    if args.backbone == "synthetic":
        Xtr, ytr, Xte, yte, Xood, _ = data.make_synthetic(
            n_per_class=args.n_per_class, K=args.K, d=args.d, seed=args.seed)
        return Xtr, ytr, Xte, yte, Xood, args.K, args.d
    Xtr, ytr = data.extract_features(args.backbone, args.dataset, "train",
                                     device=args.device, cache=f"{args.dataset}_tr.pt")
    Xte, yte = data.extract_features(args.backbone, args.dataset, "test",
                                     device=args.device, cache=f"{args.dataset}_te.pt")
    Xood, _ = data.extract_features(args.backbone, args.ood, "test",
                                    device=args.device, cache=f"{args.ood}_ood.pt")
    K = int(ytr.max().item()) + 1
    return Xtr, ytr, Xte, yte, Xood, K, Xtr.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="synthetic")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--ood", default="svhn")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--d_hidden", type=int, default=64)
    ap.add_argument("--beta_max", type=float, default=5.0)
    ap.add_argument("--Np", type=int, default=10)
    ap.add_argument("--Nm", type=int, default=10)
    # synthetic config
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--n_per_class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    Xtr, ytr, Xte, yte, Xood, K, d_in = get_data(args)
    print(f"d_in={d_in} K={K} train={len(ytr)} test={len(yte)} ood={len(Xood)}")

    rows = []
    # --- baselines ---
    lin = build_and_train("linear", d_in, args.d_hidden, K, Xtr, ytr, args.device, args.epochs, args.beta_max)
    p_id = inference.deterministic_probs(lin, Xte, args.device)
    p_ood = inference.deterministic_probs(lin, Xood, args.device)
    s_id, s_ood = 1 - p_id.max(1).values, 1 - p_ood.max(1).values
    rows.append(("Linear", p_id, yte, K, s_id, s_ood))

    mcd = build_and_train("mc_dropout", d_in, args.d_hidden, K, Xtr, ytr, args.device, args.epochs, args.beta_max)
    p_id, _ = inference.mc_dropout_probs(mcd, Xte, T=100, device=args.device)
    p_ood, _ = inference.mc_dropout_probs(mcd, Xood, T=100, device=args.device)
    rows.append(("MC Dropout", p_id, yte, K, 1 - p_id.max(1).values, 1 - p_ood.max(1).values))

    edl = build_and_train("edl", d_in, args.d_hidden, K, Xtr, ytr, args.device, args.epochs, args.beta_max)
    p_id, u_id = inference.edl_opinion(edl, Xte, args.device)
    p_ood, u_ood = inference.edl_opinion(edl, Xood, args.device)
    rows.append(("EDL", p_id, yte, K, u_id, u_ood))

    lap = laplace.LastLayerLaplace(lin, prior_prec=1.0).fit(Xtr, ytr, K, args.device)
    p_id, _ = lap.predict(Xte, T=30, device=args.device)
    p_ood, _ = lap.predict(Xood, T=30, device=args.device)
    rows.append(("Laplace", p_id, yte, K, 1 - p_id.max(1).values, 1 - p_ood.max(1).values))

    # --- SNN with three signals ---
    snn = build_and_train("snn", d_in, args.d_hidden, K, Xtr, ytr, args.device, args.epochs, args.beta_max)
    raw_id, pb_id = inference.snn_nested_samples(snn, Xte, args.Np, args.Nm, args.device)
    raw_ood, pb_ood = inference.snn_nested_samples(snn, Xood, args.Np, args.Nm, args.device)
    sig_id = inference.sl_signals(raw_id, pb_id)
    sig_ood = inference.sl_signals(raw_ood, pb_ood)
    pacc = sig_id["probs"]
    for sname in ["u", "H", "neg_b"]:
        rows.append((f"SNN ({sname})", pacc, yte, K, sig_id[sname], sig_ood[sname]))

    # --- Augmented SL opinion (new estimator: LoTV split inside the opinion) ---
    from . import augmented as aug
    op_id = aug.augmented_opinion(aug.raw_to_4d(raw_id, args.Np, args.Nm), prior=1.0/K)
    op_ood = aug.augmented_opinion(aug.raw_to_4d(raw_ood, args.Np, args.Nm), prior=1.0/K)
    rows.append(("SNN (u* aug)", op_id["P"], yte, K, op_id["u"], op_ood["u"]))
    rows.append(("SNN (H* aug)", op_id["P"], yte, K, op_id["H"], op_ood["H"]))

    # --- print table ---
    print("\n%-14s %6s %6s %6s %6s %8s %8s" %
          ("Method", "Acc", "NLL", "Brier", "ECE", "OOD-AUROC", "FPR95"))
    for name, probs, y, K_, s_id, s_ood in rows:
        om = metrics.ood_metrics(s_id, s_ood)
        print("%-14s %6.3f %6.3f %6.3f %6.3f %8.3f %8.3f" % (
            name, metrics.accuracy(probs, y), metrics.nll(probs, y),
            metrics.brier(probs, y, K_), metrics.ece(probs, y),
            om["auroc"], om["fpr95"]))

    # --- LoTV decomposition: ID vs OOD ---
    print("\nLoTV decomposition (SNN):")
    print("  ID  aleatoric=%.4f epistemic=%.4f total=%.4f" % (
        sig_id["aleatoric"].mean(), sig_id["epistemic"].mean(), sig_id["total"].mean()))
    print("  OOD aleatoric=%.4f epistemic=%.4f total=%.4f" % (
        sig_ood["aleatoric"].mean(), sig_ood["epistemic"].mean(), sig_ood["total"].mean()))
    print("  -> Prop.1 check: u should track epistemic; OOD epistemic should be LOW")
    print("     mean u: ID=%.4f OOD=%.4f" % (sig_id["u"].mean(), sig_ood["u"].mean()))
    print("  Augmented u* (epistemic share): ID=%.4f OOD=%.4f" % (
        op_id["u"].mean(), op_ood["u"].mean()))
    counts, _ = aug.regime_summary(snn)
    print("  Learned unit regimes:", dict(counts))


if __name__ == "__main__":
    main()
