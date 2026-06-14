"""Experiment 1: predictive/calibration/OOD + decomposition analysis.

Shows the three SNN signals (u, H, neg_b) are not interchangeable, and that
u collapses as an OOD detector while H and neg_b do not (Prop. 1).

Persistence:
  results/models/run_exp1_<hash>/lin.pt|mcd.pt|edl.pt|snn.pt  (keyed by train params)
  results/cache/run_exp1_<hash>_results.json                   (keyed by all params)

Use --no-cache to retrain and re-run inference from scratch.

Usage:
    python -m snn_eval.run_exp1 --backbone synthetic
    python -m snn_eval.run_exp1 --backbone dinov2_vits14 --dataset cifar10
    python -m snn_eval.run_exp1 --backbone synthetic --no-cache
"""
import argparse
import torch
from . import models, inference, metrics, data, laplace, cache
from . import augmented as aug

_TRAIN_KEYS = ("backbone", "dataset", "ood", "epochs", "d_hidden", "beta_max",
               "K", "d", "n_per_class", "seed")


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


def compute(args):
    torch.manual_seed(args.seed)
    Xtr, ytr, Xte, yte, Xood, K, d_in = get_data(args)
    print(f"d_in={d_in} K={K} train={len(ytr)} test={len(yte)} ood={len(Xood)}")

    train_params = {k: getattr(args, k) for k in _TRAIN_KEYS
                    if hasattr(args, k)}
    dh = args.d_hidden

    # build model instances (needed for both load and train paths)
    lin = models.LinearHead(d_in, dh, K)
    mcd = models.MCDropoutHead(d_in, dh, K, p_drop=0.5)
    edl = models.EDLHead(d_in, dh, K)
    snn = models.SubjectiveHead(d_in, dh, K, prior_a=7, prior_b=3, init_keep=0.7)

    heads = {"lin": lin, "mcd": mcd, "edl": edl, "snn": snn}
    kinds = {"lin": "linear", "mcd": "mc_dropout", "edl": "edl", "snn": "snn"}

    loaded = set() if args.no_cache else cache.load_models("run_exp1", train_params, **heads)
    missing = [n for n in heads if n not in loaded]
    if not missing:
        print("[training skipped — models loaded from cache]")
    else:
        for n in missing:
            heads[n] = build_and_train(kinds[n], d_in, dh, K, Xtr, ytr,
                                       args.device, args.epochs, args.beta_max)
        cache.save_models("run_exp1", train_params,
                          **{n: heads[n] for n in missing})
    lin, mcd, edl, snn = heads["lin"], heads["mcd"], heads["edl"], heads["snn"]

    # inference
    p_id  = inference.deterministic_probs(lin, Xte,  args.device)
    p_ood = inference.deterministic_probs(lin, Xood, args.device)

    pm_id, _ = inference.mc_dropout_probs(mcd, Xte,  T=100, device=args.device)
    pm_ood, _ = inference.mc_dropout_probs(mcd, Xood, T=100, device=args.device)

    pe_id, u_id   = inference.edl_opinion(edl, Xte,  args.device)
    pe_ood, u_ood = inference.edl_opinion(edl, Xood, args.device)

    lap = laplace.LastLayerLaplace(lin, prior_prec=1.0).fit(Xtr, ytr, K, args.device)
    pl_id,  _ = lap.predict(Xte,  T=30, device=args.device)
    pl_ood, _ = lap.predict(Xood, T=30, device=args.device)

    raw_id,  pb_id  = inference.snn_nested_samples(snn, Xte,  args.Np, args.Nm, args.device)
    raw_ood, pb_ood = inference.snn_nested_samples(snn, Xood, args.Np, args.Nm, args.device)
    sig_id  = inference.sl_signals(raw_id,  pb_id)
    sig_ood = inference.sl_signals(raw_ood, pb_ood)
    op_id  = aug.augmented_opinion(aug.raw_to_4d(raw_id,  args.Np, args.Nm), prior=1.0/K)
    op_ood = aug.augmented_opinion(aug.raw_to_4d(raw_ood, args.Np, args.Nm), prior=1.0/K)

    # pre-compute all metrics as floats
    raw_rows = [
        ("Linear",       p_id,            1 - p_id.max(1).values,   1 - p_ood.max(1).values),
        ("MC Dropout",   pm_id,           1 - pm_id.max(1).values,  1 - pm_ood.max(1).values),
        ("EDL",          pe_id,           u_id,                     u_ood),
        ("Laplace",      pl_id,           1 - pl_id.max(1).values,  1 - pl_ood.max(1).values),
    ]
    pacc = sig_id["probs"]
    for sname in ["u", "H", "neg_b"]:
        raw_rows.append((f"SNN ({sname})", pacc, sig_id[sname], sig_ood[sname]))
    raw_rows.append(("SNN (u* aug)", op_id["P"], op_id["u"], op_ood["u"]))
    raw_rows.append(("SNN (H* aug)", op_id["P"], op_id["H"], op_ood["H"]))

    table = []
    for name, probs, s_id, s_ood in raw_rows:
        om = metrics.ood_metrics(s_id, s_ood)
        table.append({
            "name":      name,
            "acc":       metrics.accuracy(probs, yte),
            "nll":       metrics.nll(probs, yte),
            "brier":     metrics.brier(probs, yte, K),
            "ece":       metrics.ece(probs, yte),
            "ood_auroc": om["auroc"],
            "fpr95":     om["fpr95"],
        })

    counts, _ = aug.regime_summary(snn)

    return {
        "table": table,
        "lotv": {
            "id_aleatoric":  float(sig_id["aleatoric"].mean()),
            "id_epistemic":  float(sig_id["epistemic"].mean()),
            "id_total":      float(sig_id["total"].mean()),
            "ood_aleatoric": float(sig_ood["aleatoric"].mean()),
            "ood_epistemic": float(sig_ood["epistemic"].mean()),
            "ood_total":     float(sig_ood["total"].mean()),
            "u_id":          float(sig_id["u"].mean()),
            "u_ood":         float(sig_ood["u"].mean()),
            "u_aug_id":      float(op_id["u"].mean()),
            "u_aug_ood":     float(op_ood["u"].mean()),
        },
        "unit_regimes": dict(counts),
    }


def display(res):
    print("\n%-14s %6s %6s %6s %6s %8s %8s" %
          ("Method", "Acc", "NLL", "Brier", "ECE", "OOD-AUROC", "FPR95"))
    for row in res["table"]:
        print("%-14s %6.3f %6.3f %6.3f %6.3f %8.3f %8.3f" % (
            row["name"], row["acc"], row["nll"], row["brier"],
            row["ece"], row["ood_auroc"], row["fpr95"]))

    L = res["lotv"]
    print("\nLoTV decomposition (SNN):")
    print("  ID  aleatoric=%.4f epistemic=%.4f total=%.4f" %
          (L["id_aleatoric"], L["id_epistemic"], L["id_total"]))
    print("  OOD aleatoric=%.4f epistemic=%.4f total=%.4f" %
          (L["ood_aleatoric"], L["ood_epistemic"], L["ood_total"]))
    print("  -> Prop.1 check: u should track epistemic; OOD epistemic should be LOW")
    print("     mean u: ID=%.4f OOD=%.4f" % (L["u_id"], L["u_ood"]))
    print("  Augmented u* (epistemic share): ID=%.4f OOD=%.4f" %
          (L["u_aug_id"], L["u_aug_ood"]))
    print("  Learned unit regimes:", res["unit_regimes"])


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
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--n_per_class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", dest="no_cache",
                    help="Retrain and re-run inference from scratch")
    args = ap.parse_args()

    all_params = {k: v for k, v in vars(args).items() if k not in ("device", "no_cache")}
    res = None if args.no_cache else cache.load_results("run_exp1", all_params)
    if res is None:
        res = compute(args)
        cache.save_results("run_exp1", all_params, res)
    display(res)


if __name__ == "__main__":
    main()
