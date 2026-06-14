"""Experiment 3: SL fusion across reliable and adversarially miscalibrated sources.

Two reliable sources (standard training) + three adversarial sources (labels
rotated y'=(y+1) mod K). Trust-discounted SL fusion should preserve accuracy
where logit averaging collapses.

Persistence:
  results/models/run_exp3_<hash>/source_0.pt … source_4.pt  (keyed by train params)
  results/cache/run_exp3_<hash>_results.json                 (keyed by all params)

Use --no-cache to retrain and re-run from scratch.

Usage: python -m snn_eval.run_exp3
       python -m snn_eval.run_exp3 --no-cache
"""
import argparse
import torch
from . import models, inference, metrics, data, fusion, cache

_TRAIN_KEYS = ("epochs", "d_hidden", "beta_max", "K", "d", "seed")


def _train_source(Xtr, ytr, K, d, d_hidden, epochs, beta_max, device):
    snn = models.SubjectiveHead(d, d_hidden, K, prior_a=1, prior_b=1, init_keep=0.5)
    return models.train_head(snn, Xtr, ytr, K, epochs=epochs, is_snn=True,
                             beta_max=beta_max, warmup_frac=0.15, device=device)


def compute(args):
    torch.manual_seed(args.seed)
    Xtr, ytr, _, _, _, protos = data.make_synthetic(
        n_per_class=400, K=args.K, d=args.d, seed=args.seed)
    Xcal,  ycal,  _ = data.make_regime_mixture(protos, K=args.K, seed=args.seed + 100)
    Xtest, ytest, _ = data.make_regime_mixture(protos, K=args.K, seed=args.seed + 200)

    train_params = {k: getattr(args, k) for k in _TRAIN_KEYS}
    is_adv = [False, False, True, True, True]

    # build model instances for load path
    sources = [models.SubjectiveHead(args.d, args.d_hidden, args.K,
                                     prior_a=1, prior_b=1, init_keep=0.5)
               for _ in is_adv]
    named = {f"source_{i}": m for i, m in enumerate(sources)}

    loaded = set() if args.no_cache else cache.load_models("run_exp3", train_params, **named)
    missing = [n for n in named if n not in loaded]
    if not missing:
        print("[training skipped — models loaded from cache]")
    else:
        for i, (adv, m) in enumerate(zip(is_adv, sources)):
            if f"source_{i}" in loaded:
                continue
            y_use = data.rotate_labels(ytr, args.K) if adv else ytr
            models.train_head(m, Xtr, y_use, args.K, epochs=args.epochs, is_snn=True,
                              beta_max=args.beta_max, warmup_frac=0.15, device=args.device)
        cache.save_models("run_exp3", train_params,
                          **{n: named[n] for n in missing})

    # inference
    trusts, pb_test = [], []
    for m in sources:
        raw_c, pb_c = inference.snn_nested_samples(m, Xcal, args.Np, args.Nm, args.device)
        acc_c = (pb_c.mean(1).argmax(1) == ycal).float().mean().item()
        n = len(ycal)
        trusts.append((acc_c * n + 1) / (n + 2))
        _, pb_t = inference.snn_nested_samples(m, Xtest, args.Np, args.Nm, args.device)
        pb_test.append(pb_t)

    print("source calibration trusts:", [f"{t:.3f}" for t in trusts])
    raw_out = fusion.fuse_eval(pb_test, ytest, trusts, args.K, metrics)

    # fuse_eval already returns floats from metrics.*
    out = {name: {"acc": d["acc"], "nll": d["nll"], "ece": d["ece"]}
           for name, d in raw_out.items()}

    return {"trusts": trusts, "out": out}


def display(res):
    print("source calibration trusts:", [f"{t:.3f}" for t in res["trusts"]])
    print("\n%-16s %6s %6s %6s" % ("Method", "Acc", "NLL", "ECE"))
    for name, d in res["out"].items():
        print("%-16s %6.3f %6.3f %6.3f" % (name, d["acc"], d["nll"], d["ece"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--d_hidden", type=int, default=32)
    ap.add_argument("--beta_max", type=float, default=20.0)
    ap.add_argument("--Np", type=int, default=10)
    ap.add_argument("--Nm", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", dest="no_cache",
                    help="Retrain and re-run inference from scratch")
    args = ap.parse_args()

    all_params = {k: v for k, v in vars(args).items() if k not in ("device", "no_cache")}
    res = None if args.no_cache else cache.load_results("run_exp3", all_params)
    if res is None:
        res = compute(args)
        cache.save_results("run_exp3", all_params, res)
    display(res)


if __name__ == "__main__":
    main()
