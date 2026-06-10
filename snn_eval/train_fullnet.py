"""Train SNN / baselines as FULL networks (Route 2) with a modern recipe.

Examples:
    # quick CPU sanity check, no downloads (random data):
    python -m snn_eval.train_fullnet --arch resnet18 --method snn --dataset synthetic --smoke

    # real runs (need torchvision + internet on first use):
    python -m snn_eval.train_fullnet --arch resnet18  --method snn --dataset cifar10  --epochs 200
    python -m snn_eval.train_fullnet --arch wrn2810   --method snn --dataset cifar100 --epochs 200
    python -m snn_eval.train_fullnet --arch resnet18  --method ensemble --ensemble_size 5

Baselines: deterministic, mcdropout, edl, ensemble. SNGP/DDU need spectral
normalisation in the backbone and are out of scope for this trainer.
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbones, fullnet_inference as fi, inference as inf, metrics
from .models import edl_loss

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
class Cutout:
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1:]
        y, x = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
        y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
        x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


def get_loaders(args):
    if args.dataset == "synthetic" or args.smoke:
        return _synthetic_loaders(args)
    import torchvision as tv
    import torchvision.transforms as T
    tf_tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD), Cutout(16)])
    tf_te = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    DS = {"cifar10": tv.datasets.CIFAR10, "cifar100": tv.datasets.CIFAR100}[args.dataset]
    tr = DS(args.data_root, train=True, download=True, transform=tf_tr)
    te = DS(args.data_root, train=False, download=True, transform=tf_te)
    ood = tv.datasets.SVHN(args.data_root, split="test", download=True, transform=tf_te)
    n_classes = 100 if args.dataset == "cifar100" else 10
    if args.subset:
        tr = torch.utils.data.Subset(tr, list(range(min(args.subset, len(tr)))))
    nw = args.num_workers
    mk = lambda d, s: torch.utils.data.DataLoader(d, args.bs, shuffle=s, num_workers=nw,
                                                  persistent_workers=(nw > 0))
    return mk(tr, True), mk(te, False), mk(ood, False), n_classes


def _synthetic_loaders(args):
    """Random images with a learnable signal; for code-path testing only."""
    n_classes, n = 10, 512 if args.smoke else 4096
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(3 * 32 * 32, n_classes, generator=g)

    def make(m, seed):
        gg = torch.Generator().manual_seed(seed)
        X = torch.randn(m, 3, 32, 32, generator=gg)
        y = (X.flatten(1) @ proj).argmax(1)
        return torch.utils.data.TensorDataset(X, y)

    mk = lambda d, s: torch.utils.data.DataLoader(d, args.bs, shuffle=s)
    ood = torch.utils.data.TensorDataset(torch.randn(256, 3, 32, 32) * 2.5,
                                         torch.zeros(256, dtype=torch.long))
    return mk(make(n, 1), True), mk(make(n // 4, 2), False), mk(ood, False), n_classes


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_one(net, train_loader, n_classes, args, is_snn, is_edl):
    dev = args.device
    net.to(dev).train()
    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_train = len(train_loader.dataset)
    total_steps = args.epochs * len(train_loader)
    step = 0
    t0 = time.time()
    print(f"  training: {total_steps} steps ({len(train_loader)} steps/epoch x {args.epochs} epochs)")
    for ep in range(args.epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = net(xb, sample=True) if is_snn else net(xb)
            if is_edl:
                eta = F.softplus(logits) + 1.0
                loss = edl_loss(eta, yb, n_classes, lam=min(1.0, ep / max(1, args.epochs // 2)))
            else:
                loss = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
                if is_snn:
                    beta = args.beta_max * min(1.0, step / max(1, args.warmup_frac * total_steps))
                    loss = loss + (beta / n_train) * net.kl()
            loss.backward()
            opt.step()
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                sps = step / el
                eta_min = (total_steps - step) / sps / 60
                print(f"    step {step}/{total_steps} loss={loss.item():.3f} "
                      f"{sps:.1f} steps/s  ETA {eta_min:.0f} min", flush=True)
        sched.step()
        if is_snn and (ep % max(1, args.epochs // 5) == 0):
            m, s = net.keep_stats()
            print(f"  ep{ep} loss={loss.item():.3f} E[keep] mean={m:.3f} std={s:.3f}")
    net.eval()
    return net


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(method, net, te_loader, ood_loader, n_classes, args):
    dev = args.device
    if method == "snn":
        raw, pb, y = fi.snn_nested(net, te_loader, args.Np, args.Nm, dev)
        raw_o, pb_o, _ = fi.snn_nested(net, ood_loader, args.Np, args.Nm, dev)
        sig, sig_o = inf.sl_signals(raw, pb), inf.sl_signals(raw_o, pb_o)
        probs = sig["probs"]
        _print_row("SNN (H)", probs, y, n_classes, sig["H"], sig_o["H"])
        _print_row("SNN (neg_b)", probs, y, n_classes, sig["neg_b"], sig_o["neg_b"])
        _print_row("SNN (u)", probs, y, n_classes, sig["u"], sig_o["u"])
        print("  LoTV  ID: alea=%.4f epi=%.4f | OOD: alea=%.4f epi=%.4f | u ID=%.3f OOD=%.3f" % (
            sig["aleatoric"].mean(), sig["epistemic"].mean(),
            sig_o["aleatoric"].mean(), sig_o["epistemic"].mean(),
            sig["u"].mean(), sig_o["u"].mean()))
        return
    if method == "edl":
        probs, u, y = fi.edl_opinion(net, te_loader, dev)
        _, u_o, _ = fi.edl_opinion(net, ood_loader, dev)
        _print_row("EDL", probs, y, n_classes, u, u_o)
        return
    if method == "mcdropout":
        probs, y = fi.mc_dropout(net, te_loader, args.T, dev)
        probs_o, _ = fi.mc_dropout(net, ood_loader, args.T, dev)
    else:  # deterministic / ensemble handled by caller passing probs
        probs, y = fi.det_probs(net, te_loader, dev)
        probs_o, _ = fi.det_probs(net, ood_loader, dev)
    _print_row(method, probs, y, n_classes, 1 - probs.max(1).values, 1 - probs_o.max(1).values)


def _print_row(name, probs, y, K, s_id, s_ood):
    om = metrics.ood_metrics(s_id, s_ood)
    print("%-14s acc=%.3f nll=%.3f brier=%.3f ece=%.3f | OOD-AUROC=%.3f FPR95=%.3f" % (
        name, metrics.accuracy(probs, y), metrics.nll(probs, y),
        metrics.brier(probs, y, K), metrics.ece(probs, y), om["auroc"], om["fpr95"]))


def run_ensemble(args, train_loader, te_loader, ood_loader, n_classes):
    dev = args.device
    probs_te, probs_o, y = 0, 0, None
    for k in range(args.ensemble_size):
        torch.manual_seed(args.seed + k)
        net = backbones.build_net(args.arch, n_classes, "deterministic")
        net = train_one(net, train_loader, n_classes, args, is_snn=False, is_edl=False)
        p, y = fi.det_probs(net, te_loader, dev)
        po, _ = fi.det_probs(net, ood_loader, dev)
        probs_te = probs_te + p / args.ensemble_size
        probs_o = probs_o + po / args.ensemble_size
    _print_row("DeepEnsemble", probs_te, y, n_classes,
               1 - probs_te.max(1).values, 1 - probs_o.max(1).values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "wrn2810"], default="resnet18")
    ap.add_argument("--method", choices=["snn", "mcdropout", "edl", "deterministic", "ensemble"],
                    default="snn")
    ap.add_argument("--dataset", default="cifar10")  # or cifar100 / synthetic
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--beta_max", type=float, default=1e-2)   # KL weight for SNN (tune!)
    ap.add_argument("--warmup_frac", type=float, default=0.3)
    ap.add_argument("--init_keep", type=float, default=0.9)
    ap.add_argument("--drop_p", type=float, default=0.3)
    ap.add_argument("--Np", type=int, default=10)
    ap.add_argument("--Nm", type=int, default=10)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--ensemble_size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)  # 0 = safe on Windows/CPU
    ap.add_argument("--subset", type=int, default=0)        # limit train size for quick runs
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
    torch.manual_seed(args.seed)

    train_loader, te_loader, ood_loader, n_classes = get_loaders(args)
    print(f"arch={args.arch} method={args.method} dataset={args.dataset} "
          f"K={n_classes} device={args.device} epochs={args.epochs}")
    if args.device == "cpu" and args.dataset not in ("synthetic",) and not args.smoke:
        print("  WARNING: training a full network from scratch on CPU is impractical "
              "(days for 200 epochs).")
        print("  -> Use --device cuda on a GPU, OR for a CPU laptop use the frozen-feature")
        print("     route instead: python -m snn_eval.run_exp1 --backbone dinov2_vits14")
        print("  -> For a quick CPU sanity check here, add: --subset 2000 --epochs 3 --num_workers 0")

    if args.method == "ensemble":
        run_ensemble(args, train_loader, te_loader, ood_loader, n_classes)
        return

    net = backbones.build_net(args.arch, n_classes, args.method,
                              init_keep=args.init_keep, drop_p=args.drop_p)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params={n_params/1e6:.2f}M")
    net = train_one(net, train_loader, n_classes, args,
                    is_snn=(args.method == "snn"), is_edl=(args.method == "edl"))
    evaluate(args.method, net, te_loader, ood_loader, n_classes, args)


if __name__ == "__main__":
    main()
