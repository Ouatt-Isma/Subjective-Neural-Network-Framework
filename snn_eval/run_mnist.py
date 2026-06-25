"""MNIST small-model evaluation: SNN vs MC Dropout vs EDL.

1. Calibration table: Acc / NLL / ECE for all three models (+ OOD AUROC).
2. Rotation sweep 0..180 deg: how epistemic (u_e, u*) and aleatoric (u_a)
   uncertainty evolve under covariate shift.

Three architectures available via --arch:
  mlp     2-layer MLP (default, LN+256+ReLU+10, flat 784-dim input)
  cnn     LeNet-style 3-conv + 2-pool + FC (image input, same d_hidden bottleneck)
  resnet  4-block tiny ResNet with global avg pool (image input)

Persistence (all under results/):
  models/<exp>_<hash>/snn.pt|mcd.pt|edl.pt   trained weights (keyed by train params)
  cache/<exp>_<hash>_results.json             pre-computed metrics (keyed by all params)

On re-run: loads saved models if training params match; models whose .pt
           file is missing are retrained individually (delete e.g. edl.pt
           to retrain only EDL). Loads saved results if all params match
           (skips inference too).
Use --no-cache to retrain and re-run inference from scratch.

Usage:
    python -m snn_eval.run_mnist                         # MLP, real MNIST
    python -m snn_eval.run_mnist --arch cnn              # CNN, real MNIST
    python -m snn_eval.run_mnist --arch resnet           # ResNet, real MNIST
    python -m snn_eval.run_mnist --synthetic             # MLP, synthetic data
    python -m snn_eval.run_mnist --arch cnn --no-cache   # force full re-run
"""
import argparse, math
import torch
import torch.nn.functional as F
from . import models, inference, metrics, cache
from . import augmented as aug

# params that affect model structure / training (used for model cache key)
_TRAIN_KEYS = ("arch", "epochs", "d_hidden", "beta_max", "seed", "synthetic", "n_train", "label_noise")


# ---------------- data helpers ----------------
def load_mnist(root="./data", ntr=60000, nte=10000):
    import torchvision as tv, torchvision.transforms as T
    tf = T.ToTensor()
    tr  = tv.datasets.MNIST(root, train=True,  download=True, transform=tf)
    te  = tv.datasets.MNIST(root, train=False, download=True, transform=tf)
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


def load_mnist_class_split(id_classes=(0, 1, 2, 3, 4), root="./data", nte=1000):
    """Epistemic OOD probe: train only on id_classes; OOD = unseen-class test images.

    Uses MNIST .data / .targets tensors directly for fast filtering without a
    per-item loop. Returns the same contract as load_mnist so compute() accepts it
    as a loader. K is inferred from yte (= len(id_classes)), not hardcoded.
    Expected: model shows high u_e (MI) on OOD images because it has no training
    signal for those feature regions — trust parameters remain diffuse.
    """
    import torchvision as tv
    tr = tv.datasets.MNIST(root, train=True,  download=True)
    te = tv.datasets.MNIST(root, train=False, download=True)
    id_t = torch.tensor(list(id_classes))

    def masked(ds, mask, n):
        idx = mask.nonzero(as_tuple=True)[0][:n]
        X = ds.data[idx].unsqueeze(1).float() / 255.0   # (N,1,28,28) in [0,1]
        return X, ds.targets[idx]

    id_tr  = torch.isin(tr.targets, id_t)
    id_te  = torch.isin(te.targets, id_t)
    Xtr, ytr  = masked(tr, id_tr,  60000)
    Xte, yte  = masked(te, id_te,  nte)
    Xood, _   = masked(te, ~id_te, nte)
    return (Xtr, ytr), (Xte, yte), Xood


NORM = (0.1307, 0.3081)

def normalize(X):
    """MNIST channel normalization; preserves spatial dims."""
    return (X - NORM[0]) / NORM[1]

def flat(X):
    """Normalize + flatten to (N, 784). Used for MLP arch."""
    return normalize(X).flatten(1)


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


# ---------------- model factory ----------------
def _build_models(args, K):
    """Return (snn, mcd, edl) for the selected arch."""
    arch, dh = args.arch, args.d_hidden
    if arch == "mlp":
        snn = models.SubjectiveHead(784, dh, K, prior_a=7, prior_b=3, init_keep=0.7)
        mcd = models.MCDropoutHead(784, dh, K, p_drop=0.5)
        edl = models.EDLHead(784, dh, K)
    else:
        snn = models.SubjectiveCNN(arch, dh, K, prior_a=7, prior_b=3, init_keep=0.7)
        mcd = models.MCDropoutCNN(arch, dh, K, p_drop=0.5)
        edl = models.EDLCNN(arch, dh, K)
    return snn, mcd, edl


def _prep_inputs(X_img, arch):
    """Convert (N,1,28,28) images to the format expected by the chosen arch."""
    return flat(X_img) if arch == "mlp" else normalize(X_img)


# ---------------- model train-or-load ----------------
def build_or_train_models(args, K, Xtr=None, ytr=None, Xte=None, yte=None, exp_name="run_mnist"):
    """Build SNN/MCD/EDL heads for args.arch; load cached weights where available,
    train the rest on (Xtr, ytr). Returns dict(snn=..., mcd=..., edl=...) of the
    actual trained nn.Module instances — pass Xtr/ytr only if a cache miss is
    possible (e.g. omit them to fetch already-cached models for scoring custom
    probe batches, as compute() doesn't otherwise expose the trained models).
    """
    train_params = {k: getattr(args, k) for k in _TRAIN_KEYS}
    snn, mcd, edl = _build_models(args, K)
    heads = {"snn": snn, "mcd": mcd, "edl": edl}

    loaded = set() if args.no_cache else cache.load_models(exp_name, train_params, **heads)
    missing = [n for n in heads if n not in loaded]
    if not missing:
        print("[training skipped — models loaded from cache]")
    else:
        common = dict(epochs=args.epochs, device=args.device, verbose=True,
                      Xte=Xte, yte=yte)
        if "snn" in missing:
            print("\n[training SNN]")
            models.train_head(snn, Xtr, ytr, K, is_snn=True,
                              beta_max=args.beta_max, **common)
        if "mcd" in missing:
            print("[training MC Dropout]")
            models.train_head(mcd, Xtr, ytr, K, **common)
        if "edl" in missing:
            print("[training EDL]")
            models.train_head(edl, Xtr, ytr, K, is_edl=True, **common)
        cache.save_models(exp_name, train_params,
                          **{n: heads[n] for n in missing})
    return heads


# ---------------- compute / display ----------------
def compute(args, loader=None, exp_name="run_mnist", return_models=False):
    """Train (or reload) models, run inference, return JSON-serialisable results dict.

    `loader` defaults to load_synthetic/load_mnist (picked via args.synthetic) but
    accepts any zero-arg callable returning ((Xtr,ytr),(Xte,yte),Xood) with the same
    shapes (images (N,1,28,28), int labels) — pass a different one to point this same
    pipeline at another MNIST-shaped dataset (e.g. FashionMNIST/KMNIST) without
    touching training/inference/metrics code. `exp_name` namespaces the model cache
    so alternate-dataset runs don't collide with the default MNIST cache entries.
    """
    torch.manual_seed(args.seed)
    loader = loader or (load_synthetic if args.synthetic else load_mnist)
    (Xtr_i, ytr), (Xte_i, yte), Xood_i = loader()
    K = int(torch.cat([ytr, yte]).max().item()) + 1

    n_train = getattr(args, "n_train", 60000)
    if n_train < len(ytr):
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(ytr), generator=g)[:n_train]
        Xtr_i, ytr = Xtr_i[idx], ytr[idx]
    label_noise = getattr(args, "label_noise", 0.0)
    if label_noise > 0.0:
        g = torch.Generator().manual_seed(args.seed + 1)
        flip = torch.rand(len(ytr), generator=g) < label_noise
        ytr = ytr.clone()
        ytr[flip] = torch.randint(0, K, (int(flip.sum().item()),), generator=g)

    Xtr  = _prep_inputs(Xtr_i,  args.arch)
    Xte  = _prep_inputs(Xte_i,  args.arch)
    Xood = _prep_inputs(Xood_i, args.arch)
    print(f"arch={args.arch}  train={len(ytr)}  test={len(yte)}  hidden={args.d_hidden}")

    # --- training (with per-model cache: only missing models are trained) ---
    heads = build_or_train_models(args, K, Xtr, ytr, Xte, yte, exp_name=exp_name)
    snn, mcd, edl = heads["snn"], heads["mcd"], heads["edl"]

    # --- inference ---
    n_te, n_ood = len(yte), Xood.shape[0]
    print(f"\n[inference — {n_te} test / {n_ood} OOD samples]")
    print("  SNN nested sampling (ID)...")
    raw_id, pb_id  = inference.snn_nested_samples(snn, Xte,  args.Np, args.Nm, args.device,
                                                   desc="SNN ID")
    print("  SNN nested sampling (OOD)...")
    raw_o,  pb_o   = inference.snn_nested_samples(snn, Xood, args.Np, args.Nm, args.device,
                                                   desc="SNN OOD")
    op_id  = aug.augmented_opinion(aug.raw_to_4d(raw_id, args.Np, args.Nm), prior=1.0/K)
    op_o   = aug.augmented_opinion(aug.raw_to_4d(raw_o,  args.Np, args.Nm), prior=1.0/K)
    opS_id = aug.augmented_opinion(aug.raw_to_4d(raw_id, args.Np, args.Nm), prior=1.0/K, mode="soft")
    opS_o  = aug.augmented_opinion(aug.raw_to_4d(raw_o,  args.Np, args.Nm), prior=1.0/K, mode="soft")
    sig_id  = inference.sl_signals(raw_id, pb_id)
    sig_ood = inference.sl_signals(raw_o,  pb_o)
    print(f"  MC Dropout (T={args.T}, ID)...")
    pm,   sm_id  = inference.mc_dropout_probs(mcd, Xte,  T=args.T, device=args.device,
                                         desc=f"MCD ID  T={args.T}")
    print(f"  MC Dropout (T={args.T}, OOD)...")
    pm_o, sm_ood = inference.mc_dropout_probs(mcd, Xood, T=args.T, device=args.device,
                                         desc=f"MCD OOD T={args.T}")
    mcd_id  = aug.bald_opinion(sm_id)
    mcd_ood = aug.bald_opinion(sm_ood)
    print("  EDL opinion (ID + OOD)...")
    pe,   ue   = inference.edl_opinion(edl, Xte,  args.device)
    pe_o, ue_o = inference.edl_opinion(edl, Xood, args.device)
    print("  done.")

    # pre-compute table metrics as plain floats (JSON-serialisable)
    raw_rows = [
        ("MC Dropout",       pm,              1 - pm.max(1).values,  1 - pm_o.max(1).values),
        ("MCD (mean, H)",    pm,              mcd_id["H"],            mcd_ood["H"]),
        ("MCD (u* BALD)",    pm,              mcd_id["u"],            mcd_ood["u"]),
        ("EDL",              pe,              ue,                    ue_o),
        ("SNN (mean, H)",    sig_id["probs"], sig_id["H"],           sig_ood["H"]),
        ("SNN (u* counts)",  sig_id["probs"], op_id["u"],            op_o["u"]),
        ("SNN (aug P soft)", opS_id["P"],     opS_id["u"],           opS_o["u"]),
    ]
    table = []
    for name, probs, s_id, s_ood in raw_rows:
        om = metrics.ood_metrics(s_id, s_ood)
        table.append({
            "name":      name,
            "acc":       metrics.accuracy(probs, yte),
            "nll":       metrics.nll(probs, yte),
            "ece":       metrics.ece(probs, yte),
            "ood_auroc": om["auroc"],
        })

    # rotation sweep (rot_rows is already a list of float lists)
    n = min(args.rot_n, len(yte))
    Xr_base, yr = Xte_i[:n], yte[:n]
    angles = list(range(0, 181, args.rot_step))
    print(f"\nRotation sweep ({n} digits, {len(angles)} angles) ...")
    sweep = []
    for ang in angles:
        Xa = _prep_inputs(rotate_batch(Xr_base, ang), args.arch)
        raw_a, _ = inference.snn_nested_samples(snn, Xa, args.Np, args.Nm, args.device)
        op = aug.augmented_opinion(aug.raw_to_4d(raw_a, args.Np, args.Nm), prior=1.0/K)
        acc_a = metrics.accuracy(op["P"], yr)
        pm_a, sm_a = inference.mc_dropout_probs(mcd, Xa, T=max(20, args.T // 5),
                                                device=args.device,
                                                desc=f"MCD {ang:3d}°")
        lm = lotv_from_samples(sm_a)
        mcd_a = aug.bald_opinion(sm_a)
        pe_a, ue_a = inference.edl_opinion(edl, Xa, args.device)
        He = -(pe_a.clamp_min(1e-12) * pe_a.clamp_min(1e-12).log()).sum(-1)
        r = [ang, acc_a,
             op["u"].mean().item(), op["u_e"].mean().item(),
             op["u_a"].mean().item(), op["H"].mean().item(),
             lm["H"].mean().item(), lm["epi"].mean().item(), lm["alea"].mean().item(),
             mcd_a["u"].mean().item(), mcd_a["u_e"].mean().item(), mcd_a["u_a"].mean().item(),
             ue_a.mean().item(), He.mean().item()]
        sweep.append(r)
        print(f"  {ang:3d}°  acc={acc_a:.3f}  SNN u*={r[2]:.3f} u_e={r[3]:.4f}  "
              f"MCD u*={r[9]:.3f} u_e={r[10]:.4f}  H={r[5]:.3f}")

    res = {"arch": args.arch, "n_rot": n, "table": table, "sweep": sweep}
    if return_models:
        return res, heads
    return res


def pixel_noise_probe(heads, Xte_i, yte, args, K, sigmas=(0.0, 0.1, 0.2, 0.4, 0.8)):
    """Inference-time Gaussian pixel noise sweep on fixed trained models (aleatoric probe).

    No retraining: adds N(0,sigma^2) noise to raw [0,1] test images, re-runs inference.
    Expected signature: u_a (eH) rises with sigma; u_e (MI) stays relatively low because
    all trust draws agree that the noisy image is ambiguous (no cross-draw disagreement).
    Returns a list of dicts, one per sigma.
    """
    snn, mcd, edl = heads["snn"], heads["mcd"], heads["edl"]
    rows = []
    for sigma in sigmas:
        g = torch.Generator().manual_seed(42)
        Xn_i = (Xte_i + sigma * torch.randn(*Xte_i.shape, generator=g)).clamp(0.0, 1.0)
        Xn = _prep_inputs(Xn_i, args.arch)
        raw_n, pb_n = inference.snn_nested_samples(snn, Xn, args.Np, args.Nm, args.device)
        op = aug.augmented_opinion(aug.raw_to_4d(raw_n, args.Np, args.Nm), prior=1.0 / K)
        pm_n, sm_n = inference.mc_dropout_probs(mcd, Xn, T=args.T, device=args.device,
                                                 desc=f"pixel_noise σ={sigma:.2f}")
        mcd_n = aug.bald_opinion(sm_n)
        pe_n, ue_n = inference.edl_opinion(edl, Xn, args.device)
        rows.append({
            "sigma":   float(sigma),
            "acc":     metrics.accuracy(op["P"], yte),
            "snn_u":   op["u"].mean().item(),
            "snn_u_e": op["u_e"].mean().item(),
            "snn_u_a": op["u_a"].mean().item(),
            "snn_H":   op["H"].mean().item(),
            "mcd_u":   mcd_n["u"].mean().item(),
            "mcd_u_e": mcd_n["u_e"].mean().item(),
            "mcd_u_a": mcd_n["u_a"].mean().item(),
            "edl_u":   ue_n.mean().item(),
        })
    return rows


def display(res, out_prefix="rotation_sweep"):
    """Print tables, save CSV and plot from a pre-computed results dict.

    `out_prefix` names the CSV/PNG under results/ (default "rotation_sweep",
    matching the CLI) — pass a dataset-specific prefix when calling this for
    more than one dataset so outputs don't overwrite each other. Returns the
    matplotlib Figure (or None if matplotlib is unavailable) so callers such
    as a notebook can display it inline in addition to the saved PNG.
    """
    print(f"\narch={res['arch']}")
    print("\n%-18s %6s %6s %6s %9s" % ("Model", "Acc", "NLL", "ECE", "OOD-AUROC"))
    for row in res["table"]:
        print("%-18s %6.3f %6.3f %6.3f %9.3f" %
              (row["name"], row["acc"], row["nll"], row["ece"], row["ood_auroc"]))

    sweep = res["sweep"]
    print("\nRotation sweep (means over %d digits)" % res["n_rot"])
    print("%5s | %5s  %-7s %-7s %-7s %-6s | %-6s %-7s %-7s | %-6s %-7s %-7s | %-6s %-6s" % (
        "deg", "acc", "SNN u*", "SNN u_e", "SNN u_a", "SNN H",
        "MCD H", "MCD epi", "MCD ale", "MCD u*", "MCD u_e", "MCD u_a", "EDL u", "EDL H"))
    for r in sweep:
        print("%5d | %5.3f  %-7.3f %-7.4f %-7.3f %-6.3f | %-6.3f %-7.4f %-7.3f | "
              "%-6.3f %-7.4f %-7.3f | %-6.3f %-6.3f" % tuple(r))

    import csv, os
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{out_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deg", "acc", "snn_u_star", "snn_u_e", "snn_u_a", "snn_H",
                    "mcd_H", "mcd_epi", "mcd_alea", "mcd_u_star", "mcd_u_e", "mcd_u_a",
                    "edl_u", "edl_H"])
        w.writerows(sweep)
    print(f"saved {csv_path}")
    try:
        import matplotlib
        try:
            from IPython import get_ipython
            in_notebook = get_ipython() is not None
        except ImportError:
            in_notebook = False
        if not in_notebook:
            matplotlib.use("Agg")  # headless CLI runs have no display backend
        import matplotlib.pyplot as plt
        import numpy as np
        A = np.array(sweep)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].plot(A[:, 0], A[:, 1], "k-o", ms=3); ax[0].set_title("Accuracy vs rotation")
        ax[1].plot(A[:, 0], A[:, 2], "-o", ms=3, label="SNN u* (epi share)")
        ax[1].plot(A[:, 0], A[:, 3] / max(A[:, 3].max(), 1e-9), "-s", ms=3, label="SNN u_e (norm)")
        ax[1].plot(A[:, 0], A[:, 4] / max(A[:, 4].max(), 1e-9), "-^", ms=3, label="SNN u_a (norm)")
        ax[1].plot(A[:, 0], A[:, 9], "-D", ms=3, label="MCD u* (BALD)")
        ax[1].set_title("Epistemic-share decomposition"); ax[1].legend(fontsize=7)
        ax[2].plot(A[:, 0], A[:, 5], "-o", ms=3, label="SNN H")
        ax[2].plot(A[:, 0], A[:, 6], "-s", ms=3, label="MCD H")
        ax[2].plot(A[:, 0], A[:, 13], "-^", ms=3, label="EDL H")
        ax[2].set_title("Entropy comparison"); ax[2].legend(fontsize=8)
        for a in ax: a.set_xlabel("rotation (deg)")
        fig.tight_layout()
        png_path = f"results/{out_prefix}.png"
        fig.savefig(png_path, dpi=140)
        print(f"saved {png_path}")
        return fig
    except ImportError:
        print("(matplotlib not installed; skipped plot)")
        return None


def build_argparser():
    """Build the CLI parser. Also used by callers (e.g. notebooks) that want
    a Namespace of defaults without duplicating the argument list — e.g.
    `args = build_argparser().parse_args([]); args.epochs = 3`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp", choices=["mlp", "cnn", "resnet"],
                    help="Model architecture: mlp (flat 784), cnn (LeNet), resnet (tiny ResNet)")
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
    ap.add_argument("--n_train", type=int, default=60000,
                    help="Cap training set size (sub-samples randomly; default=all)")
    ap.add_argument("--label_noise", type=float, default=0.0,
                    help="Fraction of training labels to randomly flip (aleatoric probe)")
    ap.add_argument("--no-cache", action="store_true", dest="no_cache",
                    help="Retrain and re-run inference from scratch")
    return ap


def main():
    args = build_argparser().parse_args()

    all_params   = {k: v for k, v in vars(args).items() if k not in ("device", "no_cache")}
    res = None if args.no_cache else cache.load_results("run_mnist", all_params)
    if res is None:
        res = compute(args)
        cache.save_results("run_mnist", all_params, res)
    display(res)


if __name__ == "__main__":
    main()
