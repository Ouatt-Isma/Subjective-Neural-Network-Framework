"""Data: synthetic feature generator (runs anywhere) + real frozen-backbone hook.

The synthetic generator mimics post-LayerNorm penultimate features: class
clusters around random orthogonalish prototypes of fixed magnitude, with OOD
generated from a novel prototype orthogonal to all training prototypes (same
expected norm, so a linear softmax cannot detect it by norm alone).
"""
import torch
import numpy as np


def _orthonormal_prototypes(K, d, seed):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(K + 1, d, generator=g)  # K classes + 1 OOD prototype
    Q, _ = torch.linalg.qr(M.T)             # (d, K+1) orthonormal columns
    protos = Q.T[:K + 1]                    # (K+1, d)
    return protos * (d ** 0.5)              # fixed magnitude


def make_synthetic(n_per_class=400, K=4, d=768, noise=0.6, seed=0, ood_n=400):
    """Return (Xtr, ytr, Xte, yte, Xood)."""
    protos = _orthonormal_prototypes(K, d, seed)
    g = torch.Generator().manual_seed(seed + 1)

    def cluster(proto, n):
        return proto.unsqueeze(0) + noise * torch.randn(n, d, generator=g)

    Xtr, ytr, Xte, yte = [], [], [], []
    for k in range(K):
        Xtr.append(cluster(protos[k], n_per_class)); ytr += [k] * n_per_class
        Xte.append(cluster(protos[k], n_per_class // 2)); yte += [k] * (n_per_class // 2)
    Xood = cluster(protos[K], ood_n)  # orthogonal novel prototype
    return (torch.cat(Xtr), torch.tensor(ytr),
            torch.cat(Xte), torch.tensor(yte), Xood, protos)


def make_regime_mixture(protos, K=4, n_each=300, noise=0.6, seed=0):
    """easy / split / diffuse regimes (Experiment 2)."""
    g = torch.Generator().manual_seed(seed + 7)
    d = protos.shape[1]
    # easy: single-prototype clusters
    easy_x, easy_y = [], []
    for k in range(K):
        easy_x.append(protos[k] + noise * torch.randn(n_each // K + 1, d, generator=g))
        easy_y += [k] * (n_each // K + 1)
    easy_x = torch.cat(easy_x)[:n_each]; easy_y = torch.tensor(easy_y)[:n_each]
    # split: equal mixtures of two prototypes (ambiguous between two confident answers)
    split_x, split_y = [], []
    for _ in range(n_each):
        a, b = np.random.RandomState(seed + _).choice(K, 2, replace=False)
        mix = 0.5 * protos[a] + 0.5 * protos[b]
        split_x.append(mix + noise * torch.randn(d, generator=g))
        split_y.append(int(a))  # nominal label
    split_x = torch.stack(split_x); split_y = torch.tensor(split_y)
    # diffuse: near centroid of all prototypes + extra noise (no decisive evidence)
    centroid = protos[:K].mean(0)
    diffuse_x = centroid + 1.6 * noise * torch.randn(n_each, d, generator=g)
    diffuse_y = torch.randint(0, K, (n_each,), generator=g)
    X = torch.cat([easy_x, split_x, diffuse_x])
    y = torch.cat([easy_y, split_y, diffuse_y])
    regime = (["easy"] * len(easy_y) + ["split"] * len(split_y) +
              ["diffuse"] * len(diffuse_y))
    return X, y, regime


def rotate_labels(y, K):
    """Adversarial source: deterministic label rotation y' = (y+1) mod K."""
    return (y + 1) % K


# ----------------------------------------------------------------------
# Real frozen-backbone feature extraction (used on the user's machine)
# ----------------------------------------------------------------------
def extract_features(backbone="dinov2_vits14", dataset="cifar10", split="train",
                     device="cpu", batch_size=128, root="./data", cache=None):
    """Extract frozen penultimate features for a torchvision dataset.

    Requires internet on first run to download weights/data. Returns (X, y).
    backbone: 'dinov2_vits14' (torch.hub) or a timm model name, or 'clip'.
    """
    import os
    if cache and os.path.exists(cache):
        d = torch.load(cache)
        return d["X"], d["y"]
    import torchvision as tv
    import torchvision.transforms as T

    if backbone.startswith("dinov2"):
        model = torch.hub.load("facebookresearch/dinov2", backbone)
        tfm = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        embed = lambda m, x: m(x)  # CLS token
    else:
        import timm
        model = timm.create_model(backbone, pretrained=True, num_classes=0)
        cfg = timm.data.resolve_data_config({}, model=model)
        tfm = timm.data.create_transform(**cfg)
        embed = lambda m, x: m(x)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    is_train = split == "train"
    if dataset == "cifar10":
        ds = tv.datasets.CIFAR10(root, train=is_train, download=True, transform=tfm)
    elif dataset == "cifar100":
        ds = tv.datasets.CIFAR100(root, train=is_train, download=True, transform=tfm)
    elif dataset == "svhn":
        ds = tv.datasets.SVHN(root, split="train" if is_train else "test",
                              download=True, transform=tfm)
    else:
        raise ValueError(dataset)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=2)
    Xs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            f = embed(model, xb.to(device)).cpu()
            Xs.append(f); ys.append(torch.as_tensor(yb))
    X = torch.cat(Xs); y = torch.cat(ys)
    if cache:
        torch.save({"X": X, "y": y}, cache)
    return X, y
