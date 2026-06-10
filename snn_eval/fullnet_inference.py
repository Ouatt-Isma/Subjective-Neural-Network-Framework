"""Inference over full networks (Route 2), reusing inference.sl_signals/metrics."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def _collect(net, loader, fn, device):
    net.eval().to(device)
    out, ys = [], []
    for xb, yb in loader:
        out.append(fn(xb.to(device)).cpu())
        ys.append(torch.as_tensor(yb))
    return torch.cat(out), torch.cat(ys)


@torch.no_grad()
def det_probs(net, loader, device="cpu"):
    return _collect(net, loader, lambda x: F.softmax(net(x, sample=False), dim=1), device)


@torch.no_grad()
def edl_opinion(net, loader, device="cpu"):
    def fn(x):
        eta = F.softplus(net(x, sample=False)) + 1.0
        S = eta.sum(1, keepdim=True)
        return torch.cat([eta / S, (eta.shape[1] / S)], dim=1)  # probs ++ u
    cat, ys = _collect(net, loader, fn, device)
    K = cat.shape[1] - 1
    return cat[:, :K], cat[:, K], ys


@torch.no_grad()
def mc_dropout(net, loader, T=100, device="cpu"):
    net.eval().to(device)
    net.set_sampling(True)
    means, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        s = torch.stack([F.softmax(net(xb), dim=1) for _ in range(T)], dim=1)  # (B,T,K)
        means.append(s.mean(1).cpu())
        ys.append(torch.as_tensor(yb))
    net.set_sampling(False)
    return torch.cat(means), torch.cat(ys)


@torch.no_grad()
def snn_nested(net, loader, Np=10, Nm=10, device="cpu"):
    """Return raw (N, Np*Nm, K), per_beta (N, Np, K), labels (N,)."""
    net.eval().to(device)
    net.set_sampling(True)
    raw_all, pb_all, ys = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        B = xb.size(0)
        raw_i, pb_i = [], []
        for _ in range(Np):
            net.sample_all_trust(B, device)
            mj = torch.stack([F.softmax(net(xb), dim=1) for _ in range(Nm)], dim=1)  # (B,Nm,K)
            raw_i.append(mj)
            pb_i.append(mj.mean(1))
            net.clear_all_trust()
        raw_all.append(torch.cat(raw_i, dim=1).cpu())   # (B, Np*Nm, K)
        pb_all.append(torch.stack(pb_i, dim=1).cpu())    # (B, Np, K)
        ys.append(torch.as_tensor(yb))
    net.set_sampling(False)
    return torch.cat(raw_all), torch.cat(pb_all), torch.cat(ys)
