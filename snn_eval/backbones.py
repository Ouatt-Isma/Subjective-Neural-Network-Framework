"""SNN-instrumented backbones for full-network (Route 2) training.

The SNN mechanism is channel-wise Beta-Bernoulli dropout: each channel j of a
feature map carries a trust prob p_j ~ Beta(alpha_j, beta_j); a Bernoulli mask
z_j gates the whole channel. Kumaraswamy reparam + Concrete relaxation at train,
hard Bernoulli at inference. The same layer type also implements plain
Dropout2d ('dropout') and identity ('none') so baselines share the architecture.

Nested-sampling inference needs the trust prob fixed across the Nm masks of one
Beta sample; `sample_trust`/`clear_trust` cache p for that purpose.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def softplus_inv(y):
    return math.log(math.expm1(y))


class StochasticDrop2d(nn.Module):
    def __init__(self, channels, kind="snn", p_drop=0.1,
                 prior_a=7.0, prior_b=3.0, init_keep=0.9, tau=0.5):
        super().__init__()
        self.kind = kind
        self.channels = channels
        self.p_drop = p_drop
        self.tau = tau
        self.force_sample = False
        self._cached_p = None
        if kind == "snn":
            self.register_buffer("prior_a", torch.tensor(float(prior_a)))
            self.register_buffer("prior_b", torch.tensor(float(prior_b)))
            a0 = init_keep * 6.0
            b0 = (1 - init_keep) * 6.0
            self.alpha_free = nn.Parameter(torch.full((channels,), softplus_inv(a0)))
            self.beta_free = nn.Parameter(torch.full((channels,), softplus_inv(b0)))

    # --- SNN helpers ---
    def alpha_beta(self):
        return F.softplus(self.alpha_free) + EPS, F.softplus(self.beta_free) + EPS

    def expected_keep(self):
        a, b = self.alpha_beta()
        return a / (a + b)

    def sample_p(self, B, device):
        a, b = self.alpha_beta()
        u = torch.rand(B, self.channels, device=device).clamp(EPS, 1 - EPS)
        return (1 - u.pow(1.0 / b)).pow(1.0 / a).clamp(EPS, 1 - EPS)

    def sample_mask(self, p, hard):
        if hard:
            return torch.bernoulli(p)
        u = torch.rand_like(p).clamp(EPS, 1 - EPS)
        logit = torch.log(p) - torch.log(1 - p) + torch.log(u) - torch.log(1 - u)
        return torch.sigmoid(logit / self.tau)

    def sample_trust(self, B, device):
        if self.kind == "snn":
            self._cached_p = self.sample_p(B, device)

    def clear_trust(self):
        self._cached_p = None

    def kl(self):
        if self.kind != "snn":
            return torch.zeros((), device=self.alpha_free.device) if hasattr(self, "alpha_free") else 0.0
        a, b = self.alpha_beta()
        a0, b0 = self.prior_a, self.prior_b
        lbeta = lambda x, y: torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
        kl = (lbeta(a0, b0) - lbeta(a, b)
              + (a - a0) * torch.digamma(a)
              + (b - b0) * torch.digamma(b)
              - (a + b - a0 - b0) * torch.digamma(a + b))
        return kl.sum()

    def forward(self, x):
        if self.kind == "none":
            return x
        B = x.size(0)
        if self.kind == "dropout":
            active = self.training or self.force_sample
            return F.dropout2d(x, self.p_drop, training=active)
        # snn
        if self._cached_p is not None:                      # nested inference: hard mask
            z = self.sample_mask(self._cached_p, hard=True)
            return x * z.view(B, self.channels, 1, 1)
        if self.training or self.force_sample:              # train: fresh p + concrete mask
            p = self.sample_p(B, x.device)
            z = self.sample_mask(p, hard=not self.training)
            return x * z.view(B, self.channels, 1, 1)
        return x * self.expected_keep().view(1, self.channels, 1, 1)  # deterministic pass


class _SNNNet(nn.Module):
    """Mixin: KL collection and trust caching across all StochasticDrop2d layers."""

    def _drops(self):
        return [m for m in self.modules() if isinstance(m, StochasticDrop2d) and m.kind == "snn"]

    def kl(self):
        ds = self._drops()
        if not ds:
            return torch.zeros((), device=next(self.parameters()).device)
        return sum(d.kl() for d in ds)

    def set_sampling(self, flag):
        for m in self.modules():
            if isinstance(m, StochasticDrop2d):
                m.force_sample = flag

    def sample_all_trust(self, B, device):
        for d in self._drops():
            d.sample_trust(B, device)

    def clear_all_trust(self):
        for d in self._drops():
            d.clear_trust()

    def keep_stats(self):
        ds = self._drops()
        if not ds:
            return (float("nan"), float("nan"))
        ek = torch.cat([d.expected_keep().detach() for d in ds])
        return (ek.mean().item(), ek.std().item())


# ----------------------------------------------------------------------
# CIFAR ResNet-18
# ----------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, out, stride, drop_kind, drop_kw):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.short = nn.Sequential()
        if stride != 1 or inp != out:
            self.short = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False),
                                       nn.BatchNorm2d(out))
        self.drop = StochasticDrop2d(out, kind=drop_kind, **drop_kw)

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        o = F.relu(o + self.short(x))
        return self.drop(o)


class ResNet18(_SNNNet):
    def __init__(self, n_classes=10, drop_kind="snn", drop_kw=None):
        super().__init__()
        drop_kw = drop_kw or {}
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        cfg = [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]
        layers, inp = [], 64
        for out, stride in cfg:
            layers.append(BasicBlock(inp, out, stride, drop_kind, drop_kw))
            inp = out
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x, sample=True):
        if not sample:
            self.set_sampling(False)
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.blocks(o)
        o = F.adaptive_avg_pool2d(o, 1).flatten(1)
        return self.fc(o)


# ----------------------------------------------------------------------
# WRN-28-10
# ----------------------------------------------------------------------
class WRNBlock(nn.Module):
    def __init__(self, inp, out, stride, drop_kind, drop_kw):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.equal = (inp == out and stride == 1)
        self.short = None if self.equal else nn.Conv2d(inp, out, 1, stride, 0, bias=False)
        self.drop = StochasticDrop2d(out, kind=drop_kind, **drop_kw)  # WRN dropout slot

    def forward(self, x):
        o = F.relu(self.bn1(x))
        sc = x if self.equal else self.short(o)
        o = self.conv1(o)
        o = self.drop(o)                       # between the two convs
        o = self.conv2(F.relu(self.bn2(o)))
        return o + sc


class WRN(_SNNNet):
    def __init__(self, depth=28, widen=10, n_classes=10, drop_kind="snn", drop_kw=None):
        super().__init__()
        drop_kw = drop_kw or {}
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        blocks, inp = [], widths[0]
        for g, (w, stride) in enumerate(zip(widths[1:], [1, 2, 2])):
            for i in range(n):
                blocks.append(WRNBlock(inp, w, stride if i == 0 else 1, drop_kind, drop_kw))
                inp = w
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(inp)
        self.fc = nn.Linear(inp, n_classes)

    def forward(self, x, sample=True):
        if not sample:
            self.set_sampling(False)
        o = self.conv1(x)
        o = self.blocks(o)
        o = F.relu(self.bn(o))
        o = F.adaptive_avg_pool2d(o, 1).flatten(1)
        return self.fc(o)


def build_net(arch, n_classes, method, init_keep=0.9, drop_p=0.3,
              prior_a=7.0, prior_b=3.0):
    """method in {snn, mcdropout, edl, deterministic}."""
    if method == "snn":
        kind, kw = "snn", dict(init_keep=init_keep, prior_a=prior_a, prior_b=prior_b)
    elif method == "mcdropout":
        kind, kw = "dropout", dict(p_drop=drop_p)
    else:  # edl / deterministic / ensemble member
        kind, kw = "none", {}
    if arch == "resnet18":
        return ResNet18(n_classes, kind, kw)
    if arch == "wrn2810":
        return WRN(28, 10, n_classes, kind, kw)
    raise ValueError(arch)
