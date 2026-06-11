"""Subjective Neural Network head and baseline heads.

All heads share the LN -> Linear -> ReLU -> Linear architecture so that
comparisons isolate the uncertainty mechanism, not backbone/head capacity.
The SNN places Beta-Bernoulli dropout on the bottleneck `h`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

EPS = 1e-6


def softplus_inv(y: float) -> float:
    """Inverse softplus, for initialising free params so softplus(x)=y."""
    return math.log(math.expm1(y))


# ----------------------------------------------------------------------
# Baseline heads
# ----------------------------------------------------------------------
class LinearHead(nn.Module):
    """Deterministic LN-Linear-ReLU-Linear head."""

    def __init__(self, d_in, d_hidden, n_classes, p_drop=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_classes)
        self.p_drop = p_drop

    def forward(self, x, sample=False):
        h = F.relu(self.fc1(self.ln(x)))
        if self.p_drop > 0 and (self.training or sample):
            h = F.dropout(h, p=self.p_drop, training=True)  # MC dropout if sample=True
        return self.fc2(h)


class MCDropoutHead(LinearHead):
    """Same as LinearHead but keeps dropout active at inference when sample=True."""

    def __init__(self, d_in, d_hidden, n_classes, p_drop=0.5):
        super().__init__(d_in, d_hidden, n_classes, p_drop=p_drop)


class EDLHead(nn.Module):
    """Evidential head: outputs evidence e>=0, Dirichlet eta = e + 1 (Sensoy 2018)."""

    def __init__(self, d_in, d_hidden, n_classes):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_classes)
        self.n_classes = n_classes

    def evidence(self, x):
        h = F.relu(self.fc1(self.ln(x)))
        return F.softplus(self.fc2(h))  # non-negative evidence

    def forward(self, x):
        return self.evidence(x) + 1.0  # Dirichlet concentration eta


def edl_loss(eta, y, n_classes, lam=1.0):
    """EDL SSE loss + KL regulariser to a uniform Dirichlet on misleading evidence."""
    S = eta.sum(dim=1, keepdim=True)
    p = eta / S
    y1h = F.one_hot(y, n_classes).float()
    sse = ((y1h - p) ** 2 + p * (1 - p) / (S + 1)).sum(dim=1)
    # KL( Dir(alpha_tilde) || Dir(1) ), alpha_tilde = y + (1-y)*eta
    alpha_t = y1h + (1 - y1h) * eta
    kl = _kl_dirichlet_uniform(alpha_t, n_classes)
    return (sse + lam * kl).mean()


def _kl_dirichlet_uniform(alpha, K):
    S = alpha.sum(dim=1, keepdim=True)
    t1 = torch.lgamma(S).squeeze(1) - torch.lgamma(alpha).sum(dim=1) - math.lgamma(K)
    t2 = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    return t1 + t2


# ----------------------------------------------------------------------
# Subjective Neural Network head
# ----------------------------------------------------------------------
class SubjectiveHead(nn.Module):
    """SNN head: Beta-Bernoulli dropout on the bottleneck.

    p_j ~ Beta(alpha_j, beta_j) (trust), z_j ~ Bernoulli(p_j) (mask).
    Kumaraswamy reparam for p; Concrete relaxation for z at train time.
    """

    def __init__(self, d_in, d_hidden, n_classes,
                 prior_a=7.0, prior_b=3.0, init_keep=0.7, tau=0.5):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_classes)
        self.n_classes = n_classes
        self.d_hidden = d_hidden
        self.tau = tau
        self.register_buffer("prior_a", torch.tensor(float(prior_a)))
        self.register_buffer("prior_b", torch.tensor(float(prior_b)))
        # free params -> softplus -> positive (alpha, beta)
        # initialise so E[p] ~= init_keep with a moderate concentration
        a0 = init_keep * 6.0
        b0 = (1 - init_keep) * 6.0
        self.alpha_free = nn.Parameter(torch.full((d_hidden,), softplus_inv(a0)))
        self.beta_free = nn.Parameter(torch.full((d_hidden,), softplus_inv(b0)))

    def alpha_beta(self):
        return F.softplus(self.alpha_free) + EPS, F.softplus(self.beta_free) + EPS

    def expected_keep(self):
        a, b = self.alpha_beta()
        return (a / (a + b)).detach()

    def sample_p(self, batch):
        """Kumaraswamy reparameterised sample of trust prob, shape (batch, d_hidden)."""
        a, b = self.alpha_beta()
        u = torch.rand(batch, self.d_hidden, device=a.device).clamp(EPS, 1 - EPS)
        p = (1 - u.pow(1.0 / b)).pow(1.0 / a)
        return p.clamp(EPS, 1 - EPS)

    def sample_mask(self, p, hard):
        if hard:
            return torch.bernoulli(p)
        u = torch.rand_like(p).clamp(EPS, 1 - EPS)
        logit = (torch.log(p) - torch.log(1 - p)
                 + torch.log(u) - torch.log(1 - u))
        return torch.sigmoid(logit / self.tau)

    def extract_features(self, x):
        """Hidden representation before trust masking, shape (N, d_hidden)."""
        return F.relu(self.fc1(self.ln(x)))

    def forward(self, x, sample=True, hard=None):
        if hard is None:
            hard = not self.training
        h = self.extract_features(x)
        if sample:
            p = self.sample_p(x.shape[0])
            z = self.sample_mask(p, hard=hard)
            h = h * z
        return self.fc2(h)

    def kl(self):
        """Closed-form sum_j KL(Beta(a_j,b_j) || Beta(a0,b0))."""
        a, b = self.alpha_beta()
        a0, b0 = self.prior_a, self.prior_b
        lbeta = lambda x, y: torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
        kl = (lbeta(a0, b0) - lbeta(a, b)
              + (a - a0) * torch.digamma(a)
              + (b - b0) * torch.digamma(b)
              - (a + b - a0 - b0) * torch.digamma(a + b))
        return kl.sum()


# ----------------------------------------------------------------------
# Image backbones (CNN and tiny ResNet) for run_mnist --arch cnn/resnet
# ----------------------------------------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class _CNNBackbone(nn.Module):
    """LeNet-style: three conv layers + pool -> (N, d_out). Input (N,1,28,28)."""

    def __init__(self, d_out):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                      # 28x28 -> 14x14
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                      # 14x14 -> 7x7
        )
        self.fc = nn.Linear(64 * 7 * 7, d_out)

    def forward(self, x):
        return F.relu(self.fc(self.convs(x).flatten(1)))


class _TinyResNet(nn.Module):
    """4-block residual net with global avg pool -> (N, d_out). Input (N,1,28,28)."""

    def __init__(self, d_out):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(_ResBlock(32), _ResBlock(32))
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),        # 28x28 -> 14x14
        )
        self.stage2 = nn.Sequential(_ResBlock(64), _ResBlock(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),        # 14x14 -> 7x7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, d_out)

    def forward(self, x):
        x = self.down2(self.stage2(self.down1(self.stage1(self.stem(x)))))
        return F.relu(self.fc(self.pool(x).flatten(1)))


def _make_backbone(arch, d_out):
    if arch == "cnn":    return _CNNBackbone(d_out)
    if arch == "resnet": return _TinyResNet(d_out)
    raise ValueError(f"unknown arch {arch!r}; expected 'cnn' or 'resnet'")


class SubjectiveCNN(nn.Module):
    """SNN with CNN or ResNet image backbone. Same external interface as SubjectiveHead."""

    def __init__(self, arch, d_hidden, n_classes,
                 prior_a=7.0, prior_b=3.0, init_keep=0.7, tau=0.5):
        super().__init__()
        self.backbone = _make_backbone(arch, d_hidden)
        self.head = SubjectiveHead(d_hidden, d_hidden, n_classes,
                                   prior_a=prior_a, prior_b=prior_b,
                                   init_keep=init_keep, tau=tau)

    @property
    def fc2(self): return self.head.fc2

    def sample_p(self, batch):   return self.head.sample_p(batch)
    def alpha_beta(self):        return self.head.alpha_beta()
    def expected_keep(self):     return self.head.expected_keep()
    def kl(self):                return self.head.kl()

    def extract_features(self, x):
        return self.head.extract_features(self.backbone(x))

    def forward(self, x, sample=True, hard=None):
        return self.head(self.backbone(x), sample=sample, hard=hard)


class MCDropoutCNN(nn.Module):
    """MC Dropout with CNN or ResNet image backbone."""

    def __init__(self, arch, d_hidden, n_classes, p_drop=0.5):
        super().__init__()
        self.backbone = _make_backbone(arch, d_hidden)
        self.head = MCDropoutHead(d_hidden, d_hidden, n_classes, p_drop=p_drop)

    def forward(self, x, sample=False):
        return self.head(self.backbone(x), sample=sample)


class EDLCNN(nn.Module):
    """EDL with CNN or ResNet image backbone."""

    def __init__(self, arch, d_hidden, n_classes):
        super().__init__()
        self.backbone = _make_backbone(arch, d_hidden)
        self.head = EDLHead(d_hidden, d_hidden, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        return self.head(self.backbone(x))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_head(model, Xtr, ytr, n_classes, *, epochs=15, lr=1e-3, bs=64,
               beta_max=5.0, warmup_frac=0.1, is_snn=False, is_edl=False,
               edl_lam=1.0, device="cpu", verbose=False):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    n_train = float(n)
    steps = max(1, n // bs)
    total_steps = epochs * steps
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss, ep_correct, ep_seen, ep_kl = 0.0, 0, 0, 0.0
        beta = lam_t = 0.0
        for i in tqdm(range(0, n - bs + 1, bs), desc=f"Epoch {ep+1}/{epochs}", disable=not verbose):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            if is_edl:
                eta = model(xb)
                lam_t = edl_lam * min(1.0, ep / 10.0)  # Sensoy-style annealing
                loss = edl_loss(eta, yb, n_classes, lam=lam_t)
                preds = eta.argmax(1)
            else:
                logits = model(xb, sample=True) if is_snn else model(xb)
                loss = F.cross_entropy(logits, yb)
                if is_snn:
                    beta = beta_max * min(1.0, step / max(1, warmup_frac * total_steps))
                    kl = model.kl()
                    ep_kl += kl.item()
                    loss = loss + (beta / n_train) * kl
                preds = logits.argmax(1)
            loss.backward()
            opt.step()
            step += 1
            ep_loss += loss.item() * len(yb)
            ep_correct += (preds == yb).sum().item()
            ep_seen += len(yb)
        if verbose:
            msg = (f"  ep {ep+1:>3}/{epochs} loss={ep_loss/max(1,ep_seen):.4f} "
                   f"acc={ep_correct/max(1,ep_seen):.4f}")
            if is_snn:
                msg += (f" kl={ep_kl/max(1,steps):.1f} beta={beta:.2f} "
                        f"std(E[p])={model.expected_keep().std().item():.4f}")
            if is_edl:
                msg += f" lam={lam_t:.2f}"
            print(msg, flush=True)
    model.eval()
    return model
