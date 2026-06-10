"""Last-layer Laplace approximation (diagonal GGN) on a trained LinearHead."""
import torch
import torch.nn.functional as F


class LastLayerLaplace:
    """Gaussian posterior over fc2 weights, diagonal GGN Hessian, MAP-centred."""

    def __init__(self, model, prior_prec=1.0):
        self.model = model.eval()
        self.prior_prec = prior_prec
        self.var = None  # diagonal posterior variance over fc2 params

    @torch.no_grad()
    def _features(self, X):
        return F.relu(self.model.fc1(self.model.ln(X)))

    def fit(self, Xtr, ytr, n_classes, device="cpu"):
        self.model.to(device)
        X = Xtr.to(device)
        h = self._features(X)                       # (N, Hdim)
        probs = F.softmax(self.model.fc2(h), dim=1)  # (N, K)
        # diagonal GGN for weight (k,j): sum_n h_j^2 * p_k (1-p_k)
        pk = (probs * (1 - probs))                   # (N, K)
        hsq = (h ** 2)                               # (N, Hdim)
        ggn_w = torch.einsum("nk,nj->kj", pk, hsq)   # (K, Hdim)
        ggn_b = pk.sum(0)                            # (K,)
        self.var_w = 1.0 / (ggn_w + self.prior_prec)
        self.var_b = 1.0 / (ggn_b + self.prior_prec)
        return self

    @torch.no_grad()
    def predict(self, X, T=30, device="cpu"):
        self.model.to(device)
        h = self._features(X.to(device))
        W = self.model.fc2.weight.data                # (K, Hdim)
        b = self.model.fc2.bias.data                  # (K,)
        outs = []
        for _ in range(T):
            Ws = W + self.var_w.sqrt() * torch.randn_like(W)
            bs = b + self.var_b.sqrt() * torch.randn_like(b)
            outs.append(F.softmax(h @ Ws.T + bs, dim=1))
        outs = torch.stack(outs, dim=1)               # (B, T, K)
        return outs.mean(1).cpu(), outs.cpu()
