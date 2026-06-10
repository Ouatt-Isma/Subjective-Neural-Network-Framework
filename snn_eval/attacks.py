"""FGSM / PGD attacks on the head input features.

Caveat: with a frozen backbone we attack the penultimate features directly,
which is a relaxation of full input-space attacks. For end-to-end pixel attacks
backprop through the (frozen) backbone instead; the head API is identical.
"""
import torch
import torch.nn.functional as F


def _logits_for_attack(model, x, is_snn, is_edl):
    if is_edl:
        eta = model(x)
        return (eta / eta.sum(1, keepdim=True)).clamp_min(1e-8).log()
    # deterministic mean for a stable gradient
    return model(x, sample=False) if is_snn else model(x)


def fgsm(model, X, y, eps, is_snn=False, is_edl=False, device="cpu"):
    model.eval().to(device)
    x = X.clone().to(device).requires_grad_(True)
    logits = _logits_for_attack(model, x, is_snn, is_edl)
    loss = F.cross_entropy(logits, y.to(device))
    grad, = torch.autograd.grad(loss, x)
    return (x + eps * grad.sign()).detach().cpu()


def pgd(model, X, y, eps, alpha=None, steps=10, is_snn=False, is_edl=False, device="cpu"):
    model.eval().to(device)
    if alpha is None:
        alpha = eps / 4
    x0 = X.clone().to(device)
    x = x0 + 0.001 * torch.randn_like(x0)
    for _ in range(steps):
        x.requires_grad_(True)
        logits = _logits_for_attack(model, x, is_snn, is_edl)
        loss = F.cross_entropy(logits, y.to(device))
        grad, = torch.autograd.grad(loss, x)
        x = x.detach() + alpha * grad.sign()
        x = torch.max(torch.min(x, x0 + eps), x0 - eps)
    return x.detach().cpu()
