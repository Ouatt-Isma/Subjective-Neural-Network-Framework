"""Metrics implemented with numpy to avoid heavy deps."""
import numpy as np

try:
    _trapz = np.trapezoid  # numpy>=2
except AttributeError:
    _trapz = np.trapz

EPS = 1e-12


def _np(x):
    import torch
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def accuracy(probs, y):
    probs, y = _np(probs), _np(y)
    return float((probs.argmax(1) == y).mean())


def nll(probs, y):
    probs, y = _np(probs), _np(y)
    p = probs[np.arange(len(y)), y].clip(EPS, 1.0)
    return float(-np.log(p).mean())


def brier(probs, y, K):
    probs, y = _np(probs), _np(y)
    oh = np.eye(K)[y]
    return float(((probs - oh) ** 2).sum(1).mean())


def ece(probs, y, n_bins=15):
    probs, y = _np(probs), _np(y)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


# ---- OOD detection: score is an uncertainty (higher => more OOD) ----
def _roc_auc(scores, labels):
    """labels: 1 for OOD (positive), 0 for ID."""
    scores, labels = _np(scores), _np(labels)
    order = np.argsort(-scores)
    labels = labels[order]
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = tps / P
    fpr = fps / N
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(_trapz(tpr, fpr))


def _aupr(scores, labels):
    scores, labels = _np(scores), _np(labels)
    order = np.argsort(-scores)
    labels = labels[order]
    tps = np.cumsum(labels)
    prec = tps / (np.arange(len(labels)) + 1)
    rec = tps / max(1, labels.sum())
    return float(_trapz(prec, rec))


def _fpr_at_tpr(scores, labels, tpr_target=0.95):
    scores, labels = _np(scores), _np(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 1 - tpr_target)  # detect tpr_target of OOD
    return float((neg >= thr).mean())


def ood_metrics(score_id, score_ood):
    scores = np.concatenate([_np(score_id), _np(score_ood)])
    labels = np.concatenate([np.zeros(len(_np(score_id))), np.ones(len(_np(score_ood)))])
    return dict(auroc=_roc_auc(scores, labels),
                aupr=_aupr(scores, labels),
                fpr95=_fpr_at_tpr(scores, labels, 0.95))


# ---- Selective prediction ----
def selective_ausc(confidence, correct, cov_lo=0.4, cov_hi=1.0, n=100):
    """Area under retained-accuracy vs coverage curve over [cov_lo, cov_hi].

    confidence: higher => more confident (accept first). correct: bool array.
    """
    confidence, correct = _np(confidence), _np(correct).astype(float)
    order = np.argsort(-confidence)
    correct = correct[order]
    cum_acc = np.cumsum(correct) / (np.arange(len(correct)) + 1)
    cov = (np.arange(len(correct)) + 1) / len(correct)
    grid = np.linspace(cov_lo, cov_hi, n)
    acc_at = np.interp(grid, cov, cum_acc)
    return float(_trapz(acc_at, grid) / (cov_hi - cov_lo))


def selective_auroc(confidence, correct):
    """AUROC of confidence as a detector of correctness."""
    return _roc_auc(_np(confidence), 1 - _np(correct).astype(int))
