"""Evaluation metrics — DroughtSet-comparable.

Regression metrics on the continuous drought indices, plus drought-detection metrics
after binarizing at a percentile threshold (DroughtSet uses the 30th percentile).
All metrics ignore invalid pixels via the mask.
"""

from __future__ import annotations

import numpy as np


def _flat(pred, target, mask):
    m = mask.astype(bool)
    return pred[m], target[m]


def mae(pred, target, mask):
    p, t = _flat(pred, target, mask)
    return float(np.mean(np.abs(p - t)))


def rmse(pred, target, mask):
    p, t = _flat(pred, target, mask)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def r2(pred, target, mask):
    p, t = _flat(pred, target, mask)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def drought_detection(pred, target, mask, percentile: float = 30.0):
    """Binarize both at the per-array `percentile` (drought = below threshold) and
    report accuracy / precision / recall / f1.
    """
    p, t = _flat(pred, target, mask)
    thr = np.percentile(t, percentile)
    pred_d = p < thr
    true_d = t < thr
    tp = np.sum(pred_d & true_d)
    fp = np.sum(pred_d & ~true_d)
    fn = np.sum(~pred_d & true_d)
    tn = np.sum(~pred_d & ~true_d)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "threshold": float(thr)}


def summary(pred, target, mask, percentile: float = 30.0) -> dict:
    """One call -> full metric dict for a results table."""
    out = {"mae": mae(pred, target, mask), "rmse": rmse(pred, target, mask),
           "r2": r2(pred, target, mask)}
    out.update(drought_detection(pred, target, mask, percentile))
    return out


def batch_summary(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    percentile: float = 30.0,
) -> dict:
    """Metrics over a stack of (N, H, W) predictions.

    Flattens all N time steps into one pool of valid pixels, then computes
    the same metrics as ``summary``.  NaN values (e.g. SMAP gaps) are
    dropped before any calculation.

    Parameters
    ----------
    pred, target : (N, H, W) or (H, W)
    mask         : (H, W) bool — same for all N time steps
    """
    pred   = np.asarray(pred,   dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim == 2:
        pred   = pred[None]
        target = target[None]

    mask_3d = np.broadcast_to(mask[None], pred.shape)   # (N, H, W)
    p = pred[mask_3d]
    t = target[mask_3d]

    valid = ~np.isnan(p) & ~np.isnan(t)
    p, t = p[valid], t[valid]

    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)

    thr    = np.percentile(t, percentile)
    pred_d = p < thr
    true_d = t < thr
    tp = np.sum(pred_d & true_d)
    fp = np.sum(pred_d & ~true_d)
    fn = np.sum(~pred_d & true_d)
    tn = np.sum(~pred_d & ~true_d)

    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)

    return {
        "mae":       float(np.mean(np.abs(p - t))),
        "rmse":      float(np.sqrt(np.mean((p - t) ** 2))),
        "r2":        float(1 - ss_res / (ss_tot + 1e-12)),
        "f1":        float(f1),
        "precision": float(prec),
        "recall":    float(rec),
        "accuracy":  float(acc),
        "threshold": float(thr),
        "n_samples": int(valid.sum()),
    }
