"""Compute the three continuous label arrays used as prediction targets.

WHY training-only fitting matters
----------------------------------
Both SPI and the anomaly labels are computed by fitting statistics (gamma
parameters, monthly mean/std) to the data. If we fit on the full dataset
(including 2022-2024 test years), future data leaks into the training
statistics. For example, if 2022 was an extreme drought year, it would
shift the gamma distribution fit and make the training-period SPI values
look more extreme than they really were.

Rule: every normalization constant is computed on train_end months (0 to
train_end-1) and applied to the full series.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

TRAIN_END = 228  # Jan 2003 = index 0 → Dec 2021 = index 227 (19 years, exclusive upper bound)


def spi3(
    rain: np.ndarray,
    train_end: int = TRAIN_END,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    """3-month Standardized Precipitation Index.

    The SPI answers: "how unusual is this 3-month rainfall total compared
    to the historical distribution for this calendar month?"

    Algorithm (McKee et al., 1993)
    --------------------------------
    1. Compute 3-month rolling sum (NaN for t=0,1 — not enough history).
    2. For each calendar month (Jan–Dec), fit a gamma distribution to the
       rolling sums in the TRAINING period.
       We use method-of-moments (fast, vectorized) instead of MLE.
    3. Handle zero-precip months with a mixed distribution:
         H(x) = p₀                              if x == 0
                p₀ + (1 − p₀) × Γ_cdf(x)        if x >  0
       where p₀ = fraction of training months with zero rain.
    4. SPI = Φ⁻¹(H(x)) — transform the [0,1] probability to a standard
       normal score. SPI ≤ −1 = moderate drought; ≤ −2 = severe drought.

    Parameters
    ----------
    rain      : (T, H, W) monthly precip in mm — physical units
    train_end : training period upper bound (exclusive)
    land_mask : (H, W) bool — NaN is written for False pixels

    Returns
    -------
    (T, H, W) float32 — NaN for t < 2 and outside the land mask
    """
    T, H, W = rain.shape
    result = np.full((T, H, W), np.nan, dtype=np.float32)

    # 3-month rolling sum — undefined for t=0,1
    roll3 = np.full((T, H, W), np.nan, dtype=np.float64)
    for t in range(2, T):
        roll3[t] = rain[t - 2] + rain[t - 1] + rain[t]

    for cal_m in range(12):
        train_idx = np.array([t for t in range(2, train_end) if t % 12 == cal_m])
        all_idx   = np.array([t for t in range(2, T)         if t % 12 == cal_m])
        if len(train_idx) == 0:
            continue

        x_train = roll3[train_idx]  # (n_train, H, W)

        # Fraction of months with zero precip
        n_valid = np.sum(~np.isnan(x_train), axis=0).clip(min=1)
        p0      = np.sum(x_train == 0, axis=0) / n_valid  # (H, W)

        # Gamma parameters (method of moments on positive values only)
        x_pos    = np.where(x_train > 0, x_train, np.nan)
        mu       = np.nanmean(x_pos, axis=0)   # (H, W)
        var      = np.nanvar(x_pos, axis=0)    # (H, W)
        valid_fit = (mu > 0) & (var > 1e-12)
        alpha = np.where(valid_fit, mu ** 2 / var, np.nan)   # shape
        beta  = np.where(valid_fit, var / mu,       np.nan)  # scale

        # Vectorize over all time steps for this calendar month at once
        x_all  = roll3[all_idx]                  # (n_months, H, W)
        x_safe = np.where((x_all > 0) & valid_fit, x_all, 1.0)

        gcdf  = stats.gamma.cdf(x_safe, alpha, scale=beta)   # broadcasting over n_months
        h_val = np.where(x_all > 0, p0 + (1 - p0) * gcdf, p0)
        h_val = np.clip(h_val, 0.001, 0.999)

        spi_all = stats.norm.ppf(h_val).astype(np.float32)
        spi_all = np.where(valid_fit & ~np.isnan(x_all), spi_all, np.nan)
        result[all_idx] = spi_all

    if land_mask is not None:
        result[:, ~land_mask] = np.nan

    return result


def anomaly(
    arr: np.ndarray,
    train_end: int = TRAIN_END,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-pixel monthly z-score anomaly.

    For each pixel and each calendar month, compute mean and std over the
    training period, then standardize the full time series:

        anomaly(t, h, w) = (arr(t, h, w) − μ_month(h,w)) / σ_month(h,w)

    μ and σ are fitted on training data only.

    Handles NaN transparently (e.g. SMAP has NaN before March 2015) —
    nanmean/nanstd ignores NaN values automatically.

    Parameters
    ----------
    arr       : (T, H, W) physical units
    train_end : training period upper bound (exclusive)
    land_mask : (H, W) bool

    Returns
    -------
    (T, H, W) float32 — NaN where σ == 0, input is NaN, or outside mask
    """
    T, H, W = arr.shape
    result = np.full((T, H, W), np.nan, dtype=np.float32)

    for cal_m in range(12):
        train_idx = np.array([t for t in range(train_end) if t % 12 == cal_m])
        all_idx   = np.array([t for t in range(T)          if t % 12 == cal_m])
        if len(train_idx) == 0:
            continue

        x_train = arr[train_idx].astype(np.float64)  # (n_train, H, W)
        mu  = np.nanmean(x_train, axis=0)             # (H, W)
        sig = np.nanstd(x_train, axis=0)              # (H, W)
        sig = np.where(sig < 1e-9, np.nan, sig)       # avoid division by near-zero

        x_all = arr[all_idx].astype(np.float64)       # (n_months, H, W)
        z_all = (x_all - mu) / sig                    # broadcasts over n_months
        result[all_idx] = z_all.astype(np.float32)

    if land_mask is not None:
        result[:, ~land_mask] = np.nan

    return result
