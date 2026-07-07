"""Dumb-but-essential baselines. Your fancy model MUST beat these.

- Persistence: "next month's drought index = this month's." Surprisingly strong for
  slow-onset drought.
- Climatology: "next month = the long-term average for that calendar month." Captures
  seasonality with zero learning.

These operate directly on the labels array (T, 3, H, W); no training needed. Use them
as the reference line in every metric table.
"""

from __future__ import annotations

import numpy as np

# ── XGBoost per-pixel baseline ──────────────────────────────────────────────


def xgb_forecast(
    dynamic: np.ndarray,
    static: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    train_end: int,
    input_len: int = 3,
    max_train_rows: int | None = 400_000,
    seed: int = 42,
    return_models: bool = False,
    **xgb_kwargs,
) -> tuple:
    """Per-pixel XGBoost baseline (1-month-ahead forecast).

    For each valid pixel × window, builds a flat feature vector:
        [dynamic[t-input_len:t, :, h, w] flattened]  ← lagged time series
        + [static[:, h, w]]                           ← terrain, land cover
    Trains one XGBRegressor per label channel (SPI-3, NDVI-anom, SM-anom).

    No spatial reasoning — each pixel is treated independently.  This is the
    key limitation that ConvLSTM is meant to overcome.

    Parameters
    ----------
    dynamic       : (T, C_dyn, H, W)
    static        : (C_stat, H, W)
    labels        : (T, 3, H, W)      SPI-3, NDVI-anom, SM-anom
    mask          : (H, W) bool
    train_end     : exclusive upper index of training period (e.g. 228)
    input_len     : months of history used as features
    max_train_rows: subsample training rows for speed; None = no cap
    seed          : random seed for subsampling + XGBoost

    Returns
    -------
    preds   : (n_test, 3, H, W) float32  — one prediction per test month
    targets : (n_test, 3, H, W) float32  — actual labels for the same months
    models  : list of 3 XGBRegressor     — only returned when return_models=True
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("pip install xgboost")

    T, C_dyn, H, W = dynamic.shape
    C_stat = static.shape[0]
    valid_h, valid_w = np.where(mask)
    n_valid = len(valid_h)

    # Extract valid-pixel slices once — avoids repeated fancy-indexing in the loop.
    dyn_px  = dynamic[:, :, valid_h, valid_w]   # (T, C_dyn, n_valid)
    stat_px = static[:, valid_h, valid_w].T      # (n_valid, C_stat)
    lab_px  = labels[:, :, valid_h, valid_w]     # (T, 3, n_valid)

    def _build(starts):
        """Build (X, Y) matrices for a list of window start indices."""
        n_w = len(starts)
        n_feat = input_len * C_dyn + C_stat
        X = np.empty((n_w * n_valid, n_feat), dtype=np.float32)
        Y = np.empty((n_w * n_valid, 3),      dtype=np.float32)
        for wi, s in enumerate(starts):
            # Lag features: (input_len, C_dyn, n_valid) → (n_valid, input_len*C_dyn)
            lag = dyn_px[s: s + input_len].transpose(2, 0, 1).reshape(n_valid, -1)
            X[wi * n_valid: (wi + 1) * n_valid] = np.concatenate([lag, stat_px], axis=1)
            # Target = label at s+input_len  (3, n_valid) → (n_valid, 3)
            Y[wi * n_valid: (wi + 1) * n_valid] = lab_px[s + input_len].T
        return X, Y

    # Window s → features from dynamic[s:s+input_len], target = labels[s+input_len].
    # Train: target in [0, train_end)  →  s ∈ [0, train_end - input_len)
    # Test : target in [train_end, T)  →  s ∈ [train_end - input_len, T - input_len)
    train_starts = list(range(0, train_end - input_len))
    test_starts  = list(range(train_end - input_len, T - input_len))

    X_train, Y_train = _build(train_starts)
    X_test,  _       = _build(test_starts)

    rng = np.random.default_rng(seed)
    if max_train_rows and len(X_train) > max_train_rows:
        idx = rng.choice(len(X_train), max_train_rows, replace=False)
        X_train, Y_train = X_train[idx], Y_train[idx]

    n_test = len(test_starts)
    preds   = np.full((n_test, 3, H, W), np.nan, dtype=np.float32)
    targets = np.full((n_test, 3, H, W), np.nan, dtype=np.float32)

    xgb_params = {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": seed, "n_jobs": -1, "tree_method": "hist",
    }
    xgb_params.update(xgb_kwargs)

    fitted_models = []
    for c in range(3):
        # Drop NaN target rows (SMAP has NaN before March 2015 → sm_anom NaN).
        valid_rows = ~np.isnan(Y_train[:, c])
        mdl = XGBRegressor(**xgb_params)
        mdl.fit(X_train[valid_rows], Y_train[valid_rows, c])
        y_hat = mdl.predict(X_test).reshape(n_test, n_valid).astype(np.float32)
        preds[:, c, valid_h, valid_w] = y_hat
        fitted_models.append(mdl)

    # Populate targets from the labels array.
    for wi, s in enumerate(test_starts):
        targets[wi] = labels[s + input_len]

    if return_models:
        return preds, targets, fitted_models
    return preds, targets


def persistence_forecast(labels: np.ndarray, input_len: int, horizon: int) -> np.ndarray:
    """For each window, repeat the LAST observed month across the horizon.

    Returns predictions aligned to the same windows the Dataset produces:
        preds shape = (n_windows, horizon, 3, H, W)
    """
    T = labels.shape[0]
    starts = range(0, T - (input_len + horizon) + 1)
    preds = []
    for s in starts:
        last = labels[s + input_len - 1]                 # (3, H, W)
        preds.append(np.repeat(last[None], horizon, axis=0))
    return np.stack(preds) if preds else np.empty((0,))


def climatology_forecast(
    labels: np.ndarray, input_len: int, horizon: int, calendar_offset: int = 0
) -> np.ndarray:
    """Predict each future month with its per-pixel calendar-month mean.

    calendar_offset = month index (0=Jan) of labels[0], so we map each target month
    to the right climatological mean.
    """
    T = labels.shape[0]
    # Per-pixel mean for each of the 12 calendar months.
    clim = np.full((12,) + labels.shape[1:], np.nan, dtype=np.float64)
    for m in range(12):
        sel = labels[(np.arange(T) + calendar_offset) % 12 == m]
        if len(sel):
            clim[m] = np.nanmean(sel, axis=0)

    starts = range(0, T - (input_len + horizon) + 1)
    preds = []
    for s in starts:
        window = []
        for h in range(horizon):
            month = (s + input_len + h + calendar_offset) % 12
            window.append(clim[month])
        preds.append(np.stack(window))
    return np.stack(preds).astype(np.float32) if preds else np.empty((0,))
