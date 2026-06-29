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
