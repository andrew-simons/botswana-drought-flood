"""PyTorch Dataset for BotswanaDroughtFloodSet — the GEE->Drive->PyTorch handoff.

The cube on disk (built in Week 3-4) is expected as .npy / .zarr arrays:
    dynamic : (T, C_dyn, H, W)  float32   monthly driver variables
    static  : (C_stat, H, W)    float32   elevation, slope, land cover, climatology
    labels  : (T, 3, H, W)      float32   the 3 normalized drought indices
    mask    : (H, W)            bool       valid land pixels

Forecasting setup (DroughtSet-style, scaled down for Botswana/monthly):
    input  = `input_len` months of dynamic features
    target = next `horizon` months of the labels
A sliding window of stride `stride` generates samples.

This Dataset returns whole-grid windows (good for ConvLSTM). For the per-pixel
baselines you'll instead flatten to tabular rows — see models/baselines.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - allows importing without torch
    torch = None
    Dataset = object


class BotswanaCube(Dataset):
    """Sliding-window spatio-temporal dataset over the Botswana cube."""

    def __init__(
        self,
        root: str | Path,
        input_len: int = 12,
        horizon: int = 3,
        stride: int = 1,
        time_slice: tuple[int, int] | None = None,
        norm_stats: dict | None = None,
    ):
        """
        Args:
            root: folder containing dynamic.npy, static.npy, labels.npy, mask.npy.
            input_len: months of history fed in (e.g. 12 = one year).
            horizon: months ahead to predict.
            stride: step between consecutive windows.
            time_slice: (t0, t1) to restrict to a temporal split (train vs test).
            norm_stats: {'mean': (C_dyn,), 'std': (C_dyn,)} computed on TRAIN ONLY.
                        If None, no normalization is applied (compute it once on train).
        """
        if torch is None:
            raise ImportError("PyTorch is required to use BotswanaCube.")
        root = Path(root)
        self.dynamic = np.load(root / "dynamic.npy", mmap_mode="r")   # (T, C, H, W)
        self.static = np.load(root / "static.npy")                    # (Cs, H, W)
        self.labels = np.load(root / "labels.npy", mmap_mode="r")     # (T, 3, H, W)
        self.mask = np.load(root / "mask.npy")                        # (H, W)

        T = self.dynamic.shape[0]
        self.t0, self.t1 = time_slice or (0, T)
        self.input_len = input_len
        self.horizon = horizon
        self.stride = stride
        self.norm_stats = norm_stats

        # Valid window start indices within the chosen time slice.
        last_start = self.t1 - (input_len + horizon)
        self.starts = list(range(self.t0, last_start + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return x
        mean = np.asarray(self.norm_stats["mean"]).reshape(1, -1, 1, 1)
        std = np.asarray(self.norm_stats["std"]).reshape(1, -1, 1, 1)
        return (x - mean) / (std + 1e-6)

    def __getitem__(self, i: int):
        s = self.starts[i]
        x = np.asarray(self.dynamic[s : s + self.input_len], dtype=np.float32)
        x = self._normalize(x)                                   # (L, C, H, W)
        y = np.asarray(
            self.labels[s + self.input_len : s + self.input_len + self.horizon],
            dtype=np.float32,
        )                                                        # (horizon, 3, H, W)
        static = np.asarray(self.static, dtype=np.float32)       # (Cs, H, W)

        # NaNs (clouds/no-data) -> 0 after normalization; the mask tells the loss
        # which pixels to score.
        x = np.nan_to_num(x)
        y_mask = ~np.isnan(y) & self.mask[None, None]
        y = np.nan_to_num(y)

        return {
            "x": torch.from_numpy(x),
            "static": torch.from_numpy(static),
            "y": torch.from_numpy(y),
            "y_mask": torch.from_numpy(y_mask),
        }


def compute_norm_stats(root: str | Path, train_slice: tuple[int, int]) -> dict:
    """Per-channel mean/std over the TRAIN time slice only (prevents leakage)."""
    root = Path(root)
    dyn = np.load(root / "dynamic.npy", mmap_mode="r")
    t0, t1 = train_slice
    chunk = np.asarray(dyn[t0:t1], dtype=np.float64)             # (t, C, H, W)
    mean = np.nanmean(chunk, axis=(0, 2, 3))
    std = np.nanstd(chunk, axis=(0, 2, 3))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
