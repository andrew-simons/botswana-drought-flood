"""ConvLSTM — the spatio-temporal core model (Week 6).

A ConvLSTM replaces the LSTM's fully-connected gates with convolutions, so it keeps
the (H, W) spatial structure of the grid while modeling change over time. This is the
best accuracy/effort trade-off for gridded forecasting and the natural Botswana
analogue of DroughtSet's spatio-temporal models.

Reference: Shi et al. 2015, "Convolutional LSTM Network".

Input  : (B, L, C_in, H, W)   L months of driver features
Output : (B, horizon, n_targets, H, W)   forecast of the drought indices
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        # One conv produces all four gates (input, forget, output, cell) at once.
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel, padding=pad)
        self.hidden_ch = hidden_ch

    def forward(self, x, state):
        h, c = state
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(z, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, hw, device):
        h = torch.zeros(batch, self.hidden_ch, *hw, device=device)
        c = torch.zeros(batch, self.hidden_ch, *hw, device=device)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Single-layer ConvLSTM encoder + conv head that emits the full horizon."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 32,
        n_targets: int = 3,
        horizon: int = 3,
        kernel: int = 3,
        static_ch: int = 0,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets
        self.cell = ConvLSTMCell(in_ch + static_ch, hidden_ch, kernel)
        self.head = nn.Conv2d(hidden_ch, horizon * n_targets, kernel, padding=kernel // 2)
        self.static_ch = static_ch

    def forward(self, x, static=None):
        # x: (B, L, C, H, W)
        B, L, C, H, W = x.shape
        h, c = self.cell.init_state(B, (H, W), x.device)
        for t in range(L):
            xt = x[:, t]
            if self.static_ch and static is not None:
                xt = torch.cat([xt, static], dim=1)            # broadcast static each step
            h, c = self.cell(xt, (h, c))
        out = self.head(h)                                     # (B, horizon*n_targets, H, W)
        return out.view(B, self.horizon, self.n_targets, H, W)


def masked_mae_loss(pred, target, mask):
    """MAE over valid pixels only (mask: True = score this pixel)."""
    diff = torch.abs(pred - target)
    m = mask.float()
    return (diff * m).sum() / (m.sum() + 1e-6)
