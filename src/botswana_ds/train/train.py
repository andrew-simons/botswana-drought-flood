"""Training loop for ConvLSTM (and any future spatial models).

Design decisions
----------------
- Masked MAE loss: only valid land pixels contribute — prevents ocean/outside-Botswana
  pixels from dominating gradients.
- Validation split is the last 2 years of the training period (temporal, not random)
  so the model never sees future data during training.
- Early stopping on val loss (patience=15 epochs) saves the best checkpoint and
  avoids overfitting on the 19-year training set.
- ReduceLROnPlateau: halves lr when val loss stalls for 5 epochs.
- MPS support: uses Apple Silicon GPU when available (useful for local dev on Mac).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from botswana_ds.models.convlstm import drought_weighted_mae_loss, masked_mae_loss


# ── Epoch helpers ─────────────────────────────────────────────────────────────

def _run_epoch(model, loader, optimizer, device, training: bool, loss_fn=None) -> float:
    active_fn = loss_fn or masked_mae_loss
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            x      = batch["x"].to(device)
            static = batch["static"].to(device)
            y      = batch["y"].to(device)
            y_mask = batch["y_mask"].to(device)

            pred = model(x, static)
            loss = active_fn(pred, y, y_mask)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total += loss.item()
    return total / max(len(loader), 1)


# ── Main training function ────────────────────────────────────────────────────

def fit(
    model,
    train_loader,
    val_loader,
    *,
    n_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    ckpt_path: str | Path = "best_model.pt",
    device: str | None = None,
    verbose: bool = True,
    loss_fn=None,
) -> dict:
    """Train with early stopping; save best checkpoint by val loss.

    Parameters
    ----------
    model        : nn.Module (ConvLSTMForecaster or similar)
    train_loader : DataLoader for training windows
    val_loader   : DataLoader for validation windows
    n_epochs     : maximum epochs
    lr           : initial learning rate for Adam
    patience     : early-stop patience (epochs without improvement)
    ckpt_path    : where to save the best model weights
    device       : 'cuda' | 'mps' | 'cpu' — auto-detected if None
    verbose      : print progress every 5 epochs

    loss_fn  : callable(pred, target, mask) -> scalar loss.
               Defaults to masked_mae_loss.

    Returns
    -------
    history : {'train': [float, ...], 'val': [float, ...]}
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Device: {device}   Trainable params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    history = {"train": [], "val": []}
    best_val = float("inf")
    wait = 0

    for epoch in range(n_epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, device, training=True,  loss_fn=loss_fn)
        val_loss   = _run_epoch(model, val_loader,   optimizer, device, training=False, loss_fn=loss_fn)
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(),
                 "val_loss": best_val, "history": history},
                ckpt_path,
            )
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stop — epoch {epoch + 1}  best val MAE = {best_val:.4f}")
                break

        if verbose and (epoch + 1) % 5 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            marker = " ✓" if val_loss == best_val else ""
            print(
                f"Epoch {epoch+1:4d}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={lr_now:.1e}{marker}"
            )

    return history


# ── Inference ────────────────────────────────────────────────────────────────

def predict(model, loader, device: str | None = None):
    """Run model on all batches; return stacked numpy arrays.

    Returns
    -------
    preds   : (N, horizon, 3, H, W) float32
    targets : (N, horizon, 3, H, W) float32
    masks   : (N, horizon, 3, H, W) bool
    """
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["x"].to(device), batch["static"].to(device))
            preds.append(pred.cpu().numpy())
            targets.append(batch["y"].numpy())
            masks.append(batch["y_mask"].numpy())
    return (
        np.concatenate(preds).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
        np.concatenate(masks),
    )
