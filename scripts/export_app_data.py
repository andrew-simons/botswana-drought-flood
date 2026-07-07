"""Generate data/app/ from the cube and trained models.

Run once (or after re-training) from the repo root:
    python scripts/export_app_data.py

Requires the cube in data/cube/ and (for ConvLSTM) data/cube/model_convlstm.pt.
Outputs are small enough to commit to git (~10 MB each).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

CUBE_DIR   = REPO / "data" / "cube"
APP_DIR    = REPO / "data" / "app"
ASSETS_DIR = REPO / "app" / "assets"
APP_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load cube metadata ────────────────────────────────────────────────────────
cube_meta = json.load(open(CUBE_DIR / "meta.json"))
TRAIN_END = cube_meta["splits"]["train_end_idx"]   # 228
T         = cube_meta["time"]["T"]                  # 252
N_TEST    = T - TRAIN_END                           # 24

print(f"Cube: T={T}, TRAIN_END={TRAIN_END}, N_TEST={N_TEST}")

dynamic = np.load(CUBE_DIR / "dynamic.npy", mmap_mode="r")
static  = np.load(CUBE_DIR / "static.npy")
labels  = np.load(CUBE_DIR / "labels.npy",  mmap_mode="r")
mask    = np.load(CUBE_DIR / "mask.npy")

# ── 1. Save targets (actual test-period labels) ───────────────────────────────
targets = np.asarray(labels[TRAIN_END:], dtype=np.float32)   # (24, 3, H, W)
np.save(APP_DIR / "targets.npy", targets)
print(f"Saved targets.npy  {targets.shape}")

# ── 2. Save mask ──────────────────────────────────────────────────────────────
np.save(APP_DIR / "mask.npy", mask)
print(f"Saved mask.npy  {mask.shape}")

# ── 3. XGBoost predictions ────────────────────────────────────────────────────
print("\nTraining XGBoost (this takes ~5 minutes)…")
try:
    import ctypes, pathlib
    _libomp = pathlib.Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")
    if _libomp.exists():
        ctypes.cdll.LoadLibrary(str(_libomp))
except Exception:
    pass

from botswana_ds.models.baselines import xgb_forecast

xgb_preds, _, _ = xgb_forecast(
    dynamic, static, labels, mask,
    train_end=TRAIN_END,
    input_len=3,
    max_train_rows=400_000,
    seed=42,
    return_models=True,
)
np.save(APP_DIR / "xgb_preds.npy", xgb_preds.astype(np.float32))
print(f"Saved xgb_preds.npy  {xgb_preds.shape}")

# ── 4. ConvLSTM predictions ───────────────────────────────────────────────────
ckpt_path = CUBE_DIR / "model_convlstm.pt"
if ckpt_path.exists():
    print("\nRunning ConvLSTM inference…")
    import torch
    from botswana_ds.models.convlstm import ConvLSTMForecaster
    from botswana_ds.data.dataset import BotswanaCube, compute_norm_stats
    from botswana_ds.train.train import predict

    INPUT_LEN = 24
    HORIZON   = 1
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    norm_stats = compute_norm_stats(CUBE_DIR, train_slice=(0, TRAIN_END))
    test_ds = BotswanaCube(
        CUBE_DIR, input_len=INPUT_LEN, horizon=HORIZON,
        time_slice=(TRAIN_END - INPUT_LEN, T), norm_stats=norm_stats,
    )

    C_dyn = dynamic.shape[1]
    model = ConvLSTMForecaster(
        in_ch=C_dyn + 3, hidden_ch=64, n_targets=3,
        horizon=HORIZON, static_ch=static.shape[0],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    cl_preds, _, _ = predict(model, loader, device=device)
    # cl_preds: (N_TEST, horizon=1, 3, H, W) → squeeze horizon dim
    cl_preds = cl_preds[:, 0].astype(np.float32)   # (N_TEST, 3, H, W)
    np.save(APP_DIR / "convlstm_preds.npy", cl_preds)
    print(f"Saved convlstm_preds.npy  {cl_preds.shape}")
else:
    print(f"Skipping ConvLSTM — checkpoint not found at {ckpt_path}")

# ── 5. App metadata ───────────────────────────────────────────────────────────
import datetime
g = cube_meta["grid"]
app_meta = {
    "grid": {
        "min_lon":      g["min_lon"],
        "min_lat":      g["min_lat"],
        "max_lon":      g["max_lon"],
        "max_lat":      g["max_lat"],
        "res":          g["res"],
        "H":            g["H"],
        "W":            g["W"],
        "row0_is_north": True,   # GEE exports northernmost row first
    },
    "test_start":  cube_meta["splits"]["test_start_date"],   # "2022-01"
    "T_test":      N_TEST,
    "label_channels": cube_meta["label_channels"],
    "dynamic_channels": cube_meta["dynamic_channels"],
    "last_updated": datetime.date.today().isoformat(),
}
with open(APP_DIR / "meta.json", "w") as f:
    json.dump(app_meta, f, indent=2)
print(f"\nSaved meta.json  (last_updated: {app_meta['last_updated']})")

# ── 6. Download Botswana district GeoJSON (admin level 1) ────────────────────
geojson_path = ASSETS_DIR / "botswana_districts.geojson"
if not geojson_path.exists():
    print("\nDownloading Botswana district boundaries from GADM…")
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BWA_1.json"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw = resp.json()
        # Keep only NAME_1 and geometry to reduce file size
        slim = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"NAME_1": f["properties"]["NAME_1"]},
                    "geometry": f["geometry"],
                }
                for f in raw["features"]
            ],
        }
        with open(geojson_path, "w") as fj:
            json.dump(slim, fj, separators=(",", ":"))
        print(f"Saved botswana_districts.geojson  ({geojson_path.stat().st_size // 1024} KB)")
    except Exception as e:
        print(f"Could not download district GeoJSON: {e}")
        print("The app will work without district boundaries.")
else:
    print(f"\nbotswana_districts.geojson already exists — skipping download.")

print("\nDone. Commit data/app/ and app/assets/ to git, then push to deploy.")
