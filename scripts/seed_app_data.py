"""Seed data/app/ from the cube for local UI testing.

Uses real labels as 'targets' and adds Gaussian noise to simulate
model predictions — no training required.  Replace with real model
outputs by running scripts/export_app_data.py once models are ready.

    python scripts/seed_app_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import requests

REPO      = Path(__file__).parent.parent
CUBE_DIR  = REPO / "data" / "cube"
APP_DIR   = REPO / "data" / "app"
ASSETS_DIR = REPO / "app" / "assets"
APP_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# ── Load cube ─────────────────────────────────────────────────────────────────
print("Loading cube…")
cube_meta  = json.load(open(CUBE_DIR / "meta.json"))
TRAIN_END  = cube_meta["splits"]["train_end_idx"]   # 228
T          = cube_meta["time"]["T"]                  # 252
N_TEST     = T - TRAIN_END                           # 24

labels = np.load(CUBE_DIR / "labels.npy", mmap_mode="r")   # (T, 3, H, W)
mask   = np.load(CUBE_DIR / "mask.npy")                      # (H, W) bool

targets = np.asarray(labels[TRAIN_END:], dtype=np.float32)  # (24, 3, H, W)
H, W    = mask.shape

# ── Targets ───────────────────────────────────────────────────────────────────
np.save(APP_DIR / "targets.npy", targets)
print(f"Saved targets.npy  {targets.shape}")

# ── Mask ──────────────────────────────────────────────────────────────────────
np.save(APP_DIR / "mask.npy", mask)
print(f"Saved mask.npy  {mask.shape}")

# ── Synthetic predictions (real + noise, NaN outside mask) ────────────────────
noise       = rng.normal(0, 0.3, targets.shape).astype(np.float32)
preds       = targets + noise
land_mask4d = mask[None, None, :, :]                        # broadcast (1,1,H,W)
preds       = np.where(land_mask4d, preds, np.nan)
np.save(APP_DIR / "xgb_preds.npy", preds)
print(f"Saved xgb_preds.npy  {preds.shape}  (synthetic — replace with real XGB output)")

# ── App metadata ──────────────────────────────────────────────────────────────
import datetime
g = cube_meta["grid"]
app_meta = {
    "grid": {
        "min_lon": g["min_lon"],
        "min_lat": g["min_lat"],
        "max_lon": g["max_lon"],
        "max_lat": g["max_lat"],
        "res":     g["res"],
        "H":       g["H"],
        "W":       g["W"],
        "row0_is_north": True,
    },
    "test_start":       cube_meta["splits"]["test_start_date"],
    "T_test":           N_TEST,
    "label_channels":   cube_meta["label_channels"],
    "dynamic_channels": cube_meta["dynamic_channels"],
    "last_updated":     datetime.date.today().isoformat(),
}
with open(APP_DIR / "meta.json", "w") as f:
    json.dump(app_meta, f, indent=2)
print(f"Saved meta.json")

# ── Download district GeoJSON ─────────────────────────────────────────────────
geojson_path = ASSETS_DIR / "botswana_districts.geojson"
if not geojson_path.exists():
    print("\nDownloading Botswana district boundaries from GADM…")
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BWA_1.json"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw = resp.json()
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
        print(f"Could not download GeoJSON: {e} — app will work without district boundaries.")
else:
    print(f"botswana_districts.geojson already exists.")

print("\nDone. Run: streamlit run app/app.py")
