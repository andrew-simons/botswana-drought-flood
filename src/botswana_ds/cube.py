"""Load raw GeoTIFF exports, apply scale factors, and assemble the cube.

Cube structure
--------------
dynamic : (T, C_dyn, H, W) = (252, 9, 182, 188)  float32  ~290 MB
static  : (C_sta, H, W)    = (3, 182, 188)          float32  tiny
labels  : (T, 3, H, W)     = (252, 3, 182, 188)    float32  ~104 MB
mask    : (H, W)            = (182, 188)              bool     tiny
meta.json                    — grid params, channel names, split dates

Quick usage
-----------
    from botswana_ds.cube import assemble
    assemble('data/', 'data/cube/')
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rioxarray as rxr

from .grid import make_grid
from .labels import spi3, anomaly, TRAIN_END

# ── Grid constants ────────────────────────────────────────────────────────────
_GRID = make_grid()
H, W  = _GRID.shape        # 182, 188
T     = 252                 # Jan 2003 – Dec 2023

# SMAP (SPL4SMGP) data begins March 2015 — first 2 months (Jan–Feb 2015) are unavailable.
# March 2015 in the 252-month index: (2015-2003)*12 + 2 = 146
SMAP_START = 146
SMAP_N     = 106            # months from March 2015 to December 2023

# ── Channel definitions ───────────────────────────────────────────────────────
# (channel_name, filename, scale_factor, offset)
# Physical value = raw_DN * scale + offset
# Notes:
#   MODIS NDVI/EVI raw DN: ×0.0001 → real index (−0.2 to 1.0)
#   MODIS ET raw DN:        ×0.1   → mm (approx monthly total)
#   MODIS LST raw DN:       ×0.02  → Kelvin, then −273.15 → °C
#   ERA5 temperature/dewpoint: in Kelvin → −273.15 → °C
#   CHIRPS, SMAP, wind: already in physical units

DYNAMIC_CHANNELS: list[tuple[str, str, float, float]] = [
    ("rain_mm",    "rain_mm_monthly_2003_2024.tif",    1.0,      0.0),
    ("t2m_c",      "t2m_k_monthly_2003_2024.tif",      1.0,   -273.15),
    ("ndvi",       "ndvi_monthly_2003_2024.tif",       1e-4,     0.0),
    ("evi",        "evi_monthly_2003_2024.tif",        1e-4,     0.0),
    ("et_mm",      "et_monthly_2003_2024.tif",         0.1,      0.0),
    ("lst_c",      "lst_day_k_monthly_2003_2024.tif",  0.02,  -273.15),
    ("sm_surf",    "sm_surf_monthly_2015_2024.tif",    1.0,      0.0),  # 106 bands, starts Mar 2015
    ("dewpoint_c", "dewpoint_k_monthly_2003_2024.tif", 1.0,   -273.15),
    ("wind_ms",    "wind_speed_monthly_2003_2024.tif", 1.0,      0.0),
]

# (channel_name, filename, band_index_in_file)
STATIC_CHANNELS: list[tuple[str, str, int]] = [
    ("elevation_m",   "static_elev_slope.tif", 0),
    ("slope_deg",     "static_elev_slope.tif", 1),
    ("landcover_int", "static_landcover.tif",  0),
]

# Channel indices within the dynamic array (used by build_labels)
CH = {name: i for i, (name, *_) in enumerate(DYNAMIC_CHANNELS)}


def load_tif(path: Path | str) -> np.ndarray:
    """Load a GeoTIFF → (bands, H, W) float32, NaN where masked."""
    da = rxr.open_rasterio(path, masked=True)
    return da.values.astype(np.float32)


def build_dynamic(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all 9 dynamic channels → (T=252, C=9, H, W) float32.

    SMAP has only 138 bands (January 2015 onward). We create a full 282-band
    array pre-filled with NaN and place the SMAP values at t=144..281.

    Returns
    -------
    dynamic : (282, 9, 182, 188) float32
    mask    : (182, 188) bool — True = pixel has at least one valid rain value
    """
    n_ch = len(DYNAMIC_CHANNELS)
    cube = np.full((T, n_ch, H, W), np.nan, dtype=np.float32)

    for c, (name, fname, scale, offset) in enumerate(DYNAMIC_CHANNELS):
        raw = load_tif(data_dir / fname)   # (B, H, W)
        arr = (raw * scale + offset)

        if name == "sm_surf":
            # Insert 138-band SMAP starting at t=144
            cube[SMAP_START : SMAP_START + SMAP_N, c] = arr
        else:
            cube[:, c] = arr               # (282, H, W)

    # Derive land mask from CHIRPS: valid if ANY month has rain data
    mask = ~np.all(np.isnan(cube[:, CH["rain_mm"]]), axis=0)   # (H, W)
    return cube, mask


def build_static(data_dir: Path) -> np.ndarray:
    """Load 3 static channels → (3, H, W) float32."""
    out = np.full((len(STATIC_CHANNELS), H, W), np.nan, dtype=np.float32)
    for c, (name, fname, band_idx) in enumerate(STATIC_CHANNELS):
        raw = load_tif(data_dir / fname)   # (B, H, W)
        out[c] = raw[band_idx]
    return out


def build_labels(
    dynamic: np.ndarray,
    mask: np.ndarray,
    train_end: int = TRAIN_END,
) -> np.ndarray:
    """Compute SPI-3, NDVI anomaly, SM anomaly from the dynamic cube.

    All three labels are computed on physical-unit arrays (after scale factors
    have already been applied in build_dynamic).

    Parameters
    ----------
    dynamic   : (T, 9, H, W) float32
    mask      : (H, W) bool
    train_end : fit normalization stats on indices 0 … train_end-1 only

    Returns
    -------
    (T, 3, H, W) float32 — channels: [spi3, ndvi_anom, sm_anom]
    """
    print("  Computing SPI-3 (gamma fit per pixel per calendar month) ...")
    label_spi3 = spi3(
        dynamic[:, CH["rain_mm"]],
        train_end=train_end,
        land_mask=mask,
    )

    print("  Computing NDVI anomaly ...")
    label_ndvi = anomaly(
        dynamic[:, CH["ndvi"]],
        train_end=train_end,
        land_mask=mask,
    )

    print("  Computing SM anomaly ...")
    label_sm = anomaly(
        dynamic[:, CH["sm_surf"]],
        train_end=train_end,
        land_mask=mask,
    )

    return np.stack([label_spi3, label_ndvi, label_sm], axis=1)  # (T, 3, H, W)


def assemble(data_dir: str | Path, out_dir: str | Path) -> None:
    """Full pipeline: load → scale → labels → save.

    Writes to out_dir:
        dynamic.npy    (282, 9, 182, 188)  float32
        static.npy     (3, 182, 188)       float32
        labels.npy     (282, 3, 182, 188)  float32
        mask.npy       (182, 188)          bool
        meta.json
    """
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dynamic channels ...")
    dynamic, mask = build_dynamic(data_dir)
    print(f"  shape {dynamic.shape}  valid pixels: {mask.sum():,}")

    print("Loading static channels ...")
    static = build_static(data_dir)
    print(f"  shape {static.shape}")

    print("Computing labels ...")
    labels = build_labels(dynamic, mask)
    print(f"  shape {labels.shape}")

    print("Saving cube ...")
    np.save(out_dir / "dynamic.npy", dynamic)
    np.save(out_dir / "static.npy",  static)
    np.save(out_dir / "labels.npy",  labels)
    np.save(out_dir / "mask.npy",    mask)

    meta = {
        "grid": {
            "min_lon": _GRID.min_lon,
            "min_lat": _GRID.min_lat,
            "max_lon": _GRID.max_lon,
            "max_lat": _GRID.max_lat,
            "res": _GRID.res,
            "H": H,
            "W": W,
        },
        "time": {"start": "2003-01", "end": "2023-12", "T": T},
        "dynamic_channels": [ch[0] for ch in DYNAMIC_CHANNELS],
        "static_channels":  [ch[0] for ch in STATIC_CHANNELS],
        "label_channels":   ["spi3", "ndvi_anom", "sm_anom"],
        "splits": {
            "train_end_idx":   TRAIN_END,
            "train_end_date":  "2021-12",
            "test_start_idx":  TRAIN_END,
            "test_start_date": "2022-01",
        },
        "smap": {
            "start_idx":  SMAP_START,
            "start_date": "2015-01",
            "n_bands":    SMAP_N,
        },
        "scale_notes": {
            "ndvi":  "raw DN × 1e-4 → dimensionless index",
            "evi":   "raw DN × 1e-4 → dimensionless index",
            "et_mm": "raw DN × 0.1 → approx mm/month",
            "lst_c": "raw DN × 0.02 − 273.15 → °C",
            "t2m_c": "ERA5 Kelvin − 273.15 → °C",
            "dewpoint_c": "ERA5 Kelvin − 273.15 → °C",
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    dyn_mb = dynamic.nbytes / 1e6
    lbl_mb = labels.nbytes / 1e6
    print(f"\nDone.  Cube written to {out_dir}/")
    print(f"  dynamic : {dyn_mb:.0f} MB")
    print(f"  labels  : {lbl_mb:.0f} MB")
    print(f"  total   : {dyn_mb + lbl_mb:.0f} MB (approx)")
