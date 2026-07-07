"""Load and cache pre-computed prediction arrays for the app.

All arrays live in data/app/ (relative to repo root).
st.cache_data ensures each file is read from disk only once per session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import ACTIVE_MODEL, MODEL_FILE

_REPO = Path(__file__).parent.parent
DATA_DIR   = _REPO / "data" / "app"
ASSETS_DIR = Path(__file__).parent / "assets"


def data_ready() -> bool:
    return (DATA_DIR / MODEL_FILE[ACTIVE_MODEL]).exists()


@st.cache_data(show_spinner="Loading predictions…")
def load_predictions() -> np.ndarray:
    """(T, 3, H, W) float32 — model predictions for the test period."""
    return np.load(DATA_DIR / MODEL_FILE[ACTIVE_MODEL])


@st.cache_data(show_spinner=False)
def load_targets() -> np.ndarray:
    """(T, 3, H, W) float32 — actual observed labels for the test period."""
    return np.load(DATA_DIR / "targets.npy")


@st.cache_data(show_spinner=False)
def load_mask() -> np.ndarray:
    """(H, W) bool — True = valid land pixel."""
    return np.load(DATA_DIR / "mask.npy")


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    with open(DATA_DIR / "meta.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_districts() -> dict | None:
    path = ASSETS_DIR / "botswana_districts.geojson"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def make_date_labels(meta: dict) -> list[str]:
    """Human-readable month labels, e.g. ['Jan 2022', 'Feb 2022', ...]."""
    dates = pd.date_range(meta["test_start"], periods=meta["T_test"], freq="MS")
    return [d.strftime("%b %Y") for d in dates]


def latlon_to_grid(lat: float, lon: float, meta: dict) -> tuple[int, int]:
    """Convert a clicked lat/lon to (row, col) grid indices."""
    g = meta["grid"]
    # Row 0 is the northernmost row (GEE convention)
    row = int((g["max_lat"] - lat) / g["res"])
    col = int((lon - g["min_lon"]) / g["res"])
    row = int(np.clip(row, 0, g["H"] - 1))
    col = int(np.clip(col, 0, g["W"] - 1))
    return row, col
