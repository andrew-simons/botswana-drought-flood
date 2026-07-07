"""Change ACTIVE_MODEL here to switch the map data source — nothing else needs editing."""

from __future__ import annotations

# ── Model selection ───────────────────────────────────────────────────────────
ACTIVE_MODEL: str = "xgboost"   # "xgboost" | "convlstm"

MODEL_FILE = {
    "xgboost":  "xgb_preds.npy",
    "convlstm": "convlstm_preds.npy",
}

MODEL_DISPLAY_NAME = {
    "xgboost":  "XGBoost",
    "convlstm": "ConvLSTM",
}

# ── Drought thresholds (z-score units, same as training labels) ───────────────
DROUGHT_ONSET  = -0.52   # 30th percentile — drought begins
DROUGHT_SEVERE = -1.28   # 10th percentile — severe drought

# ── Label metadata ────────────────────────────────────────────────────────────
LABEL_KEYS = ["spi3", "ndvi_anom", "sm_anom"]

LABEL_DISPLAY = {
    "spi3":      "Drought Risk (Rainfall)",
    "ndvi_anom": "Vegetation Health",
    "sm_anom":   "Soil Moisture",
}

LABEL_HELP = {
    "spi3":      "How much rainfall has fallen compared to the long-term average. Negative = drier than normal.",
    "ndvi_anom": "How green the vegetation is compared to the historical average for this time of year.",
    "sm_anom":   "How much water is in the soil compared to the historical average for this time of year.",
}

# ── Risk category thresholds (applied to any label) ──────────────────────────
def risk_category(value: float) -> str:
    if value < DROUGHT_SEVERE:
        return "Severe drought"
    if value < DROUGHT_ONSET:
        return "Moderate drought"
    if value > 0.52:
        return "Wetter than normal"
    return "Normal"

RISK_COLOR = {
    "Severe drought":    "#8B0000",
    "Moderate drought":  "#CC4400",
    "Normal":            "#14532D",
    "Wetter than normal": "#1A3A5C",
}

RISK_EMOJI = {
    "Severe drought":    "🔴",
    "Moderate drought":  "🟠",
    "Normal":            "🟢",
    "Wetter than normal": "🔵",
}

# ── Citizen-facing action guidance per risk level ─────────────────────────────
ACTION_GUIDANCE: dict[str, dict] = {
    "Severe drought": {
        "icon": "🔴",
        "headline": "Drought Alert — take action now",
        "steps": [
            "Delay new plantings — soil moisture is critically low.",
            "Reduce livestock numbers if water sources (boreholes, dams) are running low.",
            "Contact your District Administration office or the Ministry of Agriculture for drought relief.",
            "Harvest and store any available rainwater immediately.",
        ],
    },
    "Moderate drought": {
        "icon": "🟠",
        "headline": "Drought Watch — monitor closely",
        "steps": [
            "Delay non-essential planting until rainfall improves.",
            "Monitor borehole and dam levels weekly.",
            "Reduce irrigation and focus water on priority crops.",
        ],
    },
    "Normal": {
        "icon": "🟢",
        "headline": "Conditions normal — no action required",
        "steps": [
            "Conditions are within the seasonal range.",
            "Continue routine farming and water management practices.",
        ],
    },
    "Wetter than normal": {
        "icon": "🔵",
        "headline": "Wetter than normal — good for planting",
        "steps": [
            "Good soil moisture — suitable for planting if temperatures allow.",
            "Check low-lying areas and fields for waterlogging.",
            "Ensure drainage channels and culverts are clear.",
        ],
    },
}

# ── EM-DAT documented Botswana drought events ─────────────────────────────────
EMDAT_EVENTS = [
    {"name": "2015–16 El Niño drought", "start": "2015-07", "end": "2016-06"},
    {"name": "2023–24 drought",          "start": "2023-01", "end": "2024-06"},
]
