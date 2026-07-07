"""Plotly charts: pixel time series and district alert table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DROUGHT_ONSET, DROUGHT_SEVERE, EMDAT_EVENTS,
    LABEL_DISPLAY, LABEL_KEYS, RISK_COLOR, risk_category,
)

# ── Design tokens (mirrors CSS custom properties) ──────────────────────────────
_C_OBSERVED   = "#006D77"   # deep teal — observed data
_C_PREDICTED  = "#E29B36"   # warm golden — model prediction
_C_ONSET      = "#CC4400"   # burnt orange — drought onset threshold
_C_SEVERE     = "#8B0000"   # dark red — severe drought threshold
_C_EMDAT      = "rgba(140, 50, 0, 0.07)"
_C_GRID       = "#E4E8E2"
_C_ZEROLINE   = "#BEC4BA"
_C_PLOT_BG    = "#FAFAF8"
_FONT_STACK   = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
_C_TICK       = "#5A6057"
_C_INK        = "#1C1C1E"


def _base_layout(title: str, date_labels: list[str], x: list[int]) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(family=_FONT_STACK, size=13, color=_C_INK),
            x=0,
            pad=dict(l=0, t=4),
        ),
        font=dict(family=_FONT_STACK, size=12, color=_C_INK),
        plot_bgcolor=_C_PLOT_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickmode="array",
            tickvals=x[::3],
            ticktext=date_labels[::3],
            title="",
            tickfont=dict(size=11, color=_C_TICK),
            gridcolor=_C_GRID,
            gridwidth=0.5,
            zeroline=False,
            showline=True,
            linecolor=_C_GRID,
            linewidth=1,
        ),
        yaxis=dict(
            title="Anomaly (z-score)",
            title_font=dict(size=11, color=_C_TICK),
            tickfont=dict(size=11, color=_C_TICK),
            gridcolor=_C_GRID,
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor=_C_ZEROLINE,
            zerolinewidth=1,
            showline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(size=11, color=_C_INK),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        height=320,
        margin=dict(l=56, r=110, t=48, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=_C_GRID,
            font=dict(family=_FONT_STACK, size=12, color=_C_INK),
        ),
    )


def _add_emdat_shading(fig: go.Figure, date_labels: list[str]) -> None:
    for event in EMDAT_EVENTS:
        try:
            ev_dates  = pd.date_range(event["start"], event["end"], freq="MS")
            ev_labels = [d.strftime("%b %Y") for d in ev_dates]
            x0 = next((i for i, d in enumerate(date_labels) if d in ev_labels), None)
            x1 = max((i for i, d in enumerate(date_labels) if d in ev_labels), default=None)
            if x0 is not None and x1 is not None:
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=_C_EMDAT,
                    line_width=0,
                    annotation_text=event["name"],
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color=_C_ONSET,
                )
        except Exception:
            pass


def _add_threshold_lines(fig: go.Figure) -> None:
    fig.add_hline(
        y=DROUGHT_ONSET,
        line_dash="dash",
        line_color=_C_ONSET,
        line_width=1.5,
        annotation_text="Drought onset",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color=_C_ONSET,
    )
    fig.add_hline(
        y=DROUGHT_SEVERE,
        line_dash="dot",
        line_color=_C_SEVERE,
        line_width=1.5,
        annotation_text="Severe",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color=_C_SEVERE,
    )


def pixel_timeseries(
    preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    row: int,
    col: int,
    date_labels: list[str],
    label_key: str,
) -> go.Figure | None:
    """Line chart: model prediction vs observed for one pixel over the test period."""
    c = LABEL_KEYS.index(label_key)
    pred_vals   = preds[:, c, row, col]
    target_vals = targets[:, c, row, col]

    if not mask[row, col]:
        return None

    x   = list(range(len(date_labels)))
    fig = go.Figure()

    _add_emdat_shading(fig, date_labels)
    _add_threshold_lines(fig)

    fig.add_trace(go.Scatter(
        x=x, y=target_vals,
        name="Observed",
        mode="lines+markers",
        line=dict(color=_C_OBSERVED, width=2.5),
        marker=dict(size=4, color=_C_OBSERVED),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_vals,
        name="Model prediction",
        mode="lines+markers",
        line=dict(color=_C_PREDICTED, width=2.5, dash="dot"),
        marker=dict(size=4, color=_C_PREDICTED),
    ))

    fig.update_layout(**_base_layout(
        f"{LABEL_DISPLAY[label_key]} — selected location",
        date_labels,
        x,
    ))
    return fig


def alert_table(
    preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    month_idx: int,
    label_key: str,
    districts: dict | None,
    meta: dict,
) -> pd.DataFrame:
    """Return a DataFrame of drought status per district for the selected month."""
    c = LABEL_KEYS.index(label_key)
    arr = preds[month_idx, c]

    if districts is None:
        vals = arr[mask]
        mean_val = float(np.nanmean(vals))
        return pd.DataFrame([{
            "District": "Botswana (national)",
            "Index value": round(mean_val, 2),
            "Status": risk_category(mean_val),
        }])

    g = meta["grid"]
    H, W = g["H"], g["W"]

    lats = g["max_lat"] - (np.arange(H) + 0.5) * g["res"]
    lons = g["min_lon"] + (np.arange(W) + 0.5) * g["res"]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    rows = []
    for feature in districts.get("features", []):
        name = (
            feature.get("properties", {}).get("NAME_1")
            or feature.get("properties", {}).get("name")
            or "Unknown"
        )
        geom_type = feature["geometry"]["type"]
        coords     = feature["geometry"]["coordinates"]

        try:
            from matplotlib.path import Path as MplPath
            inside = np.zeros(H * W, dtype=bool)
            polys = coords if geom_type == "MultiPolygon" else [coords]
            for poly in polys:
                ring = np.array(poly[0])
                path = MplPath(ring[:, :2])
                inside |= path.contains_points(pts)
            inside &= mask.ravel()
            if not inside.any():
                continue
            mean_val = float(np.nanmean(arr.ravel()[inside]))
            rows.append({
                "District": name,
                "Index value": round(mean_val, 2),
                "Status": risk_category(mean_val),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("Index value").reset_index(drop=True)
    return df


def _style_status_cell(val: str) -> str:
    styles = {
        "Severe drought":    "background-color:#FEF2F2; color:#7F1D1D; font-weight:600",
        "Moderate drought":  "background-color:#FFFBEB; color:#78350F; font-weight:600",
        "Normal":            "background-color:#F0FDF4; color:#14532D; font-weight:600",
        "Wetter than normal":"background-color:#EFF6FF; color:#1A3A5C; font-weight:600",
    }
    return styles.get(val, "")


def get_district_pixels(
    district_name: str,
    districts: dict | None,
    meta: dict,
    mask: np.ndarray,
) -> np.ndarray | None:
    """Return a (H*W,) bool array marking pixels inside the named district."""
    if districts is None:
        return None
    g = meta["grid"]
    H, W = g["H"], g["W"]
    lats = g["max_lat"] - (np.arange(H) + 0.5) * g["res"]
    lons = g["min_lon"] + (np.arange(W) + 0.5) * g["res"]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    for feature in districts.get("features", []):
        name = (
            feature.get("properties", {}).get("NAME_1")
            or feature.get("properties", {}).get("name")
            or ""
        )
        if name != district_name:
            continue
        geom_type = feature["geometry"]["type"]
        coords    = feature["geometry"]["coordinates"]
        try:
            from matplotlib.path import Path as MplPath
            inside = np.zeros(H * W, dtype=bool)
            polys = coords if geom_type == "MultiPolygon" else [coords]
            for poly in polys:
                ring = np.array(poly[0])
                inside |= MplPath(ring[:, :2]).contains_points(pts)
            inside &= mask.ravel()
            return inside
        except Exception:
            return None
    return None


def district_timeseries(
    preds: np.ndarray,
    targets: np.ndarray,
    district_pixels: np.ndarray,
    date_labels: list[str],
    label_key: str,
) -> go.Figure | None:
    """Line chart: district-average prediction vs observed for one label channel."""
    c = LABEL_KEYS.index(label_key)
    T = preds.shape[0]
    pred_vals   = np.array([float(np.nanmean(preds[t,   c].ravel()[district_pixels])) for t in range(T)])
    target_vals = np.array([float(np.nanmean(targets[t, c].ravel()[district_pixels])) for t in range(T)])

    x   = list(range(T))
    fig = go.Figure()

    _add_emdat_shading(fig, date_labels)
    _add_threshold_lines(fig)

    fig.add_trace(go.Scatter(
        x=x, y=target_vals, name="Observed",
        mode="lines+markers",
        line=dict(color=_C_OBSERVED, width=2.5),
        marker=dict(size=4, color=_C_OBSERVED),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_vals, name="Model prediction",
        mode="lines+markers",
        line=dict(color=_C_PREDICTED, width=2.5, dash="dot"),
        marker=dict(size=4, color=_C_PREDICTED),
    ))

    fig.update_layout(**_base_layout(
        f"{LABEL_DISPLAY[label_key]} — district average",
        date_labels,
        x,
    ))
    return fig
