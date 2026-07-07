"""Botswana Drought Early Warning System — Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ACTION_GUIDANCE, ACTIVE_MODEL, LABEL_DISPLAY, LABEL_HELP,
    LABEL_KEYS, MODEL_DISPLAY_NAME, RISK_EMOJI, risk_category,
)
from data_loader import (
    data_ready, latlon_to_grid, load_districts,
    load_mask, load_meta, load_predictions, load_targets,
    make_date_labels,
)
from map_utils import build_map
from chart_utils import (
    _style_status_cell, alert_table, district_timeseries,
    get_district_pixels, pixel_timeseries,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Botswana Drought Watch",
    page_icon="🇧🇼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bds-font: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
  --bds-teal: #0A6C74;
  --bds-bg: #F8F9F6;
  --bds-ink: #1C1C1E;
  --bds-ink-muted: #5A6057;
  --bds-border: #D4D9D2;
}

/* Font override */
.stApp, .stApp * { font-family: var(--bds-font) !important; -webkit-font-smoothing: antialiased; }

/* Hide Streamlit chrome — keep hamburger accessible in dev via ?toolbar=dev */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }

/* Main container */
.block-container {
  padding-top: 28px !important;
  padding-bottom: 76px !important;
  max-width: 1180px !important;
}

/* ── Hero ──────────────────────────────────────────────────────────────────── */
.bds-hero { padding: 0 0 16px; }
.bds-hero-title {
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--bds-teal);
  letter-spacing: -0.025em;
  margin: 0 0 5px;
  line-height: 1.15;
  display: flex;
  align-items: center;
  gap: 10px;
}
.bds-hero-sub {
  font-size: 0.875rem;
  color: var(--bds-ink-muted);
  margin: 0;
  line-height: 1.5;
}
.bds-divider { height: 1px; background: var(--bds-border); margin: 4px 0 20px; }

/* ── Alert cards ───────────────────────────────────────────────────────────── */
.bds-alert {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 13px 18px;
  border-radius: 6px;
  border: 1px solid;
  margin: 0 0 12px;
  line-height: 1.5;
}
.bds-alert-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
.bds-alert-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  margin-bottom: 2px;
}
.bds-alert-title  { font-weight: 700; font-size: 0.95rem; margin-bottom: 3px; line-height: 1.3; }
.bds-alert-body   { font-size: 0.85rem; line-height: 1.5; }
.bds-alert-severe   { background:#FEF2F2; border-color:#FECACA; color:#7F1D1D; }
.bds-alert-moderate { background:#FFFBEB; border-color:#FDE68A; color:#78350F; }
.bds-alert-normal   { background:#F0FDF4; border-color:#BBF7D0; color:#14532D; }
.bds-alert-wet      { background:#EFF6FF; border-color:#BFDBFE; color:#1A3A5C; }

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
.bds-sidebar-brand {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 2px 0 14px;
  border-bottom: 1px solid var(--bds-border);
  margin-bottom: 2px;
}
.bds-sidebar-brand-name {
  font-weight: 700;
  font-size: 0.95rem;
  color: var(--bds-teal);
  letter-spacing: -0.01em;
  line-height: 1.2;
}
.bds-sidebar-brand-model {
  font-size: 0.7rem;
  color: var(--bds-ink-muted);
  margin-top: 1px;
  font-weight: 500;
}
.bds-sidebar-section {
  font-size: 0.67rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  color: var(--bds-ink-muted) !important;
  padding: 14px 0 1px !important;
  margin: 0 0 2px !important;
  line-height: 1.3 !important;
}
.bds-current-month {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--bds-teal);
  margin: -4px 0 6px;
}
.bds-index-help {
  font-size: 0.78rem;
  color: var(--bds-ink-muted);
  font-style: italic;
  margin: -2px 0 6px;
  line-height: 1.4;
}

/* ── Data coverage block ───────────────────────────────────────────────────── */
.bds-coverage {
  margin-top: 20px;
  padding: 10px 12px;
  background: rgba(10, 108, 116, 0.06);
  border: 1px solid rgba(10, 108, 116, 0.18);
  border-radius: 5px;
  font-size: 0.75rem;
  line-height: 1.7;
  color: var(--bds-ink-muted);
}
.bds-coverage-title {
  font-size: 0.67rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--bds-teal);
  margin-bottom: 4px;
}
.bds-coverage-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}
.bds-coverage-label { color: var(--bds-ink-muted); }
.bds-coverage-value { font-weight: 600; color: var(--bds-ink); }

/* ── Tabs ──────────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
  border-bottom: 2px solid var(--bds-border) !important;
  gap: 0 !important;
  background: transparent !important;
}
[data-baseweb="tab"] {
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  padding: 9px 18px !important;
  color: var(--bds-ink-muted) !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -2px !important;
  background: transparent !important;
}
[data-baseweb="tab"][aria-selected="true"] {
  color: var(--bds-teal) !important;
  border-bottom-color: var(--bds-teal) !important;
  font-weight: 600 !important;
}

/* ── Expander ──────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--bds-border) !important;
  border-radius: 6px !important;
  overflow: hidden;
}

/* ── Dataframe ─────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--bds-border) !important;
  border-radius: 6px !important;
  overflow: hidden;
}

/* ── Sticky footer ─────────────────────────────────────────────────────────── */
.bds-footer {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(238, 240, 236, 0.96);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border-top: 1px solid var(--bds-border);
  padding: 8px 28px;
  font-size: 0.71rem;
  color: var(--bds-ink-muted);
  z-index: 200;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: alert card HTML ────────────────────────────────────────────────────
def _render_alert(nat_risk: str, date_label: str) -> None:
    css_class = {
        "Severe drought":    "bds-alert-severe",
        "Moderate drought":  "bds-alert-moderate",
        "Normal":            "bds-alert-normal",
        "Wetter than normal":"bds-alert-wet",
    }[nat_risk]
    icon = {"Severe drought":"⚠️","Moderate drought":"🟠","Normal":"✅","Wetter than normal":"💧"}[nat_risk]
    badge = {
        "Severe drought":    "Drought Alert",
        "Moderate drought":  "Drought Watch",
        "Normal":            "All Clear",
        "Wetter than normal":"Above Normal",
    }[nat_risk]
    title_txt = {
        "Severe drought":    f"Severe conditions — {date_label}",
        "Moderate drought":  f"Drier than normal — {date_label}",
        "Normal":            f"Normal conditions — {date_label}",
        "Wetter than normal":f"Wetter than normal — {date_label}",
    }[nat_risk]
    body_txt = {
        "Severe drought":
            "Botswana-wide rainfall and soil moisture are critically below the seasonal norm. "
            "Review district breakdown and action guidance below.",
        "Moderate drought":
            "Conditions are below the seasonal average across much of Botswana. "
            "Monitor water sources and planting timing closely.",
        "Normal":
            "All three indices are within the expected seasonal range across Botswana.",
        "Wetter than normal":
            "Rainfall and soil moisture are above the seasonal average. "
            "Check drainage in low-lying areas.",
    }[nat_risk]
    st.markdown(f"""
    <div class="bds-alert {css_class}">
      <div class="bds-alert-icon">{icon}</div>
      <div>
        <div class="bds-alert-label">{badge}</div>
        <div class="bds-alert-title">{title_txt}</div>
        <div class="bds-alert-body">{body_txt}</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── Guard: data not yet generated ─────────────────────────────────────────────
if not data_ready():
    st.error(
        "**App data not found.** Run the export script first:\n\n"
        "```\npython scripts/export_app_data.py\n```\n\n"
        "This generates the pre-computed predictions in `data/app/`."
    )
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
preds       = load_predictions()
targets     = load_targets()
mask        = load_mask()
meta        = load_meta()
districts   = load_districts()
date_labels = make_date_labels(meta)
T           = len(date_labels)

# ── Session state ──────────────────────────────────────────────────────────────
if "clicked_row" not in st.session_state:
    st.session_state.clicked_row = None
    st.session_state.clicked_col = None
if "selected_district" not in st.session_state:
    st.session_state.selected_district = None

# ── District name list ─────────────────────────────────────────────────────────
district_names: list[str] = []
if districts is not None:
    district_names = sorted({
        f.get("properties", {}).get("NAME_1") or f.get("properties", {}).get("name")
        for f in districts.get("features", [])
        if f.get("properties", {}).get("NAME_1") or f.get("properties", {}).get("name")
    })

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="bds-sidebar-brand">
      <span style="font-size:1.5rem;line-height:1">🇧🇼</span>
      <div>
        <div class="bds-sidebar-brand-name">Botswana Drought Watch</div>
        <div class="bds-sidebar-brand-model">{MODEL_DISPLAY_NAME[ACTIVE_MODEL]} model</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="bds-sidebar-section">Time period</div>', unsafe_allow_html=True)
    month_idx = st.slider(
        "Month",
        min_value=0, max_value=T - 1, value=T - 1,
        format="%d",
        help="Scrub through available months",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="bds-current-month">{date_labels[month_idx]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bds-sidebar-section">Drought index</div>', unsafe_allow_html=True)
    label_choice = st.radio(
        "Drought index",
        options=LABEL_KEYS,
        format_func=lambda k: LABEL_DISPLAY[k],
        help="Which measurement to show on the map",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="bds-index-help">{LABEL_HELP[label_choice]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bds-sidebar-section">Data source</div>', unsafe_allow_html=True)
    view_choice = st.radio(
        "Map data",
        options=["prediction", "observed"],
        format_func=lambda v: "Model prediction" if v == "prediction" else "Observed (actual)",
        label_visibility="collapsed",
    )

    # ── Data coverage block ────────────────────────────────────────────────────
    _cov_start   = meta.get("test_start", "—")
    _cov_t       = meta.get("T_test", 0)
    _cov_updated = meta.get("last_updated", "—")
    if _cov_start != "—" and _cov_t:
        import pandas as _pd
        _end_dt  = _pd.date_range(_cov_start, periods=_cov_t, freq="MS")[-1]
        _cov_end = _end_dt.strftime("%b %Y")
        _cov_start_fmt = _pd.Timestamp(_cov_start).strftime("%b %Y")
    else:
        _cov_end = "—"
        _cov_start_fmt = _cov_start

    st.markdown(f"""
    <div class="bds-coverage">
      <div class="bds-coverage-title">Dataset info</div>
      <div class="bds-coverage-row">
        <span class="bds-coverage-label">Coverage</span>
        <span class="bds-coverage-value">{_cov_start_fmt} – {_cov_end}</span>
      </div>
      <div class="bds-coverage-row">
        <span class="bds-coverage-label">Last updated</span>
        <span class="bds-coverage-value">{_cov_updated}</span>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bds-hero">
  <h1 class="bds-hero-title">🇧🇼 Botswana Drought Watch</h1>
  <p class="bds-hero-sub">
    Satellite-based drought monitoring &middot; Updated monthly &middot;
    Select a district or click the map to explore
  </p>
</div>
<div class="bds-divider"></div>""", unsafe_allow_html=True)

# ── National risk + alert card ─────────────────────────────────────────────────
label_c      = LABEL_KEYS.index(label_choice)
arr_map      = (preds if view_choice == "prediction" else targets)[month_idx, label_c]
national_val = float(np.nanmean(arr_map[mask]))
nat_risk     = risk_category(national_val)

_render_alert(nat_risk, date_labels[month_idx])

# ── Action guidance ────────────────────────────────────────────────────────────
guidance  = ACTION_GUIDANCE[nat_risk]
auto_open = nat_risk in ("Severe drought", "Moderate drought")
with st.expander(
    f"{guidance['icon']} What should I do? — {guidance['headline']}",
    expanded=auto_open,
):
    for i, step in enumerate(guidance["steps"], 1):
        st.markdown(f"**{i}.** {step}")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── District quick-select ──────────────────────────────────────────────────────
if district_names:
    col_pick, col_clear = st.columns([5, 1])
    with col_pick:
        chosen = st.selectbox(
            "Select your district:",
            options=["— choose a district —"] + district_names,
            index=0,
            label_visibility="collapsed",
        )
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.selected_district = None
            st.session_state.clicked_row        = None
            st.session_state.clicked_col        = None

    if chosen != "— choose a district —":
        st.session_state.selected_district = chosen
        st.session_state.clicked_row        = None
        st.session_state.clicked_col        = None

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_map, tab_districts, tab_trend = st.tabs(["Map", "Districts", "Trend"])

# ── Map tab ────────────────────────────────────────────────────────────────────
with tab_map:
    st.caption("Click any location inside Botswana to load its trend in the **Trend** tab.")
    m = build_map(
        arr=arr_map,
        mask=mask,
        meta=meta,
        districts=districts,
        label_key=label_choice,
        month_label=date_labels[month_idx],
    )
    map_data = st_folium(m, width=None, height=540, returned_objects=["last_clicked"])

    clicked = map_data.get("last_clicked") if map_data else None
    if clicked:
        lat, lon   = clicked["lat"], clicked["lng"]
        row, c_idx = latlon_to_grid(lat, lon, meta)
        if mask[row, c_idx]:
            st.session_state.clicked_row       = row
            st.session_state.clicked_col       = c_idx
            st.session_state.selected_district = None
            g           = meta["grid"]
            actual_lat  = g["max_lat"] - (row + 0.5) * g["res"]
            actual_lon  = g["min_lon"] + (c_idx + 0.5) * g["res"]
            st.success(
                f"📍 Location selected: **{actual_lat:.2f}°S, {actual_lon:.2f}°E** "
                f"— open the **Trend** tab to see the time series."
            )
        else:
            st.info("That location is outside Botswana's land area. Try clicking inside the country.")

# ── Districts tab ──────────────────────────────────────────────────────────────
with tab_districts:
    st.markdown(f"**District status — {date_labels[month_idx]}**")
    df = alert_table(preds, targets, mask, month_idx, label_choice, districts, meta)
    if not df.empty:
        styled = df[["District", "Index value", "Status"]].style.map(
            _style_status_cell, subset=["Status"]
        ).format({"Index value": "{:.2f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("District boundaries not available. Run the export script to fetch GeoJSON.")

# ── Trend tab ──────────────────────────────────────────────────────────────────
with tab_trend:
    if st.session_state.clicked_row is not None:
        row   = st.session_state.clicked_row
        c_idx = st.session_state.clicked_col
        g     = meta["grid"]
        actual_lat = g["max_lat"] - (row + 0.5) * g["res"]
        actual_lon = g["min_lon"] + (c_idx + 0.5) * g["res"]
        st.markdown(f"**{actual_lat:.2f}°S, {actual_lon:.2f}°E**")
        fig = pixel_timeseries(preds, targets, mask, row, c_idx, date_labels, label_choice)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.selected_district is not None:
        dist_name = st.session_state.selected_district
        st.markdown(f"**{dist_name} District**")
        pixels = get_district_pixels(dist_name, districts, meta, mask)
        if pixels is not None and pixels.any():
            fig = district_timeseries(preds, targets, pixels, date_labels, label_choice)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No land pixels found for this district.")

    else:
        st.info(
            "No location selected yet.\n\n"
            "- Use the **district dropdown** above to pick your area, or\n"
            "- Click any point on the **Map** tab."
        )

# ── Sticky footer ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="bds-footer">
  Data:&nbsp;CHIRPS &middot; MODIS &middot; SMAP &middot; ERA5-Land &middot; ESA WorldCover
  &nbsp;&middot;&nbsp; Events: EM-DAT
  &nbsp;&middot;&nbsp; Processed with Google Earth Engine &nbsp;&middot;&nbsp; Model: PyTorch
</div>""", unsafe_allow_html=True)
