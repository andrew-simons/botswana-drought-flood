"""Build the Folium map: prediction image overlay + district boundaries."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import folium
import matplotlib
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DROUGHT_ONSET, DROUGHT_SEVERE, LABEL_DISPLAY


# Colormap: red (drought) → yellow → green (normal) → blue (wet)
_CMAP = matplotlib.colormaps["RdYlGn"]
_VMIN, _VMAX = -2.5, 2.5


def _array_to_png_b64(arr: np.ndarray, mask: np.ndarray) -> str:
    """Convert (H, W) float array + mask to a base64 RGBA PNG for ImageOverlay."""
    normed = np.clip((arr - _VMIN) / (_VMAX - _VMIN), 0.0, 1.0)
    rgba = (_CMAP(normed) * 255).astype(np.uint8)   # (H, W, 4)
    rgba[~mask] = [0, 0, 0, 0]                       # transparent outside Botswana
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _district_style(feature):
    return {
        "fillColor":   "transparent",
        "color":       "#333333",
        "weight":      1.2,
        "fillOpacity": 0.0,
    }


def _district_highlight(feature):
    return {"color": "#000000", "weight": 2.5}


def build_map(
    arr: np.ndarray,
    mask: np.ndarray,
    meta: dict,
    districts: dict | None,
    label_key: str,
    month_label: str,
) -> folium.Map:
    """Return a Folium map with prediction overlay and district labels.

    Parameters
    ----------
    arr         : (H, W) float32 — prediction for one month and one label channel
    mask        : (H, W) bool
    meta        : app meta.json dict
    districts   : GeoJSON FeatureCollection or None
    label_key   : e.g. "spi3"
    month_label : human-readable label shown in the map title, e.g. "Jan 2022"
    """
    g = meta["grid"]
    south, north = g["min_lat"], g["max_lat"]
    west,  east  = g["min_lon"], g["max_lon"]
    centre_lat = (south + north) / 2
    centre_lon = (west  + east)  / 2

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # ── Prediction image overlay ──────────────────────────────────────────────
    # Row 0 in the array is the northernmost row (GEE convention).
    # Folium's ImageOverlay maps [[south, west], [north, east]] with the image
    # top at north — so the array is already in the right orientation.
    png_url = _array_to_png_b64(arr, mask)
    folium.raster_layers.ImageOverlay(
        image=png_url,
        bounds=[[south, west], [north, east]],
        opacity=0.75,
        name=LABEL_DISPLAY[label_key],
        zindex=1,
    ).add_to(m)

    # ── District boundaries + labels ─────────────────────────────────────────
    if districts is not None:
        geo_layer = folium.GeoJson(
            districts,
            name="Districts",
            style_function=_district_style,
            highlight_function=_district_highlight,
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME_1"],
                aliases=["District:"],
                style="font-size:13px; font-weight:bold;",
            ),
            zindex=2,
        )
        geo_layer.add_to(m)

        # Add fixed district name labels at each polygon centroid
        for feature in districts.get("features", []):
            name = (
                feature.get("properties", {}).get("NAME_1")
                or feature.get("properties", {}).get("name")
                or ""
            )
            if not name:
                continue
            coords = feature["geometry"]["coordinates"]
            geom_type = feature["geometry"]["type"]
            # Extract a representative point (first ring centroid)
            try:
                if geom_type == "Polygon":
                    ring = np.array(coords[0])
                elif geom_type == "MultiPolygon":
                    # Use the largest polygon
                    ring = max(coords, key=lambda p: len(p[0]))
                    ring = np.array(ring[0])
                else:
                    continue
                clon = float(ring[:, 0].mean())
                clat = float(ring[:, 1].mean())
                folium.Marker(
                    location=[clat, clon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:10px;font-weight:600;color:#222;'
                             f'text-shadow:1px 1px 2px #fff,-1px -1px 2px #fff;'
                             f'white-space:nowrap;">{name}</div>',
                        icon_size=(120, 20),
                        icon_anchor=(60, 10),
                    ),
                ).add_to(m)
            except Exception:
                continue

    # ── Colour-scale legend ───────────────────────────────────────────────────
    _font = "system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif"
    legend_html = f"""
    <div style="position:fixed;bottom:36px;left:28px;z-index:1000;
                background:rgba(248,249,246,0.96);
                backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
                padding:12px 16px;border-radius:6px;
                border:1px solid #D4D9D2;
                font-family:{_font};font-size:12px;line-height:1.75;
                color:#1C1C1E;box-shadow:0 2px 12px rgba(0,0,0,0.08);">
        <div style="font-size:10.5px;font-weight:700;letter-spacing:0.05em;
                    text-transform:uppercase;color:#5A6057;margin-bottom:6px;">
            {LABEL_DISPLAY[label_key]}&nbsp;&middot;&nbsp;{month_label}
        </div>
        <span style="color:#8B0000;font-size:13px">&#9632;</span>&nbsp;Severe drought
        <span style="color:#999;font-size:10px">&lt; {DROUGHT_SEVERE:.2f}</span><br>
        <span style="color:#CC4400;font-size:13px">&#9632;</span>&nbsp;Moderate drought
        <span style="color:#999;font-size:10px">&lt; {DROUGHT_ONSET:.2f}</span><br>
        <span style="color:#4CAF50;font-size:13px">&#9632;</span>&nbsp;Normal<br>
        <span style="color:#1565C0;font-size:13px">&#9632;</span>&nbsp;Wetter than normal
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Click-for-coordinates plugin ──────────────────────────────────────────
    m.add_child(folium.ClickForLatLng())

    folium.LayerControl().add_to(m)
    return m
