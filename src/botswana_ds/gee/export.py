"""GEE helpers: clip to Botswana, aggregate to monthly, export to Drive.

These functions return Earth Engine objects / start export tasks; they only work
*after* you have run `ee.Authenticate()` and `ee.Initialize(project=...)` in your
session (do that in the Colab notebook, not here).

Mental model for every variable:
    collection -> filter to date range -> filter to Botswana -> reduce to a
    MONTHLY image (sum for rainfall, mean for temperature/NDVI/...) -> resample to
    the common grid -> export.

Keeping all variables on the SAME monthly cadence and SAME grid is what lets us
later stack them into a (T, C, H, W) cube.
"""

from __future__ import annotations

from ..grid import BOTSWANA_BBOX, GRID_RES_DEG

# Lazy import so the module can be imported (e.g. for docs) without ee installed.
try:
    import ee
except ImportError:  # pragma: no cover
    ee = None


def _require_ee():
    if ee is None:
        raise ImportError(
            "earthengine-api is not installed / initialized. In Colab run:\n"
            "  import ee; ee.Authenticate(); ee.Initialize(project='your-project')"
        )


def botswana_geometry(use_gaul: bool = True):
    """Return Botswana as an ee.Geometry.

    use_gaul=True  -> precise national boundary from FAO GAUL (recommended for masking).
    use_gaul=False -> the simple rectangular bbox (fast, for quick previews).
    """
    _require_ee()
    if use_gaul:
        gaul = ee.FeatureCollection("FAO/GAUL/2015/level0")
        return gaul.filter(ee.Filter.eq("ADM0_NAME", "Botswana")).geometry()
    min_lon, min_lat, max_lon, max_lat = BOTSWANA_BBOX
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def month_starts(start: str, end: str):
    """ee.List of monthly start dates in [start, end), e.g. month_starts('2003-01-01','2024-01-01')."""
    _require_ee()
    start = ee.Date(start)
    end = ee.Date(end)
    n_months = end.difference(start, "month").round()
    return ee.List.sequence(0, n_months.subtract(1)).map(
        lambda i: start.advance(ee.Number(i), "month")
    )


def monthly_reduce(collection_id, band, reducer, start, end, region=None, scale_m=5566):
    """Generic monthly aggregator.

    Args:
        collection_id: GEE ImageCollection id, e.g. 'UCSB-CHG/CHIRPS/DAILY'.
        band: band name to keep, e.g. 'precipitation'.
        reducer: ee.Reducer (ee.Reducer.sum() for rain, .mean() for temp/NDVI).
        start, end: 'YYYY-MM-DD' strings; range is [start, end).
        region: ee.Geometry to clip to (defaults to Botswana GAUL).
        scale_m: nominal export scale in meters (~5566 m = 0.05 deg at the equator).

    Returns:
        ee.ImageCollection with one image per month, each carrying a 'system:time_start'.
    """
    _require_ee()
    region = region or botswana_geometry()
    col = (
        ee.ImageCollection(collection_id)
        .filterDate(start, end)
        .filterBounds(region)
        .select(band)
    )

    def per_month(m0):
        m0 = ee.Date(m0)
        m1 = m0.advance(1, "month")
        img = col.filterDate(m0, m1).reduce(reducer).rename(band)
        return img.set("system:time_start", m0.millis()).clip(region)

    return ee.ImageCollection(month_starts(start, end).map(per_month))


def monthly_chirps(start="2003-01-01", end="2024-01-01", region=None):
    """Monthly TOTAL rainfall (mm) from CHIRPS daily — our anchor variable."""
    _require_ee()
    return monthly_reduce(
        "UCSB-CHG/CHIRPS/DAILY", "precipitation", ee.Reducer.sum(), start, end, region
    )


# Registry of the drought-core variables and how to reduce each to monthly.
# (band, reducer-name) — extend this as you add variables in Week 2.
DROUGHT_CORE = {
    "rain_mm":   ("UCSB-CHG/CHIRPS/DAILY",        "precipitation",       "sum"),
    "t2m":       ("ECMWF/ERA5_LAND/DAILY_AGGR",   "temperature_2m",      "mean"),
    "lst_day":   ("MODIS/061/MOD11A2",            "LST_Day_1km",         "mean"),
    "ndvi":      ("MODIS/061/MOD13Q1",            "NDVI",                "mean"),
    "evi":       ("MODIS/061/MOD13Q1",            "EVI",                 "mean"),
    "et":        ("MODIS/061/MOD16A2",            "ET",                  "mean"),
    "pet":       ("MODIS/061/MOD16A2",            "PET",                 "mean"),
    "sm_rz":     ("NASA/SMAP/SPL4SMGP/007",       "sm_rootzone",         "mean"),
}

_REDUCERS = {"sum": "Reducer.sum", "mean": "Reducer.mean", "min": "Reducer.min", "max": "Reducer.max"}


def export_variable_to_drive(name, start, end, folder="BotswanaDroughtFloodSet", region=None):
    """Start a Drive export of monthly images for one DROUGHT_CORE variable.

    The export writes one multi-band GeoTIFF (band per month) per variable to your
    Google Drive `folder`. Monitor progress in the GEE 'Tasks' tab.
    """
    _require_ee()
    if name not in DROUGHT_CORE:
        raise KeyError(f"{name!r} not in DROUGHT_CORE: {list(DROUGHT_CORE)}")
    cid, band, red = DROUGHT_CORE[name]
    reducer = {"sum": ee.Reducer.sum(), "mean": ee.Reducer.mean(),
               "min": ee.Reducer.min(), "max": ee.Reducer.max()}[red]
    region = region or botswana_geometry()

    monthly = monthly_reduce(cid, band, reducer, start, end, region)
    # Stack months into bands of a single image so it exports as one cube-friendly file.
    stacked = monthly.toBands()

    task = ee.batch.Export.image.toDrive(
        image=stacked,
        description=f"{name}_{start[:4]}_{end[:4]}",
        folder=folder,
        fileNamePrefix=f"{name}_monthly_{start[:4]}_{end[:4]}",
        region=region,
        scale=5566,                       # ~0.05 deg
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    return task
