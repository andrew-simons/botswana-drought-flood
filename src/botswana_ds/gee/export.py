"""GEE helpers: clip to Botswana, aggregate to monthly, export to Drive.

These functions return Earth Engine objects / start export tasks; they only work
*after* you have run `ee.Authenticate()` and `ee.Initialize(project=...)` in your
session (do that in the Colab notebook, not here).

Every export writes a single multi-band GeoTIFF to Google Drive (one band = one
month) so the entire time series for a variable fits in one file — easy to
download and stack into (T, H, W).

Unit / scale-factor notes (IMPORTANT when assembling the cube in Python):
  ERA5 temperature / dewpoint  : stored in Kelvin — subtract 273.15 for °C
  MODIS LST (LST_Day_1km)      : raw DN; GEE does NOT auto-scale MODIS bands
                                  → multiply by 0.02 for Kelvin, −273.15 for °C
  MODIS NDVI / EVI             : raw DN; multiply by 0.0001 for real NDVI/EVI
  MODIS ET                     : raw DN; multiply by 0.1 for mm/day per 8-day composite
  CHIRPS precipitation         : already mm/month — no scaling needed
  SMAP ssm                     : already in mm — no scaling needed
  ERA5 wind components         : already in m/s — no scaling needed
"""

from __future__ import annotations

from ..grid import BOTSWANA_BBOX, GRID_RES_DEG, GEE_CRS_TRANSFORM, GEE_DIMENSIONS

try:
    import ee
except ImportError:
    ee = None


def _require_ee():
    if ee is None:
        raise ImportError(
            "earthengine-api is not installed / initialized. In Colab run:\n"
            "  import ee; ee.Authenticate(); ee.Initialize(project='your-project')"
        )


def _export_region():
    """Exact bbox rectangle for export tasks.

    Using the bbox (not the GAUL polygon) as the export region ensures
    crsTransform produces exactly 182×188 pixels — the GAUL polygon
    extends slightly outside the grid bbox and would add extra edge pixels.
    Image data is already clipped to GAUL before export, so this only
    affects the output extent, not the masked pixels.
    """
    min_lon, min_lat, max_lon, max_lat = BOTSWANA_BBOX
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def botswana_geometry(use_gaul: bool = True):
    """Return Botswana as an ee.Geometry.

    use_gaul=True  -> precise national boundary from FAO GAUL (recommended for masking).
    use_gaul=False -> simple bbox (fast, for quick previews).
    """
    _require_ee()
    if use_gaul:
        gaul = ee.FeatureCollection("FAO/GAUL/2015/level0")
        return gaul.filter(ee.Filter.eq("ADM0_NAME", "Botswana")).geometry()
    min_lon, min_lat, max_lon, max_lat = BOTSWANA_BBOX
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def month_starts(start: str, end: str):
    """ee.List of monthly start dates in [start, end), e.g. '2003-01-01','2024-01-01'."""
    _require_ee()
    start_date = ee.Date(start)
    end_date = ee.Date(end)
    n_months = end_date.difference(start_date, "month").round()
    return ee.List.sequence(0, n_months.subtract(1)).map(
        lambda i: start_date.advance(ee.Number(i), "month")
    )


def monthly_reduce(
    collection_id: str,
    band: str,
    reducer,
    start: str,
    end: str,
    region=None,
    scale_m: int = 5566,
):
    """Generic monthly aggregator — one image per month in [start, end).

    Works for both sub-monthly collections (e.g. CHIRPS/DAILY, MODIS 8-day) and
    already-monthly collections (ERA5_LAND/MONTHLY_AGGR, MOD13A3). For
    already-monthly collections the reduce is a no-op (single image reduces to
    itself), but the clipping and time-stamping still happen correctly.

    Returns an ee.ImageCollection with one image per month, clipped to region.
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
        monthly_col = col.filterDate(m0, m1)
        # If the collection is empty (data gap / sensor outage), return a fully-masked
        # image so toBands() still produces a band for this month — just all NaN.
        filled = ee.Image.constant(0).rename(band).updateMask(ee.Image.constant(0))
        img = ee.Algorithms.If(
            monthly_col.size().gt(0),
            monthly_col.reduce(reducer).rename(band),
            filled,
        )
        return ee.Image(img).set("system:time_start", m0.millis()).clip(region)

    return ee.ImageCollection(month_starts(start, end).map(per_month))


def monthly_chirps(start: str = "2003-01-01", end: str = "2024-01-01", region=None):
    """Monthly total rainfall (mm) from CHIRPS — our anchor variable and grid source.

    CHIRPS/MONTHLY was removed from GEE; we sum CHIRPS/DAILY within each calendar month.
    """
    return monthly_reduce(
        "UCSB-CHG/CHIRPS/DAILY", "precipitation", ee.Reducer.sum(), start, end, region
    )


def monthly_wind_speed(start: str = "2003-01-01", end: str = "2024-01-01", region=None):
    """ERA5-Land 10 m wind speed (m/s) = sqrt(u² + v²), monthly mean.

    Wind speed requires combining two bands, so it gets its own function instead
    of a DROUGHT_CORE entry.
    """
    _require_ee()
    region = region or botswana_geometry()

    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(start, end)
        .filterBounds(region)
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
    )

    def per_month(m0):
        m0 = ee.Date(m0)
        m1 = m0.advance(1, "month")
        monthly_col = era5.filterDate(m0, m1).select(
            ["u_component_of_wind_10m", "v_component_of_wind_10m"]
        )
        # Fully-masked fallback placed first so real data (added via merge) always wins.
        # mosaic() takes the last valid pixel — monthly_col images override the masked
        # fallback wherever data exists; months with no data stay fully masked.
        # This avoids ee.Algorithms.If which has type-inference issues inside map().
        filled = (
            ee.Image.constant([0.0, 0.0])
            .rename(["u_component_of_wind_10m", "v_component_of_wind_10m"])
            .updateMask(ee.Image.constant(0))
        )
        img = ee.ImageCollection([filled]).merge(monthly_col).mosaic()
        u = img.select("u_component_of_wind_10m")
        v = img.select("v_component_of_wind_10m")
        speed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed")
        return speed.set("system:time_start", m0.millis()).clip(region)

    return ee.ImageCollection(month_starts(start, end).map(per_month))


# ── Variable registry ─────────────────────────────────────────────────────────
#
# 8 of the 9 dynamic channels that can be exported via export_variable_to_drive().
# Channel 9 (wind_speed) is computed from u+v — use export_wind_speed_to_drive().
#
# Keys map to the cube channel names used throughout the project.

DROUGHT_CORE: dict[str, tuple[str, str, str]] = {
    # name          collection_id                                band                       reducer
    "rain_mm":    ("UCSB-CHG/CHIRPS/DAILY",                    "precipitation",            "sum"),
    "t2m_k":      ("ECMWF/ERA5_LAND/MONTHLY_AGGR",             "temperature_2m",           "mean"),
    "ndvi":       ("MODIS/061/MOD13A3",                        "NDVI",                     "mean"),
    "evi":        ("MODIS/061/MOD13A3",                        "EVI",                      "mean"),
    "et":         ("MODIS/006/MOD16A2",                        "ET",                       "sum"),
    "lst_day_k":  ("MODIS/061/MOD11A2",                        "LST_Day_1km",              "mean"),
    "sm_surf":    ("NASA/SMAP/SPL4SMGP/008",                    "sm_surface",               "mean"),
    "dewpoint_k": ("ECMWF/ERA5_LAND/MONTHLY_AGGR",             "dewpoint_temperature_2m",  "mean"),
}

_REDUCERS = {
    "sum":  lambda: ee.Reducer.sum(),
    "mean": lambda: ee.Reducer.mean(),
    "min":  lambda: ee.Reducer.min(),
    "max":  lambda: ee.Reducer.max(),
}


def export_variable_to_drive(
    name: str,
    start: str = "2003-01-01",
    end: str = "2024-01-01",
    folder: str = "BotswanaDroughtFloodSet",
    region=None,
):
    """Start a Drive export of monthly images for one DROUGHT_CORE variable.

    Exports a single multi-band GeoTIFF (one band per month) to Google Drive.
    Monitor progress in the GEE Tasks tab at https://code.earthengine.google.com.
    """
    _require_ee()
    if name not in DROUGHT_CORE:
        raise KeyError(f"{name!r} not in DROUGHT_CORE: {list(DROUGHT_CORE)}")
    cid, band, red = DROUGHT_CORE[name]
    reducer = _REDUCERS[red]()
    region = region or botswana_geometry()

    monthly = monthly_reduce(cid, band, reducer, start, end, region)
    stacked = monthly.toBands().toFloat()

    task = ee.batch.Export.image.toDrive(
        image=stacked,
        description=f"{name}_{start[:4]}_{end[:4]}",
        folder=folder,
        fileNamePrefix=f"{name}_monthly_{start[:4]}_{end[:4]}",
        region=_export_region(),
        crsTransform=GEE_CRS_TRANSFORM,
        dimensions=GEE_DIMENSIONS,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    return task


def export_wind_speed_to_drive(
    start: str = "2003-01-01",
    end: str = "2024-01-01",
    folder: str = "BotswanaDroughtFloodSet",
    region=None,
):
    """Start a Drive export of monthly ERA5-Land 10 m wind speed (m/s)."""
    _require_ee()
    region = region or botswana_geometry()
    monthly = monthly_wind_speed(start, end, region)
    stacked = monthly.toBands().toFloat()

    task = ee.batch.Export.image.toDrive(
        image=stacked,
        description=f"wind_speed_{start[:4]}_{end[:4]}",
        folder=folder,
        fileNamePrefix=f"wind_speed_monthly_{start[:4]}_{end[:4]}",
        region=_export_region(),
        crsTransform=GEE_CRS_TRANSFORM,
        dimensions=GEE_DIMENSIONS,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    return task


def export_static_layers(
    folder: str = "BotswanaDroughtFloodSet",
):
    """Export the 3 static channels (elevation, slope, land cover) as one multi-band GeoTIFF.

    Returns a list of started ee.batch.Task objects (one per export).
    We export them separately so a failure in one doesn't block the others.

    Band descriptions
    -----------------
    elevation_m   : Copernicus GLO-30 DEM in metres (resampled to 0.05°)
    slope_deg     : slope in degrees, derived from GLO-30 via ee.Terrain.slope()
    landcover_int : ESA WorldCover 2021 integer class (10=Trees … 100=Moss)
    """
    _require_ee()

    region = _export_region()
    export_kwargs = dict(
        folder=folder,
        region=region,
        crsTransform=GEE_CRS_TRANSFORM,
        dimensions=GEE_DIMENSIONS,
        crs="EPSG:4326",
        maxPixels=1e10,
    )

    # --- Elevation + slope (Copernicus GLO-30, 30 m) ---
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic()
    elevation = dem.rename("elevation_m")
    slope = ee.Terrain.slope(dem).rename("slope_deg")
    elev_slope = elevation.addBands(slope).toFloat()

    task_elev = ee.batch.Export.image.toDrive(
        image=elev_slope,
        description="static_elev_slope",
        fileNamePrefix="static_elev_slope",
        **export_kwargs,
    )
    task_elev.start()

    # --- Land cover (ESA WorldCover 2021, 10 m) ---
    lc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover_int")

    task_lc = ee.batch.Export.image.toDrive(
        image=lc.toUint8(),
        description="static_landcover",
        fileNamePrefix="static_landcover",
        **export_kwargs,
    )
    task_lc.start()

    return [task_elev, task_lc]


def export_land_mask(
    folder: str = "BotswanaDroughtFloodSet",
    region=None,
):
    """Export the Botswana land mask (1 = land, 0 = outside) on the common grid.

    Exports a single-band uint8 GeoTIFF. Read it in Python with rioxarray and
    save as mask.npy (bool). The mask is used by the loss function to exclude
    pixels outside Botswana's border.
    """
    _require_ee()
    region = region or botswana_geometry()

    # Burn a constant 1 over the GAUL polygon. Any pixel whose centre falls
    # inside Botswana gets value 1; everything outside the exported region is
    # masked (nodata) by the GEE exporter, which we treat as 0 when loading.
    mask = ee.Image.constant(1).clip(region).rename("land_mask").toUint8()

    task = ee.batch.Export.image.toDrive(
        image=mask,
        description="land_mask",
        folder=folder,
        fileNamePrefix="land_mask",
        region=_export_region(),
        crsTransform=GEE_CRS_TRANSFORM,
        dimensions=GEE_DIMENSIONS,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    return task
