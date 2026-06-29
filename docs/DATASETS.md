# Datasets — Botswana Drought & Flood (all free, GEE-first)

Verified against the Earth Engine Data Catalog. Use the `061` MODIS IDs (not `006`).

## Drought core (phase 1)
| Variable | Dataset | GEE Collection ID | Res / cadence | Coverage |
|---|---|---|---|---|
| Rainfall | CHIRPS Daily v2 | `UCSB-CHG/CHIRPS/DAILY` | 5.5 km / daily | 1981– |
| Air temp + soil water | ERA5-Land | `ECMWF/ERA5_LAND/DAILY_AGGR` | 11 km / daily | 1950– |
| Land surface temp | MODIS MOD11A2 | `MODIS/061/MOD11A2` | 1 km / 8-day | 2000– |
| NDVI / EVI | MODIS MOD13Q1 | `MODIS/061/MOD13Q1` | 250 m / 16-day | 2000– |
| ET / PET | MODIS MOD16A2 | `MODIS/061/MOD16A2` | 500 m / 8-day | 2001– |
| Soil moisture (root-zone) | SMAP L4 | `NASA/SMAP/SPL4SMGP/007` | 11 km / 3-hr | 2015– |
| Terrain | Copernicus DEM GLO-30 | `COPERNICUS/DEM/GLO30` | 30 m / static | 2010–15 |
| SPEI (context) | Global SPEIbase v2.10 | `CSIC/SPEI/2_10` | 0.5° / monthly | 1901–2023 |
| Water storage | GRACE/GRACE-FO | `NASA/GRACE/MASS_GRIDS_V04/LAND` | ~0.5° / monthly | 2002– |
| Land cover | ESA WorldCover v200 | `ESA/WorldCover/v200` | 10 m / static | 2021 |

## Flood add-ons (phase 2)
| Purpose | Dataset | GEE Collection ID |
|---|---|---|
| Flood detection (all-weather) | Sentinel-1 SAR GRD | `COPERNICUS/S1_GRD` |
| Baseline water mask | JRC Global Surface Water | `JRC/GSW1_4/GlobalSurfaceWater`, `JRC/GSW1_4/MonthlyHistory` |
| Historical flood labels | Global Flood DB (MODIS) | `GLOBAL_FLOOD_DB/MODIS_EVENTS/V1` |
| Flow routing (flat terrain) | MERIT Hydro / HydroSHEDS | `MERIT/Hydro/v1_0_1`, `WWF/HydroSHEDS/15ACC`, `WWF/HydroSHEDS/15DIR` |
| Static flood hazard | GloFAS Flood Hazard | `JRC/CEMS_GLOFAS/FloodHazard/v2_1` |

## Exposure (both phases)
| Variable | Dataset | GEE Collection ID |
|---|---|---|
| Population | WorldPop | `WorldPop/GP/100m/pop` |
| Built-up area | GHSL | `JRC/GHSL/P2023A/GHS_BUILT_S` |

## Historical event records (for VALIDATION, not pixel labels)
- **Drought:** EM-DAT — https://www.emdat.be/ (filter country = Botswana). Corroborate
  with World Bank "Drought Resilience Profile: Botswana".
- **Flood:** Dartmouth Flood Observatory — https://floodobservatory.colorado.edu/ (also on
  HDX); plus the GEE Global Flood DB layer. Cross-check ReliefWeb Botswana sitreps.
- Documented Botswana events: floods 2000, 2013, ex-Dineo 2017, 2018; drought emergency 2023–24.

## Notes
- **Do NOT use `GRIDMET/DROUGHT`** — it is CONUS-only. Compute SPI yourself from CHIRPS.
- UN-SPIDER has turnkey GEE recipes for [SPI drought monitoring](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-drought-monitoring-spi/step-by-step)
  and [Sentinel-1 flood mapping](https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step).
- For the flat Okavango/Makgadikgadi, prefer Copernicus GLO-30 or MERIT Hydro over raw SRTM.
