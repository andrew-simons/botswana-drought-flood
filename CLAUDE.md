# CLAUDE.md — Botswana Drought & Flood Early-Warning System

## Project in one sentence
Build a spatio-temporal ML pipeline + web/mobile app for Botswana that lets citizens, governments, and NGOs monitor drought and flood risk — inspired by DroughtSet (AAAI-25, arXiv:2412.15075) but redesigned for Botswana with free global data only.

## User context
Undergraduate student/intern, new to geospatial ML and remote sensing. Has Python experience, learning PyTorch and GEE. Wants mentor-style explanations. Frame all responses accordingly.

## Stack
- Data extraction: Google Earth Engine (Python `ee` + `geemap`)
- Storage: Google Drive (GEE exports land here)
- Training compute: Google Colab (free GPU)
- ML framework: PyTorch
- Library: `src/botswana_ds/` (importable; notebooks call this)
- Backend (week 8+): FastAPI
- Frontend demo (week 8): Streamlit → React + Leaflet/Mapbox

---

## Key design decisions and WHY

### Monthly cadence (not weekly like DroughtSet)
DroughtSet uses weekly because USDM publishes weekly. Our free global sources (CHIRPS, MODIS, SMAP, ERA5-Land) are most reliable and complete at monthly aggregation. Flood phase 2 may need daily, but drought at monthly is appropriate.

### 0.05° (~5.5 km) grid, ~130 × 200 cells
CHIRPS native resolution is 0.05°. Aligning to CHIRPS avoids resampling our primary rainfall input. All other variables are resampled TO this grid, not the other way around. Botswana bbox: `[17.0, -27.0, 29.5, -18.0]` (lon_min, lat_min, lon_max, lat_max).

### Three continuous labels (not categorical)
Following DroughtSet's key insight: per-pixel, climatology-normalized continuous labels capture severity gradients better than categorical severity classes. Our three:
1. **SPI-3** — 3-month Standardized Precipitation Index, computed from CHIRPS using scipy gamma fit
2. **NDVI anomaly** — z-score vs that pixel's own monthly climatology (MODIS MOD13A3)
3. **SM anomaly** — z-score vs that pixel's own monthly climatology (SMAP 10KM)

### Normalization computed on training split ONLY
If you fit normalization stats on the full dataset, future data leaks into training. Per-pixel normalization preferred (each pixel has its own seasonal climatology). Stats saved to `meta.json`.

### Spatial block splits (5×5-degree blocks)
Neighboring pixels are spatially correlated — drought spreads. A random pixel-level split leaks spatial patterns across the boundary. Block splits prevent this. Same approach as DroughtSet.

### Separate drought and flood models
Drought: slow-onset, monthly, regression (continuous labels). Flood: fast-onset, daily/event, segmentation (binary mask). Different data cadence, different labels, different architectures. Phase 1 = drought only.

### GEE for data, not training
GEE is a planetary-scale geospatial compute platform excellent at spatial reduction and export. Not designed for iterative ML training. Export monthly arrays to Drive, then train in Colab with PyTorch.

---

## Cube structure
```
dynamic:  (T, C_dyn, H, W)   e.g. (264, 9, 130, 200) for 22 years × 12 months
static:   (C_static, H, W)   e.g. (3, 130, 200)
labels:   (T, 3, H, W)       SPI, NDVI-anom, SM-anom
mask:     (H, W)              True = valid land pixel
meta.json                     grid params, variable names, norm stats, split dates
```

### Dynamic channels (9)
1. CHIRPS precipitation (mm/month)
2. ERA5-Land 2m temperature (°C)
3. MODIS NDVI
4. MODIS EVI
5. MODIS ET (mm/day)
6. MODIS LST daytime (°C)
7. SMAP surface soil moisture (m³/m³)
8. ERA5-Land dewpoint temperature (°C)
9. ERA5-Land wind speed (m/s)

### Static channels (3)
1. Elevation (m) — Copernicus GLO-30
2. Slope (degrees) — derived from DEM
3. Land cover class (integer) — ESA WorldCover 2021

### Label channels (3)
1. SPI-3 (gamma-fit on CHIRPS, training period only)
2. NDVI anomaly (z-score vs pixel monthly climatology, training period only)
3. SM anomaly (z-score vs pixel monthly climatology, training period only)

---

## Datasets (GEE collection IDs)

| Variable | GEE Collection | Resolution | Notes |
|----------|----------------|------------|-------|
| Rainfall | `UCSB-CHG/CHIRPS/MONTHLY` | 5 km | Label source for SPI |
| Temperature | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | 9 km | 2m temp, dewpoint, wind |
| Soil moisture | `NASA_USDA/HSL/SMAP10KM_soil_moisture` | 10 km | Label source for SM anom |
| NDVI/EVI | `MODIS/006/MOD13A3` | 1 km | Label source for NDVI anom |
| ET | `MODIS/006/MOD16A2` | 500 m | 8-day → aggregate to monthly |
| LST | `MODIS/006/MOD11A2` | 1 km | 8-day → aggregate to monthly |
| Elevation | `COPERNICUS/DEM/GLO30` | 30 m | Static; derive slope |
| Land cover | `ESA/WorldCover/v200` | 10 m | Static; encode as integer |
| River networks | `WWF/HydroSHEDS/15ACC` | 500 m | Flood phase only |
| Flood extent | `JRC/GSW1_4/MonthlyHistory` | 30 m | Flood labels for phase 2 |
| Population | `WorldPop/GP/100m/pop` | 100 m | For exposure-weighted risk |

Validation (not in GEE): EM-DAT event records (emdat.be) — Botswana drought years 2015–16, 2023–24.

---

## Model progression (in order — beat each baseline before advancing)

1. **Persistence** — predict last month's label value; the floor
2. **Climatology** — predict historical mean for that calendar month; strong seasonal baseline
3. **Per-pixel XGBoost** — no spatial reasoning; fast to iterate; strong non-DL baseline
4. **Per-pixel LSTM** — adds temporal context per pixel; still no spatial reasoning
5. **ConvLSTM** — the target architecture: spatial + temporal jointly (use `ndrplz/ConvLSTM_pytorch`)
6. Stretch: ST-Transformer, GNN on HydroSHEDS (flood phase)

ConvLSTM key idea: standard LSTM hidden state is a vector; ConvLSTM hidden state is a 2D map (H×W). Gates use 2D convolutions — each pixel shares information with neighbors through the kernel.

---

## Evaluation

- **Primary:** MAE on continuous labels
- **Secondary:** RMSE, R²
- **Detection:** Precision/Recall/F1 at 30th-percentile threshold (SPI ≈ −0.52 = drought onset)
- **Event-level:** Do predictions align with EM-DAT drought years 2015–16, 2023–24?
- **Splits:** Temporal holdout (2022–2024 = test) + spatial block CV on remainder

---

## 8-week timeline
1. Setup + grid + CHIRPS proof-of-loop
2. Full GEE export pipeline (all 9 variables, 2003–2024)
3. Labels (SPI, NDVI-anom, SM-anom) + cube assembly
4. BotswanaDroughtFloodSet v0.1 + PyTorch Dataset + DataLoader
5. Baselines (persistence, climatology, XGBoost) + metrics module
6. Per-pixel LSTM → ConvLSTM training
7. Benchmark table + SHAP/Integrated Gradients explanations
8. Write-up + Streamlit demo + FastAPI skeleton

## Current status
- [x] Repo skeleton, requirements.txt, README, PLAN.md, DATASETS.md
- [ ] GEE account + hello world
- [ ] Common grid + land mask (`botswana_ds.grid`)
- [ ] CHIRPS proof-of-loop
- [ ] Full export pipeline
- [ ] Label computation
- [ ] Cube assembly
- [ ] PyTorch Dataset
- [ ] Baselines
- [ ] ConvLSTM
- [ ] Evaluation + benchmark
- [ ] Deployment

## Deployment target
```
FastAPI backend
  GET /risk?lat=&lon=&date=       → {spi, ndvi_anom, sm_anom}
  GET /timeseries?lat=&lon=       → historical + forecast series
  GET /forecast?lat=&lon=         → next 3-month predictions
  GET /explain?lat=&lon=&date=    → SHAP feature importances
  GET /alerts                     → cells exceeding thresholds

Frontend
  Streamlit (week 8 demo)
  → React + Leaflet/Mapbox (production)
     - Interactive Botswana map with risk heatmap layer
     - Historical trend charts (Plotly/Recharts)
     - Alert badges, EM-DAT event overlay
     - SHAP explanation panel
```

Hosting: Google Cloud Run, Render, or Railway (all have free tiers).
