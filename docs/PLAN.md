# Project Plan & Design — Botswana Drought & Flood Early-Warning

Inspired by **DroughtSet** (Tan et al., AAAI-25, arXiv:2412.15075) but redesigned for
Botswana with free global data. Compute: **Google Colab (free)**. Priority: **drought
first**, flood phase 2. Deliverable: a research-style dataset + baseline + benchmark,
with a rough Streamlit demo in the final week.

## What we borrow from DroughtSet
DroughtSet is a CONUS, 4 km, weekly, 2003–2013 spatio-temporal cube. It splits features
into **static** (elevation, canopy, land cover) vs **dynamic** (climate, soil moisture,
vegetation), and uses **three continuous, per-pixel-normalized drought indices** as labels
(soil-moisture, ESI, SIF) — not categorical labels. Task = forecast 26 weeks from 100;
spatial 5×5-block train/test split; metrics MAE + detection accuracy at the 30th pct.

**We keep:** static/dynamic split, continuous climatology-normalized labels, the
forecasting framing, block splits, MAE/detection metrics.
**We change:** region (Botswana), grid (0.05°≈5.5 km, CHIRPS-aligned, ~130×200), cadence
(monthly), span (2003–present), and label sources → **SPI (from CHIRPS), NDVI-anomaly,
soil-moisture-anomaly** (the same "three drought types" idea with free data).
**We replace:** US-only products (GridMET/NLDAS/NLCD/USDM) with CHIRPS/ERA5-Land/SMAP/
WorldCover; USDM → EM-DAT event records used only for validation.

## Dataset: `BotswanaDroughtFloodSet`
A data cube in Drive: `dynamic (T,C,H,W)`, `static (Cs,H,W)`, `labels (T,3,H,W)`,
`mask (H,W)`, plus `meta.json`. Splits: temporal holdout (last ~3 yrs) **and** spatial
5×5-block CV. See [DATASETS.md](DATASETS.md) for all sources.

## Pipeline (GEE → Drive → PyTorch)
1. Define grid/AOI (`botswana_ds.grid`, `gee.export.botswana_geometry`).
2. Monthly-reduce each variable in GEE (`gee.export.monthly_reduce` / `export_variable_to_drive`).
3. Compute labels: SPI (gamma fit on CHIRPS), NDVI & SM z-anomalies per pixel & month.
4. Stack to the cube; record NaN mask; normalize dynamics on **train split only**.
5. Window into (input_len → horizon) samples (`data.dataset.BotswanaCube`).

## Models (in order; beat the baselines first)
1. Persistence + climatology (`models.baselines`) — the bar to beat.
2. Per-pixel RandomForest/XGBoost.
3. Per-pixel LSTM/GRU.
4. **ConvLSTM** (`models.convlstm`) — the spatio-temporal core.
5. Stretch: spatio-temporal transformers / ViT; GNN on HydroSHEDS basins (flood).
Drought and flood use **separate** models (regression vs SAR segmentation/U-Net).

## Evaluation (`train.metrics`)
MAE (primary), RMSE, R²; detection Accuracy/Precision/Recall/F1 at 30th pct; event-level
check vs EM-DAT drought years (2015–16, 2023–24). Validate with temporal holdout +
spatial block CV. No leakage: normalization stats from train only.

## Deployment
`torch.save(state_dict)` + config JSON → FastAPI (`/risk`, `/timeseries`, `/alerts`) →
Streamlit/React map with heatmaps; Integrated Gradients/SHAP for explanations.

## 8-week timeline
1 setup + grid + CHIRPS loop · 2 full export pipeline · 3 labels + cube · 4 dataset v0.1 +
loader · 5 baselines + metrics · 6 LSTM → ConvLSTM · 7 benchmark + explanations · 8 write-up
+ Streamlit demo. (Later: flood phase — Sentinel-1 + U-Net + GNN.)

## First 5 tasks (this week)
1. Repo skeleton + `requirements.txt` ✅ (this scaffold).
2. EE account + project; run `notebooks/01_gee_hello.ipynb`.
3. Read DroughtSet; note keep/change (this doc is your starting point).
4. Pull + visualize + export one month of CHIRPS (in the notebook). ✅ code ready.
5. Define + save the common grid + land mask (`botswana_ds.grid`). ✅ code ready.
