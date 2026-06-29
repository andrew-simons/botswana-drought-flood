# Botswana Drought & Flood Early-Warning System

A spatio-temporal machine-learning project to monitor and forecast **drought** (phase 1)
and **flood** (phase 2) risk across Botswana, using only **free, public** global datasets
via **Google Earth Engine (GEE)** and **PyTorch**. Inspired by the
[DroughtSet paper](https://arxiv.org/abs/2412.15075) (AAAI-25), but redesigned for
Botswana with global data.

**Primary deliverable:** a reusable, research-style spatio-temporal dataset —
`BotswanaDroughtFloodSet` — plus a baseline + benchmark, and a rough demo dashboard.

See [`docs/PLAN.md`](docs/PLAN.md) for the full 8-week plan and design rationale.

---

## Architecture in one line

**GEE** (extract + preprocess satellite/climate data) → **Google Drive** (analysis-ready
arrays) → **PyTorch in Colab** (train models) → **Streamlit/FastAPI** (serve predictions).

You do *not* train models in GEE — it is only the data engine.

## Repo layout

```
botswana-drought-flood/
├── notebooks/        # Colab notebooks (drive the work; commit with outputs cleared)
├── src/botswana_ds/  # the importable library (the durable, reusable code)
│   ├── grid.py       # the common Botswana grid everything snaps to
│   ├── gee/          # GEE export + preprocessing helpers
│   ├── data/         # cube builder, PyTorch Dataset, normalization
│   ├── models/       # baselines, LSTM, ConvLSTM, (later) U-Net
│   ├── train/        # training loop, metrics, evaluation
│   └── explain/      # SHAP / Integrated Gradients
├── data/             # pointers only — real arrays live in Google Drive
├── models/           # saved weights (gitignored) + config JSON
├── api/              # FastAPI (deployment phase)
├── app/              # Streamlit demo (week 8)
└── docs/             # PLAN.md, dataset card, methodology
```

## Quickstart (Colab)

1. Push this repo to GitHub.
2. Open a new [Google Colab](https://colab.research.google.com/) notebook.
3. Run:
   ```python
   !pip install -q geemap rioxarray xgboost captum shap
   !git clone https://github.com/<you>/botswana-drought-flood.git
   %cd botswana-drought-flood

   import ee, geemap
   ee.Authenticate()                  # one-time per session; follow the link
   ee.Initialize(project='<your-ee-project>')

   from src.botswana_ds.grid import BOTSWANA_BBOX, make_grid
   from src.botswana_ds.gee.export import botswana_geometry, monthly_chirps

   m = geemap.Map(center=[-22, 24], zoom=5)
   m.addLayer(botswana_geometry(), {}, 'Botswana')
   m
   ```
4. Follow Week-1 tasks in [`docs/PLAN.md`](docs/PLAN.md).

## Datasets (all free)

Drought core: CHIRPS rainfall, ERA5-Land, MODIS LST/NDVI/ET, SMAP soil moisture,
Copernicus DEM, SPEIbase, GRACE, ESA WorldCover. Event labels: EM-DAT, Dartmouth Flood
Observatory. Full table with GEE collection IDs in [`docs/DATASETS.md`](docs/DATASETS.md).

## Status
- [ ] Week 1 — setup, grid, CHIRPS proof-of-loop
- [ ] Week 2 — full export pipeline
- [ ] Week 3 — labels + cube
- [ ] Week 4 — BotswanaDroughtFloodSet v0.1
- [ ] Week 5 — baselines + metrics
- [ ] Week 6 — ConvLSTM
- [ ] Week 7 — benchmark + explanations
- [ ] Week 8 — write-up + Streamlit demo
