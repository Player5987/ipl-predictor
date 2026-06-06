# 🏏 IPL Match Predictor (2025–2049)

ML-powered IPL predictions using ball-by-ball phase features,
Prophet player forecasting, and ELO team ratings.

## Results

| Component       | Metric  | Value  | Notes                        |
|-----------------|---------|--------|------------------------------|
| Phase features  | Corr    | 0.12   | pp_dot_diff, death_econ_diff |
| Match predictor | CV AUC  | 0.558  | 8-fold time-series CV        |
| ELO ratings     | Teams   | 10     | Validated vs real standings  |
| Player forecast | Players | 87     | Prophet 80% CI               |
| API endpoints   | Count   | 7      | FastAPI + Swagger UI         |
| Frontend tabs   | Count   | 4      | React + Recharts             |

> IPL has genuine randomness. Professional betting markets
> with squad news and pitch data achieve 0.62–0.67 AUC.
> This model uses only historical match statistics.

## Quick Start

```bash
# Terminal 1 — Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend  
cd frontend && npm run dev
```

Open http://localhost:5173

## Tech Stack

| Layer      | Technology                              |
|------------|-----------------------------------------|
| ML Models  | XGBoost, LightGBM, RandomForest, Prophet|
| Features   | Ball-by-ball phase stats, ELO, H2H, Form|
| Backend    | FastAPI, Uvicorn, Joblib               |
| Frontend   | React, Vite, Recharts, Axios           |
| Data       | Cricsheet BBB (2008–2025), 1076 matches|

## Features

- **Match Predictor** — any two IPL teams, any year 2025–2049
- **Player Forecast** — Prophet time-series for 87 players
- **Tournament Simulator** — full season standings + matrix
- **ELO Rankings** — team strength from 18 seasons of data

## Model Notes

Phase-based ball-by-ball features (powerplay dot ball %, death
bowling economy) are the strongest predictors — correlating
0.10–0.12 with match outcomes vs 0.05–0.08 for classic features.

The 2020–2024 IPL era shows lower feature correlations (0.03)
vs 2008–2015 (0.12) — confirming IPL has become genuinely more
unpredictable in the modern era.