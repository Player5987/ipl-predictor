<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/en/8/84/Indian_Premier_League_Official_Logo.svg" height="80" alt="IPL Logo"/>

# IPL Match Predictor

**ML-powered match predictions for IPL 2025–2049**

Ball-by-ball phase features · Prophet player forecasting · ELO ratings · Stacking ensemble

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB?style=flat-square&logo=react&logoColor=white)](https://react.dev)
[![XGBoost](https://img.shields.io/badge/XGBoost-Stacking_Ensemble-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

[Live Demo](#-live-demo) · [API Docs (Swagger)](#-api-endpoints) · [Quick Start](#-quick-start) · [Model Details](#-model-performance)

</div>

---

## Live Demo

| Service | URL |
|---------|-----|
| Frontend | [ipl-predictor-silk-tau.vercel.app](https://ipl-predictor-silk-tau.vercel.app) |
| API | [ipl-predictor-as4i.onrender.com](https://ipl-predictor-as4i.onrender.com) |
| API Docs | [/docs](https://ipl-predictor-as4i.onrender.com/docs) (Swagger UI) |

> **⚠️ Cold Start Notice — please read before opening the app**
>
> The backend is hosted on Render's free tier, which spins down after inactivity.
>
> 1. Open the **[API URL](https://ipl-predictor-as4i.onrender.com)** first and wait **30–60 seconds** for it to wake up — you should see a JSON health-check response.
> 2. Then open the **[Frontend](https://ipl-predictor-silk-tau.vercel.app)** — it will load and function normally once the API is warm.

---

## Overview

IPL Match Predictor is a full-stack machine learning application that predicts IPL match outcomes using historical ball-by-ball data from 1,243 matches (2008–2026). It combines a stacking ensemble classifier, ELO ratings, and Prophet time-series forecasts to generate win probabilities for any matchup from 2025 to 2049.

> **Honest benchmark:** Professional betting markets — with squad news, pitch reports, and injury data — reach 0.62–0.67 AUC. This model reaches **0.558 AUC** using only historical match statistics. The gap reflects genuine randomness in modern IPL, not a modelling failure.

---

## Model Performance

| Component | Metric | Value | Notes |
|-----------|--------|:-----:|-------|
| Match predictor | CV ROC-AUC | **0.558** | 8-fold time-series cross-validation |
| Phase features | Correlation | 0.12 | `pp_dot_diff`, `death_econ_diff` |
| ELO ratings | Teams covered | 10 | Validated against real IPL standings |
| Player forecasts | Players | 87 | Prophet with 80% confidence intervals |
| API endpoints | Count | 9 | FastAPI + Swagger UI |
| Frontend tabs | Count | 4 | React + Recharts |

### Era Analysis — Why Modern IPL Is Harder to Predict

Feature correlations have weakened significantly across IPL eras, confirming the league has become genuinely higher-parity over time:

| Era | Matches | Avg Feature Correlation |
|-----|--------:|:-----------------------:|
| 2008–2015 — Classic IPL | 508 | **0.117** |
| 2016–2019 — Mature IPL | 235 | 0.091 |
| 2020–2026 — Modern IPL | 403 | 0.031 |

A model trained on older data generalises less well to modern seasons — not because of poor modelling, but because IPL itself became a higher-parity league post-COVID.

---

## Top Predictive Features

Phase-level ball-by-ball features dominate the model, confirming that powerplay and death-over execution are the strongest drivers of match outcomes.

| # | Feature | Type | Importance | Correlation |
|---|---------|:----:|:----------:|:-----------:|
| 1 | `pp_dot_diff` | Phase | 0.146 | 0.118 |
| 2 | `death_econ_diff` | Phase | 0.100 | 0.104 |
| 3 | `pp_wicket_rate_diff` | Phase | — | 0.117 |
| 4 | `alltime_diff` | Base | 0.100 | 0.079 |
| 5 | `death_dot_diff` | Phase | 0.099 | 0.080 |
| 6 | `pp_boundary_diff` | Phase | 0.097 | 0.065 |
| 7 | `pp_run_rate_diff` | Phase | 0.092 | 0.072 |
| 8 | `elo_diff` | Base | 0.092 | 0.056 |
| 9 | `form10_diff` | Base | 0.087 | 0.044 |
| 10 | `pp_econ_diff` | Phase | 0.083 | 0.086 |

---

## Features

### Match Predictor
Select any two IPL teams and any year from 2025 to 2049. Win probabilities are blended from model output and ELO projections, with explanation cards showing projected ELO, trajectory direction, and key factors.

### Player Performance Forecast
87 top IPL players forecast via Prophet time-series modelling across five metrics: Runs, Batting Average, Strike Rate, Wickets, and Economy Rate. Includes an 80% confidence interval, full historical trend chart, and autocomplete player search.

### Tournament Simulator
Simulates all 45 round-robin matchups for any chosen season year. Outputs predicted standings ranked by expected wins, a 10×10 head-to-head win probability matrix, and ELO delta from the 2024 baseline.

### ELO Power Rankings
ELO ratings computed from all 1,243 matches (2008–2026). Expandable team cards with titles, home city, and win probability vs. a league-average opponent. Parameters: K-factor = 32, Base = 1500.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1 — Clone & set up the backend

```bash
git clone https://github.com/Player5987/ipl-predictor.git
cd ipl-predictor

python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

API available at `http://localhost:8000`  
Swagger UI at `http://localhost:8000/docs`

### 2 — Start the frontend

```bash
cd frontend
npm install
npm run dev
```

App available at `http://localhost:5173`

---

## API Reference

| Method | Endpoint | Description |
|:------:|----------|-------------|
| `GET` | `/` | Health check + model status |
| `GET` | `/teams` | All IPL teams with ELO ratings |
| `GET` | `/elo/rankings` | Current power rankings |
| `GET` | `/elo/projection/{year}` | Future ELO projections |
| `GET` | `/players` | All forecastable players |
| `GET` | `/players/search/{query}` | Search players by name |
| `POST` | `/predict/match` | Match winner prediction |
| `POST` | `/predict/player` | Player performance forecast |
| `GET` | `/predict/tournament/{year}` | Full season simulation |

### Match Prediction

```bash
curl -X POST http://localhost:8000/predict/match \
  -H "Content-Type: application/json" \
  -d '{"team1": "Mumbai Indians", "team2": "Chennai Super Kings", "year": 2026}'
```

```json
{
  "team1": "Mumbai Indians",
  "team2": "Chennai Super Kings",
  "year": 2026,
  "team1_win_prob": 0.534,
  "team2_win_prob": 0.466,
  "predicted_winner": "Mumbai Indians",
  "confidence": "low",
  "method": "model+elo"
}
```

### Player Forecast

```bash
curl -X POST http://localhost:8000/predict/player \
  -H "Content-Type: application/json" \
  -d '{"player_name": "V Kohli", "metric": "runs"}'
```

```json
{
  "player": "V Kohli",
  "forecast": {
    "predicted": 634.51,
    "lower_bound": 411.10,
    "upper_bound": 846.77,
    "confidence_interval": "80%",
    "method": "prophet"
  }
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML ensemble | XGBoost · LightGBM · RandomForest · CalibratedCV |
| Time series | Prophet |
| Features | Ball-by-ball phase stats · ELO · H2H · rolling form |
| Backend | FastAPI · Uvicorn · Joblib · scikit-learn |
| Frontend | React · Vite · Recharts · Axios |
| Data | Cricsheet BBB 2008–2026 · 1,243 matches · 87 players |

---

## Project Structure

```
ipl-predictor/
├── backend/
│   └── main.py                     # 9 FastAPI endpoints
├── frontend/
│   └── src/
│       ├── App.jsx                  # Main app with hero header
│       ├── App.css                  # Dark theme CSS
│       └── components/
│           ├── MatchPredictor.jsx
│           ├── PlayerForecast.jsx
│           ├── TournamentSim.jsx
│           └── EloRankings.jsx
├── models/
│   ├── stacking_model.pkl           # CalibratedRandomForest
│   ├── xgb_model.pkl                # XGBoost (feature importance)
│   ├── final_elo.json               # Team ELO ratings
│   ├── player_forecast.json         # 87 players (Prophet)
│   ├── player_stats.json            # 957 career stats
│   ├── players_list.json            # Frontend dropdown
│   ├── phase_features.json          # Ball-by-ball phase stats
│   └── feature_names.json           # 12 selected features
├── upgrade_features.py              # Phase feature engineering
├── final_honest_model.py            # Final model training
├── phase4_player_forecast.py        # Prophet forecasting
├── phase5_fastapi_backend.py        # Backend generator
├── requirements.txt
└── README.md
```

---

## IPL 2026 Data

Match data from IPL 2026 (up to May 31, 2026) is included:

| Stage | Match | Date |
|-------|-------|------|
| Final | Gujarat Titans vs Royal Challengers Bengaluru | May 31 |
| Qualifier 2 | Rajasthan Royals vs Gujarat Titans | May 29 |
| Qualifier 1 | Rajasthan Royals vs Sunrisers Hyderabad | May 27 |
| Eliminator | Royal Challengers Bengaluru vs Gujarat Titans | May 26 |
| League stage | 70 matches | March–May 2026 |

Source: [Cricsheet](https://cricsheet.org/downloads/) — 1,243 IPL matches in JSON format (2008–2026).

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you'd like to change.

---

## License

Distributed under the [MIT License](LICENSE).

---

<div align="center">

Built by [Gaurish](https://github.com/Player5987) &nbsp;·&nbsp; Data from [Cricsheet](https://cricsheet.org)

</div>
