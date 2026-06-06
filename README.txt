# 🏏 IPL Match Predictor (2025–2049)

> ML-powered IPL predictions using ball-by-ball phase features, Prophet player forecasting, ELO team ratings, and a stacking ensemble model. Built with FastAPI + React.

---

## 🚀 Live Demo

| Service | URL |
|---------|-----|
| Frontend | *coming soon — Vercel* |
| API | *coming soon — Render* |
| API Docs | `/docs` (Swagger UI) |

---

## 📊 Model Performance

| Component | Metric | Value | Notes |
|-----------|--------|-------|-------|
| Phase features | Correlation | 0.12 | `pp_dot_diff`, `death_econ_diff` |
| Match predictor | CV ROC-AUC | **0.558** | 8-fold time-series CV |
| ELO ratings | Teams | 10 | Validated vs real IPL standings |
| Player forecast | Players | 87 | Prophet 80% confidence interval |
| API endpoints | Count | 7 | FastAPI + Swagger UI |
| Frontend tabs | Count | 4 | React + Recharts |

> **Note:** IPL has genuine randomness. Professional betting markets with squad news, pitch conditions, and injury reports achieve 0.62–0.67 AUC. This model uses only historical match statistics — no external data sources.

---

## 🔍 Key Finding — IPL Era Analysis

Feature correlations are stronger in older eras, confirming IPL has become genuinely more unpredictable over time:

| Era | Matches | Avg Feature Correlation |
|-----|---------|------------------------|
| 2008–2015 (Classic IPL) | 508 | **0.117** |
| 2016–2019 (Mature IPL) | 235 | 0.091 |
| 2020–2024 (Modern IPL) | 333 | 0.031 |

This means a model trained on older data generalises less well to modern seasons — not because of poor modelling, but because IPL itself became a higher-parity league post-COVID.

---

## 🧠 Features

### Match Predictor
- Select any two IPL teams and any year from 2025 to 2049
- Win probabilities blending model output with ELO projection
- Year-based ELO projection using team trajectory and cyclical form patterns
- Explanation cards showing projected ELO, team trajectory direction, and key factors

### Player Performance Forecast
- 87 top IPL players forecast using Prophet time-series model
- Metrics: Runs, Batting Average, Strike Rate, Wickets, Economy Rate
- 80% confidence interval with full historical trend chart
- Autocomplete player search with career stats

### Tournament Simulator
- Simulates all 45 round-robin matchups for any season year
- Predicted standings ranked by expected wins
- Head-to-head win probability matrix (10×10)
- ELO change from 2024 baseline shown per team

### ELO Power Rankings
- ELO computed from all 1076 matches (2008–2025)
- Expandable team cards with titles, home city, win probability vs average
- K-factor = 32, Base = 1500

---

## 📈 Top Features by Importance

| Rank | Feature | Type | Importance | Correlation |
|------|---------|------|-----------|-------------|
| 1 | `pp_dot_diff` | ⭐ Phase | 0.146 | 0.118 |
| 2 | `death_econ_diff` | ⭐ Phase | 0.100 | 0.104 |
| 3 | `pp_wicket_rate_diff` | ⭐ Phase | — | 0.117 |
| 4 | `alltime_diff` | Base | 0.100 | 0.079 |
| 5 | `death_dot_diff` | ⭐ Phase | 0.099 | 0.080 |
| 6 | `pp_boundary_diff` | ⭐ Phase | 0.097 | 0.065 |
| 7 | `pp_run_rate_diff` | ⭐ Phase | 0.092 | 0.072 |
| 8 | `elo_diff` | Base | 0.092 | 0.056 |
| 9 | `form10_diff` | Base | 0.087 | 0.044 |
| 10 | `pp_econ_diff` | ⭐ Phase | 0.083 | 0.086 |

Phase features from ball-by-ball data dominate the top 10, confirming that powerplay and death-over performance are the strongest predictors of IPL match outcomes.

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend
```bash
# Clone the repo
git clone https://github.com/Player5987/ipl-predictor.git
cd ipl-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn backend.main:app --reload --port 8000
```

API available at `http://localhost:8000`
Swagger UI at `http://localhost:8000/docs`

### Frontend
```bash
cd frontend
npm install
npm run dev
```

App available at `http://localhost:5173`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost, LightGBM, RandomForest, CalibratedCV |
| Time Series | Prophet (player forecasting) |
| Features | Ball-by-ball phase stats, ELO, H2H, rolling form |
| Backend | FastAPI, Uvicorn, Joblib, scikit-learn |
| Frontend | React, Vite, Recharts, Axios |
| Data | Cricsheet BBB 2008–2025, 1076 matches, 87 players |

---

## 📁 Project Structure

```
ipl-predictor/
├── backend/
│   └── main.py                    # 7 FastAPI endpoints
├── frontend/
│   └── src/
│       ├── App.jsx                # Main app with hero header
│       ├── App.css                # Dark theme CSS
│       └── components/
│           ├── MatchPredictor.jsx
│           ├── PlayerForecast.jsx
│           ├── TournamentSim.jsx
│           └── EloRankings.jsx
├── models/
│   ├── stacking_model.pkl         # CalibratedRandomForest
│   ├── xgb_model.pkl              # XGBoost (feature importance)
│   ├── final_elo.json             # Team ELO ratings
│   ├── player_forecast.json       # 87 players (Prophet)
│   ├── player_stats.json          # 957 career stats
│   ├── players_list.json          # Frontend dropdown
│   ├── phase_features.json        # Ball-by-ball phase stats
│   └── feature_names.json         # 12 selected features
├── upgrade_features.py            # Phase feature engineering
├── final_honest_model.py          # Final model training
├── phase4_player_forecast.py      # Prophet forecasting
├── phase5_fastapi_backend.py      # Backend generator
├── requirements.txt
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + model status |
| `GET` | `/teams` | All IPL teams with ELO ratings |
| `GET` | `/elo/rankings` | Current power rankings |
| `GET` | `/elo/projection/{year}` | Future ELO projections |
| `GET` | `/players` | All forecastable players |
| `GET` | `/players/search/{query}` | Search players by name |
| `POST` | `/predict/match` | Match winner prediction |
| `POST` | `/predict/player` | Player performance forecast |
| `GET` | `/predict/tournament/{year}` | Full season simulation |

### Example: Match Prediction
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

### Example: Player Forecast
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
    "lower_bound": 411.1,
    "upper_bound": 846.77,
    "confidence_interval": "80%",
    "method": "prophet"
  }
}
```

---

## 📅 IPL 2026 Data

This project includes match data from IPL 2026 (up to May 31, 2026):

- **Final:** Gujarat Titans vs Royal Challengers Bengaluru (May 31)
- **Qualifier 2:** Rajasthan Royals vs Gujarat Titans (May 29)
- **Qualifier 1:** Rajasthan Royals vs Sunrisers Hyderabad (May 27)
- **Eliminator:** Royal Challengers Bengaluru vs Gujarat Titans (May 26)
- Full league stage: 70 matches (March–May 2026)

Source: [Cricsheet](https://cricsheet.org/downloads/) — 1243 IPL matches in JSON format (2008–2026)

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.

---

## 📄 License

MIT

---

*Built by [Gaurish](https://github.com/Player5987) · Data from [Cricsheet](https://cricsheet.org)*