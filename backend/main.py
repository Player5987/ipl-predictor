

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import joblib, json, os, numpy as np, warnings
warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDIR  = os.path.join(BASE, "models")
PROC  = os.path.join(BASE, "data", "processed")

store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    # Load stacking model
    for model_name in ["stacking_model.pkl",
                        "logisticreg_model.pkl",
                        "lightgbm_model.pkl",
                        "xgboost_model.pkl"]:
        p = os.path.join(MDIR, model_name)
        if os.path.exists(p):
            try:
                store["stack"] = joblib.load(p)
                print(f"  Model loaded: {model_name}")
                break
            except Exception as e:
                print(f"  {model_name} failed: {e}")

    # Load feature names
    for fn in ["feature_names.pkl", "feature_names.json"]:
        p = os.path.join(MDIR, fn)
        if os.path.exists(p):
            try:
                if fn.endswith(".pkl"):
                    store["features"] = joblib.load(p)
                else:
                    with open(p, encoding="utf-8") as f:
                        store["features"] = json.load(f)
                print(f"  Features: {len(store['features'])} cols")
                break
            except Exception as e:
                print(f"  {fn} failed: {e}")

    
    p = os.path.join(MDIR, "final_elo.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            store["elo"] = json.load(f)
        print(f"  ELO: {len(store['elo'])} teams")
    else:
        store["elo"] = {}

    
    for fname, key in [
        ("player_forecast.json", "player_forecast"),
        ("player_stats.json",    "player_stats"),
        ("players_list.json",    "players_list"),
    ]:
        p = os.path.join(MDIR, fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                store[key] = json.load(f)
        else:
            store[key] = {} if key != "players_list" else []

    n = len(store.get("player_forecast", {}))
    print(f"  Players: {n} forecasts loaded")
    print("API ready.")
    yield
    store.clear()


app = FastAPI(
    title="IPL Predictor API",
    description="Predict IPL match winners and player performance",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

# ── HELPERS ──────────────────────────────────────────────
def get_elo(team: str) -> float:
    return float(store["elo"].get(team, 1500.0))

def elo_win_prob(team1: str, team2: str) -> float:
    r1 = get_elo(team1); r2 = get_elo(team2)
    return round(1 / (1 + 10**((r2-r1)/400)), 4)


def build_features(team1, team2, year):
    feat_cols = store.get("features", [])
    elo1 = get_elo(team1, year)
    elo2 = get_elo(team2, year)
    elo_wr = elo_win_prob(team1, team2, year)

    mapping = {
        
        "pp_dot_diff":          0.0,
        "death_wicket_diff":    0.0,
        "pp_wicket_rate_diff":  0.0,
        "death_econ_diff":      0.0,
        "pp_econ_diff":         0.0,
        "death_dot_diff":       0.0,
        "pp_run_rate_diff":     0.0,
        "pp_boundary_diff":     0.0,
        # Base features
        "alltime_diff":  round(elo_wr - (1 - elo_wr), 4),
        "elo_diff":      round(elo1 - elo2, 2),
        "form10_diff":   0.0,
        "margin_diff":   0.0,
        # Context
        "season_num":    year - 2007,
        "month":         4,
        "toss_t1":       0,
        "bat_first":     0,
    }
    return np.array([[mapping.get(f, 0.0) for f in feat_cols]])

def predict_prob(team1, team2, year):
    elo_p = elo_win_prob(team1, team2)

    if "stack" not in store or "features" not in store:
        return round(elo_p, 3), "elo_only"

    try:
        X    = build_features(team1, team2, year)
        prob = float(store["stack"].predict_proba(X)[0][1])
        # Blend: 40% model + 60% ELO (ELO is more reliable here)
        blended = round(0.4*prob + 0.6*elo_p, 3)
        return blended, "model+elo"
    except Exception as e:
        return round(elo_p, 3), f"elo_fallback({e})"



@app.get("/")
async def root():
    return {
        "status":  "ok",
        "message": "IPL Predictor API v1.0",
        "model_loaded":   "stack" in store,
        "players_loaded": len(store.get("player_forecast",{})),
        "endpoints": ["/docs","/teams","/players",
                      "/predict/match","/predict/player",
                      "/predict/tournament/{year}",
                      "/elo/rankings"]
    }

@app.get("/teams")
async def list_teams():
    teams = []
    for t in TEAMS:
        elo  = get_elo(t)
        teams.append({
            "name": t,
            "elo":  round(elo, 1),
            "win_prob_vs_average": round(
                1/(1+10**((1500-elo)/400))*100, 1)
        })
    teams.sort(key=lambda x: -x["elo"])
    return {"teams": teams, "count": len(teams)}

# ── GET /elo/rankings ─────────────────────────────────────
@app.get("/elo/rankings")
async def elo_rankings():
    ranked = sorted(
        [(t, get_elo(t)) for t in TEAMS],
        key=lambda x: -x[1]
    )
    return {
        "rankings": [
            {"rank": i+1, "team": t, "elo": round(e, 1)}
            for i,(t,e) in enumerate(ranked)
        ]
    }

# ── GET /players ──────────────────────────────────────────
@app.get("/players")
async def list_players():
    return {
        "players": store.get("players_list", []),
        "count":   len(store.get("players_list", []))
    }

# ── GET /players/search/{q} ───────────────────────────────
@app.get("/players/search/{query}")
async def search_players(query: str):
    all_p   = list(store.get("player_forecast",{}).keys())
    results = sorted([p for p in all_p
                      if query.lower() in p.lower()])[:20]
    return {"query": query, "results": results,
            "count": len(results)}

# ── POST /predict/match ───────────────────────────────────
class MatchRequest(BaseModel):
    team1:  str
    team2:  str
    year:   int
    venue:  Optional[str] = "neutral"

@app.post("/predict/match")
async def predict_match(req: MatchRequest):
    if req.team1 not in TEAMS:
        raise HTTPException(422,
            f"Unknown team: {req.team1!r}. Valid: {TEAMS}")
    if req.team2 not in TEAMS:
        raise HTTPException(422,
            f"Unknown team: {req.team2!r}")
    if req.team1 == req.team2:
        raise HTTPException(422, "Teams must be different")
    if not 2025 <= req.year <= 2049:
        raise HTTPException(422,
            f"Year must be 2025-2049, got {req.year}")

    p1, method = predict_prob(req.team1, req.team2, req.year)
    p2 = round(1.0 - p1, 3)

    margin = abs(p1 - 0.5)
    confidence = ("high" if margin > 0.15 else
                  "medium" if margin > 0.08 else "low")

    elo1 = get_elo(req.team1); elo2 = get_elo(req.team2)
    explanation = [
        {
            "factor": "ELO rating difference",
            "value":  round(elo1 - elo2, 1),
            "detail": f"{req.team1}: {elo1:.0f} vs "
                      f"{req.team2}: {elo2:.0f}",
            "favours": req.team1 if elo1 > elo2 else req.team2,
        },
        {
            "factor": "Historical win rate",
            "value":  round(elo_win_prob(req.team1,req.team2)*100,1),
            "detail": f"{req.team1} wins "
                      f"{elo_win_prob(req.team1,req.team2)*100:.1f}% "
                      f"based on ELO history",
            "favours": (req.team1
                        if elo_win_prob(req.team1,req.team2) > 0.5
                        else req.team2),
        },
    ]

    return {
        "team1":             req.team1,
        "team2":             req.team2,
        "year":              req.year,
        "venue":             req.venue,
        "team1_win_prob":    p1,
        "team2_win_prob":    p2,
        "predicted_winner":  req.team1 if p1 >= p2 else req.team2,
        "confidence":        confidence,
        "method":            method,
        "explanation":       explanation,
        "disclaimer": ("Cricket is unpredictable. "
                       "Favourites lose ~40% of IPL matches.")
    }

# ── POST /predict/player ──────────────────────────────────
class PlayerRequest(BaseModel):
    player_name: str
    metric:      Optional[str] = "runs"

@app.post("/predict/player")
async def predict_player(req: PlayerRequest):
    forecasts = store.get("player_forecast", {})
    stats     = store.get("player_stats", {})

    # Exact match first
    player = req.player_name
    if player not in forecasts:
        # Partial match
        found = [p for p in forecasts
                 if req.player_name.lower() in p.lower()]
        if not found:
            raise HTTPException(404,
                f"Player {req.player_name!r} not found. "
                f"Try GET /players/search/{req.player_name}")
        player = found[0]

    fc_all = forecasts[player]
    if req.metric not in fc_all:
        raise HTTPException(422,
            f"Metric {req.metric!r} not available. "
            f"Available: {list(fc_all.keys())}")

    fc = fc_all[req.metric]
    st = stats.get(player, {})

    return {
        "player":      player,
        "metric":      req.metric,
        "next_season": 2026,
        "forecast": {
            "predicted":          fc["predicted"],
            "lower_bound":        fc["lower"],
            "upper_bound":        fc["upper"],
            "confidence_interval":"80%",
            "method":             fc.get("method","prophet"),
            "seasons_of_data":    fc.get("seasons_used", 0),
        },
        "historical":    fc.get("historical", []),
        "career_stats":  {
            "career_runs":    st.get("career_runs", 0),
            "career_avg":     st.get("career_avg", 0),
            "career_sr":      st.get("career_sr", 0),
            "career_wickets": st.get("career_wickets", 0),
            "career_economy": st.get("career_economy", 0),
        },
        "disclaimer": ("Based on Prophet time-series forecast. "
                       "Actual performance depends on team, role, "
                       "fitness, and match conditions.")
    }

# ── GET /predict/tournament/{year} ───────────────────────
@app.get("/predict/tournament/{year}")
async def simulate_tournament(year: int):
    if not 2025 <= year <= 2049:
        raise HTTPException(422, "Year must be 2025-2049")

    matchups     = []
    team_exp_wins = {t: 0.0 for t in TEAMS}

    for i, t1 in enumerate(TEAMS):
        for t2 in TEAMS[i+1:]:
            p1, _ = predict_prob(t1, t2, year)
            p2 = round(1.0 - p1, 3)
            matchups.append({
                "team1": t1, "team2": t2,
                "team1_prob": p1, "team2_prob": p2,
            })
            team_exp_wins[t1] += p1
            team_exp_wins[t2] += p2

    standings = sorted(
        [{"team": t, "expected_wins": round(w, 2),
          "elo": round(get_elo(t), 0)}
         for t,w in team_exp_wins.items()],
        key=lambda x: -x["expected_wins"]
    )
    for i,s in enumerate(standings):
        s["predicted_rank"] = i+1

    return {
        "year":     year,
        "matchups": matchups,
        "predicted_standings": standings,
        "n_matchups": len(matchups),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True)
