
import pandas as pd
import numpy as np
import os, json, joblib, warnings
warnings.filterwarnings("ignore")
from prophet import Prophet

BASE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
MDIR = os.path.join(BASE, "models")
os.makedirs(MDIR, exist_ok=True)

print("=" * 62)
print("  IPL PREDICTOR — PHASE 4: PLAYER FORECASTING")
print("=" * 62)

def parse_season(s):
    s = str(s).strip()
    if "/" in s:   return int(s.split("/")[0]) + 1
    if "-" in s and len(s)==7: return int(s[:4]) + 1
    try:    return int(float(s))
    except: return None

print("\n[STEP 1] Loading player career stats ...")

# Try multiple sources for batting stats
batting = None
for path in [
    os.path.join(PROC,"batting_career.csv"),
    os.path.join(RAW,"2025","ipl_batsman.csv"),
]:
    if os.path.exists(path):
        batting = pd.read_csv(path)
        print(f"  Batting: {path} ({len(batting)} rows)")
        print(f"  Cols   : {batting.columns.tolist()}")
        break

bowling = None
for path in [
    os.path.join(PROC,"bowling_career.csv"),
    os.path.join(RAW,"2025","ipl_bowler.csv"),
]:
    if os.path.exists(path):
        bowling = pd.read_csv(path)
        print(f"  Bowling: {path} ({len(bowling)} rows)")
        print(f"  Cols   : {bowling.columns.tolist()}")
        break


print("\n[STEP 2] Building career stats from deliveries ...")

DELIV = os.path.join(RAW,"bbb",
        "deliveries_updated_mens_ipl_upto_2024.csv")
BBB   = os.path.join(RAW,"bbb",
        "matches_updated_mens_ipl_upto_2024.csv")

deliv_career_batting = None
deliv_career_bowling = None

if os.path.exists(DELIV):
    try:
        print(f"  Loading BBB deliveries (up to 2M rows)...")
        raw = pd.read_csv(DELIV, nrows=2_000_000,
                          low_memory=False)
        raw.columns = [c.lower().strip().replace(" ","_")
                       for c in raw.columns]
        print(f"  Shape: {raw.shape}")
        print(f"  Cols : {raw.columns.tolist()[:12]}")

        # Rename columns
        rn={}
        for std,opts in {
            "match_id":    ["match_id","matchid","id"],
            "batter":      ["batter","batsman","striker"],
            "bowler":      ["bowler"],
            "batting_team":["batting_team","bat_team"],
            "runs":        ["batsman_runs","batter_runs",
                            "runs_off_bat","runs","batter_run"],
            "is_wicket":   ["is_wicket","player_dismissed",
                            "wicket"],
            "season":      ["season","year"],
            "over":        ["over","over_number"],
        }.items():
            for o in opts:
                if o in raw.columns and std not in raw.columns:
                    rn[o]=std; break
        raw = raw.rename(columns=rn)

        # Map season via BBB match file
        if "season" not in raw.columns and \
           "match_id" in raw.columns:
            bbb_s = pd.read_csv(BBB,
                usecols=["matchId","season"])
            bbb_s["season"] = bbb_s["season"]\
                              .apply(parse_season)
            bbb_s["matchId"] = bbb_s["matchId"].astype(str)
            raw["match_id"]  = raw["match_id"].astype(str)
            raw = raw.merge(
                bbb_s.rename(columns={"matchId":"match_id"}),
                on="match_id",how="left")

        if "season" in raw.columns:
            raw["season"] = raw["season"].apply(parse_season)
            raw = raw.dropna(subset=["season"]).copy()
            raw["season"] = raw["season"].astype(int)
            raw = raw[raw["season"].between(2008,2025)].copy()
            raw["runs"] = pd.to_numeric(
                raw.get("runs",0),errors="coerce").fillna(0)

            # ── BATTING CAREER ─────────────────────────────
            if "batter" in raw.columns:
                # Wicket flag
                if "is_wicket" in raw.columns:
                    raw["is_out"] = pd.to_numeric(
                        raw["is_wicket"].astype(str)\
                        .str.extract(r'(\d+)',
                                     expand=False),
                        errors="coerce").fillna(0).astype(int)
                    raw["is_out"] = raw["is_out"].clip(0,1)
                else:
                    raw["is_out"] = 0

                bat_grp = raw.groupby(
                    ["batter","season"]).agg(
                    runs       = ("runs","sum"),
                    balls      = ("runs","count"),
                    dismissals = ("is_out","sum"),
                ).reset_index()
                bat_grp["average"] = (
                    bat_grp["runs"] /
                    bat_grp["dismissals"].clip(lower=1))
                bat_grp["strike_rate"] = (
                    bat_grp["runs"] /
                    bat_grp["balls"].clip(lower=1)) * 100
                bat_grp = bat_grp[bat_grp["balls"]>30].copy()
                deliv_career_batting = bat_grp
                print(f"  Batting career: "
                      f"{len(bat_grp)} player-seasons  "
                      f"({bat_grp['batter'].nunique()} players)")

            # ── BOWLING CAREER ─────────────────────────────
            if "bowler" in raw.columns:
                if "is_wicket" in raw.columns:
                    raw["bowl_wkt"] = raw["is_out"].clip(0,1)
                else:
                    raw["bowl_wkt"] = 0

                bowl_grp = raw.groupby(
                    ["bowler","season"]).agg(
                    wickets        = ("bowl_wkt","sum"),
                    runs_conceded  = ("runs","sum"),
                    balls_bowled   = ("runs","count"),
                ).reset_index()
                bowl_grp["economy"] = (
                    bowl_grp["runs_conceded"] /
                    bowl_grp["balls_bowled"].clip(lower=1)) * 6
                bowl_grp["bowling_avg"] = (
                    bowl_grp["runs_conceded"] /
                    bowl_grp["wickets"].clip(lower=1))
                bowl_grp = bowl_grp[
                    bowl_grp["balls_bowled"]>30].copy()
                deliv_career_bowling = bowl_grp
                print(f"  Bowling career: "
                      f"{len(bowl_grp)} player-seasons  "
                      f"({bowl_grp['bowler'].nunique()} players)")

    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()

# Use whichever source we have
if batting is None and deliv_career_batting is not None:
    batting = deliv_career_batting.rename(
        columns={"batter":"player_name"})
elif deliv_career_batting is not None:
    # merge both if both available
    pass

if bowling is None and deliv_career_bowling is not None:
    bowling = deliv_career_bowling.rename(
        columns={"bowler":"player_name"})

# ============================================================
# STEP 3 — STANDARDISE COLUMN NAMES
# ============================================================
print("\n[STEP 3] Standardising career dataframes ...")

def standardise_batting(df):
    if df is None: return None
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    rn={}
    for std,opts in {
        "player_name": ["player_name","batter","batsman",
                        "player","name"],
        "season":      ["season","year"],
        "runs":        ["runs","total_runs","run"],
        "average":     ["average","avg","batting_avg"],
        "strike_rate": ["strike_rate","sr","strike rate"],
        "balls":       ["balls","balls_faced","ball"],
        "dismissals":  ["dismissals","out","outs"],
    }.items():
        for o in opts:
            if o in df.columns and std not in df.columns:
                rn[o]=std; break
    df = df.rename(columns=rn)
    if "season" in df.columns:
        df["season"] = df["season"].apply(parse_season)
        df = df.dropna(subset=["season"]).copy()
        df["season"] = df["season"].astype(int)
    return df

def standardise_bowling(df):
    if df is None: return None
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    rn={}
    for std,opts in {
        "player_name": ["player_name","bowler","player","name"],
        "season":      ["season","year"],
        "wickets":     ["wickets","wkts","wicket"],
        "economy":     ["economy","econ","economy_rate"],
        "bowling_avg": ["bowling_avg","avg","bowling_average"],
        "balls_bowled":["balls_bowled","balls","overs"],
    }.items():
        for o in opts:
            if o in df.columns and std not in df.columns:
                rn[o]=std; break
    df = df.rename(columns=rn)
    if "season" in df.columns:
        df["season"] = df["season"].apply(parse_season)
        df = df.dropna(subset=["season"]).copy()
        df["season"] = df["season"].astype(int)
    return df

batting = standardise_batting(batting)
bowling = standardise_bowling(bowling)

if batting is not None:
    print(f"  Batting: {len(batting)} rows  "
          f"players={batting['player_name'].nunique()}")
    print(f"  Seasons: "
          f"{sorted(batting['season'].unique().tolist())}")
if bowling is not None:
    print(f"  Bowling: {len(bowling)} rows  "
          f"players={bowling['player_name'].nunique()}")


# ============================================================
# STEP 4 — PROPHET FORECAST FOR EACH PLAYER
# ============================================================
print("\n[STEP 4] Forecasting next season with Prophet ...")

def forecast_player(df, player_col, metric,
                    player_name, next_season=2026):
    """
    Forecast next season's metric for a player.
    Returns: {predicted, lower, upper, historical}
    """
    pdf = df[df[player_col]==player_name].copy()
    pdf = pdf.sort_values("season")

    if len(pdf) < 3:
        # Not enough data — use career average
        avg = float(pdf[metric].mean()) if len(pdf)>0 else 0.
        return {
            "predicted":   round(avg,2),
            "lower":       round(avg*0.75,2),
            "upper":       round(avg*1.25,2),
            "method":      "career_avg",
            "seasons_used":len(pdf),
            "historical":  pdf[["season",metric]]\
                           .rename(columns={metric:"value"})\
                           .to_dict("records")
        }

    # Prophet needs ds (date) and y (value)
    prop_df = pd.DataFrame({
        "ds": pd.to_datetime(
            pdf["season"].astype(str)+"-04-01"),
        "y":  pdf[metric].values
    })

    try:
        m = Prophet(
            yearly_seasonality=False,
            changepoint_prior_scale=0.3,
            interval_width=0.80,
            seasonality_mode="additive"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(prop_df)

        future = m.make_future_dataframe(
            periods=1, freq="YE")
        fc = m.predict(future).tail(1)

        pred  = float(fc["yhat"].values[0])
        lower = float(fc["yhat_lower"].values[0])
        upper = float(fc["yhat_upper"].values[0])

        # Clip negative predictions to 0
        pred  = max(0, round(pred, 2))
        lower = max(0, round(lower, 2))
        upper = max(0, round(upper, 2))

        return {
            "predicted":   pred,
            "lower":       lower,
            "upper":       upper,
            "method":      "prophet",
            "seasons_used":len(pdf),
            "historical":  pdf[["season",metric]]\
                           .rename(columns={metric:"value"})\
                           .to_dict("records")
        }
    except Exception as e:
        avg = float(pdf[metric].mean())
        return {
            "predicted":   round(avg,2),
            "lower":       round(avg*0.8,2),
            "upper":       round(avg*1.2,2),
            "method":      "fallback_avg",
            "error":       str(e),
            "historical":  pdf[["season",metric]]\
                           .rename(columns={metric:"value"})\
                           .to_dict("records")
        }


# Forecast for top players
player_forecasts = {}

if batting is not None and "player_name" in batting.columns:
    # Top batters by total runs
    top_batters = batting.groupby("player_name")["runs"]\
                         .sum().nlargest(50).index.tolist()
    print(f"\n  Forecasting {len(top_batters)} top batters ...")

    for player in top_batters:
        player_forecasts[player] = {}
        for metric in ["runs","strike_rate","average"]:
            if metric in batting.columns:
                fc = forecast_player(
                    batting,"player_name",
                    metric,player)
                player_forecasts[player][metric] = fc

    print(f"  Done. Sample forecasts:")
    for p in top_batters[:3]:
        if "runs" in player_forecasts.get(p,{}):
            fc = player_forecasts[p]["runs"]
            print(f"    {p:25s}  "
                  f"pred={fc['predicted']:.0f}  "
                  f"CI=[{fc['lower']:.0f},{fc['upper']:.0f}]"
                  f"  method={fc['method']}")

if bowling is not None and "player_name" in bowling.columns:
    # Top bowlers by wickets
    top_bowlers = bowling.groupby("player_name")["wickets"]\
                         .sum().nlargest(40).index.tolist()
    print(f"\n  Forecasting {len(top_bowlers)} top bowlers ...")

    for player in top_bowlers:
        if player not in player_forecasts:
            player_forecasts[player] = {}
        for metric in ["wickets","economy","bowling_avg"]:
            if metric in bowling.columns:
                fc = forecast_player(
                    bowling,"player_name",
                    metric,player)
                player_forecasts[player][metric]=fc

    print(f"  Done. Sample bowling forecasts:")
    for p in top_bowlers[:3]:
        if "wickets" in player_forecasts.get(p,{}):
            fc = player_forecasts[p]["wickets"]
            print(f"    {p:25s}  "
                  f"pred={fc['predicted']:.1f} wkts  "
                  f"method={fc['method']}")


# ============================================================
# STEP 5 — BUILD PLAYER STATS LOOKUP
# ============================================================
print("\n[STEP 5] Building player stats lookup ...")

player_stats = {}

if batting is not None:
    for player in batting["player_name"].unique():
        pdf = batting[batting["player_name"]==player]\
              .sort_values("season")
        player_stats[player] = {
            "type":          "batter",
            "seasons_played":int(len(pdf)),
            "career_runs":   int(pdf["runs"].sum()),
            "career_avg":    round(float(pdf["average"].mean()),2)
                             if "average" in pdf.columns else 0,
            "career_sr":     round(float(pdf["strike_rate"].mean()),2)
                             if "strike_rate" in pdf.columns else 0,
            "last_season_runs":
                int(pdf.tail(1)["runs"].values[0])
                if len(pdf)>0 else 0,
            "seasons": pdf[["season","runs"]].to_dict("records")
        }

if bowling is not None:
    for player in bowling["player_name"].unique():
        pdf = bowling[bowling["player_name"]==player]\
              .sort_values("season")
        if player not in player_stats:
            player_stats[player] = {"type":"bowler"}
        player_stats[player].update({
            "career_wickets":int(pdf["wickets"].sum()),
            "career_economy":
                round(float(pdf["economy"].mean()),2)
                if "economy" in pdf.columns else 0,
        })

print(f"  Total players in lookup: {len(player_stats)}")

# ============================================================
# STEP 6 — SAVE ALL OUTPUTS
# ============================================================
print("\n[STEP 6] Saving ...")

# Save player forecasts
with open(os.path.join(MDIR,"player_forecast.json"),"w") as f:
    json.dump(player_forecasts, f, indent=2)
print(f"  Saved player_forecast.json  "
      f"({len(player_forecasts)} players)")

# Save player stats
with open(os.path.join(MDIR,"player_stats.json"),"w") as f:
    json.dump(player_stats, f, indent=2)
print(f"  Saved player_stats.json  "
      f"({len(player_stats)} players)")

# Save batting + bowling career CSVs
if deliv_career_batting is not None:
    out_bat = deliv_career_batting.copy()
    out_bat.to_csv(
        os.path.join(PROC,"batting_career_final.csv"),
        index=False)
    print(f"  Saved batting_career_final.csv")

if deliv_career_bowling is not None:
    out_bowl = deliv_career_bowling.copy()
    out_bowl.to_csv(
        os.path.join(PROC,"bowling_career_final.csv"),
        index=False)
    print(f"  Saved bowling_career_final.csv")

# Save list of forecastable players (for frontend dropdown)
forecastable = [
    {"name": p,
     "metrics": list(v.keys()),
     "career_runs": player_stats.get(p,{}).get("career_runs",0),
     "career_wickets": player_stats.get(p,{}).get("career_wickets",0)}
    for p,v in player_forecasts.items()
    if v  # non-empty
]
forecastable.sort(key=lambda x:-x["career_runs"])
with open(os.path.join(MDIR,"players_list.json"),"w") as f:
    json.dump(forecastable, f, indent=2)
print(f"  Saved players_list.json  "
      f"({len(forecastable)} players)")


# ============================================================
# FINAL REPORT
# ============================================================
print()
print("="*62)
print("  PHASE 4 COMPLETE")
print("="*62)
print(f"  Players forecast  : {len(player_forecasts)}")
print(f"  Players in lookup : {len(player_stats)}")
print()
print("  Files saved to models/:")
print("    player_forecast.json  ← used by /predict/player API")
print("    player_stats.json     ← career stats lookup")
print("    players_list.json     ← frontend dropdown list")
print()
if player_forecasts:
    p = list(player_forecasts.keys())[0]
    metrics = player_forecasts[p]
    print(f"  Example — {p}:")
    for metric, fc in list(metrics.items())[:3]:
        if isinstance(fc, dict) and "predicted" in fc:
            print(f"    {metric:15s}  "
                  f"predicted={fc['predicted']}  "
                  f"CI=[{fc['lower']}, {fc['upper']}]")
print()
print("  Next step → phase5_fastapi_backend.py")
print("="*62)
