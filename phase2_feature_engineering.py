

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


BASE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
os.makedirs(PROC, exist_ok=True)

print("=" * 60)
print("  IPL PREDICTOR — PHASE 2: FEATURE ENGINEERING")
print("=" * 60)
print()


combined_path = os.path.join(PROC, "matches_all.csv")
raw_path      = os.path.join(RAW,  "matches.csv")

if os.path.exists(combined_path):
    print("[LOAD] Reading data/processed/matches_all.csv")
    df = pd.read_csv(combined_path)
elif os.path.exists(raw_path):
    print("[LOAD] Reading data/raw/matches.csv")
    df = pd.read_csv(raw_path)
else:
    raise FileNotFoundError(
        "Could not find matches file.\n"
        "Expected: data/processed/matches_all.csv\n"
        "      or: data/raw/matches.csv"
    )

print(f"       Rows loaded  : {len(df)}")
print(f"       Columns found: {df.columns.tolist()}")
print()


date_candidates = ["date", "match_date", "Date", "start_date"]
date_col = next((c for c in date_candidates if c in df.columns), None)
if date_col is None:
    raise ValueError(f"No date column found. Columns: {df.columns.tolist()}")
if date_col != "date":
    df = df.rename(columns={date_col: "date"})

df["date"] = pd.to_datetime(df["date"], dayfirst=True, format='mixed')
df = df.sort_values("date").reset_index(drop=True)


col_map = {
    "team_1":         "team1",
    "team_2":         "team2",
    "match_winner":   "winner",
    "winning_team":   "winner",
    "toss_won":       "toss_winner",
    "toss_decision":  "toss_decision",
    "venue_name":     "venue",
    "ground":         "venue",
}
df = df.rename(columns={k: v for k, v in col_map.items()
                         if k in df.columns})

required = ["team1", "team2", "winner"]
missing  = [c for c in required if c not in df.columns]
if missing:
    print(f"[WARN] Missing columns: {missing}")
    print(f"       Available: {df.columns.tolist()}")
    print("       Fix column names above and re-run.")
    raise SystemExit(1)

print(f"[INFO] Date range : {df['date'].min().date()} "
      f"→ {df['date'].max().date()}")
print(f"[INFO] Seasons    : {sorted(df['date'].dt.year.unique().tolist())}")
print()



print("[STEP 1] Standardising team names ...")

TEAM_MAP = {
    "Delhi Daredevils":          "Delhi Capitals",
    "Deccan Chargers":           "Sunrisers Hyderabad",
    "Rising Pune Supergiant":    "Rising Pune Supergiants",
    "Kings XI Punjab":           "Punjab Kings",
    "Pune Warriors":             "Rising Pune Supergiants",
}

for col in ["team1", "team2", "winner", "toss_winner"]:
    if col in df.columns:
        df[col] = df[col].replace(TEAM_MAP)

# Remove matches with no result (rain, abandoned, etc.)
before = len(df)
df = df[df["winner"].notna() & (df["winner"] != "")].copy()
after  = len(df)
print(f"         Removed {before - after} no-result matches")
print(f"         Clean matches: {after}")
print(f"         Teams: {sorted(df['team1'].unique())}")
print()




print("[STEP 2] Computing ELO ratings ...")

def compute_elo(df, k=32, base=1500):
    elo     = {}     # live current ratings
    pre_t1  = []     # elo of team1 BEFORE this match
    pre_t2  = []     # elo of team2 BEFORE this match

    for _, row in df.iterrows():
        t1, t2  = row["team1"], row["team2"]
        winner  = row["winner"]

        r1 = elo.get(t1, base)
        r2 = elo.get(t2, base)

        # Record BEFORE updating (critical — no leakage)
        pre_t1.append(round(r1, 2))
        pre_t2.append(round(r2, 2))

        # ELO expected win probabilities
        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        exp2 = 1 - exp1

        # Actual results (1 = win, 0 = loss)
        a1 = 1.0 if winner == t1 else 0.0
        a2 = 1.0 - a1

        # Update ratings AFTER recording
        elo[t1] = r1 + k * (a1 - exp1)
        elo[t2] = r2 + k * (a2 - exp2)

    df = df.copy()
    df["elo_t1"]   = pre_t1
    df["elo_t2"]   = pre_t2
    df["elo_diff"] = [round(a - b, 2)
                      for a, b in zip(pre_t1, pre_t2)]

    print("         Final ELO ratings (all teams):")
    for team, rating in sorted(elo.items(),
                                key=lambda x: -x[1]):
        bar = "█" * int((rating - 1300) / 20)
        print(f"           {team:28s} {rating:6.0f}  {bar}")
    return df, elo

df, final_elo = compute_elo(df)
print()



print("[STEP 3] Computing head-to-head win rates ...")

h2h_wr_list = []
h2h_n_list  = []

for pos, (i, row) in enumerate(df.iterrows()):
    t1, t2 = row["team1"], row["team2"]

    # Only look at rows BEFORE current position
    past = df.iloc[:pos]
    mask = (
        ((past["team1"] == t1) & (past["team2"] == t2)) |
        ((past["team1"] == t2) & (past["team2"] == t1))
    )
    h2h = past[mask]

    if len(h2h) == 0:
        h2h_wr_list.append(0.5)   # no history → neutral
        h2h_n_list.append(0)
    else:
        wins = (h2h["winner"] == t1).sum()
        h2h_wr_list.append(round(wins / len(h2h), 4))
        h2h_n_list.append(len(h2h))

df["h2h_winrate_t1"] = h2h_wr_list
df["h2h_n_matches"]  = h2h_n_list

print(f"         Average H2H matches available per game: "
      f"{np.mean(h2h_n_list):.1f}")
print(f"         Games with H2H history: "
      f"{sum(1 for n in h2h_n_list if n > 0)}")
print()



print("[STEP 4] Computing rolling form (last 5 matches) ...")

def get_form(df, team, before_pos, window=5):
    """Win rate and streak for team in last N matches."""
    past    = df.iloc[:before_pos]
    mask    = (past["team1"] == team) | (past["team2"] == team)
    recent  = past[mask].tail(window)

    if len(recent) == 0:
        return 0.5, 0   # no history

    wins   = (recent["winner"] == team).sum()
    wr     = round(wins / len(recent), 4)

    # Winning streak: last 3 all won
    last3  = recent.tail(3)
    streak = int(len(last3) == 3 and
                 (last3["winner"] == team).all())
    return wr, streak

form_t1, form_t2       = [], []
streak_t1, streak_t2   = [], []

for pos, (i, row) in enumerate(df.iterrows()):
    f1, s1 = get_form(df, row["team1"], pos)
    f2, s2 = get_form(df, row["team2"], pos)
    form_t1.append(f1);   form_t2.append(f2)
    streak_t1.append(s1); streak_t2.append(s2)

df["form_t1"]   = form_t1
df["form_t2"]   = form_t2
df["streak_t1"] = streak_t1
df["streak_t2"] = streak_t2
df["form_diff"] = [round(a - b, 4)
                   for a, b in zip(form_t1, form_t2)]

print(f"         Avg form_t1 : {np.mean(form_t1):.3f}")
print(f"         Avg form_t2 : {np.mean(form_t2):.3f}")
print(f"         Streak events: "
      f"{sum(streak_t1)} team1 / {sum(streak_t2)} team2")
print()



print("[STEP 5] Computing venue win rates ...")

venue_t1, venue_t2 = [], []

for pos, (i, row) in enumerate(df.iterrows()):
    t1, t2  = row["team1"], row["team2"]
    venue   = row.get("venue", "Unknown")
    past    = df.iloc[:pos]

    def vwr(team, venue, past):
        mask = (
            ((past["team1"] == team) | (past["team2"] == team))
            & (past["venue"] == venue)
        )
        v = past[mask]
        if len(v) == 0:
            return 0.5
        return round((v["winner"] == team).sum() / len(v), 4)

    venue_t1.append(vwr(t1, venue, past))
    venue_t2.append(vwr(t2, venue, past))

df["venue_wr_t1"] = venue_t1
df["venue_wr_t2"] = venue_t2
df["venue_diff"]  = [round(a - b, 4)
                     for a, b in zip(venue_t1, venue_t2)]

n_venues = df["venue"].nunique() if "venue" in df.columns else 0
print(f"         Venues in dataset: {n_venues}")
print(f"         Avg venue_wr_t1 : {np.mean(venue_t1):.3f}")
print()



print("[STEP 6] Adding toss features ...")

toss_t1_col    = []
bat_col        = []
field_adv_col  = []

for _, row in df.iterrows():
    toss_winner = str(row.get("toss_winner", "")).strip()
    decision    = str(row.get("toss_decision", "")).lower().strip()

    t1_won_toss = int(toss_winner == row["team1"])
    bat_first   = int(decision == "bat")
    # Team1 wins toss AND chooses to field
    field_adv   = int(t1_won_toss == 1 and bat_first == 0)

    toss_t1_col.append(t1_won_toss)
    bat_col.append(bat_first)
    field_adv_col.append(field_adv)

df["toss_t1"]      = toss_t1_col
df["bat_first"]    = bat_col
df["field_adv_t1"] = field_adv_col

toss_impact = df.groupby("toss_t1")["target"].mean() \
    if "target" in df.columns else None
print(f"         Toss won by team1: "
      f"{sum(toss_t1_col)} / {len(df)} matches")
print()




print("[STEP 7] Adding context features ...")

df["season_num"] = df["date"].dt.year - 2007
# 2008 → 1,  2015 → 8,  2025 → 18
# Higher number = more mature league, teams more stable

df["month"] = df["date"].dt.month
# April(4) vs May(5) conditions differ — May is hotter,
# pitches slower, death bowling more important

# Playoff flag
if "match_type" in df.columns:
    df["is_playoff"] = df["match_type"].str.lower()\
        .str.contains("final|playoff|eliminator|qualifier",
                      na=False).astype(int)
elif "match_number" in df.columns:
    # Playoff matches are usually numbered 71-74 in IPL
    df["is_playoff"] = (df["match_number"] >= 71).astype(int)
else:
    df["is_playoff"] = 0

print(f"         Season range : {df['season_num'].min()} "
      f"→ {df['season_num'].max()}")
print(f"         Playoff matches: {df['is_playoff'].sum()}")
print()




print("[STEP 8] Assembling final feature matrix ...")

df["target"] = (df["winner"] == df["team1"]).astype(int)

FEATURE_COLS = [
    # ── Team strength ──
    "elo_t1",          # Team1 ELO rating before match
    "elo_t2",          # Team2 ELO rating before match
    "elo_diff",        # elo_t1 - elo_t2 (most predictive)

    # ── Head to head ──
    "h2h_winrate_t1",  # Team1 win rate vs Team2 historically
    "h2h_n_matches",   # How many H2H matches (reliability signal)

    # ── Recent form ──
    "form_t1",         # Team1 win rate last 5 matches
    "form_t2",         # Team2 win rate last 5 matches
    "form_diff",       # form_t1 - form_t2
    "streak_t1",       # 1 if Team1 won last 3 in a row
    "streak_t2",       # 1 if Team2 won last 3 in a row

    # ── Venue ──
    "venue_wr_t1",     # Team1 win rate at this venue
    "venue_wr_t2",     # Team2 win rate at this venue
    "venue_diff",      # venue_wr_t1 - venue_wr_t2

    # ── Toss ──
    "toss_t1",         # 1 if team1 won the toss
    "bat_first",       # 1 if toss winner chose to bat
    "field_adv_t1",    # 1 if team1 won toss & chose to field

    # ── Context ──
    "season_num",      # Which season (1-18)
    "month",           # Month of match (4=April, 5=May, 6=June)
    "is_playoff",      # 1 if knockout stage match
]

# Keep only columns that actually exist in the dataframe
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"         Features included: {len(FEATURE_COLS)}")
print(f"         Features: {FEATURE_COLS}")

# Check for NaN values
nan_report = df[FEATURE_COLS].isna().sum()
nan_features = nan_report[nan_report > 0]

if len(nan_features) > 0:
    print(f"\n[WARN]   NaN values found:")
    for feat, count in nan_features.items():
        print(f"           {feat}: {count} NaNs")
    print("         Filling with 0.5 (neutral)")
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.5)
else:
    print("         No NaN values — perfectly clean!")

# Final output dataframe
output = df[
    ["date", "season_num", "team1", "team2", "winner"]
    + FEATURE_COLS
    + ["target"]
].copy()

# Save to disk
out_path = os.path.join(PROC, "features.csv")
output.to_csv(out_path, index=False)

# Also save feature column list for the model training script
import json
with open(os.path.join(PROC, "feature_cols.json"), "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

print()
print("=" * 60)
print("  FEATURE MATRIX COMPLETE")
print("=" * 60)
print(f"  Saved to  : data/processed/features.csv")
print(f"  Shape     : {output.shape}")
print(f"  Features  : {len(FEATURE_COLS)}")
print(f"  Date range: {output['date'].min().date()} "
      f"→ {output['date'].max().date()}")
print(f"  Target    : {output['target'].mean():.1%} "
      f"team1 wins (should be ~50%)")
print()



print("FEATURE ANALYSIS")
print("-" * 50)

corr = output[FEATURE_COLS + ["target"]]\
       .corr()["target"].drop("target")\
       .sort_values(key=abs, ascending=False)

print("\nFeature correlation with match outcome (abs):")
print("  (Higher = more predictive)\n")
for feat, val in corr.items():
    bar   = "█" * int(abs(val) * 40)
    sign  = "+" if val > 0 else "-"
    print(f"  {feat:20s} {sign}{abs(val):.4f}  {bar}")

print()

# ELO sanity check
print("ELO rating sanity check:")
print(f"  Minimum ELO seen : {output['elo_t1'].min():.0f}")
print(f"  Maximum ELO seen : {output['elo_t1'].max():.0f}")
print(f"  Mean ELO         : {output['elo_t1'].mean():.0f} "
      f"(should be ~1500)")

# Season-wise win rates
print("\nTeam1 win rate by recent season:")
output["year"] = output["date"].dt.year
season_stats = output.groupby("year")["target"].agg(
    matches="count", team1_wins="sum"
)
season_stats["win_rate"] = (
    season_stats["team1_wins"] / season_stats["matches"]
).round(3)
print(season_stats.tail(6).to_string())

# Top winning teams
print("\nMost wins by team (overall):")
win_counts = pd.concat([
    output[output["target"] == 1]["team1"],
    output[output["target"] == 0]["team2"]
]).value_counts().head(6)
for team, wins in win_counts.items():
    bar = "█" * (wins // 10)
    print(f"  {team:28s} {wins:3d} wins  {bar}")

print()
print("=" * 60)
print("  PHASE 2 COMPLETE")
print("  Next step → run phase3_train_models.py")
print("=" * 60)
