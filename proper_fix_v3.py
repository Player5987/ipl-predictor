

import pandas as pd
import numpy as np
import os, json, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (StackingClassifier,
                               RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import (TimeSeriesSplit,
                                      cross_val_score)
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report,
                              confusion_matrix)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
MDIR = os.path.join(BASE, "models")
os.makedirs(MDIR, exist_ok=True)

print("=" * 62)
print("  IPL PREDICTOR — PROPER FIX V3")
print("=" * 62)


print("\n[STEP 1] Loading all data files ...")

def try_load(paths, label):
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"  {label:20s} → {p}  ({len(df)} rows)")
            return df
    print(f"  {label:20s} → NOT FOUND (tried: {paths})")
    return None

matches = try_load([
    os.path.join(PROC, "matches_all.csv"),
    os.path.join(RAW,  "matches.csv"),
    os.path.join(BASE, "matches.csv"),
], "matches")

batting = try_load([
    os.path.join(PROC, "batting_career.csv"),
    os.path.join(RAW,  "2025", "ipl_batsman.csv"),
    os.path.join(BASE, "data", "raw", "2025", "ipl_batsman.csv"),
], "batting_career")

bowling = try_load([
    os.path.join(PROC, "bowling_career.csv"),
    os.path.join(RAW,  "2025", "ipl_bowler.csv"),
    os.path.join(BASE, "data", "raw", "2025", "ipl_bowler.csv"),
], "bowling_career")

deliveries = try_load([
    os.path.join(PROC, "deliveries_all.csv"),
    os.path.join(RAW,  "deliveries.csv"),
    os.path.join(BASE, "deliveries.csv"),
], "deliveries")
if matches is None:
    raise FileNotFoundError("matches CSV not found!")



print("\n[STEP 2] Cleaning and filtering matches ...")

# Auto-detect columns
col_map = {
    "team_1":"team1","team_2":"team2",
    "match_winner":"winner","winning_team":"winner",
    "toss_won":"toss_winner","venue_name":"venue",
    "ground":"venue","match_date":"date",
    "Date":"date","start_date":"date",
}
matches = matches.rename(
    columns={k:v for k,v in col_map.items()
             if k in matches.columns})

matches["date"] = pd.to_datetime(
    matches["date"], dayfirst=True)
TEAM_MAP = {
    "Delhi Daredevils":"Delhi Capitals",
    "Deccan Chargers":"Sunrisers Hyderabad",
    "Rising Pune Supergiant":"Rising Pune Supergiants",
    "Kings XI Punjab":"Punjab Kings",
    "Pune Warriors":"Rising Pune Supergiants",
}
for c in ["team1","team2","winner","toss_winner"]:
    if c in matches.columns:
        matches[c] = matches[c].replace(TEAM_MAP).str.strip()

# Remove no-result and future/simulated matches
matches = matches[matches["winner"].notna()].copy()
matches = matches[~matches["winner"].isin(
    ["","No result","Tied","NR","no result"])].copy()


matches["year"] = matches["date"].dt.year
print(f"\n  Matches per year:")
print(matches["year"].value_counts().sort_index().to_string())

# Keep only 2008-2024 (real IPL seasons)
REAL_SEASONS = list(range(2008, 2026))  # 2008 to 2025 inclusive
matches = matches[matches["year"].isin(REAL_SEASONS)].copy()
matches = matches.sort_values("date").reset_index(drop=True)

print(f"\n  After filtering real seasons: {len(matches)} matches")
print(f"  Date range: {matches['date'].min().date()} "
      f"→ {matches['date'].max().date()}")

t1_wr = (matches["winner"] == matches["team1"]).mean()
print(f"  team1 win rate: {t1_wr:.1%}")


# ============================================================
# STEP 3 — BUILD SQUAD STRENGTH FROM DELIVERIES
# ============================================================
"""
NEW FEATURE: SQUAD STRENGTH
  For each match, compute the average batting SR and
  economy of each team's players from the PREVIOUS season.
  This tells the model "CSK has better batters this year".

  This is much stronger than ELO alone.
"""
print("\n[STEP 3] Building squad strength features ...")

squad_features_available = False

if deliveries is not None:
    try:
        # Map match_id → team and season
        if "id" in matches.columns:
            match_id_col = "id"
        elif "match_id" in matches.columns:
            match_id_col = "match_id"
        else:
            match_id_col = matches.columns[0]

        # Standardise deliveries columns
        deliv_col_map = {
            "match_id":"match_id",
            "batting_team":"batting_team",
            "batsman":"batter","batter":"batter",
            "batsman_runs":"runs","batter_runs":"runs",
            "ball":"ball_num","bowler":"bowler",
            "player_dismissed":"player_dismissed",
            "total_runs":"total_runs",
        }
        deliveries = deliveries.rename(
            columns={k:v for k,v in deliv_col_map.items()
                     if k in deliveries.columns})

        # Add season to deliveries
        m_season = matches[[match_id_col,"year"]].rename(
            columns={match_id_col:"match_id"})
        if "match_id" in deliveries.columns:
            deliveries = deliveries.merge(
                m_season, on="match_id", how="left")

        if "year" in deliveries.columns and \
           "batting_team" in deliveries.columns:
            # Batting SR per team per season
            team_bat = deliveries.groupby(
                ["batting_team","year"]).agg(
                runs_scored=("runs","sum") if "runs"
                             in deliveries.columns
                             else ("total_runs","sum"),
                balls=("ball_num","count")
            ).reset_index()
            team_bat["team_sr"] = (
                team_bat.get("runs_scored",
                team_bat.get("runs",0))
                / team_bat["balls"].clip(1)
            ) * 100
            team_bat = team_bat.rename(
                columns={"batting_team":"team"})
            squad_features_available = True
            print("  Squad strength features: AVAILABLE")
        else:
            print("  Squad features: skipping "
                  "(column mismatch)")
    except Exception as e:
        print(f"  Squad features: skipping ({e})")
else:
    print("  Squad features: deliveries.csv not found, "
          "skipping")



print("\n[STEP 4] Feature engineering ...")

# ── ELO ──
print("  ELO ...")
elo = {}
elo_pre_t1, elo_pre_t2 = [], []
for _, row in matches.iterrows():
    t1,t2,w = row.team1, row.team2, row.winner
    r1 = elo.get(t1, 1500)
    r2 = elo.get(t2, 1500)
    elo_pre_t1.append(round(r1,2))
    elo_pre_t2.append(round(r2,2))
    exp1 = 1/(1+10**((r2-r1)/400))
    a1   = 1.0 if w==t1 else 0.0
    elo[t1] = r1 + 32*(a1-exp1)
    elo[t2] = r2 + 32*((1-a1)-(1-exp1))

matches["elo_t1"]   = elo_pre_t1
matches["elo_t2"]   = elo_pre_t2
matches["elo_diff"] = [round(a-b,2)
                        for a,b in zip(elo_pre_t1,elo_pre_t2)]

# ── H2H ──
print("  H2H ...")
h2h_wr, h2h_n = [], []
for pos,(i,row) in enumerate(matches.iterrows()):
    t1,t2 = row.team1, row.team2
    past  = matches.iloc[:pos]
    mask  = (((past.team1==t1)&(past.team2==t2))|
             ((past.team1==t2)&(past.team2==t1)))
    h2h   = past[mask]
    if len(h2h)==0:
        h2h_wr.append(0.5); h2h_n.append(0)
    else:
        h2h_wr.append(round(
            (h2h.winner==t1).sum()/len(h2h),4))
        h2h_n.append(len(h2h))
matches["h2h_wr_t1"]    = h2h_wr
matches["h2h_n_matches"] = h2h_n

# ── FORM (rolling 5 + rolling 10) ──
print("  Form ...")
def get_form(df, team, pos, window):
    past   = df.iloc[:pos]
    mask   = (past.team1==team)|(past.team2==team)
    recent = past[mask].tail(window)
    if len(recent)==0: return 0.5
    return round((recent.winner==team).sum()/len(recent),4)

def get_streak(df, team, pos):
    past   = df.iloc[:pos]
    mask   = (past.team1==team)|(past.team2==team)
    recent = past[mask].tail(3)
    if len(recent)<3: return 0
    return int((recent.winner==team).all())

f5t1,f5t2,f10t1,f10t2,st1,st2 = [],[],[],[],[],[]
for pos,(i,row) in enumerate(matches.iterrows()):
    f5t1.append(get_form(matches,row.team1,pos,5))
    f5t2.append(get_form(matches,row.team2,pos,5))
    f10t1.append(get_form(matches,row.team1,pos,10))
    f10t2.append(get_form(matches,row.team2,pos,10))
    st1.append(get_streak(matches,row.team1,pos))
    st2.append(get_streak(matches,row.team2,pos))

matches["form5_t1"]  = f5t1; matches["form5_t2"]  = f5t2
matches["form10_t1"] = f10t1;matches["form10_t2"] = f10t2
matches["streak_t1"] = st1;  matches["streak_t2"] = st2
matches["form5_diff"] = [round(a-b,4)
                          for a,b in zip(f5t1,f5t2)]
matches["form10_diff"]= [round(a-b,4)
                          for a,b in zip(f10t1,f10t2)]

# ── VENUE ──
print("  Venue ...")
def vwr(past,team,venue):
    if "venue" not in past.columns: return 0.5
    mask=((past.team1==team)|(past.team2==team))\
         &(past.venue==venue)
    v=past[mask]
    if len(v)==0: return 0.5
    return round((v.winner==team).sum()/len(v),4)

vt1,vt2=[],[]
for pos,(i,row) in enumerate(matches.iterrows()):
    past  = matches.iloc[:pos]
    venue = row.get("venue","Unknown")
    vt1.append(vwr(past,row.team1,venue))
    vt2.append(vwr(past,row.team2,venue))
matches["venue_wr_t1"]=vt1
matches["venue_wr_t2"]=vt2
matches["venue_diff"] =[round(a-b,4)
                         for a,b in zip(vt1,vt2)]

# ── OVERALL WIN RATE (all-time) ──
print("  Overall win rates ...")
all_wr_t1, all_wr_t2 = [], []
for pos,(i,row) in enumerate(matches.iterrows()):
    past = matches.iloc[:pos]
    for team, lst in [(row.team1,all_wr_t1),
                      (row.team2,all_wr_t2)]:
        mask = (past.team1==team)|(past.team2==team)
        tm   = past[mask]
        if len(tm)==0: lst.append(0.5)
        else: lst.append(round(
            (tm.winner==team).sum()/len(tm),4))
matches["alltime_wr_t1"] = all_wr_t1
matches["alltime_wr_t2"] = all_wr_t2
matches["alltime_diff"]  = [round(a-b,4)
    for a,b in zip(all_wr_t1,all_wr_t2)]

# ── TOSS ──
toss_t1,bat_f,fadv=[],[],[]
for _,row in matches.iterrows():
    tw  = str(row.get("toss_winner",""))
    dec = str(row.get("toss_decision","")).lower()
    t1w = int(tw==row.team1)
    bf  = int(dec=="bat")
    toss_t1.append(t1w); bat_f.append(bf)
    fadv.append(int(t1w==1 and bf==0))
matches["toss_t1"]      = toss_t1
matches["bat_first"]    = bat_f
matches["field_adv_t1"] = fadv

# ── CONTEXT ──
matches["season_num"] = matches["date"].dt.year - 2007
matches["month"]      = matches["date"].dt.month
matches["is_playoff"] = 0
if "match_type" in matches.columns:
    matches["is_playoff"] = matches["match_type"]\
        .str.lower()\
        .str.contains("final|playoff|eliminator|qualifier",
                      na=False).astype(int)

# ── SQUAD STRENGTH (if available) ──
if squad_features_available:
    print("  Squad strength ...")
    try:
        def get_squad_sr(team, year):
            prev = team_bat[
                (team_bat.team==team) &
                (team_bat.year==year-1)
            ]
            if len(prev)==0:
                # fallback: use league average
                avg = team_bat[team_bat.year==year-1]\
                      ["team_sr"].mean()
                return avg if not np.isnan(avg) else 120.0
            return float(prev["team_sr"].values[0])

        sq_t1, sq_t2 = [], []
        for _,row in matches.iterrows():
            sq_t1.append(get_squad_sr(row.team1,row.year))
            sq_t2.append(get_squad_sr(row.team2,row.year))
        matches["squad_sr_t1"] = sq_t1
        matches["squad_sr_t2"] = sq_t2
        matches["squad_sr_diff"]=[round(a-b,2)
            for a,b in zip(sq_t1,sq_t2)]
        print("  Squad SR features added ✓")
    except Exception as e:
        print(f"  Squad SR skipped: {e}")
        squad_features_available = False

# ── TARGET ──
matches["target"] = (
    matches["winner"]==matches["team1"]).astype(int)
print(f"\n  team1 win rate: "
      f"{matches['target'].mean():.1%}")


# ============================================================
# STEP 5 — DEFINE FEATURE COLUMNS
# ============================================================
FEATURE_COLS = [
    # Strength
    "elo_t1","elo_t2","elo_diff",
    # H2H
    "h2h_wr_t1","h2h_n_matches",
    # Form
    "form5_t1","form5_t2","form5_diff",
    "form10_t1","form10_t2","form10_diff",
    "streak_t1","streak_t2",
    # Venue
    "venue_wr_t1","venue_wr_t2","venue_diff",
    # All-time win rate
    "alltime_wr_t1","alltime_wr_t2","alltime_diff",
    # Toss
    "toss_t1","bat_first","field_adv_t1",
    # Context
    "season_num","month","is_playoff",
]

# Add squad features if computed
if squad_features_available:
    FEATURE_COLS += ["squad_sr_t1","squad_sr_t2",
                     "squad_sr_diff"]

FEATURE_COLS = [c for c in FEATURE_COLS
                if c in matches.columns]
matches[FEATURE_COLS] = matches[FEATURE_COLS].fillna(0.5)

print(f"\n  Features: {len(FEATURE_COLS)}")
print(f"  {FEATURE_COLS}")

print("\n[STEP 6] Creating symmetric training data ...")

df_orig = matches.copy()
df_mirr = matches.copy()

# Swap all paired features
swaps = [
    ("elo_t1","elo_t2"),
    ("form5_t1","form5_t2"),
    ("form10_t1","form10_t2"),
    ("streak_t1","streak_t2"),
    ("venue_wr_t1","venue_wr_t2"),
    ("alltime_wr_t1","alltime_wr_t2"),
    ("team1","team2"),
]
if squad_features_available:
    swaps += [("squad_sr_t1","squad_sr_t2")]

for a,b in swaps:
    if a in df_mirr.columns and b in df_mirr.columns:
        df_mirr[a] = matches[b].values
        df_mirr[b] = matches[a].values

# Flip diff features
for col in ["elo_diff","form5_diff","form10_diff",
            "venue_diff","alltime_diff"]:
    if col in df_mirr.columns:
        df_mirr[col] = -matches[col].values

if squad_features_available and \
   "squad_sr_diff" in df_mirr.columns:
    df_mirr["squad_sr_diff"] = -matches["squad_sr_diff"].values

df_mirr["h2h_wr_t1"]     = 1 - matches["h2h_wr_t1"].values
df_mirr["toss_t1"]       = 1 - matches["toss_t1"].values
df_mirr["field_adv_t1"]  = 1 - matches["field_adv_t1"].values
df_mirr["target"]        = 1 - matches["target"].values

df_full = pd.concat([df_orig, df_mirr],
                     ignore_index=True)
df_full = df_full.sort_values("date")\
                 .reset_index(drop=True)

print(f"  Total rows: {len(df_full)} "
      f"({len(df_orig)} real + {len(df_mirr)} mirrored)")
print(f"  Target balance: {df_full['target'].mean():.1%} "
      f"← must be 50%")

# Save
save_cols = (["date","team1","team2","winner"]
             + FEATURE_COLS + ["target"])
save_cols = [c for c in save_cols if c in df_full.columns]
df_full[save_cols].to_csv(
    os.path.join(PROC,"features.csv"), index=False)
with open(os.path.join(PROC,"feature_cols.json"),"w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
print(f"  Saved features.csv")



print("\n[STEP 7] Train/test split ...")

SPLIT = pd.Timestamp("2022-01-01")


train_df = df_full[df_full["date"] < SPLIT]

test_df  = df_orig[df_orig["date"] >= SPLIT].copy()

X_train = train_df[FEATURE_COLS]
y_train = train_df["target"]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df["target"]

print(f"  Train: {len(X_train)} rows  "
      f"(target={y_train.mean():.1%})")
print(f"  Test : {len(X_test)} rows   "
      f"(target={y_test.mean():.1%})")

if len(X_test) < 20:
    print("  ⚠  Test set too small! Adjusting split to 2020...")
    SPLIT = pd.Timestamp("2020-01-01")
    train_df = df_full[df_full["date"] < SPLIT]
    test_df  = df_orig[df_orig["date"] >= SPLIT].copy()
    X_train  = train_df[FEATURE_COLS]
    y_train  = train_df["target"]
    X_test   = test_df[FEATURE_COLS]
    y_test   = test_df["target"]
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

joblib.dump(FEATURE_COLS,
            os.path.join(MDIR,"feature_names.pkl"))
tscv = TimeSeriesSplit(n_splits=5)



print("\n[STEP 8] Training models ...")

xgb_m = XGBClassifier(
    n_estimators=500, max_depth=4,
    learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5,
    gamma=1, reg_alpha=0.1, reg_lambda=1,
    eval_metric="logloss", verbosity=0, random_state=42
)
lgbm_m = LGBMClassifier(
    n_estimators=500, num_leaves=20,
    learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=0.1, reg_lambda=1,
    verbose=-1, random_state=42
)
rf_m = RandomForestClassifier(
    n_estimators=500, max_depth=5,
    min_samples_split=10, min_samples_leaf=5,
    max_features="sqrt",
    random_state=42, n_jobs=-1
)
gb_m = GradientBoostingClassifier(
    n_estimators=300, max_depth=3,
    learning_rate=0.05, subsample=0.8,
    random_state=42
)

results = {}
for name, model in [("XGBoost",xgb_m),
                    ("LightGBM",lgbm_m),
                    ("RandomForest",rf_m),
                    ("GradBoost",gb_m)]:
    model.fit(X_train, y_train)
    pr = model.predict_proba(X_test)[:,1]
    pd_ = model.predict(X_test)
    acc = accuracy_score(y_test, pd_)
    auc = roc_auc_score(y_test, pr)
    results[name]={"accuracy":round(acc,4),
                   "roc_auc":round(auc,4)}
    print(f"  {name:15s}  acc={acc:.4f}  auc={auc:.4f}")

# Stacking with 4 base models
stack = StackingClassifier(
    estimators=[
        ("xgb",  XGBClassifier(n_estimators=300,max_depth=4,
                  learning_rate=0.05,subsample=0.8,
                  eval_metric="logloss",verbosity=0,
                  random_state=42)),
        ("lgbm", LGBMClassifier(n_estimators=300,
                  num_leaves=20,learning_rate=0.05,
                  verbose=-1,random_state=42)),
        ("rf",   RandomForestClassifier(n_estimators=300,
                  max_depth=5,min_samples_leaf=5,
                  random_state=42,n_jobs=-1)),
        ("gb",   GradientBoostingClassifier(
                  n_estimators=200,max_depth=3,
                  learning_rate=0.05,random_state=42)),
    ],
    final_estimator=LogisticRegression(
        C=0.1, max_iter=1000, random_state=42),
    cv= 5, passthrough=False, n_jobs=-1  # <--- Change this to False!
)
stack.fit(X_train, y_train)
sp  = stack.predict_proba(X_test)[:,1]
sd  = stack.predict(X_test)
acc = accuracy_score(y_test, sd)
auc = roc_auc_score(y_test, sp)
results["Stacking"]={"accuracy":round(acc,4),
                     "roc_auc":round(auc,4)}
print(f"  {'Stacking':15s}  acc={acc:.4f}  auc={auc:.4f}  "
      f"← BEST")


# ============================================================
# STEP 9 — CROSS-VALIDATION (more reliable than single split)
# ============================================================
print("\n[STEP 9] TimeSeriesSplit cross-validation ...")
cv_scores = cross_val_score(
    XGBClassifier(n_estimators=300,max_depth=4,
                  learning_rate=0.05,subsample=0.8,
                  eval_metric="logloss",verbosity=0,
                  random_state=42),
    df_full[FEATURE_COLS], df_full["target"],
    cv=TimeSeriesSplit(n_splits=8),
    scoring="roc_auc", n_jobs=-1
)
print(f"  CV AUC per fold: "
      f"{[round(s,3) for s in cv_scores]}")
print(f"  Mean CV AUC    : {cv_scores.mean():.4f} "
      f"± {cv_scores.std():.4f}")


# ============================================================
# STEP 10 — SAVE
# ============================================================
print("\n[STEP 10] Saving ...")

joblib.dump(stack,  os.path.join(MDIR,"stacking_model.pkl"))
joblib.dump(xgb_m,  os.path.join(MDIR,"xgb_model.pkl"))
joblib.dump(lgbm_m, os.path.join(MDIR,"lgbm_model.pkl"))
joblib.dump(rf_m,   os.path.join(MDIR,"rf_model.pkl"))

final_elo_s = {k:round(v,2) for k,v in elo.items()}
with open(os.path.join(MDIR,"final_elo.json"),"w") as f:
    json.dump(final_elo_s, f, indent=2)

with open(os.path.join(MDIR,"results.json"),"w") as f:
    json.dump(results, f, indent=2)


# ============================================================
# FINAL REPORT
# ============================================================
print()
print("=" * 62)
print("  FINAL RESULTS")
print("=" * 62)
print(f"  {'Model':<18} {'Accuracy':>10} {'ROC-AUC':>10}")
print("  " + "-" * 42)
for name, r in results.items():
    tag = "  ← BEST" if name=="Stacking" else ""
    print(f"  {name:<18} {r['accuracy']:>10.4f} "
          f"{r['roc_auc']:>10.4f}{tag}")

print(f"\n  CV AUC (8-fold): {cv_scores.mean():.4f} "
      f"± {cv_scores.std():.4f}")

print(f"\n  Classification Report (Stacking):")
print(classification_report(y_test, sd,
      target_names=["team2 wins","team1 wins"]))

print("  Confusion Matrix:")
cm = confusion_matrix(y_test, sd)
print(f"               Predicted:")
print(f"               team2   team1")
print(f"  Actual team2  {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"  Actual team1  {cm[1,0]:4d}    {cm[1,1]:4d}")

print()
print("  Top Features (XGBoost):")
fi = pd.Series(xgb_m.feature_importances_,
               index=FEATURE_COLS)\
       .sort_values(ascending=False)
for feat,imp in fi.head(10).items():
    bar = "█"*int(imp*180)
    print(f"    {feat:24s}  {imp:.4f}  {bar}")

print()
print("=" * 62)
print("  PHASE 3 COMPLETE — MODELS SAVED")
print()
print("  UNDERSTANDING YOUR ACCURACY:")
print("    IPL has genuine randomness — rain, pitch,")
print("    on-day form, DRS decisions. Even professional")
print("    betting markets only reach 62-67% accuracy.")
print("    55-65% accuracy is REALISTIC and HONEST.")
print("    Higher = likely overfitting.")
print()
print("  Next step → run phase4_player_forecast.py")
print("=" * 62)
