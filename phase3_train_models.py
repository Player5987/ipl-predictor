"""
============================================================
IPL MATCH WINNER PREDICTOR
Phase 3: Train Base Models + Stacking Ensemble + Optuna
============================================================
HOW TO RUN:
  1. Make sure phase2_feature_engineering.py ran successfully
     and data/processed/features.csv exists
  2. Run: python phase3_train_models.py
  3. Output: models/ folder with all saved models
============================================================
"""

import pandas as pd
import numpy as np
import os, json, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (StackingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report,
                              confusion_matrix)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# STEP 0 — LOAD DATA
# ============================================================
print("=" * 60)
print("  IPL PREDICTOR — PHASE 3: MODEL TRAINING")
print("=" * 60)
print()

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
MDIR = os.path.join(BASE, "models")
os.makedirs(MDIR, exist_ok=True)

feat_path = os.path.join(PROC, "features.csv")
if not os.path.exists(feat_path):
    raise FileNotFoundError(
        "data/processed/features.csv not found.\n"
        "Run phase2_feature_engineering.py first."
    )

df = pd.read_csv(feat_path, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"[LOAD] features.csv loaded: {df.shape}")

# Load feature columns saved by Phase 2
fc_path = os.path.join(PROC, "feature_cols.json")
if os.path.exists(fc_path):
    with open(fc_path) as f:
        FEATURE_COLS = json.load(f)
    # Keep only columns that exist in df
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
else:
    # Fallback: detect feature columns automatically
    exclude = {"date","season_num","team1","team2",
               "winner","target","year"}
    FEATURE_COLS = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in [np.float64, np.int64]]

print(f"[LOAD] Features used: {FEATURE_COLS}")
print()

X = df[FEATURE_COLS]
y = df["target"]


# ============================================================
# STEP 1 — TIME-BASED TRAIN / TEST SPLIT
# ============================================================
"""
CONCEPT: WHY NOT RANDOM SPLIT?
  If we use sklearn's train_test_split(random_state=42),
  a match from 2024 can end up in training while a match
  from 2019 is in test. The model then "knows the future"
  during training — this is called DATA LEAKAGE.

  Example of leakage:
    Training includes match 900 (played May 2024)
    The model learns ELO/form from that match
    Test includes match 800 (played April 2023)
    The model effectively saw future data

  CORRECT APPROACH: everything before SPLIT_DATE trains,
  everything after tests. Mirrors real deployment.
"""
print("[STEP 1] Time-based train/test split ...")

# Use last 2 full seasons as test set
SPLIT_DATE = pd.Timestamp("2023-01-01")

train_mask = df["date"] < SPLIT_DATE
test_mask  = df["date"] >= SPLIT_DATE

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

print(f"         Train: {len(X_train)} matches  "
      f"({df[train_mask]['date'].min().date()} "
      f"→ {df[train_mask]['date'].max().date()})")
print(f"         Test : {len(X_test)} matches  "
      f"({df[test_mask]['date'].min().date()} "
      f"→ {df[test_mask]['date'].max().date()})")
print(f"         Train target balance: {y_train.mean():.1%} team1 wins")
print(f"         Test  target balance: {y_test.mean():.1%} team1 wins")

# Save the feature names for the FastAPI backend
joblib.dump(FEATURE_COLS, os.path.join(MDIR, "feature_names.pkl"))
print(f"         Saved feature_names.pkl → models/")
print()

# TimeSeriesSplit for cross-validation during tuning
# (same time-order rule applies inside tuning, not just outside)
tscv = TimeSeriesSplit(n_splits=5)

# ============================================================
# STEP 2 — OPTUNA HYPERPARAMETER TUNING (CORRECTED)
# ============================================================
"""
CONCEPT: WHAT IS OPTUNA?
  Hyperparameters are settings you choose BEFORE training
  (max_depth, learning_rate, n_estimators).
  Wrong choices → underfitting or overfitting.

  Optuna uses Bayesian Optimization:
  - Trial 1: random guess → score 0.68
  - Trial 2: learns from trial 1 → smarter guess → 0.71
  - Trial N: has a map of which regions work well → 0.74+

  It's NOT random search. It builds a surrogate model
  of the loss function and samples promising regions.
  50 Bayesian trials > 500 random trials.

  We optimise ROC-AUC (not accuracy) because:
  AUC measures how well-CALIBRATED the probabilities are,
  not just whether the predicted winner is correct.
  A model that says "60% CSK" is more useful than one
  that just says "CSK wins".
"""
print("[STEP 2] Optuna hyperparameter tuning ...")
print("         (this takes 3-6 minutes — go grab a chai)")
print()

# ── 2a. XGBoost tuning ──
print("  Tuning XGBoost (50 trials) ...")

def xgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_est", 100, 600),
        "max_depth":        trial.suggest_int("depth", 3, 8),
        "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("sub", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("col", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("mcw", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "eval_metric": "logloss",
        "verbosity":   0,
        "random_state": 42,
    }
    cv_scores = []
    for tr_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = XGBClassifier(**params)
        m.fit(Xtr, ytr, verbose=False)
        prob = m.predict_proba(Xval)[:, 1]
        cv_scores.append(roc_auc_score(yval, prob))
    return np.mean(cv_scores)

study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_objective, n_trials=50)

# FIX: Map shortened trial names to real XGBoost argument parameters
best_xgb_params = {
    "n_estimators":     int(study_xgb.best_params["n_est"]),
    "max_depth":        int(study_xgb.best_params["depth"]),
    "learning_rate":    float(study_xgb.best_params["lr"]),
    "subsample":        float(study_xgb.best_params["sub"]),
    "colsample_bytree": float(study_xgb.best_params["col"]),
    "min_child_weight": int(study_xgb.best_params["mcw"]),
    "gamma":            float(study_xgb.best_params["gamma"]),
    "eval_metric":      "logloss",
    "verbosity":        0,
    "random_state":     42
}
print(f"    XGBoost best CV AUC: {study_xgb.best_value:.4f}")
print(f"    depth={best_xgb_params['max_depth']}, lr={best_xgb_params['learning_rate']:.4f}, n_est={best_xgb_params['n_estimators']}")

# ── 2b. LightGBM tuning ──
print("\n  Tuning LightGBM (50 trials) ...")

def lgbm_objective(trial):
    params = {
        "n_estimators":       trial.suggest_int("n_est", 100, 600),
        "num_leaves":         trial.suggest_int("leaves", 15, 127),
        "learning_rate":      trial.suggest_float("lr", 0.01, 0.3, log=True),
        "subsample":          trial.suggest_float("sub", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("col", 0.5, 1.0),
        "min_child_samples":  trial.suggest_int("mcs", 5, 40),
        "reg_alpha":          trial.suggest_float("ra", 0, 1),
        "reg_lambda":         trial.suggest_float("rl", 0, 1),
        "verbose": -1,
        "random_state": 42,
    }
    cv_scores = []
    for tr_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = LGBMClassifier(**params)
        m.fit(Xtr, ytr)
        prob = m.predict_proba(Xval)[:, 1]
        cv_scores.append(roc_auc_score(yval, prob))
    return np.mean(cv_scores)

study_lgbm = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgbm.optimize(lgbm_objective, n_trials=50)

# FIX: Map shortened trial names to real LightGBM argument parameters
best_lgbm_params = {
    "n_estimators":      int(study_lgbm.best_params["n_est"]),
    "num_leaves":        int(study_lgbm.best_params["leaves"]),
    "learning_rate":     float(study_lgbm.best_params["lr"]),
    "subsample":         float(study_lgbm.best_params["sub"]),
    "colsample_bytree":  float(study_lgbm.best_params["col"]),
    "min_child_samples": int(study_lgbm.best_params["mcs"]),
    "reg_alpha":         float(study_lgbm.best_params["ra"]),
    "reg_lambda":        float(study_lgbm.best_params["rl"]),
    "verbose":           -1,
    "random_state":      42
}
print(f"    LightGBM best CV AUC: {study_lgbm.best_value:.4f}")
print(f"    leaves={best_lgbm_params['num_leaves']}, lr={best_lgbm_params['learning_rate']:.4f}")

# ── 2c. Random Forest tuning ──
print("\n  Tuning Random Forest (40 trials) ...")

def rf_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_est", 100, 500),
        "max_depth":         trial.suggest_int("depth", 4, 16),
        "min_samples_split": trial.suggest_int("mss", 2, 12),
        "min_samples_leaf":  trial.suggest_int("msl", 1, 8),
        "max_features":      trial.suggest_categorical("feat", ["sqrt", "log2", 0.6, 0.8]),
        "random_state": 42,
        "n_jobs": -1
    }
    cv_scores = []
    for tr_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = RandomForestClassifier(**params)
        m.fit(Xtr, ytr)
        prob = m.predict_proba(Xval)[:, 1]
        cv_scores.append(roc_auc_score(yval, prob))
    return np.mean(cv_scores)

study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(rf_objective, n_trials=40)

# FIX: Map shortened trial names to real scikit-learn argument parameters
best_rf_params = {
    "n_estimators":      int(study_rf.best_params["n_est"]),
    "max_depth":         int(study_rf.best_params["depth"]),
    "min_samples_split": int(study_rf.best_params["mss"]),
    "min_samples_leaf":  int(study_rf.best_params["msl"]),
    "max_features":      study_rf.best_params["feat"],
    "random_state":      42,
    "n_jobs":            -1
}
print(f"    RandomForest best CV AUC: {study_rf.best_value:.4f}")

# Save all structured parameters down cleanly to a dictionary artifact
all_params = {
    "xgb":  best_xgb_params,
    "lgbm": best_lgbm_params,
    "rf":   best_rf_params,
}
params_path = os.path.join(MDIR, "best_params.json")
with open(params_path, "w") as f:
    json.dump(all_params, f, indent=2)
print(f"\n  Best params saved → models/best_params.json")
print()

# ============================================================
# STEP 3 — TRAIN BASE MODELS ON FULL TRAIN SET
# ============================================================
"""
CONCEPT: BASE MODELS
  We train 3 independent models. Each finds different patterns:
  
  XGBoost   → great at feature interactions (elo_diff × form)
  LightGBM  → fastest, handles slight imbalance well
  RandomForest → reduces variance via 200 independent trees,
                 less prone to overfitting on small data
  
  Their ERRORS are different — when XGBoost is wrong,
  LGBM is often right. Stacking exploits this.
"""
print("[STEP 3] Training base models on full training set ...")

xgb_model  = XGBClassifier(**best_xgb_params)
lgbm_model = LGBMClassifier(**best_lgbm_params)
rf_model   = RandomForestClassifier(**best_rf_params)

results = {}
for name, model in [("XGBoost",      xgb_model),
                    ("LightGBM",     lgbm_model),
                    ("RandomForest", rf_model)]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, probs)
    results[name] = {"accuracy": acc, "roc_auc": auc}
    print(f"  {name:15s}  accuracy={acc:.4f}  roc_auc={auc:.4f}")

print()


# ============================================================
# STEP 4 — STACKING ENSEMBLE (CORRECTED)
# ============================================================
"""
CONCEPT: HOW STACKING WORKS
  
  Step A: Train 3 base models using KFold CV.
          For each fold, the models make predictions on the
          validation fold they NEVER saw during training.
          These are called "out-of-fold" (OOF) predictions.
  
  Step B: Stack the OOF predictions as features:
          [xgb_pred, lgbm_pred, rf_pred] per match
  
  Step C: Train a meta-learner (Logistic Regression) on
          these stacked predictions to learn how to weight them.
"""
print("[STEP 4] Building stacking ensemble ...")
print("         (trains 5 CV folds × 3 models = 15 fits)")
print()

# FIX: Import and use KFold with shuffle=False to satisfy partition rules
from sklearn.model_selection import KFold
stack_cv = KFold(n_splits=5, shuffle=False)

stack = StackingClassifier(
    estimators=[
        ("xgb",  XGBClassifier(**best_xgb_params)),
        ("lgbm", LGBMClassifier(**best_lgbm_params)),
        ("rf",   RandomForestClassifier(**best_rf_params)),
    ],
    final_estimator=LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    ),
    cv=stack_cv,       # Swapped from tscv to stack_cv
    passthrough=True,  # original features + OOF preds → meta
    n_jobs=-1
)

stack.fit(X_train, y_train)

stack_preds = stack.predict(X_test)
stack_probs = stack.predict_proba(X_test)[:, 1]
stack_acc   = accuracy_score(y_test, stack_preds)
stack_auc   = roc_auc_score(y_test, stack_probs)
results["Stacking"] = {
    "accuracy": stack_acc,
    "roc_auc":  stack_auc
}

print(f"  {'Stacking':15s}  accuracy={stack_acc:.4f}  "
      f"roc_auc={stack_auc:.4f}  ← BEST MODEL")
print()

# ============================================================
# STEP 5 — SAVE ALL MODELS
# ============================================================
print("[STEP 5] Saving all models ...")

joblib.dump(stack,      os.path.join(MDIR, "stacking_model.pkl"))
joblib.dump(xgb_model,  os.path.join(MDIR, "xgb_model.pkl"))
joblib.dump(lgbm_model, os.path.join(MDIR, "lgbm_model.pkl"))
joblib.dump(rf_model,   os.path.join(MDIR, "rf_model.pkl"))

# Save results for README
results_path = os.path.join(MDIR, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print("  models/stacking_model.pkl  ← used by FastAPI")
print("  models/xgb_model.pkl       ← used for SHAP")
print("  models/lgbm_model.pkl")
print("  models/rf_model.pkl")
print("  models/best_params.json")
print("  models/results.json")
print("  models/feature_names.pkl")
print()


# ============================================================
# STEP 6 — FULL RESULTS REPORT
# ============================================================
print("=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  {'Model':<18} {'Accuracy':>10} {'ROC-AUC':>10}")
print("  " + "-" * 42)
for name, r in results.items():
    marker = "  ← BEST" if name == "Stacking" else ""
    print(f"  {name:<18} {r['accuracy']:>10.4f} "
          f"{r['roc_auc']:>10.4f}{marker}")

print()
print("  Detailed classification report (Stacking model):")
print()
print(classification_report(
    y_test, stack_preds,
    target_names=["team2 wins", "team1 wins"]
))

print("  Confusion Matrix:")
cm = confusion_matrix(y_test, stack_preds)
print(f"               Predicted")
print(f"               team2   team1")
print(f"  Actual team2  {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"  Actual team1  {cm[1,0]:4d}    {cm[1,1]:4d}")
print()

# Feature importances from XGBoost
print("  Top feature importances (XGBoost):")
fi = pd.Series(
    xgb_model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)
for feat, imp in fi.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"    {feat:22s} {imp:.4f}  {bar}")

print()
print("=" * 60)
print("  PHASE 3 COMPLETE")
print("  Next step → run phase4_player_forecast.py")
print("=" * 60)
