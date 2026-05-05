"""
train.py
Orchestrates complete two-stage model training.
Run once. After completion, models are frozen.
Real data testing (Steps 6-20) does not retrain these models.
"""

import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import json
from datetime import datetime

from stage1_species import Stage1SpeciesClassifier, FEAT_COLS
from stage2_stress import Stage2StressBank

cfg     = yaml.safe_load(open("config/config.yaml"))
SEED    = cfg["project"]["seed"]
np.random.seed(SEED)
SPECIES = cfg["project"]["species"]
MODEL_DIR = Path(cfg["paths"]["models"])
REPORT_DIR = Path(cfg["paths"]["reports"])
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "master_features_scaled.csv")
    return df

def train_stage1(df: pd.DataFrame) -> tuple:
    """Train Stage 1 species classifier with 5-fold cross-validation."""
    print("\n=== STAGE 1: Species Classification ===")
    X = df[FEAT_COLS].values
    y = df["species_label"].values

    # Log class distribution (imbalance handled via class_weight='balanced')
    counts = Counter(y)
    print(f"Class distribution: {counts}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["training"]["test_size"],
        stratify=y, random_state=SEED
    )

    # 5-fold cross-validation on training set
    clf = Stage1SpeciesClassifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        clf_fold = Stage1SpeciesClassifier()
        clf_fold.fit(X_train[tr_idx], y_train[tr_idx])
        pred = clf_fold.model.predict(X_train[val_idx])
        acc  = accuracy_score(y_train[val_idx], pred)
        cv_scores.append(acc)
        print(f"  Fold {fold+1}/5 accuracy: {acc:.4f}")
    print(f"CV mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    # Final training on full training set
    clf.fit(X_train, y_train)
    y_pred = clf.model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Stage 1 test accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=SPECIES))

    clf.save(MODEL_DIR / "stage1_rf.pkl")
    return clf, test_acc, cv_scores, X_test, y_test

def train_stage2(df: pd.DataFrame) -> tuple:
    """Train Stage 2 species-conditioned SVM stress classifiers."""
    print("\n=== STAGE 2: Species-Conditioned Stress Classification ===")
    bank = Stage2StressBank()
    results = {}

    for sp in SPECIES:
        print(f"\n  Training Stage 2 SVM for: {sp}")
        sp_df = df[df["species"] == sp]
        X = sp_df[FEAT_COLS].values
        y = sp_df["stress_label"].values

        # Log class distribution (imbalance handled via class_weight='balanced')
        counts = Counter(y)
        print(f"  {sp} class distribution: {counts}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg["training"]["test_size"],
            stratify=y, random_state=SEED
        )
        bank.fit_species(sp, X_train, y_train)
        y_pred, _ = bank.predict(sp, X_test)
        acc = accuracy_score(y_test, y_pred)
        stresses = cfg["project"]["stress_states"]
        print(classification_report(y_test, y_pred, target_names=stresses))
        results[sp] = {"accuracy": round(acc, 4)}

    bank.save_all(MODEL_DIR)
    return bank, results

def save_training_metadata(s1_acc, s1_cv, s2_results):
    meta = {
        "trained_at": datetime.now().isoformat(),
        "stage1": {
            "test_accuracy": round(s1_acc, 4),
            "cv_mean": round(np.mean(s1_cv), 4),
            "cv_std":  round(np.std(s1_cv),  4),
        },
        "stage2": s2_results,
        "models_frozen": True,
        "real_data_compatible": True
    }
    out = REPORT_DIR / "training_metadata.json"
    json.dump(meta, open(out, "w"), indent=2)
    print(f"\nTraining metadata saved: {out}")
    return meta

if __name__ == "__main__":
    df = load_features()
    print(f"Dataset loaded: {df.shape} | Classes: {df['species'].unique()}")

    clf,  s1_acc, s1_cv, *_ = train_stage1(df)
    bank, s2_results         = train_stage2(df)
    meta = save_training_metadata(s1_acc, s1_cv, s2_results)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Stage 1 Test Accuracy: {meta['stage1']['test_accuracy']:.0%}")
    for sp, r in s2_results.items():
        print(f"Stage 2 [{sp}] Accuracy: {r['accuracy']:.0%}")
    print("All models serialised to models/")
    print("Models are now FROZEN. Do not retrain for real data testing.")
