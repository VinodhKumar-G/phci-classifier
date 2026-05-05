"""
evaluate.py
Comprehensive validation of the two-stage pipeline.
Generates confusion matrices, per-class metrics, and confidence distributions.
Must be run and all criteria passed before Step 6 (hardware) begins.
"""

import numpy as np
import pandas as pd
import yaml, json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
from pipeline import PHCIPipeline
from stage1_species import FEAT_COLS

cfg      = yaml.safe_load(open("config/config.yaml"))
SPECIES  = cfg["project"]["species"]
STRESSES = cfg["project"]["stress_states"]
FIG_DIR  = Path(cfg["paths"]["figures"])
REP_DIR  = Path(cfg["paths"]["reports"])
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_test_set():
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "master_features_scaled.csv")
    from sklearn.model_selection import train_test_split
    SEED = cfg["project"]["seed"]
    _, df_test = train_test_split(df, test_size=cfg["training"]["test_size"],
                                  stratify=df["species_label"], random_state=SEED)
    return df_test

def evaluate_stage1(df_test):
    pipe  = PHCIPipeline()
    X     = df_test[FEAT_COLS].values
    y_sp  = df_test["species_label"].values
    # Stage 1 direct evaluation (bypass pipeline for isolated stage test)
    y_pred, confs = pipe.stage1.model.predict(X), pipe.stage1.model.predict_proba(X).max(axis=1)
    y_pred = pipe.stage1.model.predict(X)
    acc    = accuracy_score(y_sp, y_pred)
    f1     = f1_score(y_sp, y_pred, average="weighted")
    cm     = confusion_matrix(y_sp, y_pred)
    print("\n=== Stage 1 Validation ===")
    print(f"Accuracy: {acc:.4f} | F1 (weighted): {f1:.4f}")
    print(classification_report(y_sp, y_pred, target_names=SPECIES))
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SPECIES, yticklabels=SPECIES, ax=ax)
    ax.set_title("Stage 1: Species Classification Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.savefig(FIG_DIR / "stage1_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    return acc, f1

def evaluate_stage2(df_test):
    pipe = PHCIPipeline()
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, sp in enumerate(SPECIES):
        sp_df = df_test[df_test["species"] == sp]
        X     = sp_df[FEAT_COLS].values
        y     = sp_df["stress_label"].values
        y_pred, _ = pipe.stage2.predict(sp, X)
        acc  = accuracy_score(y, y_pred)
        f1   = f1_score(y, y_pred, average="weighted")
        cm   = confusion_matrix(y, y_pred)
        results[sp] = {"accuracy": round(acc,4), "f1": round(f1,4)}
        print(f"\nStage 2 [{sp}]: Accuracy={acc:.4f} | F1={f1:.4f}")
        print(classification_report(y, y_pred, target_names=STRESSES))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=STRESSES, yticklabels=STRESSES,
                    ax=axes[idx])
        axes[idx].set_title(f"Stage 2: {sp.title()} Stress CM")
        axes[idx].tick_params(axis='x', rotation=30)
    fig.savefig(FIG_DIR / "stage2_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    return results

def test_end_to_end(df_test, n_samples=20):
    """Run full pipeline on random test windows and print semantic outputs."""
    pipe = PHCIPipeline()
    print("\n=== End-to-End Pipeline Inference Samples ===")
    sample = df_test.sample(n=n_samples, random_state=42)
    for _, row in sample.iterrows():
        x    = row[FEAT_COLS].values.astype(np.float32)
        # Un-scale for pipeline (pipeline re-scales internally via scaler)
        import joblib
        sc = joblib.load(Path(cfg['paths']['models']) / 'scaler.pkl')
        x_raw = sc.inverse_transform(x.reshape(1,-1))[0]
        result = pipe.run(x_raw)
        true_sp = row["species"]; true_st = row["stress_state"]
        pred_sp = result["stage1"]["species_name"]
        pred_st = result["stage2"]["stress_name"] if result["stage2"] else "N/A"
        match = (pred_sp == true_sp and pred_st == true_st)
        print(f"  {'OK' if match else 'WRONG'} | True: {true_sp}/{true_st} | Pred: {pred_sp}/{pred_st}")
        print(f"         Alert: {result['alert']}")

if __name__ == "__main__":
    df_test = load_test_set()
    s1_acc, s1_f1 = evaluate_stage1(df_test)
    s2_results    = evaluate_stage2(df_test)
    test_end_to_end(df_test)
    print("\nAll figures saved to outputs/figures/")
