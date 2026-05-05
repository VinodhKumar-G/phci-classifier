"""
stage1_species.py
Stage 1: Random Forest classifier for plant species identification.
Input:  32-feature vector from a 30-second signal window
Output: species label (0=mimosa, 1=tomato, 2=aloe) + confidence score
"""

import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

cfg      = yaml.safe_load(open("config/config.yaml"))
s1_cfg   = cfg["stage1"]
tr_cfg   = cfg["training"]
SEED     = cfg["project"]["seed"]
SPECIES  = cfg["project"]["species"]
MIN_CONF = cfg["thresholds"]["species_confidence_min"]

FEAT_COLS = [
    "mean","std","rms","p2p","skewness","kurtosis",
    "zcr","mean_slope","max_slope","pos_slope","neg_slope","energy",
    "pwr_dc","pwr_low","pwr_mid","peak_freq",
    "spec_entropy","spec_centroid","spec_rolloff","spec_flatness",
    "n_peaks","n_valleys","peak_mean","valley_mean","rise_time","fall_time",
    "ac_1s","ac_5s","ac_15s","apen","sampen","hurst"
]

class Stage1SpeciesClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators  = s1_cfg["n_estimators"],
            max_depth     = s1_cfg["max_depth"],
            min_samples_leaf = s1_cfg["min_samples_leaf"],
            max_features  = s1_cfg["max_features"],
            class_weight  = s1_cfg["class_weight"],
            random_state  = SEED,
            n_jobs        = -1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (species_labels, confidence_scores)"""
        if not self.is_fitted:
            raise RuntimeError("Stage1 model not fitted. Call fit() first.")
        labels = self.model.predict(X)
        probas = self.model.predict_proba(X)
        confidence = probas.max(axis=1)
        return labels, confidence

    def predict_single(self, x: np.ndarray) -> dict:
        """Single-window inference. Returns dict with label, name, confidence."""
        label, conf = self.predict(x.reshape(1, -1))
        return {
            "species_label":      int(label[0]),
            "species_name":       SPECIES[int(label[0])],
            "species_confidence": float(conf[0]),
            "above_threshold":    float(conf[0]) >= MIN_CONF,
        }

    def save(self, path: Path):
        joblib.dump(self.model, path)
        print(f"Stage 1 model saved: {path}")

    @classmethod
    def load(cls, path: Path) -> 'Stage1SpeciesClassifier':
        obj = cls()
        obj.model     = joblib.load(path)
        obj.is_fitted = True
        return obj
