"""
stage2_stress.py
Stage 2: Three species-specific SVM stress classifiers.
One SVM is trained per species. Stage 1 output selects which SVM runs.
This is the key architectural novelty: conditioning on species identity
before stress classification.
"""

import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import classification_report

cfg      = yaml.safe_load(open("config/config.yaml"))
s2_cfg   = cfg["stage2"]
SPECIES  = cfg["project"]["species"]
STRESSES = cfg["project"]["stress_states"]
SEED     = cfg["project"]["seed"]
MIN_CONF = cfg["thresholds"]["stress_confidence_min"]

FEAT_COLS = [
    "mean","std","rms","p2p","skewness","kurtosis",
    "zcr","mean_slope","max_slope","pos_slope","neg_slope","energy",
    "pwr_dc","pwr_low","pwr_mid","peak_freq",
    "spec_entropy","spec_centroid","spec_rolloff","spec_flatness",
    "n_peaks","n_valleys","peak_mean","valley_mean","rise_time","fall_time",
    "ac_1s","ac_5s","ac_15s","apen","sampen","hurst"
]

class Stage2StressBank:
    """
    Manages three independent SVM models, one per species.
    predict() selects the correct SVM based on Stage 1 species output.
    """
    def __init__(self):
        self.models = {
            sp: SVC(
                kernel       = s2_cfg["kernel"],
                C            = s2_cfg["C"],
                gamma        = s2_cfg["gamma"],
                class_weight = s2_cfg["class_weight"],
                probability  = s2_cfg["probability"],
                random_state = SEED
            ) for sp in SPECIES
        }
        self.fitted = {sp: False for sp in SPECIES}

    def fit_species(self, species: str, X: np.ndarray, y: np.ndarray):
        """Train the SVM for a specific species."""
        self.models[species].fit(X, y)
        self.fitted[species] = True
        print(f"Stage 2 [{species}] SVM trained on {len(y)} windows")

    def predict(self, species_name: str, X: np.ndarray) -> tuple:
        """
        Run species-conditioned stress prediction.
        Returns (stress_labels, confidence_scores)
        """
        if not self.fitted[species_name]:
            raise RuntimeError(f"Stage2 model for {species_name} not fitted.")
        labels = self.models[species_name].predict(X)
        probas = self.models[species_name].predict_proba(X)
        confidence = probas.max(axis=1)
        return labels, confidence

    def predict_single(self, species_name: str, x: np.ndarray) -> dict:
        labels, conf = self.predict(species_name, x.reshape(1, -1))
        return {
            "stress_label":      int(labels[0]),
            "stress_name":       STRESSES[int(labels[0])],
            "stress_confidence": float(conf[0]),
            "above_threshold":   float(conf[0]) >= MIN_CONF,
        }

    def save_all(self, model_dir: Path):
        for sp in SPECIES:
            p = model_dir / f"stage2_{sp}_svm.pkl"
            joblib.dump(self.models[sp], p)
            print(f"Saved Stage 2 [{sp}]: {p}")

    @classmethod
    def load_all(cls, model_dir: Path) -> 'Stage2StressBank':
        obj = cls()
        for sp in SPECIES:
            p = model_dir / f"stage2_{sp}_svm.pkl"
            obj.models[sp] = joblib.load(p)
            obj.fitted[sp] = True
        return obj
