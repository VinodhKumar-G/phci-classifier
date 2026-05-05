"""
pipeline.py
End-to-end inference pipeline.
This is the core patentable system: species identification ->
confidence gate -> species-conditioned stress classification ->
semantic output generation.

Real data integration: no changes to this file required.
Only the input feature vector changes (same shape, same columns).
"""

import numpy as np
import yaml
import joblib
from pathlib import Path
from stage1_species import Stage1SpeciesClassifier
from stage2_stress import Stage2StressBank
from semantic_output import SemanticOutputLayer

cfg = yaml.safe_load(open("config/config.yaml"))

class PHCIPipeline:
    """
    The complete two-stage plant bioelectrical signal classifier.
    Architecture:
        Feature vector (32-dim)
            -> Stage 1: species ID + confidence
            -> Confidence gate (threshold check)
            -> Stage 2: species-conditioned stress classification
            -> Semantic output layer: natural language alert
    """
    def __init__(self, model_dir: str = None):
        md = Path(model_dir or cfg["paths"]["models"])
        self.scaler   = joblib.load(md / "scaler.pkl")
        self.stage1   = Stage1SpeciesClassifier.load(md / "stage1_rf.pkl")
        self.stage2   = Stage2StressBank.load_all(md)
        self.semantic = SemanticOutputLayer()
        self.sp_thresh = cfg["thresholds"]["species_confidence_min"]
        self.st_thresh = cfg["thresholds"]["stress_confidence_min"]

    def run(self, feature_vector: np.ndarray) -> dict:
        """
        Full pipeline inference on a single 32-feature window.

        Parameters
        ----------
        feature_vector : np.ndarray, shape (32,) — NOT pre-scaled
            32 features in the exact order of FEATURE_NAMES

        Returns
        -------
        dict with full prediction details and semantic alert
        """
        # Scale input using the fitted scaler (from training)
        x_scaled = self.scaler.transform(feature_vector.reshape(1, -1))

        # Stage 1: species identification
        s1 = self.stage1.predict_single(x_scaled[0])

        if not s1["above_threshold"]:
            return {
                "stage1": s1,
                "stage2": None,
                "alert":  self.semantic.uncertain(s1),
                "status": "UNCERTAIN_SPECIES"
            }

        # Stage 2: species-conditioned stress classification
        s2 = self.stage2.predict_single(s1["species_name"], x_scaled[0])

        alert = (
            self.semantic.generate(s1, s2)
            if s2["above_threshold"]
            else self.semantic.uncertain_stress(s1, s2)
        )

        return {
            "stage1": s1,
            "stage2": s2,
            "alert":  alert,
            "status": "OK"
        }
