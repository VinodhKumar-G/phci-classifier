# tests/test_pipeline.py
# Run: pytest tests/test_pipeline.py -v
# No trained models required — all Stage 1/2 outputs are mocked

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# ── helpers ──────────────────────────────────────────────────────────
def _make_s1(species, confidence, threshold=0.70):
    return {
        'species_label':      ['mimosa','tomato','aloe'].index(species),
        'species_name':       species,
        'species_confidence': confidence,
        'above_threshold':    confidence >= threshold,
    }

def _make_s2(stress, confidence, threshold=0.60):
    STRESSES = ['healthy','water_stress','heat_stress','wound_response']
    return {
        'stress_label':      STRESSES.index(stress),
        'stress_name':       stress,
        'stress_confidence': confidence,
        'above_threshold':   confidence >= threshold,
    }

# ── tests ────────────────────────────────────────────────────────────
class TestPHCIPipelineRun:

    @patch('pipeline.joblib.load')
    @patch('pipeline.Stage1SpeciesClassifier.load')
    @patch('pipeline.Stage2StressBank.load_all')
    @patch('pipeline.SemanticOutputLayer')
    def test_both_stages_confident(self, MockSem, MockS2, MockS1, mock_joblib):
        '''Stage 1 and Stage 2 both above threshold -> status OK, alert generated.'''
        from pipeline import PHCIPipeline
        mock_joblib.return_value = MagicMock()
        mock_joblib.return_value.transform.return_value = np.zeros((1, 32))

        pipe = PHCIPipeline.__new__(PHCIPipeline)
        pipe.scaler  = MagicMock()
        pipe.scaler.transform.return_value = np.zeros((1, 32))
        pipe.stage1  = MagicMock()
        pipe.stage2  = MagicMock()
        pipe.semantic = MagicMock()
        pipe.sp_thresh = 0.70
        pipe.st_thresh = 0.60

        s1 = _make_s1('mimosa', 0.92)
        s2 = _make_s2('water_stress', 0.88)
        pipe.stage1.predict_single.return_value = s1
        pipe.stage2.predict_single.return_value = s2
        pipe.semantic.generate.return_value = 'Mimosa water stress alert.'

        result = pipe.run(np.zeros(32))

        assert result['status'] == 'OK'
        assert result['stage1'] == s1
        assert result['stage2'] == s2
        assert result['alert']  == 'Mimosa water stress alert.'
        pipe.semantic.generate.assert_called_once_with(s1, s2)

    @patch('pipeline.joblib.load')
    def test_species_below_threshold(self, mock_joblib):
        '''Stage 1 confidence below threshold -> UNCERTAIN_SPECIES, stage2 is None.'''
        from pipeline import PHCIPipeline
        pipe = PHCIPipeline.__new__(PHCIPipeline)
        pipe.scaler  = MagicMock()
        pipe.scaler.transform.return_value = np.zeros((1, 32))
        pipe.stage1  = MagicMock()
        pipe.stage2  = MagicMock()
        pipe.semantic = MagicMock()
        pipe.sp_thresh = 0.70
        pipe.st_thresh = 0.60

        s1 = _make_s1('tomato', 0.45)  # below 0.70
        pipe.stage1.predict_single.return_value = s1
        pipe.semantic.uncertain.return_value = 'UNCERTAIN: low confidence.'

        result = pipe.run(np.zeros(32))

        assert result['status'] == 'UNCERTAIN_SPECIES'
        assert result['stage2'] is None
        assert 'UNCERTAIN' in result['alert']
        pipe.stage2.predict_single.assert_not_called()

    @patch('pipeline.joblib.load')
    def test_stress_below_threshold(self, mock_joblib):
        '''Stage 1 OK, Stage 2 below threshold -> OK status, uncertain_stress alert.'''
        from pipeline import PHCIPipeline
        pipe = PHCIPipeline.__new__(PHCIPipeline)
        pipe.scaler  = MagicMock()
        pipe.scaler.transform.return_value = np.zeros((1, 32))
        pipe.stage1  = MagicMock()
        pipe.stage2  = MagicMock()
        pipe.semantic = MagicMock()
        pipe.sp_thresh = 0.70
        pipe.st_thresh = 0.60

        s1 = _make_s1('aloe', 0.85)         # above threshold
        s2 = _make_s2('heat_stress', 0.50)  # below 0.60
        pipe.stage1.predict_single.return_value = s1
        pipe.stage2.predict_single.return_value = s2
        pipe.semantic.uncertain_stress.return_value = 'Aloe: uncertain stress.'

        result = pipe.run(np.zeros(32))

        assert result['status'] == 'OK'
        assert result['stage2'] is not None
        pipe.semantic.uncertain_stress.assert_called_once_with(s1, s2)
        pipe.semantic.generate.assert_not_called()

class TestPipelineScalerTransform:

    def test_feature_vector_is_scaled(self):
        '''Pipeline must scale the raw feature vector before passing to Stage 1.'''
        from pipeline import PHCIPipeline
        pipe = PHCIPipeline.__new__(PHCIPipeline)
        sc_mock = MagicMock()
        sc_mock.transform.return_value = np.zeros((1, 32))
        pipe.scaler  = sc_mock
        pipe.stage1  = MagicMock()
        pipe.stage1.predict_single.return_value = _make_s1('mimosa', 0.40)  # uncertain
        pipe.stage2  = MagicMock()
        pipe.semantic = MagicMock()
        pipe.semantic.uncertain.return_value = 'UNCERTAIN'
        pipe.sp_thresh = 0.70
        pipe.st_thresh = 0.60

        x_raw = np.ones(32) * 5.0
        pipe.run(x_raw)
        sc_mock.transform.assert_called_once()
        call_arg = sc_mock.transform.call_args[0][0]
        assert call_arg.shape == (1, 32)
