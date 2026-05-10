# tests/test_semantic.py
# Run: pytest tests/test_semantic.py -v
# No mocks needed — SemanticOutputLayer is pure logic, no file I/O

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from semantic_output import SemanticOutputLayer

SPECIES  = ['mimosa', 'tomato', 'aloe']
STRESSES = ['healthy', 'water_stress', 'heat_stress', 'wound_response']

def _s1(species, confidence=0.87):
    return {'species_name': species, 'species_confidence': confidence,
            'species_label': SPECIES.index(species), 'above_threshold': confidence >= 0.70}

def _s2(stress, confidence=0.91):
    return {'stress_name': stress, 'stress_confidence': confidence,
            'stress_label': STRESSES.index(stress), 'above_threshold': confidence >= 0.60}


class TestSemanticOutputGenerate:

    def setup_method(self):
        self.sem = SemanticOutputLayer()

    def test_all_12_combos_return_strings(self):
        '''Every species x stress combination must return a non-empty string.'''
        for sp in SPECIES:
            for st in STRESSES:
                out = self.sem.generate(_s1(sp), _s2(st))
                assert isinstance(out, str), f'Not a string for {sp}/{st}'
                assert len(out) > 20, f'Alert too short for {sp}/{st}: {out!r}'

    def test_confidence_appears_in_output(self):
        '''generate() must include both confidence percentages.'''
        out = self.sem.generate(_s1('mimosa', 0.87), _s2('water_stress', 0.91))
        # Confidence values should appear as percentages e.g. 87% or 87
        assert '87' in out, f'Species confidence missing: {out}'
        assert '91' in out, f'Stress confidence missing: {out}'

    def test_unknown_key_returns_fallback(self):
        '''generate() must not raise on an unknown species/stress combo.'''
        s1 = {'species_name': 'cactus', 'species_confidence': 0.80,
              'species_label': 99, 'above_threshold': True}
        s2 = {'stress_name': 'cosmic_ray', 'stress_confidence': 0.75,
              'stress_label': 99, 'above_threshold': True}
        out = self.sem.generate(s1, s2)
        assert isinstance(out, str) and len(out) > 5

    @pytest.mark.parametrize('species', SPECIES)
    def test_each_species_message_contains_latin_or_common_name(self, species):
        '''Each species should have a unique base message, not just fallback.'''
        EXPECTED = {'mimosa': 'Mimosa', 'tomato': 'Solanum', 'aloe': 'Aloe'}
        out = self.sem.generate(_s1(species), _s2('healthy'))
        assert EXPECTED[species] in out, f'{EXPECTED[species]} not in alert: {out}'


class TestSemanticOutputUncertain:

    def setup_method(self):
        self.sem = SemanticOutputLayer()

    def test_uncertain_returns_string(self):
        out = self.sem.uncertain(_s1('tomato', 0.45))
        assert isinstance(out, str)
        assert len(out) > 10

    def test_uncertain_contains_uncertain_keyword(self):
        out = self.sem.uncertain(_s1('tomato', 0.45))
        assert 'UNCERTAIN' in out.upper()

    def test_uncertain_contains_confidence_value(self):
        out = self.sem.uncertain(_s1('tomato', 0.45))
        assert '45' in out, f'Confidence not in uncertain alert: {out}'

    def test_uncertain_stress_contains_species_name(self):
        out = self.sem.uncertain_stress(_s1('aloe', 0.82), _s2('heat_stress', 0.55))
        assert 'Aloe' in out or 'aloe' in out.lower()

    def test_uncertain_stress_contains_stress_confidence(self):
        out = self.sem.uncertain_stress(_s1('mimosa', 0.88), _s2('wound_response', 0.52))
        assert '52' in out, f'Stress confidence not in uncertain_stress alert: {out}'

    def test_uncertain_stress_contains_stress_name(self):
        out = self.sem.uncertain_stress(_s1('tomato', 0.75), _s2('water_stress', 0.58))
        assert 'water_stress' in out or 'water stress' in out.lower()


class TestSemanticOutputAllStressStates:

    def setup_method(self):
        self.sem = SemanticOutputLayer()

    @pytest.mark.parametrize('stress', STRESSES)
    def test_wound_response_has_dedicated_message(self, stress):
        '''wound_response must have its own message entry, not fall through to unknown.'''
        out = self.sem.generate(_s1('mimosa'), _s2(stress))
        assert 'Unknown' not in out, f'Fallback triggered for {stress}: {out}'
