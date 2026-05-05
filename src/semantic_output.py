"""
semantic_output.py
Translates two-stage classification outputs into human-readable alerts.
This is the fourth novel element in the patent claims:
semantic translation of dual-classification output.
"""

class SemanticOutputLayer:
    MESSAGES = {
        ("mimosa",  "healthy"):       "Mimosa pudica: Healthy. No stimulus detected. Electrical baseline stable.",
        ("mimosa",  "water_stress"):   "Mimosa pudica: Water deficit detected. Variation potential pattern observed. Irrigation recommended within 2-4 hours.",
        ("mimosa",  "heat_stress"):    "Mimosa pudica: Heat stress detected. Elevated action potential frequency above baseline. Provide shade immediately.",
        ("mimosa",  "wound_response"): "Mimosa pudica: Wound response detected. Large-amplitude rapid action potential cascade. Inspect plant for physical damage.",
        ("tomato",  "healthy"):        "Solanum lycopersicum: Healthy. Low-amplitude baseline potentials. Normal physiological state.",
        ("tomato",  "water_stress"):   "Solanum lycopersicum: Water stress detected. Sustained depolarisation pattern. Irrigation recommended.",
        ("tomato",  "heat_stress"):    "Solanum lycopersicum: Thermal stress detected. Increase ventilation or reduce ambient temperature.",
        ("tomato",  "wound_response"): "Solanum lycopersicum: Wound signal detected. System potential propagation observed. Check for pest or mechanical damage.",
        ("aloe",    "healthy"):        "Aloe vera: Healthy. Very low amplitude slow baseline drift. Normal for species.",
        ("aloe",    "water_stress"):   "Aloe vera: Water deficit indicated. Although drought-tolerant, prolonged stress detected. Monitor closely.",
        ("aloe",    "heat_stress"):    "Aloe vera: Heat stress signals present. Reduce direct sun exposure or increase ambient humidity.",
        ("aloe",    "wound_response"): "Aloe vera: Physical disturbance detected. Slow potential change indicates mechanical stress.",
    }

    def generate(self, s1: dict, s2: dict) -> str:
        key = (s1["species_name"], s2["stress_name"])
        base = self.MESSAGES.get(key, f"Unknown condition: {key}")
        conf_str = (
            f" [Species confidence: {s1['species_confidence']:.0%} | "
            f"Stress confidence: {s2['stress_confidence']:.0%}]"
        )
        return base + conf_str

    def uncertain(self, s1: dict) -> str:
        return (
            f"UNCERTAIN: Species identification confidence {s1['species_confidence']:.0%} "
            f"is below threshold {s1['above_threshold']}. "
            f"Signal quality insufficient for stress classification. "
            f"Check electrode contact and signal noise level."
        )

    def uncertain_stress(self, s1: dict, s2: dict) -> str:
        return (
            f"{s1['species_name'].title()}: Species identified with {s1['species_confidence']:.0%} confidence. "
            f"Stress state uncertain ({s2['stress_confidence']:.0%} confidence). "
            f"Most likely: {s2['stress_name']}. Inspect plant visually to confirm."
        )
