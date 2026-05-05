"""
generate_data.py
Generates biologically realistic plant bioelectrical signal datasets.
Parameters are derived from published electrophysiology literature.
Output: one CSV per (species, stress_state) combination in data/synthetic/raw/
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

cfg = yaml.safe_load(open("config/config.yaml"))
RNG = np.random.default_rng(cfg["project"]["seed"])
FS  = cfg["signal"]["sample_rate_hz"]

# ── Signal parameter library (literature-derived) ──────────────────────
# Each entry: (baseline_mv, noise_std, event_amp_range, event_rise_s,
#              event_decay_s, events_per_min_range)
PARAMS = {
  ("mimosa", "healthy"):       (0.5, 0.3, (0.5, 2.0), 0.5, 1.5, (0.1, 0.5)),
  ("mimosa", "water_stress"):   (1.2, 0.5, (3.0, 8.0), 1.0, 4.0, (0.3, 1.0)),
  ("mimosa", "heat_stress"):    (1.8, 0.7, (5.0,15.0), 0.6, 3.0, (0.5, 2.0)),
  ("mimosa", "wound_response"): (0.5, 0.4, (40.0,100.0),0.2, 2.0, (2.0, 5.0)),
  ("tomato", "healthy"):        (0.3, 0.2, (0.3, 1.5), 2.0, 8.0, (0.05,0.2)),
  ("tomato", "water_stress"):   (0.8, 0.4, (5.0,15.0), 4.0,20.0, (0.1, 0.5)),
  ("tomato", "heat_stress"):    (1.0, 0.6, (8.0,25.0), 3.0,15.0, (0.2, 0.8)),
  ("tomato", "wound_response"): (0.5, 0.3, (5.0,20.0), 5.0,60.0, (0.1, 0.4)),
  ("aloe",   "healthy"):        (0.2, 0.15,(0.2, 1.0), 5.0,20.0, (0.02,0.1)),
  ("aloe",   "water_stress"):   (0.5, 0.3, (1.5, 4.0),10.0,60.0, (0.05,0.2)),
  ("aloe",   "heat_stress"):    (0.7, 0.4, (2.0, 6.0), 8.0,40.0, (0.1, 0.3)),
  ("aloe",   "wound_response"): (0.3, 0.2, (1.0, 3.0),15.0,90.0, (0.02,0.1)),
}

def generate_session(species, stress, duration_min=45):
    """
    Returns a 1D numpy array: plant bioelectrical signal at FS Hz.
    duration_min: recording duration in minutes.
    """
    baseline, noise, amp_r, rise, decay, rate_r = PARAMS[(species, stress)]
    n_samples = int(duration_min * 60 * FS)
    signal = RNG.normal(baseline, noise, n_samples)  # baseline + Gaussian noise
    t      = np.arange(n_samples) / FS

    # Inject electrophysiological events
    events_per_min = RNG.uniform(*rate_r)
    n_events = int(events_per_min * duration_min)
    event_times = RNG.uniform(0, duration_min * 60, n_events)

    for t0 in event_times:
        amp = RNG.uniform(*amp_r)
        idx = int(t0 * FS)
        # Rise phase (linear)
        rise_samp = max(1, int(rise * FS))
        for i in range(min(rise_samp, n_samples - idx)):
            signal[idx + i] += amp * (i / rise_samp)
        # Decay phase (exponential)
        decay_samp = max(1, int(decay * FS))
        for i in range(min(decay_samp, n_samples - idx - rise_samp)):
            signal[idx + rise_samp + i] += amp * np.exp(-3 * i / decay_samp)

    return signal.astype(np.float32)

def save_session(species, stress, run_id, signal):
    n = len(signal)
    ts_ms = (np.arange(n) / FS * 1000).astype(np.int64)
    df = pd.DataFrame({
        "timestamp_ms":  ts_ms,
        "session_id":    f"{species}_{stress}_run{run_id}",
        "species":       species,
        "stress_state":  stress,
        "voltage_mv":    signal,
        "species_label": cfg["project"]["species"].index(species),
        "stress_label":  cfg["project"]["stress_states"].index(stress),
    })
    out = Path(cfg["paths"]["synthetic_raw"]) / f"{species}_{stress}_run{run_id}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows -> {out}")

if __name__ == "__main__":
    n_sessions = cfg["synthetic"]["sessions_per_class"]
    dur = cfg["synthetic"]["session_duration_min"]
    for sp in cfg["project"]["species"]:
        for st in cfg["project"]["stress_states"]:
            for run in range(1, n_sessions + 1):
                sig = generate_session(sp, st, dur)
                save_session(sp, st, run, sig)
    print(f"Generation complete: {3 * 4 * n_sessions} CSV files")
