"""
preprocess.py
Sliding window feature extraction producing the 32-feature vector.
Works identically for synthetic data and real hardware data.
The ONLY difference for real data: the input CSV path changes. Nothing else.
"""

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks, welch, butter, sosfiltfilt
from scipy.stats import skew, kurtosis
import antropy as ant
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

cfg = yaml.safe_load(open("config/config.yaml"))
WS  = cfg["signal"]["window_size_samples"]   # 258
HOP = cfg["signal"]["hop_size_samples"]       # 129
FS  = cfg["signal"]["sample_rate_hz"]         # 8.6
N_F = cfg["signal"]["n_features"]             # 32

FEATURE_NAMES = [
    "mean","std","rms","p2p","skewness","kurtosis",
    "zcr","mean_slope","max_slope","pos_slope","neg_slope","energy",
    "pwr_dc","pwr_low","pwr_mid","peak_freq",
    "spec_entropy","spec_centroid","spec_rolloff","spec_flatness",
    "n_peaks","n_valleys","peak_mean","valley_mean","rise_time","fall_time",
    "ac_1s","ac_5s","ac_15s","apen","sampen","hurst"
]

def extract_features(w: np.ndarray) -> dict:
    w = w.astype(np.float64)
    f = {}
    diffs = np.diff(w)
    # Time-domain
    f["mean"]       = np.mean(w)
    f["std"]        = np.std(w, ddof=1)
    f["rms"]        = np.sqrt(np.mean(w**2))
    f["p2p"]        = w.max() - w.min()
    f["skewness"]   = float(skew(w))
    f["kurtosis"]   = float(kurtosis(w))
    f["zcr"]        = np.sum(np.diff(np.sign(w)) != 0) / (len(w) / FS)
    f["mean_slope"] = np.mean(np.abs(diffs) * FS)
    f["max_slope"]  = np.max(np.abs(diffs) * FS)
    f["pos_slope"]  = float(np.mean(diffs[diffs>0]*FS)) if (diffs>0).any() else 0.0
    f["neg_slope"]  = float(np.mean(diffs[diffs<0]*FS)) if (diffs<0).any() else 0.0
    f["energy"]     = float(np.sum(w**2))
    # Frequency-domain
    freqs, psd = welch(w, fs=FS, nperseg=min(64, len(w)))
    bp = lambda lo, hi: float(np.trapezoid(psd[(freqs>=lo)&(freqs<hi)],
                                           freqs[(freqs>=lo)&(freqs<hi)]))
    f["pwr_dc"]       = bp(0.0, 0.1)
    f["pwr_low"]      = bp(0.1, 1.0)
    f["pwr_mid"]      = bp(1.0, 5.0)
    f["peak_freq"]    = float(freqs[np.argmax(psd)])
    pn = psd/(psd.sum()+1e-10)
    f["spec_entropy"] = float(-np.sum(pn*np.log2(pn+1e-10)))
    f["spec_centroid"]= float(np.sum(freqs*psd)/(psd.sum()+1e-10))
    cs = np.cumsum(psd)
    f["spec_rolloff"] = float(freqs[np.searchsorted(cs, 0.85*cs[-1])])
    f["spec_flatness"]= float(np.exp(np.mean(np.log(psd+1e-10)))/(np.mean(psd)+1e-10))
    # Morphological
    peaks,_   = find_peaks(w,  height=np.mean(w), distance=int(FS))
    valleys,_ = find_peaks(-w, height=-np.mean(w), distance=int(FS))
    f["n_peaks"]     = len(peaks)
    f["n_valleys"]   = len(valleys)
    f["peak_mean"]   = float(w[peaks].mean())   if len(peaks)   else float(np.mean(w))
    f["valley_mean"] = float(w[valleys].mean()) if len(valleys) else float(np.mean(w))
    rise = [peaks[i]-valleys[i] for i in range(min(len(peaks),len(valleys)))]
    f["rise_time"]   = float(np.mean(rise)/FS*1000) if rise else 0.0
    fall = [valleys[i+1]-peaks[i] for i in range(min(len(valleys)-1,len(peaks)))]
    f["fall_time"]   = float(np.mean(fall)/FS*1000) if fall else 0.0
    # Statistical
    lag = lambda s,l: float(np.corrcoef(s[:-l],s[l:])[0,1]) if l<len(s) else 0.0
    f["ac_1s"]  = lag(w, int(1*FS))
    f["ac_5s"]  = lag(w, int(5*FS))
    f["ac_15s"] = lag(w, int(min(15*FS, len(w)-1)))
    f["apen"]   = float(ant.app_entropy(w, order=2))
    f["sampen"] = float(ant.sample_entropy(w, order=2))
    n = len(w); dev = np.cumsum(w-np.mean(w))
    R = dev.max()-dev.min(); S = np.std(w,ddof=1)
    f["hurst"]  = float(np.log(R/S)/np.log(n)) if S>0 else 0.5
    assert len(f) == N_F, f"Feature count mismatch: {len(f)} != {N_F}"
    return f

def process_file(csv_path: Path) -> list:
    df  = pd.read_csv(csv_path)
    sig = df["voltage_mv"].values.astype(np.float32)
    rows = []
    for start in range(0, len(sig)-WS, HOP):
        w    = sig[start:start+WS]
        feat = extract_features(w)
        feat["species"]      = df["species"].iloc[0]
        feat["stress_state"] = df["stress_state"].iloc[0]
        feat["species_label"]= df["species_label"].iloc[0]
        feat["stress_label"] = df["stress_label"].iloc[0]
        feat["session_id"]   = df["session_id"].iloc[0]
        rows.append(feat)
    return rows

if __name__ == "__main__":
    raw_dir = Path(cfg["paths"]["synthetic_raw"])
    processed_dir = Path(cfg["paths"]["processed"])
    models_dir = Path(cfg["paths"]["models"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for f in sorted(raw_dir.glob("*.csv")):
        print(f"Processing {f.name}...")
        all_rows.extend(process_file(f))
    master = pd.DataFrame(all_rows)
    out = processed_dir / "master_features.csv"
    master.to_csv(out, index=False)
    print(f"Feature matrix: {master.shape} -> {out}")
    # Fit and save scaler on feature columns only
    feat_cols = FEATURE_NAMES
    scaler = StandardScaler()
    master[feat_cols] = scaler.fit_transform(master[feat_cols])
    master.to_csv(processed_dir / "master_features_scaled.csv", index=False)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print("Scaler saved to models/scaler.pkl")
