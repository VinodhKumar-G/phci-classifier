# src/feature_extractor.py
# CORRECTED for numpy 2.x compatibility
# Key fix: scipy.integrate.trapezoid replaces removed np.trapz

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid   # FIX: np.trapz removed in numpy 2.x
import antropy as ant
import yaml, warnings

# Suppress pandas 3.x copy-on-write deprecation warning (informational only)
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

cfg = yaml.safe_load(open('config/config.yaml'))
FS  = cfg['signal']['sample_rate_hz']   # 8.6 Hz

FEATURE_NAMES = [
    'mean','std','rms','p2p','skewness','kurtosis',
    'zero_cross_rate','mean_abs_slope','max_slope','pos_slope_mean','neg_slope_mean','energy',
    'pwr_dc','pwr_low','pwr_mid','peak_freq','spec_entropy','spec_centroid','spec_rolloff','spec_flatness',
    'n_peaks','n_valleys','peak_amp_mean','valley_amp_mean','mean_rise_time_ms','mean_fall_time_ms',
    'autocorr_1s','autocorr_5s','autocorr_15s','approx_entropy','sample_entropy','hurst_exponent',
]

def extract(window: np.ndarray) -> np.ndarray:
    '''
    Extract 32-dimensional feature vector from one 30-second signal window.
    COMPATIBLE: numpy 2.x, scipy 1.15.x, antropy 0.2.x
    Input:  1D numpy array of 258 voltage values (millivolts)
    Output: 1D numpy array of 32 floats, no NaN, no Inf
    '''
    w = np.asarray(window, dtype=np.float64)
    diffs = np.diff(w)
    f = {}

    # Time domain (12 features)
    f['mean']            = float(np.mean(w))
    f['std']             = float(np.std(w, ddof=1))
    f['rms']             = float(np.sqrt(np.mean(w**2)))
    f['p2p']             = float(w.max() - w.min())
    # skew/kurtosis return NaN if variance=0; replace with 0.0
    sk = skew(w)
    ku = kurtosis(w)
    f['skewness']        = float(sk) if not np.isnan(sk) else 0.0
    f['kurtosis']        = float(ku) if not np.isnan(ku) else 0.0
    f['zero_cross_rate'] = float(np.sum(np.diff(np.sign(w)) != 0) / (len(w) / FS))
    f['mean_abs_slope']  = float(np.mean(np.abs(diffs) * FS))
    f['max_slope']       = float(np.max(np.abs(diffs) * FS))
    pos = diffs[diffs > 0] * FS
    neg = diffs[diffs < 0] * FS
    f['pos_slope_mean']  = float(np.mean(pos)) if len(pos) > 0 else 0.0
    f['neg_slope_mean']  = float(np.mean(neg)) if len(neg) > 0 else 0.0
    f['energy']          = float(np.sum(w**2))

    # Frequency domain (8 features)
    freqs, psd = welch(w, fs=FS, nperseg=min(64, len(w)))

    def band_power(lo: float, hi: float) -> float:
        '''Power in frequency band [lo, hi) Hz.'''
        idx = (freqs >= lo) & (freqs < hi)
        # FIX: use scipy.integrate.trapezoid — np.trapz removed in numpy 2.x
        return float(trapezoid(psd[idx], freqs[idx])) if idx.any() else 0.0

    f['pwr_dc']          = band_power(0.0, 0.1)
    f['pwr_low']         = band_power(0.1, 1.0)
    f['pwr_mid']         = band_power(1.0, 5.0)
    f['peak_freq']       = float(freqs[np.argmax(psd)])
    psd_n = psd / (psd.sum() + 1e-12)
    f['spec_entropy']    = float(-np.sum(psd_n * np.log2(psd_n + 1e-12)))
    f['spec_centroid']   = float(np.sum(freqs * psd) / (psd.sum() + 1e-12))
    cs = np.cumsum(psd)
    f['spec_rolloff']    = float(freqs[np.searchsorted(cs, 0.85 * cs[-1])])
    geo = np.exp(np.mean(np.log(psd + 1e-12)))
    f['spec_flatness']   = float(geo / (np.mean(psd) + 1e-12))

    # Morphological (6 features)
    pks,  _ = find_peaks( w, height=np.mean(w), distance=int(FS))
    vals, _ = find_peaks(-w, height=-np.mean(w), distance=int(FS))
    f['n_peaks']           = len(pks)
    f['n_valleys']         = len(vals)
    f['peak_amp_mean']     = float(w[pks].mean())  if len(pks)  > 0 else float(np.mean(w))
    f['valley_amp_mean']   = float(w[vals].mean()) if len(vals) > 0 else float(np.mean(w))
    rises = [pks[i] - vals[i] for i in range(min(len(pks), len(vals)))]
    falls = [vals[i+1] - pks[i] for i in range(min(len(vals)-1, len(pks)))]
    f['mean_rise_time_ms'] = float(np.mean(rises) / FS * 1000) if rises else 0.0
    f['mean_fall_time_ms'] = float(np.mean(falls) / FS * 1000) if falls else 0.0

    # Statistical / entropy (6 features)
    def lag_autocorr(sig: np.ndarray, lag_sec: float) -> float:
        lag = int(lag_sec * FS)
        if lag >= len(sig): return 0.0
        # Handle constant signals (std=0) to avoid NaN
        if np.std(sig) < 1e-12: return 1.0
        corr = np.corrcoef(sig[:-lag], sig[lag:])[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    f['autocorr_1s']    = lag_autocorr(w, 1)
    f['autocorr_5s']    = lag_autocorr(w, 5)
    f['autocorr_15s']   = lag_autocorr(w, 15)
    try:
        ae = float(ant.app_entropy(w, order=2))
        se = float(ant.sample_entropy(w, order=2))
        f['approx_entropy']  = ae if not np.isnan(ae) else 0.0
        f['sample_entropy']  = se if not np.isnan(se) else 0.0
    except Exception:
        f['approx_entropy']  = 0.0
        f['sample_entropy']  = 0.0
    n   = len(w)
    dev = np.cumsum(w - np.mean(w))
    R   = dev.max() - dev.min()
    S   = float(np.std(w, ddof=1))
    f['hurst_exponent']  = float(np.log(R / S) / np.log(n)) if S > 1e-8 else 0.5

    return np.array([f[name] for name in FEATURE_NAMES], dtype=np.float64)

def extract_batch(windows: np.ndarray) -> np.ndarray:
    '''
    Batch feature extraction.
    Input:  shape (n_windows, 258)
    Output: shape (n_windows, 32)
    '''
    return np.vstack([extract(w) for w in windows])
