"""
visualize.py
Visualises bioelectrical signal differences across species and stress states.
Works for both synthetic (Step 5) and real hardware data (Step 16+).
Change DATA_DIR to switch between synthetic and real data sources.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yaml
from pathlib import Path

cfg     = yaml.safe_load(open("config/config.yaml"))
SPECIES = cfg["project"]["species"]
STRESS  = cfg["project"]["stress_states"]
FIG_DIR = Path(cfg["paths"]["figures"])
FS      = cfg["signal"]["sample_rate_hz"]

COLOURS = {
    "mimosa": "#E74C3C",  "tomato": "#2ECC71",  "aloe": "#3498DB",
    "healthy":"#2ECC71",  "water_stress":"#3498DB",
    "heat_stress":"#E74C3C", "wound_response":"#F39C12"
}

def plot_species_comparison(data_dir: str, n_seconds=60):
    """Plot 60-second signal sample for each species side by side."""
    n_samp = int(n_seconds * FS)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Plant Bioelectrical Signal Comparison by Species",
                 fontsize=14, fontweight='bold')
    for ax, sp in zip(axes, SPECIES):
        # Load one healthy session for clean species comparison
        f = next(Path(data_dir).glob(f"{sp}_healthy_run1.csv"))
        sig = pd.read_csv(f)["voltage_mv"].values[:n_samp]
        t   = np.arange(len(sig)) / FS
        ax.plot(t, sig, color=COLOURS[sp], linewidth=0.8, alpha=0.9)
        ax.set_ylabel(f"{sp.title()}\n(mV)", fontsize=10)
        ax.set_ylim(sig.mean()-sig.std()*6, sig.mean()+sig.std()*6)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (seconds)")
    fig.savefig(FIG_DIR / "species_signal_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: species_signal_comparison.png")

def plot_stress_comparison(data_dir: str, species="mimosa", n_seconds=60):
    """For one species, overlay all 4 stress states."""
    n_samp = int(n_seconds * FS)
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"{species.title()}: Signal Variation Across Stress States",
                 fontsize=13, fontweight='bold')
    for st in STRESS:
        files = list(Path(data_dir).glob(f"{species}_{st}_run1.csv"))
        if not files: continue
        sig = pd.read_csv(files[0])["voltage_mv"].values[:n_samp]
        t   = np.arange(len(sig)) / FS
        ax.plot(t, sig, color=COLOURS[st], linewidth=0.9,
                alpha=0.85, label=st.replace("_", " ").title())
    ax.set_xlabel("Time (seconds)"); ax.set_ylabel("Voltage (mV)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / f"{species}_stress_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {species}_stress_comparison.png")

def plot_feature_heatmap():
    """Heatmap of mean feature values per (species, stress) class."""
    df  = pd.read_csv(Path(cfg["paths"]["processed"]) / "master_features.csv")
    key_feats = ["mean","std","rms","p2p","zcr","pwr_dc","pwr_low",
                 "pwr_mid","spec_entropy","n_peaks","rise_time","hurst"]
    df["class"] = df["species"] + " | " + df["stress_state"]
    pivot = df.groupby("class")[key_feats].mean()
    pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-10)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot_norm, cmap="YlOrRd", annot=False, ax=ax)
    ax.set_title("Normalised Feature Means by Species-Stress Class",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Feature"); ax.set_ylabel("Species | Stress State")
    fig.savefig(FIG_DIR / "feature_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: feature_heatmap.png")

if __name__ == "__main__":
    DATA_DIR = cfg["paths"]["synthetic_raw"]
    plot_species_comparison(DATA_DIR)
    for sp in SPECIES:
        plot_stress_comparison(DATA_DIR, species=sp)
    plot_feature_heatmap()
    print(f"All figures saved to {FIG_DIR}")
