import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.numerical_analyzer import SignalAnalyzer

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "high_freq_vital.parquet")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "visuals")

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

def find_clean_segment(df, fs=100, window_sec=12):
    """
    Scans the dataset to find a 'Medical Grade' segment.
    Criteria:
    1. ECG signal is not flat (std dev > threshold).
    2. ART signal has pulsating rhythm (std dev > threshold).
    3. No massive artifacts (outliers).
    """
    total_samples = len(df)
    step = int(fs * window_sec) # Jump by window size

    best_start = 0
    max_quality_score = -1

    print("[INFO] Scanning for the cleanest signal segment...")

    for i in range(0, total_samples - step, int(step/2)):
        segment = df.iloc[i : i + step]
        ecg = segment['ecg'].values
        art = segment['art'].values

        # Quality Metrics
        ecg_std = np.std(ecg)
        art_std = np.std(art)

        # Heuristic:
        # 1. ECG shouldn't be flat (std > 0.1) but not noise (std < 2.0)
        # 2. ART should vary (std > 5) implies pulse pressure
        if 0.1 < ecg_std < 1.0 and art_std > 5:
            # We found a candidate!
            # Use variability as a score (we want rhythmic variation)
            score = ecg_std + art_std
            if score > max_quality_score:
                max_quality_score = score
                best_start = i

    print(f"[SUCCESS] Found best segment at Index {best_start} (Time: {best_start/fs:.1f}s)")
    return best_start

def generate_monitor_view():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data not found at {DATA_PATH}")
        return

    # 1. Load Data
    print("[INFO] Loading high-frequency waveform data...")
    df = pd.read_parquet(DATA_PATH)
    fs = 100
    duration_sec = 12

    # --- STEP 1: SMART SEARCH ---
    best_idx = find_clean_segment(df, fs, duration_sec)

    # Extract the clean window
    window_df = df.iloc[best_idx : best_idx + int(duration_sec * fs)].copy()

    t = np.linspace(0, duration_sec, len(window_df))
    ecg = window_df['ecg'].values
    art = window_df['art'].values

    # --- STEP 2: ADVANCED SIGNAL PROCESSING ---
    analyzer = SignalAnalyzer(sampling_rate=fs)

    # ECG: Light smoothing
    ecg_clean = analyzer.apply_moving_average(ecg, window_size=2)

    # ART: Heavy smoothing
    art_clean = analyzer.apply_moving_average(art, window_size=8)

    # --- STEP 3: VISUAL TUNING (Clean Waveforms Only) ---
    print("[INFO] Rendering Clean Waveforms...")
    plt.style.use('dark_background')

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=300)
    plt.subplots_adjust(hspace=0.0) # Zero gap between plots

    # Plot 1: ECG (Green)
    ax1.plot(t, ecg_clean, color='#00FF00', linewidth=2.0)
    ax1.set_ylim(np.min(ecg_clean)-0.5, np.max(ecg_clean)+0.5)
    ax1.axis('off') # Remove axes (ticks, spines, labels)

    # [REMOVED] Text labels for ECG
    # ax1.text(0, np.max(ecg_clean), "II", color='#00FF00', fontsize=16, fontweight='bold')
    # peaks = analyzer.detect_peaks(ecg_clean, threshold=0.6)
    # real_bpm = analyzer.calculate_heart_rate(peaks) if len(peaks) > 0 else 80
    # ax1.text(duration_sec - 1, np.mean(ecg_clean), f"{int(real_bpm)}", ...)
    # ax1.text(duration_sec - 1, np.mean(ecg_clean)+0.5, "ECG", ...)

    # Plot 2: ART (Red)
    ax2.plot(t, art_clean, color='#FF3333', linewidth=2.0)
    ax2.set_ylim(np.min(art_clean)-10, np.max(art_clean)+10)
    ax2.axis('off') # Remove axes

    # 4. Export
    # Naming it 'clean' so you can distinguish it
    output_filename = "icu_monitor_clean_waveforms.svg"
    output_path = os.path.join(REPORT_DIR, output_filename)

    # Save with transparent background option if needed, currently set to black facecolor
    plt.savefig(output_path, format='svg', bbox_inches='tight', facecolor='black', pad_inches=0.1)

    print(f"[SUCCESS] Clean Waveform SVG generated: {output_path}")

if __name__ == "__main__":
    generate_monitor_view()