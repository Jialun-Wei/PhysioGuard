import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.numerical_analyzer import SignalAnalyzer

# =============================================================================
# PATH SETUP
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "high_freq_vital.parquet")

def run_analysis_test():
    """
    Test script to verify the correctness of the Numerical Analyzer.
    Includes data visualization and physiological range validation.
    """
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found at {DATA_PATH}. Please run vital_loader.py first.")
        return

    # 1. Load the High-Frequency Data
    print("[INFO] Loading high-frequency Parquet data...")
    df = pd.read_parquet(DATA_PATH)

    # We take a small segment (e.g., 5 seconds = 500 samples at 100Hz) for clear visualization
    sample_segment = df.iloc[1000:1500].copy()
    raw_ecg = sample_segment['ecg'].values
    raw_art = sample_segment['art'].values

    # 2. Initialize Analyzer (100Hz sampling rate)
    analyzer = SignalAnalyzer(sampling_rate=100)

    # 3. Apply Processing
    print("[INFO] Applying Numerical Analysis...")
    # Denoising
    clean_ecg = analyzer.apply_moving_average(raw_ecg, window_size=5)
    # Differentiation (Project 314 requirement)
    art_derivative = analyzer.calculate_derivative(raw_art)
    # Feature Extraction (Peak detection)
    peaks = analyzer.detect_peaks(clean_ecg, threshold=0.7)
    # Result Calculation
    bpm = analyzer.calculate_heart_rate(peaks)

    # 4. VISUAL VERIFICATION (The "Does it look right?" test)
    print("[INFO] Generating validation plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: ECG Smoothing and Peak Detection
    ax1.plot(raw_ecg, label='Raw ECG (Noisy)', color='lightgray', alpha=0.7)
    ax1.plot(clean_ecg, label='Clean ECG (Moving Average)', color='green', linewidth=2)
    ax1.plot(peaks, clean_ecg[peaks], "x", label='Detected R-peaks', color='red')
    ax1.set_title(f"ECG Analysis (Calculated Heart Rate: {bpm:.2f} BPM)")
    ax1.legend()

    # Plot 2: Arterial Pressure Derivative (Calculus in action)
    ax2.plot(raw_art, label='Raw ART Pressure', color='blue')
    ax2.plot(art_derivative / 100, label='Scaled Derivative (dp/dt)', color='orange', linestyle='--')
    ax2.set_title("Arterial Pressure & Numerical Derivative")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # 5. LOGICAL VALIDATION (The "Does it make sense?" test)
    validate_results(bpm, art_derivative)

def validate_results(bpm, derivative):
    """
    Performs basic physiological and mathematical sanity checks.
    """
    print("\n" + "="*30)
    print("VALIDATION REPORT")
    print("="*30)

    # Check 1: Heart Rate Range
    if 40 <= bpm <= 180:
        print(f"[PASS] Heart Rate ({bpm:.2f} BPM) is within normal human range.")
    else:
        print(f"[FAIL] Heart Rate ({bpm:.2f} BPM) is physiologically improbable!")

    # Check 2: Numerical Stability
    if not np.isnan(derivative).any():
        print(f"[PASS] Derivative calculation is numerically stable (No NaNs).")
    else:
        print(f"[FAIL] Derivative contains NaN values. Check input data.")

if __name__ == "__main__":
    run_analysis_test()