import numpy as np
import pandas as pd

class SignalAnalyzer:
    """
    Provides numerical analysis tools for high-frequency physiological signals.
    Focuses on denoising, differentiation, and feature extraction.
    """

    def __init__(self, sampling_rate=100):
        """
        :param sampling_rate: Number of samples per second (Hz). Default is 100Hz.
        """
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate  # Time interval between samples (h)

    def apply_moving_average(self, signal, window_size=5):
        """
        Smooths the signal using a simple moving average filter.
        Addresses Project 297's need for clean clinical visualization. TODO
        """
        # We use numpy's convolution to efficiently calculate moving average
        window = np.ones(window_size) / window_size
        return np.convolve(signal, window, mode='same')

    def calculate_derivative(self, signal):
        """
        Calculates the first-order derivative using the Finite Difference Method.
        Directly relates to Project 314 (Numerical Analysis/Fluid Dynamics).
        Useful for detecting sudden spikes in blood pressure.
        """
        # np.diff calculates f(x+h) - f(x)
        # We divide by dt to get the rate of change
        derivative = np.diff(signal) / self.dt

        # Append a zero at the end to maintain original array length
        return np.append(derivative, 0)

    def detect_peaks(self, signal, threshold=0.8):
        """
        A simplified peak detection algorithm for ECG R-waves.
        Calculates local maxima that exceed a certain relative threshold.
        """
        # Normalize signal for consistent thresholding
        if np.max(signal) == np.min(signal): return []
        norm_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

        peaks = []
        for i in range(1, len(norm_signal) - 1):
            # Check if current point is greater than neighbors and exceeds threshold
            if norm_signal[i] > norm_signal[i-1] and \
                    norm_signal[i] > norm_signal[i+1] and \
                    norm_signal[i] > threshold:
                peaks.append(i)
        return np.array(peaks)

    def calculate_heart_rate(self, peak_indices):
        """
        Converts detected ECG peaks into Beats Per Minute (BPM).
        Demonstrates Project 294's ability to derive 'Truth' for AI validation.
        """
        if len(peak_indices) < 2:
            return 0

        # Calculate time difference between consecutive peaks in seconds
        intervals = np.diff(peak_indices) * self.dt
        avg_interval = np.mean(intervals)

        # BPM = 60 / average interval in seconds
        return 60.0 / avg_interval

# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Example logic to test the analyzer
    # loader = ... (read your parquet file)
    # analyzer = SignalAnalyzer(sampling_rate=100)
    # clean_ecg = analyzer.apply_moving_average(raw_ecg)
    # hr = analyzer.calculate_heart_rate(analyzer.detect_peaks(clean_ecg))
    print("[INFO] Numerical Analyzer module initialized.")