import vitaldb
import pandas as pd
import numpy as np
import os

# =============================================================================
# VITALDB HIGH-FREQUENCY LOADER
# =============================================================================
# This script fetches REAL-TIME waveforms from the VitalDB open repository.
# It matches the high-frequency requirement of SickKids AtriumDB.

class VitalHighFreqLoader:
    """
    Handles fetching of high-resolution (500Hz) physiological waveforms.
    """
    def __init__(self):
        # Specific track names in VitalDB matching SickKids requirements
        # SNUADC/ECG_II: Electrocardiogram (500Hz)
        # SNUADC/PLETH: Photoplethysmogram (SpO2 waveform, 100Hz)
        self.track_names = ['SNUADC/ECG_II', 'SNUADC/PLETH']

    def fetch_sample_case(self, case_id=1):
        """
        Fetches high-frequency data for a specific surgical case.
        """
        print(f"[INFO] Fetching high-frequency waveforms for Case {case_id}...")

        # 1. Download data (Returns a numpy array)
        # interval=0.01 means 100Hz sampling rate
        vals = vitaldb.load_vital(self.track_names, caseid=case_id, interval=0.01)

        # 2. Convert to DataFrame
        df_high_freq = pd.DataFrame(vals, columns=['ecg', 'ppg'])

        # 3. Add timestamp (Assuming 100Hz, each point is 10ms)
        df_high_freq['timestamp_ms'] = np.arange(len(df_high_freq)) * 10

        # Drop rows with NaN (common at start/end of surgery)
        df_high_freq.dropna(inplace=True)

        print(f"[SUCCESS] Loaded {len(df_high_freq)} points of high-freq data.")
        return df_high_freq

    def export_to_parquet(self, df, filename="high_freq_vital.parquet"):
        """
        Saves to parquet for use in the Streamlit Dashboard.
        """
        output_path = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)
        df.to_parquet(output_path, engine='pyarrow')
        print(f"[INFO] High-frequency data saved to {output_path}")

if __name__ == "__main__":
    loader = VitalHighFreqLoader()
    # Fetching case 10 from VitalDB
    data = loader.fetch_sample_case(case_id=10)
    if not data.empty:
        # We only take first 60 seconds to keep MVP light (100Hz * 60s = 6000 points)
        loader.export_to_parquet(data.iloc[:6000])