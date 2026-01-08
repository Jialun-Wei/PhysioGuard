import vitaldb
import pandas as pd
import numpy as np
import os

class VitalDataLoader:
    """
    Handles fetching and pre-processing of real high-frequency
    physiological waveforms from the VitalDB open dataset.
    """

    def __init__(self):
        # Tracking ECG (Heart activity) and ART (Arterial Blood Pressure)
        # These are standard high-frequency signals in ICU monitoring.
        self.track_names = ['ECG_II', 'ART']

    def fetch_real_case_data(self, duration_seconds=60):
        """
        Finds a valid case and loads a segment of high-frequency data.
        """
        print(f"[INFO] Searching for cases containing tracks: {self.track_names}...")

        # 1. Find cases that have both ECG and Arterial Pressure
        case_ids = vitaldb.find_cases(self.track_names)

        if not case_ids:
            print("[ERROR] No cases found with the specified tracks.")
            return None

        target_case_id = case_ids[0]
        print(f"[INFO] Found {len(case_ids)} cases. Loading Case ID: {target_case_id}")

        # 2. Load the case data
        # sampling_interval = 1/100 (100Hz)
        # vals is a 2D numpy array [samples, tracks]
        try:
            # We only load a specific duration to keep the MVP responsive
            # 100Hz * 60s = 6000 samples
            sample_count = 100 * duration_seconds
            vals = vitaldb.load_case(target_case_id, self.track_names, 1/100)

            # Slice the data to the desired duration
            vals = vals[:sample_count]
        except Exception as e:
            print(f"[ERROR] Failed to load case {target_case_id}: {e}")
            return None

        # 3. Data Transformation to DataFrame
        # Handling potential NaNs which are common in raw medical records
        df = pd.DataFrame(vals, columns=['ecg', 'art'])
        df['timestamp_ms'] = np.arange(len(df)) * 10 # 100Hz = 10ms per sample

        # Critical for Research: Drop initial NaNs where sensors might not be connected
        df.dropna(inplace=True)

        print(f"[SUCCESS] Successfully loaded {len(df)} points of high-frequency data.")
        return df

    def save_to_parquet(self, df, filename="high_freq_vital.parquet"):
        """
        Saves the processed high-frequency dataframe into the 'data' folder
        located at the project root directory.
        """
        # 1. Get the absolute path of the current script (src/vital_loader.py)
        script_path = os.path.abspath(__file__)

        # 2. Get the directory of the script (src/)
        src_dir = os.path.dirname(script_path)

        # 3. Get the parent directory (the Project Root: PhysioGuard/)
        project_root = os.path.dirname(src_dir)

        # 4. Define the target directory at the project root
        output_dir = os.path.join(project_root, "data")

        # 5. Create the directory if it does not already exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[INFO] Created directory: {output_dir}")

        # 6. Construct the full output path for the file
        output_path = os.path.join(output_dir, filename)

        # 7. Serialize the data to Parquet
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"[SUCCESS] High-frequency data saved to: {output_path}")

if __name__ == "__main__":
    loader = VitalDataLoader()
    data = loader.fetch_real_case_data(duration_seconds=120) # Get 2 minutes of data
    if data is not None:
        loader.save_to_parquet(data)