import pandas as pd
import os

# =============================================================================
# GLOBAL PATH RESOLUTION
# =============================================================================
# We use __file__ to locate the script's directory and derive the Project Root.
# This ensures portability across different machines (Windows, Mac, Linux).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Defining the path to the MIMIC-IV Demo dataset relative to the root
MIMIC_DIR = os.path.join(PROJECT_ROOT, "data", "mimic-iv-clinical-database-demo-2.2")

# =============================================================================
# MIMIC-IV DICTIONARY MAPPING
# =============================================================================
# Standard Item IDs used in MIMIC-IV for vital signs in the ICU
ITEM_ID_MAP = {
    220045: "heart_rate",   # Units: bpm
    220277: "spo2",         # Units: %
    220179: "systolic_bp",  # Units: mmHg
    220180: "diastolic_bp"  # Units: mmHg
}

class MIMICDataLoader:
    """
    Handles the Extraction, Transformation, and Loading (ETL) of clinical
    data from the MIMIC-IV Clinical Database.
    """

    def __init__(self, raw_data_dir):
        """
        Initializes the loader with the directory containing MIMIC CSV files.
        """
        self.raw_data_dir = raw_data_dir
        # chartevents.csv.gz contains all recorded physiologic measurements
        self.chartevents_path = os.path.join(raw_data_dir, "icu", "chartevents.csv.gz")

    def process_patient_data(self, subject_id):
        """
        Extracts raw ICU measurements and transforms them into a cleaned
        time-series format for the dashboard.
        """
        if not os.path.exists(self.chartevents_path):
            print(f"[ERROR] Cannot find file: {self.chartevents_path}")
            print("[HINT] Ensure the MIMIC-IV demo folder is placed inside 'PhysioGuard/data/'.")
            return None

        print(f"[INFO] Reading records for Subject {subject_id} from chartevents...")

        # 1. Load the raw compressed CSV
        # We read the entire file for the demo; for full MIMIC, we would use chunking.
        try:
            df_raw = pd.read_csv(self.chartevents_path, compression='gzip')
        except Exception as e:
            print(f"[ERROR] Failed to read CSV: {e}")
            return None

        # 2. Filtering: Extracting specific patient and target vital sign metrics
        target_item_ids = list(ITEM_ID_MAP.keys())
        df_filtered = df_raw[
            (df_raw['subject_id'] == subject_id) &
            (df_raw['itemid'].isin(target_item_ids))
            ].copy()

        if df_filtered.empty:
            print(f"[WARNING] No vital sign data found for Subject {subject_id}.")
            return None

        # 3. Time-Series Transformation (Pivoting)
        # Convert string timestamps to datetime objects for temporal sorting
        df_filtered['charttime'] = pd.to_datetime(df_filtered['charttime'])

        # Pivot the 'Long' format table to a 'Wide' format (Time-Series)
        # Rows = Timestamps, Columns = Metric Values
        df_pivot = df_filtered.pivot_table(
            index='charttime',
            columns='itemid',
            values='valuenum',
            aggfunc='first'
        )

        # Map cryptic Item IDs to human-readable labels
        df_pivot.rename(columns=ITEM_ID_MAP, inplace=True)

        # 4. Numerical Pre-processing (Interpolation)
        # Clinical data is often sparse. We use Linear Interpolation to approximate
        # values between measurement gaps, ensuring a continuous stream for the UI.
        # This addresses requirements for Project 314 (Numerical Analysis).
        df_interpolated = df_pivot.interpolate(method='time').dropna().reset_index()

        # Rename the time index for clarity
        df_interpolated.rename(columns={'charttime': 'timestamp'}, inplace=True)

        print(f"[SUCCESS] Extracted {len(df_interpolated)} processed data points.")
        return df_interpolated

    def save_to_parquet(self, df, filename="processed_mimic_data.parquet"):
        """
        Serializes the processed DataFrame into Parquet format for optimized I/O.
        Parquet is a columnar storage format highly efficient for big data workflows.
        """
        output_dir = os.path.join(PROJECT_ROOT, "data")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)

        # We use pyarrow as the underlying engine for fast serialization
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"[INFO] Processed dataset saved to: {output_path}")

if __name__ == "__main__":
    # Initialize the data loader with the dynamic path
    loader = MIMICDataLoader(MIMIC_DIR)

    # Process Subject 10000032 (A standard patient in the MIMIC demo)
    processed_df = loader.process_patient_data(10000032)

    # Serialize to disk if successful
    if processed_df is not None:
        loader.save_to_parquet(processed_df)