import pandas as pd
import os
import numpy as np
from pathlib import Path

def convert_data(root_path="data/raw", output_dir="data/processed"):
    '''
    Converts the RCAEval dataset into flat CSV files for incidents and metrics.
    Only processes folders that match the expected fault types and file naming conventions.
    Parameters:
        - root_path: The root directory containing the raw data.
        - output_dir: The directory where the processed CSV files will be saved.
    Output:
        - incidents.csv: Contains metadata about each incident.
        - metrics.csv: Contains the extracted metrics for each incident.
    '''
    all_metrics = []
    all_incidents = []
    incident_counter = 1
    
    # Define valid faults and mappings. This ensures we only process relevant folders and files.
    VALID_FAULTS = {'cpu', 'mem', 'disk', 'delay', 'loss', 'socket'}
    FILE_MAP = { "RE1-OB": "data.csv", "RE1-SS": "simple_data.csv", "RE1-TT": "simple_data.csv"}

    root = Path(root_path)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate top-level folders (e.g. RE1-OB, RE1-SS, RE1-TT, RE2-OB, etc.)
    for top_folder in root.iterdir():
        # Only process if a mapping exists.
        if not top_folder.is_dir() or top_folder.name not in FILE_MAP:
            continue
        
        target_filename = FILE_MAP[top_folder.name]
        print("*"*50)
        print(f"Processing Top-Level Folder: {top_folder.name}")
        
        # Iterate fault type folders (e.g. cpu, mem, disk, delay, loss, socket)
        for fault_folder in top_folder.iterdir():
            # Only process if it's a directory.
            if not fault_folder.is_dir():
                continue

            fault_type = fault_folder.name.lower()
            print(f"Processing Fault Type: {fault_type}")
            
            # Find which valid fault is in the folder name.
            matched_fault = None
            for f in VALID_FAULTS:
                if f in fault_type:
                    matched_fault = f
                    break
            
            # If the folder name doesn't contain any of our keywords, skip it.
            if not matched_fault:
                continue
            
            print(f"Matched Pattern: '{matched_fault}' in folder '{fault_folder.name}'")

            # Iterate numbered subfolders (e.g. 1, 2, 3, etc.)
            for run_folder in fault_folder.iterdir():
                # Only process if it's a directory.
                if not run_folder.is_dir():
                    continue
                
                # Only process if the directory contains the target file.
                target_path = run_folder / target_filename
                if not target_path.exists():
                    continue

                # Store the .csv file as a dataframe.
                df = pd.read_csv(target_path)
                inc_id = f"INC-{incident_counter:05d}"
                
                # Metadata for incidents.csv. Generic incident entry corresponding to valid faults.
                all_incidents.append({
                    "incident_id": inc_id,
                    "title": f"System Fault: {matched_fault.upper()}",
                    "description": f"Generic {matched_fault} instability detected in {top_folder.name}.",
                    "fault_type": matched_fault,          
                    "root_cause_service": f"{fault_type}",   # Generic service
                    "region": "unknown",              # Default region is unknown in the RCAEval dataset
                    "is_anomalous": True              # Since we are only targeting fault folders
                })

                # Metric extraction, assuming the first 60 rows represent a one hour window (one row per minute).
                for minute, (_, row) in enumerate(df.head(60).iterrows()):
                    def get_mean(pattern):
                        '''
                        Helper function to calculate mean of columns matching a pattern.
                        Parameters:
                            - pattern: The pattern to match in column names. For example, get_mean('_cpu') will average all columns containing '_cpu'.
                        Output:
                            - Returns the mean if any matching columns are found, otherwise returns 0.0.
                        '''
                        cols = [c for c in df.columns if pattern in c]
                        return row[cols].mean() if cols else 0.0

                    all_metrics.append({
                        "incident_id": inc_id,
                        "minute": minute,
                        "cpu_pct": get_mean('_cpu'),
                        "memory_pct": (get_mean('_mem') / (1024**3)) * 100, # Convert bytes to GB and normalize
                        "latency_ms": get_mean('_latency') * 1000, # Convert seconds to milliseconds
                        "error_rate": get_mean('error'),
                        "queue_depth": get_mean('load'),
                        "auth_error_rate": 0.0
                    })
                
                incident_counter += 1

    # Create a dataframe to hold all incidents.
    df_incidents = pd.DataFrame(all_incidents)
    
    # Assign splits, 80% train and 20% test.
    np.random.seed(42) 
    mask = np.random.rand(len(df_incidents)) < 0.8
    df_incidents['data_split'] = np.where(mask, 'Train', 'Test')

    # Save outputs.
    df_incidents.to_csv(f"{output_dir}/incidents.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(f"{output_dir}/metrics.csv", index=False)
    print(f"\nFinished! Compiled {incident_counter - 1} specific fault incidents.")


if __name__ == "__main__":
    convert_data()