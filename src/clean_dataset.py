import os
from pathlib import Path

def clean_dataset(root_path="data/raw"):
    """
    Clean the RCAEval dataset to only hold data we care about. 
    This is a one-time cleanup script to remove redundant files and reduce the size of the repo.

    Parameters:
        -root_path: The root directory containing the RCAEval data folders (e.g., RE1-OB, RE1-SS, etc.)
    """
    root = Path(root_path)
    deleted_count = 0
    kept_count = 0

    # Ensure we target the RE folders
    for re_folder in root.glob("RE*-*"):
        # Only process if it's a directory.
        if not re_folder.is_dir():
            continue
            
        print(f"Cleaning folder: {re_folder.name}")

        for fault_folder in re_folder.iterdir():
            # Only process if it's a directory.
            if not fault_folder.is_dir():
                continue

            for run_folder in fault_folder.iterdir():
                # Only process if it's a directory.
                if not run_folder.is_dir():
                    continue

                # Always keep inject_time.txt if it exists.
                keep_names = {"inject_time.txt"} 
                
                # Define priority for data files, first match wins. We can use any file.
                # The "simple" versions are smaller and preferred if they exist.
                # The "data" and "metrics" files don't typically exist together.
                priority_list = ["simple_data.csv", "data.csv", "simple_metrics.csv", "metrics.csv"]

                # Find the first existing file from the list.
                for filename in priority_list:
                    if (run_folder / filename).exists():
                        keep_names.add(filename)
                        break 

                if keep_names:
                    for file in run_folder.iterdir():
                        if file.is_file():
                            if file.name not in keep_names:
                                try:
                                    file.unlink()
                                    deleted_count += 1
                                except Exception as e:
                                    print(f"Error deleting {file}: {e}")
                            else:
                                kept_count += 1
                else:
                    print(f"Warning: No metrics found in {run_folder.name}! Skipping cleanup for this folder.")

    print(f"\n--- Cleanup Summary ---")
    print(f"Files Deleted: {deleted_count}")
    print(f"Files Preserved: {kept_count}")

if __name__ == "__main__":
    clean_dataset()