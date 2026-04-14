from __future__ import annotations

import pandas as pd

from .config import INCIDENTS_PATH, METRICS_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .data_converter import convert_data
from .generate_sample_data import ensure_sample_data


def ensure_processed_data() -> None:
    if INCIDENTS_PATH.exists() and METRICS_PATH.exists():
        return

    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        convert_data(root_path=str(RAW_DATA_DIR), output_dir=str(PROCESSED_DATA_DIR))
        return

    ensure_sample_data()


def load_incidents() -> pd.DataFrame:
    ensure_processed_data()
    incidents = pd.read_csv(INCIDENTS_PATH)
    incidents["is_anomalous"] = incidents["is_anomalous"].astype(bool)
    if "data_split" not in incidents.columns:
        incidents["data_split"] = "Train"
    return incidents


def load_metrics() -> pd.DataFrame:
    ensure_processed_data()
    return pd.read_csv(METRICS_PATH)
