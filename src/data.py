from __future__ import annotations

import pandas as pd

from .config import INCIDENTS_PATH, METRICS_PATH
from .generate_sample_data import ensure_sample_data


def load_incidents() -> pd.DataFrame:
    ensure_sample_data()
    incidents = pd.read_csv(INCIDENTS_PATH)
    incidents["is_anomalous"] = incidents["is_anomalous"].astype(bool)
    return incidents


def load_metrics() -> pd.DataFrame:
    ensure_sample_data()
    return pd.read_csv(METRICS_PATH)
