from __future__ import annotations

import numpy as np
import pandas as pd

from .config import METRIC_COLUMNS, TEXT_COLUMNS


def _safe_slope(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values.to_numpy(), deg=1)
    return float(coeffs[0])


def _summarize_metric(values: pd.Series, prefix: str) -> dict:
    median = float(values.median())
    maximum = float(values.max())
    minimum = float(values.min())
    mean = float(values.mean())
    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_max": maximum,
        f"{prefix}_min": minimum,
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_delta": end - start,
        f"{prefix}_spike": maximum - median,
        f"{prefix}_slope": _safe_slope(values),
    }


def build_feature_frame(
    incidents: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    feature_rows: list[dict] = []

    grouped = metrics.sort_values(["incident_id", "minute"]).groupby("incident_id")
    incident_lookup = incidents.set_index("incident_id")

    for incident_id, group in grouped:
        base = incident_lookup.loc[incident_id]
        row = {
            "incident_id": incident_id,
            "text": " ".join(str(base[column]) for column in TEXT_COLUMNS),
            "fault_type": base["fault_type"],
            "root_cause_service": base["root_cause_service"],
            "is_anomalous": bool(base["is_anomalous"]),
            "data_split": base["data_split"],
        }
        for column in METRIC_COLUMNS:
            row.update(_summarize_metric(group[column], column))
        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def get_numeric_feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    excluded = {"incident_id", "text", "fault_type", "root_cause_service", "is_anomalous"}
    return [column for column in feature_frame.columns if column not in excluded]
