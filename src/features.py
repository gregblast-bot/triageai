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


def build_feature_row(
    metrics: pd.DataFrame,
    *,
    incident_id: str,
    text: str,
    fault_type: str = "unknown",
    root_cause_service: str = "unknown",
    is_anomalous: bool = False,
    data_split: str | None = None,
) -> dict:
    working_metrics = metrics.copy()
    if "minute" in working_metrics.columns:
        working_metrics = working_metrics.sort_values("minute")
    working_metrics = working_metrics.reset_index(drop=True)

    for column in METRIC_COLUMNS:
        if column not in working_metrics.columns:
            working_metrics[column] = 0.0
        working_metrics[column] = pd.to_numeric(working_metrics[column], errors="coerce").fillna(0.0)

    row = {
        "incident_id": incident_id,
        "text": text,
        "fault_type": fault_type,
        "root_cause_service": root_cause_service,
        "is_anomalous": bool(is_anomalous),
    }
    if data_split is not None:
        row["data_split"] = data_split
    for column in METRIC_COLUMNS:
        row.update(_summarize_metric(working_metrics[column], column))
    return row


def build_feature_frame(
    incidents: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    feature_rows: list[dict] = []

    grouped = metrics.sort_values(["incident_id", "minute"]).groupby("incident_id")
    incident_lookup = incidents.set_index("incident_id")

    for incident_id, group in grouped:
        base = incident_lookup.loc[incident_id]
        feature_rows.append(
            build_feature_row(
                group,
                incident_id=incident_id,
                text=" ".join(str(base[column]) for column in TEXT_COLUMNS),
                fault_type=base["fault_type"],
                root_cause_service=base["root_cause_service"],
                is_anomalous=bool(base["is_anomalous"]),
                data_split=base["data_split"] if "data_split" in base.index else None,
            )
        )

    return pd.DataFrame(feature_rows)


def get_numeric_feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    excluded = {
        "incident_id",
        "text",
        "fault_type",
        "root_cause_service",
        "is_anomalous",
        "data_split",
    }
    return [column for column in feature_frame.columns if column not in excluded]
