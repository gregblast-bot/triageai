from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    METRIC_COLUMNS,
    OPTIONAL_METRIC_COLUMNS,
    PRECOMPUTED_SCALAR_FEATURES,
    TEXT_COLUMNS,
)


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
    precomputed_scalars: dict[str, float] | None = None,
) -> dict:
    working_metrics = metrics.copy()
    if "minute" in working_metrics.columns:
        working_metrics = working_metrics.sort_values("minute")
    working_metrics = working_metrics.reset_index(drop=True)

    all_metric_columns = list(METRIC_COLUMNS) + list(OPTIONAL_METRIC_COLUMNS)
    for column in all_metric_columns:
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
    for column in all_metric_columns:
        row.update(_summarize_metric(working_metrics[column], column))

    # Latency spread is p90 minus p50. If the source didn't ship p50 this just
    # collapses to zero, which is what we want.
    row["latency_spread_ms_mean"] = row["latency_ms_mean"] - row["latency_p50_ms_mean"]
    row["latency_spread_ms_max"] = row["latency_ms_max"] - row["latency_p50_ms_max"]

    scalars = precomputed_scalars or {}
    for column in PRECOMPUTED_SCALAR_FEATURES:
        row[column] = float(scalars.get(column, 0.0))

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
        scalars = {
            column: float(base[column])
            for column in PRECOMPUTED_SCALAR_FEATURES
            if column in base.index and pd.notna(base[column])
        }
        feature_rows.append(
            build_feature_row(
                group,
                incident_id=incident_id,
                text=" ".join(str(base[column]) for column in TEXT_COLUMNS),
                fault_type=base["fault_type"],
                root_cause_service=base["root_cause_service"],
                is_anomalous=bool(base["is_anomalous"]),
                data_split=base["data_split"] if "data_split" in base.index else None,
                precomputed_scalars=scalars or None,
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
