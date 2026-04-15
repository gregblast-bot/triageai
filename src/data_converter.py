from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


FILE_MAP = {"RE1-OB": "data.csv", "RE1-SS": "simple_data.csv", "RE1-TT": "simple_data.csv",
            "RE2-OB": "simple_metrics.csv", "RE2-SS": "simple_metrics.csv", "RE2-TT": "simple_metrics.csv"}
WINDOW_SIZE = 120
PRE_FAULT_CONTEXT = 30
NORMAL_WINDOW_GAP = 60


def _extract_labels(folder_name: str) -> tuple[str, str] | tuple[None, None]:
    match = re.search(r"_(cpu|mem|disk|delay|loss|socket)$", folder_name.lower())
    if not match:
        return None, None
    fault_type = match.group(1)
    service_name = folder_name[: -len(f"_{fault_type}")]
    return fault_type, service_name


def _get_injection_index(run_folder: Path, frame: pd.DataFrame) -> int:
    inject_file = run_folder / "inject_time.txt"
    if not inject_file.exists():
        return len(frame) // 2

    inject_time = inject_file.read_text().strip()
    timestamps = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    valid_times = timestamps.dropna()
    if valid_times.empty:
        return len(frame) // 2

    target = int(inject_time)
    indices = valid_times[valid_times >= target].index
    if len(indices) == 0:
        return len(frame) - 1
    return int(indices[0])


def _select_window(frame: pd.DataFrame, start: int, size: int) -> pd.DataFrame:
    if len(frame) <= size:
        return frame.reset_index(drop=True)

    max_start = max(len(frame) - size, 0)
    bounded_start = min(max(start, 0), max_start)
    return frame.iloc[bounded_start : bounded_start + size].reset_index(drop=True)


def _assign_data_split(*parts: str) -> str:
    # Deterministic hash-based split (not cryptographic — sha1 is fine for bucketing).
    key = "::".join(parts)
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "Train" if bucket < 8 else "Test"


def _build_metric_rows(incident_id: str, frame: pd.DataFrame) -> list[dict]:
    cpu_cols = [column for column in frame.columns if column.endswith("_cpu")]
    mem_cols = [column for column in frame.columns if column.endswith("_mem")]
    latency_cols = [column for column in frame.columns if "_latency-90" in column]
    if not latency_cols:
        latency_cols = [column for column in frame.columns if "_latency" in column]
    error_cols = [column for column in frame.columns if column.endswith("_error")]
    workload_cols = [column for column in frame.columns if column.endswith("_workload")]
    auth_error_cols = [
        column
        for column in error_cols
        if "auth" in column or "user" in column or "verification" in column
    ]

    def mean_series(columns: list[str], scale: float = 1.0) -> pd.Series:
        if not columns:
            return pd.Series(0.0, index=frame.index, dtype=float)
        numeric_frame = frame[columns].apply(pd.to_numeric, errors="coerce")
        return numeric_frame.mean(axis=1).fillna(0.0) * scale

    metric_frame = pd.DataFrame(
        {
            "incident_id": incident_id,
            "minute": range(len(frame)),
            "cpu_pct": mean_series(cpu_cols),
            "memory_pct": mean_series(mem_cols, scale=1 / (1024**3)),
            "latency_ms": mean_series(latency_cols, scale=1000.0),
            "error_rate": mean_series(error_cols),
            "queue_depth": mean_series(workload_cols),
            "auth_error_rate": mean_series(auth_error_cols),
        }
    )
    return metric_frame.to_dict(orient="records")


def _append_incident(
    all_incidents: list[dict],
    all_metrics: list[dict],
    incident_counter: int,
    system_name: str,
    frame: pd.DataFrame,
    *,
    fault_type: str,
    root_cause_service: str,
    is_anomalous: bool,
    data_split: str,
) -> int:
    incident_id = f"INC-{incident_counter:05d}"
    all_incidents.append(
        {
            "incident_id": incident_id,
            "title": f"Telemetry window from {system_name}",
            "description": f"Processed multivariate RCAEval metrics segment from {system_name}.",
            "fault_type": fault_type,
            "root_cause_service": root_cause_service,
            "region": "unknown",
            "is_anomalous": is_anomalous,
            "data_split": data_split,
        }
    )
    all_metrics.extend(_build_metric_rows(incident_id, frame))
    return incident_counter + 1


def convert_data(root_path: str = "data/raw", output_dir: str = "data/processed") -> None:
    """
    Convert RCAEval runs into incident-level CSV files used by the training pipeline.

    For each raw run, the converter emits:
    - one normal window taken before the injected fault
    - one anomalous window centered around the injection point
    """
    all_metrics: list[dict] = []
    all_incidents: list[dict] = []
    incident_counter = 1

    root = Path(root_path)
    os.makedirs(output_dir, exist_ok=True)

    for top_folder in sorted(root.iterdir()):
        if not top_folder.is_dir() or top_folder.name not in FILE_MAP:
            continue

        target_filename = FILE_MAP[top_folder.name]
        logger.info("Processing top-level folder: %s", top_folder.name)

        for fault_folder in sorted(top_folder.iterdir()):
            if not fault_folder.is_dir():
                continue

            fault_type, service_name = _extract_labels(fault_folder.name)
            if fault_type is None or service_name is None:
                continue

            logger.info("Processing folder: %s -> service=%s, fault=%s", fault_folder.name, service_name, fault_type)

            for run_folder in sorted(fault_folder.iterdir()):
                if not run_folder.is_dir():
                    continue

                target_path = run_folder / target_filename
                if not target_path.exists():
                    continue

                frame = pd.read_csv(target_path)
                inject_idx = _get_injection_index(run_folder, frame)
                data_split = _assign_data_split(
                    top_folder.name,
                    fault_folder.name,
                    run_folder.name,
                )

                normal_end = max(inject_idx - NORMAL_WINDOW_GAP, WINDOW_SIZE)
                normal_start = normal_end - WINDOW_SIZE
                normal_window = _select_window(frame, normal_start, WINDOW_SIZE)

                anomalous_start = inject_idx - PRE_FAULT_CONTEXT
                anomalous_window = _select_window(frame, anomalous_start, WINDOW_SIZE)

                incident_counter = _append_incident(
                    all_incidents,
                    all_metrics,
                    incident_counter,
                    top_folder.name,
                    normal_window,
                    fault_type="healthy",
                    root_cause_service="none",
                    is_anomalous=False,
                    data_split=data_split,
                )
                incident_counter = _append_incident(
                    all_incidents,
                    all_metrics,
                    incident_counter,
                    top_folder.name,
                    anomalous_window,
                    fault_type=fault_type,
                    root_cause_service=service_name,
                    is_anomalous=True,
                    data_split=data_split,
                )

    pd.DataFrame(all_incidents).to_csv(f"{output_dir}/incidents.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(f"{output_dir}/metrics.csv", index=False)
    logger.info("Finished! Compiled %d incident windows.", incident_counter - 1)


if __name__ == "__main__":
    convert_data()
