"""Rebuild the test/*.csv fixtures from real RCAEval training windows.

We pick one incident per fault family from the Train split so the numbers sit
where the model actually learned—not hand-tuned toy scales that drift from
production-ish data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INCIDENTS = ROOT / "data" / "processed" / "incidents.csv"
METRICS = ROOT / "data" / "processed" / "metrics.csv"
OUT = ROOT / "test"

FAULT_TO_FILE = {
    "cpu": "test_cpu_exhaustion.csv",
    "mem": "test_memory_leak.csv",
    "delay": "test_cascading_failure.csv",
    "loss": "test_auth_failure.csv",
    "socket": "test_queue_congestion.csv",
}

COLUMNS = [
    "minute",
    "error_rate",
    "latency_ms",
    "cpu_pct",
    "memory_pct",
    "queue_depth",
    "auth_error_rate",
    # Newer columns. They're populated when the source collection had the
    # data, and zero otherwise. Keeping them in the fixture makes the smoke
    # test match what live fault-lab actually sends these days.
    "latency_p50_ms",
    "load_avg",
    "disk_io",
    "socket_count",
]


def main() -> None:
    incidents = pd.read_csv(INCIDENTS)
    metrics = pd.read_csv(METRICS)
    train = incidents[incidents["data_split"] == "Train"]
    for fault, filename in FAULT_TO_FILE.items():
        candidates = train[train["fault_type"] == fault]
        if candidates.empty:
            print(f"[skip] no {fault} windows in training split")
            continue
        iid = candidates.iloc[0]["incident_id"]
        window = metrics[metrics["incident_id"] == iid].copy()
        window = window.sort_values("minute").reset_index(drop=True)
        # Keep the column set stable even when the source CSV didn't have
        # one of the newer families. Anything missing falls back to zero.
        for column in COLUMNS:
            if column not in window.columns:
                window[column] = 0.0
        window = window[COLUMNS].round(4)
        out_path = OUT / filename
        window.to_csv(out_path, index=False)
        print(f"[write] {out_path.relative_to(ROOT)} ({len(window)} rows) "
              f"<- {iid} ({fault})")


if __name__ == "__main__":
    main()
