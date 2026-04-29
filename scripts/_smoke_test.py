"""Quick sanity check: point the trained stack at each fixture CSV and see if
the predicted fault type still matches what we baked into the file. Handy
after retraining or when you tweak features.

The fixtures come from training windows, and the per-service delta scalars
live on incidents.csv, so we look them up by fault family when we run.
That way the smoke test exercises the same inference shape the live
fault-lab path uses (control plane -> live_telemetry -> triage_custom_metrics)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PRECOMPUTED_SCALAR_FEATURES
from src.triage import triage_custom_metrics

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "test"
INCIDENTS = ROOT / "data" / "processed" / "incidents.csv"

EXPECTED = {
    "test_cpu_exhaustion.csv": "cpu",
    "test_memory_leak.csv": "mem",
    "test_cascading_failure.csv": "delay",
    "test_auth_failure.csv": "loss",
    "test_queue_congestion.csv": "socket",
}


def _scalars_for_fault(incidents: pd.DataFrame, fault: str) -> dict[str, float]:
    if fault not in incidents["fault_type"].unique():
        return {}
    candidate = incidents[
        (incidents["fault_type"] == fault) & (incidents["data_split"] == "Train")
    ]
    if candidate.empty:
        return {}
    row = candidate.iloc[0]
    return {
        column: float(row[column])
        for column in PRECOMPUTED_SCALAR_FEATURES
        if column in row.index and pd.notna(row[column])
    }


def main() -> None:
    incidents = pd.read_csv(INCIDENTS)
    passed = failed = 0
    for filename, expected in EXPECTED.items():
        path = TEST_DIR / filename
        df = pd.read_csv(path)
        scalars = _scalars_for_fault(incidents, expected)
        result = triage_custom_metrics(df, precomputed_scalars=scalars or None)
        pred = result["predicted_fault_type"]
        prob = result["fault_confidence"]
        anomaly = result["unusual"]
        rc = result["predicted_root_cause_service"]
        status = "OK " if pred == expected else "MISS"
        if pred == expected:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] {filename:32s} expected={expected:<6s} "
              f"pred={pred:<6s} p={prob:.2f} anomaly={anomaly} rc={rc}")
    print(f"\n{passed} passed / {failed} failed out of {passed + failed}")


if __name__ == "__main__":
    main()
