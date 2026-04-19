"""Quick sanity check: point the trained stack at each fixture CSV and see if
the predicted fault type still matches what we baked into the file. Handy after
retraining or when you tweak features."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.triage import triage_custom_metrics

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "test"

EXPECTED = {
    "test_cpu_exhaustion.csv": "cpu",
    "test_memory_leak.csv": "mem",
    "test_cascading_failure.csv": "delay",
    "test_auth_failure.csv": "loss",
    "test_queue_congestion.csv": "socket",
}


def main() -> None:
    passed = failed = 0
    for filename, expected in EXPECTED.items():
        path = TEST_DIR / filename
        df = pd.read_csv(path)
        result = triage_custom_metrics(df)
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
