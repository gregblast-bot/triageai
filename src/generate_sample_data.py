from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import INCIDENTS_PATH, METRICS_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR


@dataclass(frozen=True)
class FaultProfile:
    fault_type: str
    root_cause_service: str
    description_template: str
    title_template: str
    anomalous: bool = True


FAULT_PROFILES = [
    FaultProfile(
        fault_type="healthy",
        root_cause_service="none",
        title_template="Stable production traffic",
        description_template="All core services are within normal operating range with no major alert spikes.",
        anomalous=False,
    ),
    FaultProfile(
        fault_type="database_overload",
        root_cause_service="orders-db",
        title_template="Latency increase in order processing",
        description_template="Order APIs show rising latency and queue depth, suggesting database pressure in the ordering path.",
    ),
    FaultProfile(
        fault_type="dependency_failure",
        root_cause_service="payment-service",
        title_template="Downstream dependency timeout",
        description_template="Checkout requests are timing out with elevated upstream errors after calls to an external dependency fail.",
    ),
    FaultProfile(
        fault_type="auth_failure",
        root_cause_service="auth-service",
        title_template="Authentication errors across login flow",
        description_template="Users report login failures while auth-related errors rise sharply and successful sessions drop.",
    ),
    FaultProfile(
        fault_type="memory_leak",
        root_cause_service="session-service",
        title_template="Steady memory growth before service instability",
        description_template="Memory usage increases over time until the session service becomes unstable and request failures appear.",
    ),
    FaultProfile(
        fault_type="latency_regression",
        root_cause_service="frontend",
        title_template="Sustained response-time regression",
        description_template="Page latency remains elevated after a recent change even though CPU is only moderately affected.",
    ),
]


def _build_metric_rows(
    incident_id: str,
    fault_profile: FaultProfile,
    rng: np.random.Generator,
    minutes: int = 60,
) -> list[dict]:
    minute_index = np.arange(minutes)

    error_rate = np.clip(rng.normal(0.4, 0.08, minutes), 0, None)
    latency_ms = np.clip(rng.normal(180, 20, minutes), 50, None)
    cpu_pct = np.clip(rng.normal(42, 6, minutes), 5, 100)
    memory_pct = np.clip(rng.normal(48, 5, minutes), 5, 100)
    queue_depth = np.clip(rng.normal(6, 2, minutes), 0, None)
    auth_error_rate = np.clip(rng.normal(0.1, 0.04, minutes), 0, None)

    if fault_profile.fault_type == "database_overload":
        ramp = np.linspace(0, 1, minutes)
        latency_ms += 180 * ramp + rng.normal(0, 8, minutes)
        queue_depth += 20 * ramp
        error_rate += 2.3 * np.clip(ramp - 0.4, 0, None)
        cpu_pct += 8 * ramp
    elif fault_profile.fault_type == "dependency_failure":
        spike = minute_index >= 35
        error_rate[spike] += rng.normal(5.5, 0.8, spike.sum())
        latency_ms[spike] += rng.normal(260, 35, spike.sum())
        queue_depth[spike] += rng.normal(12, 3, spike.sum())
    elif fault_profile.fault_type == "auth_failure":
        spike = minute_index >= 28
        auth_error_rate[spike] += rng.normal(6.0, 1.0, spike.sum())
        error_rate[spike] += rng.normal(3.5, 0.7, spike.sum())
        latency_ms[spike] += rng.normal(70, 12, spike.sum())
    elif fault_profile.fault_type == "memory_leak":
        ramp = np.linspace(0, 1.2, minutes)
        memory_pct += 35 * ramp
        error_rate += 1.8 * np.clip(ramp - 0.75, 0, None)
        latency_ms += 110 * np.clip(ramp - 0.45, 0, None)
        cpu_pct += 10 * np.clip(ramp - 0.5, 0, None)
    elif fault_profile.fault_type == "latency_regression":
        shift = minute_index >= 20
        latency_ms[shift] += rng.normal(140, 15, shift.sum())
        error_rate[shift] += rng.normal(0.7, 0.18, shift.sum())
        cpu_pct[shift] += rng.normal(6, 1.5, shift.sum())

    return [
        {
            "incident_id": incident_id,
            "minute": int(minute),
            "error_rate": round(float(error_rate[minute]), 4),
            "latency_ms": round(float(latency_ms[minute]), 4),
            "cpu_pct": round(float(cpu_pct[minute]), 4),
            "memory_pct": round(float(memory_pct[minute]), 4),
            "queue_depth": round(float(queue_depth[minute]), 4),
            "auth_error_rate": round(float(auth_error_rate[minute]), 4),
        }
        for minute in minute_index
    ]


def generate_sample_dataset(
    incident_count: int = 180,
    minutes_per_incident: int = 60,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    incident_rows: list[dict] = []
    metric_rows: list[dict] = []

    profiles = list(FAULT_PROFILES)
    weights = np.array([0.18, 0.18, 0.16, 0.16, 0.16, 0.16])

    for index in range(incident_count):
        profile = profiles[int(rng.choice(len(profiles), p=weights))]
        incident_id = f"INC-{index + 1:04d}"
        region = rng.choice(["us-east-1", "us-west-2", "eu-central-1"])

        incident_rows.append(
            {
                "incident_id": incident_id,
                "title": profile.title_template,
                "description": f"{profile.description_template} Region={region}.",
                "fault_type": profile.fault_type,
                "root_cause_service": profile.root_cause_service,
                "region": region,
                "is_anomalous": profile.anomalous,
            }
        )
        metric_rows.extend(
            _build_metric_rows(
                incident_id=incident_id,
                fault_profile=profile,
                rng=rng,
                minutes=minutes_per_incident,
            )
        )

    incidents = pd.DataFrame(incident_rows)
    metrics = pd.DataFrame(metric_rows)
    return incidents, metrics


def ensure_sample_data() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if INCIDENTS_PATH.exists() and METRICS_PATH.exists():
        return

    incidents, metrics = generate_sample_dataset()
    incidents.to_csv(INCIDENTS_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)


if __name__ == "__main__":
    ensure_sample_data()
