from __future__ import annotations

import csv
import io
import sqlite3
import statistics
import threading
from collections.abc import AsyncGenerator
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import psutil
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from fault_lab.common.config import (
    SCENARIO_EXPECTATION,
    SCENARIO_PRESETS,
    SERVICE_FAULTS,
    TELEMETRY_BUCKET_SEC,
    TELEMETRY_DB_PATH,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()
    yield


app = FastAPI(title="Fault Lab Control Plane", lifespan=lifespan)
db_lock = threading.Lock()


class FaultToggleRequest(BaseModel):
    service: str
    fault: str
    enabled: bool
    intensity: float = 1.0


class ScenarioRequest(BaseModel):
    scenario: str


class TelemetryEvent(BaseModel):
    service: str
    path: str
    status_code: int
    latency_ms: float
    cpu_pct: float
    memory_mb: float
    queue_depth: float
    error: bool
    auth_error: bool = False
    # Extras the newer runtimes emit. Defaulting to zero keeps old services
    # that don't know about these fields working without changes.
    disk_io_bytes: float = 0.0
    socket_count: float = 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    TELEMETRY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TELEMETRY_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_lock:
        conn = get_conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS faults (
                    service TEXT NOT NULL,
                    fault TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    intensity REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (service, fault)
                );

                CREATE TABLE IF NOT EXISTS telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    service TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    cpu_pct REAL NOT NULL,
                    memory_mb REAL NOT NULL,
                    queue_depth REAL NOT NULL,
                    error INTEGER NOT NULL,
                    auth_error INTEGER NOT NULL,
                    disk_io_bytes REAL NOT NULL DEFAULT 0,
                    socket_count REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS scenario_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    scenario TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            # Add the new columns if the table was created before we added
            # them. This way an upgrade doesn't throw away existing demo data.
            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(telemetry_events)").fetchall()
            }
            for column_name, definition in (
                ("disk_io_bytes", "REAL NOT NULL DEFAULT 0"),
                ("socket_count", "REAL NOT NULL DEFAULT 0"),
            ):
                if column_name not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE telemetry_events ADD COLUMN {column_name} {definition}"
                    )
            conn.execute(
                """
                INSERT OR IGNORE INTO scenario_state (id, scenario, updated_at)
                VALUES (1, 'healthy', ?)
                """,
                (utc_now(),),
            )
            for service, faults in SERVICE_FAULTS.items():
                for fault in faults:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO faults (service, fault, enabled, intensity, updated_at)
                        VALUES (?, ?, 0, 0.0, ?)
                        """,
                        (service, fault, utc_now()),
                    )
            conn.commit()
        finally:
            conn.close()


def set_all_faults_healthy(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE faults SET enabled = 0, intensity = 0.0, updated_at = ?",
        (utc_now(),),
    )


def apply_scenario(conn: sqlite3.Connection, scenario: str) -> None:
    set_all_faults_healthy(conn)
    preset = SCENARIO_PRESETS.get(scenario, {})
    for service, faults in preset.items():
        for fault, intensity in faults.items():
            conn.execute(
                """
                UPDATE faults
                SET enabled = 1, intensity = ?, updated_at = ?
                WHERE service = ? AND fault = ?
                """,
                (float(intensity), utc_now(), service, fault),
            )
    conn.execute(
        """
        INSERT INTO scenario_state (id, scenario, updated_at)
        VALUES (1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET scenario = excluded.scenario,
                                      updated_at = excluded.updated_at
        """,
        (scenario, utc_now()),
    )


def current_scenario() -> str:
    with db_lock:
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT scenario FROM scenario_state WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return "healthy"
    return str(row["scenario"])


def fault_state() -> dict:
    with db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT service, fault, enabled, intensity FROM faults ORDER BY service, fault"
            ).fetchall()
        finally:
            conn.close()

    payload: dict[str, dict[str, dict[str, float | bool]]] = {}
    for row in rows:
        payload.setdefault(row["service"], {})
        payload[row["service"]][row["fault"]] = {
            "enabled": bool(row["enabled"]),
            "intensity": float(row["intensity"]),
            "description": SERVICE_FAULTS[row["service"]][row["fault"]],
        }
    return payload


def recent_events(limit: int = 40) -> list[dict]:
    with db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                """
                SELECT created_at, service, path, status_code, latency_ms, cpu_pct, memory_mb,
                       queue_depth, error, auth_error
                FROM telemetry_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    return [dict(row) for row in rows]


def _percentile(values: list[float], pct: float) -> float:
    """Tiny percentile helper. `statistics.quantiles` is a bit much for a
    single point and refuses to run with fewer than two samples, which we
    hit all the time on sparse buckets."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[rank])


def _per_service_means(bucket_rows: list[sqlite3.Row], field: str) -> list[float]:
    """Average `field` within each service for this bucket."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in bucket_rows:
        grouped[row["service"]].append(float(row[field]))
    return [sum(values) / len(values) for values in grouped.values() if values]


def _top_service_delta(bucket_rows: list[sqlite3.Row], field: str, scale: float = 1.0) -> float:
    """Gap between the hottest service and the cross-service mean. If there's
    only one service in the bucket there's no spread to talk about, so return
    zero."""
    means = _per_service_means(bucket_rows, field)
    if len(means) < 2:
        return 0.0
    peak = max(means)
    overall = sum(means) / len(means)
    return (peak - overall) * scale


def _aggregate_bucket(bucket_rows: list[sqlite3.Row]) -> dict:
    """Roll up raw events for one bucket. No arbitrary caps, just honest
    averages.

    CPU is already normalized when we emit, so we just pass it through. Memory
    gets turned into GB because that's what the model was trained on. Latency
    keeps p90 as the headline value (to match RCAEval) but we also expose p50
    so the classifier can see how much the tail is dragging. The per-service
    deltas are the "one service is hot" feature root-cause cares about."""
    count = len(bucket_rows)
    latencies = [float(row["latency_ms"]) for row in bucket_rows]
    latency_p90 = _percentile(latencies, 90.0)
    latency_p50 = _percentile(latencies, 50.0)
    cpu = sum(float(row["cpu_pct"]) for row in bucket_rows) / count
    memory_mb = sum(float(row["memory_mb"]) for row in bucket_rows) / count
    queue = max(float(row["queue_depth"]) for row in bucket_rows)
    error_count = sum(int(row["error"]) for row in bucket_rows)
    auth_errors = sum(int(row["auth_error"]) for row in bucket_rows)
    disk_io = sum(float(row["disk_io_bytes"]) for row in bucket_rows) / count
    sockets = sum(float(row["socket_count"]) for row in bucket_rows) / count
    # Load average is host-wide, not per-bucket, so we just read it once.
    try:
        load_avg = float(psutil.getloadavg()[0])
    except (OSError, AttributeError):
        load_avg = 0.0

    return {
        "error_rate": float(error_count),
        "latency_ms": float(latency_p90),
        "cpu_pct": float(cpu),
        "memory_pct": float(memory_mb / 1024.0),
        "queue_depth": float(queue),
        "auth_error_rate": float(auth_errors),
        "latency_p50_ms": float(latency_p50),
        "load_avg": float(load_avg),
        "disk_io": float(disk_io),
        "socket_count": float(sockets),
        "cpu_top_service_delta": _top_service_delta(bucket_rows, "cpu_pct"),
        "mem_top_service_delta": _top_service_delta(
            bucket_rows, "memory_mb", scale=1 / 1024.0
        ),
        "error_top_service_delta": _top_service_delta(bucket_rows, "error"),
        "latency_top_service_delta": _top_service_delta(bucket_rows, "latency_ms"),
    }


def _idle_bucket() -> dict:
    """All zeros—only there so the first bucket has something to compare against before real data shows up."""
    return {
        "error_rate": 0.0,
        "latency_ms": 0.0,
        "cpu_pct": 0.0,
        "memory_pct": 0.0,
        "queue_depth": 0.0,
        "auth_error_rate": 0.0,
        "latency_p50_ms": 0.0,
        "load_avg": 0.0,
        "disk_io": 0.0,
        "socket_count": 0.0,
        "cpu_top_service_delta": 0.0,
        "mem_top_service_delta": 0.0,
        "error_top_service_delta": 0.0,
        "latency_top_service_delta": 0.0,
    }


def build_window(limit: int = 120) -> list[dict]:
    """Recent history as a list of buckets ending at now—what TriageAI ingests as a window.

    Bucket width comes from TELEMETRY_BUCKET_SEC (60s by default), so 120 points
    is roughly the same two-hour feel as the training windows. Gaps repeat the
    last real bucket instead of snapping back to a fake idle line; the old idle
    backfill was making bursty traffic look like a bunch of bogus spikes."""
    bucket_sec = max(1.0, float(TELEMETRY_BUCKET_SEC))
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=bucket_sec * limit)
    with db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                """
                SELECT created_at, service, latency_ms, cpu_pct, memory_mb, queue_depth,
                       error, auth_error, disk_io_bytes, socket_count
                FROM telemetry_events
                WHERE created_at >= ?
                ORDER BY created_at ASC
                """,
                (cutoff.isoformat(),),
            ).fetchall()
        finally:
            conn.close()

    buckets: dict[int, list[sqlite3.Row]] = {}
    now = datetime.now(timezone.utc)
    for row in rows:
        created = datetime.fromisoformat(row["created_at"])
        delta_sec = (now - created).total_seconds()
        if delta_sec < 0 or delta_sec >= bucket_sec * limit:
            continue
        bucket_index = limit - 1 - int(delta_sec // bucket_sec)
        if 0 <= bucket_index < limit:
            buckets.setdefault(bucket_index, []).append(row)

    window: list[dict] = []
    last = _idle_bucket()
    for bucket_index in range(limit):
        bucket_rows = buckets.get(bucket_index, [])
        if bucket_rows:
            last = _aggregate_bucket(bucket_rows)
        entry = {"minute": bucket_index, **last}
        window.append(entry)
    return window


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/faults")
def get_faults() -> dict:
    active = current_scenario()
    return {
        "scenarios": list(SCENARIO_PRESETS.keys()),
        "faults": fault_state(),
        "active_scenario": active,
        "expected": SCENARIO_EXPECTATION.get(active, {}),
    }


@app.get("/api/faults/{service_name}")
def get_faults_for_service(service_name: str) -> dict:
    state = fault_state().get(service_name, {})
    flattened = {
        fault: item["intensity"]
        for fault, item in state.items()
        if item["enabled"]
    }
    return {
        "service": service_name,
        "faults": flattened,
    }


@app.post("/api/faults/scenario")
def set_scenario(request: ScenarioRequest) -> dict:
    if request.scenario not in SCENARIO_PRESETS:
        return {"ok": False, "detail": f"Unknown scenario: {request.scenario}"}
    with db_lock:
        conn = get_conn()
        try:
            apply_scenario(conn, request.scenario)
            conn.commit()
        finally:
            conn.close()
    return {"ok": True, "scenario": request.scenario}


@app.post("/api/faults/toggle")
def toggle_fault(request: FaultToggleRequest) -> dict:
    with db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                UPDATE faults
                SET enabled = ?, intensity = ?, updated_at = ?
                WHERE service = ? AND fault = ?
                """,
                (
                    1 if request.enabled else 0,
                    max(0.0, min(1.0, request.intensity if request.enabled else 0.0)),
                    utc_now(),
                    request.service,
                    request.fault,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return {"ok": True}


@app.post("/api/telemetry/events")
def ingest_telemetry(event: TelemetryEvent) -> dict:
    with db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO telemetry_events (
                    created_at, service, path, status_code, latency_ms, cpu_pct,
                    memory_mb, queue_depth, error, auth_error,
                    disk_io_bytes, socket_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    event.service,
                    event.path,
                    event.status_code,
                    float(event.latency_ms),
                    float(event.cpu_pct),
                    float(event.memory_mb),
                    float(event.queue_depth),
                    1 if event.error else 0,
                    1 if event.auth_error else 0,
                    float(event.disk_io_bytes),
                    float(event.socket_count),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return {"ok": True}


@app.get("/api/telemetry/summary")
def telemetry_summary(limit: int = 120) -> dict:
    window = build_window(limit=limit)
    events = recent_events(limit=20)
    active_faults = {
        service: {
            fault: item
            for fault, item in faults.items()
            if item["enabled"]
        }
        for service, faults in fault_state().items()
    }
    active_faults = {service: faults for service, faults in active_faults.items() if faults}
    non_empty_rows = [row for row in window if row["latency_ms"] > 0.0 or row["error_rate"] > 0.0]
    avg_latency = sum(row["latency_ms"] for row in non_empty_rows) / len(non_empty_rows) if non_empty_rows else 0.0
    avg_cpu = sum(row["cpu_pct"] for row in non_empty_rows) / len(non_empty_rows) if non_empty_rows else 0.0
    error_sum = sum(row["error_rate"] for row in window)
    scenario = current_scenario()
    return {
        "active_faults": active_faults,
        "active_scenario": scenario,
        "expected": SCENARIO_EXPECTATION.get(scenario, {}),
        "request_buckets": len(non_empty_rows),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_cpu_pct": round(avg_cpu, 2),
        "window_error_score": round(error_sum, 2),
        "window": window[-30:],
        "recent_events": events,
    }


@app.get("/api/telemetry/window")
def telemetry_window(limit: int = 120) -> dict:
    scenario = current_scenario()
    return {
        "rows": build_window(limit=limit),
        "active_scenario": scenario,
        "expected": SCENARIO_EXPECTATION.get(scenario, {}),
    }


@app.get("/api/telemetry/window.csv", response_class=PlainTextResponse)
def telemetry_window_csv(limit: int = 120) -> str:
    rows = build_window(limit=limit)
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "minute",
            "error_rate",
            "latency_ms",
            "cpu_pct",
            "memory_pct",
            "queue_depth",
            "auth_error_rate",
            "latency_p50_ms",
            "load_avg",
            "disk_io",
            "socket_count",
            "cpu_top_service_delta",
            "mem_top_service_delta",
            "error_top_service_delta",
            "latency_top_service_delta",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


@app.post("/api/telemetry/reset")
def reset_telemetry() -> dict:
    with db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM telemetry_events")
            conn.commit()
        finally:
            conn.close()
    return {"ok": True}


@app.get("/api/scenarios/current")
def get_current_scenario() -> dict:
    scenario = current_scenario()
    return {
        "active_scenario": scenario,
        "expected": SCENARIO_EXPECTATION.get(scenario, {}),
    }
