from __future__ import annotations

import csv
import io
import sqlite3
import threading
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from fault_lab.common.config import SCENARIO_PRESETS, SERVICE_FAULTS, TELEMETRY_DB_PATH


app = FastAPI(title="Fault Lab Control Plane")
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
                auth_error INTEGER NOT NULL
            );
            """
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


def fault_state() -> dict:
    with db_lock:
        conn = get_conn()
        rows = conn.execute(
            "SELECT service, fault, enabled, intensity FROM faults ORDER BY service, fault"
        ).fetchall()
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
        conn.close()
    return [dict(row) for row in rows]


def build_window(limit: int = 120) -> list[dict]:
    with db_lock:
        conn = get_conn()
        rows = conn.execute(
            """
            SELECT created_at, latency_ms, cpu_pct, memory_mb, queue_depth, error, auth_error
            FROM telemetry_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()

    event_rows = list(reversed(rows))
    baseline_count = max(0, limit - len(event_rows))

    window = []
    for bucket_index in range(baseline_count):
        window.append(
            {
                "minute": bucket_index,
                "error_rate": 0.0,
                "latency_ms": 18.0,
                "cpu_pct": 0.9,
                "memory_pct": 0.03,
                "queue_depth": 0.0,
                "auth_error_rate": 0.0,
            }
        )

    for offset, event in enumerate(event_rows, start=baseline_count):
        window.append(
            {
                "minute": offset,
                "error_rate": 6.5 if int(event["error"]) else 0.0,
                "latency_ms": min(3000.0, max(8.0, float(event["latency_ms"]))),
                "cpu_pct": min(16.0, max(0.7, float(event["cpu_pct"]))),
                "memory_pct": min(0.19, max(0.03, float(event["memory_mb"]) / 1024.0)),
                "queue_depth": min(320.0, max(0.0, float(event["queue_depth"]) * 14.0)),
                "auth_error_rate": 0.75 if int(event["auth_error"]) else 0.0,
            }
        )
    return window


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/faults")
def get_faults() -> dict:
    return {
        "scenarios": list(SCENARIO_PRESETS.keys()),
        "faults": fault_state(),
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
        apply_scenario(conn, request.scenario)
        conn.commit()
        conn.close()
    return {"ok": True, "scenario": request.scenario}


@app.post("/api/faults/toggle")
def toggle_fault(request: FaultToggleRequest) -> dict:
    with db_lock:
        conn = get_conn()
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
        conn.close()
    return {"ok": True}


@app.post("/api/telemetry/events")
def ingest_telemetry(event: TelemetryEvent) -> dict:
    with db_lock:
        conn = get_conn()
        conn.execute(
            """
            INSERT INTO telemetry_events (
                created_at, service, path, status_code, latency_ms, cpu_pct,
                memory_mb, queue_depth, error, auth_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()
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
    non_empty_rows = [row for row in window if row["latency_ms"] > 18.0 or row["error_rate"] > 0.0]
    avg_latency = sum(row["latency_ms"] for row in non_empty_rows) / len(non_empty_rows) if non_empty_rows else 0.0
    avg_cpu = sum(row["cpu_pct"] for row in non_empty_rows) / len(non_empty_rows) if non_empty_rows else 0.0
    error_sum = sum(row["error_rate"] for row in window)
    return {
        "active_faults": active_faults,
        "request_buckets": len(non_empty_rows),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_cpu_pct": round(avg_cpu, 2),
        "window_error_score": round(error_sum, 2),
        "window": window[-30:],
        "recent_events": events,
    }


@app.get("/api/telemetry/window")
def telemetry_window(limit: int = 120) -> dict:
    return {"rows": build_window(limit=limit)}


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
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


@app.post("/api/telemetry/reset")
def reset_telemetry() -> dict:
    with db_lock:
        conn = get_conn()
        conn.execute("DELETE FROM telemetry_events")
        conn.commit()
        conn.close()
    return {"ok": True}
