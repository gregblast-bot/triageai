from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


PAGES = [
    "Home",
    "Search",
    "Product",
    "Cart",
    "Login",
    "Checkout",
    "Admin",
]

SCENARIO_DESCRIPTIONS = {
    "healthy": "Normal traffic with no deliberate fault injection.",
    "cpu_exhaustion": "A hot service starts saturating CPU as user traffic and expensive requests accumulate.",
    "memory_leak": "A memory leak grows over time and eventually drags latency and errors upward.",
    "queue_congestion": "Background workers fall behind and request queues build up across checkout-like flows.",
    "auth_failure": "Login and session validation calls begin failing while auth-related errors rise.",
    "dependency_outage": "A downstream dependency becomes slow and flaky, causing retries and elevated errors.",
    "cascading_failure": "A downstream problem spills into queue growth, resource pressure, and user-facing latency.",
}

TRAFFIC_LEVELS = {
    "Normal": 1.0,
    "Busy": 1.25,
    "Flash Sale": 1.55,
}

PAGE_BASELINES = {
    "Home": {"cpu": 2.1, "memory": 0.052, "latency": 58.0, "queue": 2.0},
    "Search": {"cpu": 2.8, "memory": 0.056, "latency": 78.0, "queue": 3.0},
    "Product": {"cpu": 2.4, "memory": 0.055, "latency": 68.0, "queue": 2.5},
    "Cart": {"cpu": 3.0, "memory": 0.060, "latency": 85.0, "queue": 4.0},
    "Login": {"cpu": 2.5, "memory": 0.054, "latency": 70.0, "queue": 3.0},
    "Checkout": {"cpu": 3.4, "memory": 0.064, "latency": 96.0, "queue": 5.0},
    "Admin": {"cpu": 3.1, "memory": 0.066, "latency": 90.0, "queue": 4.5},
}

ACTION_LOAD = {
    "navigate": 0.7,
    "background": 0.45,
    "search": 1.25,
    "open_product": 0.8,
    "add_to_cart": 1.0,
    "remove_from_cart": 0.9,
    "login_attempt": 1.1,
    "checkout_submit": 1.45,
    "coupon": 0.75,
    "inventory_refresh": 1.1,
    "admin_report": 1.0,
}

ACTION_LABELS = {
    "navigate": "Viewed page",
    "background": "Background traffic",
    "search": "Executed search query",
    "open_product": "Opened product details",
    "add_to_cart": "Added item to cart",
    "remove_from_cart": "Removed item from cart",
    "login_attempt": "Attempted login",
    "checkout_submit": "Submitted checkout",
    "coupon": "Applied coupon code",
    "inventory_refresh": "Refreshed inventory status",
    "admin_report": "Opened admin analytics report",
}


def default_simulation_state() -> dict:
    return {
        "tick": 0,
        "current_page": "Home",
        "scenario": "healthy",
        "traffic_level": "Normal",
        "history": [],
        "event_log": [],
        "cart_items": 0,
        "cpu_pressure": 0.0,
        "memory_drift": 0.0,
        "queue_backlog": 0.0,
        "auth_fail_level": 0.0,
        "dependency_pressure": 0.0,
        "last_navigation_page": "Home",
    }


def _rng_for_tick(state: dict, action: str, page: str) -> np.random.Generator:
    seed = state["tick"] * 101 + len(action) * 17 + len(page) * 13
    return np.random.default_rng(seed)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _base_metrics(state: dict, page: str, action: str) -> dict:
    traffic_multiplier = TRAFFIC_LEVELS[state["traffic_level"]]
    rng = _rng_for_tick(state, action, page)
    baseline = PAGE_BASELINES[page]
    load = ACTION_LOAD[action] * traffic_multiplier

    cpu = baseline["cpu"] + load * 0.9 + rng.normal(0.0, 0.18)
    memory = baseline["memory"] + 0.0015 * state["cart_items"] + rng.normal(0.0, 0.0014)
    latency = baseline["latency"] + load * 13.0 + rng.normal(0.0, 6.0)
    queue = baseline["queue"] + load * 3.2 + rng.normal(0.0, 0.9)
    error_rate = max(0.0, rng.normal(0.008, 0.006))
    auth_error_rate = 0.0

    if action == "search":
        latency += 22.0
        queue += 3.0
        cpu += 0.5
    elif action == "open_product":
        latency += 10.0
    elif action == "add_to_cart":
        state["cart_items"] += 1
        queue += 2.0
        cpu += 0.6
        memory += 0.002
    elif action == "remove_from_cart":
        state["cart_items"] = max(0, state["cart_items"] - 1)
        queue += 1.0
        cpu += 0.25
    elif action == "login_attempt":
        latency += 18.0
        queue += 2.5
        auth_error_rate += 0.02
    elif action == "checkout_submit":
        latency += 45.0
        queue += 6.0
        cpu += 0.9
        memory += 0.0035
    elif action == "coupon":
        latency += 15.0
        queue += 2.0
    elif action == "inventory_refresh":
        cpu += 0.8
        latency += 18.0
    elif action == "admin_report":
        cpu += 1.2
        latency += 28.0

    return {
        "cpu_pct": cpu,
        "memory_pct": memory,
        "latency_ms": latency,
        "queue_depth": queue,
        "error_rate": error_rate,
        "auth_error_rate": auth_error_rate,
    }


def _apply_healthy_decay(state: dict) -> None:
    state["cpu_pressure"] = max(0.0, state["cpu_pressure"] * 0.55 - 0.15)
    state["queue_backlog"] = max(0.0, state["queue_backlog"] * 0.45 - 2.0)
    state["memory_drift"] = max(0.0, state["memory_drift"] * 0.88 - 0.0005)
    state["auth_fail_level"] = max(0.0, state["auth_fail_level"] * 0.65 - 0.02)
    state["dependency_pressure"] = max(0.0, state["dependency_pressure"] * 0.55 - 0.05)


def _overlay_scenario(
    state: dict,
    metrics: dict,
    *,
    page: str,
    action: str,
) -> None:
    scenario = state["scenario"]
    traffic_multiplier = TRAFFIC_LEVELS[state["traffic_level"]]
    action_load = ACTION_LOAD[action] * traffic_multiplier
    rng = _rng_for_tick(state, action, f"{page}:{scenario}")

    if scenario == "healthy":
        _apply_healthy_decay(state)
        return

    if scenario == "cpu_exhaustion":
        increment = 0.55 + 0.35 * action_load
        if page in {"Search", "Checkout", "Admin"}:
            increment += 0.45
        state["cpu_pressure"] = min(11.5, state["cpu_pressure"] + increment)
        metrics["cpu_pct"] += state["cpu_pressure"]
        metrics["latency_ms"] += max(0.0, state["cpu_pressure"] - 2.5) * 16.0
        metrics["queue_depth"] += max(0.0, state["cpu_pressure"] - 4.0) * 2.2
        metrics["error_rate"] += max(0.0, state["cpu_pressure"] - 8.0) * 0.08
        state["queue_backlog"] = max(0.0, state["queue_backlog"] * 0.7)
        state["memory_drift"] = max(0.0, state["memory_drift"] * 0.9)
        return

    if scenario == "memory_leak":
        increment = 0.003 + 0.0015 * action_load
        if page in {"Cart", "Checkout", "Admin"}:
            increment += 0.001
        state["memory_drift"] = min(0.115, state["memory_drift"] + increment)
        metrics["memory_pct"] += state["memory_drift"]
        metrics["latency_ms"] += max(0.0, state["memory_drift"] - 0.035) * 2200.0
        metrics["error_rate"] += max(0.0, state["memory_drift"] - 0.055) * 18.0
        metrics["cpu_pct"] += max(0.0, state["memory_drift"] - 0.05) * 10.0
        return

    if scenario == "queue_congestion":
        increment = 5.0 + action_load * 9.0
        if page in {"Checkout", "Search"}:
            increment += 12.0
        state["queue_backlog"] = min(280.0, state["queue_backlog"] * 0.92 + increment)
        metrics["queue_depth"] += state["queue_backlog"]
        metrics["latency_ms"] += min(1800.0, state["queue_backlog"] * 4.6)
        metrics["error_rate"] += max(0.0, state["queue_backlog"] - 115.0) * 0.02
        metrics["cpu_pct"] += min(5.0, state["queue_backlog"] / 70.0)
        return

    if scenario == "auth_failure":
        if page == "Login" or action in {"login_attempt", "checkout_submit"}:
            state["auth_fail_level"] = min(1.9, state["auth_fail_level"] + 0.28)
            metrics["auth_error_rate"] += state["auth_fail_level"]
            metrics["error_rate"] += 0.35 + state["auth_fail_level"] * 0.9
            metrics["latency_ms"] += 65.0 + state["auth_fail_level"] * 115.0
            metrics["queue_depth"] += 8.0 + state["auth_fail_level"] * 9.0
        else:
            state["auth_fail_level"] = max(0.08, state["auth_fail_level"] * 0.9)
            metrics["auth_error_rate"] += state["auth_fail_level"] * 0.1
        return

    if scenario == "dependency_outage":
        state["dependency_pressure"] = min(1.0, max(0.35, state["dependency_pressure"] + 0.08))
        metrics["latency_ms"] += 140.0 + 90.0 * action_load + rng.uniform(10.0, 140.0)
        metrics["error_rate"] += 0.28 + 0.35 * action_load
        metrics["queue_depth"] += 14.0 + 5.0 * action_load
        metrics["cpu_pct"] += 1.1 + rng.uniform(0.0, 0.6)
        if page in {"Checkout", "Search"}:
            metrics["latency_ms"] += 110.0
            metrics["error_rate"] += 0.35
        return

    if scenario == "cascading_failure":
        state["dependency_pressure"] = min(1.0, max(0.45, state["dependency_pressure"] + 0.12))
        state["queue_backlog"] = min(300.0, state["queue_backlog"] * 0.9 + 10.0 + 12.0 * action_load)
        metrics["queue_depth"] += state["queue_backlog"]
        metrics["latency_ms"] += 180.0 + min(1300.0, state["queue_backlog"] * 3.8)
        metrics["error_rate"] += 0.22 + max(0.0, state["queue_backlog"] - 100.0) * 0.015
        metrics["cpu_pct"] += 2.0 + min(6.0, state["queue_backlog"] / 90.0)
        if page == "Login":
            metrics["auth_error_rate"] += 0.08
        return


def _finalize_metrics(metrics: dict) -> dict:
    return {
        "error_rate": _clip(metrics["error_rate"], 0.0, 50.0),
        "latency_ms": _clip(metrics["latency_ms"], 7.0, 3000.0),
        "cpu_pct": _clip(metrics["cpu_pct"], 0.7, 16.0),
        "memory_pct": _clip(metrics["memory_pct"], 0.03, 0.19),
        "queue_depth": _clip(metrics["queue_depth"], 0.0, 320.0),
        "auth_error_rate": _clip(metrics["auth_error_rate"], 0.0, 2.5),
    }


def record_event(state: dict, *, page: str, action: str, note: str | None = None) -> dict:
    state["current_page"] = page
    metrics = _base_metrics(state, page, action)
    _overlay_scenario(state, metrics, page=page, action=action)
    final_metrics = _finalize_metrics(metrics)
    record = {
        "minute": state["tick"],
        **final_metrics,
    }
    state["history"].append(record)
    state["event_log"].append(
        {
            "minute": state["tick"],
            "page": page,
            "action": action,
            "label": ACTION_LABELS[action],
            "scenario": state["scenario"],
            "note": note or ACTION_LABELS[action],
        }
    )
    state["tick"] += 1
    return record


def warm_start_state(state: dict, samples: int = 20) -> None:
    if state["history"]:
        return
    for _ in range(samples):
        record_event(state, page="Home", action="background", note="Warm-up traffic")


def set_scenario(state: dict, scenario: str) -> None:
    if scenario == state["scenario"]:
        return
    state["scenario"] = scenario
    state["cpu_pressure"] = 0.0
    state["memory_drift"] = 0.0
    state["queue_backlog"] = 0.0
    state["auth_fail_level"] = 0.0
    state["dependency_pressure"] = 0.0


def set_traffic_level(state: dict, traffic_level: str) -> None:
    state["traffic_level"] = traffic_level


def reset_state(state: dict) -> None:
    fresh_state = default_simulation_state()
    state.clear()
    state.update(fresh_state)
    warm_start_state(state)


def history_frame(state: dict, limit: int | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(state["history"])
    if limit is not None and len(frame) > limit:
        return frame.tail(limit).reset_index(drop=True)
    return frame


def event_log_frame(state: dict, limit: int = 20) -> pd.DataFrame:
    frame = pd.DataFrame(state["event_log"])
    if len(frame) > limit:
        return frame.tail(limit).reset_index(drop=True)
    return frame


def simulate_background_traffic(state: dict, count: int, page: str | None = None) -> None:
    chosen_page = page or state["current_page"]
    for _ in range(count):
        record_event(state, page=chosen_page, action="background", note="Background request")


def metrics_csv_bytes(state: dict, limit: int = 120) -> bytes:
    return history_frame(state, limit=limit).to_csv(index=False).encode("utf-8")
