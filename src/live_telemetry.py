from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import METRIC_COLUMNS


def fetch_telemetry_window(
    control_plane_base_url: str,
    *,
    limit: int = 120,
    timeout_sec: float = 10.0,
) -> tuple[pd.DataFrame | None, str | None, dict]:
    """
    Pull the latest aggregated metric window from a fault-lab control plane (or compatible API).

    Expects GET {base}/api/telemetry/window?limit=N returning JSON:
        {"rows": [ {...}, ... ],
         "active_scenario": "...",
         "expected": {"fault_type": "...", "root_cause_service": "..."}}
    Each row must include minute and METRIC_COLUMNS fields compatible with TriageAI.

    Returns a triple of (frame_or_none, error_or_none, meta_dict). The meta
    dict carries scenario hints when the control plane returns them; callers
    that just want the old (frame, err) tuple can ignore it.
    """
    base = control_plane_base_url.strip().rstrip("/")
    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"http://{base}"

    query = urlencode({"limit": int(limit)})
    url = f"{base}/api/telemetry/window?{query}"

    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        return None, f"HTTP {exc.code} from control plane: {exc.reason}", {}
    except URLError as exc:
        return None, f"Could not reach control plane ({url}): {exc.reason}", {}
    except OSError as exc:
        return None, f"Network error: {exc}", {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON from control plane: {exc}", {}

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None, "Control plane response missing 'rows' list.", {}

    meta = {
        "active_scenario": payload.get("active_scenario"),
        "expected": payload.get("expected") or {},
    }

    if not rows:
        return None, "No telemetry rows yet. Generate traffic in the fault lab, then refresh.", meta

    frame = pd.DataFrame(rows)
    required = ["minute", *METRIC_COLUMNS]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return None, f"Window rows missing columns: {', '.join(missing)}", meta

    for col in METRIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["minute"] = pd.to_numeric(frame["minute"], errors="coerce").fillna(0).astype(int)

    return frame, None, meta
