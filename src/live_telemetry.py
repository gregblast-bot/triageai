from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import METRIC_COLUMNS, OPTIONAL_METRIC_COLUMNS, PRECOMPUTED_SCALAR_FEATURES


def fetch_telemetry_window(
    control_plane_base_url: str,
    *,
    limit: int = 120,
    timeout_sec: float = 10.0,
) -> tuple[pd.DataFrame | None, str | None, dict]:
    """
    Fetch a rolling window of metrics from the fault-lab control plane (anything
    that speaks the same JSON shape works too).

    The response should look like GET {base}/api/telemetry/window?limit=N with
    rows plus optional active_scenario and expected labels so the UI can compare
    "what we injected" against "what the model said." Rows need the usual
    minute + METRIC_COLUMNS the rest of TriageAI expects.

    You get (dataframe or None, error string or None, meta dict). Meta is empty
    on failure; on success it holds whatever the server sent about scenario and
    expected fault—ignore it if you only care about the frame.
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
    # Newer control planes send these extras too. Default to zero if we're
    # talking to an older one so the feature shape stays consistent.
    for col in OPTIONAL_METRIC_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        else:
            frame[col] = 0.0
    frame["minute"] = pd.to_numeric(frame["minute"], errors="coerce").fillna(0).astype(int)

    # Per-service deltas come through as per-bucket scalars. Take the max
    # across the window before handing them off; the worst moment of "one
    # service is way hotter than the rest" is the bit the classifier needs.
    scalars: dict[str, float] = {}
    for col in PRECOMPUTED_SCALAR_FEATURES:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
            scalars[col] = float(frame[col].max())
    if scalars:
        meta["precomputed_scalars"] = scalars

    return frame, None, meta
