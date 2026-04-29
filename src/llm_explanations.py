from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Optional
from urllib import error, request


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]
GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_KEY_CACHE_PATH = Path(".cache") / "triageai_gemini_key.json"
GEMINI_KEY_CACHE_TTL_SECONDS = 8 * 60 * 60


def load_cached_gemini_key() -> str:
    try:
        with open(GEMINI_KEY_CACHE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""

    created_at = float(payload.get("created_at", 0.0))
    if time.time() - created_at > GEMINI_KEY_CACHE_TTL_SECONDS:
        clear_cached_gemini_key()
        return ""
    return str(payload.get("api_key", "")).strip()


def save_cached_gemini_key(api_key: str) -> None:
    key = api_key.strip()
    if not key:
        return
    GEMINI_KEY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "api_key": key,
        "created_at": time.time(),
    }
    with open(GEMINI_KEY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    try:
        os.chmod(GEMINI_KEY_CACHE_PATH, 0o600)
    except OSError:
        pass


def clear_cached_gemini_key() -> None:
    try:
        GEMINI_KEY_CACHE_PATH.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _format_signal(signal: dict[str, Any]) -> str:
    return (
        f"{signal['metric']}: max={signal['max']:.2f}, mean={signal['mean']:.2f}, "
        f"delta={signal['delta']:.2f}, spike={signal['spike']:.2f}, slope={signal['slope']:.3f}"
    )


def _top_signal_lines(result: dict, limit: int = 5) -> list[str]:
    signals = result.get("signal_highlights") or []
    return [_format_signal(signal) for signal in signals[:limit]]


def _top_context_lines(result: dict, limit: int = 4) -> list[str]:
    retrieved = result.get("retrieved_context") or {}
    documents = retrieved.get("documents") or []
    lines = []
    for doc in documents[:limit]:
        content = " ".join(str(doc.get("content", "")).split())
        lines.append(
            f"{doc.get('title', 'Retrieved note')} ({doc.get('source_type', 'context')}): {content[:360]}"
        )
    return lines


def _confidence_note(fault_conf: float, root_conf: float) -> str:
    lower = min(fault_conf, root_conf)
    if lower >= 0.75:
        return "confidence is strong enough for a clear first triage path"
    if lower >= 0.5:
        return "confidence is moderate, so confirm against the graphs before acting"
    return "confidence is low, so treat this mainly as a hypothesis to investigate"


def build_local_explanation(result: dict) -> str:
    anomaly_label = "abnormal" if result.get("unusual") else "normal"
    fault = str(result.get("predicted_fault_type", "unknown"))
    service = str(result.get("predicted_root_cause_service", "unknown"))
    fault_normalized = fault.lower()
    service_normalized = service.lower()
    fault_conf = float(result.get("fault_confidence", 0.0))
    root_conf = float(result.get("root_cause_confidence", 0.0))
    anomaly_score = float(result.get("anomaly_score", 0.0))
    clean_healthy_result = (
        fault_normalized == "healthy"
        and service_normalized in {"none", "unknown", "unlabeled", ""}
        and not bool(result.get("unusual", False))
    )
    unmatched_anomaly = (
        fault_normalized == "healthy"
        and service_normalized in {"none", "unknown", "unlabeled", ""}
        and bool(result.get("unusual", False))
    )

    signal_lines = _top_signal_lines(result, limit=3)
    signal_text = "; ".join(signal_lines) if signal_lines else "no strong metric movement was available"

    if clean_healthy_result:
        assessment = (
            f"**Assessment:** This window is classified as normal with anomaly score {anomaly_score:.3f}. "
            f"The fault classifier also predicts `healthy` at {fault_conf:.1%} confidence, with no "
            f"root-cause service at {root_conf:.1%} confidence. This should be treated as a validation "
            "result rather than an incident."
        )
        context_section = (
            "**Supporting context:** Reference cases are hidden because this window is classified as healthy "
            "with no root-cause service. The useful action is validation, not case matching."
        )
        first_checks = (
            "**First checks:** Validate that latency, queue depth, error rate, and resource metrics remain "
            "within expected ranges. Continue monitoring for sustained movement before escalating to a "
            "service-specific investigation."
        )
    elif unmatched_anomaly:
        assessment = (
            f"**Assessment:** This window is classified as abnormal with anomaly score {anomaly_score:.3f}, "
            "but the supervised classifiers did not match the telemetry to a known fault family or root-cause "
            f"service. The `healthy`/`none` class output has {fault_conf:.1%}/{root_conf:.1%} confidence, "
            "so treat this as unusual telemetry that needs validation rather than as proof that the system "
            "is operating normally."
        )
        context_section = (
            "**Supporting context:** This is an unmatched anomaly: telemetry is unusual, but the classifier "
            "did not match it to a known fault class. Use the signal changes as the primary evidence."
        )
        first_checks = (
            "**First checks:** Review the top-changing metrics for persistence, compare against recent baseline "
            "behavior, and check for recent deploys or traffic shifts before escalating to a service-specific owner."
        )
    else:
        assessment = (
            f"**Assessment:** This window is classified as {anomaly_label} with anomaly score "
            f"{anomaly_score:.3f}. The leading fault assessment is `{fault}` at {fault_conf:.1%} "
            f"confidence, with `{service}` as the likely root-cause service at {root_conf:.1%} confidence. "
            f"Overall, {_confidence_note(fault_conf, root_conf)}."
        )
        retrieved = result.get("retrieved_context") or {}
        docs = retrieved.get("documents") or []
        if docs:
            context_text = "\n".join(
                f"- {doc.get('title', 'retrieved context')}: "
                f"{' '.join(str(doc.get('content', '')).split())[:220]}"
                for doc in docs[:3]
            )
        else:
            context_text = "no retrieved incident notes were available"
        context_section = f"**Supporting context:** {context_text}"
        first_checks = (
            "**First checks:** Start with the predicted service and fault family, compare the top-changing "
            "metrics to normal behavior, then use the retrieved notes as a checklist for the first manual "
            "triage actions."
        )

    return (
        f"{assessment}\n\n"
        f"**Telemetry evidence:** The highest-changing signals are {signal_text}. These are the metric movements "
        "used as support; they should be checked against the trend chart before treating the prediction "
        "as final.\n\n"
        f"{context_section}\n\n"
        f"{first_checks}"
    )


def build_llm_prompt(result: dict) -> str:
    expected = result.get("expected") or {}
    fault = str(result.get("predicted_fault_type", "")).lower()
    service = str(result.get("predicted_root_cause_service", "")).lower()
    clean_healthy_result = (
        fault == "healthy"
        and service in {"none", "", "unknown", "unlabeled"}
        and not bool(result.get("unusual", False))
    )
    unmatched_anomaly = (
        fault == "healthy"
        and service in {"none", "", "unknown", "unlabeled"}
        and bool(result.get("unusual", False))
    )
    lines = [
        "You are explaining an incident-triage model output to an on-call engineer.",
        "Use only the model output, signal highlights, and retrieved context below.",
        "Write a production-quality explanation for an operational triage UI. Aim for 5 compact sections.",
        "Do not preface the answer with meta text such as 'Here is an explanation' or discuss the interface itself.",
        "Do not invent metrics, services, incidents, commands, or root causes.",
        "If evidence is weak or confidence is low, say that directly.",
        "Start immediately with the Assessment section.",
        "If predicted_fault_type is healthy, predicted_root_cause_service is none, and anomaly is normal, frame first steps as validation and monitoring checks.",
        "If predicted_fault_type is healthy, predicted_root_cause_service is none, and anomaly is abnormal, describe it as unusual telemetry with no known fault pattern match.",
        "For healthy/none abnormal results, do not say the system is operating normally and do not recommend a specific service owner.",
        "",
        "Model output:",
        f"- anomaly: {'abnormal' if result.get('unusual') else 'normal'}",
        f"- anomaly_score: {float(result.get('anomaly_score', 0.0)):.3f}",
        f"- predicted_fault_type: {result.get('predicted_fault_type', 'unknown')}",
        f"- fault_confidence: {float(result.get('fault_confidence', 0.0)):.3f}",
        f"- predicted_root_cause_service: {result.get('predicted_root_cause_service', 'unknown')}",
        f"- root_cause_confidence: {float(result.get('root_cause_confidence', 0.0)):.3f}",
    ]
    if expected:
        lines.extend(
            [
                "",
                "Reference label when available:",
                f"- fault_type: {expected.get('fault_type', 'unknown')}",
                f"- root_cause_service: {expected.get('root_cause_service', 'unknown')}",
            ]
        )

    signal_lines = _top_signal_lines(result)
    if signal_lines:
        lines.append("")
        lines.append("Signal highlights:")
        lines.extend(f"- {line}" for line in signal_lines)

    context_lines = [] if clean_healthy_result else _top_context_lines(result)
    if context_lines:
        lines.append("")
        lines.append("Retrieved context:")
        lines.extend(f"- {line}" for line in context_lines)

    lines.extend(
        [
            "",
            "Write with these exact section labels:",
            "1. Assessment - one paragraph explaining the likely incident.",
            "2. Model confidence - fault confidence, root-cause confidence, and what that means.",
            "3. Telemetry evidence - reference the top signal highlights and explain why they matter.",
            "4. Supporting context - for clean healthy/none results, state that reference cases are intentionally not needed; for unmatched anomalies, state that signal changes are the primary evidence; otherwise explain how retrieved notes or similar incidents support the hypothesis.",
            "5. First triage steps - 3 to 5 concrete checks an engineer should do first.",
        ]
    )
    return "\n".join(lines)


def _extract_gemini_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return "\n".join(str(part.get("text", "")).strip() for part in parts if part.get("text")).strip()


def clean_explanation_text(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(
        r"^\s*(here'?s|here is)\s+(a|the)?\s*.*?explanation[:\-\s]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"^\s*explanation[:\-\s]+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    section_aliases = {
        "Assessment": "Assessment",
        "Model confidence": "Model confidence",
        "Telemetry evidence": "Telemetry evidence",
        "Retrieved context": "Supporting context",
        "Supporting context": "Supporting context",
        "First triage steps": "First triage steps",
    }
    first_section = None
    for section in section_aliases:
        match = re.search(rf"\b{re.escape(section)}\b", cleaned, flags=re.IGNORECASE)
        if match and (first_section is None or match.start() < first_section.start()):
            first_section = match
    if first_section:
        prefix = cleaned[: first_section.start()].strip()
        if prefix and len(prefix) < 240:
            prefix_lower = prefix.lower()
            if any(term in prefix_lower for term in ["explanation", "model output", "incident triage"]):
                cleaned = cleaned[first_section.start() :].strip()

    for section, canonical in section_aliases.items():
        escaped = re.escape(section)
        cleaned = re.sub(
            rf"(?im)^\s*(?:\d+\.\s*)?(?:\*\*)?{escaped}(?:\*\*)?\s*[:\-\u2013\u2014]?\s+",
            f"**{canonical}:** ",
            cleaned,
        )
        cleaned = re.sub(
            rf"(?im)(?<!\*)\b{escaped}\b(?:\s*[:\-\u2013\u2014]|\s+(?=[A-Z]))\s*",
            f"**{canonical}:** ",
            cleaned,
            count=1,
        )

    heading_pattern = (
        r"\*\*(Assessment|Model confidence|Telemetry evidence|Supporting context|First triage steps):\*\*"
    )
    cleaned = re.sub(rf"\s+({heading_pattern})", r"\n\n\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_retryable_gemini_error(message: str) -> bool:
    retryable_terms = [
        '"code": 503',
        '"status": "UNAVAILABLE"',
        "high demand",
        '"code": 429',
        '"status": "RESOURCE_EXHAUSTED"',
        "rate limit",
    ]
    lowered = message.lower()
    return any(term.lower() in lowered for term in retryable_terms)


def _model_attempts(model_name: str) -> list[str]:
    attempts = [model_name]
    for fallback in GEMINI_FALLBACK_MODELS:
        if fallback not in attempts:
            attempts.append(fallback)
    return attempts


def _resolve_ipv4(hostname: str) -> str:
    addresses = socket.getaddrinfo(hostname, 443, family=socket.AF_INET, proto=socket.IPPROTO_TCP)
    if not addresses:
        raise RuntimeError(f"No IPv4 address found for {hostname}.")
    return addresses[0][4][0]


def _is_wsl() -> bool:
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _prefer_windows_curl() -> bool:
    return _is_wsl() and shutil.which("curl.exe") is not None


def _call_gemini_with_windows_curl(url: str, key: str, body: bytes, timeout_sec: float) -> dict:
    curl_path = shutil.which("curl.exe")
    if not curl_path:
        raise RuntimeError("curl.exe is not available for Windows-network fallback.")

    host = "generativelanguage.googleapis.com"
    ip_address = _resolve_ipv4(host)
    completed = subprocess.run(
        [
            curl_path,
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--max-time",
            str(int(max(timeout_sec, 60.0))),
            "--resolve",
            f"{host}:443:{ip_address}",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-H",
            f"x-goog-api-key: {key}",
            "--data-binary",
            "@-",
            url,
        ],
        input=body,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        detail = stdout or stderr or f"exit code {completed.returncode}"
        raise RuntimeError(detail[:300])
    return json.loads(completed.stdout.decode("utf-8"))


def generate_gemini_explanation(
    result: dict,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: float = 25.0,
) -> dict:
    model_name = model or os.environ.get("TRIAGEAI_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return {
            "provider": "gemini",
            "ok": False,
            "model": model_name,
            "error": "Missing GEMINI_API_KEY.",
            "text": build_local_explanation(result),
        }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": build_llm_prompt(result),
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 850,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    errors = []

    for attempted_model in _model_attempts(model_name):
        url = GEMINI_API_URL_TEMPLATE.format(model=attempted_model)
        data = None

        if _prefer_windows_curl():
            try:
                data = _call_gemini_with_windows_curl(url, key, body, timeout_sec)
            except Exception as windows_exc:
                error_text = f"{attempted_model}: Windows curl fallback failed: {windows_exc}"
                errors.append(error_text)
                if _is_retryable_gemini_error(str(windows_exc)):
                    continue
                return {
                    "provider": "gemini",
                    "ok": False,
                    "model": attempted_model,
                    "error": error_text,
                    "text": build_local_explanation(result),
                }

        req = request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key,
            },
            method="POST",
        )
        if data is None:
            try:
                with request.urlopen(req, timeout=timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                error_text = f"{attempted_model}: Gemini returned HTTP {exc.code}: {detail[:240]}"
                errors.append(error_text)
                if _is_retryable_gemini_error(error_text):
                    continue
                return {
                    "provider": "gemini",
                    "ok": False,
                    "model": attempted_model,
                    "error": error_text,
                    "text": build_local_explanation(result),
                }
            except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                python_error = str(exc)
                try:
                    data = _call_gemini_with_windows_curl(url, key, body, timeout_sec)
                except Exception as windows_exc:
                    error_text = (
                        f"{attempted_model}: Python HTTPS failed: {python_error}. "
                        f"Windows curl fallback failed: {windows_exc}"
                    )
                    errors.append(error_text)
                    if _is_retryable_gemini_error(str(windows_exc)):
                        continue
                    return {
                        "provider": "gemini",
                        "ok": False,
                        "model": attempted_model,
                        "error": error_text,
                        "text": build_local_explanation(result),
                    }

        text = clean_explanation_text(_extract_gemini_text(data))
        if not text:
            error_text = f"{attempted_model}: Gemini returned an empty response."
            errors.append(error_text)
            continue

        return {
            "provider": "gemini",
            "ok": True,
            "model": attempted_model,
            "error": None if attempted_model == model_name else "Retried with fallback model.",
            "text": text,
        }

    return {
        "provider": "gemini",
        "ok": False,
        "model": model_name,
        "error": "All Gemini model attempts failed: " + " | ".join(errors[-3:]),
        "text": build_local_explanation(result),
    }


def explain_triage_result(
    result: dict,
    *,
    use_gemini: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    if use_gemini:
        return generate_gemini_explanation(result, model=model, api_key=api_key)
    return {
        "provider": "local",
        "ok": True,
        "model": None,
        "error": None,
        "text": build_local_explanation(result),
    }
