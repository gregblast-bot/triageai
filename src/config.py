from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

INCIDENTS_PATH = PROCESSED_DATA_DIR / "incidents.csv"
METRICS_PATH = PROCESSED_DATA_DIR / "metrics.csv"

ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
FAULT_MODEL_PATH = MODELS_DIR / "fault_model.joblib"
ROOT_CAUSE_MODEL_PATH = MODELS_DIR / "root_cause_model.joblib"
SIMILARITY_INDEX_PATH = MODELS_DIR / "similarity_index.joblib"
RAG_INDEX_PATH = MODELS_DIR / "rag_index.joblib"
EVAL_SUMMARY_PATH = MODELS_DIR / "eval_summary.json"

MODEL_FILES = {
    "anomaly": ANOMALY_MODEL_PATH,
    "fault": FAULT_MODEL_PATH,
    "root_cause": ROOT_CAUSE_MODEL_PATH,
    "similarity": SIMILARITY_INDEX_PATH,
    "rag": RAG_INDEX_PATH,
}

METRIC_COLUMNS = [
    "error_rate",
    "latency_ms",
    "cpu_pct",
    "memory_pct",
    "queue_depth",
    "auth_error_rate",
]

TEXT_COLUMNS = ["title", "description"]

# Default fault-lab control plane (host port maps from docker: 8001 -> API)
DEFAULT_LIVE_CONTROL_PLANE_URL = "http://localhost:8001"


def get_contamination_rate(anomaly_rate: float) -> float:
    return min(max(float(anomaly_rate), 0.05), 0.35)
