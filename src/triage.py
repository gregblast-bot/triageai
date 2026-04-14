from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    ANOMALY_MODEL_PATH,
    FAULT_MODEL_PATH,
    MODEL_FILES,
    ROOT_CAUSE_MODEL_PATH,
    SIMILARITY_INDEX_PATH,
)
from .data import load_incidents, load_metrics
from .features import build_feature_frame, build_feature_row
from .rag import retrieve_context
from .train_models import train_all_models


def _ensure_models():
    if all(path.exists() for path in MODEL_FILES.values()):
        return
    train_all_models()


def _load_feature_frame() -> pd.DataFrame:
    incidents = load_incidents()
    metrics = load_metrics()
    return build_feature_frame(incidents, metrics)


def _run_models_for_feature_row(
    incident_row: pd.DataFrame,
    *,
    incident_id: str,
    similar_k: int = 3,
) -> dict:
    _ensure_models()

    anomaly_bundle = joblib.load(ANOMALY_MODEL_PATH)
    fault_bundle = joblib.load(FAULT_MODEL_PATH)
    root_cause_bundle = joblib.load(ROOT_CAUSE_MODEL_PATH)
    similarity_bundle = joblib.load(SIMILARITY_INDEX_PATH)

    numeric_columns = anomaly_bundle["numeric_columns"]
    anomaly_model = anomaly_bundle["model"]
    fault_model = fault_bundle["model"]
    root_cause_model = root_cause_bundle["model"]

    anomaly_score = float(-anomaly_model.decision_function(incident_row[numeric_columns])[0])
    unusual = bool(anomaly_model.predict(incident_row[numeric_columns])[0] == -1)

    fault_probabilities = fault_model.predict_proba(incident_row[numeric_columns + ["text"]])[0]
    fault_classes = fault_model.classes_
    fault_index = int(fault_probabilities.argmax())

    root_probabilities = root_cause_model.predict_proba(incident_row[numeric_columns + ["text"]])[0]
    root_classes = root_cause_model.classes_
    root_index = int(root_probabilities.argmax())

    scaler = similarity_bundle["scaler"]
    matrix = similarity_bundle["matrix"]
    similarity_numeric_columns = similarity_bundle["numeric_columns"]
    metadata = similarity_bundle["metadata"]
    query_vector = scaler.transform(incident_row[similarity_numeric_columns])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked_indices = scores.argsort()[::-1]

    similar_incidents = []
    for idx in ranked_indices:
        candidate = metadata.iloc[idx]
        if candidate["incident_id"] == incident_id:
            continue
        similar_incidents.append(
            {
                "incident_id": candidate["incident_id"],
                "similarity": float(scores[idx]),
                "fault_type": candidate["fault_type"],
                "root_cause_service": candidate["root_cause_service"],
            }
        )
        if len(similar_incidents) >= similar_k:
            break

    result = {
        "incident_id": incident_id,
        "unusual": unusual,
        "anomaly_score": anomaly_score,
        "predicted_fault_type": str(fault_classes[fault_index]),
        "fault_confidence": float(fault_probabilities[fault_index]),
        "predicted_root_cause_service": str(root_classes[root_index]),
        "root_cause_confidence": float(root_probabilities[root_index]),
        "top_similar_incidents": similar_incidents,
    }
    rag_context = retrieve_context(incident_row.iloc[0], result)
    result["retrieved_context"] = rag_context
    return result


def triage_incident(incident_id: str, similar_k: int = 3) -> dict:
    feature_frame = _load_feature_frame()
    incident_row = feature_frame.loc[feature_frame["incident_id"] == incident_id].copy()
    if incident_row.empty:
        raise ValueError(f"Unknown incident_id: {incident_id}")
    return _run_models_for_feature_row(incident_row, incident_id=incident_id, similar_k=similar_k)


def triage_custom_metrics(
    metrics: pd.DataFrame,
    *,
    title: str = "Uploaded real incident",
    description: str = "Custom metric window uploaded from an external application.",
    incident_id: str = "CUSTOM-REAL-001",
    similar_k: int = 3,
) -> dict:
    text = f"{title} {description}".strip()
    feature_row = build_feature_row(
        metrics,
        incident_id=incident_id,
        text=text,
        fault_type="unlabeled",
        root_cause_service="unlabeled",
        is_anomalous=False,
    )
    feature_frame = pd.DataFrame([feature_row])
    return _run_models_for_feature_row(feature_frame, incident_id=incident_id, similar_k=similar_k)


if __name__ == "__main__":
    incidents = load_incidents()
    example_id = incidents.iloc[0]["incident_id"]
    print(triage_incident(example_id))
