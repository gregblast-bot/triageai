from __future__ import annotations

from collections import Counter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import RAG_INDEX_PATH
from .data import load_incidents, load_metrics
from .features import build_feature_frame


FAULT_NOTES = {
    "healthy": {
        "investigate": "Verify that metrics remain within expected ranges and confirm there are no hidden localized failures.",
        "symptoms": "Stable latency, low error rate, and no sustained resource spikes.",
    },
    "cpu": {
        "investigate": "Check service saturation, request bursts, container limits, and whether one service is consuming most host CPU.",
        "symptoms": "Elevated CPU with throughput pressure and latency degradation during peak periods.",
    },
    "delay": {
        "investigate": "Inspect downstream latency, timeout propagation, queue growth, and slow RPC paths.",
        "symptoms": "Rising response time with degraded latency metrics even when raw error counts are modest.",
    },
    "disk": {
        "investigate": "Check I/O wait, disk saturation, storage capacity, and persistence-related bottlenecks.",
        "symptoms": "Latency increase combined with slower persistence operations and unstable resource behavior.",
    },
    "loss": {
        "investigate": "Check packet loss, connection health, retry storms, and network-dependent services.",
        "symptoms": "Intermittent failures, degraded throughput, and increased errors caused by unreliable communication.",
    },
    "mem": {
        "investigate": "Check memory growth, OOM events, garbage collection pressure, and restart loops.",
        "symptoms": "Growing memory usage followed by service instability, degraded latency, or crashes.",
    },
}


def _top_feature_terms(feature_row: pd.Series, limit: int = 6) -> list[str]:
    numeric_items = [
        (column, float(value))
        for column, value in feature_row.items()
        if column
        not in {
            "incident_id",
            "text",
            "fault_type",
            "root_cause_service",
            "is_anomalous",
            "data_split",
        }
    ]
    ranked = sorted(numeric_items, key=lambda item: abs(item[1]), reverse=True)
    terms = []
    for column, value in ranked[:limit]:
        if abs(value) < 1e-9:
            continue
        terms.append(f"{column.replace('_', ' ')} {value:.2f}")
    return terms


def _make_fault_documents() -> list[dict]:
    documents = []
    for fault_type, note in FAULT_NOTES.items():
        documents.append(
            {
                "doc_id": f"fault::{fault_type}",
                "source_type": "fault_note",
                "title": f"Fault note: {fault_type}",
                "content": (
                    f"Fault type {fault_type}. Common symptoms: {note['symptoms']} "
                    f"Suggested investigation focus: {note['investigate']}"
                ),
                "incident_id": "",
                "fault_type": fault_type,
                "root_cause_service": "",
            }
        )
    return documents


def _make_service_documents(incidents: pd.DataFrame) -> list[dict]:
    documents = []
    service_counts = Counter(incidents["root_cause_service"])
    for service_name, count in sorted(service_counts.items()):
        if service_name == "none":
            continue
        documents.append(
            {
                "doc_id": f"service::{service_name}",
                "source_type": "service_note",
                "title": f"Service note: {service_name}",
                "content": (
                    f"Service {service_name} appears in {count} labeled RCAEval incident windows. "
                    "Investigate service-level resource pressure, downstream dependencies, and service-specific latency or error changes."
                ),
                "incident_id": "",
                "fault_type": "",
                "root_cause_service": service_name,
            }
        )
    return documents


def _make_incident_documents(
    incidents: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> list[dict]:
    documents = []
    incident_lookup = incidents.set_index("incident_id")
    for row in feature_frame.itertuples(index=False):
        incident_meta = incident_lookup.loc[row.incident_id]
        feature_terms = _top_feature_terms(pd.Series(row._asdict()))
        documents.append(
            {
                "doc_id": f"incident::{row.incident_id}",
                "source_type": "incident_case",
                "title": f"Incident case {row.incident_id}",
                "content": (
                    f"Incident {row.incident_id} from {incident_meta['title']}. "
                    f"Fault type {row.fault_type}. Root cause service {row.root_cause_service}. "
                    f"Anomalous {bool(row.is_anomalous)}. Key signals: {'; '.join(feature_terms)}."
                ),
                "incident_id": row.incident_id,
                "fault_type": row.fault_type,
                "root_cause_service": row.root_cause_service,
            }
        )
    return documents


def build_knowledge_documents(
    incidents: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    documents = []
    documents.extend(_make_fault_documents())
    documents.extend(_make_service_documents(incidents))
    documents.extend(_make_incident_documents(incidents, feature_frame))
    return pd.DataFrame(documents)


def build_rag_index(
    incidents: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> dict:
    documents = build_knowledge_documents(incidents, feature_frame)
    vectorizer = TfidfVectorizer(max_features=1200, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents["content"])
    bundle = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "documents": documents.to_dict(orient="records"),
    }
    joblib.dump(bundle, RAG_INDEX_PATH)
    return bundle


def ensure_rag_index() -> dict:
    if RAG_INDEX_PATH.exists():
        return joblib.load(RAG_INDEX_PATH)

    incidents = load_incidents()
    metrics = load_metrics()
    feature_frame = build_feature_frame(incidents, metrics)
    return build_rag_index(incidents, feature_frame)


def build_retrieval_query(incident_row: pd.Series, triage_result: dict) -> str:
    feature_terms = _top_feature_terms(incident_row, limit=5)
    return " ".join(
        [
            str(incident_row.get("text", "")),
            f"fault {triage_result['predicted_fault_type']}",
            f"service {triage_result['predicted_root_cause_service']}",
            "abnormal" if triage_result["unusual"] else "normal",
            *feature_terms,
        ]
    )


def retrieve_context(
    incident_row: pd.Series,
    triage_result: dict,
    top_k: int = 5,
) -> dict:
    bundle = ensure_rag_index()
    vectorizer = bundle["vectorizer"]
    matrix = bundle["matrix"]
    documents = bundle["documents"]

    query = build_retrieval_query(incident_row, triage_result)
    query_vector = vectorizer.transform([query])
    base_scores = cosine_similarity(query_vector, matrix).flatten()

    adjusted_scores = []
    predicted_fault = triage_result["predicted_fault_type"]
    predicted_service = triage_result["predicted_root_cause_service"]
    for idx, doc in enumerate(documents):
        score = float(base_scores[idx])
        if doc["fault_type"] and doc["fault_type"] == predicted_fault:
            score += 0.25
        if doc["root_cause_service"] and doc["root_cause_service"] == predicted_service:
            score += 0.2
        if doc["source_type"] == "fault_note" and doc["fault_type"] == predicted_fault:
            score += 0.1
        if doc["source_type"] == "service_note" and doc["root_cause_service"] == predicted_service:
            score += 0.1
        if predicted_fault != "healthy" and doc["fault_type"] == "healthy":
            score -= 0.1
        adjusted_scores.append(score)

    ranked_indices = sorted(range(len(documents)), key=lambda idx: adjusted_scores[idx], reverse=True)

    selected = []
    source_limits = {"fault_note": 2, "service_note": 2, "incident_case": 2}
    source_counts = Counter()

    for idx in ranked_indices:
        doc = documents[idx]
        if doc["source_type"] == "incident_case" and doc["incident_id"] == triage_result["incident_id"]:
            continue
        if source_counts[doc["source_type"]] >= source_limits.get(doc["source_type"], top_k):
            continue

        selected.append(
            {
                "doc_id": doc["doc_id"],
                "source_type": doc["source_type"],
                "title": doc["title"],
                "score": float(adjusted_scores[idx]),
                "content": doc["content"],
                "incident_id": doc["incident_id"],
                "fault_type": doc["fault_type"],
                "root_cause_service": doc["root_cause_service"],
            }
        )
        source_counts[doc["source_type"]] += 1
        if len(selected) >= top_k:
            break

    summary = (
        f"Retrieved {len(selected)} context items for predicted fault "
        f"`{triage_result['predicted_fault_type']}` in service "
        f"`{triage_result['predicted_root_cause_service']}`."
    )
    return {
        "query": query,
        "summary": summary,
        "documents": selected,
    }


if __name__ == "__main__":
    incidents = load_incidents()
    metrics = load_metrics()
    feature_frame = build_feature_frame(incidents, metrics)
    bundle = build_rag_index(incidents, feature_frame)
    print(f"Built local retrieval index with {len(bundle['documents'])} documents.")
