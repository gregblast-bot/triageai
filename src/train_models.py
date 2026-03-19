from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ANOMALY_MODEL_PATH,
    FAULT_MODEL_PATH,
    MODELS_DIR,
    ROOT_CAUSE_MODEL_PATH,
    SIMILARITY_INDEX_PATH,
)
from .data import load_incidents, load_metrics
from .features import build_feature_frame, get_numeric_feature_columns


def _get_contamination_rate(anomaly_rate: float) -> float:
    return min(max(float(anomaly_rate), 0.05), 0.35)


def _build_classifier_pipeline(numeric_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("text", TfidfVectorizer(max_features=400, ngram_range=(1, 2)), "text"),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=250, random_state=42)),
        ]
    )


def _fit_similarity_index(feature_frame: pd.DataFrame) -> dict:
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(feature_frame["text"])
    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "metadata": feature_frame[["incident_id", "fault_type", "root_cause_service"]].reset_index(
            drop=True
        ),
    }


def train_all_models() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    incidents = load_incidents()
    metrics = load_metrics()
    feature_frame = build_feature_frame(incidents, metrics)
    numeric_columns = get_numeric_feature_columns(feature_frame)

    anomaly_model = IsolationForest(
        n_estimators=250,
        contamination=_get_contamination_rate(feature_frame["is_anomalous"].mean()),
        random_state=42,
    )
    anomaly_model.fit(feature_frame[numeric_columns])

    fault_model = _build_classifier_pipeline(numeric_columns)
    fault_model.fit(feature_frame[numeric_columns + ["text"]], feature_frame["fault_type"])

    root_cause_model = _build_classifier_pipeline(numeric_columns)
    root_cause_model.fit(
        feature_frame[numeric_columns + ["text"]],
        feature_frame["root_cause_service"],
    )

    similarity_index = _fit_similarity_index(feature_frame)

    joblib.dump(
        {
            "model": anomaly_model,
            "numeric_columns": numeric_columns,
        },
        ANOMALY_MODEL_PATH,
    )
    joblib.dump(
        {
            "model": fault_model,
            "numeric_columns": numeric_columns,
            "label_column": "fault_type",
        },
        FAULT_MODEL_PATH,
    )
    joblib.dump(
        {
            "model": root_cause_model,
            "numeric_columns": numeric_columns,
            "label_column": "root_cause_service",
        },
        ROOT_CAUSE_MODEL_PATH,
    )
    joblib.dump(similarity_index, SIMILARITY_INDEX_PATH)

    return {
        "incident_count": len(feature_frame),
        "numeric_feature_count": len(numeric_columns),
        "model_dir": str(Path(MODELS_DIR)),
    }


if __name__ == "__main__":
    summary = train_all_models()
    print("Training complete:", summary)
