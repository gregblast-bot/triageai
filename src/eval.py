from __future__ import annotations

import json

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EVAL_SUMMARY_PATH, MODELS_DIR
from .data import load_incidents, load_metrics
from .features import build_feature_frame, get_numeric_feature_columns


def _get_contamination_rate(anomaly_rate: float) -> float:
    return min(max(float(anomaly_rate), 0.05), 0.35)


def _build_pipeline(numeric_columns: list[str]) -> Pipeline:
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


def evaluate_models() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feature_frame = build_feature_frame(load_incidents(), load_metrics())
    numeric_columns = get_numeric_feature_columns(feature_frame)

    # Now we want to use the datasplit that was defined during data conversion to maintain consistency.
    train_df = feature_frame[feature_frame["data_split"] == "Train"].copy()
    test_df = feature_frame[feature_frame["data_split"] == "Test"].copy()

    # Remove data_split from features if it's there.
    if "data_split" in numeric_columns:
        numeric_columns.remove("data_split")

    anomaly_model = IsolationForest(
        n_estimators=250,
        contamination=_get_contamination_rate(train_df["is_anomalous"].mean()),
        random_state=42,
    )
    anomaly_model.fit(train_df[numeric_columns])
    anomaly_pred = anomaly_model.predict(test_df[numeric_columns])
    anomaly_pred = [pred == -1 for pred in anomaly_pred]

    fault_model = _build_pipeline(numeric_columns)
    fault_model.fit(train_df[numeric_columns + ["text"]], train_df["fault_type"])
    fault_pred = fault_model.predict(test_df[numeric_columns + ["text"]])

    root_model = _build_pipeline(numeric_columns)
    root_model.fit(train_df[numeric_columns + ["text"]], train_df["root_cause_service"])
    root_pred = root_model.predict(test_df[numeric_columns + ["text"]])

    summary = {
        "anomaly_precision": precision_score(test_df["is_anomalous"], anomaly_pred, zero_division=0),
        "anomaly_recall": recall_score(test_df["is_anomalous"], anomaly_pred, zero_division=0),
        "anomaly_f1": f1_score(test_df["is_anomalous"], anomaly_pred, zero_division=0),
        "fault_accuracy": accuracy_score(test_df["fault_type"], fault_pred),
        "fault_macro_f1": f1_score(test_df["fault_type"], fault_pred, average="macro"),
        "root_cause_accuracy": accuracy_score(test_df["root_cause_service"], root_pred),
        "root_cause_macro_f1": f1_score(
            test_df["root_cause_service"],
            root_pred,
            average="macro",
        ),
    }

    EVAL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate_models(), indent=2))
