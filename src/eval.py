from __future__ import annotations

import json

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .config import EVAL_SUMMARY_PATH, MODELS_DIR
from .data import load_incidents, load_metrics
from .features import build_feature_frame, get_numeric_feature_columns
from .train_models import build_classifier_pipeline


def _get_contamination_rate(anomaly_rate: float) -> float:
    return min(max(float(anomaly_rate), 0.05), 0.35)


def _score_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def _evaluate_classifier_family(train_df, test_df, numeric_columns, target_column: str) -> dict:
    results = {}
    for classifier_name in (
        "random_forest",
        "random_forest_balanced",
        "random_forest_balanced_subsample",
        "logistic_regression",
    ):
        model = build_classifier_pipeline(numeric_columns, classifier_name)
        model.fit(train_df[numeric_columns + ["text"]], train_df[target_column])
        predictions = model.predict(test_df[numeric_columns + ["text"]])
        results[classifier_name] = _score_predictions(test_df[target_column], predictions)
    return results


def _select_best_model(results: dict) -> str:
    ranked = sorted(
        results.items(),
        key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]),
        reverse=True,
    )
    return ranked[0][0]


def evaluate_models() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feature_frame = build_feature_frame(load_incidents(), load_metrics())
    numeric_columns = get_numeric_feature_columns(feature_frame)

    if "data_split" in feature_frame.columns:
        train_df = feature_frame[feature_frame["data_split"] == "Train"].copy()
        test_df = feature_frame[feature_frame["data_split"] == "Test"].copy()
        if train_df.empty or test_df.empty:
            train_df, test_df = train_test_split(
                feature_frame,
                test_size=0.25,
                random_state=42,
                stratify=feature_frame["fault_type"],
            )
    else:
        train_df, test_df = train_test_split(
            feature_frame,
            test_size=0.25,
            random_state=42,
            stratify=feature_frame["fault_type"],
        )

    anomaly_model = IsolationForest(
        n_estimators=250,
        contamination=_get_contamination_rate(train_df["is_anomalous"].mean()),
        random_state=42,
    )
    anomaly_model.fit(train_df[numeric_columns])
    anomaly_pred = anomaly_model.predict(test_df[numeric_columns])
    anomaly_pred = [pred == -1 for pred in anomaly_pred]

    fault_results = _evaluate_classifier_family(
        train_df,
        test_df,
        numeric_columns,
        "fault_type",
    )
    root_results = _evaluate_classifier_family(
        train_df,
        test_df,
        numeric_columns,
        "root_cause_service",
    )
    best_fault_model = _select_best_model(fault_results)
    best_root_model = _select_best_model(root_results)

    summary = {
        "anomaly_precision": precision_score(test_df["is_anomalous"], anomaly_pred, zero_division=0),
        "anomaly_recall": recall_score(test_df["is_anomalous"], anomaly_pred, zero_division=0),
        "anomaly_f1": f1_score(test_df["is_anomalous"], anomaly_pred, zero_division=0),
        "fault_accuracy": fault_results[best_fault_model]["accuracy"],
        "fault_macro_f1": fault_results[best_fault_model]["macro_f1"],
        "root_cause_accuracy": root_results[best_root_model]["accuracy"],
        "root_cause_macro_f1": root_results[best_root_model]["macro_f1"],
        "selected_models": {
            "fault_type": best_fault_model,
            "root_cause_service": best_root_model,
        },
        "comparisons": {
            "fault_type": fault_results,
            "root_cause_service": root_results,
        },
    }

    EVAL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate_models(), indent=2))
