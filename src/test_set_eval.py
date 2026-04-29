"""Batch-evaluate deployed joblib models on the Test split (same inference as triage, no RAG)."""
from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pandas as pd

from .triage import _load_feature_frame, _run_models_for_feature_row, clear_caches, models_ready


def evaluate_test_split(*, anomaly_flag_min_score: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """
    Run deployed models on every incident with data_split == "Test".

    Returns (per-incident table, aggregate metric dict). Raises if no Test rows or models missing.
    """
    if not models_ready():
        raise RuntimeError("Models are not trained yet.")

    clear_caches()
    feature_frame = _load_feature_frame()
    if "data_split" not in feature_frame.columns:
        raise ValueError("incidents.csv has no data_split column; cannot select Test rows.")

    test_df = feature_frame[feature_frame["data_split"] == "Test"].copy()
    if test_df.empty:
        raise ValueError("No Test split rows found in processed data.")

    rows_out: list[dict] = []
    for _, meta in test_df.iterrows():
        incident_id = meta["incident_id"]
        incident_row = test_df.loc[test_df["incident_id"] == incident_id].copy()
        pred = _run_models_for_feature_row(
            incident_row,
            incident_id=incident_id,
            similar_k=3,
            anomaly_flag_min_score=anomaly_flag_min_score,
            skip_similarity_and_rag=True,
        )
        truth_fault = str(meta["fault_type"])
        truth_root = str(meta["root_cause_service"])
        truth_anom = bool(meta["is_anomalous"])
        pred_fault = pred["predicted_fault_type"]
        pred_root = pred["predicted_root_cause_service"]
        pred_anom = bool(pred["unusual"])

        rows_out.append(
            {
                "incident_id": incident_id,
                "true_fault_type": truth_fault,
                "pred_fault_type": pred_fault,
                "fault_correct": truth_fault == pred_fault,
                "true_root_cause_service": truth_root,
                "pred_root_cause_service": pred_root,
                "root_correct": truth_root == pred_root,
                "true_anomalous": truth_anom,
                "pred_anomalous": pred_anom,
                "anomaly_correct": truth_anom == pred_anom,
                "anomaly_score": pred["anomaly_score"],
                "fault_confidence": pred["fault_confidence"],
                "root_confidence": pred["root_cause_confidence"],
            }
        )

    result_df = pd.DataFrame(rows_out)

    y_true_a = result_df["true_anomalous"].astype(bool)
    y_pred_a = result_df["pred_anomalous"].astype(bool)
    y_true_f = result_df["true_fault_type"]
    y_pred_f = result_df["pred_fault_type"]
    y_true_r = result_df["true_root_cause_service"]
    y_pred_r = result_df["pred_root_cause_service"]

    metrics = {
        "n_test": int(len(result_df)),
        "anomaly_precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
        "anomaly_recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
        "anomaly_f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
        "fault_accuracy": float(accuracy_score(y_true_f, y_pred_f)),
        "fault_macro_f1": float(f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)),
        "root_cause_accuracy": float(accuracy_score(y_true_r, y_pred_r)),
        "root_cause_macro_f1": float(f1_score(y_true_r, y_pred_r, average="macro", zero_division=0)),
        "fault_correct_count": int(result_df["fault_correct"].sum()),
        "fault_wrong_count": int((~result_df["fault_correct"]).sum()),
        "root_correct_count": int(result_df["root_correct"].sum()),
        "root_wrong_count": int((~result_df["root_correct"]).sum()),
        "anomaly_correct_count": int(result_df["anomaly_correct"].sum()),
        "anomaly_wrong_count": int((~result_df["anomaly_correct"]).sum()),
    }

    return result_df, metrics
