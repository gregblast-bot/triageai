from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from .config import (
    ANOMALY_MODEL_PATH,
    FAULT_MODEL_PATH,
    MODELS_DIR,
    ROOT_CAUSE_MODEL_PATH,
    SIMILARITY_INDEX_PATH,
    get_contamination_rate,
)
from .data import load_incidents, load_metrics
from .features import build_feature_frame, get_numeric_feature_columns
from .rag import build_rag_index


SUPPORTED_CLASSIFIERS = (
    "random_forest",
    "random_forest_balanced",
    "random_forest_balanced_subsample",
    "logistic_regression",
)


def build_classifier_pipeline(
    numeric_columns: list[str],
    classifier_name: str = "random_forest",
) -> Pipeline:
    if classifier_name not in SUPPORTED_CLASSIFIERS:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("text", TfidfVectorizer(max_features=400, ngram_range=(1, 2)), "text"),
        ]
    )

    if classifier_name == "random_forest":
        classifier = RandomForestClassifier(n_estimators=250, random_state=42)
    elif classifier_name == "random_forest_balanced":
        classifier = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
        )
    elif classifier_name == "random_forest_balanced_subsample":
        classifier = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced_subsample",
        )
    else:
        classifier = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _fit_similarity_index(feature_frame: pd.DataFrame, numeric_columns: list[str]) -> dict:
    scaler = StandardScaler()
    matrix = scaler.fit_transform(feature_frame[numeric_columns])
    return {
        "scaler": scaler,
        "matrix": matrix,
        "numeric_columns": numeric_columns,
        "metadata": feature_frame[["incident_id", "fault_type", "root_cause_service"]].reset_index(
            drop=True
        ),
    }


def get_param_grid(classifier_name: str) -> dict:
    """
    Returns parameter grids. Use 'classifier__' prefix to target the 
    pipeline step named 'classifier'.
    """
    if "random_forest" in classifier_name:
        return {
            # Random Forest Tuning
            'classifier__n_estimators': [250, 500],
            'classifier__max_depth': [10, 20, None],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__min_samples_split': [2, 5],
            
            # TF-IDF Tuning (reaching into the ColumnTransformer)
            'preprocessor__text__max_features': [200, 400, 600],
            'preprocessor__text__ngram_range': [(1, 1), (1, 2)],
            'preprocessor__text__use_idf': [True, False]
        }
        
    elif classifier_name == "logistic_regression":
        return {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'preprocessor__text__max_features': [400, 600]
        }
    return {}


def train_all_models(
    fault_classifier_name: str = "random_forest_balanced_subsample",
    root_cause_classifier_name: str = "random_forest_balanced_subsample",
    use_grid_search: bool = True
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    incidents = load_incidents()
    metrics = load_metrics()
    feature_frame = build_feature_frame(incidents, metrics)
    numeric_columns = get_numeric_feature_columns(feature_frame)

    training_frame = feature_frame
    if "data_split" in feature_frame.columns:
        train_subset = feature_frame[feature_frame["data_split"] == "Train"].copy()
        if not train_subset.empty:
            training_frame = train_subset

    anomaly_model = IsolationForest(
        n_estimators=250,
        contamination=get_contamination_rate(training_frame["is_anomalous"].mean()),
        random_state=42,
    )
    anomaly_model.fit(training_frame[numeric_columns])

    fault_model = build_classifier_pipeline(numeric_columns, fault_classifier_name)
    if use_grid_search:
        fault_search = GridSearchCV(
            fault_model, 
            get_param_grid(fault_classifier_name), 
            cv=3, 
            n_jobs=-1, 
            scoring='f1_weighted'
        )
        try:
            fault_search.fit(training_frame[numeric_columns + ["text"]], training_frame["fault_type"])
            fault_model = fault_search.best_estimator_
            print(f"Best Fault Params: {fault_search.best_params_}")
        except Exception as e:
            print(f"GridSearch failed: {e}")
            fault_model = fault_model.fit(training_frame[numeric_columns + ["text"]], training_frame["fault_type"])
    else:
        fault_model.fit(training_frame[numeric_columns + ["text"]], training_frame["fault_type"])

    root_cause_model = build_classifier_pipeline(numeric_columns, root_cause_classifier_name)
    if use_grid_search:
        root_cause_search = GridSearchCV(
            root_cause_model, 
            get_param_grid(root_cause_classifier_name), 
            cv=3, 
            n_jobs=-1, 
            scoring='f1_weighted'
        )
        try:
            root_cause_search.fit(training_frame[numeric_columns + ["text"]], training_frame["root_cause_service"])
            root_cause_model = root_cause_search.best_estimator_
            print(f"Best Root Cause Params: {root_cause_search.best_params_}")
        except Exception as e:
            print(f"GridSearch failed: {e}")
            root_cause_model = root_cause_model.fit(training_frame[numeric_columns + ["text"]], training_frame["root_cause_service"])
    else:
        root_cause_model.fit(training_frame[numeric_columns + ["text"]], training_frame["root_cause_service"])

    similarity_index = _fit_similarity_index(training_frame, numeric_columns)
    training_incidents = incidents.loc[
        incidents["incident_id"].isin(training_frame["incident_id"])
    ].copy()
    rag_index = build_rag_index(training_incidents, training_frame)

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
            "classifier_name": fault_classifier_name,
        },
        FAULT_MODEL_PATH,
    )
    joblib.dump(
        {
            "model": root_cause_model,
            "numeric_columns": numeric_columns,
            "label_column": "root_cause_service",
            "classifier_name": root_cause_classifier_name,
        },
        ROOT_CAUSE_MODEL_PATH,
    )
    joblib.dump(similarity_index, SIMILARITY_INDEX_PATH)

    return {
        "incident_count": len(training_frame),
        "numeric_feature_count": len(numeric_columns),
        "model_dir": str(Path(MODELS_DIR)),
        "fault_classifier_name": fault_classifier_name,
        "root_cause_classifier_name": root_cause_classifier_name,
        "rag_document_count": len(rag_index["documents"]),
    }


if __name__ == "__main__":
    summary = train_all_models()
    print("Training complete:", summary)
