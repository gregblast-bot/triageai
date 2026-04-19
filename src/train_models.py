from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

logger = logging.getLogger(__name__)

SUPPORTED_CLASSIFIERS = (
    "random_forest",
    "random_forest_balanced",
    "random_forest_balanced_subsample",
    "logistic_regression",
)

# Randomized search samples this many distinct hyperparameter configs (3-fold CV each).
RANDOM_SEARCH_N_ITER = 40
RANDOM_SEARCH_CV = 3


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


def get_param_distributions(classifier_name: str) -> dict:
    """
    Distributions for RandomizedSearchCV. Use 'classifier__' for the pipeline step
    named 'classifier'. Grid values for n_estimators override the pipeline's initial
    RF defaults; class_weight from build_classifier_pipeline is preserved (not tuned).
    """
    if "random_forest" in classifier_name:
        return {
            "classifier__n_estimators": [250, 400, 500],
            "classifier__max_depth": [10, 20, None],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__min_samples_split": [2, 5],
            "preprocessor__text__max_features": [200, 400, 600],
            "preprocessor__text__ngram_range": [(1, 1), (1, 2)],
            "preprocessor__text__use_idf": [True, False],
        }

    if classifier_name == "logistic_regression":
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l1", "l2"],
            "preprocessor__text__max_features": [400, 600],
        }
    return {}


def _fit_classifier_with_optional_search(
    pipeline: Pipeline,
    classifier_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    role: str,
    use_search: bool,
) -> tuple[Pipeline, dict | None]:
    """Fit `pipeline`, optionally via RandomizedSearchCV. Returns (estimator, tuning_meta or None)."""
    if not use_search:
        pipeline.fit(X, y)
        return pipeline, None

    param_distributions = get_param_distributions(classifier_name)
    if not param_distributions:
        pipeline.fit(X, y)
        return pipeline, None

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=RANDOM_SEARCH_N_ITER,
        cv=RANDOM_SEARCH_CV,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    try:
        search.fit(X, y)
    except (ValueError, MemoryError):
        logger.warning("%s classifier: hyperparameter search failed; using default fit", role, exc_info=True)
        pipeline.fit(X, y)
        return pipeline, {"search_failed": True, "error": "search_raised"}
    except Exception:
        logger.exception("%s classifier: unexpected error during search; using default fit", role)
        pipeline.fit(X, y)
        return pipeline, {"search_failed": True, "error": "unexpected"}

    logger.info(
        "%s classifier best CV score=%.4f params=%s",
        role,
        float(search.best_score_),
        search.best_params_,
    )
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
    }


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


def train_all_models(
    fault_classifier_name: str = "random_forest_balanced_subsample",
    root_cause_classifier_name: str = "random_forest_balanced_subsample",
    *,
    use_grid_search: bool = False,
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

    # Fit Isolation Forest on *labeled-normal* rows only so the learned
    # "typical" manifold isn't dragged toward injected-fault patterns. Fall
    # back to the whole training frame only when the dataset has no normal
    # rows at all.
    normal_frame = training_frame[~training_frame["is_anomalous"].astype(bool)]
    if len(normal_frame) < 10:
        normal_frame = training_frame
        anomaly_contamination = get_contamination_rate(
            training_frame["is_anomalous"].mean()
        )
    else:
        anomaly_contamination = "auto"
    anomaly_model = IsolationForest(
        n_estimators=250,
        contamination=anomaly_contamination,
        random_state=42,
    )
    anomaly_model.fit(normal_frame[numeric_columns])

    X = training_frame[numeric_columns + ["text"]]

    fault_pipeline = build_classifier_pipeline(numeric_columns, fault_classifier_name)
    fault_model, fault_tuning = _fit_classifier_with_optional_search(
        fault_pipeline,
        fault_classifier_name,
        X,
        training_frame["fault_type"],
        role="Fault",
        use_search=use_grid_search,
    )

    root_pipeline = build_classifier_pipeline(numeric_columns, root_cause_classifier_name)
    root_cause_model, root_tuning = _fit_classifier_with_optional_search(
        root_pipeline,
        root_cause_classifier_name,
        X,
        training_frame["root_cause_service"],
        role="Root cause",
        use_search=use_grid_search,
    )

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
    fault_bundle = {
        "model": fault_model,
        "numeric_columns": numeric_columns,
        "label_column": "fault_type",
        "classifier_name": fault_classifier_name,
    }
    if fault_tuning and not fault_tuning.get("search_failed"):
        fault_bundle["tuning"] = fault_tuning
    joblib.dump(fault_bundle, FAULT_MODEL_PATH)

    root_bundle = {
        "model": root_cause_model,
        "numeric_columns": numeric_columns,
        "label_column": "root_cause_service",
        "classifier_name": root_cause_classifier_name,
    }
    if root_tuning and not root_tuning.get("search_failed"):
        root_bundle["tuning"] = root_tuning
    joblib.dump(root_bundle, ROOT_CAUSE_MODEL_PATH)

    joblib.dump(similarity_index, SIMILARITY_INDEX_PATH)

    result = {
        "incident_count": len(training_frame),
        "numeric_feature_count": len(numeric_columns),
        "model_dir": str(Path(MODELS_DIR)),
        "fault_classifier_name": fault_classifier_name,
        "root_cause_classifier_name": root_cause_classifier_name,
        "rag_document_count": len(rag_index["documents"]),
        "fault_tuning": fault_tuning,
        "root_cause_tuning": root_tuning,
    }
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train TriageAI models")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run randomized hyperparameter search (much slower, ~many minutes on full data)",
    )
    args = parser.parse_args()
    summary = train_all_models(use_grid_search=args.tune)
    print("Training complete:", summary)
