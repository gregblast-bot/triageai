from __future__ import annotations

import joblib

import pandas as pd
import streamlit as st

from src.config import FAULT_MODEL_PATH, METRIC_COLUMNS, ROOT_CAUSE_MODEL_PATH
from src.data import load_incidents, load_metrics
from src.train_models import train_all_models
from src.triage import clear_caches, models_ready, triage_custom_metrics, triage_incident


st.set_page_config(page_title="TriageAI", layout="wide")


@st.cache_data
def get_incidents():
    return load_incidents()


@st.cache_data
def get_metrics():
    return load_metrics()


def get_active_classifier_config() -> dict:
    if not FAULT_MODEL_PATH.exists() or not ROOT_CAUSE_MODEL_PATH.exists():
        return {}

    fault_bundle = joblib.load(FAULT_MODEL_PATH)
    root_bundle = joblib.load(ROOT_CAUSE_MODEL_PATH)
    return {
        "fault_classifier_name": fault_bundle.get("classifier_name", "unknown"),
        "root_cause_classifier_name": root_bundle.get("classifier_name", "unknown"),
    }


def render_header():
    st.title("TriageAI")
    st.caption(
        "AI-assisted incident triage for anomaly detection, fault classification, and root-cause hinting."
    )


def render_training_panel():
    st.sidebar.header("Setup")
    st.sidebar.write(
        "Train the anomaly detector, classifiers, similarity index, and local retrieval index."
    )

    classifier_options = {
        "Random Forest": "random_forest",
        "Random Forest (Balanced)": "random_forest_balanced",
        "Random Forest (Balanced Subsample)": "random_forest_balanced_subsample",
        "Logistic Regression": "logistic_regression",
    }
    selected_fault_label = st.sidebar.selectbox(
        "Fault classifier",
        options=list(classifier_options.keys()),
        index=2,
    )
    selected_root_label = st.sidebar.selectbox(
        "Root-cause classifier",
        options=list(classifier_options.keys()),
        index=2,
    )
    st.sidebar.caption("Anomaly detection remains Isolation Forest.")

    use_hp_search = st.sidebar.checkbox(
        "Randomized hyperparameter search (slower)",
        value=False,
        help=(
            "Tunes fault and root-cause classifiers via RandomizedSearchCV (~40 trials × 3-fold CV each). "
            "Can take many minutes on large datasets."
        ),
    )

    active_config = get_active_classifier_config()
    if active_config:
        st.sidebar.write(
            "Active trained models: "
            f"fault=`{active_config['fault_classifier_name']}`, "
            f"root cause=`{active_config['root_cause_classifier_name']}`"
        )

    if st.sidebar.button("Generate data and train models", use_container_width=True):
        spinner_msg = (
            "Training with hyperparameter search..."
            if use_hp_search
            else "Training baseline models..."
        )
        with st.spinner(spinner_msg):
            summary = train_all_models(
                fault_classifier_name=classifier_options[selected_fault_label],
                root_cause_classifier_name=classifier_options[selected_root_label],
                use_grid_search=use_hp_search,
            )
        st.cache_data.clear()
        clear_caches()
        msg = (
            "Training complete. "
            f"fault=`{summary['fault_classifier_name']}`, "
            f"root cause=`{summary['root_cause_classifier_name']}`"
        )
        ft = summary.get("fault_tuning") or {}
        rt = summary.get("root_cause_tuning") or {}
        if ft.get("best_cv_score") is not None:
            msg += f" | fault CV f1_weighted={ft['best_cv_score']:.4f}"
        if rt.get("best_cv_score") is not None:
            msg += f" | root-cause CV f1_weighted={rt['best_cv_score']:.4f}"
        st.success(msg)


def render_incident_selector(incidents):
    incident_label_map = {
        row.incident_id: f"{row.incident_id} [{row.data_split}] - {row.title}"
        for row in incidents.itertuples(index=False)
    }
    selected = st.sidebar.selectbox(
        "Select incident",
        options=list(incident_label_map.keys()),
        format_func=lambda incident_id: incident_label_map[incident_id],
    )
    return selected


def render_mode_selector():
    st.sidebar.header("Input Mode")
    return st.sidebar.radio(
        "Choose data source",
        options=["Dataset incident", "Upload real incident"],
        index=0,
    )


def custom_metrics_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "minute": [0, 1, 2],
            "error_rate": [0.0, 0.1, 0.0],
            "latency_ms": [100.0, 140.0, 110.0],
            "cpu_pct": [35.0, 60.0, 40.0],
            "memory_pct": [55.0, 56.0, 57.0],
            "queue_depth": [2.0, 4.0, 3.0],
            "auth_error_rate": [0.0, 0.0, 0.0],
        }
    )


def parse_uploaded_metrics(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def render_upload_panel():
    st.sidebar.subheader("Real Incident Upload")
    st.sidebar.caption(
        "Upload a CSV with at least these columns: "
        "`minute`, `error_rate`, `latency_ms`, `cpu_pct`, `memory_pct`, `queue_depth`, `auth_error_rate`."
    )
    template_csv = custom_metrics_template().to_csv(index=False)
    st.sidebar.download_button(
        "Download CSV template",
        data=template_csv,
        file_name="triageai_metrics_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    title = st.sidebar.text_input("Incident title", value="Uploaded real incident")
    description = st.sidebar.text_area(
        "Incident description",
        value="Paste a short summary of what the application is doing or failing to do.",
        height=100,
    )
    uploaded_file = st.sidebar.file_uploader("Upload metric CSV", type="csv")
    return title, description, uploaded_file

def render_incident_overview(incident):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True fault type", incident["fault_type"])
    col2.metric("Root-cause service", incident["root_cause_service"])
    col3.metric("Labeled anomalous", "Yes" if incident["is_anomalous"] else "No")
    col4.metric("Data split", incident.get("data_split", "Unknown"))

    st.subheader("Incident summary")
    st.write(f"**Title:** {incident['title']}")
    st.write(incident["description"])


def render_custom_incident_overview(title: str, description: str):
    col1, col2, col3 = st.columns(3)
    col1.metric("Source", "Uploaded CSV")
    col2.metric("Training labels", "Unavailable")
    col3.metric("Generalization", "Unseen incident")

    st.subheader("Incident summary")
    st.write(f"**Title:** {title}")
    st.write(description)
    st.warning(
        "These predictions are being applied to unseen external telemetry. "
        "They are useful as pattern hints, but they are not guaranteed to map cleanly to your real application's services or fault taxonomy."
    )


def render_metric_charts(metrics):
    st.subheader("Metric trends")
    chart_columns = [column for column in METRIC_COLUMNS if column in metrics.columns]
    st.line_chart(metrics.set_index("minute")[chart_columns])


def render_triage_output(result):
    st.subheader("Triage output")
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly flag", "Abnormal" if result["unusual"] else "Normal")
    col2.metric("Fault prediction", result["predicted_fault_type"])
    col3.metric("Root-cause prediction", result["predicted_root_cause_service"])

    classifier_config = get_active_classifier_config()
    if classifier_config:
        st.caption(
            "Active classifiers: "
            f"fault=`{classifier_config['fault_classifier_name']}`, "
            f"root cause=`{classifier_config['root_cause_classifier_name']}`"
        )

    st.write(
        f"**Anomaly score:** {result['anomaly_score']:.3f} | "
        f"**Fault confidence:** {result['fault_confidence']:.3f} | "
        f"**Root-cause confidence:** {result['root_cause_confidence']:.3f}"
    )

    if result["top_similar_incidents"]:
        st.subheader("Similar incidents")
        for item in result["top_similar_incidents"]:
            st.write(
                f"- `{item['incident_id']}` | score={item['similarity']:.3f} | "
                f"fault={item['fault_type']} | root cause={item['root_cause_service']}"
            )

    retrieved_context = result.get("retrieved_context", {})
    if retrieved_context.get("documents"):
        st.subheader("Retrieved context")
        st.caption(retrieved_context["summary"])
        with st.expander("Retrieval query", expanded=False):
            st.code(retrieved_context["query"])
        for item in retrieved_context["documents"]:
            st.write(
                f"**{item['title']}** "
                f"(`{item['source_type']}`, score={item['score']:.3f})"
            )
            st.write(item["content"])


def main():
    render_header()
    render_training_panel()
    mode = render_mode_selector()

    if not models_ready():
        st.info("Baseline models are not trained yet. Use the sidebar to train them.")
        return

    if mode == "Dataset incident":
        incidents = get_incidents()
        metrics = get_metrics()
        incident_id = render_incident_selector(incidents)
        incident = incidents.loc[incidents["incident_id"] == incident_id].iloc[0]
        incident_metrics = metrics.loc[metrics["incident_id"] == incident_id].copy()
        result = triage_incident(incident_id)

        render_incident_overview(incident)
        render_metric_charts(incident_metrics)
        render_triage_output(result)
        return

    title, description, uploaded_file = render_upload_panel()
    if uploaded_file is None:
        st.info("Upload a metric CSV from a real application to run the trained models on unseen telemetry.")
        st.subheader("Expected CSV shape")
        st.dataframe(custom_metrics_template(), use_container_width=True)
        return

    try:
        uploaded_metrics = parse_uploaded_metrics(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        return

    missing_columns = [column for column in ["minute", *METRIC_COLUMNS] if column not in uploaded_metrics.columns]
    if missing_columns:
        st.error(
            "Uploaded CSV is missing required columns: "
            + ", ".join(f"`{column}`" for column in missing_columns)
        )
        st.dataframe(custom_metrics_template(), use_container_width=True)
        return

    coerced_columns = []
    for column in METRIC_COLUMNS:
        original = uploaded_metrics[column]
        numeric = pd.to_numeric(original, errors="coerce")
        non_numeric_count = int(original.notna().sum() - numeric.notna().sum())
        if non_numeric_count > 0:
            coerced_columns.append(f"`{column}` ({non_numeric_count} values)")
    if coerced_columns:
        st.warning(
            "Some metric columns contain non-numeric values that will be treated as zero: "
            + ", ".join(coerced_columns)
        )

    render_custom_incident_overview(title, description)
    render_metric_charts(uploaded_metrics)
    result = triage_custom_metrics(
        uploaded_metrics,
        title=title,
        description=description,
    )
    render_triage_output(result)


if __name__ == "__main__":
    main()
