from __future__ import annotations

import streamlit as st

from src.config import MODEL_FILES
from src.data import load_incidents, load_metrics
from src.train_models import train_all_models
from src.triage import triage_incident


st.set_page_config(page_title="TriageAI", layout="wide")


@st.cache_data
def get_incidents():
    return load_incidents()


@st.cache_data
def get_metrics():
    return load_metrics()


def models_ready() -> bool:
    return all(path.exists() for path in MODEL_FILES.values())


def render_header():
    st.title("TriageAI")
    st.caption(
        "AI-assisted incident triage for anomaly detection, fault classification, and root-cause hinting."
    )


def render_training_panel():
    st.sidebar.header("Setup")
    st.sidebar.write("Train baseline models before using the triage view.")
    if st.sidebar.button("Generate data and train models", use_container_width=True):
        with st.spinner("Training baseline models..."):
            train_all_models()
        st.cache_data.clear()
        st.success("Training complete. Reload the incident view.")


def render_incident_selector(incidents):
    incident_label_map = {
        row.incident_id: f"{row.incident_id} | {row.title}"
        for row in incidents.itertuples(index=False)
    }
    selected = st.sidebar.selectbox(
        "Select incident",
        options=list(incident_label_map.keys()),
        format_func=lambda incident_id: incident_label_map[incident_id],
    )
    return selected


def render_incident_overview(incident):
    col1, col2, col3 = st.columns(3)
    col1.metric("True fault type", incident["fault_type"])
    col2.metric("Root-cause service", incident["root_cause_service"])
    col3.metric("Labeled anomalous", "Yes" if incident["is_anomalous"] else "No")

    st.subheader("Incident summary")
    st.write(f"**Title:** {incident['title']}")
    st.write(incident["description"])


def render_metric_charts(metrics):
    st.subheader("Metric trends")
    chart_columns = [
        "error_rate",
        "latency_ms",
        "cpu_pct",
        "memory_pct",
        "queue_depth",
        "auth_error_rate",
    ]
    st.line_chart(metrics.set_index("minute")[chart_columns])


def render_triage_output(result):
    st.subheader("Triage output")
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly flag", "Abnormal" if result["unusual"] else "Normal")
    col2.metric("Fault prediction", result["predicted_fault_type"])
    col3.metric("Root-cause prediction", result["predicted_root_cause_service"])

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


def main():
    render_header()
    render_training_panel()

    if not models_ready():
        st.info("Baseline models are not trained yet. Use the sidebar to train them.")
        return

    incidents = get_incidents()
    metrics = get_metrics()
    incident_id = render_incident_selector(incidents)
    incident = incidents.loc[incidents["incident_id"] == incident_id].iloc[0]
    incident_metrics = metrics.loc[metrics["incident_id"] == incident_id].copy()
    result = triage_incident(incident_id)

    render_incident_overview(incident)
    render_metric_charts(incident_metrics)
    render_triage_output(result)


if __name__ == "__main__":
    main()
