from __future__ import annotations

from datetime import timedelta

import joblib

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_LIVE_CONTROL_PLANE_URL,
    EVAL_SUMMARY_PATH,
    FAULT_MODEL_PATH,
    METRIC_COLUMNS,
    ROOT_CAUSE_MODEL_PATH,
)
from src.live_telemetry import fetch_telemetry_window
from src.data import load_incidents, load_metrics
from src.test_set_eval import evaluate_test_split
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
    with st.sidebar.expander("Training setup", expanded=not models_ready()):
        st.write(
            "Train the anomaly detector, classifiers, similarity index, and local retrieval index."
        )

        classifier_options = {
            "Random Forest": "random_forest",
            "Random Forest (Balanced)": "random_forest_balanced",
            "Random Forest (Balanced Subsample)": "random_forest_balanced_subsample",
            "Logistic Regression": "logistic_regression",
        }
        selected_fault_label = st.selectbox(
            "Fault classifier",
            options=list(classifier_options.keys()),
            index=2,
        )
        selected_root_label = st.selectbox(
            "Root-cause classifier",
            options=list(classifier_options.keys()),
            index=2,
        )
        st.caption("Anomaly detection remains Isolation Forest (trained on normal rows).")

        use_hp_search = st.checkbox(
            "Randomized hyperparameter search (slower)",
            value=False,
            help=(
                "Tunes fault and root-cause classifiers via RandomizedSearchCV (~40 trials × 3-fold CV each). "
                "Can take many minutes on large datasets."
            ),
        )

        active_config = get_active_classifier_config()
        if active_config:
            st.write(
                "Active trained models: "
                f"fault=`{active_config['fault_classifier_name']}`, "
                f"root cause=`{active_config['root_cause_classifier_name']}`"
            )

        train_clicked = st.button("Generate data and train models", use_container_width=True)

    if train_clicked:
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
        st.sidebar.success(msg)


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
        options=[
            "Dataset incident",
            "Upload real incident",
            "Live ingest (HTTP)",
            "Test set evaluation (batch)",
        ],
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


def render_live_ingest_panel():
    st.sidebar.subheader("Live HTTP ingest")
    st.sidebar.caption(
        "Polls a fault-lab control plane `/api/telemetry/window` JSON endpoint. "
        f"Default matches Docker map `{DEFAULT_LIVE_CONTROL_PLANE_URL}`."
    )
    st.sidebar.text_input(
        "Control plane base URL",
        value=DEFAULT_LIVE_CONTROL_PLANE_URL,
        key="live_base_url",
        help="Example: http://localhost:8001 when fault_lab is running.",
    )
    st.sidebar.number_input(
        "Window length (minutes)",
        min_value=30,
        max_value=300,
        value=120,
        step=10,
        key="live_limit",
    )
    refresh_mode = st.sidebar.selectbox(
        "Refresh",
        options=["Manual (button)", "Every 5s", "Every 10s", "Every 30s"],
        index=0,
        key="live_refresh_mode",
    )
    st.sidebar.text_input(
        "Incident title",
        value="Live fault-lab window",
        key="live_incident_title",
    )
    st.sidebar.text_area(
        "Incident description",
        value="Rolling telemetry window ingested over HTTP from the running fault lab.",
        key="live_incident_desc",
        height=100,
    )
    st.sidebar.slider(
        "Min anomaly score to show Abnormal",
        min_value=0.0,
        max_value=0.4,
        value=0.05,
        step=0.01,
        key="live_anomaly_min",
        help=(
            "Models are trained on RCAEval, not fault-lab traffic—benign checkout can look "
            "like a borderline outlier. Raise this to count light spikes as Normal; use 0 "
            "for the raw IsolationForest decision only."
        ),
    )
    return refresh_mode


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


def render_live_ingest_overview(
    title: str,
    description: str,
    control_plane_url: str,
    *,
    meta: dict | None = None,
):
    meta = meta or {}
    expected = meta.get("expected") or {}
    active_scenario = meta.get("active_scenario") or "unknown"

    col1, col2, col3 = st.columns(3)
    col1.metric("Source", "Live HTTP window")
    col2.metric("Active scenario", active_scenario)
    col3.metric(
        "Expected fault",
        expected.get("fault_type", "unknown"),
    )
    st.caption(f"Control plane: `{control_plane_url}`")

    st.subheader("Incident summary")
    st.write(f"**Title:** {title}")
    st.write(description)
    if expected:
        st.caption(
            f"Expected root-cause service (per scenario preset): `{expected.get('root_cause_service', 'unknown')}`"
        )
    st.info(
        "Telemetry is polled from the control plane API (not uploaded CSV). "
        "Run `docker compose -f fault_lab/docker-compose.yml up` and generate traffic in the storefront."
    )
    st.caption(
        "**Healthy scenario** means fault-lab is not injecting scripted faults—it does not guarantee "
        "the ML pipeline will say “Normal,” because metrics still differ from RCAEval training data. "
        "Use the sidebar **Min anomaly score** slider to separate borderline spikes from clear incidents."
    )


def render_metric_charts(metrics):
    st.subheader("Metric trends")
    chart_columns = [column for column in METRIC_COLUMNS if column in metrics.columns]
    st.line_chart(metrics.set_index("minute")[chart_columns])


def _run_live_ingest_triage():
    """Wire up whatever the user typed in the sidebar and run triage on the live pull."""
    base_url = st.session_state.get("live_base_url", DEFAULT_LIVE_CONTROL_PLANE_URL)
    limit = int(st.session_state.get("live_limit", 120))
    title = st.session_state.get("live_incident_title", "Live fault-lab window")
    description = st.session_state.get("live_incident_desc", "")
    metrics_df, err, meta = fetch_telemetry_window(base_url, limit=limit)
    if err:
        st.error(err)
        return None
    render_live_ingest_overview(title, description, base_url, meta=meta)
    render_metric_charts(metrics_df)
    min_score = float(st.session_state.get("live_anomaly_min", 0.05))
    result = triage_custom_metrics(
        metrics_df,
        title=title,
        description=description,
        incident_id="LIVE-HTTP-001",
        anomaly_flag_min_score=min_score,
    )
    result["expected"] = meta.get("expected") or {}
    result["active_scenario"] = meta.get("active_scenario")
    return result


def render_triage_output(result):
    st.subheader("Triage output")
    raw = result.get("unusual_raw")
    if raw is not None and raw != result["unusual"]:
        st.caption(
            "Isolation Forest flagged this window as an outlier, but the anomaly score is below "
            f"your minimum ({result.get('anomaly_flag_min_score', 0):.2f}) — **showing Normal** for the demo."
        )
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly flag", "Abnormal" if result["unusual"] else "Normal")
    col2.metric("Fault prediction", result["predicted_fault_type"])
    col3.metric("Root-cause prediction", result["predicted_root_cause_service"])

    expected = result.get("expected") or {}
    if expected:
        expected_fault = expected.get("fault_type", "unknown")
        expected_service = expected.get("root_cause_service", "unknown")
        fault_match = expected_fault == result["predicted_fault_type"]
        service_match = expected_service == result["predicted_root_cause_service"]
        has_truth_anomaly = "is_anomalous" in expected
        if has_truth_anomaly:
            truth_anom = bool(expected["is_anomalous"])
            pred_anom = bool(result["unusual"])
            anomaly_match = truth_anom == pred_anom
        else:
            anomaly_match = True  # N/A for live-ingest scenario-only labels
        msg_parts = [
            f"Labeled fault: `{expected_fault}` → predicted `{result['predicted_fault_type']}` "
            f"({'match' if fault_match else 'mismatch'})",
            f"Labeled service: `{expected_service}` → predicted `{result['predicted_root_cause_service']}` "
            f"({'match' if service_match else 'mismatch'})",
        ]
        if has_truth_anomaly:
            msg_parts.insert(
                0,
                f"Labeled anomaly: `{'Abnormal' if truth_anom else 'Normal'}` → "
                f"predicted `{'Abnormal' if pred_anom else 'Normal'}` "
                f"({'match' if anomaly_match else 'mismatch'})",
            )
        msg = "  \n".join(msg_parts)
        matches = [anomaly_match, fault_match, service_match] if has_truth_anomaly else [fault_match, service_match]
        if all(matches):
            st.success(msg)
        elif any(matches):
            st.warning(msg)
        else:
            st.error(msg)

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


def render_test_set_evaluation():
    st.subheader("Test split — batch evaluation")
    st.caption(
        "Scores every **Test** row using the **saved** models in `models/` (same inference as Dataset incident, "
        "RAG skipped for speed). Compares predictions to labels in `data/processed/incidents.csv`."
    )
    st.slider(
        "Min anomaly score to count as Abnormal",
        min_value=0.0,
        max_value=0.4,
        value=0.0,
        step=0.01,
        key="batch_anomaly_min",
        help="Aligned with triage: outlier must clear this score to be labeled Abnormal.",
    )

    if st.button("Run evaluation on Test split", type="primary", key="batch_eval_btn"):
        with st.spinner("Running deployed models on all Test incidents..."):
            try:
                min_s = float(st.session_state.get("batch_anomaly_min", 0.0))
                df, metrics = evaluate_test_split(anomaly_flag_min_score=min_s)
                st.session_state["batch_eval_df"] = df
                st.session_state["batch_eval_metrics"] = metrics
            except ValueError as exc:
                st.error(str(exc))
            except RuntimeError as exc:
                st.error(str(exc))

    if "batch_eval_metrics" not in st.session_state:
        st.info("Click **Run evaluation on Test split** to compute metrics, charts, and per-incident results.")
        if EVAL_SUMMARY_PATH.exists():
            st.caption(
                f"Optional: `{EVAL_SUMMARY_PATH.name}` from `python -m src.eval` compares **retrained** "
                "classifier families on a holdout split — different from the deployed-model table below."
            )
        return

    metrics = st.session_state["batch_eval_metrics"]
    df = st.session_state["batch_eval_df"]

    st.markdown("##### Aggregate metrics (deployed models, Test split)")
    mcols = st.columns(4)
    mcols[0].metric("Test incidents", metrics["n_test"])
    mcols[1].metric("Fault accuracy", f"{metrics['fault_accuracy']:.3f}")
    mcols[2].metric("Root-cause accuracy", f"{metrics['root_cause_accuracy']:.3f}")
    mcols[3].metric("Anomaly F1", f"{metrics['anomaly_f1']:.3f}")

    mcols2 = st.columns(4)
    mcols2[0].metric("Fault macro-F1", f"{metrics['fault_macro_f1']:.3f}")
    mcols2[1].metric("Root macro-F1", f"{metrics['root_cause_macro_f1']:.3f}")
    mcols2[2].metric("Anomaly precision", f"{metrics['anomaly_precision']:.3f}")
    mcols2[3].metric("Anomaly recall", f"{metrics['anomaly_recall']:.3f}")

    with st.expander("Raw metric JSON", expanded=False):
        st.json(metrics)

    st.markdown("##### Correct vs wrong (counts)")
    chart_df = pd.DataFrame(
        {
            "Correct": [
                metrics["fault_correct_count"],
                metrics["root_correct_count"],
                metrics["anomaly_correct_count"],
            ],
            "Wrong": [
                metrics["fault_wrong_count"],
                metrics["root_wrong_count"],
                metrics["anomaly_wrong_count"],
            ],
        },
        index=["Fault type", "Root cause", "Anomaly flag"],
    )
    st.bar_chart(chart_df)

    wrong_fault = df[~df["fault_correct"]].copy()
    wrong_root = df[~df["root_correct"]].copy()
    wrong_anom = df[~df["anomaly_correct"]].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Fault mismatches**")
        st.caption(f"{len(wrong_fault)} incidents")
        if not wrong_fault.empty:
            st.dataframe(
                wrong_fault[
                    ["incident_id", "true_fault_type", "pred_fault_type", "fault_confidence"]
                ],
                use_container_width=True,
                height=200,
            )
    with c2:
        st.markdown("**Root-cause mismatches**")
        st.caption(f"{len(wrong_root)} incidents")
        if not wrong_root.empty:
            st.dataframe(
                wrong_root[
                    [
                        "incident_id",
                        "true_root_cause_service",
                        "pred_root_cause_service",
                        "root_confidence",
                    ]
                ],
                use_container_width=True,
                height=200,
            )
    with c3:
        st.markdown("**Anomaly mismatches**")
        st.caption(f"{len(wrong_anom)} incidents")
        if not wrong_anom.empty:
            st.dataframe(
                wrong_anom[
                    [
                        "incident_id",
                        "true_anomalous",
                        "pred_anomalous",
                        "anomaly_score",
                    ]
                ],
                use_container_width=True,
                height=200,
            )

    st.markdown("##### All Test incidents (full table)")
    st.dataframe(df, use_container_width=True, height=400)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="test_set_predictions.csv",
        mime="text/csv",
    )


def main():
    render_header()
    mode = render_mode_selector()
    render_training_panel()

    if not models_ready():
        st.info("Baseline models are not trained yet. Use the **Training setup** expander in the sidebar.")
        return

    if mode == "Dataset incident":
        incidents = get_incidents()
        metrics = get_metrics()
        incident_id = render_incident_selector(incidents)
        incident = incidents.loc[incidents["incident_id"] == incident_id].iloc[0]
        incident_metrics = metrics.loc[metrics["incident_id"] == incident_id].copy()
        result = triage_incident(incident_id)
        # Ground truth from the dataset (Train or Test) — drives match/mismatch in triage output.
        result["expected"] = {
            "fault_type": incident["fault_type"],
            "root_cause_service": incident["root_cause_service"],
            "is_anomalous": bool(incident["is_anomalous"]),
        }

        render_incident_overview(incident)
        render_metric_charts(incident_metrics)
        render_triage_output(result)
        return

    if mode == "Live ingest (HTTP)":
        refresh_mode = render_live_ingest_panel()
        interval_map = {"Manual (button)": 0, "Every 5s": 5, "Every 10s": 10, "Every 30s": 30}
        interval_sec = interval_map.get(refresh_mode, 0)

        if interval_sec > 0:
            st.caption(f"Auto-refreshing every {interval_sec}s. Select **Manual (button)** to pause.")

            @st.fragment(run_every=timedelta(seconds=interval_sec))
            def _live_auto():
                result = _run_live_ingest_triage()
                if result is not None:
                    render_triage_output(result)

            _live_auto()
        else:
            if st.button("Fetch latest window & run triage", type="primary", key="live_fetch_btn"):
                result = _run_live_ingest_triage()
                if result is not None:
                    render_triage_output(result)
            st.caption(
                "Tip: start fault_lab (`docker compose -f fault_lab/docker-compose.yml up`), "
                "then browse the storefront."
            )
        return

    if mode == "Test set evaluation (batch)":
        render_test_set_evaluation()
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
