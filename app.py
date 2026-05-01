from __future__ import annotations

from datetime import timedelta

import joblib

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_LIVE_CONTROL_PLANE_URL,
    FAULT_MODEL_PATH,
    METRIC_COLUMNS,
    ROOT_CAUSE_MODEL_PATH,
)
from src.llm_explanations import (
    DEFAULT_GEMINI_MODEL,
    explain_triage_result,
    load_cached_gemini_key,
    save_cached_gemini_key,
)
from src.live_telemetry import fetch_telemetry_window
from src.data import load_incidents, load_metrics
from src.test_set_eval import evaluate_test_split
from src.train_models import train_all_models
from src.triage import clear_caches, models_ready, triage_custom_metrics, triage_incident


st.set_page_config(page_title="TriageAI", layout="wide")

MODE_HISTORICAL = "Historical incident"
MODE_UPLOAD = "Upload telemetry CSV"
MODE_LIVE = "Live telemetry"
MODE_EVALUATION = "Model evaluation"


def display_label(value) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def display_metric_name(value: str) -> str:
    replacements = {
        "pct": "%",
        "ms": "ms",
        "cpu": "CPU",
        "io": "I/O",
        "p50": "p50",
    }
    parts = str(value).replace("_", " ").split()
    return " ".join(replacements.get(part, part.title()) for part in parts)


def is_healthy_none_result(result: dict) -> bool:
    fault = str(result.get("predicted_fault_type", "")).lower()
    service = str(result.get("predicted_root_cause_service", "")).lower()
    return (
        fault == "healthy"
        and service in {"none", "", "unknown", "unlabeled"}
        and not bool(result.get("unusual", False))
    )


def is_unmatched_anomaly_result(result: dict) -> bool:
    fault = str(result.get("predicted_fault_type", "")).lower()
    service = str(result.get("predicted_root_cause_service", "")).lower()
    return (
        fault == "healthy"
        and service in {"none", "", "unknown", "unlabeled"}
        and bool(result.get("unusual", False))
    )


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
    with st.sidebar.expander("Model management", expanded=not models_ready()):
        st.write(
            "Refresh the anomaly, classification, root-cause, and reference-case models."
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
        st.caption("Anomaly detection is trained on normal reference windows.")

        use_hp_search = st.checkbox(
            "Extended model search",
            value=False,
            help=(
                "Runs a broader search over classifier settings. This is slower and intended for model refreshes."
            ),
        )

        active_config = get_active_classifier_config()
        if active_config:
            st.write(
                "Active model profile: "
                f"fault={display_label(active_config['fault_classifier_name'])}, "
                f"root cause={display_label(active_config['root_cause_classifier_name'])}"
            )

        train_clicked = st.button("Refresh models", use_container_width=True)

    if train_clicked:
        spinner_msg = (
            "Running extended model search..."
            if use_hp_search
            else "Refreshing models..."
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
            "Model refresh complete. "
            f"fault={display_label(summary['fault_classifier_name'])}, "
            f"root cause={display_label(summary['root_cause_classifier_name'])}"
        )
        ft = summary.get("fault_tuning") or {}
        rt = summary.get("root_cause_tuning") or {}
        if ft.get("best_cv_score") is not None:
            msg += f" | fault validation={ft['best_cv_score']:.4f}"
        if rt.get("best_cv_score") is not None:
            msg += f" | root-cause validation={rt['best_cv_score']:.4f}"
        st.sidebar.success(msg)


def render_explanation_panel():
    with st.sidebar.expander("Explanation assistant", expanded=False):
        st.caption(
            "Optional external language model support for incident summaries."
        )
        cached_key = load_cached_gemini_key()
        if cached_key and not st.session_state.get("gemini_api_key"):
            st.session_state["gemini_api_key"] = cached_key
        st.checkbox(
            "Use Gemini LLM explanations",
            value=False,
            key="use_gemini_explanation",
            help="Uses the configured external language model. The app falls back to the built-in explanation if unavailable.",
        )
        entered_key = st.text_input(
            "Explanation API key",
            value=st.session_state.get("gemini_api_key", ""),
            key="gemini_api_key",
            type="password",
            help="Optional. You can also set this in the environment.",
        )
        if entered_key:
            save_cached_gemini_key(entered_key)
            st.caption("Gemini key loaded. Enable the Gemini checkbox above to use it.")
        elif cached_key:
            st.caption("Cached Gemini key found. Enable the Gemini checkbox above to use it.")
        st.text_input(
            "Explanation model",
            value=DEFAULT_GEMINI_MODEL,
            key="gemini_model",
        )


def render_incident_selector(incidents):
    split_options = ["All", *sorted(str(split) for split in incidents["data_split"].dropna().unique())]
    selected_split = st.sidebar.selectbox(
        "Data split",
        options=split_options,
        index=0,
        key="incident_split_filter",
    )
    sort_by = st.sidebar.selectbox(
        "Sort incidents by",
        options=["Split, then ID", "ID", "Fault type", "Root-cause service"],
        index=0,
        key="incident_sort_by",
    )

    filtered = incidents.copy()
    if selected_split != "All":
        filtered = filtered.loc[filtered["data_split"].astype(str).eq(selected_split)].copy()

    sort_columns = {
        "Split, then ID": ["data_split", "incident_id"],
        "ID": ["incident_id"],
        "Fault type": ["fault_type", "incident_id"],
        "Root-cause service": ["root_cause_service", "incident_id"],
    }[sort_by]
    filtered = filtered.sort_values(sort_columns).reset_index(drop=True)

    incident_label_map = {
        row.incident_id: (
            f"{row.incident_id} | {row.data_split} | "
            f"{display_label(row.fault_type)} / {display_label(row.root_cause_service)}"
        )
        for row in filtered.itertuples(index=False)
    }
    selected = st.sidebar.selectbox(
        "Select incident",
        options=list(incident_label_map.keys()),
        format_func=lambda incident_id: incident_label_map[incident_id],
    )
    selected_row = filtered.loc[filtered["incident_id"].eq(selected)].iloc[0]
    st.sidebar.caption(f"Title: {selected_row['title']}")
    return selected


def render_mode_selector():
    st.sidebar.header("Input Mode")
    return st.sidebar.radio(
        "Choose data source",
        options=[
            MODE_HISTORICAL,
            MODE_UPLOAD,
            MODE_LIVE,
            MODE_EVALUATION,
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
    st.sidebar.subheader("Telemetry Upload")
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
    title = st.sidebar.text_input("Incident title", value="Uploaded incident")
    description = st.sidebar.text_area(
        "Incident description",
        value="Paste a short summary of what the application is doing or failing to do.",
        height=100,
    )
    uploaded_file = st.sidebar.file_uploader("Upload metric CSV", type="csv")
    return title, description, uploaded_file

def render_incident_overview(incident):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reference fault", display_label(incident["fault_type"]))
    col2.metric("Reference service", display_label(incident["root_cause_service"]))
    col3.metric("Reference anomaly", "Yes" if incident["is_anomalous"] else "No")
    col4.metric("Data split", incident.get("data_split", "Unknown"))

    st.subheader("Incident summary")
    st.write(f"**Title:** {incident['title']}")
    st.write(incident["description"])


def render_live_ingest_panel():
    st.sidebar.subheader("Live Telemetry")
    st.sidebar.caption(
        "Pulls a rolling metrics window from a compatible telemetry endpoint."
    )
    st.sidebar.text_input(
        "Telemetry endpoint",
        value=DEFAULT_LIVE_CONTROL_PLANE_URL,
        key="live_base_url",
        help="Base URL for the telemetry provider.",
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
        value="Live telemetry window",
        key="live_incident_title",
    )
    st.sidebar.text_area(
        "Incident description",
        value="Rolling telemetry window ingested from the connected environment.",
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
            "Raise this threshold to reduce borderline anomaly alerts in noisier environments."
        ),
    )
    return refresh_mode


def render_custom_incident_overview(title: str, description: str):
    col1, col2, col3 = st.columns(3)
    col1.metric("Source", "Uploaded CSV")
    col2.metric("Reference labels", "Not provided")
    col3.metric("Evaluation", "Prediction only")

    st.subheader("Incident summary")
    st.write(f"**Title:** {title}")
    st.write(description)
    st.warning(
        "These predictions are being applied to unseen external telemetry. "
        "Use them as triage guidance and confirm against operational context before acting."
    )


def render_live_ingest_overview(
    title: str,
    description: str,
    control_plane_url: str,
    *,
    meta: dict | None = None,
):
    col1, col2, col3 = st.columns(3)
    col1.metric("Source", "Live telemetry")
    col2.metric("Connection", "Active")
    col3.metric(
        "Window size",
        "Rolling",
    )
    st.caption(f"Telemetry source: `{control_plane_url}`")

    st.subheader("Incident summary")
    st.write(f"**Title:** {title}")
    st.write(description)
    st.info(
        "Telemetry is pulled from the connected metrics endpoint and scored as a rolling incident window."
    )


def render_metric_charts(metrics):
    st.subheader("Metric trends")
    chart_columns = [column for column in METRIC_COLUMNS if column in metrics.columns]
    st.line_chart(metrics.set_index("minute")[chart_columns])


def _run_live_ingest_triage():
    """Fetch the configured live telemetry window and run triage."""
    base_url = st.session_state.get("live_base_url", DEFAULT_LIVE_CONTROL_PLANE_URL)
    limit = int(st.session_state.get("live_limit", 120))
    title = st.session_state.get("live_incident_title", "Live telemetry window")
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
        precomputed_scalars=meta.get("precomputed_scalars"),
    )
    return result


def render_triage_output(result):
    st.subheader("Triage output")
    raw = result.get("unusual_raw")
    if raw is not None and raw != result["unusual"]:
        st.caption(
            "The window was borderline anomalous, but it did not clear the configured alert threshold."
        )
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly flag", "Abnormal" if result["unusual"] else "Normal")
    col2.metric("Fault prediction", display_label(result["predicted_fault_type"]))
    col3.metric("Root-cause prediction", display_label(result["predicted_root_cause_service"]))

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
            f"Reference fault: `{display_label(expected_fault)}` -> predicted `{display_label(result['predicted_fault_type'])}` "
            f"({'match' if fault_match else 'mismatch'})",
            f"Reference service: `{display_label(expected_service)}` -> predicted `{display_label(result['predicted_root_cause_service'])}` "
            f"({'match' if service_match else 'mismatch'})",
        ]
        if has_truth_anomaly:
            msg_parts.insert(
                0,
                f"Reference anomaly: `{'Abnormal' if truth_anom else 'Normal'}` -> "
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

    st.write(
        f"**Anomaly score:** {result['anomaly_score']:.3f} | "
        f"**Fault confidence:** {result['fault_confidence']:.3f} | "
        f"**Root-cause confidence:** {result['root_cause_confidence']:.3f}"
    )

    signal_highlights = result.get("signal_highlights") or []
    if signal_highlights:
        st.subheader("Top signal changes")
        signal_df = pd.DataFrame(signal_highlights).copy()
        signal_df["metric"] = signal_df["metric"].map(display_metric_name)
        signal_df = signal_df[
            ["metric", "max", "mean", "delta", "spike", "slope", "score"]
        ]
        signal_df = signal_df.rename(
            columns={
                "metric": "Signal",
                "max": "Max",
                "mean": "Mean",
                "delta": "Delta",
                "spike": "Spike",
                "slope": "Slope",
                "score": "Impact",
            }
        )
        st.dataframe(signal_df, use_container_width=True, hide_index=True)

    healthy_none = is_healthy_none_result(result)
    unmatched_anomaly = is_unmatched_anomaly_result(result)
    if healthy_none:
        st.info("No incident pattern detected. Reference cases and supporting context are hidden for healthy windows.")
    elif unmatched_anomaly:
        st.warning(
            "Unusual telemetry detected, but no known fault pattern matched confidently. Review the signal changes before escalating."
        )

    if result["top_similar_incidents"] and not healthy_none:
        st.subheader("Reference cases")
        for item in result["top_similar_incidents"]:
            st.write(
                f"- `{item['incident_id']}` | similarity={item['similarity']:.3f} | "
                f"fault={display_label(item['fault_type'])} | service={display_label(item['root_cause_service'])}"
            )

    retrieved_context = result.get("retrieved_context", {})
    if retrieved_context.get("documents") and not healthy_none:
        st.subheader("Supporting context")
        st.caption(
            "Showing supporting references for "
            f"{display_label(result['predicted_fault_type'])} in "
            f"{display_label(result['predicted_root_cause_service'])}."
        )
        for item in retrieved_context["documents"]:
            st.write(
                f"**{item['title']}** "
                f"(relevance={item['score']:.3f})"
            )
            st.write(item["content"])

    st.subheader("Incident explanation")
    use_gemini = bool(st.session_state.get("use_gemini_explanation", False))
    if use_gemini:
        with st.spinner("Generating enhanced explanation..."):
            explanation = explain_triage_result(
                result,
                use_gemini=True,
                model=st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL),
                api_key=st.session_state.get("gemini_api_key") or None,
            )
        if explanation["ok"]:
            st.caption(f"Enhanced explanation from Gemini ({explanation['model']}).")
        else:
            st.warning(
                "Enhanced explanation is currently unavailable. Showing the built-in incident explanation."
            )
            if explanation.get("error"):
                st.caption(f"Gemini error: {explanation['error']}")
    else:
        explanation = explain_triage_result(result, use_gemini=False)
        st.caption("Built-in incident explanation.")
    st.write(explanation["text"])


def render_test_set_evaluation():
    st.subheader("Holdout evaluation")
    st.caption(
        "Scores held-out incidents with the active model pipeline and compares predictions to reference labels."
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

    if st.button("Run holdout evaluation", type="primary", key="batch_eval_btn"):
        with st.spinner("Running holdout evaluation..."):
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
        st.info("Click **Run holdout evaluation** to compute metrics, charts, and per-incident results.")
        return

    metrics = st.session_state["batch_eval_metrics"]
    df = st.session_state["batch_eval_df"]

    st.markdown("##### Aggregate metrics")
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

    st.markdown("##### Incident-level results")
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
    render_explanation_panel()

    if not models_ready():
        st.info("Models are not ready yet. Use **Model management** in the sidebar to refresh them.")
        return

    if mode == MODE_HISTORICAL:
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

    if mode == MODE_LIVE:
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
            st.caption("Connect a compatible telemetry source, then fetch the latest window.")
        return

    if mode == MODE_EVALUATION:
        render_test_set_evaluation()
        return

    title, description, uploaded_file = render_upload_panel()
    if uploaded_file is None:
        st.info("Upload a metric CSV to score an incident window.")
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
