# TriageAI

TriageAI is an incident-triage project focused on:

- anomaly detection from incident metrics
- fault-type classification
- root-cause service prediction
- similar-incident retrieval
- local retrieval-only RAG for supporting context

The project includes:

- RCAEval preprocessing into incident windows
- feature extraction from incident metrics
- train/test-aware model training and evaluation
- a free local retrieval index built from incident cases and curated notes
- a main Streamlit triage app
- a live telemetry simulator app
- a real fault-injected local microservice stack for external testing

## Current Model Setup

- anomaly detection: `IsolationForest`
- fault-type classification: `RandomForestClassifier` with `class_weight="balanced_subsample"`
- root-cause prediction: `RandomForestClassifier` with `class_weight="balanced_subsample"`
- supporting context: local retrieval-only RAG

The main app can score:

- processed RCAEval incidents
- uploaded external telemetry CSVs with the same metric schema

## Project Structure

```text
.
├── app.py
├── simulator_app.py
├── data
│   ├── processed
│   │   ├── incidents.csv
│   │   └── metrics.csv
│   └── raw
├── fault_lab
├── models
├── requirements.txt
└── src
    ├── config.py
    ├── data.py
    ├── data_converter.py
    ├── eval.py
    ├── features.py
    ├── generate_sample_data.py
    ├── rag.py
    ├── simulator.py
    ├── train_models.py
    └── triage.py
```

## Quick Start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Convert RCAEval data into digestible file format:

```bash
./.venv/bin/python -m src.data_converter
```

Generate processed data, train models, and build the retrieval index:

```bash
./.venv/bin/python -m src.train_models
```

Run evaluation:

```bash
./.venv/bin/python -m src.eval
```

Start the app:

```bash
streamlit run app.py
```

Run the companion simulator app:

```bash
streamlit run simulator_app.py
```

Run the real fault-injected stack:

```bash
docker-compose -f fault_lab/docker-compose.yml up -d --build
```

Then open:

- Storefront: `http://localhost:8090`
- Control plane: `http://localhost:8001`

## GitHub Actions

This repository includes a CI workflow at `.github/workflows/ci.yml`.
It runs on push, pull request, and manual dispatch.

The workflow:

- installs the main TriageAI dependencies and Fault Lab dependencies
- compiles and imports the Streamlit, ML, and FastAPI modules
- validates the processed incident and metric dataset
- trains the models and runs evaluation
- smoke-tests dataset and uploaded-CSV triage paths
- validates, builds, starts, and smoke-tests the Fault Lab Docker Compose stack

GitHub Actions is for validation/build automation. It does not permanently host the Streamlit or Fault Lab apps. For a public live demo, deploy the app separately on a hosting service such as Streamlit Community Cloud, Render, Railway, or Hugging Face Spaces.

## Main App

`app.py` supports two input modes:

- `Dataset incident`: inspect RCAEval-derived incidents with true labels and train/test split markers
- `Upload real incident`: upload a CSV with:
  - `minute`
  - `error_rate`
  - `latency_ms`
  - `cpu_pct`
  - `memory_pct`
  - `queue_depth`
  - `auth_error_rate`

The sidebar also lets you retrain with different classifier families for comparison, although the default selected model is the weighted Random Forest variant that performed best in evaluation.

## Evaluation

Processed incidents carry a `data_split` field. Training uses only `Train` incidents, and evaluation runs against held-out `Test` incidents.

Current default evaluation results are written to `models/eval_summary.json`.

## Notes

- If raw RCAEval data is available in `data/raw/`, the code converts it into processed incidents and metrics. If not, it falls back to synthetic starter data in `data/processed/`.
- The current scaffold does not train remediation-action recommendations because the project scope was narrowed to tasks that public AIOps datasets can support more honestly.
- The starter scaffold is pinned for Python 3.8+.
- The current RAG layer is retrieval-only and fully local. It does not require any paid LLM API.
- `simulator_app.py` provides a small ecommerce-like app that generates TriageAI-compatible telemetry and can deliberately inject failure scenarios such as CPU exhaustion, memory leak, queue congestion, auth failure, dependency outage, and cascading failure.
- `fault_lab/` contains a real local multi-service stack with deliberate fault injection and telemetry export. See `fault_lab/README.md` for service-level details.
