# TriageAI

TriageAI is a starter incident-triage project focused on:

- anomaly detection from incident metrics
- fault-type classification
- root-cause service prediction
- similar-incident retrieval
- local retrieval-only RAG for supporting context

This scaffold is built to get a course project moving quickly. It includes:

- a synthetic starter dataset generator
- feature extraction from incident metrics
- baseline machine learning training scripts
- evaluation scripts
- a free local retrieval index built from incident cases and curated notes
- a Streamlit demo app

## Project Structure

```text
.
├── app.py
├── data
│   ├── processed
│   │   ├── incidents.csv
│   │   └── metrics.csv
│   └── raw
├── models
├── requirements.txt
└── src
    ├── config.py
    ├── data.py
    ├── eval.py
    ├── features.py
    ├── generate_sample_data.py
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
python3 -m src.data_converter
```

Generate processed data, train baseline models, and build the retrieval index:

```bash
python3 -m src.train_models
```

Run evaluation:

```bash
python3 -m src.eval
```

Start the app:

```bash
streamlit run app.py
```

Run the companion simulator app:

```bash
streamlit run simulator_app.py
```

## Notes

- If no dataset exists yet, the code auto-generates a synthetic starter dataset in `data/processed/`.
- The current scaffold does not train remediation-action recommendations because the project scope was narrowed to tasks that public AIOps datasets can support more honestly.
- Replace the synthetic dataset with your selected public dataset once your preprocessing pipeline is ready.
- The starter scaffold is pinned for Python 3.8+.
- The current RAG layer is retrieval-only and fully local. It does not require any paid LLM API.
- `simulator_app.py` provides a small ecommerce-like app that generates TriageAI-compatible telemetry and can deliberately inject failure scenarios such as CPU exhaustion, memory leak, queue congestion, auth failure, dependency outage, and cascading failure.
