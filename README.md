# RiskLens

Synthetic transaction review. An Isolation Forest is fit on generated activity and used to flag rows for a person to review. A high score is a reason to look, not a finding of fraud.

## Python service

The original console is Streamlit, with SQLite reviews and Docker. It lives in [python](python/).

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## TypeScript console

The later console (holdout scoring, drift, slice gaps, and the voice assistant) is the TypeScript app at the repository root.

```bash
npm install
npm run dev
```

It listens on port 8080. Voice answers need `XAI_API_KEY` in the environment. Do not commit that key.

`python/brief_server.py` answers the first line of an open case on port 8091, before the spoken reply. The dev startup script launches it.

## Scoring API

The holdout numbers in the Model lab are the last local training run, written to `models/metrics.json`. They are not targets.

```bash
PYTHONPATH=. python -m training.train
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. uvicorn backend.risklens_api.main:app --port 8000
```

`POST /v1/score` returns a probability and whether an analyst should look. It does not freeze an account. PyTorch is installed from the CPU wheel index, not from the default pip index.

`docker compose up --build` is the local API plus PostgreSQL path. PostgreSQL is started by Compose and is not yet read by the API. Kubernetes, Vertex AI, MLflow, and Feast are not part of this build. See [docs/COVERAGE.md](docs/COVERAGE.md).

