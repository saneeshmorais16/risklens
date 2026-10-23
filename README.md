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

It listens on port 8080. Voice answers need `XAI_API_KEY`. `npm run dev` reads it from `.env` when the variable is not already set. Do not commit that key.

`python/brief_server.py` answers the first line of an open case on port 8091, before the spoken reply. The dev startup script launches it.

## Scoring API

The holdout numbers in the Model lab are the last local training run, written to `models/metrics.json`. They are not targets.

```bash
PYTHONPATH=. python -m training.train
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. uvicorn backend.risklens_api.main:app --port 8000
```

`POST /v1/score` returns a probability and whether an analyst should look. It does not freeze an account. PyTorch is installed from the CPU wheel index, not from the default pip index.

`docker compose up --build` starts the API and PostgreSQL. The API writes scores to PostgreSQL when `DATABASE_URL` is set, and to SQLite when it is not. Kubernetes and Vertex AI are not part of this build. See [docs/COVERAGE.md](docs/COVERAGE.md).

## Phase 3 registries and drift

```bash
pip install -r backend/requirements.txt -r requirements-mlops.txt
python -m pytest -q
```

MLflow and Feast run on the local disk. `GET /v1/monitor` compares recent scores with `models/reference_profile.json`. `GET /v1/registry` reads a champion alias only after `MLFLOW_TRACKING_URI` points at a file store, and it reports `deployed: false`. Details are in [docs/PHASE3.md](docs/PHASE3.md).

## Phase 4 traffic split

`GET /v1/route` reads a route file from `ROUTE_PATH`. `POST /v1/route/score` loads the two named versions from the local MLflow store and applies a canary, shadow, or A/B weight. The response keeps `action` as `none`, `cluster` as false, and `deployed` as false. The champion alias stays where it was. Details are in [docs/PHASE4.md](docs/PHASE4.md).

## Phase 5 version mix

`GET /v1/monitor` still computes one PSI for the latest scored rows. The same payload lists how many of those rows each stored model served, for example `baseline_logistic` and `baseline_logistic@2`. A short book includes the counts before PSI is ready. Details are in [docs/PHASE5.md](docs/PHASE5.md).

## Phase 6 PSI by version

`GET /v1/monitor` adds `by_version`. A model gets its own PSI only after 30 served rows. A smaller slice stays `ready: false`. The book PSI is still the whole window. `action` stays `none`. Details are in [docs/PHASE6.md](docs/PHASE6.md).

## Phase 7 look

`GET /v1/monitor` adds `look`. It names the ready model whose PSI alerts when another ready model is quiet. A model under 30 rows is not that quiet comparison. The route file and the champion alias stay where they are. Details are in [docs/PHASE7.md](docs/PHASE7.md).

