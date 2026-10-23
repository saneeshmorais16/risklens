# Technology coverage

| Requirement | Status | Where |
|---|---|---|
| Python | Implemented and tested | `training/`, `backend/`, `agents/` |
| PyTorch | Implemented and tested | `training/train.py` RiskNet |
| Transformer | Implemented and tested | `training/train.py` narrative encoder |
| Synthetic transactions | Implemented and tested | `training/generate.py` |
| Temporal split, no label leakage into features | Implemented and tested | `training/train.py`, `tests/test_phase1.py` |
| Precision, recall, F1, PR-AUC, confusion | Measured on days 25–30 | `models/metrics.json` |
| FastAPI scoring | Implemented and tested | `backend/risklens_api/main.py` |
| PostgreSQL | Live check passed | `mlops/score_log.py` writes `score_events` when `DATABASE_URL` is postgresql. `tests/test_live_postgres.py` passed 3 tests against `postgres:16-alpine`. Compose also stored a scored row with `action` `none` |
| Cloud database support | Configured, not provisioned | Same `DATABASE_URL`. No Cloud SQL instance was created |
| Docker | Desktop 4.96.3 running on WSL 2.7.13 | Engine 29.9.2. `risklens-api` built. Compose `/health` returned `{"ok":true,"model":"baseline_logistic"}`. The stack was stopped after the check |
| React + TypeScript dashboard | Existing console plus Model lab | `src/components/risk-console.tsx` |
| Next.js | Not used | The running UI is TanStack Start |
| Local investigation agent | Implemented and tested | `agents/react_agent.py` allow-listed tools, audit line |
| RAG over fictional policies | TF-IDF index, seven notes, tested | `rag/index.py`, `rag/policies/` |
| ReAct with an LLM | Local four-step loop is tested. An LLM does not choose the tools | `agents/react_agent.py` |
| MLflow model registry | Local file store, tested | `mlops/registry.py`. Champion alias can move back to an older version. Not a remote tracking server |
| Feast feature registry | Local provider, tested | `mlops/feature_registry.py`. Parquet offline store, SQLite online store. Not a feature server |
| Model monitoring and drift | Book PSI, per-model PSI after 30 rows, and a look when those PSIs disagree | `mlops/drift.py` `version_drift` and `look_at`, `GET /v1/monitor`. A model under 30 rows is not the quiet comparison. The route and the champion alias stay put |
| GitHub Actions | Python job passed. UI job failed, lockfile updated locally | `.github/workflows/ci.yml`. Run `39715178903` passed the Python tests and failed `npm ci`. `package-lock.json` now keeps Ajv 6 for ESLint and Ajv 8 for the optional peer. `npm ci` succeeded with npm 10.9.9 on this machine. The workflow has not been re-run, because this lockfile has not been pushed |
| Canary, shadow, and A/B | Local in-process split, tested | `mlops/deploy.py`, `GET /v1/route`, `POST /v1/route/score`. The route pins two registry versions. Shadow scores stay off the served decision. `cluster` is false |
| KServe, Vertex, Kubernetes | Not implemented | No cluster and no cloud project were provisioned |
