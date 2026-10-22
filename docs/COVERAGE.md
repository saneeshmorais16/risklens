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
| PostgreSQL | Live check written, not executed here | `mlops/score_log.py` writes `score_events` when `DATABASE_URL` is postgresql. `tests/test_live_postgres.py` starts PostgreSQL when `docker info` succeeds. Docker is not installed on this machine, so that test was skipped and no server was contacted |
| Cloud database support | Configured, not provisioned | Same `DATABASE_URL`. No Cloud SQL instance was created |
| Docker | Dockerfile and Compose written | Docker is not installed in this environment, so the image was not built and the live PostgreSQL test was skipped |
| React + TypeScript dashboard | Existing console plus Model lab | `src/components/risk-console.tsx` |
| Next.js | Not used | The running UI is TanStack Start |
| Local investigation agent | Implemented and tested | `agents/react_agent.py` allow-listed tools, audit line |
| RAG over fictional policies | TF-IDF index, seven notes, tested | `rag/index.py`, `rag/policies/` |
| ReAct with an LLM | Local four-step loop is tested. An LLM does not choose the tools | `agents/react_agent.py` |
| MLflow model registry | Local file store, tested | `mlops/registry.py`. Champion alias can move back to an older version. Not a remote tracking server |
| Feast feature registry | Local provider, tested | `mlops/feature_registry.py`. Parquet offline store, SQLite online store. Not a feature server |
| Model monitoring and drift | PSI on features and scores, tested | `mlops/drift.py`, `GET /v1/monitor`, `models/reference_profile.json` |
| GitHub Actions | Workflow installs MLOps requirements and runs pytest | `.github/workflows/ci.yml`. This environment did not push, so the workflow was not executed here |
| Canary, shadow, and A/B | Local in-process split, tested | `mlops/deploy.py`, `GET /v1/route`, `POST /v1/route/score`. The route pins two registry versions. Shadow scores stay off the served decision. `cluster` is false |
| KServe, Vertex, Kubernetes | Not implemented | No cluster and no cloud project were provisioned |
