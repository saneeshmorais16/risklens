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
| PostgreSQL | Compose service only | `docker-compose.yml`. The API does not use it yet |
| Docker | Dockerfile and Compose written | Not executed in this environment |
| React + TypeScript dashboard | Existing console plus Model lab | `src/components/risk-console.tsx` |
| Next.js | Not used | The running UI is TanStack Start |
| Local investigation agent | Implemented and tested | `agents/react_agent.py` allow-listed tools, audit line |
| RAG over fictional policies | TF-IDF index, seven notes, tested | `rag/index.py`, `rag/policies/` |
| ReAct with an LLM | Local four-step loop is tested. An LLM does not choose the tools | `agents/react_agent.py` |
| MLflow, Feast, KServe, Vertex, Prometheus, Grafana, Kubernetes | Not implemented | No deployment was attempted |
| GitHub Actions | Workflow file added | Not executed here |
| CI/CD cloud deploy, canary, shadow, A/B | Not implemented | |
