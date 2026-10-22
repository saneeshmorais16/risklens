# RiskLens — Explainable Transaction Investigation MVP (Phase 1)

An interactive financial anomaly investigation console using **fully synthetic data**, Isolation Forest, and analyst-friendly review context.

**Phase 1 hardening** adds:
- Basic username/password authentication (demo credentials; override via env)
- SQLite persistence of analysis runs and human review outcomes
- Structured logging (stdout + `risklens.log`)
- Docker + docker-compose for containerized deployment
- Improved error handling

## Run locally (Python)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Default login: username `analyst` / password `demo`  
(or set `RISKLENS_PASSWORD` environment variable)

Run tests with `python -m pytest -q`.

## Run with Docker

```bash
docker compose up --build
```

Then open http://localhost:8501

The SQLite database (`risklens.db`) and log file are created inside the container working directory.  
Mount a volume in `docker-compose.yml` if you need persistence across container restarts.

## Features
- Reproducible synthetic transactions with deliberately injected unusual scenarios
- Isolation Forest ranking and adjustable anomaly proportion
- Interactive risk scatterplot, searchable investigation queue and CSV export
- Transaction details with interpretable *heuristic indicators*
- Optional CSV upload with columns `amount,hour,velocity_24h,distance_km,new_device`
- Persist runs and record analyst decisions (approved / rejected / escalated / needs_more_info)

## Important limitations
- Anomaly ≠ fraud. This is not suitable for automated adverse decisions.
- Data are synthetic; injected scenario flags are not verified fraud labels.
- Review reasons are threshold-based context and **not** faithful feature attributions for Isolation Forest.
- Model is fitted to the current dataset, so rankings can change as the input changes.
- Authentication is basic (single demo user). No production-grade secrets management, encryption, or multi-tenant controls yet.
- SQLite is suitable for single-instance / low-concurrency use only.
- No authentication, persistence, privacy controls or production monitoring in the original MVP (partially addressed in Phase 1).

## Next phases
- Phase 2: Split scoring into a FastAPI service, proper model registry, PostgreSQL, OIDC auth, temporal evaluation
- Phase 3: Full observability, drift/fairness monitoring, audit immutability, compliance controls, production deployment (K8s / managed)

Human-review outcomes and audit trail; feature validation; temporal train/test split; evaluation against verified labels; fairness/drift monitoring; authenticated API and database.
