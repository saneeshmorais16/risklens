# Phase 3 — local registry, features, and drift

Phase 1 scores a held-out synthetic book and refuses account actions. Phase 2 retrieves policy notes through a four-step allow-listed loop. Phase 3 adds the MLOps pieces that can run on one machine.

## What is implemented

- `mlops/registry.py` logs the saved logistic baseline to a local MLflow file store, registers `risklens-baseline`, and points the `champion` alias at a version. Moving that alias back to an older version is the rollback.
- `mlops/feature_registry.py` registers the six scoring features in Feast. The offline store is parquet. The online store is SQLite. The label and the scenario name are not features.
- `mlops/drift.py` computes population stability on those features and on the served score. The alert band is PSI ≥ 0.25.
- `mlops/score_log.py` appends each `/v1/score` call. SQLite is the default. `DATABASE_URL=postgresql://...` selects PostgreSQL, which is also how a managed Postgres such as Cloud SQL would be addressed. Nothing in this phase creates a cloud database.
- `GET /v1/monitor` compares the latest scored rows with `models/reference_profile.json`.
- `GET /v1/registry` reads the champion alias when `MLFLOW_TRACKING_URI` is set. The payload says `deployed: false`.

## What was tested

`python -m pytest -q` on Python 3.12: 18 passed. That includes `tests/test_phase1.py`, `tests/test_phase2.py`, and `tests/test_phase3.py`. The Phase 3 tests cover a stable PSI, a shifted hour, the checked-in training-window profile (7,646 rows, the same count as `models/metrics.json`), the SQLite score log, an MLflow champion alias moved back to an older version, a Feast online read of the later row, and the monitor and registry endpoints. Both endpoints keep `action` as `none`, and the registry endpoint keeps `deployed` as false.

PostgreSQL and Docker were not executed in this phase. Both were run later. The live PostgreSQL tests passed, and Compose served the API. See [COVERAGE.md](COVERAGE.md).

## What was not deployed

MLflow is a directory, not a tracking server. Feast is the local provider, not a feature server. PostgreSQL is used only when a URL is set and the server accepts a connection. Vertex AI, KServe, and Kubernetes are not in this phase. The local canary, shadow, and A/B split is Phase 4.

`models/reference_profile.json` is the training window (days 1–21) of `generate(n_customers=360, seed=42)`, scored by the checked-in baseline. It is a measured profile of that synthetic book.
