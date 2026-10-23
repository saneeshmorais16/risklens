# Phase 7 — which model to look at

Phase 6 reports a PSI per model once that model has 30 rows. Phase 7 names the model to inspect when those PSIs disagree.

## What is implemented

- `look_at` in `mlops/drift.py` reads the `by_version` list. It names a model only when at least one ready model has alerts and at least one ready model has none. The named models are the alerting ones.
- A model under 30 rows is not ready, so it is not the quiet comparison. Two alerting models are not a disagreement. A window with one ready model is not a disagreement.
- `GET /v1/monitor` adds `look`. `alias_moved` and `route_changed` stay false. `action` stays `none`. The route file is not written, and the champion alias is not read.

## What was tested

`python -m pytest -q` on Python 3.12 when this phase landed: 33 passed, 3 skipped. A later run with Docker up was 36 passed.

The Phase 7 tests cover a short model that cannot stand in as the quiet one, a shifted `baseline_logistic` named beside a steady `baseline_logistic@2`, and two shifted models that produce no look. The route file bytes are the same after the monitor call. The live PostgreSQL check has since passed. WSL 2.7.13 is installed and Docker Desktop 4.96.3 is running.

## What this slice does not do

It does not change the route. It does not move the champion alias. It does not log a shadow score. It does not create a cluster.
