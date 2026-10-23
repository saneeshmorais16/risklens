# Phase 6 — PSI by served model

Phase 5 counts rows by model and keeps one PSI for the whole book. Phase 6 adds a PSI for each model that has enough rows of its own.

## What is implemented

- `version_drift` in `mlops/drift.py` groups the same window the monitor already reads. A model with fewer than 30 rows is `ready: false`. A model with 30 or more gets the same feature PSI and score PSI as the book report, computed on that model's rows only.
- `GET /v1/monitor` adds `by_version` and `by_version_note`. The book PSI stays one report for every row in the window. `action` stays `none` on the book and on each model.
- A blank model name still counts as `baseline_logistic`, the same rule as the version mix. Shadow scores are still left out of the score log, so they are left out of this split.

## What was tested

`python -m pytest -q` on Python 3.12 when this phase landed: 30 passed, 3 skipped. A later run with Docker up was 36 passed.

The Phase 6 tests use a reference book fixed at hour 14. Twenty shifted rows plus ten more shifted rows make the whole book ready, and both models stay not ready. Thirty shifted rows of `baseline_logistic` alert on hour and score. Thirty steady rows of `baseline_logistic@2` do not. The book alert remains the mixed window. The live PostgreSQL check has since passed. WSL 2.7.13 is installed and Docker Desktop 4.96.3 is running.

## What this slice does not do

It does not move the champion alias. It does not create a cluster. It does not log a shadow score. A model under 30 rows does not get a PSI. Naming which model to look at is Phase 7.
