# Phase 5 — served version mix

Phase 4 writes the served model on each scored row. Phase 5 counts those rows. PSI stays one number for the whole book.

## What is implemented

- `version_mix` in `mlops/score_log.py` groups the score log by the `model` column. A blank model is counted as `baseline_logistic`. Shares sum to 1. The largest count is first, and equal counts are ordered by name.
- `GET /v1/monitor` adds `versions` and `versions_note` to the existing report. The feature PSI and the score PSI are still computed once, on every row in the window.
- A book with fewer than 30 rows is still `ready: false`. That payload includes the same counts. `action` stays `none`.

## What was tested

`python -m pytest -q` on Python 3.12 when this phase landed: 28 passed, 3 skipped. A later run with Docker up was 36 passed.

The Phase 5 tests cover a mix ordered by count, a blank model counted as the baseline, the short-book payload, and a 30-row monitor report whose PSI alerts are for the whole book. The version list on that report is 20 rows of `baseline_logistic` and 10 of `baseline_logistic@2`. The live PostgreSQL check from Phase 4 has since passed. WSL 2.7.13 is installed and Docker Desktop 4.96.3 is running.

## What this slice does not do

Per-version PSI is Phase 6. This phase only counts. Shadow scores stay out of the score log, so they stay out of the mix. The champion alias is unchanged. No cluster is created.
