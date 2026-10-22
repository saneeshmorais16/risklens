# Phase 4 — local traffic split

Phase 3 registers a champion and measures drift. Phase 4 chooses which registered version scores a row. The choice is canary, shadow, or A/B. A score still asks for a look. The field `action` is `none`. The split runs in this process, and `cluster` stays false.

## What is implemented

- `mlops/deploy.py` maps a key into a unit interval with SHA-256. The same key always lands in the same place. `candidate_weight` is the share of keys that select the candidate.
- Canary and A/B serve the selected version. They use the same split. The mode is stored so a canary slice stays a canary slice, and an A/B split stays an experiment.
- Shadow serves the primary. For keys inside the weight it also scores the candidate. `shadow_served` is false. The shadow score is left out of the score log.
- The route names two versions in the file at `ROUTE_PATH`. It does not follow the champion alias. A score does not move that alias, and moving the alias leaves the route on the versions it names.
- `GET /v1/route` reports the file. `POST /v1/route/score` loads those versions from the local MLflow file store. Both payloads keep `action` as `none`, `cluster` as false, `deployed` as false, and `alias_moved` as false.
- The served score is appended to the same log as `/v1/score`, with `model` set to `baseline_logistic@<version>`. `GET /v1/monitor` still reads that log as one book. It does not compute PSI per version.
- `tests/test_live_postgres.py` is the live PostgreSQL check. When `docker info` succeeds it starts `postgres:16-alpine`, writes through the score log and the API, and removes the container.

## What was tested

`python -m pytest -q` on Python 3.12: 26 passed, 3 skipped.

The Phase 4 tests cover a stable split at the requested weight, shadow serving the primary while scoring the candidate, a route that is rejected unless the two versions differ, the route endpoints, a pinned version that stays put when the champion alias moves, and a load of the registered baseline. Served rows keep `action` as `none`. The route payloads keep `cluster` and `deployed` false.

The three skipped tests are `tests/test_live_postgres.py`. Docker is not installed on this machine, so PostgreSQL was not started and Compose was not run.

## What was not deployed

No Kubernetes Service, KServe InferenceService, or cloud endpoint was created. The champion alias is a local registry pointer. Vertex AI remains unused.
