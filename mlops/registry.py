"""File-based MLflow registry for the logistic baseline.

Versions and the champion alias live in a local tracking directory.
Pointing the alias at an older version is the rollback this module can do.
It does not deploy that version anywhere.
"""

from __future__ import annotations

MODEL_NAME = "risklens-baseline"
CHAMPION = "champion"


# Save the baseline in a local tracking directory and point champion at that version. deployed stays false.
def register_baseline(tracking_uri: str, model, metrics: dict) -> dict:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("risklens-baseline")
    with mlflow.start_run(run_name="baseline_logistic") as run:
        for key in ("precision", "recall", "f1", "pr_auc"):
            mlflow.log_metric(key, float(metrics[key]))
        mlflow.set_tag("served_action", "none")
        info = mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id
        model_uri = info.model_uri
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException as exc:
        if "already exists" not in str(exc).lower():
            raise
    version = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=run_id)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version.version)
    return {
        "name": MODEL_NAME,
        "version": str(version.version),
        "alias": CHAMPION,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "deployed": False,
    }


# Point the alias at a version. That is the rollback this module can do. It does not deploy the version.
def set_alias(tracking_uri: str, version: str, alias: str = CHAMPION) -> None:
    from mlflow.tracking import MlflowClient

    MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(MODEL_NAME, alias, str(version))


# Which version the alias names. Missing alias means there is nothing to read. It does not mean a deployment.
def alias_version(tracking_uri: str, alias: str = CHAMPION) -> str | None:
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    try:
        found = MlflowClient(tracking_uri=tracking_uri).get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None
    return str(found.version)
