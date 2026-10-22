"""Local Feast registry for the scoring features.

The online store is SQLite and the offline store is a parquet file.
Labels and scenario names are not registered. This is not a Feast server cluster.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from features.build import FEATURE_COLUMNS, add_features

VIEW_NAME = "transaction_features"


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    table = add_features(frame).copy()
    origin = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=40)
    table["event_timestamp"] = origin + pd.to_timedelta(table["day"].astype(int), unit="D")
    columns = ["customer_id", "event_timestamp", *FEATURE_COLUMNS]
    return table.loc[:, columns]


def feature_view(parquet_path: Path):
    from feast import Entity, FeatureView, Field, FileSource
    from feast.types import Float64, Int64
    from feast.value_type import ValueType

    customer = Entity(name="customer", join_keys=["customer_id"], value_type=ValueType.STRING)
    source_path = parquet_path.resolve().as_uri() if parquet_path.is_absolute() else parquet_path.as_posix()
    source = FileSource(
        name="transactions",
        path=source_path,
        timestamp_field="event_timestamp",
    )
    types = {
        "log_amount": Float64,
        "hour": Int64,
        "velocity_24h": Int64,
        "distance_km": Float64,
        "new_device": Int64,
        "merchant_risk": Float64,
    }
    view = FeatureView(
        name=VIEW_NAME,
        entities=[customer],
        ttl=timedelta(days=60),
        schema=[Field(name=name, dtype=types[name]) for name in FEATURE_COLUMNS],
        source=source,
        online=True,
    )
    return customer, view


def apply_store(repo: Path, frame: pd.DataFrame):
    from feast import FeatureStore

    data = repo / "data"
    data.mkdir(parents=True, exist_ok=True)
    table = prepare_frame(frame)
    parquet = data / "transactions.parquet"
    table.to_parquet(parquet, index=False)
    (repo / "feature_store.yaml").write_text(
        "\n".join(
            [
                "project: risklens",
                "provider: local",
                "registry: data/registry.db",
                "online_store:",
                "  type: sqlite",
                "  path: data/online.db",
                "offline_store:",
                "  type: file",
                "entity_key_serialization_version: 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    customer, view = feature_view(Path("data/transactions.parquet"))
    store = FeatureStore(repo_path=str(repo))
    store.apply([customer, view])
    start = table["event_timestamp"].min().to_pydatetime()
    end = (table["event_timestamp"].max() + pd.Timedelta(days=1)).to_pydatetime()
    store.materialize(start, end)
    return store


def online_features(store, customer_id: str) -> dict:
    names = [f"{VIEW_NAME}:{name}" for name in FEATURE_COLUMNS]
    result = store.get_online_features(
        features=names,
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()
    return {name: result[name][0] for name in FEATURE_COLUMNS}


def registered_feature_names(view) -> list[str]:
    fields = getattr(view, "schema", None) or getattr(view, "features", [])
    return [field.name for field in fields]
