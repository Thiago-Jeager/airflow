from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from sklearn import datasets

from .config import Settings
from .db import get_engine
from .schemas import create_iris_table_sql


RENAME_MAP = {
    # map the odd feature name containing a slash to a safe column name
    "od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines",
}


def load_iris_df(ds: str | None = None) -> pd.DataFrame:
    """Load the wine dataset (keeps function name for compatibility).

    Returns a dataframe with feature columns, `target`, `target_name`, and optionally
    `ingestion_date` when `ds` is provided.
    """
    wine = datasets.load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    # normalize any uncommon column names
    df = df.rename(columns=RENAME_MAP)
    df["target"] = wine.target
    # sklearn wine dataset does not provide readable target names by default; create them
    try:
        target_names = wine.target_names
    except Exception:
        target_names = [str(x) for x in sorted(set(wine.target))]
    df["target_name"] = df["target"].apply(lambda i: target_names[i])
    if ds:
        df["ingestion_date"] = pd.to_datetime(ds).normalize()
    return df


def ensure_iris_table(settings: Settings) -> None:
    engine = get_engine(settings)
    with engine.begin() as conn:
        conn.execute(create_iris_table_sql(settings.iris_table))


def write_iris(settings: Settings, df: pd.DataFrame) -> int:
    engine = get_engine(settings)
    df.to_sql(settings.iris_table, engine, if_exists="append", index=False)
    return len(df)
