from __future__ import annotations


def create_iris_table_sql(table: str) -> str:
    # Schema adapted for the wine dataset features
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        alcohol double precision,
        malic_acid double precision,
        ash double precision,
        alcalinity_of_ash double precision,
        magnesium double precision,
        total_phenols double precision,
        flavanoids double precision,
        nonflavanoid_phenols double precision,
        proanthocyanins double precision,
        color_intensity double precision,
        hue double precision,
        od280_od315_of_diluted_wines double precision,
        proline double precision,
        target integer,
        target_name text,
        ingestion_date date
    )
    """


def create_eval_table_sql(table: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        run_id text,
        accuracy double precision,
        precision_weighted double precision,
        recall_weighted double precision,
        execution_date date
    )
    """
